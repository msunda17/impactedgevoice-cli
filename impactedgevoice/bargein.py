"""
bargein.py — Conversational State Machine + Interrupt Controller (Crown Jewel #2)

States:
    IDLE          — waiting for user to speak
    USER_SPEAKING — user is talking; we are capturing audio
    THINKING      — user finished; ASR + LLM prefill + decode in progress
    SPEAKING      — TTS playing back assistant response
    INTERRUPTED   — user spoke over assistant; cleanup in flight

Transitions:
    IDLE          -> USER_SPEAKING  (VAD start)
    USER_SPEAKING -> THINKING       (VAD end)
    THINKING      -> SPEAKING       (first TTS audio ready)
    SPEAKING      -> INTERRUPTED    (VAD start while playing)
    INTERRUPTED   -> USER_SPEAKING  (cleanup complete)
    SPEAKING      -> IDLE           (TTS queue drained)
    THINKING      -> IDLE           (no transcript / empty response)

On interrupt:
    1. Set cancel_event so the LLM decode loop in kv_cache.generate() bails out
    2. Cancel the asyncio decode task
    3. sd.stop() to flush the speaker buffer
    4. Drain the TTS sentence queue
    5. Truncate KV cache back to the pre-assistant checkpoint
    6. Reset VAD internal state so we don't carry stale hangover counters
"""

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Optional

import numpy as np
import sounddevice as sd

from impactedgevoice.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE          = "IDLE"
    USER_SPEAKING = "USER_SPEAKING"
    THINKING      = "THINKING"
    SPEAKING      = "SPEAKING"
    INTERRUPTED   = "INTERRUPTED"


class BargeInController:
    """
    Owns conversational state + interrupt machinery.

    The orchestrator transitions states by calling the on_* methods.
    The controller exposes a `cancel_event` (threading.Event) that the
    LLM decode loop checks every token — this is how we kill in-flight
    inference cleanly without process-level signals.
    """

    # Minimum RMS amplitude required to trigger barge-in while assistant
    # is speaking. Tuned to reject the assistant's own TTS audio bleed
    # through the mic. Adjust per hardware.
    BARGE_IN_RMS_THRESHOLD = 0.02

    # User must produce loud-enough speech for at least this many ms
    # before we treat it as a real interrupt (avoid coughs / clicks).
    BARGE_IN_MIN_DURATION_MS = 200

    def __init__(self, kv: KVCacheManager, tts_queue: asyncio.Queue):
        self._state: State = State.IDLE
        self.kv = kv
        self.tts_queue = tts_queue

        # Cross-thread cancellation signal.
        # The LLM decode loop runs in a background thread (asyncio.to_thread).
        # threading.Event is the right primitive: cheap to check, safe across
        # threads, no event loop dependency.
        self.cancel_event = threading.Event()

        # Asyncio task wrapping the per-turn generation pipeline.
        # We keep a handle so we can .cancel() it on interrupt.
        self.current_turn_task: Optional[asyncio.Task] = None

        # KV checkpoint label we roll back to on interrupt.
        # Set by the orchestrator just before assistant decode begins.
        self.pre_assistant_checkpoint: Optional[int] = None

        # Tracks consecutive loud frames during SPEAKING for debounced
        # barge-in detection.
        self._loud_frame_count: int = 0

        # State transition log for tests / debugging.
        self.transition_log: list[tuple[float, State, State, str]] = []

    # ------------------------------------------------------------------ #
    #  State accessor                                                     #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> State:
        return self._state

    def _transition(self, new_state: State, reason: str) -> None:
        old = self._state
        self._state = new_state
        self.transition_log.append((time.perf_counter(), old, new_state, reason))
        logger.debug("[STATE] %s -> %s (%s)", old.value, new_state.value, reason)

    # ------------------------------------------------------------------ #
    #  Public API — orchestrator calls these                              #
    # ------------------------------------------------------------------ #

    def on_user_speech_start(self) -> bool:
        """
        Called when VAD emits a start event. Returns True if the orchestrator
        should begin capturing audio for a new turn.
        """
        if self._state == State.IDLE:
            self._transition(State.USER_SPEAKING, "vad_start")
            return True
        # Already in USER_SPEAKING etc. — ignore duplicate triggers
        return False

    def on_user_speech_end(self) -> bool:
        """Called when VAD emits an end event."""
        if self._state == State.USER_SPEAKING:
            self._transition(State.THINKING, "vad_end")
            return True
        return False

    def on_first_tts_audio(self) -> None:
        """Orchestrator signals that TTS playback has started."""
        if self._state == State.THINKING:
            self._transition(State.SPEAKING, "first_tts_audio")

    def on_turn_complete(self) -> None:
        """All TTS sentences for this turn have played out."""
        if self._state in (State.SPEAKING, State.THINKING):
            self._transition(State.IDLE, "turn_complete")
            self.pre_assistant_checkpoint = None

    def enter_speaking_for_announcement(self) -> None:
        """
        Enter SPEAKING state from IDLE to play a pre-canned announcement
        (e.g. a document summary read aloud). Used by the orchestrator when
        opening_announcement is provided before the mic takes over.
        """
        if self._state == State.IDLE:
            self._transition(State.SPEAKING, "opening_announcement")

    def save_pre_assistant_checkpoint(self) -> None:
        """
        Called right before the LLM starts generating the assistant turn.
        We snapshot the KV length here so we can roll back on interrupt.
        """
        self.pre_assistant_checkpoint = self.kv.save_checkpoint(
            f"turn_{self.kv._turn_index + 1}_pre_assistant_bargein"
        )

    # ------------------------------------------------------------------ #
    #  Barge-in detection                                                 #
    # ------------------------------------------------------------------ #

    def should_interrupt(self, audio_chunk: np.ndarray, vad_result: Optional[dict]) -> bool:
        """
        Returns True if the audio chunk indicates a real user interruption
        during SPEAKING. False otherwise.

        Heuristic (debounced): require VAD-start signal AND chunk RMS above
        threshold AND BARGE_IN_MIN_DURATION_MS sustained loud audio.
        """
        if self._state != State.SPEAKING:
            return False

        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))
        loud = rms > self.BARGE_IN_RMS_THRESHOLD

        if loud:
            self._loud_frame_count += 1
        else:
            self._loud_frame_count = 0

        # 16 kHz / 512 sample chunks = 32 ms per chunk.
        # 200 ms / 32 ms ≈ 7 chunks.
        ms_per_chunk = (len(audio_chunk) / 16000.0) * 1000.0
        required_chunks = max(1, int(self.BARGE_IN_MIN_DURATION_MS / ms_per_chunk))

        # VAD-start is the primary signal; RMS gating filters out the
        # assistant's own TTS audio bleeding through the mic.
        triggered = (
            vad_result is not None
            and "start" in vad_result
            and self._loud_frame_count >= required_chunks
        )
        return triggered

    # ------------------------------------------------------------------ #
    #  Interrupt sequence                                                 #
    # ------------------------------------------------------------------ #

    async def interrupt(self) -> None:
        """
        Kill the in-flight assistant turn cleanly:
            1. Signal LLM decode thread to stop on its next token
            2. Stop sounddevice playback
            3. Drain the TTS sentence queue
            4. Cancel the turn task (raises CancelledError in the coroutine)
            5. Truncate the KV cache back to pre-assistant checkpoint
            6. Reset internal counters
        """
        if self._state not in (State.SPEAKING, State.THINKING):
            return

        self._transition(State.INTERRUPTED, "barge_in")

        # 1. LLM decode thread — set flag, it polls between tokens.
        self.cancel_event.set()

        # 2. Stop speaker output immediately.
        try:
            sd.stop()
        except Exception as e:
            logger.warning("[BARGE-IN] sd.stop() raised: %s", e)

        # 3. Drain TTS queue so the playback worker doesn't keep going.
        drained = 0
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break
        logger.info("[BARGE-IN] Drained %d pending TTS sentences", drained)

        # 4. Cancel the turn coroutine. Catches CancelledError upstream.
        if self.current_turn_task and not self.current_turn_task.done():
            self.current_turn_task.cancel()
            try:
                await self.current_turn_task
            except (asyncio.CancelledError, Exception):
                pass

        # 5. Roll back KV cache.
        if self.pre_assistant_checkpoint is not None:
            self.kv.truncate_to(self.pre_assistant_checkpoint)
            logger.info(
                "[BARGE-IN] Truncated KV cache to seq_len=%d",
                self.pre_assistant_checkpoint,
            )

        # 6. Reset for next turn.
        self.cancel_event.clear()
        self._loud_frame_count = 0
        self.pre_assistant_checkpoint = None
        self.current_turn_task = None

        # Immediately transition into USER_SPEAKING because the user is
        # actively talking right now (that's why we interrupted).
        self._transition(State.USER_SPEAKING, "post_interrupt")
