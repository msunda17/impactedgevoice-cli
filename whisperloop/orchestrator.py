"""
orchestrator.py — Async event-driven conversational state machine.

Owns:
  * The conversational state machine (delegated to BargeInController)
  * Audio capture → VAD → ASR → LLM → TTS pipeline coordination
  * Latency instrumentation (delegated to LatencyLogger)
  * Self-trigger suppression while the assistant is speaking

Self-trigger suppression:
  The mic continues to capture during TTS playback so we can detect
  barge-in, but VAD signals are gated through BargeInController.should_interrupt
  which requires sustained loud audio (RMS > threshold for >=200ms) before
  treating it as a real interrupt. This prevents the assistant from hearing
  its own voice and looping.
"""

import asyncio
import logging
import re
import threading
from typing import Optional

import numpy as np

from whisperloop.audio_io import AudioStreamer, play_audio_async
from whisperloop.bargein import BargeInController, State
from whisperloop.console import (
    style_bot, style_muted, style_separator, style_system, style_you,
)
from whisperloop.instrumentation import LatencyLogger
from whisperloop.kv_cache import KVCacheManager
from whisperloop.asr import ASR, StreamingASR
from whisperloop.speculative_prefill import SpeculativePrefillController
from whisperloop.memex import Memex
from whisperloop.memex.pruner import LivePruner
from whisperloop.tts import TTS
from whisperloop.vad import VAD

logger = logging.getLogger(__name__)

# Sentence splitter for the opening announcement — small enough to keep TTS
# latency tight, large enough to not chop mid-clause.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class Orchestrator:
    """
    Canonical voice conversation engine.

    Modes:
        Orchestrator()                                    → fresh voice chat
        Orchestrator(kv=seeded_kv)                        → reuse a primed KV
        run(opening_announcement=...)                     → speak intro then listen
        run(initial_user_text=...)                        → generate first reply
                                                            from typed text, then
                                                            listen on the mic.
    """

    def __init__(
        self,
        kv: Optional[KVCacheManager] = None,
        memex: Optional[Memex] = None,
        baseline_checkpoint: str = "system",
    ):
        print("--- Initializing Whisperloop Orchestrator ---")
        self.audio_streamer = AudioStreamer(chunk_size=512)
        self.vad = VAD()
        self.asr = ASR(model_size="small.en")
        self.tts = TTS(model_path="models/piper/en_US-lessac-medium.onnx")

        if kv is not None:
            # Caller has already loaded + warmed-up the KV (e.g. file/text mode
            # primed it with document context). Reuse it as-is.
            self.kv = kv
            logger.info("[ORCH] Reusing pre-loaded KV (seq_len=%d)", self.kv.kv_len)
        else:
            # Auto-pick the simple tier via the shared ModelRouter so we use
            # whatever GGUF is actually available rather than a hardcoded path.
            from whisperloop.model_router import ModelRouter, TaskTier
            router = ModelRouter()
            self.kv = router.select_model(query="voice chat", force_tier=TaskTier.SIMPLE)

        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.bargein = BargeInController(self.kv, self.tts_queue)
        self.latency = LatencyLogger("bench/latency.jsonl")
        
        # Speculative prefill: overlap ASR with LLM prefill using partials
        self._streaming_asr: Optional[StreamingASR] = None
        self._spec_prefill: Optional[SpeculativePrefillController] = None

        # Speculative recall: Memex recall pre-fetched during streaming ASR so
        # the [MEMEX] block is ready the moment speech ends (no recall latency
        # on the hot path between VAD-end and LLM prefill).
        self._speculative_memex_block: str = ""
        self._speculative_recall_partial: str = ""  # partial that triggered the pre-fetch

        # Memex: long-term memory across sessions + live in-session pruning.
        self.memex = memex or Memex()
        self.pruner = LivePruner(
            self.kv, self.memex, baseline_checkpoint=baseline_checkpoint
        )
        
        print("--- Ready ---")

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        opening_announcement: Optional[str] = None,
        initial_user_text: Optional[str] = None,
    ) -> None:
        """
        Main event loop.

        Args:
            opening_announcement: Speak this text via TTS before listening.
                Used by File mode to read a document summary aloud. The KV
                cache is NOT modified by this — it's pure narration of text
                the assistant has already "said" via priming.
            initial_user_text: Treat this as the user's first turn (no ASR).
                The LLM generates a response that's streamed to TTS, then the
                mic takes over for follow-ups. Used by Text mode.
        """
        playback_task = asyncio.create_task(self.playback_worker())

        # 1) Optional opening announcement (e.g. document summary aloud)
        if opening_announcement:
            await self._speak_text(opening_announcement)

        # 2) Optional first "user turn" sourced from typed text
        if initial_user_text:
            self.latency.next_turn()
            self.latency.event("speech_start")
            self.bargein.on_user_speech_start()
            self.bargein.on_user_speech_end()
            self.bargein.save_pre_assistant_checkpoint()
            await self._generate_and_stream(initial_user_text)
            await self.tts_queue.join()
            self.bargein.on_turn_complete()

        self.audio_streamer.start_recording()
        audio_buffer: list[np.ndarray] = []

        print(style_system("\n[Orchestrator] Listening…"))
        try:
            while True:
                chunk = await self.audio_streamer.audio_queue.get()
                vad_res = self.vad.process_chunk(chunk)
                st = self.bargein.state

                # ── IDLE: waiting for user ─────────────────────────────
                if st == State.IDLE:
                    if vad_res and "start" in vad_res:
                        if self.bargein.on_user_speech_start():
                            print("  (hearing you…)", end="\r")
                            self.latency.next_turn()
                            self.latency.event("speech_start")
                            audio_buffer = [chunk]
                            
                            # Start streaming ASR for speculative prefill + recall
                            self._spec_prefill = SpeculativePrefillController(
                                self.kv,
                                on_prefill_start=lambda p: logger.debug("[SPEC] Started: %r", p),
                                on_prefill_confirm=lambda p, f, r: logger.debug("[SPEC] %s: %r -> %r", r, p, f),
                            )
                            self._speculative_memex_block = ""
                            self._speculative_recall_partial = ""
                            self._streaming_asr = StreamingASR(
                                self.asr,
                                on_partial=self._on_streaming_partial,
                                interval_ms=400,
                            )
                            self._streaming_asr.start()

                # ── USER_SPEAKING: capture until VAD end ───────────────
                elif st == State.USER_SPEAKING:
                    audio_buffer.append(chunk)
                    # Feed streaming ASR for speculative prefill
                    if self._streaming_asr:
                        self._streaming_asr.feed(chunk)
                    
                    if vad_res and "end" in vad_res:
                        pass  # Visual feedback handled in _run_turn
                        self.latency.event("speech_end", chunks=len(audio_buffer))
                        self.bargein.on_user_speech_end()

                        full_audio = np.concatenate(audio_buffer)
                        audio_buffer = []
                        
                        # Finalize streaming ASR and speculative prefill
                        final_transcript = ""
                        if self._streaming_asr:
                            final_transcript = self._streaming_asr.finalize()
                            self._streaming_asr.stop()
                            self._streaming_asr = None
                        if self._spec_prefill:
                            self._spec_prefill.finalize(final_transcript)
                            self._spec_prefill = None
                        
                        await self._run_turn_with_transcript(final_transcript)

                        self.vad.reset()
                        # Drain queue: any audio captured while we were
                        # synthesising/playing could be stale or our own TTS
                        self._drain_audio_queue()
                        # Print listening indicator without metrics clutter
                        print(style_system("\n  [Listening…]"))

                # ── SPEAKING: monitor for barge-in ─────────────────────
                elif st == State.SPEAKING:
                    if self.bargein.should_interrupt(chunk, vad_res):
                        print("\n  [INTERRUPTED] You cut in — stopping the assistant.")
                        self.latency.event("barge_in")
                        await self.bargein.interrupt()
                        # We are now USER_SPEAKING (post_interrupt transition)
                        audio_buffer = [chunk]

                # THINKING / INTERRUPTED: ignore mic for now (transient)

        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self.audio_streamer.stop_recording()
            playback_task.cancel()
            self.latency.close()
            try:
                self.memex.close()
            except Exception as e:
                logger.debug("[ORCH] memex.close failed: %s", e)
            self._print_metrics()

    # ------------------------------------------------------------------ #
    #  Per-turn pipeline                                                  #
    # ------------------------------------------------------------------ #

    async def _run_turn(self, audio: np.ndarray) -> None:
        """ASR → LLM → TTS for one user turn, with cancellation support."""
        self.bargein.current_turn_task = asyncio.current_task()
        try:
            # ── ASR ────────────────────────────────────────────────────
            self.latency.mark_start("asr")
            transcript = await asyncio.to_thread(self.asr.transcribe, audio)
            self.latency.mark_end("asr", text=transcript)

            if not transcript:
                self.bargein.on_turn_complete()
                return

            await self._run_turn_with_transcript(transcript)

        except asyncio.CancelledError:
            logger.info("[ORCH] Turn cancelled (barge-in)")
            raise

    async def _run_turn_with_transcript(self, transcript: str) -> None:
        """
        LLM → TTS for a turn where ASR already completed.
        Used by streaming ASR path where transcription happened during speech.
        """
        if not transcript:
            self.bargein.on_turn_complete()
            return

        # ── Live KV pruning: if context window is filling up, compact the
        #    oldest turns into Memex BEFORE we add the new user turn. ──
        try:
            self.pruner.prune_if_needed()
        except Exception as e:
            logger.warning("[ORCH] prune_if_needed failed: %s", e)

        # ── Memex recall: use the speculatively pre-fetched block if the
        #    partial that triggered it is a prefix of the final transcript
        #    (meaning the query intent hasn't changed). Otherwise re-fetch. ──
        memex_block = ""
        try:
            spec_partial = self._speculative_recall_partial
            spec_block = self._speculative_memex_block
            if spec_block and transcript.lower().startswith(spec_partial.lower()[:40]):
                memex_block = spec_block
                logger.debug("[SPEC-RECALL] Used pre-fetched Memex block (%d chars)", len(memex_block))
            else:
                memex_block = self.memex.recall_block(transcript, k=4)
        except Exception as e:
            logger.warning("[ORCH] memex.recall_block failed: %s", e)
        finally:
            self._speculative_memex_block = ""
            self._speculative_recall_partial = ""

        prefill_text = transcript
        if memex_block:
            prefill_text = f"{memex_block}\n\n{transcript}"

        # Visual separator for new turn (dim) + bold weighted speaker labels.
        sep = "─" * 60
        print(style_separator(f"\n{sep}"))
        print(f"  {style_you('YOU:')} {style_you(transcript)}")
        if memex_block:
            n_mem = max(0, memex_block.count(chr(10)) - 1)
            print(style_muted(f"  [MEMEX] {n_mem} memories injected"))
        print(style_separator(sep))

        # ── KV checkpoint right before LLM decode (for barge-in) ──
        self.bargein.save_pre_assistant_checkpoint()

        # ── LLM generation (streamed to TTS queue per sentence) ───
        print(f"  {style_bot('BOT:')} ", end="", flush=True)
        response_text = await self._generate_and_stream(prefill_text)
        print()  # newline after response

        # ── Wait for playback drain ───────────────────────────────
        await self.tts_queue.join()
        self.latency.event("tts_done")
        self.bargein.on_turn_complete()
        self.latency.event("turn_complete")

        # ── Memex store + pruner log (off the hot path) ──
        try:
            self.pruner.record_turn(transcript, response_text or "")
            # Run summarization in a thread so it doesn't block the next turn.
            asyncio.create_task(asyncio.to_thread(
                self.memex.store, transcript, response_text or "", "voice", None,
            ))
        except Exception as e:
            logger.debug("[ORCH] memex post-turn store failed: %s", e)

    def _on_streaming_partial(self, partial: str) -> None:
        """
        Called by StreamingASR every ~400ms with a partial transcript.
        Feeds the speculative prefill controller AND triggers a speculative
        Memex recall on the first stable partial so results are ready by
        the time speech ends.
        """
        if self._spec_prefill:
            self._spec_prefill.on_partial(partial)

        # Speculative recall: trigger once per turn on the first stable partial
        # (>=3 words, >=15 chars — same gate as speculative prefill).
        if (
            not self._speculative_recall_partial
            and partial
            and len(partial.split()) >= 3
            and len(partial) >= 15
        ):
            self._speculative_recall_partial = partial
            # Run in a daemon thread — don't block the ASR loop.
            threading.Thread(
                target=self._run_speculative_recall,
                args=(partial,),
                daemon=True,
            ).start()

    def _run_speculative_recall(self, partial: str) -> None:
        """Background thread: pre-fetch Memex block for a partial transcript."""
        try:
            block = self.memex.recall_block(partial, k=4)
            # Only store if the partial hasn't been superseded
            if self._speculative_recall_partial == partial:
                self._speculative_memex_block = block
                logger.debug("[SPEC-RECALL] Pre-fetched %d chars for %r", len(block), partial[:40])
        except Exception as e:
            logger.debug("[SPEC-RECALL] Pre-fetch failed: %s", e)

    async def _generate_and_stream(self, transcript: str) -> str:
        """
        Run kv.generate() in a background thread; route tokens into a
        sentence buffer; flush each sentence to the TTS queue.

        Returns the full assistant response text (used by Memex to summarize).
        """
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue = asyncio.Queue()
        self.latency.event("llm_prefill_start")
        first_token_logged = False

        def generator_thread():
            try:
                for tok in self.kv.generate(transcript, cancel_event=self.bargein.cancel_event):
                    loop.call_soon_threadsafe(token_queue.put_nowait, tok)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        gen_task = asyncio.create_task(asyncio.to_thread(generator_thread))

        sentence_buf = ""
        full_response = ""
        try:
            while True:
                tok = await token_queue.get()
                if tok is None:
                    break
                if not first_token_logged:
                    self.latency.event("llm_first_token")
                    first_token_logged = True
                sentence_buf += tok
                full_response += tok
                print(tok, end="", flush=True)
                if any(p in tok for p in [".", "?", "!"]):
                    chunk = sentence_buf.strip()
                    sentence_buf = ""
                    if chunk:
                        await self.tts_queue.put(chunk)
            if sentence_buf.strip():
                await self.tts_queue.put(sentence_buf.strip())
            self.latency.event("llm_decode_done", kv_after=self.kv.kv_len)
            return full_response
        except asyncio.CancelledError:
            # Drain token queue so the generator thread doesn't deadlock
            self.bargein.cancel_event.set()
            await gen_task
            raise

    # ------------------------------------------------------------------ #
    #  Opening announcement (no LLM — pure TTS readout)                   #
    # ------------------------------------------------------------------ #

    async def _speak_text(self, text: str) -> None:
        """
        Push a pre-existing text into the TTS queue, sentence by sentence,
        and wait for it to finish playing. Does NOT touch the KV cache.
        """
        sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
        if not sentences:
            return
        # Visual separator for opening announcement
        sep = "─" * 60
        print(style_separator(f"\n{sep}"))
        print(f"  {style_bot('BOT:')} {style_muted('(opening summary)')}")
        print(style_separator(sep))
        # Briefly enter SPEAKING state so the playback worker logs cleanly
        # and so any mic noise during readout isn't mistaken for user turns.
        self.bargein.enter_speaking_for_announcement()
        for sent in sentences:
            await self.tts_queue.put(sent)
        await self.tts_queue.join()
        self.bargein.on_turn_complete()

    # ------------------------------------------------------------------ #
    #  TTS playback worker                                                #
    # ------------------------------------------------------------------ #

    async def playback_worker(self):
        first_audio_logged = False
        while True:
            try:
                text = await self.tts_queue.get()
                if text == "[END]":
                    self.tts_queue.task_done()
                    continue
                audio_out = await asyncio.to_thread(self.tts.synthesize, text)
                if not first_audio_logged:
                    self.latency.event("tts_first_audio")
                    self.bargein.on_first_tts_audio()
                    first_audio_logged = True
                await play_audio_async(audio_out)
                self.tts_queue.task_done()
                # Reset first-audio flag once the queue is fully drained,
                # so the next turn logs its own first-audio timestamp.
                if self.tts_queue.empty():
                    first_audio_logged = False
            except asyncio.CancelledError:
                raise
            except Exception as e:
                import traceback
                print(f"\n[ERROR in playback_worker] {e}")
                traceback.print_exc()
                try:
                    self.tts_queue.task_done()
                except ValueError:
                    pass

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _drain_audio_queue(self) -> None:
        while not self.audio_streamer.audio_queue.empty():
            try:
                self.audio_streamer.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _print_metrics(self):
        if not self.kv.turn_metrics:
            return
        print("\n\n=== SESSION LATENCY SUMMARY ===")
        print(f"{'Turn':>5} | {'Prefill ms':>10} | {'TTFT ms':>8} | {'Tok/s':>7} | {'KV len':>7}")
        print("-" * 52)
        for m in self.kv.turn_metrics:
            print(
                f"{m['turn']:>5} | {m['prefill_ms']:>10.1f} | "
                f"{m['ttft_ms']:>8.1f} | {m['tokens_per_sec']:>7.1f} | {m['kv_after']:>7}"
            )
