"""
asr.py — faster-whisper wrapper.

Two modes:
  1. transcribe(audio)             — batch, used by orchestrator at speech end
  2. StreamingASR.update(chunk)    — incremental, re-transcribes a rolling
                                     buffer every ~400ms and emits partials.

Streaming is the prerequisite for speculative prefill (Day 5).
"""

import logging
import time
import threading
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ASR:
    """Batch ASR — call transcribe(audio) on a complete utterance.

    Defaults tuned for quality over raw speed:
      * model_size="small.en" — ~3x WER reduction vs tiny.en on noisy/accented
        English, still runs ~real-time on CPU (int8) for short utterances.
      * beam_size=5, best_of=5 — robust decoding.
      * VAD-filter on — drops silence / breath / mic hiss segments that
        otherwise cause whisper to hallucinate ("Thank you.", "Bye.", etc.).
      * condition_on_previous_text=False — prevents cross-utterance
        hallucination loops (a known whisper failure mode).
      * language="en" — skips the language-id step (saves ~80ms and avoids
        misdetection on short clips).
      * no_speech_threshold=0.6 — aggressively drop silent segments.
    """

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        print(f"Loading ASR model {model_size}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size
        print(f"ASR model loaded: {model_size}")

    def transcribe(self, audio: np.ndarray, beam_size: int = 5) -> str:
        """Synchronous, full-utterance transcription (high quality)."""
        segments, _info = self.model.transcribe(
            audio,
            beam_size=beam_size,
            best_of=beam_size,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.6,
            temperature=[0.0, 0.2, 0.4],  # fallback ladder on low-confidence
        )
        text = "".join(segment.text for segment in segments)
        return text.strip()

    def transcribe_partial(self, audio: np.ndarray) -> str:
        """Faster, lower-quality pass for streaming partials (beam=1)."""
        segments, _info = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,           # let partials see in-progress speech
            no_speech_threshold=0.6,
            temperature=0.0,
        )
        text = "".join(segment.text for segment in segments)
        return text.strip()


class StreamingASR:
    """
    Wraps an ASR model with a rolling buffer + periodic re-transcription.

    Usage:
        sasr = StreamingASR(asr_model, on_partial=lambda t: print(t))
        sasr.start()
        for chunk in mic_stream:
            sasr.feed(chunk)
        final = sasr.finalize()
        sasr.stop()

    Design:
      * feed() appends to an in-memory float32 buffer (lock-protected).
      * A background worker thread re-runs whisper on the buffer every
        `interval_ms`. Each result is emitted via on_partial(text).
      * finalize() runs one last transcription on the full buffer and
        returns the final text.
      * The 'stable partial' tracking (unchanged for N consecutive runs)
        is what speculative prefill will consume in Day 5.

    Notes:
      * We re-transcribe from scratch every time (whisper has no streaming
        decoder state we can persist). With small.en + beam=1 on CPU this
        is ~150-300ms per call for short utterances, fast enough at 400ms
        intervals. We deliberately call the lighter transcribe_partial()
        path here — partials only need to be roughly right; the final
        utterance is re-transcribed at full beam=5 quality by finalize().
      * For utterances longer than ~10s this becomes expensive. Day-5 work
        will switch to a sliding window if needed.
    """

    def __init__(
        self,
        asr: ASR,
        on_partial: Optional[Callable[[str], None]] = None,
        interval_ms: int = 400,
        sample_rate: int = 16000,
    ):
        self._asr = asr
        self._on_partial = on_partial or (lambda t: None)
        self._interval_s = interval_ms / 1000.0
        self._sample_rate = sample_rate

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._running = False
        self._worker: Optional[threading.Thread] = None

        # Stable-partial detection
        self._last_partial: str = ""
        self._stable_count: int = 0
        self.partial_history: list[tuple[float, str]] = []

    def start(self) -> None:
        self._running = True
        with self._buffer_lock:
            self._buffer = np.zeros(0, dtype=np.float32)
        self._last_partial = ""
        self._stable_count = 0
        self.partial_history = []
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        if self._worker:
            self._worker.join(timeout=2.0)
            self._worker = None

    def feed(self, chunk: np.ndarray) -> None:
        """Append one audio chunk to the rolling buffer. Thread-safe."""
        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])

    def finalize(self) -> str:
        """Stop streaming, run one last transcription on the full buffer."""
        self._running = False
        with self._buffer_lock:
            audio = self._buffer.copy()
        if len(audio) == 0:
            return ""
        text = self._asr.transcribe(audio)
        self._on_partial(text)  # final emission
        self.partial_history.append((time.perf_counter(), text))
        return text

    @property
    def stable_partial(self) -> Optional[str]:
        """
        Returns the current partial only if it has been unchanged for
        >=2 consecutive runs and is >=3 words. None otherwise.

        This is what the speculative-prefill controller subscribes to.
        """
        if self._stable_count >= 2 and len(self._last_partial.split()) >= 3:
            return self._last_partial
        return None

    # ── worker ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval_s)
            if not self._running:
                break
            with self._buffer_lock:
                audio = self._buffer.copy()
            if len(audio) < self._sample_rate * 0.3:  # need >=300ms
                continue
            try:
                text = self._asr.transcribe_partial(audio)
            except Exception as e:
                logger.warning("[StreamingASR] transcribe failed: %s", e)
                continue
            if not text:
                continue
            self.partial_history.append((time.perf_counter(), text))
            if text == self._last_partial:
                self._stable_count += 1
            else:
                self._last_partial = text
                self._stable_count = 1
            try:
                self._on_partial(text)
            except Exception as e:
                logger.warning("[StreamingASR] on_partial callback raised: %s", e)
