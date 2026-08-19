"""
instrumentation.py — Structured latency logging.

Every stage boundary in the pipeline writes a timestamped record to a JSONL
file. The benchmark script reads these to compute p50/p95/p99.

Schema (one JSON object per line):
    {
      "ts":         float,    # time.perf_counter()
      "turn":       int,      # turn index (0 = warm-up)
      "event":      str,      # stage label, e.g. "speech_end", "first_token"
      "data":       dict,     # optional extras
    }

Events emitted by the orchestrator:
    speech_start        — VAD start
    speech_end          — VAD end
    asr_done            — final transcript ready
    llm_prefill_done    — user tokens prefilled
    llm_first_token     — first decoded token
    llm_decode_done     — assistant decode complete
    tts_first_audio     — first audio sample queued for playback
    tts_done            — all sentences played
    barge_in            — user interrupted assistant
    turn_complete       — full turn finished
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class LatencyLogger:
    """Writes JSONL latency events. Cheap (one line per event)."""

    def __init__(self, path: Union[str, Path] = "bench/latency.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._turn = 0
        self._stage_starts: Dict[str, float] = {}
        logger.info("[LATENCY] Logging to %s", self.path)

    def next_turn(self) -> int:
        self._turn += 1
        self._stage_starts.clear()
        return self._turn

    def event(self, name: str, **data: Any) -> float:
        ts = time.perf_counter()
        record = {"ts": ts, "turn": self._turn, "event": name, "data": data}
        self._file.write(json.dumps(record) + "\n")
        return ts

    def mark_start(self, stage: str) -> float:
        ts = time.perf_counter()
        self._stage_starts[stage] = ts
        return ts

    def mark_end(self, stage: str, **data: Any) -> Optional[float]:
        """Records `<stage>_ms` derived from the matching mark_start."""
        start = self._stage_starts.pop(stage, None)
        if start is None:
            return None
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.event(f"{stage}_done", elapsed_ms=elapsed_ms, **data)
        return elapsed_ms

    def close(self) -> None:
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass
