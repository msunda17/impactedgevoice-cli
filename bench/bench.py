"""
bench.py — Reproducible latency benchmark for the LLM + TTS portion of
the impactedgevoice pipeline.

What it measures (per prompt, per turn):
    prefill_ms          — user tokens prefilled into the KV cache
    ttft_ms             — end-of-prefill to first decoded token
    decode_ms           — first token to EOS
    tts_first_audio_ms  — first LLM token to first synthesized audio sample
    e2e_ms              — start-of-prefill to first audio sample ready

What it does NOT measure here:
    Mic capture, VAD endpointing, ASR — these are real-time and best
    measured via a separate audio-input benchmark using pre-recorded WAVs.
    For LLM/serving-layer characterization the text-input path is the
    right primitive: it isolates the model + cache + TTS from network
    and audio-driver jitter.

Usage:
    python -m bench.bench --prompts bench/prompts.json --turns 15
    python -m bench.bench --quick      # 5 prompts, useful for smoke tests

Output:
    Prints a p50/p95/p99 table per stage.
    Writes bench/results/<timestamp>-bench.json with the raw per-turn data.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow `python -m bench.bench` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from impactedgevoice.kv_cache import KVCacheManager
from impactedgevoice.tts import TTS

MODEL_PATH = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
TTS_PATH = "models/piper/en_US-lessac-medium.onnx"


# ---------------------------------------------------------------------- #
# Memory probe                                                            #
# ---------------------------------------------------------------------- #

def get_process_memory_mb() -> float:
    """Resident memory of the current Python process, in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


# ---------------------------------------------------------------------- #
# Single-turn benchmark                                                   #
# ---------------------------------------------------------------------- #

def run_one_turn(kv: KVCacheManager, tts: TTS, prompt: str) -> dict:
    """Run one prompt through prefill→decode→TTS, return per-stage timings."""
    t_e2e_start = time.perf_counter()
    kv_before = kv.kv_len

    t_first_token = None
    t_decode_end = None
    full_text = ""
    for tok in kv.generate(prompt, max_tokens=150):
        if t_first_token is None:
            t_first_token = time.perf_counter()
        full_text += tok
    t_decode_end = time.perf_counter()

    # First-sentence TTS — what would actually be played first
    first_sentence = ""
    for ch in full_text:
        first_sentence += ch
        if ch in ".?!":
            break
    if not first_sentence.strip():
        first_sentence = full_text[:80] or "(empty)"

    t_tts_start = time.perf_counter()
    _ = tts.synthesize(first_sentence)
    t_tts_end = time.perf_counter()

    # Pull the per-turn metrics that kv_cache.py just recorded
    kv_metric = kv.turn_metrics[-1]

    return {
        "prompt":             prompt,
        "kv_before":          kv_before,
        "kv_after":           kv.kv_len,
        "prefill_ms":         kv_metric["prefill_ms"],
        "decode_tokens":      kv_metric["decode_tokens"],
        "decode_ms":          kv_metric["decode_ms"],
        "tokens_per_sec":     kv_metric["tokens_per_sec"],
        "ttft_ms":            kv_metric["ttft_ms"],
        "tts_first_audio_ms": (t_tts_end - t_tts_start) * 1000,
        "e2e_to_first_audio_ms": (t_tts_end - t_e2e_start) * 1000,
        "response_text":      full_text.strip(),
    }


# ---------------------------------------------------------------------- #
# Stats                                                                   #
# ---------------------------------------------------------------------- #

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def summarize(turns: list[dict], cold_turns: int = 1) -> dict:
    """Compute p50/p95/p99 separately for cold (first N) and warm turns."""
    warm = turns[cold_turns:]

    def stats(values: list[float]) -> dict:
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "n": 0}
        return {
            "p50":  round(percentile(values, 50), 1),
            "p95":  round(percentile(values, 95), 1),
            "p99":  round(percentile(values, 99), 1),
            "mean": round(statistics.fmean(values), 1),
            "n":    len(values),
        }

    out = {"cold_turns": cold_turns, "warm_turns": len(warm)}
    for key in ("prefill_ms", "ttft_ms", "decode_ms",
                "tts_first_audio_ms", "e2e_to_first_audio_ms",
                "tokens_per_sec"):
        out[f"cold_{key}"] = stats([t[key] for t in turns[:cold_turns]])
        out[f"warm_{key}"] = stats([t[key] for t in warm])
    return out


def print_table(summary: dict) -> None:
    print("\n=== LATENCY SUMMARY ===")
    cold_n = summary["cold_turns"]
    warm_n = summary["warm_turns"]
    print(f"Cold turns: {cold_n} | Warm turns: {warm_n}\n")

    headers = ("Stage", "p50", "p95", "p99", "mean")
    fmt = "{:>22} | {:>8} | {:>8} | {:>8} | {:>8}"
    print(fmt.format(*headers))
    print("-" * 70)
    for key in ("prefill_ms", "ttft_ms", "decode_ms",
                "tts_first_audio_ms", "e2e_to_first_audio_ms"):
        s = summary[f"warm_{key}"]
        print(fmt.format(key, s["p50"], s["p95"], s["p99"], s["mean"]))
    print()
    tps = summary["warm_tokens_per_sec"]
    print(f"warm tokens/sec — p50: {tps['p50']} | mean: {tps['mean']}\n")


# ---------------------------------------------------------------------- #
# Entry                                                                   #
# ---------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="bench/prompts.json")
    parser.add_argument("--turns", type=int, default=None,
                        help="Override prompt count (default: all in file)")
    parser.add_argument("--quick", action="store_true", help="Run 5 prompts only")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if args.quick:
        prompts = prompts[:5]
    elif args.turns:
        prompts = prompts[:args.turns]

    print(f"Loading model: {args.model}")
    kv = KVCacheManager(model_path=args.model, n_ctx=4096, n_gpu_layers=-1)
    kv.load()
    kv.warm_up()

    print(f"Loading TTS: {TTS_PATH}")
    tts = TTS(model_path=TTS_PATH)

    print(f"\nMemory before warmup: {get_process_memory_mb():.0f} MB")
    print(f"Running {len(prompts)} prompts…\n")

    turns = []
    for i, prompt in enumerate(prompts, 1):
        t = run_one_turn(kv, tts, prompt)
        t["turn_index"] = i
        turns.append(t)
        print(
            f"[{i:>3}] prefill={t['prefill_ms']:>6.1f}ms  "
            f"ttft={t['ttft_ms']:>6.1f}ms  "
            f"decode={t['decode_ms']:>6.1f}ms  "
            f"tts1st={t['tts_first_audio_ms']:>6.1f}ms  "
            f"e2e={t['e2e_to_first_audio_ms']:>6.1f}ms"
        )

    peak_mem_mb = get_process_memory_mb()
    summary = summarize(turns, cold_turns=1)
    summary["peak_memory_mb"] = round(peak_mem_mb, 1)
    summary["model"] = args.model

    print_table(summary)
    print(f"Peak process memory: {peak_mem_mb:.0f} MB")

    # Persist results
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}-bench.json"
    out_path.write_text(
        json.dumps({"summary": summary, "turns": turns}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
