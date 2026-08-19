"""
bench_memex.py — Performance + quality benchmark for the Memex subsystem.

We measure four things end-to-end, **without** loading the LLM:

  1. STORE LATENCY        — full store path: summarize (extractive) + embed +
                             SQLite insert + numpy append. Per-call p50/p95/p99.
  2. RECALL LATENCY       — full retrieval pipeline: BM25 prefilter -> embedding
                             rerank -> recency boost -> importance weighting.
  3. RECALL QUALITY (@k)  — golden-set Recall@{1,3,5} on hand-crafted
                             paraphrased queries.
  4. SCALING              — how do (1) and (2) grow as the corpus grows
                             from 100 -> 1k -> 10k memories?

Why no LLM here:
  Memex summarization currently uses the extractive fallback when sharing the
  conversation KV (see impactedgevoice/memex/summarizer.py). The LLM-based path is
  benchmarked separately by bench_context.py once a dedicated summarizer
  model is wired in. Isolating memex from the model keeps these numbers
  comparable across runs and reproducible on machines that can't load llama.

Usage:
    python -m bench.bench_memex
    python -m bench.bench_memex --scale 10000   # large-corpus stress test
    python -m bench.bench_memex --quick         # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from impactedgevoice.memex import Memex


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
    return data[f] if f == c else data[f] + (data[c] - data[f]) * (k - f)


def stats(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "n": 0}
    return {
        "p50":  round(percentile(values, 50), 2),
        "p95":  round(percentile(values, 95), 2),
        "p99":  round(percentile(values, 99), 2),
        "mean": round(statistics.fmean(values), 2),
        "n":    len(values),
    }


# ---------------------------------------------------------------------- #
# Synthetic corpus for scaling tests                                      #
# ---------------------------------------------------------------------- #

_TOPICS = [
    "kv cache", "speculative prefill", "barge in", "tts streaming",
    "vad endpointing", "asr whisper", "model router", "memex retrieval",
    "document summarization", "rag pipeline", "embedding similarity",
    "bm25 keyword search", "checkpointing", "context window", "prompt template",
]


def _synth_pair(i: int) -> tuple[str, str]:
    """Deterministic synthetic (user, response) pair for corpus scaling."""
    topic = _TOPICS[i % len(_TOPICS)]
    user = f"Turn {i}: tell me about {topic}."
    response = (
        f"Here is what we discussed about {topic} in turn {i}. "
        f"It relates to local-first inference and low-latency pipelines, "
        f"with relevant numbers like {i * 7 % 1000} ms and entities like "
        f"ImpactEdgeVoice, llama.cpp, and faster-whisper."
    )
    return user, response


# ---------------------------------------------------------------------- #
# Benchmarks                                                              #
# ---------------------------------------------------------------------- #

def bench_store_recall(memex: Memex, n_store: int, n_recall: int) -> dict:
    """
    Time `n_store` store() calls and `n_recall` recall() calls against the
    resulting corpus.
    """
    store_times: list[float] = []
    for i in range(n_store):
        u, r = _synth_pair(i)
        t = time.perf_counter()
        memex.store(u, r, modality="text")
        store_times.append((time.perf_counter() - t) * 1000)

    # Force memex to think these are "prior" memories by clearing the
    # in-session flag so cross-session recall actually returns them.
    memex.session_id = "_other_session"

    recall_times: list[float] = []
    for i in range(n_recall):
        topic = _TOPICS[i % len(_TOPICS)]
        query = f"reminder about {topic}"
        t = time.perf_counter()
        _ = memex.recall(query, k=5)
        recall_times.append((time.perf_counter() - t) * 1000)

    return {
        "corpus_size":     n_store,
        "store_ms":        stats(store_times),
        "recall_ms":       stats(recall_times),
    }


def bench_recall_quality(memex: Memex, golden_path: Path) -> dict:
    """
    Plant the golden seeds, then run every query and check whether the
    top-k retrieved summaries contain ALL `relevant_keywords` for the item.
    Reports Recall@1, @3, @5 and the average rank of the first hit.
    """
    golden = json.loads(golden_path.read_text(encoding="utf-8"))["items"]

    # Plant the seeds (treat them as cross-session memories)
    memex.session_id = "_seed_session"
    for item in golden:
        memex.store(item["seed"]["user"], item["seed"]["response"], modality="text")
    memex.session_id = "_query_session"  # so seeds are reachable

    hits_at = {1: 0, 3: 0, 5: 0}
    first_hit_ranks: list[int] = []
    total_queries = 0
    per_item = []

    for item in golden:
        keywords = [k.lower() for k in item["relevant_keywords"]]
        item_hits = 0
        for q in item["queries"]:
            total_queries += 1
            results = memex.recall(q, k=5)
            rank = None
            for i, mem in enumerate(results, start=1):
                hay = (mem.summary + " " + " ".join(mem.topics)).lower()
                if all(kw in hay for kw in keywords):
                    rank = i
                    break
            if rank is not None:
                first_hit_ranks.append(rank)
                if rank <= 1: hits_at[1] += 1
                if rank <= 3: hits_at[3] += 1
                if rank <= 5: hits_at[5] += 1
                item_hits += 1
        per_item.append({
            "seed_topic":   item["seed"]["user"][:60],
            "queries_total": len(item["queries"]),
            "queries_hit":   item_hits,
        })

    avg_rank = round(statistics.fmean(first_hit_ranks), 2) if first_hit_ranks else None
    return {
        "queries":         total_queries,
        "recall_at_1":     round(hits_at[1] / total_queries, 3),
        "recall_at_3":     round(hits_at[3] / total_queries, 3),
        "recall_at_5":     round(hits_at[5] / total_queries, 3),
        "avg_rank_of_hit": avg_rank,
        "per_item":        per_item,
    }


def bench_scaling(data_root: Path) -> list[dict]:
    """Run store+recall benchmark at increasing corpus sizes."""
    sizes = [100, 500, 2000]
    out = []
    for size in sizes:
        run_dir = data_root / f"scale_{size}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        memex = Memex(data_dir=str(run_dir))
        r = bench_store_recall(memex, n_store=size, n_recall=min(200, size))
        # DB size on disk
        db = (run_dir / "memex.db")
        emb = (run_dir / "embeddings.npy")
        r["db_kb"]  = round(db.stat().st_size / 1024, 1) if db.exists() else 0
        r["emb_kb"] = round(emb.stat().st_size / 1024, 1) if emb.exists() else 0
        memex.close()
        out.append(r)
        print_block(f"Scaling @ corpus={size}", r)
    return out


# ---------------------------------------------------------------------- #
# Output                                                                  #
# ---------------------------------------------------------------------- #

def print_block(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------- #
# Entry                                                                   #
# ---------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="100-memory smoke test only")
    ap.add_argument("--scale", type=int, default=None,
                    help="Run one large-corpus test at this size instead of the default ladder")
    ap.add_argument("--golden", default="bench/golden_recall.json")
    ap.add_argument("--data-root", default="bench/_memex_bench_data")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if data_root.exists():
        shutil.rmtree(data_root)

    # ---- 1. Quality ---------------------------------------------------
    quality_dir = data_root / "quality"
    memex_q = Memex(data_dir=str(quality_dir))
    quality = bench_recall_quality(memex_q, Path(args.golden))
    memex_q.close()
    print_block("Recall Quality (golden set)", quality)

    # ---- 2. Scaling ---------------------------------------------------
    if args.quick:
        scaling = [bench_store_recall(Memex(data_dir=str(data_root / "quick")), 100, 50)]
        print_block("Quick: corpus=100", scaling[0])
    elif args.scale is not None:
        run_dir = data_root / f"scale_{args.scale}"
        memex = Memex(data_dir=str(run_dir))
        r = bench_store_recall(memex, n_store=args.scale, n_recall=200)
        memex.close()
        scaling = [r]
        print_block(f"Scale @ {args.scale}", r)
    else:
        scaling = bench_scaling(data_root)

    # ---- Persist ------------------------------------------------------
    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{ts}-memex-bench.json"
    out_path.write_text(json.dumps({
        "quality": quality,
        "scaling": scaling,
    }, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
