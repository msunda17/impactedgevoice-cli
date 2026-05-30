"""
bench_context.py — Context-awareness benchmark for the live conversation loop.

For each conversation chain in `conversation_chains.json`, we run TWO passes:

  A. BASELINE  — system + per-turn KV growth, NO memex injection.
                 Models pure in-cache context awareness (i.e. what the live
                 KV cache alone can remember).

  B. MEMEX     — same chain, but we wipe the KV between the prime and the
                 follow-ups, force memex.store/recall calls, and inject the
                 [MEMEX] block into the user turn. Models cross-session /
                 post-prune context awareness.

For each turn we record:

  * `ttft_ms`            — time to first decoded token after prefill starts
  * `decode_ms`          — first token to EOS
  * `prefill_ms`         — user tokens prefilled
  * `memex_block_chars`  — size of the injected [MEMEX] block (0 in baseline)
  * `recall_ms`          — time spent in memex.recall (0 in baseline)
  * `kv_after`           — KV occupancy after the turn
  * `hit`                — 1.0 if any `expected_keyword` is in the response

Aggregations:

  * accuracy@chain       — fraction of follow-up queries that hit
  * memex_uplift         — accuracy(MEMEX) - accuracy(BASELINE)
  * memex_overhead_ms    — added prefill+recall cost when memex is on

Usage:
    python -m bench.bench_context
    python -m bench.bench_context --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisperloop.kv_cache import KVCacheManager
from whisperloop.memex import Memex
from whisperloop.console import configure_logging

MODEL_PATH = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_CHAINS = "bench/conversation_chains.json"


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
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
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0}
    return {
        "p50":  round(percentile(values, 50), 1),
        "p95":  round(percentile(values, 95), 1),
        "mean": round(statistics.fmean(values), 1),
        "n":    len(values),
    }


def hit_for(response: str, expected: list[str]) -> bool:
    r = response.lower()
    return any(kw.lower() in r for kw in expected)


def run_turn(kv: KVCacheManager, prompt: str, max_tokens: int = 80) -> tuple[str, dict]:
    """One forward pass returning (response_text, timings)."""
    full = ""
    for tok in kv.generate(prompt, max_tokens=max_tokens):
        full += tok
    m = kv.turn_metrics[-1]
    return full.strip(), {
        "prefill_ms": m["prefill_ms"],
        "ttft_ms":    m["ttft_ms"],
        "decode_ms":  m["decode_ms"],
        "tokens_per_sec": m["tokens_per_sec"],
        "kv_after":   kv.kv_len,
    }


# ---------------------------------------------------------------------- #
# Per-chain runners                                                       #
# ---------------------------------------------------------------------- #

def run_chain_baseline(kv: KVCacheManager, chain: dict) -> dict:
    """Baseline: keep the KV cache growing across turns; no memex."""
    kv.restore_checkpoint("system")
    turns = []

    # Prime turn
    prime = chain["prime"]
    resp, t = run_turn(kv, prime["query"])
    turns.append({
        "role":   "prime",
        "query":  prime["query"],
        "resp":   resp[:200],
        "hit":    int(hit_for(resp, prime["expected_keyword"])),
        **t,
        "memex_block_chars": 0,
        "recall_ms": 0.0,
    })

    # Follow-ups
    for fu in chain["follow_ups"]:
        resp, t = run_turn(kv, fu["query"])
        turns.append({
            "role":   "follow_up",
            "query":  fu["query"],
            "resp":   resp[:200],
            "hit":    int(hit_for(resp, fu["expected_keyword"])),
            **t,
            "memex_block_chars": 0,
            "recall_ms": 0.0,
        })

    return {"chain": chain["name"], "mode": "baseline", "turns": turns}


def run_chain_memex(kv: KVCacheManager, memex: Memex, chain: dict) -> dict:
    """
    Memex mode: simulate a post-prune session. Store the prime in memex,
    then RESET the KV (to baseline) so the follow-ups can only succeed via
    the injected MEMEX block.
    """
    kv.restore_checkpoint("system")
    turns = []

    # Prime turn — store its (user, response) pair in memex; do not keep
    # it in the live KV cache.
    prime = chain["prime"]
    # Generate a response to the prime so we have something to store.
    resp, t = run_turn(kv, prime["query"])
    memex.session_id = "_chain_session"
    memex.store(prime["query"], resp, modality="text")
    # WIPE the live KV — this simulates a fresh session / post-prune state.
    kv.restore_checkpoint("system")
    memex.session_id = "_chain_session_followup"

    turns.append({
        "role":   "prime",
        "query":  prime["query"],
        "resp":   resp[:200],
        "hit":    int(hit_for(resp, prime["expected_keyword"])),
        **t,
        "memex_block_chars": 0,
        "recall_ms": 0.0,
    })

    # Follow-ups with memex injection
    for fu in chain["follow_ups"]:
        t_recall = time.perf_counter()
        memex_block = memex.recall_block(fu["query"], k=4)
        recall_ms = (time.perf_counter() - t_recall) * 1000

        full_prompt = f"{memex_block}\n\n{fu['query']}" if memex_block else fu["query"]
        resp, t = run_turn(kv, full_prompt)

        # Keep follow-ups isolated: reset between turns so MEMEX is the
        # sole source of context (strict test).
        kv.restore_checkpoint("system")

        turns.append({
            "role":   "follow_up",
            "query":  fu["query"],
            "resp":   resp[:200],
            "hit":    int(hit_for(resp, fu["expected_keyword"])),
            **t,
            "memex_block_chars": len(memex_block),
            "recall_ms":          round(recall_ms, 2),
        })

    return {"chain": chain["name"], "mode": "memex", "turns": turns}


# ---------------------------------------------------------------------- #
# Aggregation                                                             #
# ---------------------------------------------------------------------- #

def summarize_runs(runs: list[dict]) -> dict:
    by_chain: dict[str, dict] = {}
    for r in runs:
        cur = by_chain.setdefault(r["chain"], {"baseline": [], "memex": []})
        cur[r["mode"]].append(r)

    out = {"chains": []}
    base_ttft, memex_ttft = [], []
    base_pref, memex_pref = [], []
    base_acc_total, base_acc_hit = 0, 0
    memex_acc_total, memex_acc_hit = 0, 0
    recall_ms_all = []
    memex_block_chars_all = []

    for chain_name, modes in by_chain.items():
        baseline = modes["baseline"][0] if modes["baseline"] else None
        memex_r  = modes["memex"][0]   if modes["memex"]   else None

        def acc_of(run):
            fus = [t for t in run["turns"] if t["role"] == "follow_up"]
            if not fus:
                return 0.0, 0, 0
            hits = sum(t["hit"] for t in fus)
            return hits / len(fus), hits, len(fus)

        b_acc = b_hit = b_n = 0
        m_acc = m_hit = m_n = 0
        if baseline:
            b_acc, b_hit, b_n = acc_of(baseline)
            base_acc_hit += b_hit; base_acc_total += b_n
            for t in baseline["turns"]:
                base_ttft.append(t["ttft_ms"]); base_pref.append(t["prefill_ms"])
        if memex_r:
            m_acc, m_hit, m_n = acc_of(memex_r)
            memex_acc_hit += m_hit; memex_acc_total += m_n
            for t in memex_r["turns"]:
                memex_ttft.append(t["ttft_ms"]); memex_pref.append(t["prefill_ms"])
                recall_ms_all.append(t["recall_ms"])
                memex_block_chars_all.append(t["memex_block_chars"])

        out["chains"].append({
            "name": chain_name,
            "follow_up_accuracy": {
                "baseline": round(b_acc, 3),
                "memex":    round(m_acc, 3),
                "uplift":   round(m_acc - b_acc, 3),
            },
            "follow_up_hits": {
                "baseline": f"{b_hit}/{b_n}",
                "memex":    f"{m_hit}/{m_n}",
            },
        })

    out["overall"] = {
        "baseline_accuracy": round(base_acc_hit / max(1, base_acc_total), 3),
        "memex_accuracy":    round(memex_acc_hit / max(1, memex_acc_total), 3),
        "uplift":            round(
            memex_acc_hit / max(1, memex_acc_total)
            - base_acc_hit / max(1, base_acc_total), 3),
    }
    out["latency"] = {
        "baseline_ttft_ms": stats(base_ttft),
        "memex_ttft_ms":    stats(memex_ttft),
        "baseline_prefill_ms": stats(base_pref),
        "memex_prefill_ms":    stats(memex_pref),
        "recall_ms":           stats(recall_ms_all),
        "memex_block_chars":   stats([float(x) for x in memex_block_chars_all]),
    }
    # Memex overhead per turn (extra ms vs baseline)
    if base_ttft and memex_ttft:
        out["memex_overhead_ttft_ms"] = round(
            statistics.fmean(memex_ttft) - statistics.fmean(base_ttft), 1
        )
    return out


# ---------------------------------------------------------------------- #
# Entry                                                                   #
# ---------------------------------------------------------------------- #

def main() -> None:
    configure_logging(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--chains", default=DEFAULT_CHAINS)
    ap.add_argument("--quick", action="store_true",
                    help="Run the first 2 chains only")
    ap.add_argument("--data-root", default="bench/_context_bench_data")
    args = ap.parse_args()

    chains_doc = json.loads(Path(args.chains).read_text(encoding="utf-8"))
    chains = chains_doc["chains"]
    if args.quick:
        chains = chains[:2]

    data_root = Path(args.data_root)
    if data_root.exists():
        shutil.rmtree(data_root)

    print(f"Loading model: {args.model}")
    kv = KVCacheManager(model_path=args.model, n_ctx=4096, n_gpu_layers=-1)
    kv.load()
    kv.warm_up()

    runs: list[dict] = []
    print(f"\nRunning {len(chains)} chain(s) × 2 modes …\n")
    for chain in chains:
        # Baseline
        print(f"[{chain['name']}] baseline")
        runs.append(run_chain_baseline(kv, chain))
        # Memex (fresh memex per chain so memories don't leak across)
        memex = Memex(data_dir=str(data_root / chain["name"]))
        print(f"[{chain['name']}] memex")
        runs.append(run_chain_memex(kv, memex, chain))
        memex.close()

    summary = summarize_runs(runs)

    print("\n=== CONTEXT-AWARENESS SUMMARY ===")
    print(json.dumps(summary, indent=2))

    out_dir = Path("bench/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{ts}-context-bench.json"
    out_path.write_text(json.dumps({
        "summary": summary,
        "runs":    runs,
    }, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
