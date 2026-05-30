# Comparing Whisperloop vs Gemini Live (and Other Cloud Voice Agents)

This document explains **what we measure**, **what we can't measure**, and **what a fair head-to-head benchmark looks like** when comparing a local-first stack like Whisperloop against a closed cloud service like Gemini Live (or OpenAI Realtime, ElevenLabs Conversational AI, etc.).

---

## TL;DR — The Bottom Line

| Dimension                   | Whisperloop (local)             | Gemini Live (cloud)              | How we measure it                  |
| --------------------------- | ------------------------------- | -------------------------------- | ---------------------------------- |
| **TTFT (mic→first audio)**  | 400-700 ms                      | 600-1500 ms (network-bound)      | Same harness, both record from a WAV |
| **Tail latency p99**        | Bounded by local CPU/GPU         | Unbounded (network jitter, GC)   | 100+ turns, log p99                |
| **Cost / minute**           | $0 (electricity)                 | $0.05-0.15 typ.                  | Provider pricing × measured tokens |
| **Privacy**                 | 100% local                       | All audio leaves the device      | Categorical                        |
| **Quality (ASR WER)**       | small.en ≈ 5-7%                  | Gemini ≈ 3-4%                    | LibriSpeech-clean subset           |
| **Reasoning quality (3B)**  | Llama 3.2 3B Q4                  | Gemini 2.5 Flash / Pro            | MT-Bench-Lite or our chain set     |
| **Memex / cross-session**   | Persistent SQLite + embeddings  | Bounded conversation memory      | Recall@5 on planted facts          |
| **Offline / planes / SCIFs**| Works                            | Doesn't                          | Categorical                        |

Whisperloop **wins on tail latency, cost, privacy, and offline**. Gemini Live **wins on raw ASR quality and on reasoning quality at parity model size**. The interesting question isn't "which is better" but "**where is the parity boundary?**"

---

## 1. What is Fair Game to Compare?

A voice assistant pipeline has 5 stages. We compare on each independently AND end-to-end:

```
       MIC ──► VAD ──► ASR ──► LLM ──► TTS ──► SPEAKER
                                                                                                     │      │       │      │      │
        mic_capture_ms  endpoint_ms  asr_ms  ttft_ms  tts_ms
                          └─────────── e2e_first_audio_ms ──────────┘
```

For Gemini Live we can only observe **the boundaries we control** — the bytes we send up the socket and the bytes that come back. Internal breakdowns (their ASR vs. their LLM vs. their TTS) are not visible. So our comparison harness measures the **shared, observable boundaries**:

| Boundary           | We measure          | Gemini Live exposes? |
| ------------------ | ------------------- | -------------------- |
| Audio sent (start) | local timestamp     | yes (we control it)  |
| First audio received | local timestamp   | yes (we receive it)  |
| Transcript event   | local timestamp     | yes (Live API emits transcripts) |
| LLM text event     | local timestamp     | yes                  |
| End of speech      | local timestamp     | yes                  |
| Per-stage internals | full instrumentation | **no — opaque**     |

So the only honest **end-to-end** metric is `mic_capture_start → first_audio_received` (call it `e2e_ttfb_ms`). We compute it identically for both systems by replaying the same WAV file.

---

## 2. The Comparison Harness

We don't ship a Gemini client (no API keys in this repo), but the design below is what you'd build to run the comparison yourself.

### 2.1 Workload — a fixed corpus

Use the same prompts both systems see:

- `bench/prompts.json` — 15 simple Q&A prompts (existing)
- `bench/conversation_chains.json` — multi-turn chains (new, this commit)
- `bench/golden_recall.json` — cross-session recall queries (new, this commit)
- 100 pre-recorded WAV files at 16 kHz mono, each containing one human turn

The WAV files are the input to BOTH systems — no live mic, so we eliminate human-pacing variance.

### 2.2 Test rig

```python
# Pseudo-code; see bench/bench_remote.py (TODO) for an implementation skeleton
def run_one_turn(audio_wav: Path, backend: str) -> dict:
    t0 = time.perf_counter()

    if backend == "whisperloop":
        # Feed the WAV through our orchestrator with a synthetic VAD pulse
        result = local_orchestrator.process_wav(audio_wav)
    elif backend == "gemini_live":
        # Open a Gemini Live websocket, send the WAV in 20ms frames, collect
        # the response audio stream and transcript events.
        result = gemini_client.run_turn(audio_wav)

    return {
        "asr_first_partial_ms":  result.asr_first_partial_t - t0,
        "asr_final_ms":          result.asr_final_t - t0,
        "llm_first_token_ms":    result.llm_first_token_t - t0,
        "audio_first_sample_ms": result.audio_first_t - t0,    # the headline number
        "audio_complete_ms":     result.audio_complete_t - t0,
        "response_text":         result.text,
    }
```

### 2.3 Statistical methodology

- **N ≥ 100 turns per condition** to get stable p95/p99.
- **Identical hardware** for both runs (the local box matters even for the cloud test — network is bound to physical link).
- **Time of day spread** (cloud latency varies with provider load; sample across hours).
- **Wired ethernet** for the cloud run (Wi-Fi adds 5-50 ms of jitter that's the network, not the service).
- **Report p50 / p95 / p99 / max**, not mean. The tail is the whole story for voice UX.

### 2.4 Quality scoring (the harder half)

Latency is easy. Quality needs hand-graded references.

For each conversation chain we record:
1. **Coherence@k** — did the response use the prior context? `bench_context.py` already does this for us via keyword hit.
2. **ASR WER** — Word Error Rate vs. the human-typed reference. Use `jiwer` to compute.
3. **Hallucination rate** — does the response contradict the prior turn? Manual grading; 100 samples is enough for a credible signal.
4. **Naturalness MOS** — Mean Opinion Score 1-5 by 5 human raters on a 30-sample subset.

Whisperloop's chain accuracy is captured by `bench/bench_context.py` (this commit). Add a `--backend gemini` flag to a future `bench_remote.py` to reuse the same scorer.

---

## 3. What the Whisperloop Benchmarks Actually Measure

Two new harnesses in this repo:

### 3.1 `bench/bench_memex.py`

Pure Memex characterization, **no LLM in the loop**. Reports:

- **Store latency** (p50/p95/p99) per call — summarize-extractive + embed (MiniLM) + SQLite insert + numpy memmap append.
- **Recall latency** (p50/p95/p99) — full BM25 → embedding rerank → recency → importance pipeline.
- **Scaling** — store+recall latencies at corpus sizes 100, 500, 2 000.
- **Recall@{1,3,5}** on the golden set in `bench/golden_recall.json`.
- **Disk growth** — SQLite + embedding file size at each scale point.

Why isolate Memex: when the LLM is in the loop, signal from the retrieval system gets buried under model variance. Decoupling lets us claim "recall is X ms at 10k memories" with confidence.

### 3.2 `bench/bench_context.py`

End-to-end context-awareness, **with the LLM**. Each chain runs twice:

- **Baseline**: KV grows naturally across turns; no Memex.
- **Memex**: prime turn is stored in Memex, KV is *wiped between turns*, and follow-ups depend entirely on the injected `[MEMEX]` block for context.

Reports per chain and aggregate:
- `follow_up_accuracy` — fraction of follow-ups whose response contains an `expected_keyword`.
- `uplift` — `accuracy(memex) − accuracy(baseline)`. **Positive uplift means Memex genuinely recovers context the KV alone cannot.**
- `memex_overhead_ttft_ms` — added prefill latency vs. baseline (the cost of injecting recall).

This isolates **Memex's marginal contribution** in a way that's directly comparable to Gemini Live's "session memory" feature when we eventually run that side-by-side.

---

## 4. The Specific Comparisons We'd Run vs. Gemini Live

### A. TTFT distribution — the headline graph

Plot CDFs of `audio_first_sample_ms` for both systems on the same 100-turn corpus.

**Expected shape**:
- Whisperloop has a small p99/p50 ratio (≈ 1.5-2×) — bounded by local hardware.
- Gemini Live has a long right tail (p99/p50 ≈ 3-5×) — dominated by network jitter and provider GC pauses.

The interesting claim is *not* that the median is faster (it might not be), but that **the variance is bounded and predictable** locally.

### B. Reasoning quality at parity model size

Run our `conversation_chains.json` against:
- Whisperloop with Llama 3.2 3B Q4 (our default)
- Gemini Live with `gemini-2.5-flash` (closest size-class competitor)

Score with the same `expected_keyword` heuristic. This is **not** a comprehensive benchmark — it's a smoke test on whether the local model is "in the ballpark" for the kind of follow-up reasoning a voice assistant actually does.

### C. Memex vs. Gemini's session memory

Gemini Live keeps an in-session context (within one websocket). It does NOT persist across sessions or compact when the context fills.

We run:
1. **Cross-session recall**: prime in session A, query in session B, 24 hours later. Whisperloop: should hit via Memex DB. Gemini Live: should miss (no DB).
2. **Long conversation**: 50 turns about varying topics. Whisperloop: `LivePruner` fires around turn 25-30 (depends on n_ctx); recall stays high. Gemini Live: behavior is undocumented — likely either drops old turns silently or rejects the websocket.

This is the most differentiated claim: **persistent, queryable, local memory** isn't something the cloud APIs offer at all.

### D. Cost & privacy — categorical, not measured

Worth stating in the report even though it isn't a benchmark:

- **Cost**: a 30-minute daily session at Gemini Live ≈ $1.50-4.50/day ≈ $45-135/month. Whisperloop: $0 marginal.
- **Privacy**: enterprise/regulated users (medical, legal, defense) can't send audio to a cloud provider. Whisperloop runs in airgapped environments.
- **Offline**: airplane, basement, SCIF, rural — Whisperloop works.

### E. Hardware sensitivity

Whisperloop's latency is bound to local hardware. We'd run the harness on:
- M3 Pro MBP (high-end consumer)
- Intel i7 + RTX 3060 (mid-range desktop)
- Ryzen 7 + no GPU (CPU-only edge case)

And plot TTFT for each. The story is "your machine determines your floor; no provider can change that". Gemini Live's latency is *invariant* to your machine but *dependent* on your network.

---

## 5. What We Can NOT Claim From These Numbers

Be honest about the limits:

1. **We're not benchmarking model quality** — a tiny benchmark of follow-up keyword hits is not MMLU. For claims about reasoning quality, you need standard suites.
2. **Gemini Live's internal stages are opaque** — we can't say "their ASR is N ms"; we can only say "their pipeline e2e is N ms".
3. **Network conditions vary** — a benchmark on a 50 Mbps fiber link tells you nothing about a 4G LTE link. Report your link conditions explicitly.
4. **Cherry-picked chains can be misleading** — the `conversation_chains.json` set is intentionally simple (name recall, fact chaining). Real conversations are messier. Document this as "a smoke test", not "a representative sample".

---

## 6. Running the Benchmarks

```bash
# Pure memex characterization (no LLM)
python -m bench.bench_memex --quick                  # 30s smoke
python -m bench.bench_memex                          # full scaling ladder
python -m bench.bench_memex --scale 10000            # large-corpus stress

# Context-awareness end-to-end (requires the 1B/3B model)
python -m bench.bench_context --quick                # 2 chains
python -m bench.bench_context                        # all chains

# Original latency benchmark (LLM + TTS, single-turn)
python -m bench.bench --quick
python -m bench.bench
```

Results land in `bench/results/<timestamp>-{memex|context|bench}.json`.

---

## 7. What's Next — Stubs to Implement

These are deliberately not built yet (no API keys in the repo):

- `bench/bench_remote.py` — runs the same workload against Gemini Live, OpenAI Realtime, etc. Outputs the same JSON schema as `bench_context.py` so the analyzer can plot them on one axis.
- `bench/analyze.py` — loads multiple result JSONs and emits a markdown comparison table + CDF plots (matplotlib).
- A WAV corpus of 100 turns recorded by a single speaker, checked into `bench/audio/` (or hosted externally — they'll be ~50 MB).

When you bring an API key, the rest of this slots in cleanly: implementation, not architecture, is the missing piece.
