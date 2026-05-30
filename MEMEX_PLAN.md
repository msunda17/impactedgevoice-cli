# Memex — Implemented

> **STATUS: FULLY IMPLEMENTED** (all 6 phases complete). This document is preserved as the design record. For the current API and architecture, see `ARCHITECTURE.md § Day 6: Memex — Persistent + Live Memory`.

Vannevar Bush's "Memex" (1945) reimagined for local LLMs. Karpathy-style compact, queryable memory of every past conversation, document, and audio interaction — without bloating the active context window.

---

## 1. Design Principles

1. **Compact by default** — each memory is 50–200 tokens, not a raw transcript.
2. **Timestamped** — wall-clock time + monotonic sequence number.
3. **Topic-indexed** — extracted topic tags enable fast keyword recall.
4. **Hybrid retrieval** — BM25 keyword filtering + dense embedding similarity.
5. **Always queryable** — `recall(topic)` is called automatically before each LLM prefill.
6. **Decay-aware** — old memories get further compressed (summary-of-summaries).
7. **Single-user, local-first** — SQLite + memory-mapped numpy. No DB server.

---

## 2. System Architecture

### Conversation flow with Memex

- User submits query.
- `Memex.recall(query)` returns top-k relevant past summaries.
- Injector builds prompt: `[system_prompt] + [memex_block] + [recent_turns] + [user_query]`.
- LLM generates response.
- Turn-end hook calls `Memex.store(...)` to compact + embed + persist.

### Storage layout

- `memex_data/memex.db` — SQLite holding metadata, summaries, topics, importance.
- `memex_data/embeddings.npy` — numpy array of 384-dim vectors, memory-mapped.
- `memex_data/` is gitignored; created at runtime.

---

## 3. Database Schema

The `memories` table stores one row per turn (user query + assistant response pair).

Fields:
- `id` — primary key
- `timestamp` — Unix epoch
- `session_id` — groups turns of one conversation
- `turn_index` — sequence within session
- `modality` — voice / file / text
- `summary` — 50–200 token compact summary (the canonical recall payload)
- `topics` — JSON list, e.g. `["kv_cache", "barge_in"]`
- `importance` — float 0–1, used for decay weighting
- `raw_user` / `raw_response` — truncated originals (for audit/debug)
- `source_file` — file path if memory originated from a document
- `embedding_idx` — row index into `embeddings.npy`
- `parent_id` — set when this row is itself a summary-of-summaries
- `access_count` / `last_accessed` — for LRU-style eviction and importance boosting

A second `sessions` table tracks session-level summaries (one row per session).

Indexes on `(session_id, turn_index)`, `topics`, and `timestamp`.

---

## 4. Component Breakdown

### 4.1 `memex/storage.py` — Persistence Layer
- SQLite connection with WAL mode (crash-safe concurrent reads).
- NumPy memmap file for embeddings (`embeddings.npy`), loaded into memory for in-place append.
- Append-only embedding store; deletions are logical (memories marked `active=0`).

### 4.2 `memex/summarizer.py` — Memory Compaction
- **Hybrid design**: schema-guided LLM prompt for summarization, with extractive fallback.
- Three operations: `summarize_turn()`, `summarize_session()`, `summarize_pruned_block()`.
- Returns `MemorySummary(summary, topics, importance)`.
- **CRITICAL FIX IMPLEMENTED**: LLM-based summarization is **disabled on the shared conversation KV** because `llama_cpp.Llama.__call__` internally resets the KV cache, which corrupts the conversation state. The production path uses the extractive fallback. A dedicated summarizer model instance is the planned long-term fix.

### 4.3 `memex/embedder.py` — Semantic Indexing
- Model: `sentence-transformers/all-MiniLM-L6-v2` (90 MB, 384-dim).
- ~5 ms per embedding on CPU. Loaded once, kept warm.
- Vectors persisted to `embeddings.npy` via append + atomic rename.

### 4.4 `memex/retriever.py` — Hybrid Recall
Four-stage pipeline:
1. **BM25 prefilter** — keyword match on topics + summary text → top 50.
2. **Embedding rerank** — cosine similarity against query embedding → top 10.
3. **Recency boost** — multiply score by `exp(-age_days / 30)`.
4. **Importance weighting** — multiply by stored `importance`.

Returns the top-k after all four stages (default k = 5).

### 4.5 `memex/manager.py` — Public API
- `Memex.store(user_query, response, modality, source=None)` — async turn-end hook.
- `Memex.recall(query, k=5) -> list[Memory]` — sync, called pre-prefill.
- `Memex.summarize_session()` — runs at session close, builds session-level overview.
- `Memex.compact_old(days=30)` — recursively summarizes old memories into super-summaries.

### 4.6 `memex/injector.py` — Context Assembly
Builds the augmented system prompt block. Layout:

- System prompt (unchanged).
- A `MEMEX` block listing the top-k recalled summaries, each prefixed with relative time ("2 days ago", "1 hour ago") and topic tags.
- The current conversation transcript (recent turns held in the live KV cache).
- The user query.

The MEMEX block is capped at a hard token budget (e.g. 512 tokens) — if more memories qualify, lower-scoring ones are dropped.

---

## 5. Integration with Existing System

### 5.1 Orchestrator hook
After ASR produces the final transcript and before `kv.generate()`:
- Call `memex.recall(transcript)`.
- If memories are returned, prepend the `MEMEX` block to the user turn tokens.
- Critically: this means the **first turn** of a session has zero MEMEX overhead.

### 5.2 KV cache implications
The MEMEX block is **not** persisted in the KV cache across turns — it's regenerated each turn with fresh retrievals. This requires:
- Saving a checkpoint **before** the MEMEX block is prefilled.
- Truncating back to that checkpoint after the turn, then re-prefilling with the new MEMEX block on the next turn.

Trade-off: ~50–100 ms extra prefill per turn vs. perpetually growing cache. Worth it because MEMEX content changes turn-to-turn.

**Alternative**: keep MEMEX permanently in the cache and accept that stale memories linger. Simpler but lossier.

### 5.3 Document & audio file handling
When a file is processed via the adaptive router:
- The final summary is stored in Memex with `modality="file"` and `source_file=<path>`.
- Topics are extracted from the summary itself.
- Subsequent voice queries about that file can `recall()` it semantically.

This is the killer feature: ask the voice assistant a week later "what was that paper on speculative decoding?" and Memex returns the summary.

---

## 6. Memory Lifecycle & Decay

### Tiered retention
- **Hot tier** (0–7 days): full summary stored, high recall weight.
- **Warm tier** (7–30 days): full summary, reduced recall weight (decay factor).
- **Cold tier** (30+ days): rolled into a **super-summary** with siblings via `compact_old()`. Originals moved to `archive/`.

### Compaction job (`compact_old`)
Run on session start when DB has >1000 entries:
- Group memories older than 30 days by topic clusters.
- For each cluster, call the 2B model to summarize the summaries.
- Insert the super-summary with `parent_id` pointing to its children.
- Mark children as `archived=true` (still searchable but lower-priority).

### Importance scoring
Importance starts at 0.5 and is updated by:
- `+0.1` each time the memory is recalled (capped at 1.0).
- `+0.2` if the user explicitly references it ("remember when we talked about…").
- `-0.05` per month of zero access (slow decay toward 0).

---

## 7. Implementation Phases

### Phase 1 — Foundation ✅ COMPLETE
- `memex/storage.py` — SQLite schema, migrations, basic CRUD.
- `memex/manager.py` — minimal store/recall stubs, no retrieval intelligence.
- Wire into orchestrator: every turn calls `store()` after completion.

### Phase 2 — Summarization ✅ COMPLETE
- `memex/summarizer.py` — hybrid LLM + extractive fallback.
- Stores compact summaries with topics and importance.
- Note: LLM path disabled on shared KV; extractive fallback is production path.

### Phase 3 — Embeddings ✅ COMPLETE
- `memex/embedder.py` — MiniLM lazy-load, numpy memmap persistence.
- Graceful fallback to BM25-only when sentence-transformers unavailable.

### Phase 4 — Hybrid Retrieval ✅ COMPLETE
- `memex/retriever.py` — BM25 + embedding + recency + importance pipeline.
- Golden-set tests in `bench/golden_recall.json` verify top-ranked recall.

### Phase 5 — Context Injection ✅ COMPLETE
- `memex/injector.py` — assembles `[MEMEX]` block with relative timestamps.
- Orchestrator integration: pre-prefill recall, inject before generation.
- End-to-end demo: assistant recalls prior session/document context.

### Phase 6 — Live Pruning ✅ COMPLETE
- `memex/pruner.py` — live in-session KV cache pruning.
- When KV >80%, compacts oldest turns, stores in Memex, truncates KV.
- File mode: "primed" checkpoint protects document primer from pruning.

### Phase 7 — Lifecycle & Decay (partial)
- Importance updates on recall ✅ (bump +0.1).
- `compact_old()` super-summary job — *not yet implemented*.
- Background thread for session-end summarization — *not yet implemented*.
- CLI commands — *not yet implemented*.
- `--no-memex` flag — *not yet implemented*.

---

## 8. Testing Strategy

### Unit tests
- Storage: CRUD round-trip, embedding memmap correctness, schema migration.
- Summarizer: deterministic output on fixed seed, topic extraction sanity.
- Retriever: golden retrieval set — handcraft 20 query/expected-memory pairs.

### Integration tests
- End-to-end turn: store → close session → new session → recall → verify summary appears in MEMEX block.
- File-to-voice recall: summarize a PDF, then ask voice query about it next session.

### Performance tests ✅ (automated in bench/)
- `bench/bench_memex.py` — store/recall latency at 100, 500, 2 000 memories.
- Recall@k quality on golden set (`bench/golden_recall.json`).
- `bench/bench_context.py` — multi-turn context-awareness with/without Memex.
- Real numbers (BM25-only, Windows laptop): store p50 ≈ 0.23 ms, recall p50 ≈ 5.7 ms, recall@5 = 0.80.

### Failure modes to test
- DB corruption recovery (WAL replay).
- Embedding model unavailable → fall back to BM25-only.
- Summary LLM call fails → store raw turn with a `compaction_pending` flag.

---

## 9. Whiteboarding Defense (Why This Design)

**Q: Why SQLite + numpy instead of a vector DB like Chroma or LanceDB?**
For a single-user, single-machine assistant with <1 M memories, SQLite + memory-mapped numpy is faster (no IPC, no network), simpler (one file each), and zero-dependency. A vector DB is justified only when we hit multi-tenant or >10 M vectors.

**Q: Why a 2B model for summarization instead of the 9B?**
Each turn already pays the 2B cost for the response (in voice mode). Summarization is a "free" reuse — same model warm in memory. 9B summarization would more than double turn latency for marginal quality gain on 80-token outputs.

**Q: Why hybrid retrieval — isn't embedding similarity enough?**
Embedding similarity has weak grounding on rare proper nouns ("Whisperloop", "kv_cache_seq_rm"). BM25 catches those. The combo gives recall on both semantic paraphrases and specific terminology — the same architecture Google uses (BM25 prefilter, neural rerank).

**Q: Why per-turn re-prefill of the MEMEX block?**
Memories change every turn (new ones added, old ones recalled). Caching stale MEMEX content in the KV cache means the model is always one turn behind. The 50–100 ms extra prefill is the price of fresh recall. The system prompt and conversation history still live in the persistent cache — only the MEMEX block re-prefills.

**Q: What about privacy?**
All data is local — no cloud sync. Users can `wipe`, `clear-session`, or run with `--no-memex`. The archive directory holds raw transcripts but is gitignored and can be deleted independently of the summary DB.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Summarizer hallucinates facts into memory | Store both summary and raw — flag mismatches via embedding distance check |
| MEMEX block bloats prefill latency | Hard cap at ~1800 chars (~450 tokens) |
| Retrieval returns irrelevant old memories | Importance decay + recency boost + user feedback loop |
| DB grows unboundedly | `compact_old()` job + tiered retention — *not yet implemented* |
| Embedding model bias | MiniLM is general-purpose; consider domain-tuning later if needed |
| Crash mid-write corrupts state | SQLite WAL mode + atomic numpy append (write to `.tmp`, fsync, rename) ✅ |
| Summarizer corrupts conversation KV | Disabled LLM summarization on shared KV; extractive fallback ✅ |

---

## 11. Future Extensions

1. **Multi-session linking** — detect when two sessions discuss the same topic, merge into a "thread".
2. **Active learning** — let the user mark memories as important/forgettable.
3. **Cross-modal recall** — search by voice ("what did we say about X?") with VAD-triggered recall queries.
4. **Memex sharing** — export a session's MEMEX block as a portable JSON for handoff to another agent.
5. **Speculative recall** — start embedding the partial transcript during streaming ASR; memories are ready by the time speech ends.

---

## 12. File Layout (Implemented)

```
whisperloop/
  memex/
    __init__.py       # Public API (Memex, Memory)
    manager.py        # Memex.store(), .recall(), .recall_block()
    storage.py        # SQLite WAL + numpy memmap persistence
    summarizer.py     # Hybrid LLM + extractive memory compaction
    embedder.py       # Lazy MiniLM sentence embeddings (384-dim)
    retriever.py      # BM25 → embedding → recency → importance
    injector.py       # [MEMEX] context block builder
    pruner.py         # Live in-session KV cache pruning
memex_data/           # gitignored; created at runtime
  memex.db
  embeddings.npy
tests/                # (existing test files)
  test_kv_cache.py
  test_bargein.py
bench/
  bench_memex.py      # Memex store/recall latency + recall@k
  bench_context.py    # Multi-turn context-awareness benchmark
  golden_recall.json  # Hand-crafted query→memory pairs
  conversation_chains.json  # Multi-turn coherence test scenarios
```

---

## Next Action

All core phases (1–6) are implemented and tested. The next items from Phase 7 are:
1. `compact_old()` super-summary job for long-term DB compaction.
2. CLI commands for memex inspection (`stats`, `search`, `wipe`).
3. `--no-memex` runtime flag.
4. Dedicated summarizer model instance to re-enable LLM-based summarization without KV corruption.
