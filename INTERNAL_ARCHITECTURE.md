# Whisperloop Architecture & Knowledge Base

This document serves as a persistent record of the core concepts, prerequisites, and underlying architecture for each day's implementation of the Whisperloop project. As the project evolves from a synchronous pipeline to an asynchronous, highly optimized serving layer, this document will expand.

---

## Day 0 & Day 1: The Synchronous Baseline

### Objective
Establish the foundational pieces: Audio I/O, ASR (Automatic Speech Recognition), LLM Inference, and TTS (Text-to-Speech), wired together in a simple, synchronous, blocking pipeline.

### Prerequisite Knowledge & Under-the-Hood Mechanics

#### 1. Digital Audio Representation (`sounddevice` & `numpy`)
Audio processed by computers is fundamentally a continuous stream of numerical values (samples) representing the amplitude of sound waves at discrete intervals. 
- **Sample Rate**: The number of samples captured per second. Our microphone captures at **16,000 Hz (16kHz)**, the standard resolution required by most modern speech recognition models.
- **Data Types**: In Python, we represent this audio using `numpy` arrays of 32-bit floating-point numbers (`float32`) or 16-bit integers (`int16`), which allows for highly efficient matrix operations in memory.

#### 2. Automatic Speech Recognition (`faster-whisper`)
The ASR component is responsible for turning the raw audio arrays into text.
- **Model**: We use `faster-whisper` (running `tiny.en`), which utilizes CTranslate2 under the hood. CTranslate2 is a fast inference engine for Transformer models.
- **Mechanic**: In Day 1, the pipeline requires the *entire* 5-second audio chunk to be recorded and held in memory before it is passed to the transcription engine. The transcription engine processes the chunk and emits a final string.

#### 3. LLM Inference & The GGUF Format (`llama-cpp-python`)
We are running a 3 Billion parameter instruction-tuned model (`Llama-3.2-3B-Instruct`) locally. Doing this on consumer hardware requires specific optimizations:
- **GGUF (GPT-Generated Unified Format)**: This file format stores the model's weights in a quantized state. We use `Q4_K_M` (4-bit quantization), which mathematically compresses the precision of the model's weights. This allows a model that would normally require ~12GB of memory to fit comfortably into ~2.2GB of RAM/VRAM, with negligible loss in reasoning quality.
- **Prompt Engineering & Formatting**: LLMs don't just take raw text; they require specific control tokens to understand the conversational structure. We wrap our inputs in Llama 3 headers (e.g., `<|start_header_id|>user<|end_header_id|>`) so the model knows what context is the system prompt vs. user speech.
- **Inference Pipeline**: The LLM pre-fills the context (reading the prompt) and then autoregressively decodes (generates) tokens one by one until it hits a stop token (`<|eot_id|>`). In Day 1, this blocks the entire program until the final token is generated.

#### 4. Text-to-Speech (`piper-tts`)
The TTS component takes the generated text and turns it back into audio waveforms.
- **In-Memory Buffering**: Writing audio to a `.wav` file on a hard drive is slow (Disk I/O). To minimize latency, we use `io.BytesIO()` to synthesize the audio directly into an in-memory byte buffer, which is immediately converted back into a `numpy` array and handed to the speaker for playback.

#### 5. The "Ugly" Synchronous Blocking Pipeline
The architectural hallmark of Day 1 is its purely sequential, blocking nature:
1. `Record` runs for exactly 5 seconds.
2. `ASR` runs and blocks until transcription is complete.
3. `LLM` runs and blocks until the full sentence is generated.
4. `TTS` runs and blocks until the entire audio waveform is synthesized.
5. `Playback` blocks until the audio finishes playing through the speakers.

**Why this matters for Serving Layer Optimization:** This synchronous design is the baseline enemy. In a real-time system, these processes must overlap (stream). The LLM should start thinking before the user finishes speaking, and the TTS should start playing audio before the LLM finishes generating the final word.

#### 6. Whiteboarding Reinforcement — Day 1
- **Say**: "The baseline pipeline is the strawman — each stage blocks until complete. End-to-end latency is the *sum* of all stages, not the max. For a typical query that's record(5s) + ASR(500ms) + LLM(2s) + TTS(1s) ≈ 8.5s. The whole project is a sequence of overlaps that turn this sum into something closer to a max."
- **Numbers to memorize**: 16kHz audio = 32,000 bytes/sec (int16) or 64,000 bytes/sec (float32). Q4_K_M Llama 3.2 1B = ~1.3 GB on disk; in-memory it expands moderately with the KV cache (~80 KB/token at fp16).

---

## Day 2: Streaming + VAD (Persistent Interaction)

### Objective
Replace the fixed 5-second recording window with a continuously-running pipeline driven by Voice Activity Detection. Wire every stage together with `asyncio` queues so the system feels alive and reactive.

### Prerequisite Knowledge & Under-the-Hood Mechanics

#### 1. Voice Activity Detection — Silero VAD
VAD is the mechanism that answers: *"Is the user speaking right now?"* without requiring explicit user input (e.g., pressing a button).

- **Silero VAD** uses a small ONNX neural network (~1MB) trained to classify 512-sample audio windows as speech or not-speech, returning a confidence float between 0.0 and 1.0.
- **Threshold**: We declare speech "started" when confidence > 0.5. We declare speech "ended" when confidence drops below 0.5 for approximately 500ms. This hangover window prevents the VAD from triggering on a brief pause mid-sentence.
- **Why 512 samples?** At 16kHz, 512 samples = **32ms of audio** — the minimum window the Silero model requires for reliable classification. Smaller windows are too short to detect speech patterns; larger windows add latency.

#### 2. AsyncIO Event Loop Architecture
The core shift in Day 2 is from sequential Python to concurrent Python using `asyncio`.

- **`asyncio.Queue`**: The glue between stages. The microphone callback pushes raw audio chunks into a queue. The VAD coroutine reads from that queue. When the VAD triggers, the ASR and LLM coroutines are chained in. Queues are the primary tool to decouple producers and consumers without threads.
- **`asyncio.to_thread()`**: Python's asyncio is single-threaded. If we call `faster-whisper` or `llama.cpp` directly in a coroutine, it blocks the entire event loop. `asyncio.to_thread()` offloads blocking I/O-bound and compute-bound work to a thread pool, yielding control back to the event loop while the heavy computation runs in parallel.
- **`sounddevice.InputStream` with callback**: Instead of `sd.rec()` (blocking), we open a continuous stream that calls a Python callback every 512 samples. This callback pushes chunks to the queue without blocking.

#### 3. Streaming LLM Inference
`llama-cpp-python`'s `stream=True` parameter activates a **generator interface** — instead of waiting for all tokens, each sampled token is yielded immediately as it is generated on the GPU. This allows TTS to begin playing audio for the *first sentence* while the LLM is still generating the *second sentence*.

#### 4. Sentence-Boundary Chunked TTS
Piper TTS can only synthesize a complete text string — it can't synthesize token-by-token. The orchestrator buffers streaming LLM tokens and flushes to Piper whenever it sees a sentence boundary (`.`, `?`, `!`). This creates a pipeline effect: **TTS latency of sentence N overlaps with LLM generation of sentence N+1**.

#### 5. Whiteboarding Reinforcement — Day 2
- **Draw**: a producer-consumer diagram. mic-callback → audio_queue → VAD coroutine. token-stream → token_queue → sentence_buffer → tts_queue → playback worker. Highlight the queues — they're the seams between stages where buffering hides latency.
- **Say**: "asyncio is single-threaded. Heavy compute (whisper, llama_cpp) is offloaded to a thread pool via `asyncio.to_thread`. The event loop's only job is to shuffle data between queues. Queues + `to_thread` is enough for single-user serving — you only need `multiprocessing` if the GIL becomes a bottleneck, which it doesn't here because every heavy library releases the GIL in its C extension."
- **Key trade-off**: "Why sentence-boundary chunking instead of phrase-boundary? Piper requires a complete prosodic unit to produce natural intonation. Splitting on commas works but degrades quality. The sentence is the smallest unit Piper handles well, so it's the natural chunk size."
- **Common follow-up**: "What about the first TTS sentence — what's its latency?" → That's the dominant component of end-to-end TTFT and what we measure as `tts_first_audio_ms`. Piper synthesizes a short sentence in 80-150ms on CPU.

---

## Day 3: Persistent KV Cache (Crown Jewel #1)

### Objective
This is the defining technical achievement of the project. Implement raw, low-level KV cache persistence across conversation turns so the system prompt is only processed *once*, not re-processed at every turn.

### Prerequisite Knowledge & Under-the-Hood Mechanics

#### 1. The Transformer KV Cache — Why It Exists
Every Transformer layer performs **Attention** — each token attends to every previous token in the sequence. This requires computing Key and Value matrices for every prior token. During generation, this is expensive because you'd recompute all past tokens' K/V matrices for every new token.

The **KV Cache** is a memory buffer that stores the Key and Value matrices for every previously processed token. When new tokens arrive, the model only needs to compute K/V for the *new* tokens and read the rest from cache.

**The opportunity**: If the system prompt (e.g., 30 tokens) is prefilled once and its K/V matrices are kept in the buffer, subsequent turns with a fresh user question only need to process the new user tokens — not the system prompt again. This is a **10-20x speedup** on the prefill phase for multi-turn conversations.

#### 2. The Problem with Stateless APIs
The high-level `llama-cpp-python` `Llama()(prompt)` interface is *stateless*: every call re-encodes the entire conversation history from scratch. To get persistent caching, we must use the **low-level eval/sample loop**:

```
llm.eval(tokens)    →  Appends tokens to context, extends KV cache
llm.sample()        →  Samples the next token from current logits
```

This is the difference between "calling the LLM" and "owning the LLM context."

#### 3. Token Sequence Architecture
A Llama 3 conversation in raw token IDs looks exactly like:

```
[BOS] [START_HEADER] "system" [END_HEADER] "\n\n" <system text> [EOT]
[START_HEADER] "user"      [END_HEADER] "\n\n" <turn 1 user>  [EOT]
[START_HEADER] "assistant" [END_HEADER] "\n\n" <turn 1 asst>  [EOT]
[START_HEADER] "user"      [END_HEADER] "\n\n" <turn 2 user>  [EOT]
[START_HEADER] "assistant" [END_HEADER] "\n\n" <turn 2 asst>  [EOT]
...
```

The KV cache grows by exactly the number of tokens added in each `eval()` call. We track and assert this at every turn.

#### 4. The KV Length Invariant
After every turn, the following must hold exactly:

```
kv_len_after_turn_N == kv_len_after_turn_(N-1) + user_tokens + asst_header_tokens + asst_tokens + 1 (EOT)
```

A violation of this invariant means either:
- Tokens are being silently discarded (context overflow / truncation)
- The model state was corrupted

Our test suite (`test_kv_cache.py`) asserts this invariant after every turn. This is not paranoia — silent context corruption is the most common and hardest-to-debug failure mode in any serving system.

#### 5. Cache Truncation with `llama_kv_cache_seq_rm()`
The C-level API `llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)` removes tokens from position `p0` to `p1` in sequence `seq_id`. Setting `p1=-1` means "remove from p0 to the end."

This is used for **barge-in** (Day 4): when the user interrupts the assistant mid-sentence, we call `truncate_to(checkpoint_len)` to roll the KV cache back to before the assistant's response began. The model continues from that state, as if the incomplete response never happened.

#### 6. Latency Profile — What to Expect
| Turn | System Prefill | User Prefill | Decode |
|------|---------------|--------------|--------|
| 1    | ~200-400ms (once) | ~30-80ms | ~200-500ms |
| 2+   | 0ms (cached) | ~30-80ms | ~200-500ms |

**The headline**: Turn 1 prefill is 200-400ms (Llama 3.2 1B). Turn 2+ prefill is ~30-80ms. That's a 5-10x reduction on the prefill phase, directly improving Time-to-First-Token (TTFT) for every turn after the first.

**With Llama 3.2 3B**: Turn 1 prefill is ~400-800ms due to larger model size, but subsequent turns still benefit from the persistent cache.

#### 7. Whiteboarding Reinforcement — Day 3
When asked "explain the KV cache":
- **Draw**: a row of boxes labeled K1,V1 | K2,V2 | … per layer. Show that each new token (Kt,Vt) attends across all previous (K,V) entries.
- **Say**: "Without a cache, the cost per generated token is O(n) recomputation for n past tokens, per layer. With a cache, we read previous K/V from memory and only compute the new token's K/V — so per-token cost becomes O(1) in the past length (for the attention bookkeeping)."
- **Trade-off**: Memory grows linearly with sequence length: `2 × n_layers × n_heads × head_dim × seq_len × sizeof(dtype)`. For Llama 3.2 1B at fp16, this is ~80 KB per token. At 4096 ctx, that's ~320 MB just for the KV cache.
- **Common follow-up**: "What if the context overflows?" → Truncate from the head (drop oldest user/assistant turns), OR use a paged KV cache (vLLM), OR sliding-window attention (Mistral-style).

---

## Day 4: Barge-In + Conversational State Machine (Crown Jewel #2)

### Objective
Implement a clean interrupt mechanism so the user can speak over the assistant mid-sentence. This requires three coordinated pieces of cleanup: kill the in-flight LLM decode, flush the audio playback buffer, and roll back the KV cache to a pre-assistant checkpoint. All three must happen atomically; any one of them being missed produces an audibly broken experience.

### Architecture: The State Machine
We split conversational state out of the orchestrator into a dedicated controller (`bargein.py`). The orchestrator becomes a thin event dispatcher.

```
       ┌────────┐  vad_start    ┌───────────────┐  vad_end   ┌──────────┐
       │  IDLE  │ ─────────────▶│ USER_SPEAKING │ ──────────▶│ THINKING │
       └────────┘               └───────────────┘            └────┬─────┘
            ▲                          ▲                          │
            │ turn_complete            │ post_interrupt           │ first_tts_audio
            │                          │                          ▼
            │                   ┌──────────────┐               ┌──────────┐
            └───────────────────│ INTERRUPTED  │◀──────────────│ SPEAKING │
                                └──────────────┘   barge_in    └──────────┘
```

### Prerequisite Knowledge & Under-the-Hood Mechanics

#### 1. Why Cross-Thread Cancellation Is Hard
The LLM decode loop runs in a background thread (`asyncio.to_thread`). Python's GIL means the thread can't be "killed" externally — there is no `thread.kill()`. The only safe interrupt is **cooperative**: the worker periodically checks a flag.

We use `threading.Event` because:
- It is the cheapest cross-thread primitive (no event loop dependency).
- It works regardless of whether the worker is on the asyncio thread or a separate one.
- Checking `.is_set()` is a single C-level read — adds <1µs per token.

The decode loop in `kv_cache.py:generate()` checks `cancel_event.is_set()` once per token, immediately before sampling. Latency to honor a cancel signal is therefore one token-decode worst case (~5-10ms).

#### 2. The Three Cleanup Operations (Atomic Sequence)
On interrupt the controller performs these in order:

| # | Operation | Why this order |
|---|-----------|----------------|
| 1 | `cancel_event.set()` | Signal the LLM thread to stop *first* — otherwise it keeps appending tokens to the KV cache while we're trying to truncate it (race). |
| 2 | `sd.stop()` | Flush the speaker output buffer immediately. Stops audio bleeding into the mic during the rest of cleanup. |
| 3 | Drain `tts_queue` | Discard any sentences that were queued but not yet synthesized — otherwise the playback worker keeps speaking after the interrupt. |
| 4 | `task.cancel()` + `await task` | Raises `CancelledError` in the per-turn coroutine; we `await` to drain it cleanly before mutating shared state. |
| 5 | `kv.truncate_to(pre_assistant_checkpoint)` | Roll the cache back so the next turn doesn't think the partial assistant utterance happened. |

#### 3. The KV Cache Checkpoint — When to Take It
A checkpoint must be taken **after** the user prefill but **before** the assistant header is appended. The exact slot is in `_run_turn()` between `asr.transcribe()` and `_generate_and_stream()`. On interrupt, truncating to this checkpoint discards:
- The assistant header (`<|start_header_id|>assistant<|end_header_id|>\n\n`)
- All sampled assistant tokens
- Any EOT that may have been appended (none, since we skip it on cancel)

What is **preserved**:
- The system prompt (never touched)
- All prior completed turns
- The current turn's user message + its terminating EOT

This means the model's next turn sees a clean history that includes "user said X" but not "assistant started saying Y." This is critical for context coherence — if you preserve the partial assistant response, follow-up turns generate weirdness like "as I was saying…"

#### 4. Self-Trigger Suppression (The Bug You Saw Today)
Your earlier run showed the assistant transcribing its own TTS as new user input — 15 turns from 2 actual utterances. The naive fix ("disable mic during TTS") breaks barge-in. The correct fix is **debounced amplitude gating**:

- Mic continues capturing during `SPEAKING`.
- VAD-start signals are gated through `BargeInController.should_interrupt()`.
- A real interrupt requires: VAD-start AND chunk RMS > 0.02 AND sustained for ≥200ms.
- TTS audio bleeding through the mic typically has RMS ~0.005 (speaker volume × room acoustics × mic gain). Direct user speech is 0.05–0.3. The threshold sits cleanly in the gap.

This is a **single-microphone acoustic echo cancellation lite**. Real-world products (Alexa, Google Home) use multi-mic arrays with proper AEC. We trade quality for engineering simplicity.

#### 5. Whiteboarding Reinforcement — Day 4
When asked "how does barge-in work":
- **Draw**: the 5-state FSM (IDLE → USER_SPEAKING → THINKING → SPEAKING → INTERRUPTED) with edges labeled.
- **Say**: "Three things must happen atomically on interrupt: signal the decode thread via a `threading.Event` (cooperative cancellation, GIL-safe), flush the audio output ring buffer with `sd.stop()`, and truncate the KV cache to the checkpoint we saved right before the assistant header was appended."
- **Anti-pattern callout**: "We do NOT preserve the partial assistant utterance in the cache, because that corrupts the conversation context for the next turn. The user spoke, the assistant didn't get to reply — that's the truth the cache reflects."
- **Acoustic note**: "We do amplitude-debounced VAD gating during SPEAKING — sustained loud audio for 200ms — to reject our own TTS bleeding through the microphone. Proper AEC would need a multi-mic array."
- **Common follow-up**: "What if the user barges in but the decode already finished?" → State is `SPEAKING`, not `THINKING`; `cancel_event` is set but has no effect (decode loop already exited); we still drain the queue and truncate. Idempotent by design.
- **Common follow-up**: "How do you test cache truncation without a real LLM?" → `test_bargein.py` calls `kv._eval(junk_tokens)` to grow the cache deterministically, then verifies `truncate_to(checkpoint) == checkpoint_len` exactly.

---

## Day 5: Speculative Prefill — Overlapping ASR with LLM Inference (Crown Jewel #3)

### Objective
Reduce Time-to-First-Token (TTFT) by 100-300ms by starting LLM prefill *before* the user finishes speaking. This is the inference-system analog of CPU branch prediction: we speculate on the final transcript using stable partial ASR hypotheses, and either confirm the work or roll back on misprediction.

### The Problem: Sequential ASR → LLM Wastes Latency

In the naive pipeline:
```
[User speaks 1.5s] → [ASR 300ms] → [LLM prefill 50ms] → [First token]
                    ↑______________↑
                    Wasted wall-clock time!
```

The LLM sits idle while ASR runs. With speculative prefill:
```
[User speaks 1.5s]
  ├─ at 800ms: stable partial "How does it work" → start speculative prefill
  ├─ at 1200ms: still speaking...
  └─ at 1500ms: final ASR "How does it work exactly?"
     
[LLM prefill "How does it work" already done!] → [prefill "exactly?" 10ms] → [First token]
```

**Net TTFT reduction**: ~200-300ms (the ASR time that overlapped with prefill).

### Architecture: Three Components

#### 1. StreamingASR (Rolling Buffer Re-transcription)

Whisper has no native streaming API — it was trained on 30-second chunks. We approximate streaming:

```python
class StreamingASR:
    def feed(chunk):        # Append audio to rolling buffer
    def _loop():            # Every 400ms, re-transcribe entire buffer
    @property
    def stable_partial:     # Returns text if unchanged for >=2 runs AND >=3 words
```

**Key insight**: Re-transcribing the full buffer every 400ms is expensive (~150ms per call for short utterances) but acceptable because:
- It runs in a background thread — doesn't block the audio pipeline
- Beam=1 for partials (fast), beam=5 for final (accurate)
- The 400ms interval means we get ~2-3 partials during a typical 1-2s utterance

**Stable partial detection**:
```python
if text == last_partial:
    stable_count += 1
else:
    last_partial = text
    stable_count = 1

stable = stable_count >= 2 and len(text.split()) >= 3
```

We require ≥3 words to avoid speculating on "hey" or "the" — single-word partials are unstable and cause mispredicts.

#### 2. SpeculativePrefillController (The Speculation Engine)

Manages the lifecycle of a speculative prefill:

```python
class SpeculativePrefillController:
    def on_partial(text):     # Called when StreamingASR emits stable partial
        if first_valid_partial:
            save_checkpoint("pre_speculative")
            start_background_prefill_thread(text)
    
    def finalize(final_text): # Called at VAD end with final ASR
        if exact_match(final, speculative):
            # Jackpot! prefill already done, continue to generation
        elif prefix_match(final, speculative):
            # Keep common prefix, re-prefill only the suffix
        else:
            # Mispredict: truncate to checkpoint, prefill normally
```

**Checkpointing strategy**:
1. **Pre-speculative checkpoint**: Saved right before any speculative work. On mispredict, we truncate here and start fresh.
2. **Incremental speculation**: We only prefill user tokens — not the assistant header or response. This means even on exact match, we still need to add the assistant header and start sampling.

**Background prefill thread**:
- Runs `kv._eval(user_tokens)` in chunks of 32 tokens
- Checks `cancel_event.is_set()` between chunks for cooperative cancellation
- Non-blocking — the main asyncio loop continues capturing audio

**Match classification**:
| Result | Criteria | Action | Probability |
|--------|----------|--------|-------------|
| **Exact** | final == speculative (case-insensitive) | Continue from current state | ~60% |
| **Prefix** | final starts with speculative, len>=15 | Keep prefix, re-prefill suffix | ~25% |
| **Partial** | ≥50% word overlap | Rollback, re-prefill (optimization possible) | ~10% |
| **Mismatch** | No substantial overlap | Rollback to checkpoint, normal prefill | ~5% |

#### 3. Orchestrator Integration (Event Loop Wiring)

The main loop coordinates the three pieces:

```python
# VAD start → Start streaming ASR + speculation
if vad_start:
    streaming_asr = StreamingASR(asr, on_partial=spec.on_partial)
    streaming_asr.start()

# During USER_SPEAKING → Feed audio to streaming ASR
elif user_speaking:
    streaming_asr.feed(chunk)

# VAD end → Finalize and run turn
elif vad_end:
    final = streaming_asr.finalize()
    spec.finalize(final)  # Fixes KV cache if mispredict
    await run_turn_with_transcript(final)
```

### Correctness: Rollback on Mispredict

The critical requirement: **a wrong speculation must not corrupt the conversation state**.

**The invariant**: After `finalize()`, the KV cache must be in exactly the same state as if no speculation had occurred.

**Implementation**:
```python
# Before any speculation:
kv.save_checkpoint("pre_speculative")

# During speculation (background thread):
kv._eval(speculative_tokens)  # May be partially complete when VAD ends

# On finalize (mismatch):
kv.restore_checkpoint("pre_speculative")  # Atomic rollback
kv.generate(final_text)  # Clean prefill
```

The `restore_checkpoint()` uses `llama_kv_cache_seq_rm()` to truncate the cache to the saved sequence length. This is O(1) — just updating pointers — regardless of how many tokens were speculatively added.

### Latency Analysis: When It Helps

**Best case** (exact match, typical short queries like "How does it work?"):
- User speaks: 1200ms
- Streaming ASR stable at: 800ms
- Speculative prefill during: 800-1200ms
- Final ASR (beam=5): 300ms (overlaps with prefill)
- **TTFT**: ~50ms (just assistant header + first sample)
- **Savings**: ~250ms vs non-speculative

**Worst case** (mismatch, user changes mind mid-sentence):
- Speculation runs on "Explain quantum..."
- User actually says: "Never mind, tell me the weather"
- Rollback cost: ~5ms (truncate call)
- Re-prefill: ~30ms
- **Net cost**: +35ms vs non-speculative

**Expected improvement**:
- Typical voice queries: 3-8 words, spoken clearly
- Stable partial accuracy: ~85% exact or prefix match
- Average TTFT reduction: **150-200ms**

### Code Structure

```
whisperloop/
├── asr.py                    # ASR + StreamingASR
├── speculative_prefill.py    # SpeculativePrefillController (NEW)
├── orchestrator.py           # Integration point
└── kv_cache.py               # Checkpoint/restore primitives
```

### Whiteboarding Reinforcement — Day 5

**The 60-second pitch**:
> "Speculative prefill overlaps ASR with LLM prefill. We use a streaming ASR wrapper that re-transcribes a rolling buffer every 400ms. When we see a stable partial (unchanged for 2+ runs, 3+ words), we optimistically start pre-filling the KV cache with those tokens in a background thread. When the final ASR arrives, we compare: exact match means we saved the full prefill time; prefix match means we keep the common part; mismatch means we roll back to a checkpoint and prefill normally. The rollback is atomic using `truncate_to()`, so correctness is guaranteed."

**Key trade-offs**:
- **CPU cost**: Streaming ASR uses ~15% of one core for the 400ms polling. Acceptable for the latency win.
- **Memory**: The rolling audio buffer holds 1-2s of float32 audio (~64-128KB). Negligible.
- **Mispredict penalty**: ~30-50ms of wasted prefill work plus 5ms rollback. Rare enough that expected value is positive.

**Common follow-up**: "Why not use Whisper's streaming mode?" → There isn't one. The encoder processes 30-second frames; there's no incremental state to persist. We approximate streaming via overlapping re-transcription, which is the industry-standard workaround (see whisper.cpp streaming example).

---

## Day 6: Memex — Persistent + Live Memory (Crown Jewel #4)

### Objective
Give the assistant a long-lived, queryable memory that spans **across sessions** AND survives **mid-session KV cache pruning** when conversations or document summaries blow past the active context window.

### The Two Problems Memex Solves

**Problem 1 — Cross-session recall**: A user asks today "what was that paper on speculative decoding?" The assistant should recall a summary they discussed last week — without that summary having to live forever in the KV cache.

**Problem 2 — In-session context overflow**: When the user has had a 20-turn conversation, or summarized a 50-page PDF, the live KV cache approaches its `n_ctx` limit. The naive failure mode is a hard error or runaway memory. The Memex solution is graceful: compact the oldest turns into a summary, store it in Memex, truncate the KV cache, and surface the summary back in subsequent prompts.

### Architecture: 6 Components

```
┌──────────────────────────────────────────────────────────────────┐
│  Orchestrator (per-turn loop)                                    │
│    ├─ pre-prefill:  Memex.recall_block(query) → MEMEX block     │
│    ├─ if KV >80%:   Pruner.prune_if_needed()  → compact + store │
│    └─ post-turn:    Memex.store(user, response) (async)         │
└──────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
         ┌─────────────┐                  ┌──────────────┐
         │   Memex     │                  │  LivePruner  │
         │ (manager)   │                  │              │
         └──┬──────┬───┘                  └──────┬───────┘
            │      │                             │
   ┌────────▼─┐  ┌─▼──────────┐         ┌────────▼────────┐
   │ Storage  │  │ Retriever  │         │  Summarizer     │
   │ (SQLite+ │  │  (BM25 +   │         │  (LLM, hybrid   │
   │  numpy)  │  │  embeds)   │         │   prompt schema)│
   └──────────┘  └────────────┘         └─────────────────┘
                       │
                ┌──────▼──────┐
                │  Embedder   │
                │  (MiniLM)   │
                └─────────────┘
```

### Component Detail

#### 1. `memex/storage.py` — Persistence
- **SQLite WAL** for `memories` and `sessions` tables — crash-safe, concurrent-read.
- **NumPy memmap** for the embedding matrix (`embeddings.npy`) — append-only, fast load.
- One row per turn with: `summary, topics[], importance, modality, timestamp, embedding_idx, raw_user, raw_response`.
- Indexes on `(session_id, turn_index)` and `timestamp` for cheap session-scoped queries.

#### 2. `memex/summarizer.py` — Compaction
**Hybrid design**: schema-guided LLM call. We do **not** hardcode token weights — we tell the LLM what fields to fill (intent, entities, key facts, topics, importance) and let it choose salience.

Three operations:
- `summarize_turn(user, response)` → 60-120 word summary + 3-5 topic tags + importance.
- `summarize_session(turn_summaries)` → 100-200 word session rollup at session close.
- `summarize_pruned_block(turn_pairs)` → 120-200 word summary of evicted turns (used by `LivePruner`).

The LLM is the **simple-tier model already loaded** for voice — no extra model load. Falls back to extractive (regex-based) summarization if the LLM call fails so Memex never blocks the conversation.

#### 3. `memex/embedder.py` — Semantic Indexing
- Model: `sentence-transformers/all-MiniLM-L6-v2` (90 MB, 384 dims).
- ~5 ms per embedding on CPU. Lazy-loaded.
- Vectors are stored normalized → cosine similarity becomes a single dot product.
- **Graceful degradation**: if `sentence-transformers` isn't installed, Memex falls back to BM25-only retrieval with no functional break.

#### 4. `memex/retriever.py` — Hybrid Recall
Four-stage pipeline:

| Stage | What | Why |
|-------|------|-----|
| **1. BM25 prefilter** | Keyword score on (summary + topics) → top 50 | Catches rare proper nouns and exact terms |
| **2. Embedding rerank** | Cosine vs query embedding → top 20 | Catches semantic paraphrases |
| **3. Recency boost** | `× exp(-age_days / 30)` | Recent memories more likely relevant |
| **4. Importance** | `× (0.5 + importance)` | User-feedback weighted |

Final top-k (default 5) returned. Access bumps `importance` by +0.1 (capped at 1.0) — recall reinforces utility.

#### 5. `memex/injector.py` — Context Assembly
Builds an injectable string with relative timestamps and topic tags:

```
[MEMEX — relevant prior memories]
- (2d ago | topics: kv_cache, barge_in) User asked about KV truncation; explained...
- (45m ago | topics: pdf, summarization) Summarized a paper on…
[LIVE SESSION CONTEXT — pruned from active conversation]
- earlier this session: Discussed clock replacement algorithm and LRU…
[/MEMEX]
```

Hard-capped at ~1800 chars (~450 tokens) to bound prefill latency. Lower-scoring memories drop first.

#### 6. `memex/pruner.py` — Live In-Session KV Pruning
**The killer feature for long conversations.** When `kv.kv_len / n_ctx >= 0.80`:

1. **Estimate**: Compute how many oldest turns to drop to bring occupancy back to 40%.
2. **Compact**: Send those turn pairs to `Summarizer.summarize_pruned_block()`.
3. **Persist**: Store summary in Memex with `modality="prune"` (also added to `_live_summaries`).
4. **Truncate**: `kv.restore_checkpoint(baseline)` — rolls back to system prompt OR the document primer in file mode.
5. **Inject on next turn**: The MEMEX block automatically picks up the live summary.

Critical invariant: **the pruner never truncates below `baseline_checkpoint`** — that's the system prompt (`"system"`) for voice mode, or system+document primer (`"primed"`) for file mode. Document context is sacred.

### Integration Points in the Orchestrator

```python
# In _run_turn_with_transcript():

# 1. Prune BEFORE adding new turn (free space in KV)
self.pruner.prune_if_needed()

# 2. Build MEMEX context block
memex_block = self.memex.recall_block(transcript, k=4)
prefill_text = f"{memex_block}\n\n{transcript}"

# 3. Generate as normal
response = await self._generate_and_stream(prefill_text)

# 4. Record in pruner log + async store in Memex
self.pruner.record_turn(transcript, response)
asyncio.create_task(asyncio.to_thread(
    self.memex.store, transcript, response, "voice"
))
```

### Should We Define Summarization Logic Manually?

**No** — schema-guided LLM summarization is the right call. Reasoning:

| Approach | Quality | Latency | Robustness |
|----------|---------|---------|------------|
| **Pure rules** (TF-IDF, NER) | Misses intent | <1ms | Fails on paraphrasing |
| **Pure freeform LLM** | Best | ~150ms | Schema violations |
| **Hybrid (we use this)** | High | ~150ms | Bounded length, key fields guaranteed |

The structured prompt (`_SUMMARY_PROMPT` in `summarizer.py`) tells the LLM:
- **What to capture**: intent, entities, numbers, file names, technical terms.
- **What to drop**: pleasantries, hedging, repetition.
- **Output format**: strict JSON with `summary`, `topics`, `importance`.

The LLM picks **which tokens are salient** (the genuinely hard part) but the prompt enforces **consistent structure**. Output is parsed and falls back to extractive summarization on any parse failure.

### Latency Profile

| Operation | Cost | When |
|-----------|------|------|
| `recall_block()` | 5-30ms (BM25+embed lookup) | Every turn |
| `store()` (async) | 100-200ms LLM + 5ms embed | Every turn (off hot path) |
| `prune_if_needed()` | 0ms (just check occupancy) | Every turn |
| Actual prune | 200-400ms summarization | When KV fills (rare) |

**Net per-turn cost**: ~10-30ms added to TTFT (recall lookup + MEMEX block prefill). Storage is async so it never blocks.

### Privacy & Data Layout
```
memex_data/                 # gitignored, all-local
├── memex.db               # SQLite WAL
└── embeddings.npy         # numpy memmap, 384 × N float32
```

CLI controls (planned): `whisperloop memex stats`, `whisperloop memex search`, `whisperloop memex wipe`, `--no-memex` flag.

### Whiteboarding — Day 6

**The 60-second pitch**:
> "Memex gives the assistant memory that survives both session boundaries and live KV pressure. Every turn, we recall the top-k relevant prior memories via hybrid BM25+embedding retrieval and inject them as a bounded MEMEX block before prefill. Every turn we also store a schema-guided LLM summary of the (user, response) pair into SQLite + a numpy memmap of MiniLM embeddings. When the live KV cache crosses 80% occupancy, a Pruner compacts the oldest turns into one Memex memory, truncates the KV back to the document primer baseline, and the next turn's MEMEX block carries that summary forward. Net result: infinite-feeling context with bounded prefill latency."

**Common follow-up**: "Why not just keep the KV growing?" → Hard cap is `n_ctx` (model architecture limit). Past that, you OOM or silently truncate. Pruning lets us keep talking past that limit.

**Common follow-up**: "How is this different from RAG?" → RAG queries an external doc store. Memex queries *itself* — every prior turn is a document. Storage and embedding pipeline is identical to a RAG system, but the corpus is the conversation history.

---

## Latency Instrumentation & Benchmarking

### Objective
You cannot defend latency numbers you didn't measure. Every stage boundary writes a structured event to `bench/latency.jsonl`. The benchmark script (`bench/bench.py`) aggregates these into p50/p95/p99 distributions.

### Why p50/p95/p99 Instead of "Mean"
- **p50 (median)** describes the typical user experience.
- **p95** describes the bad-but-not-rare case — what 1 in 20 turns feels like.
- **p99** describes tail latency, where local inference wins over cloud (no network jitter, no GC pause, no shared-tenant contention).

A mean is misleading because LLM latency is bimodal: cold turn (first one, no cache) is 2-5× warm. Always report cold and warm separately.

### Events Emitted Per Turn
```
speech_start        — VAD start (t=0)
speech_end          — VAD end
asr_done            — final transcript ready          (asr_ms)
llm_prefill_start   — user tokens enter the cache
llm_first_token     — TTFT marker
llm_decode_done     — assistant EOS or cancel
tts_first_audio     — first audio sample queued
tts_done            — last audio sample played
barge_in            — user interrupted (if any)
turn_complete       — fully done
```

End-to-end TTFT = `tts_first_audio.ts - speech_end.ts`. This is the number that goes in the README headline.

### Whiteboarding Reinforcement — Benchmarking
- **Say**: "I log a JSONL event at every stage boundary in the pipeline. The bench script reads these and computes p50/p95/p99 separately for cold and warm turns. Cold means turn 1; warm is turns 2+. I report both because the user feels the warm experience after the first interaction."
- **Anti-pattern**: "Mean latency lies. Voice systems are bimodal — there's a fat tail of unfortunate GC/contention spikes that the mean drowns out."

---

## Context-Awareness & Memex Benchmarking

The single-turn `bench.py` measures the **inference pipeline**; it does not measure the assistant's ability to **remember** or its retrieval system. Two additional harnesses fill that gap:

### `bench/bench_memex.py` — Memex System Characterization (no LLM)

Pure characterization of the retrieval subsystem. Isolating Memex from the LLM lets us claim concrete latency numbers without the model variance burying the signal.

**Metrics**:
| Metric | Description |
|---|---|
| `store_ms` | Full store path: extractive summary + MiniLM embed + SQLite insert + numpy append (p50/p95/p99). |
| `recall_ms` | BM25 prefilter → embedding rerank → recency decay → importance weighting (p50/p95/p99). |
| `recall_at_{1,3,5}` | Golden-set recall — hand-crafted `query → expected_memory` pairs in `bench/golden_recall.json`. A "hit" requires ALL `relevant_keywords` to appear in the retrieved summary. |
| `avg_rank_of_hit` | Average position of the first correct memory in the top-5. Lower is better. |
| `db_kb`, `emb_kb` | Disk footprint at each corpus size. |

**Scaling ladder**: corpus sizes 100 → 500 → 2 000 (extensible via `--scale 10000`). Each step is a fresh DB so the timings are clean.

**Why no LLM**: Memex's LLM-summarization path is currently disabled on the shared conversation KV (corrupts conversation state — see `summarizer.py:_llm_call`). The extractive fallback is what actually runs in production today, so that's what we benchmark. When a dedicated summarizer model is wired in, this same script will accept a `--with-llm` flag.

**Expected numbers** (from a quick run, BM25-only on a Windows laptop):
- `store_ms`: p50 ≈ 0.2 ms, p99 ≈ 1 ms
- `recall_ms`: p50 ≈ 6 ms, p99 ≈ 10 ms at corpus=100
- `recall_at_5`: 0.80 on the golden set (room to grow with embeddings on)

### `bench/bench_context.py` — End-to-End Context-Awareness (with LLM)

For each conversation chain in `bench/conversation_chains.json`, runs **two passes**:

1. **Baseline**: KV cache grows naturally across turns; no Memex involvement.
2. **Memex**: prime turn stored in Memex, KV cache **wiped between turns**, follow-ups depend entirely on the injected `[MEMEX]` block.

**Per-turn metrics**:
- `ttft_ms`, `prefill_ms`, `decode_ms`, `tokens_per_sec`, `kv_after` — standard pipeline timings.
- `memex_block_chars` — size of the injected block (0 in baseline).
- `recall_ms` — time spent in `memex.recall_block()`.
- `hit` — 1 if any `expected_keyword` appears in the response.

**Aggregations**:
| Metric | What it tells us |
|---|---|
| `baseline_accuracy` | Pure KV-cache context awareness — what the model can answer when prior turns are still in the live cache. |
| `memex_accuracy` | Post-prune context awareness — what Memex alone can recover. |
| `uplift` | `memex_accuracy − baseline_accuracy`. Positive uplift = Memex genuinely recovers context the KV alone cannot. **This is the headline number for memex.** |
| `memex_overhead_ttft_ms` | Added prefill latency from the injected block. The cost of recall. |
| `recall_ms` | Recall latency in the live loop (should match `bench_memex.py`'s isolated numbers). |

**The chain design** (in `conversation_chains.json`) covers five context-shapes:
- `name_recall` — entity recall ("what's my name?")
- `preference_recall` — long-tail user preferences
- `fact_chaining` — numeric facts that must be transformed ("double the number I told you")
- `document_followup` — voice questions about a previously summarized document
- `topic_pivot` — connecting back to an earlier subject after a tangent

Each runs as one prime + 2-3 follow-ups, so we get 10-15 follow-up data points per pass per chain set.

### Strategy: Comparison with Gemini Live (and other cloud voice agents)

Full strategy doc: `bench/gemini_comparison.md`. The key points:

#### What is fair to compare?

Cloud APIs are **opaque** — we can see `audio_sent → audio_received` boundaries but not internal stages. So a head-to-head comparison must use **identical, observable boundaries**:

| Boundary | Whisperloop measures | Gemini Live exposes |
|---|---|---|
| Audio sent (start) | yes | yes (we control) |
| First audio received | yes | yes (we receive) |
| Transcript event | yes | yes (Live API emits it) |
| First LLM token | yes | yes |
| Per-stage internals | yes (full instrumentation) | **no — opaque** |

The only honest end-to-end metric: `e2e_ttfb_ms = mic_capture_start → first_audio_received`, computed identically by feeding the SAME pre-recorded WAVs through both backends.

#### What the comparison should measure

| Dimension | How |
|---|---|
| **TTFT p50/p95/p99** | Same 100 WAVs through both; report CDFs not means. |
| **Tail bound** | p99/p50 ratio — local should be < 2×, cloud often 3-5× (network jitter). |
| **ASR WER** | `jiwer` on LibriSpeech-clean subset. |
| **Reasoning quality** | `bench_context.py` chain accuracy at parity model size (3B Q4 vs Gemini Flash). |
| **Cross-session memory** | Plant in session A, query in session B 24h later. Whisperloop hits via Memex; Gemini Live structurally cannot. |
| **Long-conversation handling** | 50-turn varying-topic test. Whisperloop: `LivePruner` fires; recall stays high. Gemini Live: behavior undocumented. |
| **Cost** | Provider $/min × measured tokens. Local = $0. |
| **Privacy / offline** | Categorical (not measured — stated). |

#### Where Whisperloop wins, structurally

- **p99 latency is bounded** by local hardware, not by a shared-tenant network.
- **Persistent cross-session memory** — Gemini Live keeps an in-session context only; Memex survives reboot.
- **Live KV pruning** — Whisperloop handles infinite-length conversations gracefully via `LivePruner`; cloud APIs typically drop the websocket past their context limit.
- **$0 marginal cost, fully offline, no audio leaves the device.**

#### Where it doesn't, and we should be honest

- ASR small.en at ~5-7% WER vs Gemini's ~3-4%.
- Reasoning quality at 3B Q4 trails Gemini 2.5 Flash on hard prompts. The chain set in this repo is intentionally lightweight; a real claim about reasoning quality needs MMLU/MT-Bench, not keyword-match heuristics.

#### What we can NOT claim from the numbers in this repo

1. **Internal stage timings for Gemini** — we don't have them.
2. **Universal latency comparison** — network conditions are local; results vary by link.
3. **General reasoning quality** — `conversation_chains.json` is a smoke test, not MMLU.
4. **Quality of cloud TTS** — that needs MOS rating with human listeners.

#### What's stubbed for future work

- `bench/bench_remote.py` — same workload, `--backend gemini|openai|elevenlabs` flag. Not in this repo (no API keys).
- `bench/analyze.py` — multi-result joiner that produces markdown tables + matplotlib CDF plots.
- A 100-WAV speaker-controlled corpus checked into `bench/audio/`.

The architecture is ready; the missing piece is **implementation, not design**.

### Whiteboarding Reinforcement — Comparison Strategy

> "The honest comparison is **CDF of e2e_ttfb_ms** on a fixed 100-WAV corpus through both pipelines. Means lie. We report p50, p95, p99 separately. The win for local isn't necessarily a faster median — it's a **bounded p99**, plus structural wins on persistent memory, cost, offline, and privacy that the cloud APIs cannot match by construction. The wins for cloud are ASR quality and raw reasoning at parity size. The interesting question is 'where is the parity boundary?' and `bench_context.py` is built so the same scorer drops onto a Gemini backend the day we wire up the API client."

---

## Metrics Calculation Reference

When the orchestrator logs metrics at each turn, these four numbers are computed in `kv_cache.py:generate()`:

### 1. Prefill ms
**What it measures**: Time to process ("prefill") the user's transcript tokens into the KV cache.

**Calculation**:
```python
t_prefill_start = time.perf_counter()
self._eval(user_tokens)           # runs llama_cpp eval (forward pass, no sampling)
t_prefill_end = time.perf_counter()
prefill_ms = (t_prefill_end - t_prefill_start) * 1000
```

**Why it matters**: This is the "thinking before speaking" latency. With persistent KV cache, turn 2+ prefill is just the new user tokens (~5-30ms), not the full conversation history.

**Typical values**:
- Turn 1 (cold): includes system prompt prefill — ~200-400ms
- Turn 2+ (warm): user tokens only — ~5-30ms for short queries

---

### 2. TTFT ms (Time To First Token)
**What it measures**: Wall-clock time from the **start of user prefill** to the **first decoded assistant token**.

**Calculation**:
```python
t_prefill_start = time.perf_counter()   # same start as prefill
# ... prefill happens ...
for _ in range(max_tokens):
    token_id = self._sample()           # first call sets t_first_token
    if t_first_token is None:
        t_first_token = time.perf_counter()
    ...
ttft_ms = (t_first_token - t_prefill_start) * 1000
```

**Critical distinction**: TTFT includes BOTH prefill time AND the first sampling step. It is the true "latency until response starts generating."

**Relationship to other metrics**:
```
TTFT = Prefill + First_Sample_Step
     ≈ Prefill + (1 / tok_per_sec)  # first token is one decode step
```

**Typical values**: 30-50ms warm, 200-400ms cold (dominated by system prefill).

---

### 3. Tok/s (Tokens Per Second)
**What it measures**: Decode throughput — how fast the model generates assistant tokens after prefill is complete.

**Calculation**:
```python
t_decode_start = time.perf_counter()    # right after assistant header prefill
for _ in range(max_tokens):
    token_id = self._sample()
    self._eval([token_id])              # extends KV with sampled token
    assistant_tokens.append(token_id)
t_decode_end = time.perf_counter()

decode_elapsed = t_decode_end - t_decode_start
tok_per_sec = len(assistant_tokens) / decode_elapsed
```

**Why it varies**:
- **Context length**: At 264 tokens (your turn 6), the attention computation is larger than at 50 tokens (turn 1). You see 80-90 tok/s at long context vs 115-125 tok/s at short context.
- **CPU vs GPU**: This is CPU inference (llama.cpp). On GPU this would be ~300-800 tok/s.
- **Batching**: Single token per call (no speculative decoding yet).

**Formula check**: `tok/s = decode_tokens / (decode_ms / 1000)`

---

### 4. KV len (KV Cache Length)
**What it measures**: Total number of tokens stored in the persistent KV cache after this turn completes.

**Calculation**:
```python
kv_before_prefill = self._kv_len()   # checkpoint before this turn
# ... prefill user + assistant header + decode ...
kv_after_turn = self._kv_len()

# Invariant check (asserted in code):
expected = kv_before_prefill + len(user_tokens) + len(asst_header) + len(assistant_tokens) + 1 (EOT)
assert kv_after_turn == expected
```

**Growth pattern** (from your log):
```
Turn 1:  50   (system prompt ~30 + user + assistant response ~15)
Turn 6:  264  (grows ~35-40 tokens per turn depending on query length)
Turn 13: 645  (conversation history accumulating)
```

**Why it matters**: KV cache memory grows linearly with `kv_len`. At fp16, Llama 3.2 1B uses ~80KB per token across all layers. At 645 tokens, that's ~52 MB of KV cache — still well within the 4096 context budget.

---

## Why Your Response Was Too Short

**The root cause**: `max_tokens=150` was the default in `generate()`, and the orchestrator wasn't overriding it.

**What 150 tokens looks like**:
```
"Barge In exploits the audio model's ability to predict speaker locations and noise
patterns, allowing it to accurately identify the source of the barge and suppress it."
```
That output is ~35 tokens. The model emitted `<|eot_id|>` early because:
1. **Low confidence continuation**: At 1B parameters, the model is less capable of long-form reasoning than 3B. It "runs out of ideas" faster without explicit guidance.
2. **Temperature settings**: The default sampler in llama.cpp may be too conservative for this model size.
3. **System prompt shaping**: If the system prompt doesn't explicitly ask for detailed answers, small models default to terse responses.

**The fix**: Changed default `max_tokens` from 150 → 512. The model now has headroom to generate 3-4 sentence answers comparable to chat LLMs. You can also pass `max_tokens=1024` for specific long-form queries.

**Why ChatGPT responses are longer**: GPT-3.5/4 have:
- 100B+ parameters (vs your 1B/3B) — more "knowledge" to elaborate
- RLHF tuning to be verbose and helpful
- Default `max_tokens=4096` or higher in their API

---

## Adaptive Model Routing (New Architecture)

### The Problem: One Size Doesn't Fit All

A 1B model is perfect for "What's the weather?" but struggles with "Summarize this 50-page legal contract." Conversely, loading a 3B model for every voice query wastes memory and latency on simple tasks.

### Solution: Tiered Model Selection

| Tier | Model | Use Case | Latency | Memory |
|------|-------|----------|---------|--------|
| **SIMPLE** | Llama 3.2 1B | Voice Q&A, chitchat, facts | ~50ms TTFT | ~1.3GB |
| **COMPLEX** | Llama 3.2 3B | Documents, reasoning, summarization | ~250ms TTFT | ~2.2GB |

### ModelRouter Design

```python
router = ModelRouter()
kv = router.select_model(
    query="Summarize this 10-page paper...",
    task_hint="summarize"  # Optional override
)
# Returns 3B KVCacheManager automatically
```

**Classification heuristics** (in `model_router.py`):
- Keyword detection: "summarize", "analyze", "compare", "explain in detail" → COMPLEX
- Length-based: >100 words → COMPLEX (likely a pasted document)
- Reasoning patterns: "and then", "therefore", "if X then Y" → COMPLEX

**Lazy loading**: Models are loaded on first use, not at startup. This keeps memory low for voice-only sessions.

### Why Llama 3.2?

Llama 3.2 (Sept 2024) is the instruction-tuned variant optimized for assistant tasks:

| Feature | Llama 3.2 3B | Qwen 2.5 7B |
|---------|-------------|-------------|
| Params | 3B | 7.6B |
| MATH benchmark | ~75 | 79.6 |
| Multilingual | 8 languages | 14 languages |
| Tool calling | Native | Via prompting |
| Context | 128K native | 128K native |
| GGUF ready | ✓ (bartowski) | ✓ |

The 1B model is perfect for voice — sub-100ms TTFT on CPU with minimal memory footprint. The 3B model handles documents and reasoning while still fitting in ~2.2GB RAM/VRAM. Both use the same Llama 3 chat template, so the prompt formatting is identical across tiers.

---

## Document Processing Pipeline

**Separate from voice orchestrator** — batch mode for text documents.

### Architecture

```
document.pdf → text extraction → chunking → 3B model → summary.txt
```

### Chunking Strategy

Chunk size is **adaptive** based on the model's `n_ctx` (default `CHUNK_SIZE = 6000`, scaled down for smaller context windows with `CTX_RESERVE_FRACTION = 0.45`).

For documents that exceed the effective chunk size:
1. Split into overlapping chunks (adaptive size, 200 token overlap)
2. **Map**: Summarize each chunk independently
3. **Reduce**: Combine summaries into final output
4. Recursive if combined summaries still too long

**Critical fix**: Each chunk summary resets the KV cache to the `"system"` checkpoint before processing. Without this, `llama_decode` OOMs on large documents because the KV cache accumulates across chunks.

### Entry Points

```bash
# CLI for batch processing
python doccli.py summarize paper.pdf --method map_reduce --max-tokens 1024

# Q&A mode
python doccli.py ask paper.pdf "What is the main contribution?"
```

### Pre-Flight Size Limits

When a user provides a document path in interactive mode, the CLI displays explicit size limits before processing:
- **Hard limit**: ~50 pages (varies by density)
- **Soft limit**: >15 pages triggers a confirmation prompt
- **Abort condition**: Files >100 pages or >500KB of raw text are rejected with a clear message

These limits prevent the OOM crash that previously occurred with large files.

### Voice vs Document Mode Comparison

| Aspect | Voice Mode | Document Mode |
|--------|-----------|---------------|
| Primary model | 1B (simple), 3B (complex) | 3B only |
| Context window | 4K (simple), 32K (complex) | 32K |
| Chunking | No (real-time) | Yes (map-reduce) |
| Output | TTS streaming | Text to file/stdout |
| Latency target | <500ms TTFT | Minutes acceptable |
| Barge-in | Yes (critical) | N/A |

---

## Inference Runtime: End-to-End Data Flow

This section is the definitive reference for how the system actually runs during a voice conversation. It ties together all four crown jewels into a single coherent execution narrative. Read this when you need to explain — or debug — the live inference path.

---

### 1. The Main Event Loop (`orchestrator.run()`)

The orchestrator owns one `asyncio` event loop that never blocks. It is a **state-driven consumer** of audio chunks, not a polling loop.

```
mic callback ──► asyncio.Queue ──► orchestrator loop (one thread)
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │   BargeInController.state      │
                        │   (IDLE | USER_SPEAKING |     │
                        │    THINKING | SPEAKING |       │
                        │    INTERRUPTED)                │
                        └──────────────────────────────┘
```

**No threads for audio**: `sounddevice.InputStream` runs in a PortAudio callback thread, but it only pushes `np.ndarray` chunks into an `asyncio.Queue`. The orchestrator's loop pulls from that queue. This is the only seam between real-time audio and Python asyncio.

**State transitions are single-writer**: Only the orchestrator calls `bargein.on_*()`. The BargeInController validates that each transition is legal; illegal transitions are ignored (defensive, not exceptional).

---

### 2. A Normal Turn (No Interrupt, No Pruning, Cold Speculation)

Here is the exact sequence of events from the moment the user stops speaking to the moment the assistant finishes playing audio.

#### Step 0: Startup (once per session)
```python
KVCacheManager.load()      # Load GGUF model, negotiate n_ctx
KVCacheManager.warm_up() # Prefill system prompt → save checkpoint "system"
```

The system prompt is tokenized as:
```
BOS <|start_header_id|>system<|end_header_id|>\n\n<system text><|eot_id|>
```

This is `eval()`'d once. The resulting KV cache state is snapshotted as checkpoint `"system"`. For voice mode, this is also the `baseline_checkpoint` the Pruner uses as its floor.

#### Step 1: VAD Detects Speech Start
- `VAD.process_chunk()` returns `{"start": true}`
- `bargein.on_user_speech_start()` → state `IDLE → USER_SPEAKING`
- Orchestrator creates:
  - `StreamingASR` — begins feeding audio chunks to `faster-whisper` every 400ms
  - `SpeculativePrefillController` — awaits stable partials

#### Step 2: Streaming ASR Emits Stable Partial
```python
# In StreamingASR worker thread (not blocking event loop)
partial = "how does the kv cache"
# If unchanged for 2+ consecutive runs AND >=3 words AND >=15 chars:
SpeculativePrefillController.on_partial(partial)
```

Inside `on_partial`:
1. Save checkpoint `"pre_speculative"` at current `kv_len`
2. Launch `threading.Thread(target=_prefill_worker, args=(partial,))`
3. Thread calls `kv._build_user_tokens(partial)` → `kv._eval(tokens)` chunk-by-chunk
4. **The KV cache now contains the speculative user turn**

The main event loop continues capturing audio. The speculative prefill runs in parallel.

#### Step 3: VAD Detects Speech End
- `VAD.process_chunk()` returns `{"end": true}`
- `bargein.on_user_speech_end()` → state `USER_SPEAKING → THINKING`
- Orchestrator calls:
  ```python
  final_transcript = streaming_asr.finalize()  # "how does the kv cache work"
  speculative_prefill.finalize(final_transcript)
  ```

Inside `finalize`:
1. Classify match: `"how does the kv cache"` vs `"how does the kv cache work"` → **prefix match**
2. Current implementation: **always rollback to `"pre_speculative"` checkpoint** (safe path)
3. The KV cache is truncated back to where it was before speculation started
4. `finalize()` returns `final_transcript`

> **Future optimization**: On exact match, skip the re-prefill entirely. On prefix match, keep the common prefix and only prefill the delta. The current safe path costs ~20-50ms of re-prefill.

#### Step 4: Pruning Check (before adding new turn)
```python
pruner.prune_if_needed()
```

If `kv_len / n_ctx < 0.80`: nothing happens. The turn proceeds normally.

#### Step 5: Memex Recall (before prefill)
```python
memex_block = memex.recall_block(transcript, k=4)
# Queries:
#   1. Cross-session memories (from SQLite, exclude current session)
#   2. Live in-session summaries (from Memex._live_summaries in RAM)
# Formats into injectable [MEMEX] block, capped at ~1800 chars
```

If memories exist:
```python
prefill_text = "[MEMEX]\n- (2d ago | topics: kv_cache) ...\n[/MEMEX]\n\n" + transcript
```

If no memories: `prefill_text = transcript`

#### Step 6: LLM Generation
```python
bargein.save_pre_assistant_checkpoint()  # Checkpoint at kv_len AFTER user prefill
response = await _generate_and_stream(prefill_text)
```

Inside `kv.generate(prefill_text)`:
```
1. _build_user_tokens(prefill_text)  →  user_tokens
2. _eval(user_tokens)                →  KV grows by len(user_tokens)
3. SAVE checkpoint "turn_N_pre_assistant"
4. _build_assistant_header_tokens() →  asst_header_tokens
5. _eval(asst_header_tokens)         →  KV grows by len(header)
6. FOR each decode token:
     IF cancel_event.is_set(): BREAK  (barge-in path)
     token = _sample()
     IF token == <|eot_id|>: BREAK
     _eval([token])                   →  KV grows by 1 per token
     YIELD text_piece
7. IF NOT cancelled: _eval([<_EOT>])
8. VERIFY invariant: kv_after == kv_before + user + header + decode + (EOT?)
```

The `_generate_and_stream` coroutine runs `kv.generate()` in a `asyncio.to_thread` worker. Tokens are pushed back into an `asyncio.Queue` via `loop.call_soon_threadsafe()`. The orchestrator drains the queue, buffers sentences, and pushes each sentence to the `tts_queue`.

#### Step 7: TTS Playback
```python
while sentence from token_queue:
    await tts_queue.put(sentence)
    # Playback worker (started at session init) pulls from tts_queue
    # and calls Piper → sounddevice
```

When `tts_queue.join()` returns, all audio has been queued to the speaker.

#### Step 8: Turn Complete
```python
bargein.on_turn_complete()  # THINKING/SPEAKING → IDLE
latency.event("turn_complete")

# Off hot path:
pruner.record_turn(transcript, response_text)
asyncio.create_task(asyncio.to_thread(
    memex.store, transcript, response_text, "voice"
))
```

`memex.store()` is fire-and-forget. It summarizes, embeds, and persists to SQLite. If it takes 200ms, the next turn has already started.

---

### 3. The Checkpoint System

Checkpoints are the backbone of safe state management. Every crown jewel depends on them.

| Checkpoint Label | Set By | Purpose | Used By |
|---|---|---|---|
| `"system"` | `kv.warm_up()` | After system prompt prefill | Barge-in rollback (if no turns yet), Pruner baseline (voice mode) |
| `"primed"` | `menu._prime_kv_with_document()` | After document primer prefill | Pruner baseline (file mode) — document context is sacred |
| `"pre_speculative"` | `SpeculativePrefillController.on_partial()` | Before speculative user prefill | Rollback on mismatch |
| `"turn_N_pre_assistant"` | `kv.generate()` at line 198 | After user prefill, before assistant header | Barge-in: truncate assistant's partial response |
| Custom (e.g. `turn_5_pre_assistant_bargein`) | `bargein.save_pre_assistant_checkpoint()` | Snapshot for interrupt | `bargein.interrupt()` → `kv.truncate_to(seq_len)` |

**Critical invariant**: A checkpoint is just an integer `seq_len`. `truncate_to(seq_len)` calls `llama_kv_cache_seq_rm(ctx, 0, seq_len, -1)` then sets `n_tokens = seq_len`. This is O(1) — it does not recompute anything.

---

### 4. Crown Jewel Interactions During Edge Cases

#### 4A: Barge-In During Speculative Prefill

**Scenario**: User interrupts while the speculative prefill thread is still running.

**Sequence**:
1. `should_interrupt()` returns True (VAD start + RMS threshold for 200ms)
2. `bargein.interrupt()`:
   - Sets `cancel_event` → speculative thread sees it between chunks and exits
   - `sd.stop()` flushes speaker
   - Drains `tts_queue`
   - Cancels the turn `asyncio.Task`
   - **No KV truncation needed** — the assistant hadn't started yet (interrupt during THINKING, before `save_pre_assistant_checkpoint`)
   - State: `INTERRUPTED → USER_SPEAKING`
3. Next turn: speculative prefill starts fresh from current KV state

**Key point**: The speculative thread uses its own `threading.Event` (`_cancel_event`), separate from the barge-in `cancel_event`. The barge-in controller doesn't know about speculation; it just kills the decode loop. The speculative thread independently exits when it sees its own cancel flag.

#### 4B: Barge-In During Assistant Decode

**Scenario**: Assistant is speaking; user cuts in mid-sentence.

**Sequence**:
1. `should_interrupt()` returns True
2. `bargein.interrupt()`:
   - Sets `cancel_event` → decode loop breaks at next token check
   - `sd.stop()` stops current audio
   - Drains pending TTS sentences
   - Cancels turn task
   - `kv.truncate_to(pre_assistant_checkpoint)` — **removes the entire assistant turn from KV cache**
   - State: `INTERRUPTED → USER_SPEAKING`
3. The user's message (already in KV from prefill) is intact
4. Next turn starts from the truncated checkpoint

**Invariant preserved**: After truncation, `kv_len == pre_assistant_checkpoint`. The assistant's partial response tokens never existed.

#### 4C: Pruning During a Long Conversation

**Scenario**: Turn 15. `kv_len / n_ctx == 0.82`. The Pruner triggers.

**Sequence** (inside `_run_turn_with_transcript`, BEFORE user prefill):
1. `pruner.prune_if_needed()`:
   - `occupancy_ratio() = 0.82 >= 0.80` → prune needed
   - Oldest 6 turns in `_turn_log` → `summarize_pruned_block()` → summary text
   - `memex.add_live_summary(summary)` → adds to `_live_summaries` (RAM) + persists to DB
   - `kv.restore_checkpoint("system")` → **KV cache truncated back to system prompt only**
   - `_turn_log = remaining_turns` (last 2+ turns kept in log but not in KV)
2. Orchestrator continues:
   - `memex.recall_block()` → now includes the live summary under `[LIVE SESSION CONTEXT]`
   - User prefill → KV grows from system prompt baseline
   - Assistant decode → normal path

**Result**: The KV cache dropped from ~3300 tokens to ~50 tokens (system prompt). The conversation can continue indefinitely. The model "remembers" the pruned turns via the `[MEMEX]` block injected into the prefill, not via the KV cache.

**Key point**: The Pruner truncates to `"system"` (or `"primed"` in file mode). It does NOT re-prefill the remaining recent turns. Those recent turns are lost from KV — the model relies on the MEMEX block to carry them forward. This is a deliberate trade-off: bounded KV size vs. perfect fidelity of every token.

#### 4D: File Mode + Pruning

**Scenario**: User summarized a 30-page PDF, then had a 10-turn conversation about it.

**Difference from voice mode**: `baseline_checkpoint = "primed"`, not `"system"`.

The `"primed"` checkpoint was saved after the document summary was prefilled into KV. So when the Pruner triggers:
- `kv.restore_checkpoint("primed")` — keeps system prompt + document primer
- Only the *conversation turns after the document* are summarized and pruned

**Result**: The assistant never loses the document context, even on turn 50. The document primer is sacred.

---

### 5. Threading Model & Safety Boundaries

| Thread/Context | Owns | Notes |
|---|---|---|
| **Main asyncio loop** | Orchestrator state machine, queue coordination, Memex recall (BM25 is CPU-only, fast) | Single-threaded. Never blocks. |
| **PortAudio callback** | Pushes audio chunks to `asyncio.Queue` | Runs in C thread, releases GIL |
| **`asyncio.to_thread` (ASR)** | `faster-whisper.transcribe()` | One-shot per turn, or streaming partials every 400ms |
| **`asyncio.to_thread` (LLM)** | `kv.generate()` | The decode loop. Polls `cancel_event` between tokens. |
| **`asyncio.to_thread` (Memex store)** | `memex.store()` — summarization + embedding + SQLite | Fire-and-forget. Failure is logged, not fatal. |
| **`threading.Thread` (speculative)** | `_prefill_worker()` — speculative `kv._eval()` | Daemon thread. Own cancel_event. |
| **TTS playback worker** | `play_audio_async()` coroutine | Drains `tts_queue`, calls Piper, plays audio |

**Why `threading.Event` for cancellation?**

The LLM decode loop runs in a thread (via `asyncio.to_thread`). Inside that thread, there are **no `await` points** — it's a tight loop of `_sample()` → `_eval()`. `asyncio.Task.cancel()` only raises `CancelledError` at await boundaries, so it would never fire. `threading.Event.is_set()` is a lock-free, GIL-safe, sub-microsecond check that works inside a thread.

**Why not `multiprocessing`?**

`multiprocessing` would give true process-level cancellation (SIGTERM), but IPC overhead (pickling tensors across process boundaries) dwarfs the per-token cost. For a single-user assistant, cooperative cancellation via a flag is the right primitive.

---

### 6. KV Cache Invariant Verification

Every turn verifies this invariant at line 247 of `kv_cache.py`:

```python
kv_after_turn == kv_before_prefill + len(user_tokens) + len(asst_header) + len(assistant_tokens) + (1 if not cancelled else 0)
```

If violated, a `WARNING` is logged with the delta. In practice, this catches:
- Mismatched token counts (tokenizer vs. model discrepancy)
- Forgotten EOT tokens
- Truncation that didn't update `n_tokens`

This invariant is what makes the checkpoint system reliable. If `kv_len` drifts from reality, checkpoints point to wrong positions and `truncate_to()` corrupts the cache.

---

### 7. Self-Trigger Suppression (No AEC)

The system has no Acoustic Echo Cancellation. Instead:

1. **Mic stays hot during TTS** — so barge-in is always possible
2. **VAD gating**: `should_interrupt()` requires BOTH:
   - VAD "start" signal (Silero thinks it's speech)
   - Chunk RMS > `BARGE_IN_RMS_THRESHOLD` (0.02) for `BARGE_IN_MIN_DURATION_MS` (200ms)
3. **Speaker bleed is quiet**: TTS audio through the laptop speaker barely registers at the mic. On the rare occasion it does, it's not sustained for 200ms.

**Trade-off**: In a very loud room, ambient noise may trigger the RMS threshold. AEC would be better but requires multi-mic hardware or DSP. This is a pragmatic single-mic compromise.

---

## Repository Map (Current State)

```
whisperloop/
├── whisperloop/
│   ├── orchestrator.py       — async FSM dispatcher, Memex + Pruner hooks
│   ├── bargein.py            — Crown Jewel #2: state machine + interrupt
│   ├── kv_cache.py           — Crown Jewel #1: persistent KV + truncation
│   ├── speculative_prefill.py — Crown Jewel #3: ASR/LLM overlap
│   ├── memex/                — Crown Jewel #4: persistent + live memory
│   │   ├── __init__.py       — Public API exports
│   │   ├── manager.py        — Memex.store() / .recall() / .recall_block()
│   │   ├── storage.py        — SQLite WAL + numpy memmap persistence
│   │   ├── summarizer.py     — Hybrid LLM + extractive memory compaction
│   │   ├── embedder.py       — Lazy MiniLM sentence embeddings (384-dim)
│   │   ├── retriever.py      — BM25 → embedding → recency → importance
│   │   ├── injector.py       — [MEMEX] context block builder
│   │   ├── linker.py         — Cross-session memory thread linking
│   │   └── pruner.py         — Live in-session KV cache pruning
│   ├── model_router.py       — Adaptive 1B/3B model selection
│   ├── document_processor.py — Batch document summarization (3B) + adaptive chunking
│   ├── instrumentation.py    — JSONL latency logger
│   ├── asr.py                — batch ASR + StreamingASR foundation
│   ├── tts.py                — Piper wrapper
│   ├── vad.py                — Silero VAD wrapper
│   ├── audio_io.py           — sounddevice + asyncio glue
│   ├── console.py            — ANSI styling + muted grey log formatter
│   ├── menu.py               — Interactive CLI (voice / file / text modes)
│   └── main.py               — CLI entry point
├── doccli.py                — CLI entry (document mode)
├── download_models.py       — Download 1B + 3B model suite
├── bench/
│   ├── bench.py             — Single-turn LLM+TTS latency benchmark
│   ├── bench_memex.py       — Memex store/recall latency + recall@k
│   ├── bench_context.py     — Multi-turn context-awareness benchmark
│   ├── prompts.json         — 15 reference prompts
│   ├── golden_recall.json   — Hand-crafted query→memory pairs for recall@k
│   ├── conversation_chains.json — Multi-turn coherence test scenarios
│   ├── gemini_comparison.md — Strategy doc for cloud benchmark comparison
│   └── results/             — timestamped JSON outputs
├── tests/
│   ├── test_kv_cache.py     — KV invariants + truncation correctness
│   └── test_bargein.py      — FSM transitions + cancel_event reaches decode
└── ARCHITECTURE.md          — this document
```

---

## Whiteboarding Cheat Sheet (1-Page Interview Defense)

### The 60-Second Pitch
> "Whisperloop is a fully local voice assistant — mic to speaker, no network — running on a consumer laptop. Five things make it interesting as an inference systems project:
>
> 1. **Persistent KV cache across turns.** I own the llama.cpp context directly via the low-level `eval`/`sample` API, prefill the system prompt once, and verify a strict KV-growth invariant after every turn. Turn 2+ prefill is 10× faster than turn 1.
>
> 2. **Clean barge-in.** A 5-state async FSM coordinates three atomic cleanup ops on interrupt: cooperative cancellation of the decode thread via a `threading.Event`, flushing the audio output, and truncating the KV cache to a pre-assistant checkpoint. The model's next turn sees a coherent context.
>
> 3. **Speculative prefill.** I run a streaming ASR wrapper that emits partials every 400ms. When a stable partial is detected (unchanged for 2+ runs, 3+ words), I optimistically start LLM prefill in a background thread. On final ASR, I either confirm the speculation, roll back via `truncate_to(checkpoint)`, or re-prefill the delta. This overlaps ASR with LLM, saving 150-250ms TTFT.
>
> 4. **Memex — persistent + live memory.** Every turn is summarized and stored in SQLite + MiniLM embeddings. Before each prefill, hybrid BM25+embedding retrieval fetches the top-k relevant memories and injects them as a bounded MEMEX block. When the KV cache fills past 80%, a LivePruner compacts the oldest turns into a Memex summary, truncates the KV, and the next turn's MEMEX block carries that context forward. Result: infinite-feeling conversations with bounded latency.
>
> 5. **Adaptive model routing + clean UX.** I maintain both Llama 3.2 1B (fast voice) and 3B (document reasoning) models with a heuristic router. Simple queries get 50ms TTFT; complex documents get map-reduce summarization. The terminal uses ANSI styling (bold cyan for the user, bold green for the assistant, dim grey for logs) so the conversation trace is visually prominent and the log noise recedes into the background."},{

### Architecture Sketch (Drawable in 90 seconds)
```
mic → sounddevice → asyncio.Queue → VAD(Silero) ──┐
                                                   ▼
                                          ┌─────────────────┐
                                          │  Orchestrator   │
                                          │  + BargeIn FSM  │
                                          │  + Memex hook   │
                                          └────┬────────────┘
                                               │
                       ┌───────────────────────┼─────────────────┐
                       ▼                       ▼                 ▼
                ASR(faster-whisper)   KVCacheManager        tts_queue
                       │              (owns llama_cpp        │
                       │               low-level ctx)        ▼
                       │              ┌──────────┐    Piper → sounddevice
                       └──transcript─▶│ eval()   │
                                      │ sample() │
                                      └──────────┘
                                       cancel_event ◀── interrupt()
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                         Memex.recall()  Pruner.prune()   Memex.store()
                         [MEMEX block]   (if KV >80%)    (async, off hot path)
                              │                │
                         ┌────▼─────────┐  ┌──▼─────────────┐
                         │  Retriever   │  │  Summarizer    │
                         │ BM25+embed   │  │ (extractive    │
                         │ SQLite+numpy │  │  LLM hybrid)   │
                         └──────────────┘  └────────────────┘
```

### Five Defenses You Must Have Memorized

**Q: "How does cache truncation work?"**
> "I call `llama_kv_cache_seq_rm(ctx, seq_id=0, p0=checkpoint_len, p1=-1)` which drops K/V entries from position `checkpoint_len` to the end. Then I manually set `llm.n_tokens = checkpoint_len` to keep the high-level wrapper consistent with the low-level cache. The checkpoint is saved between user-prefill and assistant-header so the user's message survives the interrupt but the assistant's partial reply does not."

**Q: "Why a `threading.Event` instead of `asyncio.Task.cancel()`?"**
> "The decode loop runs in a thread, not a coroutine. `Task.cancel()` only raises `CancelledError` at `await` points — there are no awaits inside the thread. `threading.Event` is the right primitive: lock-free check, GIL-safe, sub-microsecond overhead per token. We do also cancel the outer asyncio Task to unblock the awaiter, but the actual stop signal is the Event."

**Q: "What's your tail latency story versus a cloud API?"**
> "Local p50 will be similar or slightly slower than a warm cloud endpoint. The win is p99: no network jitter, no shared-tenant queueing, no cold start. The hiring narrative is that local inference has different latency *shape*, not strictly better latency — and for interactive voice, predictable p99 beats lower p50."

**Q: "Why two models instead of just using the 7B for everything?"**
> "The 1B model does ~100 tok/s on CPU with ~50ms TTFT — that's your weather, timers, quick facts. The 3B does ~25 tok/s with ~250ms TTFT but handles longer context for documents. Loading 3B for every 'set a timer' query wastes ~2GB RAM and 200ms latency. The router uses keyword + length heuristics to choose, and lazy loading means voice-only users never pay the 3B cost."

**Q: "How does Memex handle context overflow mid-conversation?"**
> "When `kv_len / n_ctx >= 0.80`, the LivePruner estimates how many oldest turns to evict to drop occupancy back to 40%. It sends those turn pairs to the Summarizer, stores the compacted summary in Memex with `modality='prune'`, then truncates the KV cache back to a baseline checkpoint — the system prompt for voice, or the document primer for file mode. The next turn's `recall_block()` automatically picks up that live summary via the `[LIVE SESSION CONTEXT]` section. Result: the conversation can continue indefinitely without OOMing or losing coherence."

**Q: "How do you prevent the summarizer from corrupting the conversation KV?"**
> "We discovered that `llama_cpp.Llama.__call__` internally calls `self.reset()`, which wipes the KV cache. If Memex summarization shared the conversation model, it would silently corrupt the context — the model would start generating from the summarization template instead of the conversation on the next turn. The fix: a dedicated `llama_cpp.Llama` instance (separate from the conversation model) is auto-loaded from `models/` for Memex compaction. Each call is fully stateless — its internal KV reset is harmless because we never persist its context. Falls back to extractive summarization if no model file is found."

### The "What Would You Do Next" List

✅ = **Implemented**

1. **Speculative decoding** with a 0.5B draft + 3B target — biggest inference-systems signal.
2. **Paged KV cache** + continuous batching for multi-user serving.
3. **Proper AEC** via multi-mic + RNNoise so barge-in works in noisy rooms.
4. **ONNX Runtime + DirectML** — skipped; would break KV cache truncation control (Crown Jewel #1 incompatible).
5. **Learned routing** — train a small classifier (distilled BERT) to predict task complexity from the query, replacing keyword heuristics.
6. ✅ **Dedicated memex summarizer model** — a separate `llama_cpp.Llama` instance (1B Q4, isolated from conversation KV) is auto-discovered from `models/` and used for all Memex compaction. Falls back to extractive if no model found. Implemented in `memex/summarizer.py`.
7. ✅ **Cross-session memory linking** — `memex/linker.py` (`MemoryLinker`) links newly stored memories to related memories from past sessions via topic Jaccard + embedding cosine similarity. Creates a `parent_id` chain in SQLite. `Memex.get_thread(mem_id)` walks the full chain. Called automatically on every `store()`.
8. **User feedback on memory quality** — let the user mark memories as important/forgettable to train importance scoring.
9. ✅ **Speculative recall** — `_on_streaming_partial()` in the orchestrator triggers `memex.recall_block()` in a daemon thread on the first stable ASR partial (≥3 words, ≥15 chars). By the time VAD-end fires, the `[MEMEX]` block is pre-fetched. The turn uses it if the final transcript starts with the partial; otherwise falls back to a live recall. Eliminates ~6-10ms recall latency from the hot path.
10. **Head-to-head cloud benchmark** — wire `bench/bench_remote.py` with Gemini Live / OpenAI Realtime APIs to produce comparative CDFs on the same WAV corpus.

### Common Anti-Patterns to Call Out (Shows Maturity)
- "I considered using the high-level `Llama()(prompt)` API but it's stateless — every call re-encodes the full history. The whole point of this project is to own the context."
- "I considered killing the mic during TTS to prevent self-trigger but that breaks barge-in. The amplitude-gated approach is a pragmatic single-mic compromise."
- "I considered using `multiprocessing` for the LLM thread for true cancellation but the IPC overhead would dwarf the per-token cost. Cooperative cancellation via a flag is the right call for a single-user, single-process design."
- "I considered loading both models at startup but that wastes ~3.5GB RAM for voice-only users. Lazy loading keeps the baseline footprint at ~1.3GB, and the 3B model only loads if a document task is detected or explicitly requested."
- "I considered caching the MEMEX block in the KV cache across turns, but memories change every turn (new ones added, old ones recalled). Caching stale MEMEX content means the model is always one turn behind. The 50-100 ms re-prefill is the price of fresh recall."
- "I considered using Chroma or LanceDB for vector storage, but for a single-user machine with <1M memories, SQLite + memory-mapped numpy is faster (no IPC, no network), simpler (one file each), and zero-dependency."
- "I considered letting Memex summarization call `Llama()(prompt)` on the shared conversation model for convenience, but that internally resets the KV cache and corrupts the conversation state. We had to disable it and use extractive fallback until we wire a dedicated summarizer instance."

