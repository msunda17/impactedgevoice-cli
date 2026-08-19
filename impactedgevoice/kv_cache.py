"""
kv_cache.py — Persistent KV Cache Manager (Crown Jewel #1)

Architecture:
  At startup, prefill the system prompt once via low-level eval().
  Save a "system-prompt checkpoint" — the exact sequence length after prefill.
  Each user turn appends via eval() and decodes via sampling loop.
  On barge-in, truncate back to a saved KV checkpoint with llama_kv_cache_seq_rm().

Key invariant:
  kv_seq_len after turn N == system_tokens + Σ(user_k + assistant_k) for k in 1..N

This is verified at every turn and logged for the blog post latency graph.
"""

import time
import logging
import ctypes
import threading
from typing import Iterator, Optional
import numpy as np
import llama_cpp
from llama_cpp import Llama

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful, brief, and concise voice assistant. "
)

# Llama 3 special token IDs — verified against Meta tokenizer
_BOS         = 128000   # <|begin_of_text|>
_EOT         = 128009   # <|eot_id|>
_HEADER_START = 128006  # <|start_header_id|>
_HEADER_END   = 128007  # <|end_header_id|>
_NL_NL       = None     # "\n\n" — encoded dynamically


class KVCacheManager:
    """
    Owns the Llama context and manages the KV cache lifecycle.

    Public API:
        warm_up()            — prefill the system prompt once; call once at boot.
        generate(transcript) — append user turn, decode assistant response.
        truncate_to(seq_len) — roll back the cache to a saved checkpoint.
        kv_len               — current KV sequence length (read-only property).
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.model_path   = model_path
        self.n_ctx        = n_ctx
        self.n_gpu_layers = n_gpu_layers

        # Latency instrumentation — every turn gets logged
        self.turn_metrics: list[dict] = []
        self._turn_index = 0

        # Checkpoints: seq_len snapshots we can roll back to
        # Key: label (str), Value: seq_len (int)
        self._checkpoints: dict[str, int] = {}

        self._llm: Llama | None = None
        self._nl_nl_tokens: list[int] = []
        self._system_seq_len: int = 0  # set after warm_up()

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    # Context size fallback ladder — tried in order when GPU OOMs
    _CTX_FALLBACKS = [32768, 16384, 8192, 4096, 2048]

    def load(self) -> None:
        """Load the model. Separated from __init__ so caller controls timing.

        If the requested n_ctx causes an OOM on the GPU, automatically retries
        with progressively smaller context windows before hard-failing.
        """
        logger.info("Loading model from %s …", self.model_path)
        t0 = time.perf_counter()

        # Build a fallback ladder starting from n_ctx, then each smaller step
        ctx_attempts = [self.n_ctx] + [
            c for c in self._CTX_FALLBACKS if c < self.n_ctx
        ]
        last_exc: Exception | None = None
        for ctx in ctx_attempts:
            try:
                self._llm = Llama(
                    model_path   = self.model_path,
                    n_gpu_layers = self.n_gpu_layers,
                    n_ctx        = ctx,
                    logits_all   = False,
                    verbose      = False,
                )
                if ctx != self.n_ctx:
                    logger.warning(
                        "[KV] n_ctx reduced from %d to %d due to GPU memory constraints.",
                        self.n_ctx, ctx,
                    )
                self.n_ctx = ctx  # update so callers see the actual value
                break
            except (ValueError, MemoryError, RuntimeError) as exc:
                logger.warning("[KV] Context %d failed (%s), trying smaller…", ctx, exc)
                last_exc = exc
                self._llm = None
        else:
            raise RuntimeError(
                f"Failed to load model at any context size {ctx_attempts}. "
                f"Last error: {last_exc}"
            ) from last_exc

        self._nl_nl_tokens = self._encode("\n\n", add_bos=False)
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.2fs  (n_ctx=%d)", elapsed, self.n_ctx)

    def warm_up(self) -> None:
        """
        Prefill the system prompt exactly once.

        Token sequence fed to context:
          BOS  HEADER_START "system" HEADER_END "\n\n"  <system text tokens>  EOT
        """
        assert self._llm is not None, "Call load() first."

        logger.info("[KV] Prefilling system prompt …")
        t0 = time.perf_counter()

        tokens = self._build_system_tokens()
        self._eval(tokens)

        self._system_seq_len = self._kv_len()
        self._save_checkpoint("system")

        elapsed = time.perf_counter() - t0
        logger.info(
            "[KV] System prompt prefill done: %d tokens, %.0fms",
            len(tokens), elapsed * 1000,
        )

    # ------------------------------------------------------------------ #
    #  Per-turn generation                                                 #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        transcript: str,
        max_tokens: int = 512,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """
        Append the user turn to the KV cache, then decode the assistant response.

        Args:
            transcript: user turn text
            max_tokens: decode budget
            cancel_event: if set during decode, the loop terminates cleanly.
                          Used by the barge-in controller for interrupts.

        Yields:
            str tokens one by one (suitable for piping into a TTS sentence buffer).

        Side-effects:
            * Appends (user turn + assistant response) tokens to the KV cache.
            * Records per-turn latency metrics in self.turn_metrics.
        """
        assert self._llm is not None, "Call load() first."
        assert self._system_seq_len > 0, "Call warm_up() first."

        self._turn_index += 1
        turn_id = self._turn_index

        kv_before_prefill = self._kv_len()
        logger.debug(
            "[KV] Turn %d start — KV seq_len = %d", turn_id, kv_before_prefill
        )

        # ── 1. Prefill user tokens ──────────────────────────────────────
        user_tokens = self._build_user_tokens(transcript)
        t_prefill_start = time.perf_counter()
        self._eval(user_tokens)
        t_prefill_end = time.perf_counter()

        kv_after_user = self._kv_len()
        assert kv_after_user == kv_before_prefill + len(user_tokens), (
            f"KV length mismatch after user eval: "
            f"expected {kv_before_prefill + len(user_tokens)}, got {kv_after_user}"
        )
        logger.debug(
            "[KV] User prefill: +%d tokens, %.0fms",
            len(user_tokens), (t_prefill_end - t_prefill_start) * 1000,
        )

        # Save checkpoint right before we start sampling — enables clean barge-in
        checkpoint_key = f"turn_{turn_id}_pre_assistant"
        self._checkpoints[checkpoint_key] = self._kv_len()

        # ── 2. Prefill the assistant header (no output yet) ─────────────
        asst_header_tokens = self._build_assistant_header_tokens()
        self._eval(asst_header_tokens)

        # ── 3. Decode loop ───────────────────────────────────────────────
        t_first_token = None
        assistant_tokens: list[int] = []
        stop_tokens = {_EOT}

        t_decode_start = time.perf_counter()
        cancelled = False
        for _ in range(max_tokens):
            # Check cancellation BEFORE sampling — barge-in path.
            # The orchestrator/bargein controller sets this event; if it
            # fires, we bail out and let the caller truncate the KV cache.
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                logger.info("[KV] Decode cancelled by barge-in signal")
                break

            token_id = self._sample()

            if token_id in stop_tokens or token_id < 0:
                break

            if t_first_token is None:
                t_first_token = time.perf_counter()

            # Append sampled token back to context (extends KV cache)
            self._eval([token_id])
            assistant_tokens.append(token_id)

            text_piece = self._token_to_str(token_id)
            yield text_piece

        t_decode_end = time.perf_counter()

        # Append EOT for the assistant turn — but only if we weren't
        # interrupted. On interrupt, the caller truncates the KV cache
        # back to the pre-assistant checkpoint, so any partial state we
        # would otherwise write here is about to be discarded anyway.
        if not cancelled:
            self._eval([_EOT])

        # ── 4. Verify KV growth invariant ───────────────────────────────
        # When cancelled, EOT was never appended; expected_kv reflects that.
        kv_after_turn = self._kv_len()
        expected_kv = (
            kv_before_prefill
            + len(user_tokens)
            + len(asst_header_tokens)
            + len(assistant_tokens)
            + (0 if cancelled else 1)   # EOT only appended on clean finish
        )
        if kv_after_turn != expected_kv:
            logger.warning(
                "[KV] Invariant violated: expected=%d, got=%d (delta=%d)",
                expected_kv, kv_after_turn, kv_after_turn - expected_kv,
            )
        else:
            logger.debug(
                "[KV] Turn %d complete — KV grew by %d tokens (invariant OK)",
                turn_id, kv_after_turn - kv_before_prefill,
            )

        # ── 5. Record metrics ────────────────────────────────────────────
        n_decode_tokens  = len(assistant_tokens)
        decode_elapsed   = t_decode_end - t_decode_start
        prefill_elapsed  = t_prefill_end - t_prefill_start
        ttft_ms          = ((t_first_token or t_decode_end) - t_prefill_start) * 1000

        metrics = {
            "turn":             turn_id,
            "prefill_tokens":   len(user_tokens),
            "prefill_ms":       round(prefill_elapsed * 1000, 1),
            "decode_tokens":    n_decode_tokens,
            "decode_ms":        round(decode_elapsed * 1000, 1),
            "tokens_per_sec":   round(n_decode_tokens / decode_elapsed, 1) if decode_elapsed > 0 else 0,
            "ttft_ms":          round(ttft_ms, 1),
            "kv_before":        kv_before_prefill,
            "kv_after":         kv_after_turn,
        }
        self.turn_metrics.append(metrics)
        logger.debug("[METRICS] %s", metrics)

    # ------------------------------------------------------------------ #
    #  Cache control                                                       #
    # ------------------------------------------------------------------ #

    def truncate_to(self, seq_len: int) -> None:
        """
        Roll the KV cache back to `seq_len` tokens.
        Used for barge-in: discard the in-flight assistant response.
        Removes all KV entries from position seq_len to end of context.
        """
        # kv_cache_seq_rm(ctx, seq_id, p0, p1)
        # seq_id=0 (our single sequence), p0=seq_len (first token to remove),
        # p1=-1 means remove to the end of the sequence
        self._llm._ctx.kv_cache_seq_rm(0, seq_len, -1)
        # Update n_tokens to match the truncated length
        self._llm.n_tokens = seq_len
        logger.debug("[KV] Truncated to seq_len=%d", seq_len)

    def save_checkpoint(self, label: str) -> int:
        """Snapshot the current KV length under a label. Returns the seq_len."""
        seq_len = self._kv_len()
        self._checkpoints[label] = seq_len
        logger.debug("[KV] Checkpoint '%s' saved at seq_len=%d", label, seq_len)
        return seq_len

    def restore_checkpoint(self, label: str) -> None:
        """Truncate KV cache back to a named checkpoint."""
        seq_len = self._checkpoints[label]
        self.truncate_to(seq_len)
        logger.debug("[KV] Restored checkpoint '%s' (seq_len=%d)", label, seq_len)

    @property
    def kv_len(self) -> int:
        return self._kv_len()

    @property
    def system_seq_len(self) -> int:
        return self._system_seq_len

    # ------------------------------------------------------------------ #
    #  Low-level helpers                                                   #
    # ------------------------------------------------------------------ #

    def _kv_len(self) -> int:
        """Current number of tokens in the KV cache (= n_tokens eval'd so far)."""
        return self._llm.n_tokens

    def _eval(self, tokens: list[int]) -> None:
        """Push tokens into the context. Grows the KV cache by len(tokens)."""
        self._llm.eval(tokens)

    def _sample(self) -> int:
        """Sample the next token using the model's greedy/temperature sampler."""
        return self._llm.sample()

    def _encode(self, text: str, add_bos: bool = False) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def _token_to_str(self, token_id: int) -> str:
        return self._llm.detokenize([token_id]).decode("utf-8", errors="replace")

    # ── Token sequence builders ──────────────────────────────────────────

    def _build_system_tokens(self) -> list[int]:
        """
        BOS + <|start_header_id|> + "system" + <|end_header_id|> + "\n\n"
        + <system text> + <|eot_id|>
        """
        tokens: list[int] = [_BOS, _HEADER_START]
        tokens += self._encode("system", add_bos=False)
        tokens += [_HEADER_END]
        tokens += self._nl_nl_tokens
        tokens += self._encode(SYSTEM_PROMPT, add_bos=False)
        tokens += [_EOT]
        return tokens

    def _build_user_tokens(self, transcript: str) -> list[int]:
        """
        <|start_header_id|> + "user" + <|end_header_id|> + "\n\n"
        + <transcript> + <|eot_id|>
        """
        tokens: list[int] = [_HEADER_START]
        tokens += self._encode("user", add_bos=False)
        tokens += [_HEADER_END]
        tokens += self._nl_nl_tokens
        tokens += self._encode(transcript, add_bos=False)
        tokens += [_EOT]
        return tokens

    def _build_assistant_header_tokens(self) -> list[int]:
        """<|start_header_id|> + "assistant" + <|end_header_id|> + "\n\n"""
        tokens: list[int] = [_HEADER_START]
        tokens += self._encode("assistant", add_bos=False)
        tokens += [_HEADER_END]
        tokens += self._nl_nl_tokens
        return tokens

    def _save_checkpoint(self, label: str) -> None:
        self._checkpoints[label] = self._kv_len()
