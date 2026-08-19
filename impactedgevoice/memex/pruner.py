"""
pruner.py — Live in-session KV cache pruning with Memex offload.

Triggers when the active KV cache approaches its context window limit. Selects
the oldest user/assistant turns (after the system prompt + any document primer),
compacts them via the Summarizer, stores the summary in Memex (both as a live
in-session summary and as a persistent memory row), and truncates the KV cache
to free space.

This keeps long conversations and post-document chats running without crashing
on context overflow, at the cost of a one-time ~100-200 ms summarization stall
when pruning fires.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from impactedgevoice.kv_cache import KVCacheManager
    from impactedgevoice.memex.manager import Memex

logger = logging.getLogger(__name__)


class LivePruner:
    """
    Watches a KVCacheManager's fill ratio and, on demand, prunes the oldest
    conversational turns by:
      1. Reading the raw text of the N oldest turns from the conversation log.
      2. Asking the Summarizer to compact them into a single block.
      3. Pushing that block to Memex.add_live_summary() (visible to subsequent turns).
      4. Truncating the KV cache to a saved checkpoint that excludes those turns.

    Important caveats:
      * KVCacheManager doesn't currently store raw turn text by token-range, so
        we maintain our own log here keyed off turn boundaries the orchestrator
        records via record_turn().
      * On truncate, we rewind to the system+primer checkpoint and DO NOT
        re-prefill anything — the pruned content is now represented only via
        the live summary in the next turn's MEMEX block.
    """

    # Trigger pruning when KV occupancy crosses this fraction of n_ctx.
    DEFAULT_HIGH_WATER = 0.80
    # After pruning, target this fraction of n_ctx as the new occupancy.
    DEFAULT_LOW_WATER = 0.40
    # Don't prune unless we have at least this many recorded turns.
    MIN_TURNS_BEFORE_PRUNE = 4

    def __init__(
        self,
        kv: "KVCacheManager",
        memex: "Memex",
        baseline_checkpoint: str = "system",
        high_water: float = DEFAULT_HIGH_WATER,
        low_water: float = DEFAULT_LOW_WATER,
    ):
        """
        Args:
            kv: the live KV cache manager
            memex: Memex instance for offloading
            baseline_checkpoint: KV checkpoint label representing the "untouchable"
                prefix (e.g. system prompt + document primer). Pruning never
                truncates below this point.
        """
        self.kv = kv
        self.memex = memex
        self.baseline_checkpoint = baseline_checkpoint
        self.high_water = high_water
        self.low_water = low_water

        # Turn log: list of (user_text, response_text) in chronological order.
        # Filled by record_turn(); consumed by prune_if_needed().
        self._turn_log: list[tuple[str, str]] = []

    # ---- recording ----------------------------------------------------

    def record_turn(self, user_text: str, response_text: str) -> None:
        """Called by the orchestrator after each completed turn."""
        if user_text or response_text:
            self._turn_log.append((user_text or "", response_text or ""))

    # ---- pruning ------------------------------------------------------

    def occupancy_ratio(self) -> float:
        n_ctx = max(1, getattr(self.kv, "n_ctx", 0) or 1)
        return self.kv.kv_len / n_ctx

    def should_prune(self) -> bool:
        return (
            len(self._turn_log) >= self.MIN_TURNS_BEFORE_PRUNE
            and self.occupancy_ratio() >= self.high_water
        )

    def prune_if_needed(self) -> Optional[str]:
        """
        If the KV cache is past the high-water mark, summarize the oldest turns
        and truncate. Returns the summary text on success, None otherwise.
        """
        if not self.should_prune():
            return None

        # How many turns to drop? Aim to bring occupancy down to low_water.
        # Approximation: each turn takes ~roughly even share of the post-baseline KV.
        try:
            baseline_len = self.kv._checkpoints.get(self.baseline_checkpoint, 0)
        except AttributeError:
            baseline_len = 0
        n_ctx = max(1, getattr(self.kv, "n_ctx", 0) or 1)
        target_kv_len = int(self.low_water * n_ctx)
        excess = self.kv.kv_len - target_kv_len
        if excess <= 0:
            return None

        # Estimate avg tokens/turn from how many turns sit above baseline
        post_baseline = max(1, self.kv.kv_len - baseline_len)
        avg_per_turn = post_baseline / max(1, len(self._turn_log))
        n_to_prune = max(2, int(excess / max(1, avg_per_turn)))
        n_to_prune = min(n_to_prune, len(self._turn_log) - 2)  # always keep last 2 turns
        if n_to_prune <= 0:
            return None

        oldest = self._turn_log[:n_to_prune]
        remaining = self._turn_log[n_to_prune:]

        logger.info(
            "[PRUNE] KV %d/%d (%.0f%%) — compacting %d oldest turns into Memex",
            self.kv.kv_len, n_ctx, 100 * self.occupancy_ratio(), n_to_prune,
        )

        # Step 1: summarize the dropped block.
        try:
            mem = self.memex.summarizer.summarize_pruned_block(oldest)
        except Exception as e:
            logger.warning("[PRUNE] summarize_pruned_block failed: %s", e)
            return None
        if not mem.summary:
            logger.warning("[PRUNE] empty summary — aborting prune")
            return None

        # Step 2: persist + surface as live summary.
        self.memex.add_live_summary(mem.summary, topics=mem.topics)

        # Step 3: truncate KV cache back to baseline and replay nothing.
        # The live MEMEX block on the next turn carries the dropped context.
        try:
            self.kv.restore_checkpoint(self.baseline_checkpoint)
            logger.info(
                "[PRUNE] Truncated KV to baseline (seq_len=%d). Pruned summary: %s",
                self.kv.kv_len, mem.summary[:120],
            )
        except Exception as e:
            logger.warning("[PRUNE] truncate failed: %s", e)
            return None

        # Step 4: keep the most-recent turns in our local log; they're not in
        # the live KV anymore (we just truncated), but the MEMEX block + the
        # next user turn's prefill will rebuild needed context.
        self._turn_log = remaining
        return mem.summary
