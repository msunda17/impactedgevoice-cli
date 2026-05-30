"""
linker.py — Cross-session memory linking via topic clustering.

When two memories from different sessions share highly overlapping topics
(Jaccard similarity >= threshold) AND similar embedding vectors (cosine >= threshold),
they are linked into a "thread" by setting the newer memory's parent_id to the
older one.

This creates a chain: session A memory → session B memory → session C memory,
all about the same topic. The retriever then surfaces the entire thread when
any member is recalled, giving richer context than isolated summaries.

Usage:
    linker = MemoryLinker(storage, embedder)
    linker.link_new(new_row)           # called after each store()
    linker.rebuild_all_links()         # one-shot repair / migration
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from whisperloop.memex.storage import MemexStorage, MemoryRow

logger = logging.getLogger(__name__)


class MemoryLinker:
    """
    Links semantically related memories across sessions.

    Linking strategy:
        1. Topic Jaccard similarity  >= TOPIC_THRESHOLD (fast keyword gate)
        2. Embedding cosine similarity >= EMB_THRESHOLD  (semantic gate)
        3. Memory must be from a DIFFERENT session (same-session chaining
           is already implicit via turn_index ordering)
        4. Prefer linking to the most-recent qualifying memory so chains
           stay short rather than pointing to ancient roots.

    The link is recorded by setting parent_id on the newer memory to the
    older memory's id. This is a lightweight reference — it never copies
    or modifies summaries.
    """

    TOPIC_THRESHOLD = 0.25   # Jaccard: at least 1 shared topic in every 4
    EMB_THRESHOLD   = 0.55   # cosine similarity on MiniLM 384-dim embeddings
    MAX_CANDIDATES  = 200    # limit full-scan to keep linking O(n) not O(n²)

    def __init__(self, storage: MemexStorage, embedder=None):
        self.storage = storage
        self.embedder = embedder

    # ---- public API ---------------------------------------------------

    def link_new(self, new_row: MemoryRow) -> Optional[int]:
        """
        Find the best matching memory from a different session and link
        new_row to it. Returns the parent_id set, or None if no match.

        Called by Memex.store() after a successful insert so every new
        memory is immediately woven into the thread graph.
        """
        if new_row.id is None or not new_row.topics:
            return None

        candidates = self._fetch_candidates(new_row)
        if not candidates:
            return None

        best_id = self._best_match(new_row, candidates)
        if best_id is None:
            return None

        self.storage.set_parent(new_row.id, best_id)
        logger.debug(
            "[LINKER] Linked memory %d → parent %d (cross-session thread)",
            new_row.id, best_id,
        )
        return best_id

    def rebuild_all_links(self) -> int:
        """
        Re-run linking over all memories. Useful after importing a bulk
        corpus or after changing thresholds. Returns number of links set.
        """
        all_mems = self.storage.fetch_all(include_archived=False)
        # Sort oldest-first so we always link newer → older
        all_mems.sort(key=lambda m: m.timestamp)
        linked = 0
        for i, mem in enumerate(all_mems):
            if not mem.topics or mem.id is None:
                continue
            candidates = [m for m in all_mems[:i] if m.session_id != mem.session_id]
            if not candidates:
                continue
            # Take the most-recent MAX_CANDIDATES from other sessions
            candidates = candidates[-self.MAX_CANDIDATES:]
            best_id = self._best_match(mem, candidates)
            if best_id is not None:
                self.storage.set_parent(mem.id, best_id)
                linked += 1
        logger.info("[LINKER] rebuild_all_links: set %d links", linked)
        return linked

    def get_thread(self, mem_id: int, max_depth: int = 10) -> list[MemoryRow]:
        """
        Walk the parent_id chain from mem_id upward. Returns the full
        thread in chronological order (oldest first).
        """
        thread: list[MemoryRow] = []
        seen: set[int] = set()
        current_id: Optional[int] = mem_id
        depth = 0
        while current_id is not None and depth < max_depth:
            if current_id in seen:
                break  # cycle guard
            seen.add(current_id)
            row = self.storage.fetch_by_id(current_id)
            if row is None:
                break
            thread.append(row)
            current_id = row.parent_id
            depth += 1
        thread.reverse()  # chronological order (root first)
        return thread

    # ---- internals ----------------------------------------------------

    def _fetch_candidates(self, new_row: MemoryRow) -> list[MemoryRow]:
        """Fetch recent memories from OTHER sessions as link candidates."""
        all_mems = self.storage.fetch_all(include_archived=False)
        other = [
            m for m in all_mems
            if m.session_id != new_row.session_id and m.id != new_row.id
        ]
        # Most recent first, limited to MAX_CANDIDATES
        other.sort(key=lambda m: m.timestamp, reverse=True)
        return other[: self.MAX_CANDIDATES]

    def _best_match(
        self, new_row: MemoryRow, candidates: list[MemoryRow]
    ) -> Optional[int]:
        """Return the id of the best linking candidate, or None."""
        new_topics = set(new_row.topics)
        new_vec = self._get_vec(new_row)

        best_id: Optional[int] = None
        best_score = -1.0

        for cand in candidates:
            # Stage 1: topic Jaccard gate (fast)
            cand_topics = set(cand.topics)
            union = new_topics | cand_topics
            if not union:
                continue
            jaccard = len(new_topics & cand_topics) / len(union)
            if jaccard < self.TOPIC_THRESHOLD:
                continue

            # Stage 2: embedding cosine (only if embedder available)
            emb_sim = 0.0
            if new_vec is not None:
                cand_vec = self._get_vec(cand)
                if cand_vec is not None:
                    emb_sim = float(np.dot(new_vec, cand_vec))  # both normalized
                    if emb_sim < self.EMB_THRESHOLD:
                        continue

            score = jaccard * 0.4 + emb_sim * 0.6
            if score > best_score:
                best_score = score
                best_id = cand.id

        return best_id

    def _get_vec(self, row: MemoryRow) -> Optional[np.ndarray]:
        if row.embedding_idx is None:
            return None
        return self.storage.get_embedding(row.embedding_idx)
