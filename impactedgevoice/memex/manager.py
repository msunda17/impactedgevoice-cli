"""
manager.py — Public API for Memex.

Top-level operations:
  Memex.store(...)       — persist a turn (summarize + embed + index)
  Memex.recall(...)      — hybrid recall, returns top-k MemoryRow
  Memex.recall_block(..) — recall + format as injectable string
  Memex.add_live_summary(...) — for in-session pruner
  Memex.live_summaries() — current session's pruned-block summaries
  Memex.summarize_session(...) — flush session-level rollup
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from impactedgevoice.memex.storage import MemexStorage, MemoryRow
from impactedgevoice.memex.summarizer import Summarizer, MemorySummary
from impactedgevoice.memex.embedder import Embedder
from impactedgevoice.memex.retriever import Retriever
from impactedgevoice.memex.injector import build_memex_block
from impactedgevoice.memex.linker import MemoryLinker

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Lightweight public view of a stored memory."""
    id: int
    timestamp: float
    summary: str
    topics: list[str]
    importance: float
    modality: str
    source_file: Optional[str] = None
    session_id: str = ""


def _to_public(row: MemoryRow) -> Memory:
    return Memory(
        id=row.id or 0,
        timestamp=row.timestamp,
        summary=row.summary,
        topics=row.topics,
        importance=row.importance,
        modality=row.modality,
        source_file=row.source_file,
        session_id=row.session_id,
    )


class Memex:
    """
    Public Memex façade. One instance per process.

    Args:
        data_dir: directory for memex.db + embeddings.npy
        kv_provider: callable returning a KVCacheManager for summarization.
            If None, summarization uses extractive fallback (no LLM).
    """

    def __init__(
        self,
        data_dir: str = "memex_data",
        kv_provider: Optional[Callable] = None,
    ):
        self.storage = MemexStorage(data_dir=data_dir)
        self.embedder = Embedder()
        self.summarizer = Summarizer()   # loads its own dedicated model
        self.retriever = Retriever(self.storage, self.embedder)
        self.linker = MemoryLinker(self.storage, self.embedder)

        self.session_id = uuid.uuid4().hex[:12]
        self.session_started_at = time.time()
        self._turn_index = 0
        self._session_turn_summaries: list[str] = []
        # In-session "live" summaries created by the pruner when the active KV
        # cache is trimmed mid-session. These are surfaced to the LLM via
        # `recall_block(...)` so the assistant doesn't lose context on long
        # conversations or after large document summaries.
        self._live_summaries: list[str] = []

        self.storage.upsert_session(
            self.session_id,
            started_at=self.session_started_at,
            primary_modality="voice",
        )
        logger.info("[MEMEX] Session %s started (data_dir=%s)", self.session_id, data_dir)

    # ---- store --------------------------------------------------------

    def store(
        self,
        user_text: str,
        response_text: str,
        modality: str = "voice",
        source_file: Optional[str] = None,
        precomputed_summary: Optional[MemorySummary] = None,
    ) -> Optional[Memory]:
        """Compact + embed + persist one turn. Safe to call from any thread."""
        if not user_text and not response_text:
            return None

        try:
            mem = precomputed_summary or self.summarizer.summarize_turn(
                user_text, response_text
            )
            if not mem.summary:
                return None

            # Embed the summary text (so similar future queries find it).
            emb_idx: Optional[int] = None
            vec = self.embedder.embed(mem.summary) if self.embedder.is_available else None
            if vec is not None:
                emb_idx = self.storage.append_embedding(vec)

            self._turn_index += 1
            row = MemoryRow(
                timestamp=time.time(),
                session_id=self.session_id,
                turn_index=self._turn_index,
                modality=modality,
                summary=mem.summary,
                topics=mem.topics,
                importance=mem.importance,
                raw_user=user_text[:2000] if user_text else None,
                raw_response=response_text[:4000] if response_text else None,
                source_file=source_file,
                embedding_idx=emb_idx,
            )
            row.id = self.storage.insert(row)
            # Cross-session linking: find and link to related memories from
            # other sessions so the thread graph stays connected.
            try:
                self.linker.link_new(row)
            except Exception as link_err:
                logger.debug("[MEMEX] link_new failed: %s", link_err)
            self._session_turn_summaries.append(mem.summary)
            self.storage.upsert_session(
                self.session_id,
                started_at=self.session_started_at,
                turn_count=self._turn_index,
                primary_modality=modality,
            )
            logger.debug("[MEMEX] Stored memory id=%s topics=%s", row.id, mem.topics)
            return _to_public(row)
        except Exception as e:
            logger.warning("[MEMEX] store() failed: %s", e)
            return None

    # ---- recall -------------------------------------------------------

    def recall(self, query: str, k: int = 5, exclude_current_session: bool = True) -> list[Memory]:
        """Top-k cross-session memories."""
        rows = self.retriever.recall(
            query, k=k,
            exclude_session=self.session_id if exclude_current_session else None,
        )
        return [_to_public(r) for r in rows]

    def recall_block(
        self,
        query: str,
        k: int = 5,
        include_live: bool = True,
    ) -> str:
        """Recall + format ready to prepend to a user turn."""
        rows = self.retriever.recall(query, k=k, exclude_session=self.session_id)
        live = self._live_summaries if include_live else ()
        return build_memex_block(rows, live_summaries=live)

    # ---- live in-session memex ---------------------------------------

    def add_live_summary(self, summary: str, topics: Optional[list[str]] = None) -> None:
        """
        Add a compact summary of pruned-from-KV content. The orchestrator/pruner
        calls this when it evicts old turns from the active context window.

        These summaries are surfaced to the LLM on subsequent turns via
        `recall_block(...)` so the assistant retains a thread of continuity
        even after the live KV has been compacted.
        """
        if not summary or not summary.strip():
            return
        s = summary.strip()
        # Keep last N — bounded memory; older live summaries are still in DB.
        self._live_summaries.append(s)
        if len(self._live_summaries) > 16:
            # Compact pairwise: roll oldest two into one to keep budget tight.
            head = self._live_summaries[:2]
            self._live_summaries = [
                " | ".join(head)
            ] + self._live_summaries[2:]
        # Also persist as a memory row so it survives across sessions.
        try:
            mem = MemorySummary(summary=s, topics=topics or [], importance=0.55)
            self.store(
                user_text="(pruned conversation block)",
                response_text=s,
                modality="prune",
                precomputed_summary=mem,
            )
        except Exception as e:
            logger.debug("[MEMEX] add_live_summary persistence failed: %s", e)

    def live_summaries(self) -> list[str]:
        return list(self._live_summaries)

    def get_thread(self, mem_id: int) -> list[Memory]:
        """
        Return the full cross-session thread that mem_id belongs to,
        in chronological order (oldest first). Useful for surfacing
        a richer context chain rather than a single isolated summary.
        """
        rows = self.linker.get_thread(mem_id)
        return [_to_public(r) for r in rows]

    # ---- session lifecycle -------------------------------------------

    def summarize_session(self) -> Optional[Memory]:
        """Build a session-level overview from all turn summaries."""
        if not self._session_turn_summaries:
            return None
        session_sum = self.summarizer.summarize_session(self._session_turn_summaries)
        self.storage.upsert_session(
            self.session_id,
            started_at=self.session_started_at,
            ended_at=time.time(),
            summary=session_sum.summary,
            turn_count=self._turn_index,
        )
        return None

    def stats(self) -> dict:
        return self.storage.stats()

    def close(self) -> None:
        try:
            self.summarize_session()
        except Exception as e:
            logger.debug("[MEMEX] summarize_session at close failed: %s", e)
        self.storage.close()
