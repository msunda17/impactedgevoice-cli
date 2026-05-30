"""
retriever.py — Hybrid recall: BM25 keyword + dense embedding + recency + importance.

Pipeline:
  1. BM25 prefilter on (summary + topics)        → top 50
  2. Embedding cosine rerank                       → top 20
  3. Recency boost: exp(-age_days / 30)
  4. Importance multiplier
  → final top-k
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Optional

import numpy as np

from whisperloop.memex.storage import MemexStorage, MemoryRow
from whisperloop.memex.embedder import Embedder

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it",
    "for", "on", "with", "as", "this", "that", "be", "are", "was", "were",
    "what", "how", "why", "when", "where", "do", "does", "did",
}


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


class _BM25:
    """Tiny in-memory BM25 — no external dependency."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs: list[list[str]]):
        self.docs = docs
        self.N = len(docs) or 1
        self.avgdl = (sum(len(d) for d in docs) / self.N) if docs else 1.0
        self.df: Counter = Counter()
        for d in docs:
            for term in set(d):
                self.df[term] += 1

    def score(self, query_terms: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for term in query_terms:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for i, d in enumerate(self.docs):
                tf = d.count(term)
                if tf == 0:
                    continue
                norm = 1 - self.B + self.B * (len(d) / self.avgdl)
                scores[i] += idf * (tf * (self.K1 + 1)) / (tf + self.K1 * norm)
        return scores


class Retriever:
    """Hybrid retriever over MemexStorage."""

    BM25_TOP = 50
    EMB_TOP = 20
    RECENCY_HALFLIFE_DAYS = 30.0

    def __init__(self, storage: MemexStorage, embedder: Optional[Embedder] = None):
        self.storage = storage
        self.embedder = embedder or Embedder()

    def recall(
        self, query: str, k: int = 5, exclude_session: Optional[str] = None,
    ) -> list[MemoryRow]:
        if not query.strip():
            return []
        memories = self.storage.fetch_all(include_archived=False)
        if exclude_session:
            memories = [m for m in memories if m.session_id != exclude_session]
        if not memories:
            return []

        # Stage 1: BM25
        docs = [_tokenize(f"{m.summary} {' '.join(m.topics)}") for m in memories]
        bm25 = _BM25(docs)
        q_terms = _tokenize(query)
        bm25_scores = bm25.score(q_terms)

        top_n = min(self.BM25_TOP, len(memories))
        bm25_idx = np.argsort(-bm25_scores)[:top_n]
        candidates = [memories[i] for i in bm25_idx]
        candidate_bm25 = bm25_scores[bm25_idx]

        # Stage 2: embedding rerank (if available)
        emb_scores = np.zeros(len(candidates), dtype=np.float32)
        if self.embedder.is_available:
            qv = self.embedder.embed(query)
            if qv is not None:
                for i, m in enumerate(candidates):
                    if m.embedding_idx is None:
                        continue
                    v = self.storage.get_embedding(m.embedding_idx)
                    if v is None:
                        continue
                    emb_scores[i] = float(np.dot(qv, v))  # cosine since both normalized

        # Stage 3 + 4: recency + importance
        now = time.time()
        final_scores = np.zeros(len(candidates), dtype=np.float32)
        # Normalize bm25 + emb to 0-1 then combine
        bmax = candidate_bm25.max() or 1.0
        emax = emb_scores.max() or 1.0
        for i, m in enumerate(candidates):
            bm = candidate_bm25[i] / bmax
            em = emb_scores[i] / emax if emax > 0 else 0.0
            base = 0.5 * bm + 0.5 * em if self.embedder.is_available else bm
            age_days = max(0.0, (now - m.timestamp) / 86400.0)
            recency = math.exp(-age_days / self.RECENCY_HALFLIFE_DAYS)
            final_scores[i] = base * recency * (0.5 + m.importance)

        order = np.argsort(-final_scores)[:k]
        results = [candidates[i] for i in order if final_scores[i] > 0]

        # Update access counters (importance bump)
        for m in results:
            if m.id is not None:
                self.storage.update_access(m.id)
        return results
