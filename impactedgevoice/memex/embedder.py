"""
embedder.py — Semantic indexing via sentence-transformers MiniLM-L6-v2.

384-dim vectors, ~5ms/embedding on CPU, ~90MB model.
Lazy-loaded so the rest of Memex works without sentence-transformers installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Lazy wrapper around sentence-transformers MiniLM."""

    def __init__(self, model_name: str = MODEL_NAME):
        self._model_name = model_name
        self._model = None
        self._available: Optional[bool] = None

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[MEMEX] Loading embedding model %s …", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._available = True
        except Exception as e:
            logger.warning(
                "[MEMEX] sentence-transformers unavailable (%s) — "
                "Memex will fall back to BM25-only retrieval", e,
            )
            self._available = False
        return self._available

    @property
    def is_available(self) -> bool:
        return self._ensure_loaded()

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Returns a 384-dim float32 vector or None if unavailable."""
        if not self._ensure_loaded():
            return None
        try:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.astype(np.float32)
        except Exception as e:
            logger.warning("[MEMEX] embed() failed: %s", e)
            return None

    def embed_batch(self, texts: list[str]) -> Optional[np.ndarray]:
        if not self._ensure_loaded():
            return None
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
            return vecs.astype(np.float32)
        except Exception as e:
            logger.warning("[MEMEX] embed_batch() failed: %s", e)
            return None
