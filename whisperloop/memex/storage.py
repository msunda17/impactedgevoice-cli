"""
storage.py — Persistence layer for Memex.

SQLite for metadata (WAL mode for crash safety + concurrent reads).
NumPy memmap for embedding vectors (append-only, mmap'd for fast load).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    session_id      TEXT    NOT NULL,
    turn_index      INTEGER NOT NULL,
    modality        TEXT    NOT NULL,
    summary         TEXT    NOT NULL,
    topics          TEXT    NOT NULL DEFAULT '[]',
    importance      REAL    NOT NULL DEFAULT 0.5,
    raw_user        TEXT,
    raw_response    TEXT,
    source_file     TEXT,
    embedding_idx   INTEGER,
    parent_id       INTEGER REFERENCES memories(id),
    archived        INTEGER NOT NULL DEFAULT 0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   REAL
);

CREATE INDEX IF NOT EXISTS idx_session       ON memories(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_timestamp     ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_archived      ON memories(archived);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    started_at     REAL NOT NULL,
    ended_at       REAL,
    summary        TEXT,
    turn_count     INTEGER NOT NULL DEFAULT 0,
    primary_modality TEXT
);
"""


@dataclass
class MemoryRow:
    """Database row representation."""
    id: Optional[int] = None
    timestamp: float = 0.0
    session_id: str = ""
    turn_index: int = 0
    modality: str = "voice"
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    importance: float = 0.5
    raw_user: Optional[str] = None
    raw_response: Optional[str] = None
    source_file: Optional[str] = None
    embedding_idx: Optional[int] = None
    parent_id: Optional[int] = None
    archived: bool = False
    access_count: int = 0
    last_accessed: Optional[float] = None


class MemexStorage:
    """SQLite + numpy memmap storage backend."""

    def __init__(self, data_dir: str | Path = "memex_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "memex.db"
        self.emb_path = self.data_dir / "embeddings.npy"

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

        self._embeddings = self._load_embeddings()

    # ---- Embedding store ----------------------------------------------

    def _load_embeddings(self) -> np.ndarray:
        if self.emb_path.exists():
            arr = np.load(self.emb_path, mmap_mode="r")
            if arr.shape[1] == EMBEDDING_DIM:
                return np.array(arr)  # load to memory for in-place append
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    def _save_embeddings(self) -> None:
        # Atomic save: write to a tmp file handle (np.save won't append a
        # suffix when given a file object), fsync, then rename onto target.
        tmp = self.emb_path.with_name(self.emb_path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, self._embeddings, allow_pickle=False)
            f.flush()
        tmp.replace(self.emb_path)

    def append_embedding(self, vec: np.ndarray) -> int:
        """Append a 384-dim vector. Returns its row index."""
        assert vec.shape == (EMBEDDING_DIM,), f"bad shape {vec.shape}"
        idx = self._embeddings.shape[0]
        self._embeddings = np.vstack([self._embeddings, vec.astype(np.float32)])
        self._save_embeddings()
        return idx

    def get_embedding(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < self._embeddings.shape[0]:
            return self._embeddings[idx]
        return None

    @property
    def all_embeddings(self) -> np.ndarray:
        return self._embeddings

    # ---- Memory CRUD --------------------------------------------------

    def insert(self, mem: MemoryRow) -> int:
        cur = self._conn.execute(
            """INSERT INTO memories (
                timestamp, session_id, turn_index, modality, summary, topics,
                importance, raw_user, raw_response, source_file, embedding_idx,
                parent_id, archived
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.timestamp or time.time(),
                mem.session_id,
                mem.turn_index,
                mem.modality,
                mem.summary,
                json.dumps(mem.topics),
                mem.importance,
                mem.raw_user,
                mem.raw_response,
                mem.source_file,
                mem.embedding_idx,
                mem.parent_id,
                int(mem.archived),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def fetch_all(self, include_archived: bool = False) -> list[MemoryRow]:
        sql = "SELECT * FROM memories"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY timestamp DESC"
        rows = self._conn.execute(sql).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def fetch_by_session(self, session_id: str) -> list[MemoryRow]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE session_id=? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def fetch_by_id(self, mem_id: int) -> Optional["MemoryRow"]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def set_parent(self, mem_id: int, parent_id: int) -> None:
        """Link mem_id to parent_id (cross-session thread)."""
        self._conn.execute(
            "UPDATE memories SET parent_id=? WHERE id=?", (parent_id, mem_id)
        )
        self._conn.commit()

    def update_access(self, mem_id: int) -> None:
        self._conn.execute(
            "UPDATE memories SET access_count = access_count+1, last_accessed=?, "
            "importance = MIN(1.0, importance + 0.1) WHERE id=?",
            (time.time(), mem_id),
        )
        self._conn.commit()

    def upsert_session(
        self, session_id: str, started_at: float,
        ended_at: Optional[float] = None, summary: Optional[str] = None,
        turn_count: int = 0, primary_modality: str = "voice",
    ) -> None:
        self._conn.execute(
            """INSERT INTO sessions (session_id, started_at, ended_at, summary, turn_count, primary_modality)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 ended_at=excluded.ended_at,
                 summary=COALESCE(excluded.summary, sessions.summary),
                 turn_count=excluded.turn_count,
                 primary_modality=excluded.primary_modality""",
            (session_id, started_at, ended_at, summary, turn_count, primary_modality),
        )
        self._conn.commit()

    def stats(self) -> dict:
        n_total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        n_active = self._conn.execute("SELECT COUNT(*) FROM memories WHERE archived=0").fetchone()[0]
        n_sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {
            "total_memories": n_total,
            "active_memories": n_active,
            "sessions": n_sessions,
            "embeddings": int(self._embeddings.shape[0]),
            "db_size_kb": self.db_path.stat().st_size // 1024 if self.db_path.exists() else 0,
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _row_to_memory(r: sqlite3.Row) -> MemoryRow:
        return MemoryRow(
            id=r["id"],
            timestamp=r["timestamp"],
            session_id=r["session_id"],
            turn_index=r["turn_index"],
            modality=r["modality"],
            summary=r["summary"],
            topics=json.loads(r["topics"] or "[]"),
            importance=r["importance"],
            raw_user=r["raw_user"],
            raw_response=r["raw_response"],
            source_file=r["source_file"],
            embedding_idx=r["embedding_idx"],
            parent_id=r["parent_id"],
            archived=bool(r["archived"]),
            access_count=r["access_count"],
            last_accessed=r["last_accessed"],
        )
