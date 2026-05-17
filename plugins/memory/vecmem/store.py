"""SQLite-vec 向量存储层 — 增删查改 + 混合检索。"""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DIMENSION = 384  # default, overridden when probing real embedding


def _detect_vec0_dimension(db: sqlite3.Connection) -> Optional[int]:
    """Read dimension from existing vec0 virtual table, if any."""
    cur = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='virtual' AND name='vec_memories'"
    )
    row = cur.fetchone()
    if row is None:
        return None
    import re
    m = re.search(r'float\[(\d+)\]', row[0])
    if m:
        return int(m.group(1))
    return None


def _make_float32_vec(values: List[float]) -> bytes:
    """Pack a list of floats into a binary float32 blob (sqlite-vec format)."""
    return struct.pack(f"{len(values)}f", *values)


def _parse_float32_vec(blob: bytes) -> List[float]:
    """Unpack a binary float32 blob back to list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class VecStore:
    """Vector memory store backed by sqlite-vec + FTS5."""

    def __init__(self, hermes_home: Optional[str] = None, dimension: int = DIMENSION):
        if hermes_home:
            db_dir = Path(hermes_home)
        else:
            from hermes_constants import get_hermes_home
            db_dir = get_hermes_home()
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_dir / "vecmem.db")
        self._db: Optional[sqlite3.Connection] = None
        self._dimension = dimension

    def initialize(self) -> None:
        """Open DB, load vec extension, create tables."""
        self._db = sqlite3.connect(self._db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")

        # Load sqlite-vec extension
        self._db.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(self._db)
        except Exception as e:
            logger.error("Failed to load sqlite-vec: %s", e)
            raise

        # Main content table
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # FTS5 for keyword search
        self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, category, content=memories, content_rowid=id)
        """)

        # Vec0 virtual table — respect existing dimension
        existing_dim = _detect_vec0_dimension(self._db)
        if existing_dim:
            self._dimension = existing_dim
            logger.info("vec_memories table exists with dim=%d", self._dimension)
        else:
            self._db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories
                USING vec0(embedding float[{self._dimension}])
            """)
            logger.info("Created vec_memories table with dim=%d", self._dimension)

        # Embedding cache table (text_hash → embedding)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS embed_cache (
                text_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        self._db.commit()

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------ #
    # Embedding cache (text → vector cache, avoids repeated API calls)
    # ------------------------------------------------------------------ #

    def get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Return cached embedding for text+model, or None."""
        import hashlib
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cur = self._db.execute(
            "SELECT embedding FROM embed_cache WHERE text_hash=? AND model=?",
            (text_hash, model),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _parse_float32_vec(row[0])

    def set_cached_embedding(self, text: str, embedding: List[float], model: str) -> None:
        """Cache an embedding for text+model."""
        import hashlib
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        blob = _make_float32_vec(embedding)
        self._db.execute(
            "INSERT OR REPLACE INTO embed_cache (text_hash, text, embedding, model, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (text_hash, text[:500], blob, model, time.time()),
        )
        self._db.commit()

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def add(self, content: str, embedding: List[float],
            category: str = "general") -> int:
        """Store a memory with its embedding vector. Returns the row id."""
        now = time.time()
        cur = self._db.execute(
            "INSERT INTO memories (content, category, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (content, category, now, now),
        )
        fid = cur.lastrowid

        # Insert into FTS5
        self._db.execute(
            "INSERT INTO memories_fts (rowid, content, category) VALUES (?, ?, ?)",
            (fid, content, category),
        )

        # Insert vector
        vec_blob = _make_float32_vec(embedding)
        self._db.execute(
            "INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)",
            (fid, vec_blob),
        )

        self._db.commit()
        return fid

    def delete(self, fid: int) -> None:
        """Delete a memory by id."""
        self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (fid,))
        self._db.execute("DELETE FROM memories_fts WHERE rowid = ?", (fid,))
        self._db.execute("DELETE FROM memories WHERE id = ?", (fid,))
        self._db.commit()

    def get(self, fid: int) -> Optional[Dict[str, Any]]:
        """Get a single memory by id."""
        cur = self._db.execute(
            "SELECT id, content, category, created_at, updated_at FROM memories WHERE id = ?",
            (fid,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def list_all(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """List recent memories."""
        cur = self._db.execute(
            "SELECT id, content, category, created_at, updated_at FROM memories "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        cur = self._db.execute("SELECT COUNT(*) FROM memories")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------ #
    # Vector search (semantic)
    # ------------------------------------------------------------------ #

    def search(self, query_embedding: List[float],
               limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic search by vector similarity. Returns [{id, content, score}, ...]."""
        vec_blob = _make_float32_vec(query_embedding)

        cur = self._db.execute(
            "SELECT rowid, distance FROM vec_memories "
            "WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (vec_blob, limit),
        )
        vec_results = cur.fetchall()

        if not vec_results:
            return []

        # Fetch content from main table
        ids = [r[0] for r in vec_results]
        # sqlite-vec distance is L2^2, convert to similarity score (0-1)
        # score = 1 / (1 + distance)
        id_to_dist = {r[0]: r[1] for r in vec_results}

        placeholders = ",".join("?" * len(ids))
        cur = self._db.execute(
            f"SELECT id, content, category FROM memories WHERE id IN ({placeholders})",
            ids,
        )
        content_map = {r["id"]: r for r in cur.fetchall()}

        results = []
        for fid in ids:
            item = content_map.get(fid)
            if item is None:
                continue
            dist = id_to_dist[fid]
            score = 1.0 / (1.0 + dist)
            results.append({
                "id": fid,
                "content": item["content"],
                "category": item["category"],
                "score": round(score, 4),
            })

        return results

    # ------------------------------------------------------------------ #
    # Keyword search (FTS5)
    # ------------------------------------------------------------------ #

    def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Full-text search by keywords."""
        # Escape special FTS5 characters
        safe_query = query.replace('"', '""')
        cur = self._db.execute(
            "SELECT m.id, m.content, m.category, m.created_at, "
            "  rank FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (safe_query, limit),
        )
        results = []
        for r in cur.fetchall():
            results.append({
                "id": r["id"],
                "content": r["content"],
                "category": r["category"],
                "created_at": r["created_at"],
            })
        return results

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        total = self.count()
        # Count by category
        cur = self._db.execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        )
        by_category = {r["category"]: r["cnt"] for r in cur.fetchall()}

        # DB file size
        try:
            size_bytes = Path(self._db_path).stat().st_size
        except OSError:
            size_bytes = 0

        return {
            "total": total,
            "by_category": by_category,
            "db_size_bytes": size_bytes,
            "dimension": DIMENSION,
        }
