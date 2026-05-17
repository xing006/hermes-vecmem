"""SQLite-vec 向量存储层 — 增删查改 + 混合检索 + IVF 索引。"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DIMENSION = 384


def _detect_vec0_dimension(db: sqlite3.Connection) -> Optional[int]:
    cur = db.execute(
        "SELECT sql FROM sqlite_master WHERE name='vec_memories'"
    )
    row = cur.fetchone()
    if row is None:
        return None
    m = re.search(r'float\[(\d+)\]', row[0])
    if m:
        return int(m.group(1))
    return None


def _make_float32_vec(values: List[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _parse_float32_vec(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _l2_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _average_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            sums[i] += v[i]
    n = len(vectors)
    return [s / n for s in sums]


class VecStore:
    """Vector memory store backed by sqlite-vec + FTS5 + optional IVF index."""

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
        self._ivf_trained = False
        self._ivf_k = 0
        self._ivf_probe = 1  # cells to probe on search

    def initialize(self) -> None:
        """Open DB, load vec extension, create tables."""
        self._db = sqlite3.connect(self._db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")

        self._db.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(self._db)
        except Exception as e:
            logger.error("Failed to load sqlite-vec: %s", e)
            raise

        # Main content
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # FTS5
        self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, category, content=memories, content_rowid=id)
        """)

        # Vec0 — auto-migrate on dimension change
        existing_dim = _detect_vec0_dimension(self._db)
        if existing_dim:
            if existing_dim != self._dimension:
                logger.warning(
                    "vec_memories dimension changed: %d → %d. "
                    "Recreating vec0 table, clearing IVF + embed cache.",
                    existing_dim, self._dimension,
                )
                # Drop ALL sqlite-vec internal tables
                for t in ["vec_memories", "vec_memories_info",
                          "vec_memories_chunks", "vec_memories_rowids",
                          "vec_memories_vector_chunks00"]:
                    self._db.execute(f"DROP TABLE IF EXISTS {t}")
                self._db.execute("DELETE FROM embed_cache")
                self._db.execute("DELETE FROM ivf_centroids")
                self._db.execute("DELETE FROM ivf_membership")
                self._db.execute("DELETE FROM ivf_config")
                self._db.commit()
                self._ivf_k = 0
                self._ivf_trained = False
                self._db.execute(f"""
                    CREATE VIRTUAL TABLE vec_memories
                    USING vec0(embedding float[{self._dimension}])
                """)
            # else: dimension matches, use existing — no change needed
        else:
            self._db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories
                USING vec0(embedding float[{self._dimension}])
            """)

        # Embedding cache
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS embed_cache (
                text_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        # --- IVF tables ---
        # Centroids
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS ivf_centroids (
                centroid_id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector BLOB NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        # Membership: each memory → nearest centroid
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS ivf_membership (
                mem_id INTEGER PRIMARY KEY,
                centroid_id INTEGER NOT NULL,
                FOREIGN KEY (mem_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)

        # Config/metadata
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS ivf_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Read IVF state
        cur = self._db.execute(
            "SELECT value FROM ivf_config WHERE key='ivf_k'"
        )
        row = cur.fetchone()
        if row:
            self._ivf_k = int(row["value"])
            self._ivf_trained = self._ivf_k > 0
            logger.info("IVF loaded: K=%d, dim=%d", self._ivf_k, self._dimension)

        self._db.commit()

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------ #
    # IVF index management
    # ------------------------------------------------------------------ #

    def build_index(self, force: bool = False) -> Dict[str, Any]:
        """Train IVF centroids on existing data.

        Uses k-means with sqrt(N/2) clusters, max 20 iterations.
        Only builds when count > K * 2.
        """
        count = self.count()
        if count < 4:
            return {"status": "skipped", "reason": f"too few vectors ({count})"}

        K = max(2, int(math.sqrt(count // 2)))
        if K > count // 2:
            K = max(2, count // 2)

        if not force and self._ivf_trained and self._ivf_k == K:
            return {"status": "skipped", "reason": "already trained"}

        # Read all vectors from vec0 table
        cur = self._db.execute(
            "SELECT rowid, embedding FROM vec_memories ORDER BY rowid"
        )
        rows = cur.fetchall()
        all_ids = [r[0] for r in rows]
        all_vecs = [_parse_float32_vec(r[1]) for r in rows]

        if len(all_vecs) < K:
            return {"status": "skipped", "reason": f"need >= {K} vectors"}

        # --- K-means ---
        # Init: pick K random vectors
        rng = hashlib.sha256(str(time.time()).encode()).digest()
        centroids = []
        chosen = set()
        seed_offsets = [int(rng[i % len(rng)]) for i in range(K * 2)]
        for i in range(K):
            idx = seed_offsets[i] % len(all_vecs)
            while idx in chosen:
                idx = (idx + 1) % len(all_vecs)
            chosen.add(idx)
            centroids.append(all_vecs[idx][:])

        assignments = [0] * len(all_vecs)
        for iteration in range(20):
            changed = 0
            # Assign
            for i, vec in enumerate(all_vecs):
                best_d = float("inf")
                best_c = 0
                for c, cent in enumerate(centroids):
                    d = _l2_distance(vec, cent)
                    if d < best_d:
                        best_d = d
                        best_c = c
                if assignments[i] != best_c:
                    assignments[i] = best_c
                    changed += 1

            if changed == 0:
                break

            # Update centroids
            for c in range(K):
                members = [all_vecs[i] for i in range(len(all_vecs)) if assignments[i] == c]
                if members:
                    centroids[c] = _average_vectors(members)

        # --- Store centroids ---
        self._db.execute("DELETE FROM ivf_centroids")
        now = time.time()
        for c_idx, vec in enumerate(centroids):
            self._db.execute(
                "INSERT INTO ivf_centroids (centroid_id, vector, created_at) VALUES (?, ?, ?)",
                (c_idx + 1, _make_float32_vec(vec), now),
            )

        # --- Store membership ---
        self._db.execute("DELETE FROM ivf_membership")
        for i, fid in enumerate(all_ids):
            self._db.execute(
                "INSERT INTO ivf_membership (mem_id, centroid_id) VALUES (?, ?)",
                (fid, assignments[i] + 1),
            )

        # --- Save config ---
        self._db.execute(
            "INSERT OR REPLACE INTO ivf_config (key, value) VALUES ('ivf_k', ?)",
            (str(K),),
        )
        self._db.commit()

        self._ivf_k = K
        self._ivf_trained = True

        return {
            "status": "built",
            "k": K,
            "vectors": len(all_vecs),
            "iterations": iteration + 1,
            "dimension": self._dimension,
        }

    def set_ivf_probe(self, n: int) -> None:
        """Set how many nearest centroids to probe during search (default: 1)."""
        self._ivf_probe = max(1, n)

    # ------------------------------------------------------------------ #
    # Embedding cache
    # ------------------------------------------------------------------ #

    def get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
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
        now = time.time()
        cur = self._db.execute(
            "INSERT INTO memories (content, category, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (content, category, now, now),
        )
        fid = cur.lastrowid

        self._db.execute(
            "INSERT INTO memories_fts (rowid, content, category) VALUES (?, ?, ?)",
            (fid, content, category),
        )

        vec_blob = _make_float32_vec(embedding)
        self._db.execute(
            "INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)",
            (fid, vec_blob),
        )

        # IVF membership (if trained): assign to nearest centroid
        if self._ivf_trained:
            cid = self._ivf_find_nearest(embedding)
            self._db.execute(
                "INSERT OR REPLACE INTO ivf_membership (mem_id, centroid_id) VALUES (?, ?)",
                (fid, cid),
            )

        self._db.commit()
        return fid

    def _ivf_find_nearest(self, embedding: List[float]) -> int:
        """Find nearest centroid for a vector. Returns centroid_id (1-based)."""
        cur = self._db.execute("SELECT centroid_id, vector FROM ivf_centroids")
        best_c = 1
        best_d = float("inf")
        for row in cur.fetchall():
            cent_vec = _parse_float32_vec(row["vector"])
            d = _l2_distance(embedding, cent_vec)
            if d < best_d:
                best_d = d
                best_c = row["centroid_id"]
        return best_c

    def delete(self, fid: int) -> None:
        self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (fid,))
        self._db.execute("DELETE FROM memories_fts WHERE rowid = ?", (fid,))
        self._db.execute("DELETE FROM ivf_membership WHERE mem_id = ?", (fid,))
        self._db.execute("DELETE FROM memories WHERE id = ?", (fid,))
        self._db.commit()

    def get(self, fid: int) -> Optional[Dict[str, Any]]:
        cur = self._db.execute(
            "SELECT id, content, category, created_at, updated_at FROM memories WHERE id = ?",
            (fid,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_all(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
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
    # Vector search — uses IVF when available
    # ------------------------------------------------------------------ #

    def search(self, query_embedding: List[float],
               limit: int = 5) -> List[Dict[str, Any]]:
        vec_blob = _make_float32_vec(query_embedding)

        if self._ivf_trained and self._ivf_k > 0:
            # IVF path: only search vectors in nearest centroid(s)
            results = self._search_ivf(vec_blob, query_embedding, limit)
        else:
            # Brute force: search all
            results = self._search_bruteforce(vec_blob, limit)

        return results

    def _search_bruteforce(self, vec_blob: bytes, limit: int) -> List[Dict[str, Any]]:
        cur = self._db.execute(
            "SELECT rowid, distance FROM vec_memories "
            "WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (vec_blob, limit),
        )
        return self._build_results(cur.fetchall(), limit)

    def _search_ivf(self, vec_blob: bytes, query_embedding: List[float],
                    limit: int) -> List[Dict[str, Any]]:
        # Find nearest centroids
        cur = self._db.execute("SELECT centroid_id, vector FROM ivf_centroids")
        centroids = cur.fetchall()

        scored = []
        for row in centroids:
            cent_vec = _parse_float32_vec(row["vector"])
            d = _l2_distance(query_embedding, cent_vec)
            scored.append((d, row["centroid_id"]))
        scored.sort(key=lambda x: x[0])
        probe_ids = [cid for _, cid in scored[:self._ivf_probe]]

        if not probe_ids:
            return self._search_bruteforce(vec_blob, limit)

        # Get candidate row IDs from centroid membership
        placeholders = ",".join("?" * len(probe_ids))
        cur = self._db.execute(
            f"SELECT mem_id FROM ivf_membership WHERE centroid_id IN ({placeholders})",
            probe_ids,
        )
        candidate_ids = [r["mem_id"] for r in cur.fetchall()]

        if not candidate_ids:
            return self._search_bruteforce(vec_blob, limit)

        if len(candidate_ids) <= limit:
            # Too few candidates, just exhaustive search over them
            placeholders = ",".join("?" * len(candidate_ids))
            cur = self._db.execute(
                f"SELECT rowid, distance FROM vec_memories "
                f"WHERE embedding MATCH ? AND rowid IN ({placeholders}) "
                f"ORDER BY distance LIMIT ?",
                (vec_blob, *candidate_ids, limit),
            )
        else:
            # Use subquery for efficiency
            id_list = ",".join(str(fid) for fid in candidate_ids[:1000])
            cur = self._db.execute(
                f"SELECT rowid, distance FROM vec_memories "
                f"WHERE embedding MATCH ? AND rowid IN ({id_list}) "
                f"ORDER BY distance LIMIT ?",
                (vec_blob, limit),
            )

        results = cur.fetchall()
        if not results:
            return self._search_bruteforce(vec_blob, limit)
        return self._build_results(results, limit)

    def _build_results(self, vec_results, limit: int) -> List[Dict[str, Any]]:
        if not vec_results:
            return []

        ids = [r[0] for r in vec_results]
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
    # Keyword search
    # ------------------------------------------------------------------ #

    def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        safe_query = query.replace('"', '""')
        cur = self._db.execute(
            "SELECT m.id, m.content, m.category, m.created_at "
            "FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (safe_query, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        total = self.count()
        cur = self._db.execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        )
        by_category = {r["category"]: r["cnt"] for r in cur.fetchall()}
        try:
            size_bytes = Path(self._db_path).stat().st_size
        except OSError:
            size_bytes = 0

        result = {
            "total": total,
            "by_category": by_category,
            "db_size_bytes": size_bytes,
            "dimension": self._dimension,
        }

        if self._ivf_trained:
            result["ivf"] = {
                "k": self._ivf_k,
                "probe": self._ivf_probe,
            }

        return result
