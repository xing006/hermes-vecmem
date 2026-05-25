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


def normalize_content(content: str) -> str:
    """Normalize memory text for deterministic duplicate detection."""
    text = (content or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def content_hash(content: str) -> str:
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


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
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
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
                content_hash TEXT,
                source TEXT DEFAULT 'manual',
                hit_count INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                topic_key TEXT,
                memory_type TEXT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                confidence REAL DEFAULT 1.0,
                decision_reason TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        self._ensure_memory_columns()
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_category_updated ON memories(category, updated_at)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_status_updated ON memories(status, updated_at)"
        )

        # Audit/event log for safe memory updates and rollback.
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                action TEXT NOT NULL,
                before_content TEXT,
                after_content TEXT,
                before_status TEXT,
                after_status TEXT,
                reason TEXT,
                source TEXT DEFAULT 'system',
                created_at REAL NOT NULL
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_memory_created ON memory_events(memory_id, created_at)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_action_created ON memory_events(action, created_at)"
        )

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

    def _ensure_memory_columns(self) -> None:
        """Migrate older vecmem.db files in-place."""
        cur = self._db.execute("PRAGMA table_info(memories)")
        columns = {row["name"] for row in cur.fetchall()}
        migrations = [
            ("content_hash", "ALTER TABLE memories ADD COLUMN content_hash TEXT"),
            ("source", "ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'manual'"),
            ("hit_count", "ALTER TABLE memories ADD COLUMN hit_count INTEGER DEFAULT 1"),
            ("status", "ALTER TABLE memories ADD COLUMN status TEXT DEFAULT 'active'"),
            ("topic_key", "ALTER TABLE memories ADD COLUMN topic_key TEXT"),
            ("memory_type", "ALTER TABLE memories ADD COLUMN memory_type TEXT"),
            ("subject", "ALTER TABLE memories ADD COLUMN subject TEXT"),
            ("predicate", "ALTER TABLE memories ADD COLUMN predicate TEXT"),
            ("object", "ALTER TABLE memories ADD COLUMN object TEXT"),
            ("confidence", "ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0"),
            ("decision_reason", "ALTER TABLE memories ADD COLUMN decision_reason TEXT"),
        ]
        for name, sql in migrations:
            if name not in columns:
                self._db.execute(sql)
        # Backfill hashes for existing rows.
        cur = self._db.execute("SELECT id, content FROM memories WHERE content_hash IS NULL OR content_hash = ''")
        for row in cur.fetchall():
            self._db.execute(
                "UPDATE memories SET content_hash = ? WHERE id = ?",
                (content_hash(row["content"]), row["id"]),
            )

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
    # Audit events
    # ------------------------------------------------------------------ #

    def _record_event(self, memory_id: Optional[int], action: str,
                      before: Optional[Dict[str, Any]] = None,
                      after: Optional[Dict[str, Any]] = None,
                      reason: Optional[str] = None, source: str = "system") -> None:
        self._db.execute(
            "INSERT INTO memory_events (memory_id, action, before_content, after_content, "
            "before_status, after_status, reason, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory_id, action,
                before.get("content") if before else None,
                after.get("content") if after else None,
                before.get("status") if before else None,
                after.get("status") if after else None,
                reason, source, time.time(),
            ),
        )

    def list_events(self, memory_id: Optional[int] = None, limit: int = 50,
                    action: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if action:
            clauses.append("action = ?")
            params.append(action)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        cur = self._db.execute(
            "SELECT event_id, memory_id, action, before_content, after_content, "
            "before_status, after_status, reason, source, created_at FROM memory_events"
            f"{where} ORDER BY event_id ASC LIMIT ?",
            params,
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def derive_metadata(self, content: str, category: str = "general") -> Dict[str, Optional[str]]:
        """Derive coarse structured metadata from memory text without LLM calls."""
        text = (content or "").strip()
        lowered = text.casefold()
        metadata: Dict[str, Optional[str]] = {
            "memory_type": None,
            "subject": None,
            "predicate": None,
            "object": None,
        }

        def set_meta(memory_type: str, subject: str, predicate: str, obj: str) -> Dict[str, Optional[str]]:
            metadata.update({
                "memory_type": memory_type,
                "subject": subject,
                "predicate": predicate,
                "object": obj.strip() if isinstance(obj, str) else obj,
            })
            return metadata

        controlled_prefix_types = {
            "active_context:": "active_context",
            "current_project:": "project",
            "idea_backlog:": "idea_index",
            "paused_project:": "idea_index",
            "abandoned_project:": "idea_index",
            "mvp_backlog:": "idea_index",
        }
        for prefix, memory_type in controlled_prefix_types.items():
            if lowered.startswith(prefix):
                return set_meta(memory_type, prefix.rstrip(":"), "is", text[len(prefix):].strip())

        pref_match = re.match(r"^(?:用户偏好|偏好)[:：]\s*(.+)$", text)
        if pref_match:
            return set_meta("preference", "user.preference", "prefers", pref_match.group(1))

        default_match = re.match(r"^(?:默认模型|默认)[:：]\s*(.+)$", text)
        if default_match:
            return set_meta("environment", "config.default_model", "is", default_match.group(1))

        project_path_match = re.search(r"项目\s*([^\s，,:：]+)\s*(?:路径|目录|位置)[:：]\s*(.+)$", text)
        if project_path_match:
            name = project_path_match.group(1).strip()
            return set_meta("project", f"project.{name}", "path", project_path_match.group(2))

        path_match = re.search(r"(?:路径|目录|位置)[:：]\s*([a-zA-Z]:[/\\].+)$", text)
        if path_match:
            return set_meta("environment", "path", "is", path_match.group(1))

        if any(word in lowered for word in ("已配置", "api key", "api_key", "token", "密钥", "凭据")):
            return set_meta("credential_hint", "credential", "configured", text)

        if any(word in lowered for word in ("工作流", "流程", "步骤", "约定", "优先")):
            return set_meta("workflow", "workflow", "uses", text)

        if any(word in lowered for word in ("报错", "坑", "quirk", "注意", "必须", "需要")):
            return set_meta("tool_quirk", "tool_quirk", "note", text)

        return set_meta("environment" if category in {"memory", "general"} else category, category, "is", text)

    def derive_topic_key(self, content: str, category: str = "general",
                         memory_type: Optional[str] = None, subject: Optional[str] = None,
                         predicate: Optional[str] = None, object: Optional[str] = None) -> Optional[str]:
        """Derive a stable topic key used for conflict detection and updates."""
        text = (content or "").strip()
        lowered = text.casefold()
        memory_type = (memory_type or "").strip()
        subject = (subject or "").strip()
        predicate = (predicate or "").strip()

        if lowered.startswith("active_context:"):
            return "active_context"
        if lowered.startswith("current_project:"):
            return "current_project"

        idea_prefixes = ("idea_backlog:", "paused_project:", "abandoned_project:", "mvp_backlog:")
        for prefix in idea_prefixes:
            if lowered.startswith(prefix):
                rest = text[len(prefix):].strip()
                name = rest.split("，", 1)[0].split(",", 1)[0].split("；", 1)[0].split(";", 1)[0].strip()
                return f"idea_index.{self._topic_slug(name)}" if name else "idea_index"

        if subject == "config.default_model" or re.match(r"^(?:默认模型|默认)[:：]", text):
            return "config.default_model"

        project_path_match = re.search(r"项目\s*([^\s，,:：]+)\s*(?:路径|目录|位置)[:：]", text)
        if project_path_match:
            return f"project.path.{self._topic_slug(project_path_match.group(1))}"
        if memory_type == "project" and subject.startswith("project.") and predicate == "path":
            return f"project.path.{self._topic_slug(subject.split('.', 1)[1])}"

        if memory_type == "preference" or text.startswith(("用户偏好:", "用户偏好：", "偏好:", "偏好：")):
            pref_text = str(object or "") or text
            topic = self._preference_topic(pref_text)
            return f"user.preference.{topic}"

        if subject and predicate:
            # Avoid over-broad topic keys for generic fallback metadata such as
            # category=memory -> subject=memory, predicate=is. Those would turn
            # unrelated memories into false conflicts and bypass LLM merge.
            if subject in {category, "memory", "general"} and predicate == "is":
                return None
            return f"{self._topic_slug(subject)}.{self._topic_slug(predicate)}"
        return None

    def _topic_slug(self, value: str) -> str:
        value = (value or "").strip().casefold()
        value = re.sub(r"[^a-z0-9_.-]+", "-", value)
        value = value.strip("-._")
        return value or "unknown"

    def _preference_topic(self, value: str) -> str:
        text = (value or "").casefold()
        if any(word in text for word in ("回答", "回复", "响应", "简洁", "废话", "解释", "长篇", "短")):
            return "response_style"
        if any(word in text for word in ("中文", "英文", "语言")):
            return "language"
        if any(word in text for word in ("markdown", "格式", "结构", "列表")):
            return "format"
        return self._topic_slug(text[:40])

    def add(self, content: str, embedding: List[float],
            category: str = "general", source: str = "manual",
            status: str = "active", topic_key: Optional[str] = None,
            confidence: float = 1.0, decision_reason: Optional[str] = None,
            memory_type: Optional[str] = None, subject: Optional[str] = None,
            predicate: Optional[str] = None, object: Optional[str] = None) -> int:
        now = time.time()
        h = content_hash(content)
        derived = self.derive_metadata(content, category=category)
        memory_type = memory_type or derived.get("memory_type")
        subject = subject or derived.get("subject")
        predicate = predicate or derived.get("predicate")
        object = object or derived.get("object")
        topic_key = topic_key or self.derive_topic_key(
            content, category=category, memory_type=memory_type,
            subject=subject, predicate=predicate, object=object,
        )
        cur = self._db.execute(
            "INSERT INTO memories (content, category, content_hash, source, hit_count, status, "
            "topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (content, category, h, source, status, topic_key, memory_type, subject, predicate, object,
             confidence, decision_reason, now, now),
        )
        fid = cur.lastrowid
        self._write_indexes(fid, content, category, embedding)
        after = self.get(fid)
        self._record_event(fid, "review" if status == "review" else "add",
                           before=None, after=after, reason=decision_reason, source=source)
        self._db.commit()
        return fid

    def _write_indexes(self, fid: int, content: str, category: str, embedding: List[float]) -> None:
        # FTS5 external-content tables do not support normal DELETE reliably;
        # INSERT OR REPLACE keeps add/update paths simple and avoids malformed
        # index errors on fresh databases.
        self._db.execute(
            "INSERT OR REPLACE INTO memories_fts (rowid, content, category) VALUES (?, ?, ?)",
            (fid, content, category),
        )
        vec_blob = _make_float32_vec(embedding)
        self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (fid,))
        self._db.execute(
            "INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)",
            (fid, vec_blob),
        )
        if self._ivf_trained:
            cid = self._ivf_find_nearest(embedding)
            self._db.execute(
                "INSERT OR REPLACE INTO ivf_membership (mem_id, centroid_id) VALUES (?, ?)",
                (fid, cid),
            )

    def update(self, fid: int, content: str, embedding: List[float],
               category: Optional[str] = None, source: Optional[str] = None,
               status: Optional[str] = None, topic_key: Optional[str] = None,
               confidence: Optional[float] = None, decision_reason: Optional[str] = None,
               memory_type: Optional[str] = None, subject: Optional[str] = None,
               predicate: Optional[str] = None, object: Optional[str] = None) -> None:
        now = time.time()
        existing = self.get(fid)
        if not existing:
            raise KeyError(f"memory id not found: {fid}")
        new_category = category or existing.get("category") or "general"
        new_source = source or existing.get("source") or "manual"
        new_status = status or existing.get("status") or "active"
        derived = self.derive_metadata(content, category=new_category)
        new_memory_type = memory_type if memory_type is not None else (derived.get("memory_type") or existing.get("memory_type"))
        new_subject = subject if subject is not None else (derived.get("subject") or existing.get("subject"))
        new_predicate = predicate if predicate is not None else (derived.get("predicate") or existing.get("predicate"))
        new_object = object if object is not None else (derived.get("object") or existing.get("object"))
        new_topic_key = topic_key if topic_key is not None else self.derive_topic_key(
            content, category=new_category, memory_type=new_memory_type,
            subject=new_subject, predicate=new_predicate, object=new_object,
        ) or existing.get("topic_key")
        new_confidence = confidence if confidence is not None else existing.get("confidence", 1.0)
        new_reason = decision_reason if decision_reason is not None else existing.get("decision_reason")
        self._db.execute(
            "UPDATE memories SET content = ?, category = ?, content_hash = ?, source = ?, "
            "hit_count = COALESCE(hit_count, 0) + 1, status = ?, topic_key = ?, "
            "memory_type = ?, subject = ?, predicate = ?, object = ?, "
            "confidence = ?, decision_reason = ?, updated_at = ? WHERE id = ?",
            (content, new_category, content_hash(content), new_source, new_status,
             new_topic_key, new_memory_type, new_subject, new_predicate, new_object,
             new_confidence, new_reason, now, fid),
        )
        self._write_indexes(fid, content, new_category, embedding)
        after = self.get(fid)
        self._record_event(fid, "update", before=existing, after=after,
                           reason=decision_reason, source=new_source)
        self._db.commit()

    def touch(self, fid: int, action: str = "duplicate", source: str = "system", reason: Optional[str] = None) -> None:
        before = self.get(fid)
        self._db.execute(
            "UPDATE memories SET hit_count = COALESCE(hit_count, 0) + 1, updated_at = ? WHERE id = ?",
            (time.time(), fid),
        )
        after = self.get(fid)
        if before and after and action:
            self._record_event(fid, action, before=before, after=after, reason=reason, source=source)
        self._db.commit()

    def find_by_hash(self, content: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        h = content_hash(content)
        if category:
            cur = self._db.execute(
                "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at "
                "FROM memories WHERE content_hash = ? AND category = ? ORDER BY updated_at DESC LIMIT 1",
                (h, category),
            )
        else:
            cur = self._db.execute(
                "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at "
                "FROM memories WHERE content_hash = ? ORDER BY updated_at DESC LIMIT 1",
                (h,),
            )
        row = cur.fetchone()
        return dict(row) if row else None

    def find_by_prefix(self, prefix: str, category: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        like = f"{prefix}%"
        if category:
            cur = self._db.execute(
                "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at "
                "FROM memories WHERE content LIKE ? AND category = ? ORDER BY updated_at DESC LIMIT ?",
                (like, category, limit),
            )
        else:
            cur = self._db.execute(
                "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at "
                "FROM memories WHERE content LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (like, limit),
            )
        return [dict(r) for r in cur.fetchall()]

    def find_by_topic_key(self, topic_key: str, category: Optional[str] = None,
                          include_inactive: bool = False, limit: int = 20) -> List[Dict[str, Any]]:
        clauses = ["topic_key = ?"]
        params: List[Any] = [topic_key]
        if category:
            clauses.append("category = ?")
            params.append(category)
        if not include_inactive:
            clauses.append("COALESCE(status, 'active') = 'active'")
        params.append(limit)
        cur = self._db.execute(
            "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at "
            f"FROM memories WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
            params,
        )
        return [dict(r) for r in cur.fetchall()]

    def upsert(self, content: str, embedding: List[float], category: str = "general",
               source: str = "manual", unique_prefix: Optional[str] = None,
               topic_key: Optional[str] = None, memory_type: Optional[str] = None,
               subject: Optional[str] = None, predicate: Optional[str] = None,
               object: Optional[str] = None) -> Dict[str, Any]:
        existing = self.find_by_hash(content, category=category)
        if existing:
            self.touch(existing["id"], action="duplicate", source=source)
            return {"status": "duplicate", "id": existing["id"], "content": existing["content"]}

        if unique_prefix:
            matches = self.find_by_prefix(unique_prefix, category=category, limit=50)
            if matches:
                keep = matches[0]
                self.update(
                    keep["id"], content, embedding, category=category, source=source,
                    topic_key=topic_key, memory_type=memory_type, subject=subject,
                    predicate=predicate, object=object,
                )
                for stale in matches[1:]:
                    self.delete(stale["id"])
                return {"status": "updated", "id": keep["id"], "content": content, "removed": max(0, len(matches) - 1)}

        fid = self.add(
            content, embedding, category=category, source=source,
            topic_key=topic_key, memory_type=memory_type, subject=subject,
            predicate=predicate, object=object,
        )
        return {"status": "stored", "id": fid, "content": content}

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
        before = self.get(fid)
        self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (fid,))
        self._db.execute("DELETE FROM memories_fts WHERE rowid = ?", (fid,))
        self._db.execute("DELETE FROM ivf_membership WHERE mem_id = ?", (fid,))
        self._db.execute("DELETE FROM memories WHERE id = ?", (fid,))
        if before:
            self._record_event(fid, "delete", before=before, after=None, source="system")
        self._db.commit()

    def get(self, fid: int) -> Optional[Dict[str, Any]]:
        cur = self._db.execute(
            "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at FROM memories WHERE id = ?",
            (fid,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_all(self, limit: int = 20, offset: int = 0, include_inactive: bool = False) -> List[Dict[str, Any]]:
        if include_inactive:
            cur = self._db.execute(
                "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at FROM memories "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            cur = self._db.execute(
                "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at FROM memories "
                "WHERE COALESCE(status, 'active') = 'active' ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [dict(r) for r in cur.fetchall()]

    def count(self, include_inactive: bool = False) -> int:
        if include_inactive:
            cur = self._db.execute("SELECT COUNT(*) FROM memories")
        else:
            cur = self._db.execute("SELECT COUNT(*) FROM memories WHERE COALESCE(status, 'active') = 'active'")
        return cur.fetchone()[0]

    def mark_status(self, fid: int, status: str, decision_reason: Optional[str] = None,
                    action: Optional[str] = None, source: str = "system") -> None:
        before = self.get(fid)
        if not before:
            raise KeyError(f"memory id not found: {fid}")
        self._db.execute(
            "UPDATE memories SET status = ?, decision_reason = COALESCE(?, decision_reason), updated_at = ? WHERE id = ?",
            (status, decision_reason, time.time(), fid),
        )
        after = self.get(fid)
        event_action = action or status
        self._record_event(fid, event_action, before=before, after=after,
                           reason=decision_reason, source=source)
        self._db.commit()

    def list_by_status(self, status: str, limit: int = 20,
                       category: Optional[str] = None,
                       memory_type: Optional[str] = None,
                       topic_key: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ["COALESCE(status, 'active') = ?"]
        params: List[Any] = [status]
        if category:
            clauses.append("category = ?")
            params.append(category)
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if topic_key:
            clauses.append("topic_key = ?")
            params.append(topic_key)
        cur = self._db.execute(
            "SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason, created_at, updated_at "
            f"FROM memories WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Vector search — uses IVF when available
    # ------------------------------------------------------------------ #

    def search(self, query_embedding: List[float],
               limit: int = 5, include_inactive: bool = False) -> List[Dict[str, Any]]:
        vec_blob = _make_float32_vec(query_embedding)

        if self._ivf_trained and self._ivf_k > 0:
            # IVF path: only search vectors in nearest centroid(s)
            results = self._search_ivf(vec_blob, query_embedding, limit, include_inactive=include_inactive)
        else:
            # Brute force: search all
            results = self._search_bruteforce(vec_blob, limit, include_inactive=include_inactive)

        return results

    def _search_bruteforce(self, vec_blob: bytes, limit: int, include_inactive: bool = False) -> List[Dict[str, Any]]:
        cur = self._db.execute(
            "SELECT rowid, distance FROM vec_memories "
            "WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (vec_blob, limit),
        )
        return self._build_results(cur.fetchall(), limit, include_inactive=include_inactive)

    def _search_ivf(self, vec_blob: bytes, query_embedding: List[float],
                    limit: int, include_inactive: bool = False) -> List[Dict[str, Any]]:
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
            return self._search_bruteforce(vec_blob, limit, include_inactive=include_inactive)

        # Get candidate row IDs from centroid membership
        placeholders = ",".join("?" * len(probe_ids))
        cur = self._db.execute(
            f"SELECT mem_id FROM ivf_membership WHERE centroid_id IN ({placeholders})",
            probe_ids,
        )
        candidate_ids = [r["mem_id"] for r in cur.fetchall()]

        if not candidate_ids:
            return self._search_bruteforce(vec_blob, limit, include_inactive=include_inactive)

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
            return self._search_bruteforce(vec_blob, limit, include_inactive=include_inactive)
        return self._build_results(results, limit, include_inactive=include_inactive)

    def _build_results(self, vec_results, limit: int, include_inactive: bool = False) -> List[Dict[str, Any]]:
        if not vec_results:
            return []

        ids = [r[0] for r in vec_results]
        id_to_dist = {r[0]: r[1] for r in vec_results}

        placeholders = ",".join("?" * len(ids))
        status_filter = "" if include_inactive else " AND COALESCE(status, 'active') = 'active'"
        cur = self._db.execute(
            f"SELECT id, content, category, source, hit_count, status, topic_key, memory_type, subject, predicate, object, confidence, decision_reason FROM memories WHERE id IN ({placeholders}){status_filter}",
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
                "source": item["source"],
                "hit_count": item["hit_count"],
                "status": item["status"],
                "topic_key": item["topic_key"],
                "memory_type": item["memory_type"],
                "subject": item["subject"],
                "predicate": item["predicate"],
                "object": item["object"],
                "confidence": item["confidence"],
                "decision_reason": item["decision_reason"],
                "score": round(score, 4),
            })
        return results

    # ------------------------------------------------------------------ #
    # Keyword search
    # ------------------------------------------------------------------ #

    def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        safe_query = query.replace('"', '""')
        cur = self._db.execute(
            "SELECT m.id, m.content, m.category, m.source, m.hit_count, m.status, m.topic_key, m.memory_type, m.subject, m.predicate, m.object, m.confidence, m.decision_reason, m.created_at, m.updated_at "
            "FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND COALESCE(m.status, 'active') = 'active' "
            "ORDER BY rank LIMIT ?",
            (safe_query, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        total = self.count(include_inactive=True)
        active_count = self.count()
        cur = self._db.execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        )
        by_category = {r["category"]: r["cnt"] for r in cur.fetchall()}
        cur = self._db.execute(
            "SELECT COALESCE(status, 'active') as st, COUNT(*) as cnt FROM memories GROUP BY st"
        )
        by_status = {r["st"]: r["cnt"] for r in cur.fetchall()}
        cur = self._db.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories WHERE memory_type IS NOT NULL GROUP BY memory_type"
        )
        by_type = {r["memory_type"]: r["cnt"] for r in cur.fetchall()}
        try:
            size_bytes = Path(self._db_path).stat().st_size
        except OSError:
            size_bytes = 0

        # Event counts for key actions
        cur = self._db.execute(
            "SELECT action, COUNT(*) as cnt FROM memory_events GROUP BY action"
        )
        events = {r["action"]: r["cnt"] for r in cur.fetchall()}

        # Consistency checks
        mem_count = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        vec_count = self._db.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0]
        fts_count = self._db.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        orphan_vec = self._db.execute(
            "SELECT COUNT(*) FROM vec_memories WHERE rowid NOT IN (SELECT id FROM memories)"
        ).fetchone()[0]
        missing_vec = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT rowid FROM vec_memories)"
        ).fetchone()[0]

        result = {
            "total": mem_count,
            "active": active_count,
            "by_status": by_status,
            "by_category": by_category,
            "by_memory_type": by_type,
            "events": events,
            "db_size_bytes": size_bytes,
            "dimension": self._dimension,
            "consistency": {
                "memories": mem_count,
                "vec_memories": vec_count,
                "memories_fts": fts_count,
                "orphan_vec": orphan_vec,
                "missing_vec": missing_vec,
                "ok": orphan_vec == 0 and missing_vec == 0,
            },
        }

        if self._ivf_trained:
            result["ivf"] = {
                "k": self._ivf_k,
                "probe": self._ivf_probe,
            }

        return result
