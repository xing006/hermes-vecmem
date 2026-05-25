"""VecMem Manager Dashboard plugin API.

Mounted at /api/plugins/vecmem-manager/ by Hermes Dashboard.
User-level plugin only; does not modify Hermes core code.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / "AppData" / "Local" / "hermes"


def db_path() -> Path:
    return hermes_home() / "vecmem.db"


def connect() -> sqlite3.Connection:
    path = db_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"vecmem.db not found: {path}")
    db = sqlite3.connect(str(path), check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def row_dict(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
    return dict(row) if row else None


def memory_columns(db: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in db.execute("PRAGMA table_info(memories)").fetchall()}


def memory_select_expr(db: sqlite3.Connection) -> str:
    cols = memory_columns(db)
    base = ["id", "content", "category", "source", "hit_count", "status", "topic_key", "confidence", "decision_reason", "created_at", "updated_at"]
    optional = ["memory_type", "subject", "predicate", "object"]
    parts = base[:]
    for col in optional:
        parts.append(col if col in cols else f"NULL AS {col}")
    return ", ".join(parts)


def ensure_tables(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS memory_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            action TEXT NOT NULL,
            before_content TEXT,
            after_content TEXT,
            before_status TEXT,
            after_status TEXT,
            reason TEXT,
            source TEXT DEFAULT 'dashboard',
            created_at REAL NOT NULL
        )
    """)
    db.commit()


def get_record(db: sqlite3.Connection, fid: int) -> Optional[Dict[str, Any]]:
    cur = db.execute(
        f"SELECT {memory_select_expr(db)} FROM memories WHERE id = ?",
        (fid,),
    )
    return row_dict(cur.fetchone())


def record_event(db: sqlite3.Connection, fid: int, action: str, before: Dict[str, Any], after: Dict[str, Any], reason: str) -> None:
    db.execute(
        "INSERT INTO memory_events(memory_id, action, before_content, after_content, before_status, after_status, reason, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'dashboard', ?)",
        (
            fid,
            action,
            before.get("content"),
            after.get("content"),
            before.get("status"),
            after.get("status"),
            reason,
            time.time(),
        ),
    )


def consistency(db: sqlite3.Connection) -> Dict[str, Any]:
    def count(sql: str) -> int:
        try:
            return int(db.execute(sql).fetchone()[0])
        except Exception:
            return -1

    mem_count = count("SELECT COUNT(*) FROM memories")
    vec_count = count("SELECT COUNT(*) FROM vec_memories")
    fts_count = count("SELECT COUNT(*) FROM memories_fts")
    orphan_vec = count("SELECT COUNT(*) FROM vec_memories WHERE rowid NOT IN (SELECT id FROM memories)")
    missing_vec = count("SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT rowid FROM vec_memories)")
    return {
        "memories": mem_count,
        "vec_memories": vec_count,
        "memories_fts": fts_count,
        "orphan_vec": orphan_vec,
        "missing_vec": missing_vec,
        "ok": orphan_vec == 0 and missing_vec == 0,
    }


class StatusBody(BaseModel):
    reason: str = "dashboard action"


@router.get("/stats")
async def stats():
    with connect() as db:
        cols = memory_columns(db)
        total = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        active = db.execute("SELECT COUNT(*) FROM memories WHERE COALESCE(status, 'active') = 'active'").fetchone()[0]
        by_status = {r["st"]: r["cnt"] for r in db.execute("SELECT COALESCE(status, 'active') st, COUNT(*) cnt FROM memories GROUP BY st")}
        by_category = {r["category"]: r["cnt"] for r in db.execute("SELECT category, COUNT(*) cnt FROM memories GROUP BY category")}
        by_type = {}
        if "memory_type" in cols:
            by_type = {r["memory_type"]: r["cnt"] for r in db.execute("SELECT memory_type, COUNT(*) cnt FROM memories WHERE memory_type IS NOT NULL GROUP BY memory_type")}
        events = {r["action"]: r["cnt"] for r in db.execute("SELECT action, COUNT(*) cnt FROM memory_events GROUP BY action")}
        size = db_path().stat().st_size if db_path().exists() else 0
        return {
            "db_path": str(db_path()),
            "total": total,
            "active": active,
            "by_status": by_status,
            "by_category": by_category,
            "by_memory_type": by_type,
            "events": events,
            "db_size_bytes": size,
            "consistency": consistency(db),
        }


@router.get("/records")
async def records(
    status: str = Query("active"),
    category: str = Query(""),
    memory_type: str = Query(""),
    topic_key: str = Query(""),
    q: str = Query(""),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    status = "active" if not isinstance(status, str) else status
    category = "" if not isinstance(category, str) else category
    memory_type = "" if not isinstance(memory_type, str) else memory_type
    topic_key = "" if not isinstance(topic_key, str) else topic_key
    q = "" if not isinstance(q, str) else q
    limit = int(limit)
    offset = int(offset)
    clauses: List[str] = []
    params: List[Any] = []
    if status and status != "all":
        clauses.append("COALESCE(status, 'active') = ?")
        params.append(status)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if memory_type:
        # Applied after opening DB only when this schema has memory_type.
        pass
    if topic_key:
        clauses.append("topic_key = ?")
        params.append(topic_key)
    if q:
        clauses.append("(content LIKE ? OR topic_key LIKE ? OR decision_reason LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with connect() as db:
        cols = memory_columns(db)
        if memory_type and "memory_type" in cols:
            clauses.append("memory_type = ?")
            params.append(memory_type)
            where = "WHERE " + " AND ".join(clauses) if clauses else ""
        total = db.execute(f"SELECT COUNT(*) FROM memories {where}", params).fetchone()[0]
        cur = db.execute(
            f"SELECT {memory_select_expr(db)} FROM memories {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        rows = [dict(r) for r in cur.fetchall()]
        return {"records": rows, "count": len(rows), "total": total, "limit": limit, "offset": offset}


@router.get("/records/{fid}")
async def record_detail(fid: int):
    with connect() as db:
        rec = get_record(db, fid)
        if not rec:
            raise HTTPException(status_code=404, detail="Record not found")
        events = [dict(r) for r in db.execute(
            "SELECT event_id, memory_id, action, before_content, after_content, before_status, after_status, reason, source, created_at "
            "FROM memory_events WHERE memory_id = ? ORDER BY created_at DESC LIMIT 50",
            (fid,),
        )]
        same_topic: List[Dict[str, Any]] = []
        if rec.get("topic_key"):
            same_topic = [dict(r) for r in db.execute(
                "SELECT id, content, category, status, topic_key, confidence, updated_at FROM memories "
                "WHERE topic_key = ? AND id != ? ORDER BY updated_at DESC LIMIT 20",
                (rec["topic_key"], fid),
            )]
        return {"record": rec, "events": events, "same_topic_records": same_topic}


@router.get("/choices")
async def filter_choices():
    with connect() as db:
        cols = memory_columns(db)
        categories = [dict(r) for r in db.execute(
            "SELECT category, COUNT(*) cnt FROM memories GROUP BY category ORDER BY cnt DESC"
        )]
        types = []
        if "memory_type" in cols:
            types = [dict(r) for r in db.execute(
                "SELECT memory_type, COUNT(*) cnt FROM memories WHERE memory_type IS NOT NULL AND memory_type != '' GROUP BY memory_type ORDER BY cnt DESC"
            )]
        topic_keys = [dict(r) for r in db.execute(
            "SELECT topic_key, COUNT(*) cnt FROM memories WHERE topic_key IS NOT NULL AND topic_key != '' GROUP BY topic_key ORDER BY cnt DESC"
        )]
        return {"categories": categories, "memory_types": types, "topic_keys": topic_keys}


def set_status(fid: int, status: str, action: str, reason: str) -> Dict[str, Any]:
    with connect() as db:
        ensure_tables(db)
        before = get_record(db, fid)
        if not before:
            raise HTTPException(status_code=404, detail="Record not found")
        db.execute(
            "UPDATE memories SET status = ?, decision_reason = COALESCE(?, decision_reason), updated_at = ? WHERE id = ?",
            (status, reason, time.time(), fid),
        )
        after = get_record(db, fid)
        record_event(db, fid, action, before, after, reason)
        db.commit()
        return {"ok": True, "id": fid, "action": action, "status": status, "record": after}


@router.post("/records/{fid}/archive")
async def archive(fid: int, body: StatusBody = StatusBody()):
    return set_status(fid, "archived", "archive", body.reason)


@router.post("/records/{fid}/restore")
async def restore(fid: int, body: StatusBody = StatusBody()):
    return set_status(fid, "active", "restore", body.reason)


@router.post("/records/{fid}/approve")
async def approve(fid: int, body: StatusBody = StatusBody()):
    return set_status(fid, "active", "approve", body.reason)


@router.post("/records/{fid}/reject")
async def reject(fid: int, body: StatusBody = StatusBody()):
    return set_status(fid, "archived", "reject", body.reason)


@router.get("/health")
async def health():
    data = await stats()
    alerts = []
    if not data["consistency"].get("ok"):
        alerts.append({"level": "critical", "code": "consistency_failed", "message": "vec/FTS consistency check failed"})
    review_count = data["by_status"].get("review", 0)
    if review_count:
        alerts.append({"level": "warning", "code": "review_queue_nonempty", "message": f"{review_count} records waiting for review"})
    return {"status": "attention" if alerts else "ok", "alerts": alerts, "stats": data}
