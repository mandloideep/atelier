"""SQLite-backed durability for observability data (transcripts + eval runs).

Lives next to `checkpoints.db` and `usage.db` in the `atelier-checkpoints`
volume, so all three travel together. Two concerns, two tables, one file.

Why a separate DB file from the LangGraph checkpointer:
- LangGraph manages its schema; we don't want to share it.
- Lets us drop/rotate this file without touching conversation state.

Why not Turso/Neon: we're a single container in a single region. SQLite already
runs the rest of our state. Migration to libSQL stays SQL-compatible if/when
multi-region matters.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

_LOCK: Final = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _db_path() -> Path:
    checkpoints = os.getenv("ATELIER_CHECKPOINTS_DB", ".data/checkpoints/checkpoints.db")
    return Path(checkpoints).parent / "observability.db"


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            node TEXT,
            turn INTEGER,
            summary TEXT NOT NULL,
            data_json TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id, ts)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            grading_model TEXT,
            results_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_runs_session ON eval_runs(session_id, ts)")
    conn.commit()
    _CONN = conn
    return conn


def _enabled() -> bool:
    return os.getenv("PERSIST_OBSERVABILITY", "1") == "1"


# ── Transcripts ──────────────────────────────────────────────────────────────


def write_transcript_event(session_id: str, event: dict[str, Any]) -> None:
    """Append one event. Best-effort: swallows DB errors so the app never breaks on a write."""
    if not _enabled() or not session_id:
        return
    try:
        with _LOCK:
            _conn().execute(
                "INSERT INTO transcripts (session_id, ts, kind, node, turn, summary, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    event.get("ts") or datetime.now(UTC).isoformat(),
                    event["kind"],
                    event.get("node"),
                    event.get("turn"),
                    event.get("summary", ""),
                    json.dumps(event.get("data") or {}, ensure_ascii=False, default=str),
                ),
            )
            _conn().commit()
    except sqlite3.Error:
        pass  # never let observability break the app


def read_transcript(session_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    if not session_id:
        return []
    with _LOCK:
        rows = (
            _conn()
            .execute(
                "SELECT ts, kind, node, turn, summary, data_json "
                "FROM transcripts WHERE session_id = ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            )
            .fetchall()
        )
    events: list[dict[str, Any]] = []
    for ts, kind, node, turn, summary, data_json in rows:
        try:
            data = json.loads(data_json) if data_json else {}
        except json.JSONDecodeError:
            data = {}
        events.append(
            {"ts": ts, "kind": kind, "node": node, "turn": turn, "summary": summary, "data": data}
        )
    return events


def clear_transcript(session_id: str) -> int:
    if not session_id:
        return 0
    with _LOCK:
        cur = _conn().execute("DELETE FROM transcripts WHERE session_id = ?", (session_id,))
        _conn().commit()
        return cur.rowcount


# ── Eval runs ────────────────────────────────────────────────────────────────


def write_eval_run(session_id: str, results: list[dict[str, Any]], grading_model: str) -> None:
    if not _enabled() or not session_id:
        return
    try:
        with _LOCK:
            _conn().execute(
                "INSERT INTO eval_runs (session_id, ts, grading_model, results_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    datetime.now(UTC).isoformat(),
                    grading_model,
                    json.dumps(results, ensure_ascii=False, default=str),
                ),
            )
            _conn().commit()
    except sqlite3.Error:
        pass


def read_eval_runs(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if not session_id:
        return []
    with _LOCK:
        rows = (
            _conn()
            .execute(
                "SELECT ts, grading_model, results_json FROM eval_runs "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            .fetchall()
        )
    runs: list[dict[str, Any]] = []
    for ts, grading_model, results_json in rows:
        try:
            results = json.loads(results_json)
        except json.JSONDecodeError:
            results = []
        runs.append({"ts": ts, "grading_model": grading_model, "results": results})
    return runs
