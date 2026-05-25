"""Persistent per-IP daily counter, backed by SQLite.

Lives in a file next to the LangGraph checkpoints DB so it ends up in the same
Dokploy volume (`atelier-checkpoints`) and survives restarts/redeploys.
"""

import os
import sqlite3
import threading
from datetime import date
from pathlib import Path
from typing import Final

_LOCK: Final = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _db_path() -> Path:
    checkpoints = os.getenv("ATELIER_CHECKPOINTS_DB", "checkpoints.db")
    return Path(checkpoints).parent / "usage.db"


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS usage ("
        "  ip TEXT NOT NULL,"
        "  day TEXT NOT NULL,"
        "  count INTEGER NOT NULL DEFAULT 0,"
        "  PRIMARY KEY (ip, day)"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS eval_usage ("
        "  ip TEXT NOT NULL,"
        "  day TEXT NOT NULL,"
        "  count INTEGER NOT NULL DEFAULT 0,"
        "  PRIMARY KEY (ip, day)"
        ")"
    )
    conn.commit()
    _CONN = conn
    return conn


def daily_ip_cap() -> int:
    try:
        return int(os.getenv("DAILY_IP_CAP", "20"))
    except ValueError:
        return 20


def _today() -> str:
    return date.today().isoformat()


def daily_count(ip: str) -> int:
    if not ip:
        return 0
    with _LOCK:
        row = _conn().execute(
            "SELECT count FROM usage WHERE ip = ? AND day = ?",
            (ip, _today()),
        ).fetchone()
    return int(row[0]) if row else 0


def increment(ip: str) -> int:
    """Atomically increment today's counter for an IP and return the new value."""
    if not ip:
        return 0
    with _LOCK:
        conn = _conn()
        conn.execute(
            "INSERT INTO usage (ip, day, count) VALUES (?, ?, 1) "
            "ON CONFLICT(ip, day) DO UPDATE SET count = count + 1",
            (ip, _today()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT count FROM usage WHERE ip = ? AND day = ?",
            (ip, _today()),
        ).fetchone()
    return int(row[0]) if row else 0


def over_ip_cap(ip: str) -> bool:
    return daily_count(ip) >= daily_ip_cap()


# ── Ad-hoc eval cap ────────────────────────────────────────────────────────────
# Tracked separately from chat-message cap so a visitor can both chat and
# trigger a small eval batch without the two caps cannibalising each other.


def eval_daily_ip_cap() -> int:
    try:
        return int(os.getenv("EVAL_AD_HOC_DAILY_IP_CAP", "2"))
    except ValueError:
        return 2


def eval_daily_count(ip: str) -> int:
    if not ip:
        return 0
    with _LOCK:
        row = _conn().execute(
            "SELECT count FROM eval_usage WHERE ip = ? AND day = ?",
            (ip, _today()),
        ).fetchone()
    return int(row[0]) if row else 0


def record_eval_run(ip: str) -> int:
    if not ip:
        return 0
    with _LOCK:
        conn = _conn()
        conn.execute(
            "INSERT INTO eval_usage (ip, day, count) VALUES (?, ?, 1) "
            "ON CONFLICT(ip, day) DO UPDATE SET count = count + 1",
            (ip, _today()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT count FROM eval_usage WHERE ip = ? AND day = ?",
            (ip, _today()),
        ).fetchone()
    return int(row[0]) if row else 0


def over_eval_cap(ip: str) -> bool:
    return eval_daily_count(ip) >= eval_daily_ip_cap()
