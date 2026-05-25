"""In-memory per-session transcript of graph node events.

Captures router decisions, retrieval results, tool calls, claim verdicts, and
final answers so the Transcript page can render a human-readable timeline of
what the agent did. No persistence — survives container restart is explicitly
not a goal; the LangGraph checkpointer holds the durable state.

Eviction is bounded by three knobs (env-tunable):

* TRANSCRIPT_MAX_SESSIONS         — LRU cap on distinct sessions kept
* TRANSCRIPT_MAX_EVENTS_PER_SESSION — ring buffer per session
* TRANSCRIPT_TTL_HOURS            — drop sessions untouched for this long

And one safety valve:

* TRANSCRIPT_MEMORY_THRESHOLD_PCT — when process RSS exceeds this fraction of
  the cgroup memory limit, drop the oldest half of sessions immediately.

Eviction runs on every append. All operations are O(N_sessions) at worst,
which is fine because N_sessions is capped.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Literal

import psutil

EventKind = Literal[
    "router", "retrieval", "tool_call", "tool_result",
    "verdict", "answer", "rewrite", "relevancy", "error",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _is_enabled() -> bool:
    return os.getenv("TRANSCRIPT_ENABLED", "1") == "1"


def _cgroup_memory_limit_bytes() -> int | None:
    """Read the container's memory limit from cgroup v2, then v1. None if unreadable."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw == "max":
                return None
            value = int(raw)
            # cgroup v1 reports a huge sentinel when uncapped
            if value > 1 << 60:
                return None
            return value
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return None


class TranscriptStore:
    def __init__(self) -> None:
        self._sessions: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._last_touched: dict[str, float] = {}
        self._evicted_count = 0
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────────────────────

    def append(
        self,
        session_id: str,
        kind: EventKind,
        summary: str,
        *,
        node: str | None = None,
        turn: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if not _is_enabled() or not session_id:
            return
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "summary": summary,
            "node": node,
            "turn": turn,
            "data": _truncate(data or {}),
        }
        with self._lock:
            events = self._sessions.get(session_id)
            if events is None:
                events = []
                self._sessions[session_id] = events
            else:
                self._sessions.move_to_end(session_id)
            events.append(event)
            self._last_touched[session_id] = time.time()
            self._evict_locked(len(events))

    def get(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_touched.pop(session_id, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rss = psutil.Process().memory_info().rss
            limit = _cgroup_memory_limit_bytes()
            return {
                "sessions": len(self._sessions),
                "events": sum(len(e) for e in self._sessions.values()),
                "evicted": self._evicted_count,
                "rss_bytes": rss,
                "cgroup_limit_bytes": limit,
                "rss_pct_of_limit": (rss / limit * 100) if limit else None,
            }

    # ── eviction (caller holds lock) ────────────────────────────────────────

    def _evict_locked(self, current_session_len: int) -> None:
        max_events = _env_int("TRANSCRIPT_MAX_EVENTS_PER_SESSION", 500)
        max_sessions = _env_int("TRANSCRIPT_MAX_SESSIONS", 50)
        ttl_hours = _env_int("TRANSCRIPT_TTL_HOURS", 24)
        memory_threshold_pct = _env_int("TRANSCRIPT_MEMORY_THRESHOLD_PCT", 80)

        # Per-session ring buffer: drop oldest events on the session we just wrote to.
        if current_session_len > max_events:
            # The session being appended is at the end of the OrderedDict.
            sid, events = next(reversed(self._sessions.items()))
            overflow = len(events) - max_events
            if overflow > 0:
                del events[:overflow]

        # TTL sweep: drop sessions untouched for too long.
        if ttl_hours > 0:
            cutoff = time.time() - ttl_hours * 3600
            stale = [sid for sid, t in self._last_touched.items() if t < cutoff]
            for sid in stale:
                if self._sessions.pop(sid, None) is not None:
                    self._evicted_count += 1
                self._last_touched.pop(sid, None)

        # LRU cap: drop the oldest until we're back under the limit.
        while len(self._sessions) > max_sessions:
            self._sessions.popitem(last=False)
            self._evicted_count += 1

        # Memory-pressure safety valve.
        limit = _cgroup_memory_limit_bytes()
        if limit:
            try:
                rss = psutil.Process().memory_info().rss
            except Exception:
                return
            if rss / limit * 100 >= memory_threshold_pct:
                # Drop oldest 50% of sessions; aggressive on purpose.
                victims = max(1, len(self._sessions) // 2)
                for _ in range(victims):
                    if not self._sessions:
                        break
                    self._sessions.popitem(last=False)
                    self._evicted_count += 1


def _truncate(data: dict[str, Any]) -> dict[str, Any]:
    """Bound the size of per-event payloads so a runaway tool result can't blow memory."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > 600:
            out[k] = v[:600] + f"… [+{len(v) - 600} chars]"
        elif isinstance(v, list) and len(v) > 20:
            out[k] = v[:20] + [f"… [+{len(v) - 20} items]"]
        else:
            out[k] = v
    return out


# Module-level singleton. Streamlit reruns import this module fresh in dev,
# but in a long-running process the OrderedDict survives across requests.
store = TranscriptStore()
