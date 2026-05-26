"""Transcript page.

Renders the per-session timeline of graph node events: router decisions,
retrieval results, tool calls, claim verdicts, and final answers. Events are
in-memory only — see `backend/transcript.py` for retention/eviction policy.
"""

import streamlit as st

from backend import demo_guard
from backend.transcript import store as transcript_store

st.set_page_config(page_title="Transcript · Atelier", page_icon="🪧", layout="wide")

if demo_guard.is_offline():
    st.error(demo_guard.offline_message())

st.title("🪧 Transcript")
st.caption(
    "What the agent did, step by step. Useful for understanding why an answer "
    "looks the way it does — which tool was called, what came back, whether the "
    "retrieved context was judged relevant, and so on."
)

active_sid = st.session_state.get("active_session_id")
sessions_meta = st.session_state.get("sessions_meta", {})
if not sessions_meta:
    st.info("No sessions yet. Start a chat on the main page.")
    st.stop()

# Sort sessions newest-first for the picker
sorted_sids = sorted(
    sessions_meta.keys(),
    key=lambda s: sessions_meta[s].get("created_at", ""),
    reverse=True,
)


def _label(sid: str) -> str:
    meta = sessions_meta.get(sid, {})
    name = meta.get("name", sid[:8])
    marker = " (active)" if sid == active_sid else ""
    has = "🪧" if transcript_store.get(sid) else "·"
    return f"{has} {name}{marker}"


default_idx = sorted_sids.index(active_sid) if active_sid in sorted_sids else 0
col1, col2 = st.columns([4, 1])
picked_sid = col1.selectbox(
    "Session",
    options=sorted_sids,
    index=default_idx,
    format_func=_label,
    label_visibility="collapsed",
)
if col2.button("🧹 Clear transcript", use_container_width=True):
    transcript_store.clear(picked_sid)
    st.rerun()

st.caption(
    f"`{picked_sid[:8]}…` — 🪧 marks sessions with in-memory events. "
    "Sessions created in a previous container run or evicted (LRU/TTL) won't have one."
)

events = transcript_store.get(picked_sid)

if not events:
    if picked_sid == active_sid:
        st.info(
            "No events recorded for this session yet. Send a chat message — "
            "events appear here as the graph runs."
        )
    else:
        st.info(
            "No transcript in memory for this session. "
            "Transcripts are in-process only — this one was probably created in a previous run "
            "or evicted. Switch back to the active session, or chat in this one to start a new transcript."
        )
else:
    # Group events into turns. A new turn starts at every kind="router" event with
    # node="user" (the marker we emit in app.py). Falls back to one big group.
    turns: list[list[dict]] = []
    current: list[dict] = []
    for ev in events:
        if ev["kind"] == "router" and ev.get("node") == "user":
            if current:
                turns.append(current)
            current = [ev]
        else:
            if not current:
                current = []
            current.append(ev)
    if current:
        turns.append(current)

    st.markdown(f"**{len(turns)} turn(s) · {len(events)} event(s)**")

    KIND_ICONS = {
        "router": "🧭",
        "retrieval": "🔎",
        "tool_call": "🛠️",
        "tool_result": "📥",
        "verdict": "⚖️",
        "rewrite": "✏️",
        "relevancy": "🎯",
        "answer": "💬",
        "error": "🛑",
    }

    for i, turn_events in enumerate(turns, 1):
        marker = (
            turn_events[0]
            if turn_events[0]["kind"] == "router" and turn_events[0].get("node") == "user"
            else None
        )
        header = f"Turn {i}"
        if marker and marker.get("data", {}).get("user_message"):
            header += f" — {marker['data']['user_message'][:80]}"
        with st.expander(header, expanded=(i == len(turns))):
            for ev in turn_events:
                icon = KIND_ICONS.get(ev["kind"], "•")
                ts = ev["ts"].split("T")[-1][:8]
                node = ev.get("node") or ""
                st.markdown(f"{icon} `{ts}` **{ev['kind']}** _({node})_ — {ev['summary']}")
                if ev.get("data"):
                    with st.expander("data", expanded=False):
                        st.json(ev["data"])

# ── Footer: memory stats ──────────────────────────────────────────────────────

st.divider()
stats = transcript_store.stats()
limit_mb = stats["cgroup_limit_bytes"] / (1024 * 1024) if stats["cgroup_limit_bytes"] else None
rss_mb = stats["rss_bytes"] / (1024 * 1024)
pct = stats["rss_pct_of_limit"]

cols = st.columns(4)
cols[0].metric("Sessions in memory", stats["sessions"])
cols[1].metric("Total events", stats["events"])
cols[2].metric("Evicted (lifetime)", stats["evicted"])
if limit_mb:
    cols[3].metric("Process RSS", f"{rss_mb:.0f} MB", f"{pct:.1f}% of {limit_mb:.0f} MB")
else:
    cols[3].metric("Process RSS", f"{rss_mb:.0f} MB", "no cgroup limit")

st.caption(
    "Transcripts are kept in-memory only. They are evicted automatically by an LRU + TTL policy "
    "(see `TRANSCRIPT_MAX_SESSIONS`, `TRANSCRIPT_TTL_HOURS` env vars), and aggressively when "
    "process memory exceeds `TRANSCRIPT_MEMORY_THRESHOLD_PCT` of the container limit. "
    "Use **Clear transcript** above to drop the current session yourself."
)
