"""Transcript page.

Renders the per-session timeline of graph node events: router decisions,
retrieval results, tool calls, claim verdicts, and final answers. Events are
in-memory only — see `backend/transcript.py` for retention/eviction policy.
"""

import streamlit as st

from backend.transcript import store as transcript_store

st.set_page_config(page_title="Transcript · Atelier", page_icon="🪧", layout="wide")

st.title("🪧 Transcript")
st.caption(
    "What the agent did, step by step. Useful for understanding why an answer "
    "looks the way it does — which tool was called, what came back, whether the "
    "retrieved context was judged relevant, and so on."
)

active_sid = st.session_state.get("active_session_id")
if not active_sid:
    st.info("No active session. Send a message on the main page first.")
    st.stop()

session_meta = st.session_state.get("sessions_meta", {}).get(active_sid, {})
session_name = session_meta.get("name", active_sid[:8])

col1, col2 = st.columns([4, 1])
col1.markdown(f"**Session:** _{session_name}_  &nbsp;·&nbsp;  `{active_sid[:8]}…`")
if col2.button("🧹 Clear transcript", use_container_width=True):
    transcript_store.clear(active_sid)
    st.rerun()

events = transcript_store.get(active_sid)

if not events:
    st.info(
        "No events recorded for this session yet. Send a chat message — "
        "events appear here as the graph runs."
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
        marker = turn_events[0] if turn_events[0]["kind"] == "router" and turn_events[0].get("node") == "user" else None
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
