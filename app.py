import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from backend import demo_guard, keep_alive
from backend.btw_handler import handle_btw
from backend.llm_factory import content_to_text, get_llm, llm_provider_label
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.transcript import store as transcript_store
from backend.vector_store import add_paper, list_papers

st.set_page_config(page_title="Atelier", page_icon="📚", layout="centered")


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_resource
def _start_keep_alive() -> bool:
    keep_alive.start()
    return True


_start_keep_alive()


SESSIONS_FILE = Path(os.getenv("ATELIER_SESSIONS_FILE", ".data/state/sessions.json"))
SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
_rename_llm = get_llm()


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError, json.JSONDecodeError:
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300] if isinstance(m.content, str) else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [{"content": d.page_content[:300], "metadata": d.metadata} for d in (v or [])]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append(
                    {"role": "assistant", "content": content, "turn": turn, "graph_state": {}}
                )
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


def maybe_preload_demo_docs(session_id: str) -> None:
    """Ingest pre-bundled demo PDFs into a fresh session so reviewers land
    on a working sandbox. No-op unless DEMO_MODE=1 and DEMO_PRELOAD_DIR is set."""
    if not demo_guard.is_demo_mode():
        return
    preload_dir = os.getenv("DEMO_PRELOAD_DIR", "").strip()
    if not preload_dir:
        return
    preloaded = st.session_state.setdefault("preloaded_sids", set())
    if session_id in preloaded:
        return
    preloaded.add(session_id)
    path = Path(preload_dir)
    if not path.is_dir():
        return
    try:
        existing = set(list_papers(session_id))
    except Exception:
        existing = set()
    for pdf_path in sorted(path.glob("*.pdf")):
        title = pdf_path.stem
        if title in existing:
            continue
        try:
            docs = load_document(str(pdf_path))
            for doc in docs:
                doc.metadata["title"] = title
            add_paper(docs, session_id)
        except Exception:
            # Preload is best-effort; never block the UI.
            pass


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id
OFFLINE = demo_guard.is_offline()
if not OFFLINE:
    maybe_preload_demo_docs(active_sid)


def _enforce_ingest_caps(new_chunk_count: int) -> str | None:
    """Return an error message if the new doc would violate any ingest cap, else None."""
    max_chunks = demo_guard.max_chunks_per_doc()
    if new_chunk_count > max_chunks:
        return (
            f"Document produced {new_chunk_count} chunks but the per-document limit is {max_chunks}. "
            f"Try a shorter file."
        )
    try:
        existing = len(list_papers(active_sid) or [])
    except Exception:
        existing = 0
    max_docs = demo_guard.max_docs_per_session()
    if existing >= max_docs:
        return (
            f"This session already has {existing} document(s) — the per-session limit is {max_docs}. "
            f"Start a new chat to add more."
        )
    return None


def _render_status_badge(session_id: str) -> None:
    if not demo_guard.is_demo_mode():
        return
    status = demo_guard.llm_status()
    used = demo_guard.turns_used(session_id)
    cap = demo_guard.session_message_cap()
    ip_used = demo_guard.ip_count()
    ip_cap = demo_guard.ip_cap()
    cta = demo_guard.cta_message()
    if status == "quota_exhausted":
        st.warning(f"🟡 Daily free quota exhausted — try again tomorrow or {cta}.")
    elif status == "unhealthy":
        st.error(f"🔴 API unavailable — please {cta}.")
    elif ip_used >= ip_cap:
        st.warning(f"🟡 Daily limit reached for your network — try again tomorrow or {cta}.")
    elif used >= cap:
        st.warning(f"⚠️ Session limit reached — {cta}.")
    else:
        st.success(f"🟢 API healthy · session {used}/{cap} · today {ip_used}/{ip_cap}")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if OFFLINE:
        st.error(demo_guard.offline_message())
    _render_status_badge(active_sid)
    if st.button("+ New Chat", use_container_width=True, disabled=OFFLINE):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown("## 📄 Documents")

    # ── Section 1: File upload ─────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        f"PDF, TXT, or Markdown (max {demo_guard.max_upload_mb()} MB each)",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
        disabled=OFFLINE,
    )
    if st.button(
        "Add Files",
        use_container_width=True,
        key="btn_add_files",
        disabled=OFFLINE,
    ):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            max_bytes = demo_guard.max_upload_mb() * 1024 * 1024
            with st.spinner("Processing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    if getattr(f, "size", 0) > max_bytes:
                        st.error(
                            f"Rejected: {f.name} — {f.size / (1024 * 1024):.1f} MB exceeds the "
                            f"{demo_guard.max_upload_mb()} MB limit."
                        )
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        cap_err = _enforce_ingest_caps(len(docs))
                        if cap_err:
                            st.error(f"Rejected: {f.name} — {cap_err}")
                            continue
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button(
        "Load URLs",
        use_container_width=True,
        key="btn_load_urls",
        disabled=OFFLINE,
    ):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        cap_err = _enforce_ingest_caps(len(docs))
                        if cap_err:
                            st.error(f"Rejected: {url[:60]} — {cap_err}")
                            continue
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button(
        "Load ArXiv Paper",
        use_container_width=True,
        key="btn_load_arxiv",
        disabled=OFFLINE,
    ):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    cap_err = _enforce_ingest_caps(len(docs))
                    if cap_err:
                        st.error(f"Rejected: {arxiv_title.strip()} — {cap_err}")
                    else:
                        add_paper(docs, active_sid)
                        loaded_title = (
                            docs[0].metadata.get("title") if docs else arxiv_title.strip()
                        )
                        st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Loaded Documents")
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📚 Atelier — Research Paper Assistant")
st.markdown(
    "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
    "🌐 **Search the web** for the latest findings\n\n"
    "> Upload documents in the sidebar and start chatting below."
)

if OFFLINE:
    st.error(demo_guard.offline_message())
elif demo_guard.is_demo_mode():
    cap = demo_guard.session_message_cap()
    email = demo_guard.contact_email()
    contact = f" — email **{email}** for full access" if email else ""
    st.info(
        f"**Demo mode** — running on **{llm_provider_label()}** (free tier). "
        f"Up to **{cap} messages per session**{contact}."
    )

st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
_chat_placeholder = (
    "🚫 AI disabled by maintainer — browse-only mode"
    if OFFLINE
    else "Ask about your papers, verify a claim, or search the web…"
)
if prompt := st.chat_input(_chat_placeholder, disabled=OFFLINE):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})

        # ── Gates: refuse to invoke the graph if offline, over cap, or LLM down.
        blocked_reason: str | None = None
        if OFFLINE:
            blocked_reason = demo_guard.offline_message()
        elif demo_guard.over_ip_cap():
            blocked_reason = (
                f"You've reached today's {demo_guard.ip_cap()}-message limit for your network. "
                f"Please try again tomorrow or {demo_guard.cta_message()}."
            )
        elif demo_guard.over_cap(active_sid):
            blocked_reason = (
                f"You've reached the {demo_guard.session_message_cap()}-message demo limit "
                f"for this session. Please {demo_guard.cta_message()}."
            )
        elif demo_guard.llm_status() == "quota_exhausted":
            blocked_reason = (
                "The daily free quota for the demo LLM is exhausted. "
                f"Please try again tomorrow or {demo_guard.cta_message()}."
            )
        elif demo_guard.llm_status() == "unhealthy":
            blocked_reason = (
                f"The model is currently unavailable. Please {demo_guard.cta_message()}."
            )

        if blocked_reason is not None:
            with st.chat_message("assistant"):
                st.warning(blocked_reason)
            st.session_state.chats[active_sid].append(
                {
                    "role": "assistant",
                    "content": blocked_reason,
                    "graph_state": {"blocked": True},
                    "turn": st.session_state.turns[active_sid],
                }
            )
            st.stop()

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        next_turn = st.session_state.turns[active_sid] + 1
        transcript_store.append(
            active_sid,
            kind="router",
            summary=f"turn {next_turn} started — '{prompt[:80]}'",
            node="user",
            turn=next_turn,
            data={"user_message": prompt},
        )

        response_text = ""
        error_text = ""
        state_snapshot: dict = {}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                    if (
                        metadata.get("langgraph_node") == "generate_answer"
                        and hasattr(chunk, "content")
                        and chunk.content
                    ):
                        chunk_text = content_to_text(chunk.content)
                        if chunk_text:
                            response_text += chunk_text
                            placeholder.markdown(response_text + "▌")

                if not response_text:
                    final_values = graph.get_state(config).values
                    response_text = (
                        content_to_text(final_values.get("answer")) or "No response generated."
                    )

                placeholder.markdown(response_text)

                final_values = graph.get_state(config).values
                state_snapshot = _serialize_state(final_values)
                retrieval_context = [
                    d.page_content for d in (final_values.get("retrieved_docs") or [])
                ]

                # Success: bump turn counter and clear any previous unhealthy state.
                st.session_state.turns[active_sid] += 1
                current_turn = st.session_state.turns[active_sid]
                demo_guard.mark_llm_healthy()
                demo_guard.record_successful_turn()

                with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                    st.json(state_snapshot)
            except Exception as e:
                status = demo_guard.classify_exception(e)
                if status == "quota_exhausted":
                    demo_guard.mark_quota_exhausted(str(e))
                    error_text = (
                        "Daily free quota for the demo LLM is exhausted. "
                        f"Please try again tomorrow or {demo_guard.cta_message()}."
                    )
                else:
                    demo_guard.mark_llm_unhealthy(str(e))
                    error_text = (
                        f"The model is currently unavailable. Please {demo_guard.cta_message()}."
                    )
                placeholder.empty()
                st.error(error_text)
                state_snapshot = {"error": str(e), "error_type": type(e).__name__}
                current_turn = demo_guard.turns_used(active_sid)
                retrieval_context = []

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text or error_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
                "user_message": prompt,
                "retrieval_context": retrieval_context,
            }
        )

        if is_first_message:
            st.rerun()
