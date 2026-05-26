"""Per-session demo guardrails: message cap + LLM health flag.

State lives in `st.session_state` so it resets per browser session, which is
the correct scope for a public portfolio demo. Hard caps that need to survive
restarts belong on the provider side (Google AI Studio free tier, etc.).
"""

import os
from typing import Literal

import streamlit as st

from backend import usage_tracker

LLMStatus = Literal["healthy", "unhealthy", "quota_exhausted"]


def client_ip() -> str:
    """Best-effort client IP, sourced from Traefik's X-Forwarded-For header.

    Falls back to a literal so local dev still has a stable bucket.
    """
    try:
        headers = st.context.headers or {}
    except Exception:
        headers = {}
    xff = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for") or ""
    if xff:
        return xff.split(",")[0].strip()
    real = headers.get("X-Real-Ip") or headers.get("x-real-ip")
    if real:
        return real.strip()
    return "local"


def over_ip_cap(ip: str | None = None) -> bool:
    if not is_demo_mode():
        return False
    return usage_tracker.over_ip_cap(ip or client_ip())


def ip_count(ip: str | None = None) -> int:
    return usage_tracker.daily_count(ip or client_ip())


def ip_cap() -> int:
    return usage_tracker.daily_ip_cap()


def record_successful_turn(ip: str | None = None) -> None:
    """Bump the per-IP daily counter. Call only on success."""
    if not is_demo_mode():
        return
    usage_tracker.increment(ip or client_ip())


def is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "0") == "1"


def is_offline() -> bool:
    """Master kill switch. When true, all AI features are gated off but
    read-only browsing (history, transcript, evals baseline) still works."""
    return os.getenv("APP_OFFLINE", "0") == "1"


def offline_message() -> str:
    email = contact_email()
    if email:
        return (
            f"🚫 **AI features are currently disabled by the maintainer.** "
            f"You can still browse past sessions, the transcript, and the evals baseline. "
            f"Email **{email}** to request access."
        )
    return (
        "🚫 **AI features are currently disabled by the maintainer.** "
        "You can still browse past sessions, the transcript, and the evals baseline."
    )


def session_message_cap() -> int:
    try:
        return int(os.getenv("SESSION_MESSAGE_CAP", "5"))
    except ValueError:
        return 5


def max_upload_mb() -> int:
    try:
        return int(os.getenv("MAX_UPLOAD_MB", "10"))
    except ValueError:
        return 10


def max_chunks_per_doc() -> int:
    try:
        return int(os.getenv("MAX_CHUNKS_PER_DOC", "200"))
    except ValueError:
        return 200


def max_docs_per_session() -> int:
    try:
        return int(os.getenv("MAX_DOCS_PER_SESSION", "5"))
    except ValueError:
        return 5


def contact_email() -> str:
    return os.getenv("CONTACT_EMAIL", "").strip()


def turns_used(session_id: str) -> int:
    return int(st.session_state.get("turns", {}).get(session_id, 0))


def over_cap(session_id: str) -> bool:
    if not is_demo_mode():
        return False
    return turns_used(session_id) >= session_message_cap()


def llm_status() -> LLMStatus:
    return st.session_state.get("llm_status", "healthy")


def mark_llm_unhealthy(reason: str = "") -> None:
    st.session_state["llm_status"] = "unhealthy"
    st.session_state["llm_status_reason"] = reason


def mark_quota_exhausted(reason: str = "") -> None:
    st.session_state["llm_status"] = "quota_exhausted"
    st.session_state["llm_status_reason"] = reason


def mark_llm_healthy() -> None:
    st.session_state["llm_status"] = "healthy"
    st.session_state.pop("llm_status_reason", None)


def classify_exception(exc: BaseException) -> LLMStatus:
    """Map a provider exception to a UI status."""
    name = type(exc).__name__
    msg = str(exc).lower()
    # google.api_core.exceptions.ResourceExhausted (429) or 'quota'
    if name in {"ResourceExhausted"} or "quota" in msg or "rate limit" in msg or "429" in msg:
        return "quota_exhausted"
    return "unhealthy"


def cta_message() -> str:
    email = contact_email()
    if email:
        return f"email **{email}** for full access"
    return "contact the maintainer for full access"
