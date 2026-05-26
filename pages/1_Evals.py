"""Evals page.

Two sections:

1. **Baseline** — renders the static `eval_results.json` produced by `evaluate.py`.
   Free, zero-risk, always available.
2. **Run on this session** — scores the active chat session's last N turns with
   DeepEval, using Gemini as the grading model. Gated by a per-IP daily cap
   stored in `usage.db`.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from backend import demo_guard, persistence, usage_tracker

st.set_page_config(page_title="Evals · Atelier", page_icon="📊", layout="wide")

st.title("📊 Evals")
st.caption(
    "How well does the system retrieve, ground, and answer? "
    "These metrics score retrieved context relevance, answer relevance, and faithfulness "
    "of the answer to the retrieved context."
)

# ── Baseline (static) ─────────────────────────────────────────────────────────

st.subheader("Baseline — `Openclaw_Research_Report.pdf`")

results_path = Path("artifacts/eval_results.json")
if not results_path.exists():
    st.info("No baseline `artifacts/eval_results.json` found. Run `make eval` to generate one.")
else:
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Failed to parse `eval_results.json`.")
        results = []

    last_run = datetime.fromtimestamp(results_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Last run: {last_run} · {len(results)} test case(s)")

    if results:
        # Per-case summary table
        rows = []
        metric_names: list[str] = []
        for r in results:
            row = {
                "Input": (r.get("input") or "")[:120],
                "Output": (r.get("actual_output") or "")[:120],
                "Pass": "✅" if r.get("success") else "❌",
            }
            for m in r.get("metrics") or []:
                col = m["name"]
                if col not in metric_names:
                    metric_names.append(col)
                row[col] = f"{m.get('score', 0):.2f}"
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Aggregate per-metric pass rate
        st.markdown("**Per-metric summary**")
        agg_rows = []
        for name in metric_names:
            scores = []
            passes = 0
            total = 0
            for r in results:
                for m in r.get("metrics") or []:
                    if m["name"] != name:
                        continue
                    total += 1
                    if m.get("passed"):
                        passes += 1
                    if isinstance(m.get("score"), (int, float)):
                        scores.append(m["score"])
            agg_rows.append(
                {
                    "Metric": name,
                    "Pass rate": f"{passes}/{total}" if total else "—",
                    "Mean score": f"{sum(scores) / len(scores):.2f}" if scores else "—",
                }
            )
        st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)

        with st.expander("Per-case detail (failure reasons, full outputs)"):
            for r in results:
                st.markdown(f"**Input:** {r.get('input')}")
                st.markdown(f"**Output:** {r.get('actual_output')}")
                for m in r.get("metrics") or []:
                    icon = "✅" if m.get("passed") else "❌"
                    st.markdown(f"- {icon} **{m['name']}** — {m.get('score', 0):.2f}")
                    if m.get("reason"):
                        st.caption(m["reason"])
                st.divider()

st.divider()

# ── Ad-hoc: score the active session ─────────────────────────────────────────

st.subheader("Run on this session")

if demo_guard.is_offline():
    st.error(demo_guard.offline_message())
elif os.getenv("EVAL_AD_HOC_ENABLED", "1") != "1":
    st.info("Ad-hoc evals are disabled on this deployment.")
elif not os.getenv("GEMINI_API_KEY"):
    st.warning(
        "Ad-hoc evals need `GEMINI_API_KEY` set on the server. "
        "The baseline above does not require it."
    )
else:
    try:
        max_turns = int(os.getenv("EVAL_AD_HOC_MAX_TURNS", "3"))
    except ValueError:
        max_turns = 3

    active_sid = st.session_state.get("active_session_id")
    sessions_meta = st.session_state.get("sessions_meta", {})
    chats_by_sid = st.session_state.get("chats", {})

    if not sessions_meta:
        st.info("No sessions yet. Start a chat on the main page.")
        st.stop()

    sorted_sids = sorted(
        sessions_meta.keys(),
        key=lambda s: sessions_meta[s].get("created_at", ""),
        reverse=True,
    )

    def _eligible_count(sid: str) -> int:
        return sum(
            1
            for m in chats_by_sid.get(sid, [])
            if m.get("role") == "assistant" and m.get("retrieval_context") and m.get("user_message")
        )

    def _label(sid: str) -> str:
        meta = sessions_meta.get(sid, {})
        name = meta.get("name", sid[:8])
        marker = " (active)" if sid == active_sid else ""
        n = _eligible_count(sid)
        flag = f"📊 {n}" if n > 0 else "·"
        return f"{flag} {name}{marker}"

    default_idx = sorted_sids.index(active_sid) if active_sid in sorted_sids else 0
    picked_sid = st.selectbox(
        "Score which session?",
        options=sorted_sids,
        index=default_idx,
        format_func=_label,
        help="📊 N = N turns with retrieval context in memory. Sessions only loaded from disk (not chatted in the current process) show 0.",
    )

    chats = chats_by_sid.get(picked_sid, [])
    eligible = [
        m
        for m in chats
        if m.get("role") == "assistant" and m.get("retrieval_context") and m.get("user_message")
    ]
    last_n = eligible[-max_turns:]

    ip = demo_guard.client_ip()
    used = usage_tracker.eval_daily_count(ip)
    cap = usage_tracker.eval_daily_ip_cap()
    cta = demo_guard.cta_message()

    st.caption(
        f"Scores the **last {max_turns} turn(s)** of `{picked_sid[:8]}…` against the documents uploaded "
        f"to that session. Grading model: **{os.getenv('EVAL_METRIC_MODEL', 'gemini-2.5-flash')}** (Gemini). "
        f"Daily cap per network: **{used}/{cap}**."
    )

    if not eligible:
        if picked_sid == active_sid:
            st.info(
                "No turns to score yet. Send a chat message in the main page first — "
                "ad-hoc evals need at least one assistant response with retrieval context."
            )
        else:
            st.info(
                "This session has no scorable turns in memory. Chats loaded from disk after a restart "
                "lose their per-turn retrieval context, so only sessions chatted in the current process "
                "can be evaluated. Pick the active session, or send a new message in this one to capture context."
            )
    else:
        st.markdown(f"Will score these **{len(last_n)} turn(s)**:")
        for t in last_n:
            st.markdown(f"- _turn {t.get('turn', '?')}_: {t['user_message'][:120]}")

        disabled = used >= cap
        if disabled:
            st.warning(
                f"Daily eval limit reached for your network ({used}/{cap}). "
                f"Try again tomorrow or {cta}."
            )

        if st.button("Run eval on these turns", disabled=disabled, type="primary"):
            with st.spinner("Scoring with Gemini…"):
                try:
                    from deepeval.metrics import (
                        AnswerRelevancyMetric,
                        ContextualRelevancyMetric,
                        FaithfulnessMetric,
                    )
                    from deepeval.test_case import LLMTestCase

                    from backend.eval_judge import GeminiJudge

                    judge = GeminiJudge()
                    threshold = 0.7

                    def metrics_factory():
                        return [
                            AnswerRelevancyMetric(threshold=threshold, model=judge),
                            FaithfulnessMetric(threshold=threshold, model=judge),
                            ContextualRelevancyMetric(threshold=threshold, model=judge),
                        ]

                    per_turn = []
                    for t in last_n:
                        test_case = LLMTestCase(
                            input=t["user_message"],
                            actual_output=t["content"],
                            retrieval_context=t["retrieval_context"],
                        )
                        metric_results = []
                        for metric in metrics_factory():
                            try:
                                metric.measure(test_case)
                                metric_results.append(
                                    {
                                        "name": metric.__class__.__name__,
                                        "score": float(metric.score)
                                        if metric.score is not None
                                        else None,
                                        "passed": bool(metric.is_successful()),
                                        "reason": metric.reason,
                                    }
                                )
                            except Exception as exc:
                                metric_results.append(
                                    {
                                        "name": metric.__class__.__name__,
                                        "score": None,
                                        "passed": False,
                                        "reason": f"metric error: {exc}",
                                    }
                                )
                        per_turn.append(
                            {
                                "turn": t.get("turn"),
                                "input": t["user_message"],
                                "output": t["content"],
                                "metrics": metric_results,
                            }
                        )

                    usage_tracker.record_eval_run(ip)
                    persistence.write_eval_run(
                        picked_sid,
                        per_turn,
                        grading_model=os.getenv("EVAL_METRIC_MODEL", "gemini-2.5-flash"),
                    )
                    st.session_state["last_ad_hoc_eval"] = per_turn
                    st.success(f"Scored {len(per_turn)} turn(s).")
                except Exception as exc:
                    st.error(f"Eval run failed: {exc}")

        if "last_ad_hoc_eval" in st.session_state:
            st.divider()
            st.markdown("**Results**")
            for result in st.session_state["last_ad_hoc_eval"]:
                st.markdown(f"### Turn {result['turn']}")
                st.markdown(f"> {result['input']}")
                cols = st.columns(len(result["metrics"]))
                for col, m in zip(cols, result["metrics"], strict=True):
                    icon = "✅" if m["passed"] else "❌"
                    score = f"{m['score']:.2f}" if m["score"] is not None else "n/a"
                    col.metric(f"{icon} {m['name']}", score)
                with st.expander("Reasons"):
                    for m in result["metrics"]:
                        st.markdown(f"**{m['name']}** — {m.get('reason') or '(no reason)'}")

st.divider()

# ── Past runs (persisted; visible even when offline) ─────────────────────────

_session_for_history = st.session_state.get("active_session_id")
# If the picker was rendered (online + has chats), it sets `picked_sid` above.
if "picked_sid" in dir():
    _session_for_history = picked_sid

if _session_for_history:
    st.subheader("Past runs for this session")
    past_runs = persistence.read_eval_runs(_session_for_history, limit=20)
    if not past_runs:
        st.caption(
            "No past runs persisted for this session. Run an eval above (when online) "
            "and the result is saved to `observability.db` so you can revisit it later."
        )
    else:
        st.caption(f"`{_session_for_history[:8]}…` · {len(past_runs)} past run(s), newest first")
        for run in past_runs:
            run_ts = run["ts"].split(".")[0].replace("T", " ")
            with st.expander(
                f"🕒 {run_ts} · {run.get('grading_model') or 'n/a'} · {len(run['results'])} turn(s)",
                expanded=False,
            ):
                for result in run["results"]:
                    st.markdown(f"**Turn {result.get('turn', '?')}** — {result.get('input', '')}")
                    metrics = result.get("metrics") or []
                    if metrics:
                        cols = st.columns(len(metrics))
                        for col, m in zip(cols, metrics, strict=True):
                            icon = "✅" if m.get("passed") else "❌"
                            score = (
                                f"{m['score']:.2f}"
                                if isinstance(m.get("score"), (int, float))
                                else "n/a"
                            )
                            col.metric(f"{icon} {m.get('name', '?')}", score)

st.divider()
st.caption(
    "Baseline was generated by `evaluate.py` against a static dataset of goldens "
    "synthesised from a single PDF. Ad-hoc runs score *your* actual conversation "
    "in this session, against the documents *you* uploaded — so the two are not "
    "directly comparable, but together they show how the system grades itself "
    "on both a fixed reference and on real use."
)
