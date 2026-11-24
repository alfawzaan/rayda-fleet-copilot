"""Rayda Fleet Copilot — Linear-inspired Enterprise Streamlit Workspace.

Navigation views (via clean left rail):
1. Copilot          — Split-screen workbench: scrollable chat + live Evidence/Trace & Fleet Pulse inspector + inline HITL approval cards.
2. Fleet Health     — Native progress-bar inventory table, Altair time-series drill-down, and 1-click Copilot investigation triggers.
3. Operations       — Unified governance console for persisted ITSM actions and immutable security/operations audit logs.
4. Live Ingestion   — Clean 2-column realtime snapshot builder and batch NDJSON/JSON uploader.
"""

from __future__ import annotations

import html
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from fleet_copilot.agent.service import FleetCopilotService, TurnResult
from fleet_copilot.analytics.trends import generate_fleet_insights
from fleet_copilot.config import get_settings

st.set_page_config(
    page_title="Rayda Fleet Copilot",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Restrained Enterprise Design System (Zero Emojis, Clean Hairlines, Inter)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
}

/* Hide Streamlit deploy chrome & tighten top bar cleanly without clipping content */
[data-testid="stAppDeployButton"],
[data-testid="stDeployButton"],
[data-testid="stMainMenu"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.6rem !important;
}
footer { visibility: hidden; }

.block-container {
    padding-top: 3.25rem !important;
    padding-bottom: 1.6rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px;
}

/* Crisp Left Rail Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E5E7EB;
    min-width: 265px;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.65rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.25rem !important;
}
section[data-testid="stSidebar"] hr {
    margin: 0.5rem 0 !important;
    border-color: #E5E7EB !important;
}
/* Neutralize Streamlit's built-in -1rem negative margin on sidebar markdown/caption containers */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    margin-bottom: 0 !important;
}
/* Clean native sidebar captions used as section headers (compact & zero clipping) */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: #64748B !important;
    margin: 0.05rem 0 0.1rem 0.15rem !important;
    line-height: 1.25 !important;
}

/* Strictly left-align ALL sidebar buttons and tertiary links globally */
section[data-testid="stSidebar"] button,
button[data-testid="stBaseButton-tertiary"] {
    justify-content: flex-start !important;
    text-align: left !important;
    gap: 0.5rem !important;
}
section[data-testid="stSidebar"] button > div,
button[data-testid="stBaseButton-tertiary"] > div {
    justify-content: flex-start !important;
    text-align: left !important;
    width: 100%;
}

/* Sidebar ghost/inactive navigation & thread items */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] {
    padding: 0.26rem 0.6rem !important;
    border-radius: 7px !important;
    color: #374151 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    min-height: 1.85rem !important;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"]:hover {
    background-color: #EEF2F6 !important;
    color: #0F172A !important;
}

/* Sidebar ACTIVE navigation / thread item (subtle indigo highlight, no heavy box) */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    padding: 0.26rem 0.6rem !important;
    border-radius: 7px !important;
    background-color: #EEF2FF !important;
    color: #3730A3 !important;
    border: 1px solid #C7D2FE !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    min-height: 1.85rem !important;
}

/* Sidebar PRIMARY action button (+ New conversation) */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    justify-content: center !important;
    padding: 0.42rem 0.75rem !important;
    border-radius: 8px !important;
    background-color: #111827 !important;
    color: #FFFFFF !important;
    border: 1px solid #111827 !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    min-height: 2.15rem !important;
    margin-bottom: 0.25rem !important;
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] > div {
    justify-content: center !important;
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {
    background-color: #1F2937 !important;
}

/* Sidebar Popover trigger button */
section[data-testid="stSidebar"] [data-testid="stPopover"] > button {
    justify-content: flex-start !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    color: #374151 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 0.38rem 0.65rem !important;
}

/* Compact source/trace trigger pill inside chat messages */
[data-testid="stChatMessage"] button[data-testid="stBaseButton-tertiary"],
[data-testid="stChatMessage"] button[data-testid="stBaseButton-secondary"] {
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    padding: 0.22rem 0.6rem !important;
    min-height: 1.75rem !important;
    border-radius: 999px !important;
    width: auto !important;
}
[data-testid="stChatMessage"] button[data-testid="stBaseButton-tertiary"] {
    color: #475569 !important;
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
}
[data-testid="stChatMessage"] button[data-testid="stBaseButton-tertiary"]:hover {
    background-color: #EEF2FF !important;
    border-color: #C7D2FE !important;
    color: #3730A3 !important;
}
[data-testid="stChatMessage"] button[data-testid="stBaseButton-secondary"] {
    color: #3730A3 !important;
    background-color: #EEF2FF !important;
    border: 1px solid #A5B4FC !important;
}

/* Chat message typography scale (prevents giant LLM markdown headings) */
[data-testid="stChatMessage"] {
    padding: 0.7rem 0.85rem;
    border-radius: 10px;
}
[data-testid="stChatMessage"] h1 { font-size: 1.12rem !important; font-weight: 700 !important; margin: 0.5rem 0 0.3rem 0 !important; }
[data-testid="stChatMessage"] h2 { font-size: 1.02rem !important; font-weight: 700 !important; margin: 0.5rem 0 0.25rem 0 !important; }
[data-testid="stChatMessage"] h3 { font-size: 0.94rem !important; font-weight: 600 !important; margin: 0.45rem 0 0.2rem 0 !important; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-size: 0.88rem !important;
    line-height: 1.55 !important;
    color: #1E293B;
}
[data-testid="stChatMessage"] code {
    font-size: 0.8rem !important;
    background: #F1F5F9;
    padding: 1px 5px;
    border-radius: 4px;
    color: #0F172A;
}

/* Quick investigation buttons in right inspector */
button[data-testid="stBaseButton-tertiary"] {
    padding: 0.3rem 0.5rem !important;
    font-size: 0.82rem !important;
    color: #334155 !important;
    border-radius: 6px !important;
}
button[data-testid="stBaseButton-tertiary"]:hover {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
}

/* Brand Lockup */
.brand-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.1rem 0.15rem 0.65rem 0.15rem;
    margin-bottom: 0.35rem;
    border-bottom: 1px solid #E5E7EB;
}
.brand-mark {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: #111827;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: -0.02em;
}
.brand-name {
    font-size: 0.94rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.15;
}
.brand-tagline {
    font-size: 0.72rem;
    color: #6B7280;
}

/* Page Title Header */
.page-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
}
.page-sub {
    font-size: 0.84rem;
    color: #64748B;
    margin: 0.15rem 0 0 0;
}

/* Chat Empty State */
.empty-hero {
    padding: 2.4rem 1.25rem 1.4rem 1.25rem;
    text-align: left;
}
.empty-hero h3 {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
}
.empty-hero p {
    font-size: 0.88rem;
    color: #64748B;
    margin: 0;
    max-width: 560px;
    line-height: 1.5;
}

/* Evidence Inspector Items */
.ev-item {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.45rem;
    background: #FFFFFF;
}
.ev-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin-bottom: 3px;
}
.ev-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #4338CA;
    background: #EEF2FF;
    padding: 1px 6px;
    border-radius: 4px;
}
.ev-entity {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #111827;
}
.ev-body {
    font-size: 0.8rem;
    color: #374151;
    line-height: 1.4;
}
.ev-foot {
    font-size: 0.71rem;
    color: #6B7280;
    margin-top: 3px;
}

/* Trace Step Row */
.trace-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 0.45rem 0;
    border-bottom: 1px solid #F3F4F6;
    font-size: 0.8rem;
}
.trace-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.dot-ok    { background: #10B981; }
.dot-stage { background: #F59E0B; }
.dot-err   { background: #EF4444; }

/* Fleet Pulse Row */
.pulse-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F3F4F6;
    font-size: 0.83rem;
}
.pulse-label { color: #374151; font-weight: 500; }
.pulse-val-ok { color: #059669; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.pulse-val-warn { color: #D97706; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.pulse-val-crit { color: #DC2626; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

/* Parameter Chips inside HITL Cards */
.kv-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0.4rem 0 0.55rem 0;
}
.kv-chip {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 0.78rem;
    color: #1F2937;
}
.kv-chip span {
    color: #6B7280;
    margin-right: 4px;
    font-weight: 500;
}

/* Clean metric borders */
[data-testid="stMetric"] {
    padding: 0.15rem 0;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #6B7280 !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}
</style>
"""

_ACTION_LABELS = {
    "create_upgrade_order": "Hardware Upgrade Order",
    "open_remediation_ticket": "IT Remediation Ticket",
    "flag_device_for_replacement": "Device Replacement Flag",
    "notify_employee": "Employee Notification",
}

_NAV_ITEMS = [
    ("copilot", "Copilot", ":material/forum:"),
    ("fleet", "Fleet Health", ":material/monitoring:"),
    ("operations", "Operations & Audit", ":material/verified_user:"),
    ("ingestion", "Live Ingestion", ":material/upload_file:"),
]


# ---------------------------------------------------------------------------
# Service & Session State
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_cached_service(
    llm_provider: str,
    llm_model: str,
    llm_temperature: float,
    has_google_key: bool,
) -> FleetCopilotService:
    return FleetCopilotService(settings=get_settings(), auto_ingest_default_dataset=True)


def get_service() -> FleetCopilotService:
    s = get_settings()
    return _get_cached_service(
        s.llm_provider,
        s.llm_model,
        s.llm_temperature,
        bool(s.google_api_key),
    )


def _hydrate_thread_if_needed(service: FleetCopilotService, thread_id: str, tenant: str) -> None:
    if thread_id not in st.session_state.chat_history_by_thread:
        st.session_state.chat_history_by_thread[thread_id] = service.get_thread_history(
            thread_id, tenant
        )


def _ensure_session_state(service: FleetCopilotService, companies: list[str]) -> None:
    if "active_view" not in st.session_state:
        st.session_state.active_view = "copilot"
    if "active_tenant" not in st.session_state:
        st.session_state.active_tenant = companies[0] if companies else "acme-001"
    if "actor_id" not in st.session_state:
        st.session_state.actor_id = "it-admin"
    if "threads_by_tenant" not in st.session_state:
        st.session_state.threads_by_tenant = {}
    if "session_thread_ids_by_tenant" not in st.session_state:
        st.session_state.session_thread_ids_by_tenant = {}
    if "chat_history_by_thread" not in st.session_state:
        st.session_state.chat_history_by_thread = {}
    if "selected_msg_by_thread" not in st.session_state:
        st.session_state.selected_msg_by_thread = {}

    tenant = st.session_state.active_tenant
    if tenant not in st.session_state.threads_by_tenant:
        persisted = service.list_threads(tenant)
        st.session_state.threads_by_tenant[tenant] = (
            persisted[0]["thread_id"] if persisted else f"{tenant}-{uuid.uuid4().hex[:8]}"
        )

    tid = st.session_state.threads_by_tenant[tenant]
    st.session_state.session_thread_ids_by_tenant.setdefault(tenant, [])
    if tid not in st.session_state.session_thread_ids_by_tenant[tenant]:
        st.session_state.session_thread_ids_by_tenant[tenant].insert(0, tid)
    _hydrate_thread_if_needed(service, tid, tenant)


def _send_prompt_to_copilot(prompt: str) -> None:
    """Switch to Copilot view and queue a prompt for immediate execution."""
    st.session_state.active_view = "copilot"
    st.session_state["queued_prompt"] = prompt
    st.rerun()


def _execute_turn(
    service: FleetCopilotService,
    tenant: str,
    actor_id: str,
    thread_id: str,
    prompt_text: str,
) -> None:
    history = st.session_state.chat_history_by_thread[thread_id]
    history.append({"role": "user", "content": prompt_text})

    try:
        with st.status("Running Fleet Copilot...", expanded=False) as status_box:
            status_box.write("1. Planning and verifying tenant scope...")
            result: TurnResult = service.run_turn(
                thread_id=thread_id,
                company_id=tenant,
                user_message=prompt_text,
                actor_id=actor_id,
            )
            status_box.update(
                label=f"Completed ({len(result.tool_traces)} tool call(s))",
                state="complete",
                expanded=False,
            )
    except Exception as exc:
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"**Error executing agent turn:** `{exc}`\n\n"
                    "Verify your LLM provider key and model settings in `.env`."
                ),
            }
        )
        return

    content = result.answer
    if result.status == "awaiting_approval" and not content:
        content = (
            "Telemetry evidence has been verified and action proposal(s) have been staged. "
            "Review and submit your approval decision below."
        )

    history.append(
        {
            "role": "assistant",
            "content": content,
            "plan": result.plan,
            "intent": result.intent,
            "tool_traces": result.tool_traces,
            "evidence_store": result.evidence_store,
            "cited_evidence_ids": result.cited_evidence_ids,
            "blocked_proposals": result.blocked_proposals,
            "executed_actions": result.executed_actions,
        }
    )


# ---------------------------------------------------------------------------
# View 1: Split-Screen Copilot Workbench
# ---------------------------------------------------------------------------


def _render_hitl_approval_box(
    service: FleetCopilotService,
    tenant: str,
    actor_id: str,
    thread_id: str,
    pending_interrupt: dict[str, Any],
) -> None:
    proposals = pending_interrupt.get("proposals") or []
    if not proposals:
        return

    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.markdown("##### Administrator Approval Required")
            st.badge(
                f"{len(proposals)} pending",
                icon=":material/pause_circle:",
                color="orange",
            )
        st.caption(
            "Execution is paused at LangGraph's durable checkpoint. "
            "Review each proposed action and its verified telemetry evidence before resuming."
        )

        with st.form(key=f"hitl_form_{thread_id}_{len(proposals)}", border=False):
            decision_inputs: list[tuple[str, bool, str]] = []
            for idx, prop in enumerate(proposals):
                pid = prop["proposal_id"]
                action_title = _ACTION_LABELS.get(prop["action_type"], prop["action_type"])
                risk = str(prop.get("risk_level", "medium")).lower()
                risk_color = "red" if risk == "high" else ("orange" if risk == "medium" else "blue")
                target = prop.get("target_device_id") or prop.get("target_employee_id") or "-"

                if idx > 0:
                    st.divider()

                with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                    st.markdown(f"**{idx + 1}. {action_title}** · `{target}`")
                    st.badge(f"{risk.upper()} RISK", color=risk_color)

                st.caption(prop.get("rationale", ""))

                params_html = "".join(
                    f'<span class="kv-chip"><span>{html.escape(str(k))}</span>{html.escape(str(v))}</span>'
                    for k, v in (prop.get("parameters") or {}).items()
                )
                if params_html:
                    st.markdown(f'<div class="kv-row">{params_html}</div>', unsafe_allow_html=True)

                cits = prop.get("citations") or []
                if cits:
                    with st.expander(f"Supporting evidence ({len(cits)} citation(s))", expanded=False):
                        for c in cits:
                            st.markdown(f"- **`{c['evidence_id']}`** — {c['summary']}")

                c1, c2 = st.columns([1.2, 2.3])
                with c1:
                    choice = st.segmented_control(
                        f"Decision {pid}",
                        options=["Approve", "Reject"],
                        default="Reject",  # Fail-safe default
                        key=f"dec_{thread_id}_{pid}",
                        label_visibility="collapsed",
                    )
                with c2:
                    comment = st.text_input(
                        f"Note {pid}",
                        placeholder="Optional audit note...",
                        key=f"note_{thread_id}_{pid}",
                        label_visibility="collapsed",
                    )
                decision_inputs.append((pid, choice == "Approve", comment))

            submitted = st.form_submit_button(
                "Submit Decisions & Resume",
                type="primary",
                icon=":material/check_circle:",
                width="stretch",
            )
            if submitted:
                decisions = [
                    {
                        "proposal_id": pid,
                        "approved": approved,
                        "actor_id": actor_id,
                        "comment": comment,
                    }
                    for pid, approved, comment in decision_inputs
                ]
                with st.spinner("Resuming checkpoint and persisting action outcomes..."):
                    resumed = service.resume_approval(
                        thread_id=thread_id,
                        company_id=tenant,
                        decisions=decisions,
                    )
                st.session_state.chat_history_by_thread[thread_id].append(
                    {
                        "role": "assistant",
                        "content": resumed.answer,
                        "plan": resumed.plan,
                        "intent": resumed.intent,
                        "tool_traces": resumed.tool_traces,
                        "evidence_store": resumed.evidence_store,
                        "cited_evidence_ids": resumed.cited_evidence_ids,
                        "blocked_proposals": resumed.blocked_proposals,
                        "executed_actions": resumed.executed_actions,
                    }
                )
                st.rerun()


def _render_inspector_panel(
    service: FleetCopilotService,
    tenant: str,
    thread_id: str,
    history: list[dict[str, Any]],
) -> None:
    """Right-hand live Evidence, Reasoning Trace & Fleet Pulse Inspector."""
    assistant_entries = [
        (idx, m) for idx, m in enumerate(history) if m.get("role") == "assistant"
    ]

    selected_idx = st.session_state.selected_msg_by_thread.get(thread_id)
    valid_indices = [idx for idx, _ in assistant_entries]
    if selected_idx not in valid_indices:
        selected_idx = valid_indices[-1] if valid_indices else None

    target_assistant = history[selected_idx] if selected_idx is not None else None
    target_turn_num = (
        valid_indices.index(selected_idx) + 1 if selected_idx in valid_indices else None
    )

    has_trace = bool(
        target_assistant
        and (
            target_assistant.get("tool_traces")
            or target_assistant.get("cited_evidence_ids")
            or target_assistant.get("plan")
            or target_assistant.get("blocked_proposals")
            or target_assistant.get("executed_actions")
        )
    )

    ev_count = 0
    if target_assistant:
        traces = target_assistant.get("tool_traces") or []
        cited_ids = target_assistant.get("cited_evidence_ids") or []
        ev_count = len(
            dict.fromkeys(
                cited_ids or [eid for t in traces for eid in (t.get("evidence_ids") or [])]
            )
        )
    ev_tab_label = f"Sources & Tools ({ev_count})" if ev_count else "Sources & Tools"

    with st.container(border=True):
        mode = st.segmented_control(
            "Inspector mode",
            options=[ev_tab_label, "Fleet Pulse"],
            default=ev_tab_label if has_trace else "Fleet Pulse",
            key=f"inspector_mode_{thread_id}_{selected_idx}_{len(history)}",
            label_visibility="collapsed",
            width="stretch",
        )

        if mode == ev_tab_label:
            if not assistant_entries or not target_assistant:
                st.caption(
                    "Ask a question to inspect the agent's execution plan, tool calls, and verified telemetry citations here."
                )
                return

            # Clean contextual header showing which message is being inspected (no clunky dropdown)
            is_latest = selected_idx == valid_indices[-1]
            q_context = ""
            if (
                selected_idx is not None
                and selected_idx > 0
                and history[selected_idx - 1].get("role") == "user"
            ):
                raw_q = history[selected_idx - 1]["content"].strip()
                q_context = raw_q[:68] + ("..." if len(raw_q) > 68 else "")

            if q_context or not is_latest:
                with st.container(
                    horizontal=True,
                    horizontal_alignment="distribute",
                    vertical_alignment="center",
                ):
                    st.caption(q_context or f"Response #{target_turn_num}")
                    if not is_latest and st.button(
                        "Latest",
                        key=f"back_latest_{thread_id}_{selected_idx}",
                        icon=":material/arrow_downward:",
                        type="tertiary",
                    ):
                        st.session_state.selected_msg_by_thread[thread_id] = valid_indices[-1]
                        st.rerun()

            plan = target_assistant.get("plan") or []
            traces = target_assistant.get("tool_traces") or []
            ev_store = target_assistant.get("evidence_store") or {}
            cited_ids = target_assistant.get("cited_evidence_ids") or []
            show_ids = list(
                dict.fromkeys(
                    cited_ids or [eid for t in traces for eid in (t.get("evidence_ids") or [])]
                )
            )

            if target_assistant.get("intent"):
                st.badge(
                    target_assistant["intent"].replace("_", " "),
                    icon=":material/route:",
                    color="violet",
                )

            if not traces and not show_ids and not plan:
                st.caption("No tool calls or telemetry citations were used for this response.")

            if traces:
                st.markdown("**Tool Calls**")
                for tr in traces:
                    status = tr.get("status", "ok")
                    dot_cls = (
                        "dot-ok"
                        if status == "ok"
                        else ("dot-stage" if status == "staged_proposal" else "dot-err")
                    )
                    rows_txt = (
                        f"{tr.get('result_count')} rows"
                        if tr.get("result_count") is not None
                        else status
                    )
                    st.markdown(
                        f"""
                        <div class="trace-row">
                            <div class="trace-dot {dot_cls}"></div>
                            <div>
                                <div style="font-weight:600; color:#111827;"><code>{html.escape(str(tr.get('tool_name')))}</code></div>
                                <div style="color:#6B7280; font-size:0.74rem;">Step {tr.get('step')} · {html.escape(rows_txt)}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if show_ids:
                st.markdown(f"**Verified Citations ({len(show_ids)})**")
                with st.container(height=280, border=False):
                    for eid in show_ids:
                        cit = ev_store.get(eid)
                        if not cit:
                            continue
                        entity = cit.get("device_id") or cit.get("employee_id") or cit.get("company_id")
                        ts = cit.get("collected_at") or (
                            f"{str(cit.get('window_start', ''))[:10]}..{str(cit.get('window_end', ''))[:10]}"
                            if cit.get("window_start")
                            else "aggregate"
                        )
                        st.markdown(
                            f"""
                            <div class="ev-item">
                                <div class="ev-head">
                                    <span class="ev-tag">{html.escape(eid)}</span>
                                    <span class="ev-entity">{html.escape(str(entity))}</span>
                                </div>
                                <div class="ev-body">{html.escape(str(cit.get('summary', '')))}</div>
                                <div class="ev-foot">{html.escape(str(cit.get('source_tool', '')))} · {html.escape(str(ts))}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if plan:
                with st.expander("Execution Plan", expanded=False):
                    for step in plan:
                        st.caption(step)

        else:
            # Fleet Pulse live summary for active tenant
            repo = service.get_repository(tenant)
            insights = generate_fleet_insights(repo)
            high_fails = len(repo.query_compliance_failures(severity="high"))
            low_disk = len(repo.query_low_storage_devices(threshold_free_pct=15.0))
            ram_cnt = len(insights["ram_constraints"])
            batt_cnt = len(insights["battery_degradation"])
            drift_cnt = len(insights["compliance_drift"])

            rows = [
                ("High-severity compliance fails", high_fails, high_fails > 0, True),
                ("Devices < 15% free storage", low_disk, low_disk > 0, False),
                ("Sustained RAM pressure", ram_cnt, ram_cnt > 0, False),
                ("Degrading batteries (EOL)", batt_cnt, batt_cnt > 0, False),
                ("Compliance drift regressions", drift_cnt, drift_cnt > 0, False),
            ]
            for label, val, is_issue, is_crit in rows:
                cls = (
                    "pulse-val-crit"
                    if (is_issue and is_crit)
                    else ("pulse-val-warn" if is_issue else "pulse-val-ok")
                )
                st.markdown(
                    f'<div class="pulse-row"><span class="pulse-label">{label}</span>'
                    f'<span class="{cls}">{val}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.caption("Quick Investigations")
            quick = [
                ("Audit low disk devices", "Which devices are currently running low on disk space (< 15% free)?"),
                ("Check high-sev compliance", "Show me laptops failing high-severity compliance checks."),
                ("Summarize 30-day trends", "Analyze fleet health trends across battery, RAM, storage, and compliance drift."),
                ("Propose disk remediation", "Find devices under 10% free storage and open remediation tickets for them."),
            ]
            for idx, (label, prompt) in enumerate(quick):
                if st.button(
                    label,
                    key=f"pulse_q_{tenant}_{idx}",
                    type="tertiary",
                    icon=":material/arrow_forward:",
                    width="stretch",
                ):
                    st.session_state["queued_prompt"] = prompt
                    st.rerun()


def _render_copilot_view(
    service: FleetCopilotService,
    tenant: str,
    actor_id: str,
    thread_id: str,
    key_configured: bool,
    companies: list[str],
) -> None:
    overview = service.get_repository(tenant).get_fleet_overview()
    history = st.session_state.chat_history_by_thread.get(thread_id, [])

    pending_interrupt = None
    if key_configured:
        try:
            pending_interrupt = service.get_pending_interrupt(thread_id)
        except Exception:
            pending_interrupt = None

    # Crisp single-row page header
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.markdown(
            f'<div class="page-title">Fleet Copilot</div>'
            f'<div class="page-sub">Tenant-isolated reasoning over {overview["device_count"]} devices · {overview["snapshot_count"]} snapshots</div>',
            unsafe_allow_html=True,
        )
        with st.container(horizontal=True, gap="small"):
            st.badge(tenant, icon=":material/domain:", color="blue")
            if pending_interrupt:
                st.badge("Approval required", icon=":material/pause_circle:", color="orange")
            if key_configured:
                st.badge(service.settings.llm_model, icon=":material/check_circle:", color="green")
            else:
                st.badge("API key missing", icon=":material/key_off:", color="red")

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    # Split-Screen Workbench: 68% Conversation | 32% Live Inspector
    col_chat, col_insp = st.columns([2.15, 1.0], gap="medium")

    with col_chat:
        chat_box = st.container(height=530, border=True, autoscroll=True)
        with chat_box:
            if not history and not pending_interrupt:
                st.markdown(
                    f"""
                    <div class="empty-hero">
                        <h3>How can I help with <code>{html.escape(tenant)}</code> today?</h3>
                        <p>Query real-time fleet health, detect multi-day hardware or compliance trends,
                        or propose operational actions backed by verified telemetry citations.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                other_tenant = next((c for c in companies if c != tenant), "globex-002")
                starters = [
                    (
                        "Low disk space devices",
                        ":material/hard_drive:",
                        "Which devices are currently low on disk space (< 15% free)?",
                    ),
                    (
                        "High-severity compliance check",
                        ":material/verified_user:",
                        "Show me laptops failing high-severity compliance checks.",
                    ),
                    (
                        "30-day fleet health trends",
                        ":material/trending_up:",
                        "Analyze fleet health trends across battery, RAM, storage, and compliance drift.",
                    ),
                    (
                        "Propose storage remediation tickets",
                        ":material/build:",
                        "Find devices with critically low storage (< 10% free) and open remediation tickets for them.",
                    ),
                    (
                        "Devices older than macOS 15",
                        ":material/laptop_mac:",
                        "How many devices are running an OS older than macOS 15?",
                    ),
                    (
                        f"Test tenant guardrail ({other_tenant})",
                        ":material/lock:",
                        f"Show me all devices belonging to {other_tenant}.",
                    ),
                ]
                for row_i in (0, 2, 4):
                    c1, c2 = st.columns(2)
                    for offset, c in enumerate((c1, c2)):
                        lbl, icn, prm = starters[row_i + offset]
                        with c:
                            if st.button(
                                lbl,
                                key=f"starter_{thread_id}_{row_i + offset}",
                                icon=icn,
                                type="secondary",
                                width="stretch",
                            ):
                                st.session_state["queued_prompt"] = prm
                                st.rerun()
            else:
                selected_msg = st.session_state.selected_msg_by_thread.get(thread_id)
                asst_indices = [
                    i for i, m in enumerate(history) if m.get("role") == "assistant"
                ]
                effective_selected = (
                    selected_msg
                    if selected_msg in asst_indices
                    else (asst_indices[-1] if asst_indices else None)
                )

                for msg_idx, entry in enumerate(history):
                    avatar = ":material/person:" if entry["role"] == "user" else ":material/shield:"
                    with st.chat_message(entry["role"], avatar=avatar):
                        st.markdown(entry["content"])
                        if entry["role"] == "assistant":
                            traces = entry.get("tool_traces") or []
                            cits = entry.get("cited_evidence_ids") or []
                            plan = entry.get("plan") or []
                            blocked = entry.get("blocked_proposals") or []
                            has_inspectable = bool(traces or cits or plan or blocked)
                            if has_inspectable:
                                is_inspected = msg_idx == effective_selected
                                label = "Sources & tools"
                                icon = ":material/description:"

                                if st.button(
                                    label,
                                    key=f"src_btn_{thread_id}_{msg_idx}",
                                    icon=icon,
                                    type="secondary" if is_inspected else "tertiary",
                                ):
                                    st.session_state.selected_msg_by_thread[thread_id] = msg_idx
                                    st.rerun()

            if pending_interrupt:
                _render_hitl_approval_box(service, tenant, actor_id, thread_id, pending_interrupt)

        # Chat input cleanly anchored directly under the scrollable chat box
        if pending_interrupt:
            st.chat_input("Submit your approval decision above to continue...", disabled=True)
        else:
            queued = st.session_state.pop("queued_prompt", None)
            user_prompt = st.chat_input(
                f"Message Fleet Copilot ({tenant})..."
            )
            prompt_to_run = queued or user_prompt
            if prompt_to_run:
                _execute_turn(service, tenant, actor_id, thread_id, prompt_to_run)
                # Focus inspector on the newly added assistant response
                new_hist = st.session_state.chat_history_by_thread.get(thread_id, [])
                if new_hist:
                    st.session_state.selected_msg_by_thread[thread_id] = len(new_hist) - 1
                st.rerun()

    with col_insp:
        _render_inspector_panel(service, tenant, thread_id, history)


# ---------------------------------------------------------------------------
# View 2: Fleet Health & Time-Series Analytics
# ---------------------------------------------------------------------------


def _render_fleet_view(service: FleetCopilotService, tenant: str) -> None:
    repo = service.get_repository(tenant)
    insights = generate_fleet_insights(repo)
    ov = insights["overview"]
    devices = repo.list_latest_devices()
    high_fails = len(repo.query_compliance_failures(severity="high"))

    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.markdown(
            f'<div class="page-title">Fleet Health & Trends</div>'
            f'<div class="page-sub">{html.escape(tenant)} · Telemetry window {str(ov.get("earliest_snapshot") or "-")[:10]} to {str(ov.get("latest_snapshot") or "-")[:10]}</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Ask Copilot for Full Report",
            icon=":material/auto_awesome:",
            type="primary",
        ):
            _send_prompt_to_copilot(
                "Run a full fleet health report across battery, RAM, storage, compliance drift, and software anomalies, and recommend prioritized actions."
            )

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    # Clean 4-column native metric row
    m1, m2, m3, m4 = st.columns(4)
    with m1, st.container(border=True):
        st.metric("Managed Devices", ov["device_count"], f"{ov['employee_count']} employees")
    with m2, st.container(border=True):
        st.metric(
            "Low Disk (<15%)",
            len(insights["storage_trends"]),
            "depletion alerts" if insights["storage_trends"] else "all healthy",
            delta_color="inverse" if insights["storage_trends"] else "off",
        )
    with m3, st.container(border=True):
        st.metric(
            "RAM / Battery Alerts",
            len(insights["ram_constraints"]) + len(insights["battery_degradation"]),
            f"{len(insights['ram_constraints'])} RAM · {len(insights['battery_degradation'])} battery",
            delta_color="inverse"
            if (insights["ram_constraints"] or insights["battery_degradation"])
            else "off",
        )
    with m4, st.container(border=True):
        st.metric(
            "High-Sev Compliance",
            high_fails,
            "failing checks" if high_fails else "100% passing",
            delta_color="inverse" if high_fails else "off",
        )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Device Inventory Table with native ProgressColumn bars
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.markdown("##### Device Inventory (Latest Snapshot)")
            os_filter = st.segmented_control(
                "OS Filter",
                options=["All", "macOS", "Windows", "Ubuntu"],
                default="All",
                label_visibility="collapsed",
            )

        filtered_devs = [
            d
            for d in devices
            if (os_filter or "All") == "All"
            or (os_filter or "").lower() in (d["os_product_name"] or "").lower()
        ]
        if filtered_devs:
            inv_df = pd.DataFrame(
                [
                    {
                        "Device ID": d["device_id"],
                        "Employee": d["employee_id"],
                        "Model": d["model_name"],
                        "OS": f"{d['os_product_name']} {d['os_product_version']}",
                        "Disk Free %": float(d["min_disk_available_pct"] or 0.0),
                        "RAM Used %": float(d["memory_used_pct"] or 0.0),
                        "Battery": d["battery_condition"] or "Desktop",
                        "Failing Checks": ", ".join(
                            f"{c['check_id']} ({c['severity']})"
                            for c in d["failing_compliance_checks"]
                        )
                        or "Passing",
                    }
                    for d in filtered_devs
                ]
            )
            st.dataframe(
                inv_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Disk Free %": st.column_config.ProgressColumn(
                        "Disk Free %",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                    "RAM Used %": st.column_config.ProgressColumn(
                        "RAM Used %",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                },
            )
        else:
            st.caption("No devices match the selected OS filter.")

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    # Time-Series Device Drill-Down + Trend Findings
    col_ts, col_findings = st.columns([1.45, 1.0], gap="medium")

    with col_ts, st.container(border=True):
        st.markdown("##### Device Time-Series Telemetry")
        if devices:
            dev_map = {
                f"{d['device_id']} · {d['model_name']} ({d['os_product_name']} {d['os_product_version']})": d[
                    "device_id"
                ]
                for d in devices
            }
            chosen = st.selectbox(
                "Select device",
                options=list(dev_map.keys()),
                label_visibility="collapsed",
            )
            chosen_id = dev_map[chosen]
            hist = repo.get_device_history(chosen_id)
            if hist:
                ts_df = pd.DataFrame(
                    [
                        {
                            "Date": pd.to_datetime(r["collected_at"]),
                            "RAM Used %": r["memory_used_pct"],
                            "Disk Free %": r["disk_volumes"][0]["available_pct"]
                            if r["disk_volumes"]
                            else None,
                            "Battery (mAh)": r["battery_full_charge_capacity"],
                        }
                        for r in hist
                    ]
                )
                melted = ts_df.melt(
                    id_vars=["Date"],
                    value_vars=["RAM Used %", "Disk Free %"],
                    var_name="Metric",
                    value_name="Value",
                ).dropna()
                chart = (
                    alt.Chart(melted)
                    .mark_line(point=True, strokeWidth=2.2)
                    .encode(
                        x=alt.X("Date:T", title=None, axis=alt.Axis(grid=False)),
                        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 100]), title="%"),
                        color=alt.Color(
                            "Metric:N",
                            scale=alt.Scale(
                                domain=["RAM Used %", "Disk Free %"],
                                range=["#4F46E5", "#10B981"],
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=["Date:T", "Metric:N", "Value:Q"],
                    )
                    .properties(height=225)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, width="stretch")

                batt_df = ts_df[["Date", "Battery (mAh)"]].dropna()
                if not batt_df.empty:
                    b_chart = (
                        alt.Chart(batt_df)
                        .mark_area(
                            line={"color": "#D97706", "strokeWidth": 2},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[
                                    alt.GradientStop(color="rgba(217,119,6,0.22)", offset=0),
                                    alt.GradientStop(color="rgba(217,119,6,0.0)", offset=1),
                                ],
                                x1=1,
                                x2=1,
                                y1=0,
                                y2=1,
                            ),
                        )
                        .encode(
                            x=alt.X("Date:T", title=None, axis=alt.Axis(grid=False)),
                            y=alt.Y(
                                "Battery (mAh):Q",
                                scale=alt.Scale(zero=False),
                                title="Battery Capacity (mAh)",
                            ),
                            tooltip=["Date:T", "Battery (mAh):Q"],
                        )
                        .properties(height=155)
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(b_chart, width="stretch")
                else:
                    st.caption("No battery sensor data reported for this device.")

                if st.button(
                    f"Investigate {chosen_id} in Copilot",
                    icon=":material/arrow_forward:",
                    type="tertiary",
                ):
                    _send_prompt_to_copilot(
                        f"Inspect the full telemetry history of device {chosen_id}, summarize any risks, and recommend next steps."
                    )

    with col_findings, st.container(border=True):
        st.markdown("##### Detected Trend Signals")
        cat = st.segmented_control(
            "Signal category",
            options=["Battery", "RAM", "Storage", "Drift"],
            default="Storage",
            label_visibility="collapsed",
            width="stretch",
        )
        cat_map = {
            "Battery": insights["battery_degradation"],
            "RAM": insights["ram_constraints"],
            "Storage": insights["storage_trends"],
            "Drift": insights["compliance_drift"],
        }
        items = cat_map.get(cat or "Storage", [])
        if not items:
            st.caption(f"No {str(cat).lower()} anomalies detected for `{tenant}`.")
        else:
            with st.container(height=360, border=False):
                for item in items:
                    st.markdown(
                        f"""
                        <div class="ev-item">
                            <div class="ev-head">
                                <span class="ev-entity">{html.escape(item['device_id'])}</span>
                                <span style="font-size:0.74rem; color:#6B7280;">{html.escape(item.get('model_name', ''))}</span>
                            </div>
                            <div class="ev-body">{html.escape(item['finding'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# View 3: Operations & Audit Governance Console
# ---------------------------------------------------------------------------


def _render_operations_view(service: FleetCopilotService, tenant: str) -> None:
    actions = service.action_executor.list_actions(tenant, limit=100)
    events = service.audit.list_events(tenant, limit=150)

    exec_cnt = sum(1 for a in actions if a.status.value == "executed")
    rej_cnt = sum(1 for a in actions if a.status.value == "rejected")
    block_cnt = sum(
        1
        for e in events
        if e.event_type in ("CROSS_TENANT_BLOCKED", "ACTION_POLICY_BLOCKED", "SQL_QUERY_REJECTED")
    )

    st.markdown(
        f'<div class="page-title">Operations & Audit Log</div>'
        f'<div class="page-sub">Transactional ITSM mutations and immutable security audit trail for {html.escape(tenant)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(border=True):
        st.metric("Executed Actions", exec_cnt, "persisted & idempotent")
    with c2, st.container(border=True):
        st.metric("Rejected by Admin", rej_cnt, "human vetoed")
    with c3, st.container(border=True):
        st.metric(
            "Guardrail Blocks",
            block_cnt,
            "tenant / policy / SQL",
            delta_color="inverse" if block_cnt else "off",
        )
    with c4, st.container(border=True):
        st.metric("Audit Events", len(events), f"tenant {tenant}")

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    sub_view = st.segmented_control(
        "Governance view",
        options=["Operational Actions", "Security & Audit Log"],
        default="Operational Actions",
        label_visibility="collapsed",
    )

    with st.container(border=True):
        if sub_view == "Operational Actions":
            if not actions:
                st.caption("No operational actions have been executed or rejected for this tenant yet.")
            else:
                df = pd.DataFrame(
                    [
                        {
                            "Action ID": a.action_id,
                            "Action": _ACTION_LABELS.get(a.action_type.value, a.action_type.value),
                            "Target": a.target_device_id or a.target_employee_id or "-",
                            "Status": a.status.value.upper(),
                            "Reviewer": a.approved_by or "-",
                            "Evidence": ", ".join(a.evidence_ids),
                            "Summary": " · ".join(f"{k}: {v}" for k, v in a.parameters.items()),
                            "Dispatch": "Webhook + DB" if a.external_dispatch else "Local ITSM DB",
                            "Time (UTC)": a.created_at[:19].replace("T", " "),
                        }
                        for a in actions
                    ]
                )
                st.dataframe(df, width="stretch", hide_index=True)
        else:
            event_types = sorted({e.event_type for e in events})
            chosen_types = st.multiselect(
                "Filter event types",
                options=event_types,
                default=[],
                placeholder="All audit event types",
                label_visibility="collapsed",
            )
            filtered = [e for e in events if not chosen_types or e.event_type in chosen_types]
            if not filtered:
                st.caption("No audit events match the current filter.")
            else:
                ev_df = pd.DataFrame(
                    [
                        {
                            "Time (UTC)": e.timestamp[:19].replace("T", " "),
                            "Event Type": e.event_type,
                            "Status": e.status,
                            "Actor": e.actor_id,
                            "Target": e.target_entity or "-",
                            "Thread": e.thread_id,
                            "Details": json.dumps(e.details),
                        }
                        for e in filtered
                    ]
                )
                st.dataframe(ev_df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# View 4: Live Telemetry Ingestion Studio
# ---------------------------------------------------------------------------


def _render_ingestion_view(service: FleetCopilotService, tenant: str, actor_id: str) -> None:
    st.markdown(
        f'<div class="page-title">Live Telemetry Ingestion</div>'
        f'<div class="page-sub">Stream single snapshots or upload batch NDJSON into {html.escape(tenant)} with zero downtime</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.25, 1.0], gap="medium")

    with col_left, st.container(border=True):
        st.markdown("##### Realtime Snapshot Simulator")
        st.caption("Simulate an incoming endpoint telemetry event including custom/unseen compliance checks and JSON attributes.")

        with st.form("realtime_sim_form", border=False):
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                dev_id = st.text_input("Device ID", value="LIVE-DEV-001")
                model_name = st.selectbox(
                    "Model",
                    ["MacBook Pro", "MacBook Air", "ThinkPad X1 Carbon", "Dell XPS 15", "Framework Laptop 16", "Mac mini"],
                )
                ram_pct = st.slider("RAM Used (%)", 10, 99, 88)
            with r1c2:
                emp_id = st.text_input("Employee ID", value=f"emp-{tenant}-999")
                os_choice = st.selectbox(
                    "Operating System",
                    [
                        ("darwin", "macOS", "15.4"),
                        ("darwin", "macOS", "14.5"),
                        ("win32", "Windows", "11 23H2"),
                        ("linux", "Ubuntu", "24.04"),
                    ],
                    format_func=lambda x: f"{x[1]} {x[2]}",
                )
                disk_pct = st.slider("Disk Free (%)", 1, 95, 4)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                batt_cond = st.selectbox(
                    "Battery Condition",
                    ["Normal", "Service Recommended", "Desktop (No Battery)"],
                )
                custom_region = st.text_input("Custom attribute: office_region", value="eu-west-1")
            with r2c2:
                st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
                fail_enc = st.checkbox("Fail disk_encryption (high)", value=True)
                fail_edr = st.checkbox("Fail edr_sensor_active (high, unseen check)", value=True)

            if st.form_submit_button(
                "Ingest Live Snapshot",
                type="primary",
                icon=":material/bolt:",
                width="stretch",
            ):
                now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                total_ram = 34_359_738_368
                used_ram = int(total_ram * (ram_pct / 100.0))
                total_disk = 549_755_813_888
                avail_disk = int(total_disk * (disk_pct / 100.0))
                payload: dict[str, Any] = {
                    "device_id": dev_id.strip(),
                    "company_id": tenant,
                    "employee_id": emp_id.strip(),
                    "collected_at": now_iso,
                    "agent_version": "44.0.0",
                    "os": {
                        "platform": os_choice[0],
                        "product_name": os_choice[1],
                        "product_version": os_choice[2],
                        "hostname": f"{tenant}-{dev_id.strip().lower()}",
                    },
                    "device_identity": {
                        "serial_number": dev_id.strip(),
                        "model_name": model_name,
                        "total_memory": "32 GB",
                    },
                    "memory": {
                        "total_memory_bytes": total_ram,
                        "used_memory_bytes": used_ram,
                        "free_memory_bytes": total_ram - used_ram,
                    },
                    "disk_volumes": [
                        {
                            "volume_name": "Macintosh HD" if os_choice[0] == "darwin" else "OS",
                            "mount_point": "/",
                            "size_bytes": total_disk,
                            "available_bytes": avail_disk,
                            "encrypted": not fail_enc,
                        }
                    ],
                    "installed_software": [
                        {"name": "Google Chrome", "version": "130.0", "publisher": "Google LLC"},
                        {"name": "Slack", "version": "4.40", "publisher": "Slack Technologies"},
                    ],
                    "compliance_results": [
                        {
                            "check_id": "disk_encryption",
                            "status": "fail" if fail_enc else "pass",
                            "severity": "high",
                        },
                        {
                            "check_id": "edr_sensor_active",
                            "status": "fail" if fail_edr else "pass",
                            "severity": "high",
                        },
                    ],
                    "custom_telemetry_ext": {"office_region": custom_region.strip()},
                }
                if batt_cond != "Desktop (No Battery)":
                    payload["battery"] = {
                        "battery_present": True,
                        "charging_status": "discharging",
                        "percentage": 54,
                        "condition": batt_cond,
                        "cycle_count": 980 if batt_cond == "Service Recommended" else 48,
                        "full_charge_capacity": 3600 if batt_cond == "Service Recommended" else 6100,
                    }

                report = service.ingest_text(json.dumps(payload), actor_id=actor_id)
                if report.errors:
                    st.error("\n".join(report.errors))
                else:
                    st.success(
                        f"Ingested `{dev_id}` into `{tenant}` (inserted={report.inserted}, updated={report.updated})."
                    )
                    st.rerun()

    with col_right, st.container(border=True):
        st.markdown("##### Batch File or Raw JSON")
        st.caption("Supports NDJSON streams, single pretty-printed JSON objects, or JSON arrays.")

        uploaded = st.file_uploader(
            "Drop `.ndjson` or `.json` file",
            type=["ndjson", "json", "jsonl"],
            label_visibility="collapsed",
        )
        if uploaded is not None and st.button(
            "Ingest Uploaded File",
            icon=":material/upload:",
            type="primary",
            width="stretch",
        ):
            report = service.ingest_text(uploaded.getvalue().decode("utf-8"), actor_id=actor_id)
            st.success(
                f"Processed {report.processed} snapshot(s) "
                f"(inserted={report.inserted}, updated={report.updated}, errors={len(report.errors)})."
            )
            if report.errors:
                st.error("\n".join(report.errors[:8]))
            st.rerun()

        st.divider()
        raw_text = st.text_area(
            "Paste JSON or NDJSON",
            placeholder='{"device_id": "DEV-01", "company_id": "acme-001", ...}',
            height=190,
        )
        if st.button("Ingest Raw Payload", icon=":material/send:", width="stretch") and raw_text.strip():
            report = service.ingest_text(raw_text, actor_id=actor_id)
            if report.errors:
                st.error("\n".join(report.errors[:8]))
            else:
                st.success(
                    f"Processed {report.processed} snapshot(s) "
                    f"(inserted={report.inserted}, updated={report.updated})."
                )
                st.rerun()


# ---------------------------------------------------------------------------
# Main Application Shell (Left Rail + Active View)
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    service = get_service()
    companies = service.list_companies()
    _ensure_session_state(service, companies)

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-row">
                <div class="brand-mark">R</div>
                <div>
                    <div class="brand-name">Rayda</div>
                    <div class="brand-tagline">Fleet Copilot</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tenant Scope Selector
        st.caption("Tenant Scope")
        if companies:
            selected_tenant = st.selectbox(
                "Tenant",
                options=companies,
                index=companies.index(st.session_state.active_tenant)
                if st.session_state.active_tenant in companies
                else 0,
                label_visibility="collapsed",
            )
        else:
            selected_tenant = st.text_input("Tenant", value="acme-001", label_visibility="collapsed")

        if selected_tenant != st.session_state.active_tenant:
            st.session_state.active_tenant = selected_tenant
            if selected_tenant not in st.session_state.threads_by_tenant:
                persisted_for_tenant = service.list_threads(selected_tenant)
                st.session_state.threads_by_tenant[selected_tenant] = (
                    persisted_for_tenant[0]["thread_id"]
                    if persisted_for_tenant
                    else f"{selected_tenant}-{uuid.uuid4().hex[:8]}"
                )
            tid = st.session_state.threads_by_tenant[selected_tenant]
            st.session_state.session_thread_ids_by_tenant.setdefault(selected_tenant, [])
            if tid not in st.session_state.session_thread_ids_by_tenant[selected_tenant]:
                st.session_state.session_thread_ids_by_tenant[selected_tenant].insert(0, tid)
            _hydrate_thread_if_needed(service, tid, selected_tenant)
            st.rerun()

        tenant = st.session_state.active_tenant
        thread_id = st.session_state.threads_by_tenant[tenant]

        st.divider()

        # Primary Workspace Navigation
        st.caption("Workspace")
        for view_key, label, icon in _NAV_ITEMS:
            is_curr = st.session_state.active_view == view_key
            if st.button(
                label,
                key=f"nav_{view_key}",
                icon=icon,
                type="secondary" if is_curr else "tertiary",
                width="stretch",
            ) and not is_curr:
                st.session_state.active_view = view_key
                st.rerun()

        st.divider()

        # Conversations Section
        st.caption("Conversations")
        if st.button(
            "New conversation",
            icon=":material/add:",
            type="primary",
            width="stretch",
        ):
            new_tid = f"{tenant}-{uuid.uuid4().hex[:8]}"
            st.session_state.threads_by_tenant[tenant] = new_tid
            st.session_state.session_thread_ids_by_tenant.setdefault(tenant, []).insert(0, new_tid)
            st.session_state.chat_history_by_thread[new_tid] = []
            st.session_state.active_view = "copilot"
            st.rerun()

        persisted_threads = service.list_threads(tenant)
        persisted_by_id = {t["thread_id"]: t for t in persisted_threads}
        session_tids = st.session_state.session_thread_ids_by_tenant.setdefault(tenant, [])

        ordered_tids: list[str] = []
        for candidate in [thread_id, *session_tids, *persisted_by_id.keys()]:
            if candidate not in ordered_tids:
                ordered_tids.append(candidate)

        for tid_item in ordered_tids[:10]:
            meta = persisted_by_id.get(tid_item)
            if meta:
                icon = (
                    ":material/pause_circle:"
                    if meta.get("awaiting_approval")
                    else ":material/chat_bubble_outline:"
                )
                label = meta["title"][:34] + ("..." if len(meta["title"]) > 34 else "")
            else:
                local_msgs = st.session_state.chat_history_by_thread.get(tid_item) or []
                first_u = next((m["content"] for m in local_msgs if m.get("role") == "user"), None)
                icon = ":material/chat_bubble_outline:"
                label = (
                    (first_u[:34] + ("..." if len(first_u) > 34 else ""))
                    if first_u
                    else f"Untitled ({tid_item[-6:]})"
                )

            is_active = tid_item == thread_id and st.session_state.active_view == "copilot"
            if st.button(
                label,
                key=f"th_{tenant}_{tid_item}",
                icon=icon,
                type="secondary" if is_active else "tertiary",
                width="stretch",
            ):
                st.session_state.threads_by_tenant[tenant] = tid_item
                st.session_state.active_view = "copilot"
                _hydrate_thread_if_needed(service, tid_item, tenant)
                st.rerun()

        # Subtle Runtime & Actor Settings Popover
        st.divider()
        settings = service.settings
        key_configured = bool(
            (settings.llm_provider == "gemini" and settings.google_api_key)
            or (settings.llm_provider == "openai" and settings.openai_api_key)
            or (settings.llm_provider == "anthropic" and settings.anthropic_api_key)
        )
        with st.popover("Runtime & Admin Settings", icon=":material/tune:", width="stretch"):
            st.session_state.actor_id = st.text_input(
                "Admin Actor ID", value=st.session_state.actor_id
            )
            st.caption(f"Provider: `{settings.llm_provider}`")
            st.caption(f"Model: `{settings.llm_model}`")
            if settings.llm_fallback_models:
                st.caption(f"Fallbacks: `{settings.llm_fallback_models}`")
            st.caption(f"Action Backend: `{settings.action_webhook_url or 'Local SQLite ITSM'}`")
            st.caption(f"Active Thread: `{thread_id}`")

    actor_id = st.session_state.actor_id
    active_view = st.session_state.active_view

    if active_view == "copilot":
        _render_copilot_view(service, tenant, actor_id, thread_id, key_configured, companies)
    elif active_view == "fleet":
        _render_fleet_view(service, tenant)
    elif active_view == "operations":
        _render_operations_view(service, tenant)
    elif active_view == "ingestion":
        _render_ingestion_view(service, tenant, actor_id)


if __name__ == "__main__":
    main()
