"""Explicit LangGraph execution graph for the Agentic AI Fleet Copilot.

Architecture:
    planner -> reasoner <-> tool_executor -> policy_gate -> hitl_approval (interrupt) -> action_executor -> synthesizer
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from fleet_copilot.actions.audit import AuditLogger
from fleet_copilot.actions.backend import ActionExecutor, build_idempotency_key
from fleet_copilot.actions.policy import ActionPolicyValidator
from fleet_copilot.agent.state import FleetAgentState
from fleet_copilot.agent.tools import ACTION_TOOL_NAMES, build_tenant_tools
from fleet_copilot.models.actions import ActionProposal, ActionType, HumanDecision
from fleet_copilot.models.evidence import Citation, ToolTrace
from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.repository import TenantTelemetryRepository

_RISK_BY_ACTION = {
    ActionType.CREATE_UPGRADE_ORDER: "high",
    ActionType.FLAG_DEVICE_FOR_REPLACEMENT: "high",
    ActionType.OPEN_REMEDIATION_TICKET: "medium",
    ActionType.NOTIFY_EMPLOYEE: "low",
}

_CITATION_TAG_RE = re.compile(r"\b(E\d+)\b")
_BRACKET_CITATION_RE = re.compile(r"\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")


def _sanitize_inline_citations(text: str, evidence_store: dict[str, Any]) -> str:
    """Deterministically strip fabricated [E#] citation tags not present in evidence_store."""
    if not text:
        return text

    def _replace_group(match: re.Match[str]) -> str:
        raw_ids = [part.strip() for part in match.group(1).split(",")]
        valid = [eid for eid in raw_ids if eid in evidence_store]
        if not valid:
            return ""
        return "[" + ", ".join(valid) + "]"

    cleaned = _BRACKET_CITATION_RE.sub(_replace_group, text)
    # Clean up double spaces or space before punctuation left by stripped tags
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def extract_text_content(content: Any) -> str:
    """Normalize LangChain message content (str or list of Gemini content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content or "")


def _detect_explicit_cross_tenant_request(
    user_text: str, active_company: str, db: FleetDatabase
) -> str | None:
    """Dynamically check whether user text explicitly targets another tenant in the DB.

    Company names/IDs are discovered dynamically from live telemetry—never hardcoded.
    """
    lowered = user_text.lower()
    companies = db.list_companies()

    for cid in companies:
        if cid.lower() == active_company.lower():
            continue
        # Match full company_id (e.g. 'globex-002') or alphabetic slug prefix (e.g. 'globex')
        slug = re.split(r"[-_\d]+", cid.lower())[0]
        if re.search(rf"\b{re.escape(cid.lower())}\b", lowered):
            return (
                f"Access denied: your session is scoped to tenant '{active_company}'. "
                f"Cross-company access to '{cid}' is prohibited by tenant isolation controls."
            )
        if len(slug) >= 4 and re.search(rf"\b{re.escape(slug)}\b", lowered):
            return (
                f"Access denied: your session is scoped to tenant '{active_company}'. "
                f"Cross-company access to another organization ('{slug}') is prohibited."
            )

    # Also detect explicit cross-tenant device_id or employee_id references
    candidate_ids = re.findall(r"\b(?:emp-[a-z0-9-]+|[A-Z0-9]{10,16})\b", user_text)
    for token in candidate_ids:
        if token.lower().startswith("emp-"):
            owner = db.get_entity_owner(employee_id=token)
            if owner and owner != active_company:
                return (
                    f"Access denied: employee '{token}' does not belong to tenant '{active_company}'. "
                    "Cross-company data access is prohibited."
                )
        else:
            owner = db.get_entity_owner(device_id=token)
            if owner and owner != active_company:
                return (
                    f"Access denied: device '{token}' does not belong to tenant '{active_company}'. "
                    "Cross-company data access is prohibited."
                )

    return None


class ExecutionPlan(BaseModel):
    """Structured multi-step execution plan generated by the LLM planner node."""

    intent: Literal[
        "grounded_qa",
        "trend_and_insight_analysis",
        "investigate_and_propose_action",
        "ad_hoc_telemetry_query",
    ] = Field(description="Primary operational intent of the administrator's request.")
    steps: list[str] = Field(
        min_length=2,
        max_length=5,
        description="2 to 5 concise, query-specific numbered execution steps tailored to this request.",
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Tool names from the catalog expected to be invoked.",
    )


def _build_heuristic_plan(user_text: str) -> tuple[str, list[str]]:
    """Deterministic fallback plan used for offline scripted tests or planner failover."""
    lowered = user_text.lower()
    wants_action = any(
        kw in lowered
        for kw in (
            "ticket",
            "remediat",
            "upgrade",
            "order",
            "replace",
            "flag",
            "notify",
            "email",
            "message",
            "propose action",
            "take action",
            "fix",
        )
    )
    wants_trends = any(
        kw in lowered
        for kw in (
            "trend",
            "drift",
            "over time",
            "consistent",
            "degrad",
            "end-of-life",
            "end of life",
            "eol",
            "insight",
            "health",
            "summary",
            "attention",
            "anomal",
            "predict",
            "history",
            "past",
        )
    )

    if wants_action:
        intent = "investigate_and_propose_action"
        plan = [
            "1. Query tenant-scoped telemetry and/or time-series trend tools to gather verifiable evidence.",
            "2. Verify target devices/employees and confirm concrete evidence supports each candidate action.",
            "3. Stage justified action proposal(s) with exact evidence_ids (never execute without approval).",
            "4. Run deterministic Policy & Evidence Gate, then pause for explicit human approval before execution.",
            "5. Synthesize a grounded final response citing all telemetry records and action outcomes.",
        ]
    elif wants_trends:
        intent = "trend_and_insight_analysis"
        plan = [
            "1. Execute deterministic time-series trend/anomaly detectors over tenant telemetry.",
            "2. Cross-check specific devices or thresholds if additional detail is needed.",
            "3. Synthesize findings with supporting evidence citations and brief operational explanations.",
        ]
    else:
        intent = "grounded_qa"
        plan = [
            "1. Select the appropriate tenant-scoped query tool(s) (or tenant-isolated read-only SQL for ad-hoc fields).",
            "2. Verify exact counts and device/snapshot metrics (explicitly report zero if no matches exist).",
            "3. Synthesize a concise, evidence-grounded answer citing exact telemetry records.",
        ]
    return intent, plan


def _generate_execution_plan(
    *,
    user_text: str,
    company_id: str,
    tool_names: list[str],
    llm: BaseChatModel,
    fallbacks: list[BaseChatModel],
) -> tuple[str, list[str]]:
    """Generate a query-specific structured plan via LLM, with graceful deterministic fallback."""
    if getattr(llm, "_llm_type", "") == "scripted-test-chat-model" or not user_text.strip():
        return _build_heuristic_plan(user_text)

    planner_prompt = (
        f"You are the planning module for Rayda Fleet Copilot (tenant `{company_id}`).\n"
        f"Available tools: {', '.join(tool_names)}.\n"
        "Create a concise, task-specific 2-to-5 step execution plan for the administrator's request.\n"
        "Rules:\n"
        "- Always gather verifiable telemetry evidence BEFORE staging any state-changing action.\n"
        "- Use `query_telemetry_sql` when the user asks about custom JSON fields (in `raw_payload`) "
        "or telemetry fields not covered by specialized query tools.\n"
        "- For action requests, explicitly include policy verification and human approval before execution.\n"
        "- Keep each step crisp (1 short sentence, numbered `1. ...`, `2. ...`)."
    )

    try:
        structured = llm.with_structured_output(ExecutionPlan)
        if fallbacks:
            structured = structured.with_fallbacks(
                [fb.with_structured_output(ExecutionPlan) for fb in fallbacks]
            )
        result = structured.invoke(
            [
                SystemMessage(content=planner_prompt),
                HumanMessage(content=user_text),
            ]
        )
        if isinstance(result, ExecutionPlan) and result.steps:
            steps = [
                s if re.match(r"^\d+\.\s", s.strip()) else f"{idx}. {s.strip()}"
                for idx, s in enumerate(result.steps, start=1)
            ]
            if result.suggested_tools:
                steps.append(f"Planned tools: {', '.join(result.suggested_tools)}")
            return result.intent, steps
    except Exception:
        # Resilient fallback: never block a turn if structured planning hits a transient error
        pass

    return _build_heuristic_plan(user_text)


def _build_system_prompt(
    *,
    company_id: str,
    plan: list[str],
    existing_evidence: dict[str, dict[str, Any]],
) -> str:
    plan_block = "\n".join(plan)
    ev_lines: list[str] = []
    if existing_evidence:
        # Provide compact summary of previously gathered verified evidence in this thread
        for eid, c in list(existing_evidence.items())[-30:]:
            ev_lines.append(f"- {eid}: {c.get('summary', '')}")
    ev_block = "\n".join(ev_lines) if ev_lines else "(none yet in this conversation)"

    return f"""You are Rayda's Agentic AI Fleet Copilot assisting an IT administrator for tenant `{company_id}`.

ACTIVE EXECUTION PLAN:
{plan_block}

STRICT GROUNDING & SAFETY RULES (NON-NEGOTIABLE):
1. TENANT SCOPE: You are strictly scoped to `{company_id}`. Never claim access to or speculate about other companies.
2. EVIDENCE ONLY: Answer exclusively from telemetry tool outputs retrieved in this session. Never invent device IDs, employee IDs, dates, metrics, or compliance failures.
3. ZERO-RESULT HONESTY: If a query returns 0 matching records (for example, 0 high-severity compliance failures), state clearly that 0 devices matched. Never fabricate failures or mislabel medium-severity failures as high-severity. You may separately note other findings (e.g. medium-severity failures) ONLY if you also retrieved them and clearly distinguish their severity.
4. MANDATORY CITATIONS: Cite verified evidence IDs inline (e.g. `[E1]`, `[E2]`) for every factual claim and device finding.
5. TREND INSIGHTS FORMAT: When reporting insights or trends, include for each finding: (a) The finding, (b) Supporting evidence with `[E#]` citations and exact metrics/dates, and (c) A brief operational explanation.
6. GENERALIZATION / AD-HOC FIELDS: For any telemetry question not covered by a specialized tool (e.g., network interfaces, MAC addresses, agent versions, kernel releases, build versions, hardware UUIDs, or custom JSON attributes in `raw_payload`), use `query_telemetry_sql` against `v_snapshots`, `v_disk_volumes`, `v_compliance_results`, `v_installed_software`, or `v_network_interfaces`.
7. GUARDED ACTIONS:
   - Whenever the administrator asks you to propose, prepare, recommend-and-stage, or take operational actions (upgrade orders, remediation tickets, replacement flags, employee notifications), retrieve supporting telemetry first and then invoke the appropriate action tool(s) so proposals flow through the Policy Gate and Human Approval workflow.
   - Never call an action tool (`create_upgrade_order`, `open_remediation_ticket`, `flag_device_for_replacement`, `notify_employee`) without first retrieving telemetry evidence proving the issue exists for that exact target device/employee.
   - If the user asks you to take an action on a healthy device or without supporting telemetry evidence, gather telemetry first; if the telemetry does NOT support the action, REFUSE the action clearly and explain what the telemetry actually shows.
   - Always pass the exact verified `evidence_ids` (e.g. `["E1"]`) returned by read/trend tools when calling an action tool.
   - When an action tool returns `staged_for_policy_and_approval`, DO NOT claim the action has been executed. Briefly present your grounded findings and state that the proposed action(s) have been queued for policy verification and administrator approval, then finish your turn.

PREVIOUSLY VERIFIED EVIDENCE IN THIS SESSION:
{ev_block}
"""


def _build_proposal_from_tool_call(
    tool_name: str,
    args: dict[str, Any],
    company_id: str,
) -> ActionProposal:
    action_type = ActionType(tool_name)
    proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"

    target_device_id = args.get("device_id")
    target_employee_id = args.get("employee_id")
    rationale = str(args.get("rationale") or "").strip()
    raw_eids = args.get("evidence_ids") or []
    evidence_ids = [str(x).strip() for x in raw_eids if str(x).strip()]

    params = {
        k: v
        for k, v in args.items()
        if k not in ("device_id", "employee_id", "rationale", "evidence_ids")
    }
    if target_device_id is not None:
        params["device_id"] = target_device_id
    if target_employee_id is not None:
        params["employee_id"] = target_employee_id

    proposal = ActionProposal(
        proposal_id=proposal_id,
        action_type=action_type,
        company_id=company_id,
        target_device_id=target_device_id,
        target_employee_id=target_employee_id,
        parameters=params,
        rationale=rationale,
        evidence_ids=evidence_ids,
        risk_level=_RISK_BY_ACTION[action_type],
    )
    proposal.idempotency_key = build_idempotency_key(proposal)
    return proposal


def build_fleet_copilot_graph(
    *,
    db: FleetDatabase,
    audit_logger: AuditLogger,
    action_executor: ActionExecutor,
    llm: BaseChatModel,
    fallback_llms: list[BaseChatModel] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    enable_llm_planner: bool = True,
    max_steps: int = 8,
    max_sql_rows: int = 100,
) -> Any:
    """Compile the explicit LangGraph StateGraph for the Fleet Copilot."""
    fallbacks = list(fallback_llms or [])

    def planner_node(state: FleetAgentState) -> dict[str, Any]:
        company_id = state["company_id"]
        actor_id = state.get("actor_id", "it-admin")
        thread_id = state.get("thread_id", "default")

        # Find latest user message
        last_user = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                last_user = extract_text_content(msg.content)
                break

        cross_tenant_msg = _detect_explicit_cross_tenant_request(last_user, company_id, db)
        if cross_tenant_msg:
            audit_logger.record(
                thread_id=thread_id,
                company_id=company_id,
                actor_id=actor_id,
                event_type="CROSS_TENANT_BLOCKED",
                status="denied",
                details={"stage": "planner", "query": last_user[:300], "reason": cross_tenant_msg},
            )
            return {
                "intent": "cross_tenant_denied",
                "plan": ["1. Detect cross-tenant request boundary violation.", "2. Refuse request and log audit event."],
                "cross_tenant_blocked": True,
                "cross_tenant_reason": cross_tenant_msg,
                "step_count": 0,
                "tool_traces": [],
                "candidate_proposals": [],
                "validated_proposals": [],
                "blocked_proposals": [],
                "human_decisions": [],
                "executed_actions": [],
                "draft_answer": cross_tenant_msg,
            }

        repo = TenantTelemetryRepository(db, company_id)
        tools, _ = build_tenant_tools(repo, max_sql_rows=max_sql_rows)
        if enable_llm_planner:
            intent, plan = _generate_execution_plan(
                user_text=last_user,
                company_id=company_id,
                tool_names=[t.name for t in tools],
                llm=llm,
                fallbacks=fallbacks,
            )
        else:
            intent, plan = _build_heuristic_plan(last_user)
        return {
            "intent": intent,
            "plan": plan,
            "cross_tenant_blocked": False,
            "cross_tenant_reason": "",
            "step_count": 0,
            "evidence_counter": state.get("evidence_counter", 0),
            "evidence_store": dict(state.get("evidence_store") or {}),
            "tool_traces": [],
            "candidate_proposals": [],
            "validated_proposals": [],
            "blocked_proposals": [],
            "human_decisions": [],
            "executed_actions": [],
            "draft_answer": "",
            "final_response": "",
            "cited_evidence_ids": [],
        }

    def route_after_planner(state: FleetAgentState) -> str:
        if state.get("cross_tenant_blocked"):
            return "synthesizer"
        return "reasoner"

    def reasoner_node(state: FleetAgentState) -> dict[str, Any]:
        company_id = state["company_id"]
        repo = TenantTelemetryRepository(db, company_id)
        tools, _ = build_tenant_tools(repo, max_sql_rows=max_sql_rows)

        sys_prompt = _build_system_prompt(
            company_id=company_id,
            plan=state.get("plan", []),
            existing_evidence=state.get("evidence_store", {}),
        )
        conversation = [SystemMessage(content=sys_prompt), *state.get("messages", [])]

        step_count = state.get("step_count", 0)
        if step_count >= max_steps:
            # Hard step bound reached: force grounded finalization without more tool calls
            runnable = llm.with_fallbacks(fallbacks) if fallbacks else llm
            bound_msg = runnable.invoke(
                conversation
                + [
                    SystemMessage(
                        content="Maximum tool execution steps reached. Synthesize your grounded answer now using only the evidence already collected."
                    )
                ]
            )
            return {
                "messages": [bound_msg],
                "draft_answer": extract_text_content(bound_msg.content),
            }

        model_with_tools = llm.bind_tools(tools)
        if fallbacks:
            model_with_tools = model_with_tools.with_fallbacks(
                [fb.bind_tools(tools) for fb in fallbacks]
            )
        response = model_with_tools.invoke(conversation)
        draft = extract_text_content(response.content)
        return {
            "messages": [response],
            "draft_answer": draft if draft else state.get("draft_answer", ""),
        }

    def route_after_reasoner(state: FleetAgentState) -> str:
        messages = state.get("messages", [])
        if (
            messages
            and isinstance(messages[-1], AIMessage)
            and messages[-1].tool_calls
            and state.get("step_count", 0) < max_steps
        ):
            return "tool_executor"
        if state.get("candidate_proposals"):
            return "policy_gate"
        return "synthesizer"

    def tool_executor_node(state: FleetAgentState) -> dict[str, Any]:
        company_id = state["company_id"]
        actor_id = state.get("actor_id", "it-admin")
        thread_id = state.get("thread_id", "default")
        repo = TenantTelemetryRepository(db, company_id)
        _, callable_map = build_tenant_tools(repo, max_sql_rows=max_sql_rows)

        last_msg = state["messages"][-1]
        assert isinstance(last_msg, AIMessage)

        evidence_store = dict(state.get("evidence_store") or {})
        counter = int(state.get("evidence_counter", 0))
        traces = list(state.get("tool_traces") or [])
        proposals = list(state.get("candidate_proposals") or [])
        step = int(state.get("step_count", 0)) + 1
        tool_messages: list[ToolMessage] = []

        for tc in last_msg.tool_calls:
            tname = tc["name"]
            targs = tc.get("args") or {}
            tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"

            # Intercept state-changing action tools: stage proposal, NEVER execute directly
            if tname in ACTION_TOOL_NAMES:
                proposal = _build_proposal_from_tool_call(tname, targs, company_id)
                proposals.append(proposal.model_dump(mode="json"))
                traces.append(
                    ToolTrace(
                        step=step,
                        tool_name=tname,
                        arguments=targs,
                        result_count=1,
                        evidence_ids=proposal.evidence_ids,
                        status="staged_proposal",
                        summary=f"Staged {tname} ({proposal.proposal_id}) for Policy Gate & HITL approval.",
                    ).model_dump(mode="json")
                )
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tc_id,
                        name=tname,
                        content=json.dumps(
                            {
                                "status": "staged_for_policy_and_approval",
                                "proposal_id": proposal.proposal_id,
                                "action_type": tname,
                                "message": (
                                    "Action has been staged for deterministic Policy Gate validation and "
                                    "explicit IT administrator approval. It has NOT been executed yet."
                                ),
                            }
                        ),
                    )
                )
                continue

            fn = callable_map.get(tname)
            if fn is None:
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tc_id,
                        name=tname,
                        content=json.dumps({"status": "error", "error": f"Unknown tool '{tname}'"}),
                    )
                )
                continue

            try:
                raw_output = fn(**targs)
            except Exception as exc:
                traces.append(
                    ToolTrace(
                        step=step,
                        tool_name=tname,
                        arguments=targs,
                        status="error",
                        summary=str(exc),
                    ).model_dump(mode="json")
                )
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tc_id,
                        name=tname,
                        content=json.dumps({"status": "error", "error": str(exc)}),
                    )
                )
                continue

            status = raw_output.get("status", "ok")
            if status == "denied_cross_tenant":
                audit_logger.record(
                    thread_id=thread_id,
                    company_id=company_id,
                    actor_id=actor_id,
                    event_type="CROSS_TENANT_BLOCKED",
                    status="denied",
                    details={"tool_name": tname, "arguments": targs, "error": raw_output.get("error")},
                )
            elif status == "rejected_sql":
                audit_logger.record(
                    thread_id=thread_id,
                    company_id=company_id,
                    actor_id=actor_id,
                    event_type="SQL_QUERY_REJECTED",
                    status="denied",
                    details={"tool_name": tname, "arguments": targs, "error": raw_output.get("error")},
                )

            # Mint canonical sequential evidence_ids for every citation produced
            raw_citations = raw_output.pop("raw_citations", [])
            minted_cits: list[dict[str, Any]] = []
            minted_ids: list[str] = []
            for rc in raw_citations:
                counter += 1
                eid = f"E{counter}"
                cit = Citation(evidence_id=eid, **rc).model_dump(mode="json")
                evidence_store[eid] = cit
                minted_cits.append({"evidence_id": eid, "summary": cit["summary"]})
                minted_ids.append(eid)

            raw_output["evidence"] = minted_cits
            traces.append(
                ToolTrace(
                    step=step,
                    tool_name=tname,
                    arguments=targs,
                    result_count=raw_output.get("count"),
                    evidence_ids=minted_ids,
                    status=status,
                    summary=f"{tname} -> status={status}, count={raw_output.get('count')}, citations={len(minted_ids)}",
                ).model_dump(mode="json")
            )
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tc_id,
                    name=tname,
                    content=json.dumps(raw_output, default=str),
                )
            )

        return {
            "messages": tool_messages,
            "evidence_store": evidence_store,
            "evidence_counter": counter,
            "tool_traces": traces,
            "candidate_proposals": proposals,
            "step_count": step,
        }

    def policy_gate_node(state: FleetAgentState) -> dict[str, Any]:
        company_id = state["company_id"]
        actor_id = state.get("actor_id", "it-admin")
        thread_id = state.get("thread_id", "default")
        repo = TenantTelemetryRepository(db, company_id)
        validator = ActionPolicyValidator(repo)

        evidence_objs = {
            eid: Citation.model_validate(c)
            for eid, c in (state.get("evidence_store") or {}).items()
        }

        validated: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []

        for raw_prop in state.get("candidate_proposals") or []:
            proposal = ActionProposal.model_validate(raw_prop)
            verdict = validator.validate(proposal, evidence_objs)
            if verdict.allowed:
                proposal.citations = [
                    evidence_objs[eid] for eid in proposal.evidence_ids if eid in evidence_objs
                ]
                validated.append(proposal.model_dump(mode="json"))
                audit_logger.record(
                    thread_id=thread_id,
                    company_id=company_id,
                    actor_id=actor_id,
                    event_type="ACTION_PROPOSED",
                    status="awaiting_approval",
                    target_entity=proposal.target_device_id or proposal.target_employee_id,
                    details={
                        "proposal_id": proposal.proposal_id,
                        "action_type": proposal.action_type.value,
                        "parameters": proposal.parameters,
                        "evidence_ids": proposal.evidence_ids,
                        "risk_level": proposal.risk_level,
                    },
                )
            else:
                blocked_entry = {
                    "proposal": proposal.model_dump(mode="json"),
                    "verdict": verdict.model_dump(mode="json"),
                }
                blocked.append(blocked_entry)
                event_type = (
                    "CROSS_TENANT_BLOCKED"
                    if verdict.violation_code in ("CROSS_TENANT_ACCESS", "CROSS_TENANT_EVIDENCE")
                    else "ACTION_POLICY_BLOCKED"
                )
                audit_logger.record(
                    thread_id=thread_id,
                    company_id=company_id,
                    actor_id=actor_id,
                    event_type=event_type,
                    status="blocked",
                    target_entity=proposal.target_device_id or proposal.target_employee_id,
                    details={
                        "proposal_id": proposal.proposal_id,
                        "action_type": proposal.action_type.value,
                        "violation_code": verdict.violation_code,
                        "reasons": verdict.reasons,
                    },
                )

        return {
            "validated_proposals": validated,
            "blocked_proposals": blocked,
        }

    def route_after_policy_gate(state: FleetAgentState) -> str:
        if state.get("validated_proposals"):
            return "hitl_approval"
        return "synthesizer"

    def hitl_approval_node(state: FleetAgentState) -> dict[str, Any]:
        """Native LangGraph Human-in-the-Loop interrupt pausing execution for explicit approval."""
        validated = state.get("validated_proposals") or []
        payload = {
            "type": "action_approval_required",
            "company_id": state["company_id"],
            "thread_id": state.get("thread_id", "default"),
            "proposals": validated,
            "draft_analysis": state.get("draft_answer", ""),
            "instructions": (
                "Review each proposed action and its verified telemetry citations. "
                "Resume with {'decisions': [{'proposal_id': ..., 'approved': True/False, 'actor_id': ..., 'comment': ...}]}."
            ),
        }

        # Graph checkpoints here and suspends until caller invokes Command(resume=...)
        resume_payload = interrupt(payload)

        decisions_raw = (
            resume_payload.get("decisions", [])
            if isinstance(resume_payload, dict)
            else []
        )
        by_id: dict[str, dict[str, Any]] = {
            str(d.get("proposal_id")): d for d in decisions_raw if isinstance(d, dict)
        }

        normalized_decisions: list[dict[str, Any]] = []
        default_actor = state.get("actor_id", "it-admin")
        for raw_prop in validated:
            pid = raw_prop["proposal_id"]
            d = by_id.get(pid)
            if d is None:
                # Fail-closed safety: missing explicit decision defaults to REJECTED
                decision = HumanDecision(
                    proposal_id=pid,
                    approved=False,
                    actor_id=default_actor,
                    comment="No explicit approval provided (fail-closed rejection).",
                )
            else:
                decision = HumanDecision(
                    proposal_id=pid,
                    approved=bool(d.get("approved", False)),
                    actor_id=str(d.get("actor_id") or default_actor),
                    comment=str(d.get("comment") or ""),
                )
            normalized_decisions.append(decision.model_dump(mode="json"))

        return {"human_decisions": normalized_decisions}

    def action_executor_node(state: FleetAgentState) -> dict[str, Any]:
        thread_id = state.get("thread_id", "default")
        decisions_by_id = {
            d["proposal_id"]: HumanDecision.model_validate(d)
            for d in (state.get("human_decisions") or [])
        }

        executed: list[dict[str, Any]] = []
        for raw_prop in state.get("validated_proposals") or []:
            proposal = ActionProposal.model_validate(raw_prop)
            decision = decisions_by_id.get(
                proposal.proposal_id,
                HumanDecision(
                    proposal_id=proposal.proposal_id,
                    approved=False,
                    actor_id=state.get("actor_id", "it-admin"),
                    comment="Missing approval record (fail-closed).",
                ),
            )
            record = action_executor.execute(proposal, decision, thread_id=thread_id)
            executed.append(record.model_dump(mode="json"))

        return {"executed_actions": executed}

    def synthesizer_node(state: FleetAgentState) -> dict[str, Any]:
        if state.get("cross_tenant_blocked"):
            msg = state.get("cross_tenant_reason") or "Access denied due to tenant isolation."
            return {
                "final_response": msg,
                "cited_evidence_ids": [],
                "messages": [AIMessage(content=msg)],
            }

        evidence_store = state.get("evidence_store") or {}
        draft = _sanitize_inline_citations((state.get("draft_answer") or "").strip(), evidence_store)

        # Verify which cited evidence IDs actually exist in the verified evidence store
        cited_in_text = [
            eid for eid in dict.fromkeys(_CITATION_TAG_RE.findall(draft)) if eid in evidence_store
        ]
        # Also include evidence_ids attached to executed actions
        for act in state.get("executed_actions") or []:
            for eid in act.get("evidence_ids", []):
                if eid in evidence_store and eid not in cited_in_text:
                    cited_in_text.append(eid)

        sections: list[str] = []
        if draft:
            sections.append(draft)

        # Append deterministic policy-blocked explanations (cannot be hidden by LLM)
        blocked = state.get("blocked_proposals") or []
        if blocked:
            lines = ["### Guardrail Refusals (Blocked by Policy & Evidence Gate)"]
            for item in blocked:
                prop = item["proposal"]
                verdict = item["verdict"]
                target = prop.get("target_device_id") or prop.get("target_employee_id") or "unspecified"
                lines.append(
                    f"- **{prop['action_type']}** on `{target}` **BLOCKED** "
                    f"(`{verdict.get('violation_code')}`): {'; '.join(verdict.get('reasons', []))}"
                )
            sections.append("\n".join(lines))

        # Append deterministic executed/rejected action outcomes
        executed = state.get("executed_actions") or []
        if executed:
            lines = ["### Action Outcomes (Persisted & Audited)"]
            for rec in executed:
                target = rec.get("target_device_id") or rec.get("target_employee_id") or "unspecified"
                status = rec["status"].upper()
                line = (
                    f"- **{rec['action_type']}** on `{target}` — **{status}** "
                    f"(ID: `{rec['action_id']}`, reviewer: `{rec.get('approved_by')}`)"
                )
                if rec.get("approval_comment"):
                    line += f" — _{rec['approval_comment']}_"
                if rec.get("external_dispatch"):
                    line += f" | external dispatch: `{rec['external_dispatch']}`"
                lines.append(line)
            sections.append("\n".join(lines))

        final_text = "\n\n".join(sections).strip() or "No findings to report."
        return {
            "final_response": final_text,
            "cited_evidence_ids": cited_in_text,
        }

    # Assemble explicit LangGraph StateGraph
    builder = StateGraph(FleetAgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("reasoner", reasoner_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("policy_gate", policy_gate_node)
    builder.add_node("hitl_approval", hitl_approval_node)
    builder.add_node("action_executor", action_executor_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"synthesizer": "synthesizer", "reasoner": "reasoner"},
    )
    builder.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {
            "tool_executor": "tool_executor",
            "policy_gate": "policy_gate",
            "synthesizer": "synthesizer",
        },
    )
    builder.add_edge("tool_executor", "reasoner")
    builder.add_conditional_edges(
        "policy_gate",
        route_after_policy_gate,
        {"hitl_approval": "hitl_approval", "synthesizer": "synthesizer"},
    )
    builder.add_edge("hitl_approval", "action_executor")
    builder.add_edge("action_executor", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)
