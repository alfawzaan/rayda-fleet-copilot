"""Typed LangGraph state definition for the Fleet Copilot agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class FleetAgentState(TypedDict, total=False):
    """Explicit state tracked across the LangGraph planning, tool, policy, and HITL loop."""

    # Conversation history (persisted across turns via checkpointer)
    messages: Annotated[list[AnyMessage], add_messages]

    # Immutable session security context
    company_id: str
    actor_id: str
    thread_id: str

    # Per-turn planning & bounded execution loop
    plan: list[str]
    intent: str
    cross_tenant_blocked: bool
    cross_tenant_reason: str
    step_count: int
    evidence_counter: int

    # Grounding & traceability store (persisted across turns in the thread)
    evidence_store: dict[str, dict[str, Any]]
    tool_traces: list[dict[str, Any]]

    # Action governance lifecycle (per-turn)
    candidate_proposals: list[dict[str, Any]]
    validated_proposals: list[dict[str, Any]]
    blocked_proposals: list[dict[str, Any]]
    human_decisions: list[dict[str, Any]]
    executed_actions: list[dict[str, Any]]

    # Grounded final answer
    draft_answer: str
    final_response: str
    cited_evidence_ids: list[str]
