"""Action proposals, policy verdicts, HITL decisions, and audit log models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from fleet_copilot.models.evidence import Citation


class ActionType(StrEnum):
    """Supported state-changing operational actions."""

    CREATE_UPGRADE_ORDER = "create_upgrade_order"
    OPEN_REMEDIATION_TICKET = "open_remediation_ticket"
    FLAG_DEVICE_FOR_REPLACEMENT = "flag_device_for_replacement"
    NOTIFY_EMPLOYEE = "notify_employee"


class ActionStatus(StrEnum):
    """Lifecycle status of an operational action."""

    PROPOSED = "proposed"
    BLOCKED_POLICY = "blocked_policy"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"


class ActionProposal(BaseModel):
    """A candidate state-changing action awaiting policy validation & human approval."""

    proposal_id: str
    action_type: ActionType
    company_id: str
    target_device_id: str | None = None
    target_employee_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    risk_level: str = "medium"
    idempotency_key: str = ""


class PolicyVerdict(BaseModel):
    """Deterministic outcome of the Policy & Evidence Sufficiency Gate."""

    proposal_id: str
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    violation_code: str | None = None


class HumanDecision(BaseModel):
    """Explicit IT administrator approval or rejection decision."""

    proposal_id: str
    approved: bool
    actor_id: str
    comment: str = ""
    decided_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )


class ActionExecutionRecord(BaseModel):
    """Persisted record of an executed operational action."""

    action_id: str
    proposal_id: str
    action_type: ActionType
    company_id: str
    target_device_id: str | None = None
    target_employee_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus
    approved_by: str | None = None
    approval_comment: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    external_dispatch: dict[str, Any] | None = None
    idempotency_key: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )


class AuditEvent(BaseModel):
    """Immutable audit log entry for security, compliance, and operational traceability."""

    event_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    thread_id: str
    company_id: str
    actor_id: str
    event_type: str
    status: str
    target_entity: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
