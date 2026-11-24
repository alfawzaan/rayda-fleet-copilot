"""Tests for ActionPolicyValidator, real transactional ActionExecutor, webhooks, and audit logs."""

from __future__ import annotations

import json
from typing import Any

import httpx

from fleet_copilot.actions.audit import AuditLogger
from fleet_copilot.actions.backend import ActionExecutor, build_idempotency_key
from fleet_copilot.actions.policy import ActionPolicyValidator
from fleet_copilot.models.actions import ActionProposal, ActionStatus, ActionType, HumanDecision
from fleet_copilot.models.evidence import Citation
from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.repository import TenantTelemetryRepository


def _sample_citation(eid: str, company_id: str, device_id: str, employee_id: str) -> Citation:
    return Citation(
        evidence_id=eid,
        source_tool="query_low_storage_devices",
        company_id=company_id,
        device_id=device_id,
        employee_id=employee_id,
        collected_at="2026-06-12T09:00:00Z",
        metric="disk_available_pct",
        value="2.0%",
        summary=f"{device_id} has 2.0% free storage.",
    )


def test_policy_gate_enforces_tenant_existence_and_evidence(populated_db: FleetDatabase) -> None:
    repo = TenantTelemetryRepository(populated_db, "acme-001")
    validator = ActionPolicyValidator(repo)
    ev_store = {
        "E1": _sample_citation("E1", "acme-001", "MT7PJB7N5LRE", "emp-acme-1003"),
    }

    # 1. Cross-tenant target device blocked
    globex_dev = TenantTelemetryRepository(populated_db, "globex-002").list_latest_devices()[0]["device_id"]
    p_cross = ActionProposal(
        proposal_id="PROP-1",
        action_type=ActionType.FLAG_DEVICE_FOR_REPLACEMENT,
        company_id="acme-001",
        target_device_id=globex_dev,
        parameters={"device_id": globex_dev, "reason": "Test cross tenant"},
        rationale="Attempting cross-tenant action with valid looking text.",
        evidence_ids=["E1"],
    )
    v = validator.validate(p_cross, ev_store)
    assert not v.allowed
    assert v.violation_code == "CROSS_TENANT_ACCESS"

    # 2. Missing evidence_ids refused
    p_no_ev = ActionProposal(
        proposal_id="PROP-2",
        action_type=ActionType.OPEN_REMEDIATION_TICKET,
        company_id="acme-001",
        target_device_id="MT7PJB7N5LRE",
        parameters={"device_id": "MT7PJB7N5LRE", "check_id": "screen_lock", "note": "Enable screen lock"},
        rationale="Device needs screen lock remediation.",
        evidence_ids=[],
    )
    v = validator.validate(p_no_ev, ev_store)
    assert not v.allowed
    assert v.violation_code == "INSUFFICIENT_EVIDENCE"

    # 3. Fabricated evidence_id refused
    p_fake_ev = p_no_ev.model_copy(update={"proposal_id": "PROP-3", "evidence_ids": ["E999"]})
    v = validator.validate(p_fake_ev, ev_store)
    assert not v.allowed
    assert v.violation_code == "INSUFFICIENT_EVIDENCE"

    # 4. Citation pointing to a DIFFERENT device refused (entity-to-evidence binding)
    p_mismatch = ActionProposal(
        proposal_id="PROP-4",
        action_type=ActionType.FLAG_DEVICE_FOR_REPLACEMENT,
        company_id="acme-001",
        target_device_id="1LYSSFD074BB",  # Healthy device, whereas E1 is for MT7PJB7N5LRE
        parameters={"device_id": "1LYSSFD074BB", "reason": "Replace laptop"},
        rationale="Trying to cite another device's evidence.",
        evidence_ids=["E1"],
    )
    v = validator.validate(p_mismatch, ev_store)
    assert not v.allowed
    assert v.violation_code == "INSUFFICIENT_EVIDENCE"

    # 5. Valid proposal with verified target-matching evidence passes
    p_ok = ActionProposal(
        proposal_id="PROP-5",
        action_type=ActionType.CREATE_UPGRADE_ORDER,
        company_id="acme-001",
        target_device_id="MT7PJB7N5LRE",
        parameters={"device_id": "MT7PJB7N5LRE", "component": "Storage", "spec": "1 TB SSD"},
        rationale="Storage dropped to 2.0% free space on Macintosh HD.",
        evidence_ids=["E1"],
    )
    v = validator.validate(p_ok, ev_store)
    assert v.allowed


def test_action_executor_persists_idempotently_and_dispatches_webhook(
    populated_db: FleetDatabase,
    audit_logger: AuditLogger,
    monkeypatch: Any,
) -> None:
    captured_requests: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured_requests.append({"url": url, "headers": kwargs.get("headers"), "content": kwargs.get("content")})
        return httpx.Response(200, json={"accepted": True})

    monkeypatch.setattr(httpx, "post", fake_post)

    executor = ActionExecutor(
        populated_db,
        audit_logger,
        action_webhook_url="https://itsm.example.com/webhooks/rayda",
        action_webhook_secret="super-secret",
    )

    proposal = ActionProposal(
        proposal_id="PROP-100",
        action_type=ActionType.OPEN_REMEDIATION_TICKET,
        company_id="acme-001",
        target_device_id="MT7PJB7N5LRE",
        parameters={"device_id": "MT7PJB7N5LRE", "check_id": "low_storage", "note": "Free disk space"},
        rationale="Disk critically low at 2.0% free.",
        evidence_ids=["E1"],
    )
    proposal.idempotency_key = build_idempotency_key(proposal)

    decision = HumanDecision(
        proposal_id="PROP-100", approved=True, actor_id="alice-admin", comment="Approved for immediate ticket"
    )

    rec1 = executor.execute(proposal, decision, thread_id="t-1")
    assert rec1.status == ActionStatus.EXECUTED
    assert rec1.action_id.startswith("TKT-")
    assert rec1.external_dispatch is not None
    assert rec1.external_dispatch["action_webhook"]["ok"] is True
    assert len(captured_requests) == 1
    assert captured_requests[0]["headers"]["X-Rayda-Signature"].startswith("sha256=")
    assert json.loads(captured_requests[0]["content"])["action_id"] == rec1.action_id

    # Idempotency: re-executing the identical action returns existing record without double-dispatching
    rec2 = executor.execute(proposal, decision, thread_id="t-1")
    assert rec2.action_id == rec1.action_id
    assert len(captured_requests) == 1

    # Persisted and tenant-scoped
    assert len(executor.list_actions("acme-001")) == 1
    assert len(executor.list_actions("globex-002")) == 0

    events = audit_logger.list_events("acme-001")
    event_types = [e.event_type for e in events]
    assert "ACTION_EXECUTED" in event_types
    assert "ACTION_DUPLICATE_PREVENTED" in event_types
