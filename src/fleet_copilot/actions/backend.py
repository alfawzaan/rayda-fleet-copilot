"""Real transactional action execution backend with optional external webhooks.

Approved operational actions are persisted transactionally in SQLite (with
idempotency guarantees and tenant scoping) and optionally dispatched over HTTP
to real external ITSM / procurement / Slack webhook endpoints when configured.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from typing import Any

import httpx

from fleet_copilot.actions.audit import AuditLogger
from fleet_copilot.models.actions import (
    ActionExecutionRecord,
    ActionProposal,
    ActionStatus,
    ActionType,
    HumanDecision,
)
from fleet_copilot.storage.database import FleetDatabase

_PREFIX_BY_ACTION = {
    ActionType.CREATE_UPGRADE_ORDER: "ORD",
    ActionType.OPEN_REMEDIATION_TICKET: "TKT",
    ActionType.FLAG_DEVICE_FOR_REPLACEMENT: "RPL",
    ActionType.NOTIFY_EMPLOYEE: "NTF",
}


def build_idempotency_key(proposal: ActionProposal) -> str:
    """Compute a deterministic idempotency key preventing duplicate executed actions."""
    canonical = {
        "company_id": proposal.company_id,
        "action_type": proposal.action_type.value,
        "target_device_id": proposal.target_device_id,
        "target_employee_id": proposal.target_employee_id,
        "parameters": proposal.parameters,
    }
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


class ActionExecutor:
    """Executes human-approved proposals against local ITSM storage and optional webhooks."""

    def __init__(
        self,
        db: FleetDatabase,
        audit_logger: AuditLogger,
        *,
        action_webhook_url: str | None = None,
        action_webhook_secret: str | None = None,
        slack_webhook_url: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.db = db
        self.audit = audit_logger
        self.action_webhook_url = action_webhook_url
        self.action_webhook_secret = action_webhook_secret
        self.slack_webhook_url = slack_webhook_url
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        proposal: ActionProposal,
        decision: HumanDecision,
        *,
        thread_id: str,
    ) -> ActionExecutionRecord:
        """Execute an approved proposal or persist its rejection."""
        idem_key = proposal.idempotency_key or build_idempotency_key(proposal)

        if not decision.approved:
            record = ActionExecutionRecord(
                action_id=f"REJ-{uuid.uuid4().hex[:10].upper()}",
                proposal_id=proposal.proposal_id,
                action_type=proposal.action_type,
                company_id=proposal.company_id,
                target_device_id=proposal.target_device_id,
                target_employee_id=proposal.target_employee_id,
                parameters=proposal.parameters,
                status=ActionStatus.REJECTED,
                approved_by=decision.actor_id,
                approval_comment=decision.comment,
                evidence_ids=proposal.evidence_ids,
                idempotency_key=idem_key,
            )
            self._persist_record(record)
            self.audit.record(
                thread_id=thread_id,
                company_id=proposal.company_id,
                actor_id=decision.actor_id,
                event_type="ACTION_REJECTED",
                status="rejected",
                target_entity=proposal.target_device_id or proposal.target_employee_id,
                details={
                    "proposal_id": proposal.proposal_id,
                    "action_type": proposal.action_type.value,
                    "comment": decision.comment,
                },
            )
            return record

        # Idempotency check: prevent duplicate executed state mutations
        existing = self._find_existing_executed(proposal.company_id, idem_key)
        if existing is not None:
            self.audit.record(
                thread_id=thread_id,
                company_id=proposal.company_id,
                actor_id=decision.actor_id,
                event_type="ACTION_DUPLICATE_PREVENTED",
                status="deduplicated",
                target_entity=proposal.target_device_id or proposal.target_employee_id,
                details={
                    "existing_action_id": existing.action_id,
                    "proposal_id": proposal.proposal_id,
                    "idempotency_key": idem_key,
                },
            )
            return existing

        prefix = _PREFIX_BY_ACTION[proposal.action_type]
        action_id = f"{prefix}-{uuid.uuid4().hex[:10].upper()}"

        dispatch_result = self._dispatch_external(action_id, proposal, decision)

        record = ActionExecutionRecord(
            action_id=action_id,
            proposal_id=proposal.proposal_id,
            action_type=proposal.action_type,
            company_id=proposal.company_id,
            target_device_id=proposal.target_device_id,
            target_employee_id=proposal.target_employee_id,
            parameters=proposal.parameters,
            status=ActionStatus.EXECUTED,
            approved_by=decision.actor_id,
            approval_comment=decision.comment,
            evidence_ids=proposal.evidence_ids,
            external_dispatch=dispatch_result,
            idempotency_key=idem_key,
        )
        self._persist_record(record)

        self.audit.record(
            thread_id=thread_id,
            company_id=proposal.company_id,
            actor_id=decision.actor_id,
            event_type="ACTION_EXECUTED",
            status="executed",
            target_entity=proposal.target_device_id or proposal.target_employee_id,
            details={
                "action_id": record.action_id,
                "proposal_id": proposal.proposal_id,
                "action_type": proposal.action_type.value,
                "parameters": proposal.parameters,
                "evidence_ids": proposal.evidence_ids,
                "external_dispatch": dispatch_result,
            },
        )
        return record

    def _find_existing_executed(
        self, company_id: str, idempotency_key: str
    ) -> ActionExecutionRecord | None:
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                SELECT * FROM executed_actions
                WHERE company_id = ? AND idempotency_key = ? AND status = 'executed'
                LIMIT 1
                """,
                (company_id, idempotency_key),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def _persist_record(self, record: ActionExecutionRecord) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO executed_actions (
                    action_id, proposal_id, action_type, company_id,
                    target_device_id, target_employee_id, parameters_json,
                    status, approved_by, approval_comment, evidence_ids_json,
                    external_dispatch_json, idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.action_id,
                    record.proposal_id,
                    record.action_type.value,
                    record.company_id,
                    record.target_device_id,
                    record.target_employee_id,
                    json.dumps(record.parameters, sort_keys=True),
                    record.status.value,
                    record.approved_by,
                    record.approval_comment,
                    json.dumps(record.evidence_ids),
                    json.dumps(record.external_dispatch) if record.external_dispatch else None,
                    record.idempotency_key,
                    record.created_at,
                ),
            )

    def _dispatch_external(
        self, action_id: str, proposal: ActionProposal, decision: HumanDecision
    ) -> dict[str, Any] | None:
        """Dispatch real HTTP payloads to configured external webhook / Slack endpoints."""
        dispatch_summary: dict[str, Any] = {}

        if self.action_webhook_url:
            payload = {
                "action_id": action_id,
                "proposal_id": proposal.proposal_id,
                "action_type": proposal.action_type.value,
                "company_id": proposal.company_id,
                "target_device_id": proposal.target_device_id,
                "target_employee_id": proposal.target_employee_id,
                "parameters": proposal.parameters,
                "rationale": proposal.rationale,
                "evidence_ids": proposal.evidence_ids,
                "approved_by": decision.actor_id,
                "decided_at": decision.decided_at,
            }
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.action_webhook_secret:
                sig = hmac.new(
                    self.action_webhook_secret.encode("utf-8"), body, hashlib.sha256
                ).hexdigest()
                headers["X-Rayda-Signature"] = f"sha256={sig}"
            try:
                resp = httpx.post(
                    self.action_webhook_url,
                    content=body,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                dispatch_summary["action_webhook"] = {
                    "url": self.action_webhook_url,
                    "status_code": resp.status_code,
                    "ok": resp.is_success,
                }
            except httpx.HTTPError as exc:
                dispatch_summary["action_webhook"] = {
                    "url": self.action_webhook_url,
                    "ok": False,
                    "error": str(exc),
                }

        if self.slack_webhook_url and proposal.action_type == ActionType.NOTIFY_EMPLOYEE:
            text = (
                f"[Rayda Fleet Copilot | {proposal.company_id}] "
                f"Notification for `{proposal.target_employee_id}` "
                f"(approved by `{decision.actor_id}`): {proposal.parameters.get('message', '')}"
            )
            try:
                resp = httpx.post(
                    self.slack_webhook_url,
                    json={"text": text},
                    timeout=self.timeout_seconds,
                )
                dispatch_summary["slack_webhook"] = {
                    "status_code": resp.status_code,
                    "ok": resp.is_success,
                }
            except httpx.HTTPError as exc:
                dispatch_summary["slack_webhook"] = {"ok": False, "error": str(exc)}

        return dispatch_summary or None

    def list_actions(self, company_id: str, *, limit: int = 100) -> list[ActionExecutionRecord]:
        """List persisted operational actions strictly scoped to a single tenant."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM executed_actions
                WHERE company_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (company_id, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: Any) -> ActionExecutionRecord:
        return ActionExecutionRecord(
            action_id=row["action_id"],
            proposal_id=row["proposal_id"],
            action_type=ActionType(row["action_type"]),
            company_id=row["company_id"],
            target_device_id=row["target_device_id"],
            target_employee_id=row["target_employee_id"],
            parameters=json.loads(row["parameters_json"]),
            status=ActionStatus(row["status"]),
            approved_by=row["approved_by"],
            approval_comment=row["approval_comment"] or "",
            evidence_ids=json.loads(row["evidence_ids_json"]),
            external_dispatch=(
                json.loads(row["external_dispatch_json"])
                if row["external_dispatch_json"]
                else None
            ),
            idempotency_key=row["idempotency_key"],
            created_at=row["created_at"],
        )
