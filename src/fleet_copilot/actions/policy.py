"""Deterministic Policy & Evidence Sufficiency Gate for operational actions.

Runs strictly in code before any proposal is allowed to reach human approval.
Prevents cross-tenant actions, unknown targets, fabricated citations, and
unsupported/evidence-free action proposals.
"""

from __future__ import annotations

from fleet_copilot.models.actions import ActionProposal, ActionType, PolicyVerdict
from fleet_copilot.models.evidence import Citation
from fleet_copilot.storage.repository import CrossTenantAccessError, TenantTelemetryRepository


class ActionPolicyValidator:
    """Validates candidate action proposals against tenant boundaries and evidence rules."""

    def __init__(self, repo: TenantTelemetryRepository) -> None:
        self.repo = repo

    def validate(
        self,
        proposal: ActionProposal,
        evidence_store: dict[str, Citation],
    ) -> PolicyVerdict:
        """Deterministically evaluate whether a proposal may proceed to HITL approval."""
        reasons: list[str] = []

        # 1. Tenant scope check on proposal envelope
        if proposal.company_id != self.repo.company_id:
            return PolicyVerdict(
                proposal_id=proposal.proposal_id,
                allowed=False,
                violation_code="CROSS_TENANT_ACCESS",
                reasons=[
                    f"Proposal company_id '{proposal.company_id}' does not match active tenant '{self.repo.company_id}'."
                ],
            )

        # 2. Target entity ownership & existence check
        target_device = proposal.target_device_id
        target_employee = proposal.target_employee_id

        if proposal.action_type in (
            ActionType.CREATE_UPGRADE_ORDER,
            ActionType.OPEN_REMEDIATION_TICKET,
            ActionType.FLAG_DEVICE_FOR_REPLACEMENT,
        ):
            if not target_device:
                return PolicyVerdict(
                    proposal_id=proposal.proposal_id,
                    allowed=False,
                    violation_code="MISSING_TARGET",
                    reasons=[f"Action '{proposal.action_type.value}' requires a valid target device_id."],
                )
            try:
                exists = self.repo.assert_device_in_tenant(target_device)
            except CrossTenantAccessError as exc:
                return PolicyVerdict(
                    proposal_id=proposal.proposal_id,
                    allowed=False,
                    violation_code="CROSS_TENANT_ACCESS",
                    reasons=[str(exc)],
                )
            if not exists:
                return PolicyVerdict(
                    proposal_id=proposal.proposal_id,
                    allowed=False,
                    violation_code="UNKNOWN_ENTITY",
                    reasons=[
                        f"Device '{target_device}' was not found in tenant '{self.repo.company_id}' telemetry."
                    ],
                )

        if proposal.action_type == ActionType.NOTIFY_EMPLOYEE:
            if not target_employee:
                return PolicyVerdict(
                    proposal_id=proposal.proposal_id,
                    allowed=False,
                    violation_code="MISSING_TARGET",
                    reasons=["Action 'notify_employee' requires a valid employee_id."],
                )
            try:
                exists = self.repo.assert_employee_in_tenant(target_employee)
            except CrossTenantAccessError as exc:
                return PolicyVerdict(
                    proposal_id=proposal.proposal_id,
                    allowed=False,
                    violation_code="CROSS_TENANT_ACCESS",
                    reasons=[str(exc)],
                )
            if not exists:
                return PolicyVerdict(
                    proposal_id=proposal.proposal_id,
                    allowed=False,
                    violation_code="UNKNOWN_ENTITY",
                    reasons=[
                        f"Employee '{target_employee}' was not found in tenant '{self.repo.company_id}' telemetry."
                    ],
                )

        # 3. Required action parameters validation
        param_errors = self._validate_parameters(proposal)
        if param_errors:
            return PolicyVerdict(
                proposal_id=proposal.proposal_id,
                allowed=False,
                violation_code="INVALID_PARAMETERS",
                reasons=param_errors,
            )

        # 4. Rationale quality check
        if not proposal.rationale or len(proposal.rationale.strip()) < 15:
            reasons.append("Action rationale must clearly explain the operational justification (>= 15 chars).")

        # 5. Evidence sufficiency check — citations MUST exist in the run's verified evidence_store
        if not proposal.evidence_ids:
            reasons.append(
                "Insufficient evidence: no verified telemetry evidence_ids were attached to this action."
            )
            return PolicyVerdict(
                proposal_id=proposal.proposal_id,
                allowed=False,
                violation_code="INSUFFICIENT_EVIDENCE",
                reasons=reasons,
            )

        verified_citations: list[Citation] = []
        missing_ids: list[str] = []
        for eid in proposal.evidence_ids:
            cit = evidence_store.get(eid)
            if cit is None:
                missing_ids.append(eid)
            elif cit.company_id != self.repo.company_id:
                return PolicyVerdict(
                    proposal_id=proposal.proposal_id,
                    allowed=False,
                    violation_code="CROSS_TENANT_EVIDENCE",
                    reasons=[f"Citation '{eid}' belongs to another tenant."],
                )
            else:
                verified_citations.append(cit)

        if missing_ids:
            reasons.append(
                f"Unverified/fabricated evidence_ids referenced: {', '.join(missing_ids)}. "
                "Actions must cite evidence actually retrieved from telemetry tools in this session."
            )

        if not verified_citations:
            return PolicyVerdict(
                proposal_id=proposal.proposal_id,
                allowed=False,
                violation_code="INSUFFICIENT_EVIDENCE",
                reasons=reasons or ["No verified telemetry citations found."],
            )

        # 6. Entity-to-evidence binding check: cited evidence must actually cover the target entity
        if target_device:
            matching_device = [c for c in verified_citations if c.device_id == target_device]
            if not matching_device:
                reasons.append(
                    f"Insufficient entity-specific evidence: none of the cited records ({proposal.evidence_ids}) "
                    f"actually reference target device '{target_device}'."
                )

        if target_employee and proposal.action_type == ActionType.NOTIFY_EMPLOYEE:
            # Evidence matches either directly on employee_id or on a device owned by that employee
            employee_devices = {
                d["device_id"] for d in self.repo.list_latest_devices(employee_id=target_employee)
            }
            matching_emp = [
                c
                for c in verified_citations
                if c.employee_id == target_employee or (c.device_id and c.device_id in employee_devices)
            ]
            if not matching_emp:
                reasons.append(
                    f"Insufficient entity-specific evidence: none of the cited records ({proposal.evidence_ids}) "
                    f"reference employee '{target_employee}' or any of their assigned devices."
                )

        if reasons:
            return PolicyVerdict(
                proposal_id=proposal.proposal_id,
                allowed=False,
                violation_code="INSUFFICIENT_EVIDENCE",
                reasons=reasons,
            )

        return PolicyVerdict(proposal_id=proposal.proposal_id, allowed=True, reasons=[])

    @staticmethod
    def _validate_parameters(proposal: ActionProposal) -> list[str]:
        errs: list[str] = []
        params = proposal.parameters
        if proposal.action_type == ActionType.CREATE_UPGRADE_ORDER:
            if not str(params.get("component", "")).strip():
                errs.append("create_upgrade_order requires a non-empty 'component' (e.g., RAM, SSD, Battery).")
            if not str(params.get("spec", "")).strip():
                errs.append("create_upgrade_order requires a non-empty 'spec' (e.g., 32GB RAM, 1TB NVMe).")
        elif proposal.action_type == ActionType.OPEN_REMEDIATION_TICKET:
            if not str(params.get("check_id", "")).strip():
                errs.append("open_remediation_ticket requires a non-empty 'check_id'.")
            if not str(params.get("note", "")).strip():
                errs.append("open_remediation_ticket requires a non-empty 'note'.")
        elif proposal.action_type == ActionType.FLAG_DEVICE_FOR_REPLACEMENT:
            if not str(params.get("reason", "")).strip():
                errs.append("flag_device_for_replacement requires a non-empty 'reason'.")
        elif proposal.action_type == ActionType.NOTIFY_EMPLOYEE and not str(params.get("message", "")).strip():
            errs.append("notify_employee requires a non-empty 'message'.")
        return errs
