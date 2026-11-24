"""Tenant-bound tool definitions for the Fleet Copilot LangGraph agent.

Security guarantee:
- NO read, analytics, or action tool accepts ``company_id`` as an LLM argument.
- Every tool is bound by closure to a ``TenantTelemetryRepository`` pinned to
  the authenticated session's ``company_id``.
- Action tools NEVER execute state mutations directly; they are intercepted by
  the graph's ``tool_executor`` node and staged as ``ActionProposal`` objects
  that must pass the deterministic Policy & Evidence Gate and Human-in-the-Loop
  ``interrupt()`` approval before execution.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from fleet_copilot.analytics.trends import (
    detect_battery_degradation,
    detect_compliance_drift,
    detect_ram_constraints,
    detect_software_anomalies,
    detect_storage_trends,
    generate_fleet_insights,
)
from fleet_copilot.models.actions import ActionType
from fleet_copilot.storage.repository import (
    CrossTenantAccessError,
    TenantTelemetryRepository,
    UnsafeQueryError,
)

ACTION_TOOL_NAMES = frozenset(t.value for t in ActionType)


# ---------------------------------------------------------------------------
# Tool input schemas (explicit Pydantic models for accurate LLM tool calling)
# ---------------------------------------------------------------------------


class EmptyArgs(BaseModel):
    pass


class ListDevicesArgs(BaseModel):
    os_name: str | None = Field(
        default=None,
        description="Optional case-insensitive OS filter (e.g., 'macOS', 'Windows', 'Linux').",
    )
    device_type: str | None = Field(
        default=None,
        description="Optional form-factor filter: 'laptop', 'desktop', or 'all'.",
    )
    employee_id: str | None = Field(
        default=None,
        description="Optional employee_id filter within the active company.",
    )


class GetDeviceTelemetryArgs(BaseModel):
    device_id: str = Field(description="Target device_id within the active company.")
    start_date: str | None = Field(
        default=None, description="Optional ISO date/timestamp lower bound (inclusive)."
    )
    end_date: str | None = Field(
        default=None, description="Optional ISO date/timestamp upper bound (inclusive)."
    )
    latest_only: bool = Field(
        default=False,
        description="If true, return only the most recent telemetry snapshot for the device.",
    )


class LowStorageArgs(BaseModel):
    threshold_free_pct: float = Field(
        default=15.0,
        description="Disk free percentage threshold below which a device is considered low on storage.",
    )
    as_of_latest: bool = Field(
        default=True,
        description="If true, evaluate each device's latest snapshot; if false, search across all history.",
    )
    device_type: str | None = Field(
        default=None, description="Optional form-factor filter: 'laptop', 'desktop', or 'all'."
    )


class ComplianceFailuresArgs(BaseModel):
    severity: str | None = Field(
        default=None,
        description="Optional severity filter (e.g., 'high', 'medium', 'low'). Omit for all severities.",
    )
    check_id: str | None = Field(
        default=None,
        description="Optional specific compliance check_id (e.g., 'disk_encryption', 'screen_lock', 'os_up_to_date').",
    )
    as_of_latest: bool = Field(
        default=True,
        description="If true, evaluate each device's latest snapshot; if false, aggregate historical failures.",
    )
    device_type: str | None = Field(
        default=None, description="Optional form-factor filter: 'laptop', 'desktop', or 'all'."
    )


class OsVersionArgs(BaseModel):
    product_name: str | None = Field(
        default=None, description="Optional OS product name filter (e.g., 'macOS', 'Windows')."
    )
    below_version: str | None = Field(
        default=None,
        description="Optional version ceiling (e.g., '15' or '15.0' returns versions strictly older than 15.0).",
    )
    device_type: str | None = Field(
        default=None, description="Optional form-factor filter: 'laptop', 'desktop', or 'all'."
    )


class InstalledSoftwareArgs(BaseModel):
    software_name: str | None = Field(
        default=None,
        description="Optional case-insensitive software name substring to search for. Omit to list all.",
    )
    as_of_latest: bool = Field(
        default=True,
        description="If true, inspect latest snapshot per device; if false, search entire history.",
    )


class ReadonlySqlArgs(BaseModel):
    sql: str = Field(
        description=(
            "Single read-only SELECT or WITH SQL query against the tenant-isolated views: "
            "v_snapshots, v_disk_volumes, v_compliance_results, v_installed_software, v_network_interfaces. "
            "Supports json_extract(raw_payload, '$.field') for any custom/unseen telemetry attributes."
        )
    )


class BatteryTrendArgs(BaseModel):
    cycle_threshold: int = Field(default=800, description="Cycle count threshold indicating high wear.")
    min_capacity_drop_pct: float = Field(
        default=2.0, description="Minimum full-charge capacity drop percentage to flag degradation."
    )
    window_days: int | None = Field(
        default=None, description="Optional trailing lookback window in days. Omit for full history."
    )


class RamTrendArgs(BaseModel):
    used_pct_threshold: float = Field(
        default=80.0, description="Memory utilization percentage threshold."
    )
    min_breach_ratio: float = Field(
        default=0.5, description="Minimum fraction of snapshots above threshold (0.0 to 1.0)."
    )
    window_days: int | None = Field(default=None, description="Optional trailing lookback window in days.")


class StorageTrendArgs(BaseModel):
    low_free_pct_threshold: float = Field(
        default=15.0, description="Disk free space percentage considered low."
    )
    min_breach_ratio: float = Field(
        default=0.25, description="Minimum fraction of snapshots below threshold."
    )
    rapid_decline_slope_pct_per_day: float = Field(
        default=-0.35, description="Daily percentage decline slope threshold flagging rapid exhaustion."
    )
    window_days: int | None = Field(default=None, description="Optional trailing lookback window in days.")


class ComplianceDriftArgs(BaseModel):
    check_id: str | None = Field(
        default=None, description="Optional specific check_id to inspect for pass->fail regressions."
    )
    window_days: int | None = Field(default=None, description="Optional trailing lookback window in days.")


class SoftwareAnomaliesArgs(BaseModel):
    max_fleet_prevalence_ratio: float = Field(
        default=0.4,
        description="Maximum fraction of fleet devices having the app for it to be flagged as rare/outlier.",
    )
    window_days: int | None = Field(default=None, description="Optional trailing lookback window in days.")


class FleetInsightsArgs(BaseModel):
    window_days: int | None = Field(default=None, description="Optional trailing lookback window in days.")


# Action Tool Input Schemas (exact required signatures + evidence binding)
class CreateUpgradeOrderArgs(BaseModel):
    device_id: str = Field(description="Target device_id within the active company.")
    component: str = Field(description="Hardware component to upgrade (e.g., 'RAM', 'Storage', 'Battery').")
    spec: str = Field(description="Target upgrade specification (e.g., '32 GB RAM', '1 TB NVMe SSD').")
    rationale: str = Field(description="Concrete justification grounded in telemetry findings.")
    evidence_ids: list[str] = Field(
        description="List of verified evidence_ids (e.g., ['E1', 'E2']) retrieved earlier in this session supporting this order."
    )


class OpenRemediationTicketArgs(BaseModel):
    device_id: str = Field(description="Target device_id within the active company.")
    check_id: str = Field(description="Compliance check_id or issue identifier to remediate.")
    note: str = Field(description="Actionable remediation note for the IT engineering team.")
    rationale: str = Field(description="Concrete justification grounded in telemetry findings.")
    evidence_ids: list[str] = Field(
        description="List of verified evidence_ids (e.g., ['E1']) supporting this ticket."
    )


class FlagDeviceForReplacementArgs(BaseModel):
    device_id: str = Field(description="Target device_id within the active company.")
    reason: str = Field(description="Clear technical reason for flagging this device for replacement.")
    rationale: str = Field(description="Concrete justification grounded in telemetry findings.")
    evidence_ids: list[str] = Field(
        description="List of verified evidence_ids (e.g., ['E1']) supporting replacement."
    )


class NotifyEmployeeArgs(BaseModel):
    employee_id: str = Field(description="Target employee_id within the active company.")
    message: str = Field(description="Clear, helpful message to send to the employee.")
    rationale: str = Field(description="Concrete justification grounded in telemetry findings.")
    evidence_ids: list[str] = Field(
        description="List of verified evidence_ids (e.g., ['E1']) supporting this notification."
    )


# ---------------------------------------------------------------------------
# Tool runtime registry bound to a single TenantTelemetryRepository
# ---------------------------------------------------------------------------


def _citation_payload(
    *,
    source_tool: str,
    company_id: str,
    metric: str,
    value: Any,
    summary: str,
    device_id: str | None = None,
    employee_id: str | None = None,
    collected_at: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> dict[str, Any]:
    return {
        "source_tool": source_tool,
        "company_id": company_id,
        "device_id": device_id,
        "employee_id": employee_id,
        "collected_at": collected_at,
        "window_start": window_start,
        "window_end": window_end,
        "metric": metric,
        "value": value,
        "summary": summary,
    }


def build_tenant_tools(
    repo: TenantTelemetryRepository,
    *,
    max_sql_rows: int = 100,
) -> tuple[list[BaseTool], dict[str, Any]]:
    """Build LangChain StructuredTools and raw callables pinned to ``repo.company_id``.

    Returns ``(langchain_tools, callable_map)`` where ``callable_map`` returns
    raw structured dicts containing ``records`` and ``raw_citations`` so the
    graph's ``tool_executor`` node can mint canonical ``E#`` evidence IDs.
    """
    company_id = repo.company_id

    def _get_fleet_overview() -> dict[str, Any]:
        ov = repo.get_fleet_overview()
        cits = [
            _citation_payload(
                source_tool="get_fleet_overview",
                company_id=company_id,
                window_start=ov.get("earliest_snapshot"),
                window_end=ov.get("latest_snapshot"),
                metric="fleet_summary",
                value=f"{ov['device_count']} devices / {ov['snapshot_count']} snapshots",
                summary=(
                    f"Tenant {company_id}: {ov['device_count']} devices, "
                    f"{ov['employee_count']} employees, {ov['snapshot_count']} snapshots "
                    f"({ov.get('earliest_snapshot')} to {ov.get('latest_snapshot')})."
                ),
            )
        ]
        return {"status": "ok", "count": ov["device_count"], "overview": ov, "raw_citations": cits}

    def _list_devices(
        os_name: str | None = None,
        device_type: str | None = None,
        employee_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            rows = repo.list_latest_devices(
                os_name=os_name, device_type=device_type, employee_id=employee_id
            )
        except CrossTenantAccessError as exc:
            return {"status": "denied_cross_tenant", "error": str(exc), "records": [], "raw_citations": []}
        cits = [
            _citation_payload(
                source_tool="list_devices",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                collected_at=r["collected_at"],
                metric="device_status",
                value=f"{r['model_name']} | {r['os_product_name']} {r['os_product_version']}",
                summary=(
                    f"{r['device_id']} ({r['model_name']}, {r['form_factor']}, emp={r['employee_id']}) "
                    f"OS={r['os_product_name']} {r['os_product_version']}, "
                    f"RAM={r['memory_used_pct']}%, min_disk_free={r['min_disk_available_pct']}%"
                ),
            )
            for r in rows
        ]
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _get_device_telemetry(
        device_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        latest_only: bool = False,
    ) -> dict[str, Any]:
        try:
            rows = repo.get_device_history(
                device_id, start_date=start_date, end_date=end_date, latest_only=latest_only
            )
        except CrossTenantAccessError as exc:
            return {"status": "denied_cross_tenant", "error": str(exc), "records": [], "raw_citations": []}

        if not rows:
            return {
                "status": "ok",
                "count": 0,
                "records": [],
                "message": f"No telemetry records found for device '{device_id}' in tenant '{company_id}'.",
                "raw_citations": [],
            }

        latest = rows[-1]
        earliest = rows[0]
        disk_pct = latest["disk_volumes"][0]["available_pct"] if latest["disk_volumes"] else None
        failing = [c["check_id"] for c in latest["compliance_results"] if c["status"] == "fail"]
        cits = [
            _citation_payload(
                source_tool="get_device_telemetry",
                company_id=company_id,
                device_id=device_id,
                employee_id=latest["employee_id"],
                collected_at=latest["collected_at"] if latest_only else None,
                window_start=None if latest_only else earliest["collected_at"],
                window_end=None if latest_only else latest["collected_at"],
                metric="device_telemetry",
                value=(
                    f"snapshots={len(rows)}, RAM={latest['memory_used_pct']}%, "
                    f"disk_free={disk_pct}%, batt={latest['battery_condition']}, fails={failing}"
                ),
                summary=(
                    f"{device_id} ({latest['model_name']}, {latest['os_product_name']} {latest['os_product_version']}) "
                    f"latest={latest['collected_at']}: RAM={latest['memory_used_pct']}%, "
                    f"disk_free={disk_pct}%, battery={latest['battery_condition']} "
                    f"(cycles={latest['battery_cycle_count']}, fcc={latest['battery_full_charge_capacity']}), "
                    f"failing_checks={failing}"
                ),
            )
        ]
        # Keep response payload compact for the LLM if full 30-day history requested
        compact = [
            {
                "collected_at": r["collected_at"],
                "os": f"{r['os_product_name']} {r['os_product_version']}",
                "memory_used_pct": r["memory_used_pct"],
                "disk_available_pct": [v["available_pct"] for v in r["disk_volumes"]],
                "battery_condition": r["battery_condition"],
                "battery_cycle_count": r["battery_cycle_count"],
                "battery_full_charge_capacity": r["battery_full_charge_capacity"],
                "failing_checks": [
                    f"{c['check_id']}({c['severity']})"
                    for c in r["compliance_results"]
                    if c["status"] == "fail"
                ],
                "software_names": [s["name"] for s in r["installed_software"]],
            }
            for r in rows
        ]
        return {
            "status": "ok",
            "count": len(rows),
            "device_id": device_id,
            "employee_id": latest["employee_id"],
            "model_name": latest["model_name"],
            "records": compact,
            "raw_citations": cits,
        }

    def _query_low_storage_devices(
        threshold_free_pct: float = 15.0,
        as_of_latest: bool = True,
        device_type: str | None = None,
    ) -> dict[str, Any]:
        rows = repo.query_low_storage_devices(
            threshold_free_pct=threshold_free_pct,
            as_of_latest=as_of_latest,
            device_type=device_type,
        )
        cits = [
            _citation_payload(
                source_tool="query_low_storage_devices",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                collected_at=r["collected_at"],
                metric="disk_available_pct",
                value=f"{r['available_pct']}% ({r['available_gb']} GB / {r['size_gb']} GB)",
                summary=(
                    f"{r['device_id']} ({r['model_name']}, {r['form_factor']}) on {r['collected_at']}: "
                    f"{r['volume_name']} has {r['available_pct']}% free ({r['available_gb']} GB / {r['size_gb']} GB)."
                ),
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="query_low_storage_devices",
                    company_id=company_id,
                    metric="low_storage_count",
                    value=0,
                    summary=f"0 devices matched disk free space < {threshold_free_pct}% (device_type={device_type or 'all'}).",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _query_compliance_failures(
        severity: str | None = None,
        check_id: str | None = None,
        as_of_latest: bool = True,
        device_type: str | None = None,
    ) -> dict[str, Any]:
        rows = repo.query_compliance_failures(
            severity=severity,
            check_id=check_id,
            as_of_latest=as_of_latest,
            device_type=device_type,
        )
        cits = [
            _citation_payload(
                source_tool="query_compliance_failures",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                collected_at=r["collected_at"],
                metric=f"compliance:{r['check_id']}",
                value=f"fail (severity={r['severity']})",
                summary=(
                    f"{r['device_id']} ({r['model_name']}, {r['form_factor']}) on {r['collected_at']}: "
                    f"failed '{r['check_id']}' (severity={r['severity']})."
                ),
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="query_compliance_failures",
                    company_id=company_id,
                    metric="compliance_failure_count",
                    value=0,
                    summary=(
                        f"0 compliance failures found for tenant {company_id} "
                        f"(severity={severity or 'any'}, check_id={check_id or 'any'}, "
                        f"device_type={device_type or 'all'}, as_of_latest={as_of_latest})."
                    ),
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _query_os_versions(
        product_name: str | None = None,
        below_version: str | None = None,
        device_type: str | None = None,
    ) -> dict[str, Any]:
        rows = repo.query_os_versions(
            product_name=product_name, below_version=below_version, device_type=device_type
        )
        cits = [
            _citation_payload(
                source_tool="query_os_versions",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                collected_at=r["collected_at"],
                metric="os_version",
                value=f"{r['os_product_name']} {r['os_product_version']}",
                summary=(
                    f"{r['device_id']} ({r['model_name']}, {r['form_factor']}) on {r['collected_at']}: "
                    f"running {r['os_product_name']} {r['os_product_version']}."
                ),
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="query_os_versions",
                    company_id=company_id,
                    metric="os_version_match_count",
                    value=0,
                    summary=(
                        f"0 devices matched product_name={product_name or 'any'}, "
                        f"below_version={below_version or 'none'}."
                    ),
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _query_installed_software(
        software_name: str | None = None,
        as_of_latest: bool = True,
    ) -> dict[str, Any]:
        rows = repo.query_installed_software(
            software_name=software_name, as_of_latest=as_of_latest
        )
        cits = [
            _citation_payload(
                source_tool="query_installed_software",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                collected_at=r["collected_at"],
                metric="installed_software",
                value=f"{r['name']} {r.get('version') or ''}".strip(),
                summary=(
                    f"{r['device_id']} ({r['model_name']}) on {r['collected_at']}: "
                    f"software '{r['name']}' v{r.get('version')} ({r.get('publisher')})."
                ),
            )
            for r in rows[:40]
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="query_installed_software",
                    company_id=company_id,
                    metric="installed_software_count",
                    value=0,
                    summary=f"0 devices found with software matching '{software_name}'.",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows[:60], "raw_citations": cits}

    def _query_telemetry_sql(sql: str) -> dict[str, Any]:
        try:
            rows = repo.execute_readonly_sql(sql, max_rows=max_sql_rows)
        except UnsafeQueryError as exc:
            return {"status": "rejected_sql", "error": str(exc), "records": [], "raw_citations": []}

        cits: list[dict[str, Any]] = []
        for r in rows[:25]:
            cits.append(
                _citation_payload(
                    source_tool="query_telemetry_sql",
                    company_id=company_id,
                    device_id=r.get("device_id"),
                    employee_id=r.get("employee_id"),
                    collected_at=r.get("collected_at"),
                    metric="sql_query_result",
                    value=str(r)[:180],
                    summary=f"SQL row: {r}",
                )
            )
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="query_telemetry_sql",
                    company_id=company_id,
                    metric="sql_query_count",
                    value=0,
                    summary=f"Tenant-scoped SQL query returned 0 rows: {sql[:140]}",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _detect_battery_degradation(
        cycle_threshold: int = 800,
        min_capacity_drop_pct: float = 2.0,
        window_days: int | None = None,
    ) -> dict[str, Any]:
        rows = detect_battery_degradation(
            repo,
            cycle_threshold=cycle_threshold,
            min_capacity_drop_pct=min_capacity_drop_pct,
            window_days=window_days,
        )
        cits = [
            _citation_payload(
                source_tool="detect_battery_degradation",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                window_start=r["window_start"],
                window_end=r["window_end"],
                metric="battery_health_trend",
                value=(
                    f"cond={r['latest_condition']}, cycles={r['start_cycle_count']}->{r['latest_cycle_count']}, "
                    f"fcc={r['start_full_charge_capacity']}->{r['latest_full_charge_capacity']} (-{r['capacity_drop_pct']}%)"
                ),
                summary=f"{r['device_id']}: {r['finding']}",
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="detect_battery_degradation",
                    company_id=company_id,
                    metric="battery_degradation_count",
                    value=0,
                    summary=f"0 devices showed battery degradation for tenant {company_id}.",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _detect_ram_constraints(
        used_pct_threshold: float = 80.0,
        min_breach_ratio: float = 0.5,
        window_days: int | None = None,
    ) -> dict[str, Any]:
        rows = detect_ram_constraints(
            repo,
            used_pct_threshold=used_pct_threshold,
            min_breach_ratio=min_breach_ratio,
            window_days=window_days,
        )
        cits = [
            _citation_payload(
                source_tool="detect_ram_constraints",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                window_start=r["window_start"],
                window_end=r["window_end"],
                metric="ram_utilization_trend",
                value=f"avg={r['avg_memory_used_pct']}%, max={r['max_memory_used_pct']}% ({r['snapshots_above_threshold']}/{r['snapshot_count']})",
                summary=f"{r['device_id']} ({r['total_memory_label']}): {r['finding']}",
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="detect_ram_constraints",
                    company_id=company_id,
                    metric="ram_constrained_count",
                    value=0,
                    summary=f"0 devices showed sustained RAM pressure >= {used_pct_threshold}%.",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _detect_storage_trends(
        low_free_pct_threshold: float = 15.0,
        min_breach_ratio: float = 0.25,
        rapid_decline_slope_pct_per_day: float = -0.35,
        window_days: int | None = None,
    ) -> dict[str, Any]:
        rows = detect_storage_trends(
            repo,
            low_free_pct_threshold=low_free_pct_threshold,
            min_breach_ratio=min_breach_ratio,
            rapid_decline_slope_pct_per_day=rapid_decline_slope_pct_per_day,
            window_days=window_days,
        )
        cits = [
            _citation_payload(
                source_tool="detect_storage_trends",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                window_start=r["window_start"],
                window_end=r["window_end"],
                metric="storage_trend",
                value=f"{r['start_available_pct']}%->{r['latest_available_pct']}% (slope={r['slope_pct_per_day']}%/day)",
                summary=f"{r['device_id']}: {r['finding']}",
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="detect_storage_trends",
                    company_id=company_id,
                    metric="storage_trend_count",
                    value=0,
                    summary="0 devices showed persistently low or rapidly shrinking storage.",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _detect_compliance_drift(
        check_id: str | None = None,
        window_days: int | None = None,
    ) -> dict[str, Any]:
        rows = detect_compliance_drift(repo, check_id=check_id, window_days=window_days)
        cits = [
            _citation_payload(
                source_tool="detect_compliance_drift",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                window_start=r["window_start"],
                window_end=r["window_end"],
                metric=f"compliance_drift:{r['check_id']}",
                value=f"pass->fail on {r['first_regression_at']} (latest={r['latest_status']})",
                summary=f"{r['device_id']}: {r['finding']}",
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="detect_compliance_drift",
                    company_id=company_id,
                    metric="compliance_drift_count",
                    value=0,
                    summary=f"0 compliance drift regressions detected (check_id={check_id or 'any'}).",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _detect_software_anomalies(
        max_fleet_prevalence_ratio: float = 0.4,
        window_days: int | None = None,
    ) -> dict[str, Any]:
        rows = detect_software_anomalies(
            repo, max_fleet_prevalence_ratio=max_fleet_prevalence_ratio, window_days=window_days
        )
        cits = [
            _citation_payload(
                source_tool="detect_software_anomalies",
                company_id=company_id,
                device_id=r["device_id"],
                employee_id=r["employee_id"],
                window_start=r["first_seen_at"],
                window_end=r["last_seen_at"],
                metric="software_anomaly",
                value=f"{r['software_name']} v{r['version']} (prevalence={r['fleet_prevalence_ratio']})",
                summary=f"{r['device_id']}: {r['finding']}",
            )
            for r in rows
        ]
        if not rows:
            cits.append(
                _citation_payload(
                    source_tool="detect_software_anomalies",
                    company_id=company_id,
                    metric="software_anomaly_count",
                    value=0,
                    summary="0 rare or post-baseline software installations detected.",
                )
            )
        return {"status": "ok", "count": len(rows), "records": rows, "raw_citations": cits}

    def _run_fleet_insights_report(window_days: int | None = None) -> dict[str, Any]:
        insights = generate_fleet_insights(repo, window_days=window_days)
        cits: list[dict[str, Any]] = []
        for key, metric_name in (
            ("battery_degradation", "battery_health_trend"),
            ("ram_constraints", "ram_utilization_trend"),
            ("storage_trends", "storage_trend"),
            ("compliance_drift", "compliance_drift"),
            ("software_anomalies", "software_anomaly"),
        ):
            for r in insights[key]:
                cits.append(
                    _citation_payload(
                        source_tool="run_fleet_insights_report",
                        company_id=company_id,
                        device_id=r["device_id"],
                        employee_id=r.get("employee_id"),
                        window_start=r.get("window_start") or r.get("first_seen_at"),
                        window_end=r.get("window_end") or r.get("last_seen_at"),
                        metric=metric_name,
                        value=r["finding"][:120],
                        summary=f"{r['device_id']}: {r['finding']}",
                    )
                )
        total_findings = sum(
            len(insights[k])
            for k in (
                "battery_degradation",
                "ram_constraints",
                "storage_trends",
                "compliance_drift",
                "software_anomalies",
            )
        )
        return {
            "status": "ok",
            "count": total_findings,
            "insights": insights,
            "raw_citations": cits,
        }

    # Action tool placeholders (intercepted by graph tool_executor; never executed directly)
    def _action_placeholder(**kwargs: Any) -> dict[str, Any]:
        return {"status": "staged_for_approval", "args": kwargs}

    callable_map: dict[str, Any] = {
        "get_fleet_overview": _get_fleet_overview,
        "list_devices": _list_devices,
        "get_device_telemetry": _get_device_telemetry,
        "query_low_storage_devices": _query_low_storage_devices,
        "query_compliance_failures": _query_compliance_failures,
        "query_os_versions": _query_os_versions,
        "query_installed_software": _query_installed_software,
        "query_telemetry_sql": _query_telemetry_sql,
        "detect_battery_degradation": _detect_battery_degradation,
        "detect_ram_constraints": _detect_ram_constraints,
        "detect_storage_trends": _detect_storage_trends,
        "detect_compliance_drift": _detect_compliance_drift,
        "detect_software_anomalies": _detect_software_anomalies,
        "run_fleet_insights_report": _run_fleet_insights_report,
        "create_upgrade_order": _action_placeholder,
        "open_remediation_ticket": _action_placeholder,
        "flag_device_for_replacement": _action_placeholder,
        "notify_employee": _action_placeholder,
    }

    tools: list[BaseTool] = [
        StructuredTool.from_function(
            func=_get_fleet_overview,
            name="get_fleet_overview",
            description=(
                "Get dynamic overview of the active company's fleet: device count, employee count, "
                "date range, OS distribution, observed compliance checks, and observed software."
            ),
            args_schema=EmptyArgs,
        ),
        StructuredTool.from_function(
            func=_list_devices,
            name="list_devices",
            description="List latest telemetry status for devices in the active company, optionally filtered by OS, device_type ('laptop'/'desktop'), or employee_id.",
            args_schema=ListDevicesArgs,
        ),
        StructuredTool.from_function(
            func=_get_device_telemetry,
            name="get_device_telemetry",
            description="Retrieve latest snapshot or multi-day telemetry history for a specific device_id in the active company.",
            args_schema=GetDeviceTelemetryArgs,
        ),
        StructuredTool.from_function(
            func=_query_low_storage_devices,
            name="query_low_storage_devices",
            description="Find devices in the active company with disk free space below a percentage threshold.",
            args_schema=LowStorageArgs,
        ),
        StructuredTool.from_function(
            func=_query_compliance_failures,
            name="query_compliance_failures",
            description="Find compliance check failures in the active company, optionally filtered by severity ('high', 'medium', 'low'), check_id, or device_type.",
            args_schema=ComplianceFailuresArgs,
        ),
        StructuredTool.from_function(
            func=_query_os_versions,
            name="query_os_versions",
            description="Find devices running a given OS and/or strictly older than a target semantic version (e.g., product_name='macOS', below_version='15').",
            args_schema=OsVersionArgs,
        ),
        StructuredTool.from_function(
            func=_query_installed_software,
            name="query_installed_software",
            description="Search installed software across devices in the active company.",
            args_schema=InstalledSoftwareArgs,
        ),
        StructuredTool.from_function(
            func=_query_telemetry_sql,
            name="query_telemetry_sql",
            description=(
                "Execute a read-only SELECT/WITH SQL query against tenant-isolated views "
                "(v_snapshots, v_disk_volumes, v_compliance_results, v_installed_software, v_network_interfaces). "
                "Use for novel, ad-hoc, or unseen telemetry fields (including json_extract(raw_payload, '$.path'))."
            ),
            args_schema=ReadonlySqlArgs,
        ),
        StructuredTool.from_function(
            func=_detect_battery_degradation,
            name="detect_battery_degradation",
            description="Analyze time-series battery telemetry to detect end-of-life batteries, degraded conditions, high cycle counts, and declining full-charge capacity.",
            args_schema=BatteryTrendArgs,
        ),
        StructuredTool.from_function(
            func=_detect_ram_constraints,
            name="detect_ram_constraints",
            description="Analyze time-series memory telemetry to identify devices consistently constrained by high RAM usage.",
            args_schema=RamTrendArgs,
        ),
        StructuredTool.from_function(
            func=_detect_storage_trends,
            name="detect_storage_trends",
            description="Analyze time-series disk volume telemetry to detect persistently low storage or rapidly declining free space with depletion projections.",
            args_schema=StorageTrendArgs,
        ),
        StructuredTool.from_function(
            func=_detect_compliance_drift,
            name="detect_compliance_drift",
            description="Analyze time-series compliance telemetry to detect pass->fail policy regressions over time.",
            args_schema=ComplianceDriftArgs,
        ),
        StructuredTool.from_function(
            func=_detect_software_anomalies,
            name="detect_software_anomalies",
            description="Statistically detect rare/outlier or newly installed software across the fleet without relying on hardcoded blocklists.",
            args_schema=SoftwareAnomaliesArgs,
        ),
        StructuredTool.from_function(
            func=_run_fleet_insights_report,
            name="run_fleet_insights_report",
            description="Run all time-series trend and anomaly detectors (battery, RAM, storage, compliance drift, software anomalies) across the active company's fleet.",
            args_schema=FleetInsightsArgs,
        ),
        # Action Proposal Tools (require Policy Gate + HITL approval)
        StructuredTool.from_function(
            func=_action_placeholder,
            name="create_upgrade_order",
            description=(
                "Propose a hardware upgrade order for a device (e.g., RAM or SSD upgrade). "
                "Requires verified evidence_ids from earlier telemetry tool calls and explicit human approval."
            ),
            args_schema=CreateUpgradeOrderArgs,
        ),
        StructuredTool.from_function(
            func=_action_placeholder,
            name="open_remediation_ticket",
            description=(
                "Propose opening an IT remediation ticket for a compliance failure or health issue on a device. "
                "Requires verified evidence_ids and explicit human approval."
            ),
            args_schema=OpenRemediationTicketArgs,
        ),
        StructuredTool.from_function(
            func=_action_placeholder,
            name="flag_device_for_replacement",
            description=(
                "Propose flagging a device for hardware replacement (e.g., severe battery EOL or multi-factor degradation). "
                "Requires verified evidence_ids and explicit human approval."
            ),
            args_schema=FlagDeviceForReplacementArgs,
        ),
        StructuredTool.from_function(
            func=_action_placeholder,
            name="notify_employee",
            description=(
                "Propose sending a notification to an employee about their device issue or remediation steps. "
                "Requires verified evidence_ids and explicit human approval."
            ),
            args_schema=NotifyEmployeeArgs,
        ),
    ]

    return tools, callable_map
