"""Data-agnostic time-series trend and anomaly detection over tenant telemetry.

All detectors compute deterministic statistics (averages, slopes, state
transitions, persistence ratios) over whatever telemetry is currently stored
for the tenant, without hardcoding company names, device IDs, compliance check
IDs, or software blocklists.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from fleet_copilot.storage.repository import TenantTelemetryRepository


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _linear_slope_per_day(points: list[tuple[datetime, float]]) -> float:
    """Compute ordinary least-squares slope in units per day."""
    if len(points) < 2:
        return 0.0
    t0 = points[0][0]
    xs = [(dt - t0).total_seconds() / 86400.0 for dt, _ in points]
    ys = [val for _, val in points]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return round(numer / denom, 4)


def _cutoff_iso(repo: TenantTelemetryRepository, window_days: int | None) -> str | None:
    if window_days is None:
        return None
    overview = repo.get_fleet_overview()
    latest = overview.get("latest_snapshot")
    if not latest:
        return None
    cutoff = _parse_iso(latest) - timedelta(days=window_days)
    return cutoff.isoformat().replace("+00:00", "Z")


def detect_battery_degradation(
    repo: TenantTelemetryRepository,
    *,
    cycle_threshold: int = 800,
    min_capacity_drop_pct: float = 2.0,
    window_days: int | None = None,
) -> list[dict[str, Any]]:
    """Identify devices with batteries approaching end-of-life or degrading over time."""
    cutoff = _cutoff_iso(repo, window_days)
    with repo.db.transaction() as conn:
        sql = """
            SELECT device_id, employee_id, model_name, collected_at,
                   battery_present, battery_condition, battery_cycle_count,
                   battery_full_charge_capacity
            FROM snapshots
            WHERE company_id = ?
        """
        params: list[Any] = [repo.company_id]
        if cutoff:
            sql += " AND collected_at >= ?"
            params.append(cutoff)
        sql += " ORDER BY device_id ASC, collected_at ASC"
        rows = conn.execute(sql, params).fetchall()

    by_dev: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dev[r["device_id"]].append(dict(r))

    findings: list[dict[str, Any]] = []
    for device_id, recs in by_dev.items():
        batt_recs = [r for r in recs if r["battery_present"] == 1 or r["battery_condition"] is not None]
        if not batt_recs:
            continue  # Desktop or no battery telemetry at all

        missing_count = len(recs) - len(batt_recs)
        conditions = [r["battery_condition"] for r in batt_recs if r["battery_condition"]]
        latest_condition = conditions[-1] if conditions else "Unknown"
        non_normal_recs = [
            r for r in batt_recs if r["battery_condition"] and r["battery_condition"].lower() != "normal"
        ]

        cycles = [
            (r["collected_at"], int(r["battery_cycle_count"]))
            for r in batt_recs
            if r["battery_cycle_count"] is not None
        ]
        capacities = [
            (r["collected_at"], float(r["battery_full_charge_capacity"]))
            for r in batt_recs
            if r["battery_full_charge_capacity"] is not None
        ]

        start_cycle = cycles[0][1] if cycles else None
        latest_cycle = cycles[-1][1] if cycles else None

        start_cap = capacities[0][1] if capacities else None
        latest_cap = capacities[-1][1] if capacities else None
        cap_drop_pct = (
            round(((start_cap - latest_cap) / start_cap) * 100.0, 2)
            if (start_cap and latest_cap is not None and start_cap > 0)
            else 0.0
        )
        cap_slope_per_day = (
            _linear_slope_per_day([(_parse_iso(ts), val) for ts, val in capacities])
            if len(capacities) >= 2
            else 0.0
        )

        reasons: list[str] = []
        if non_normal_recs:
            first_degraded = non_normal_recs[0]["collected_at"]
            reasons.append(
                f"Battery condition degraded to '{latest_condition}' starting {first_degraded} "
                f"({len(non_normal_recs)}/{len(batt_recs)} battery snapshots)."
            )
        if latest_cycle is not None and latest_cycle >= cycle_threshold:
            reasons.append(
                f"High cycle count: {start_cycle} -> {latest_cycle} (threshold {cycle_threshold})."
            )
        if cap_drop_pct >= min_capacity_drop_pct:
            reasons.append(
                f"Full charge capacity declined by {cap_drop_pct}% "
                f"({int(start_cap)} -> {int(latest_cap)} mAh, slope {cap_slope_per_day} mAh/day)."
            )

        if reasons:
            findings.append(
                {
                    "device_id": device_id,
                    "company_id": repo.company_id,
                    "employee_id": recs[-1]["employee_id"],
                    "model_name": recs[-1]["model_name"],
                    "window_start": recs[0]["collected_at"],
                    "window_end": recs[-1]["collected_at"],
                    "snapshots_with_battery": len(batt_recs),
                    "snapshots_missing_battery": missing_count,
                    "latest_condition": latest_condition,
                    "first_degraded_at": non_normal_recs[0]["collected_at"] if non_normal_recs else None,
                    "start_cycle_count": start_cycle,
                    "latest_cycle_count": latest_cycle,
                    "start_full_charge_capacity": int(start_cap) if start_cap is not None else None,
                    "latest_full_charge_capacity": int(latest_cap) if latest_cap is not None else None,
                    "capacity_drop_pct": cap_drop_pct,
                    "capacity_slope_per_day": cap_slope_per_day,
                    "finding": " ".join(reasons),
                    "explanation": (
                        "Sustained battery wear and capacity loss indicate the battery is approaching "
                        "end-of-life and risks unexpected shutdowns or reduced mobility."
                    ),
                }
            )

    findings.sort(
        key=lambda x: (
            0 if (x["latest_condition"] or "").lower() != "normal" else 1,
            -(x["capacity_drop_pct"] or 0.0),
            -(x["latest_cycle_count"] or 0),
        )
    )
    return findings


def detect_ram_constraints(
    repo: TenantTelemetryRepository,
    *,
    used_pct_threshold: float = 80.0,
    min_breach_ratio: float = 0.5,
    window_days: int | None = None,
) -> list[dict[str, Any]]:
    """Identify devices consistently constrained by high RAM utilization."""
    cutoff = _cutoff_iso(repo, window_days)
    with repo.db.transaction() as conn:
        sql = """
            SELECT device_id, employee_id, model_name, total_memory_label,
                   total_memory_bytes, memory_used_pct, collected_at
            FROM snapshots
            WHERE company_id = ? AND memory_used_pct IS NOT NULL
        """
        params: list[Any] = [repo.company_id]
        if cutoff:
            sql += " AND collected_at >= ?"
            params.append(cutoff)
        sql += " ORDER BY device_id ASC, collected_at ASC"
        rows = conn.execute(sql, params).fetchall()

    by_dev: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dev[r["device_id"]].append(dict(r))

    findings: list[dict[str, Any]] = []
    for device_id, recs in by_dev.items():
        pcts = [float(r["memory_used_pct"]) for r in recs]
        avg_pct = round(sum(pcts) / len(pcts), 2)
        min_pct = round(min(pcts), 2)
        max_pct = round(max(pcts), 2)
        breaches = sum(1 for p in pcts if p >= used_pct_threshold)
        breach_ratio = round(breaches / len(pcts), 3)
        slope = _linear_slope_per_day(
            [(_parse_iso(r["collected_at"]), float(r["memory_used_pct"])) for r in recs]
        )

        if avg_pct >= used_pct_threshold or breach_ratio >= min_breach_ratio:
            findings.append(
                {
                    "device_id": device_id,
                    "company_id": repo.company_id,
                    "employee_id": recs[-1]["employee_id"],
                    "model_name": recs[-1]["model_name"],
                    "total_memory_label": recs[-1]["total_memory_label"],
                    "window_start": recs[0]["collected_at"],
                    "window_end": recs[-1]["collected_at"],
                    "snapshot_count": len(recs),
                    "avg_memory_used_pct": avg_pct,
                    "min_memory_used_pct": min_pct,
                    "max_memory_used_pct": max_pct,
                    "snapshots_above_threshold": breaches,
                    "breach_ratio": breach_ratio,
                    "slope_pct_per_day": slope,
                    "finding": (
                        f"Sustained high RAM utilization averaging {avg_pct}% "
                        f"(min {min_pct}%, max {max_pct}%), exceeding {used_pct_threshold}% in "
                        f"{breaches}/{len(recs)} snapshots ({round(breach_ratio * 100, 1)}%)."
                    ),
                    "explanation": (
                        "Persistent memory pressure causes heavy swapping and degrades user "
                        "productivity; evaluate workload or RAM upgrade/replacement."
                    ),
                }
            )

    findings.sort(key=lambda x: -x["avg_memory_used_pct"])
    return findings


def detect_storage_trends(
    repo: TenantTelemetryRepository,
    *,
    low_free_pct_threshold: float = 15.0,
    min_breach_ratio: float = 0.25,
    rapid_decline_slope_pct_per_day: float = -0.35,
    window_days: int | None = None,
) -> list[dict[str, Any]]:
    """Identify devices consistently low on storage or rapidly exhausting disk space."""
    cutoff = _cutoff_iso(repo, window_days)
    with repo.db.transaction() as conn:
        sql = """
            SELECT d.device_id, s.employee_id, s.model_name, d.volume_name, d.mount_point,
                   d.size_bytes, d.available_bytes, d.available_pct, d.collected_at
            FROM disk_volumes d
            INNER JOIN snapshots s ON d.snapshot_id = s.id
            WHERE d.company_id = ?
        """
        params: list[Any] = [repo.company_id]
        if cutoff:
            sql += " AND d.collected_at >= ?"
            params.append(cutoff)
        sql += " ORDER BY d.device_id ASC, d.mount_point ASC, d.collected_at ASC"
        rows = conn.execute(sql, params).fetchall()

    by_vol: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_vol[(r["device_id"], r["mount_point"])].append(dict(r))

    findings: list[dict[str, Any]] = []
    for (device_id, mount_point), recs in by_vol.items():
        pcts = [float(r["available_pct"]) for r in recs]
        start_pct = round(pcts[0], 2)
        latest_pct = round(pcts[-1], 2)
        min_pct = round(min(pcts), 2)
        avg_pct = round(sum(pcts) / len(pcts), 2)
        low_count = sum(1 for p in pcts if p < low_free_pct_threshold)
        low_ratio = round(low_count / len(pcts), 3)
        slope = _linear_slope_per_day(
            [(_parse_iso(r["collected_at"]), float(r["available_pct"])) for r in recs]
        )

        projected_days_to_zero: float | None = None
        if slope < -0.05 and latest_pct > 0:
            projected_days_to_zero = round(latest_pct / abs(slope), 1)

        is_consistently_low = low_ratio >= min_breach_ratio or latest_pct < low_free_pct_threshold
        is_rapidly_declining = slope <= rapid_decline_slope_pct_per_day

        if is_consistently_low or is_rapidly_declining:
            latest_gb = round(recs[-1]["available_bytes"] / (1024**3), 2)
            size_gb = round(recs[-1]["size_bytes"] / (1024**3), 2)
            parts = [
                f"Free space on {recs[-1]['volume_name']} ({mount_point}) moved from "
                f"{start_pct}% to {latest_pct}% ({latest_gb} GB / {size_gb} GB free), "
                f"below {low_free_pct_threshold}% in {low_count}/{len(recs)} snapshots "
                f"(slope {slope}%/day)."
            ]
            if projected_days_to_zero is not None and projected_days_to_zero <= 60:
                parts.append(f"At current consumption rate, disk reaches 0% in ~{projected_days_to_zero} days.")
            findings.append(
                {
                    "device_id": device_id,
                    "company_id": repo.company_id,
                    "employee_id": recs[-1]["employee_id"],
                    "model_name": recs[-1]["model_name"],
                    "volume_name": recs[-1]["volume_name"],
                    "mount_point": mount_point,
                    "window_start": recs[0]["collected_at"],
                    "window_end": recs[-1]["collected_at"],
                    "snapshot_count": len(recs),
                    "start_available_pct": start_pct,
                    "latest_available_pct": latest_pct,
                    "min_available_pct": min_pct,
                    "avg_available_pct": avg_pct,
                    "latest_available_gb": latest_gb,
                    "size_gb": size_gb,
                    "snapshots_below_threshold": low_count,
                    "low_ratio": low_ratio,
                    "slope_pct_per_day": slope,
                    "projected_days_to_zero": projected_days_to_zero,
                    "finding": " ".join(parts),
                    "explanation": (
                        "Critically low or steadily shrinking disk space impairs OS updates, "
                        "virtual memory paging, and application stability."
                    ),
                }
            )

    findings.sort(key=lambda x: (x["latest_available_pct"], x["slope_pct_per_day"]))
    return findings


def detect_compliance_drift(
    repo: TenantTelemetryRepository,
    *,
    check_id: str | None = None,
    window_days: int | None = None,
) -> list[dict[str, Any]]:
    """Detect devices where compliance checks regressed (pass -> fail) over time."""
    cutoff = _cutoff_iso(repo, window_days)
    with repo.db.transaction() as conn:
        sql = """
            SELECT c.device_id, s.employee_id, s.model_name,
                   c.check_id, c.severity, c.status, c.collected_at
            FROM compliance_results c
            INNER JOIN snapshots s ON c.snapshot_id = s.id
            WHERE c.company_id = ?
        """
        params: list[Any] = [repo.company_id]
        if cutoff:
            sql += " AND c.collected_at >= ?"
            params.append(cutoff)
        if check_id:
            sql += " AND LOWER(c.check_id) = ?"
            params.append(check_id.lower())
        sql += " ORDER BY c.device_id ASC, c.check_id ASC, c.collected_at ASC"
        rows = conn.execute(sql, params).fetchall()

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["device_id"], r["check_id"])].append(dict(r))

    findings: list[dict[str, Any]] = []
    for (device_id, cid), recs in by_key.items():
        if len(recs) < 2:
            continue
        statuses = [r["status"].lower() for r in recs]
        # Detect pass -> fail regression transitions
        regressions: list[str] = []
        for prev, curr in zip(recs, recs[1:], strict=False):
            if prev["status"].lower() == "pass" and curr["status"].lower() == "fail":
                regressions.append(curr["collected_at"])

        if regressions:
            fail_count = sum(1 for st in statuses if st == "fail")
            findings.append(
                {
                    "device_id": device_id,
                    "company_id": repo.company_id,
                    "employee_id": recs[-1]["employee_id"],
                    "model_name": recs[-1]["model_name"],
                    "check_id": cid,
                    "severity": recs[-1]["severity"],
                    "window_start": recs[0]["collected_at"],
                    "window_end": recs[-1]["collected_at"],
                    "initial_status": statuses[0],
                    "latest_status": statuses[-1],
                    "first_regression_at": regressions[0],
                    "regression_count": len(regressions),
                    "failed_snapshots": fail_count,
                    "total_snapshots": len(recs),
                    "finding": (
                        f"Compliance check '{cid}' ({recs[-1]['severity']} severity) regressed "
                        f"from 'pass' to 'fail' on {regressions[0]} "
                        f"(latest status: '{statuses[-1]}', failed in {fail_count}/{len(recs)} snapshots)."
                    ),
                    "explanation": (
                        "Post-enrollment policy drift indicates local configuration changes or "
                        "MDM profile desynchronization that should be remediated."
                    ),
                }
            )

    findings.sort(key=lambda x: (x["severity"] != "high", x["first_regression_at"], x["device_id"]))
    return findings


def detect_software_anomalies(
    repo: TenantTelemetryRepository,
    *,
    max_fleet_prevalence_ratio: float = 0.4,
    window_days: int | None = None,
) -> list[dict[str, Any]]:
    """Detect rare/outlier or newly appearing software installations purely statistically."""
    cutoff = _cutoff_iso(repo, window_days)
    with repo.db.transaction() as conn:
        total_dev_row = conn.execute(
            "SELECT COUNT(DISTINCT device_id) AS cnt FROM snapshots WHERE company_id = ?",
            (repo.company_id,),
        ).fetchone()
        total_devices = int(total_dev_row["cnt"]) if total_dev_row else 0
        if total_devices == 0:
            return []

        sql = """
            SELECT sw.device_id, s.employee_id, s.model_name,
                   sw.name, sw.version, sw.publisher,
                   MIN(sw.collected_at) AS first_seen_at,
                   MAX(sw.collected_at) AS last_seen_at,
                   COUNT(*) AS snapshot_count
            FROM installed_software sw
            INNER JOIN snapshots s ON sw.snapshot_id = s.id
            WHERE sw.company_id = ?
        """
        params: list[Any] = [repo.company_id]
        if cutoff:
            sql += " AND sw.collected_at >= ?"
            params.append(cutoff)
        sql += " GROUP BY sw.device_id, sw.name"
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

        # Also check each device's earliest snapshot to detect software added AFTER baseline
        first_snap_rows = conn.execute(
            "SELECT device_id, MIN(collected_at) AS first_ts FROM snapshots WHERE company_id = ? GROUP BY device_id",
            (repo.company_id,),
        ).fetchall()
        device_first_ts = {r["device_id"]: r["first_ts"] for r in first_snap_rows}

    by_sw_devices: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_sw_devices[r["name"]].add(r["device_id"])

    findings: list[dict[str, Any]] = []
    for r in rows:
        prevalence = round(len(by_sw_devices[r["name"]]) / total_devices, 3)
        installed_after_baseline = r["first_seen_at"] > device_first_ts.get(r["device_id"], "")
        if prevalence <= max_fleet_prevalence_ratio or installed_after_baseline:
            reasons: list[str] = []
            if prevalence <= max_fleet_prevalence_ratio:
                reasons.append(
                    f"low fleet prevalence ({len(by_sw_devices[r['name']])}/{total_devices} devices)"
                )
            if installed_after_baseline:
                reasons.append(f"appeared after device baseline on {r['first_seen_at']}")
            findings.append(
                {
                    "device_id": r["device_id"],
                    "company_id": repo.company_id,
                    "employee_id": r["employee_id"],
                    "model_name": r["model_name"],
                    "software_name": r["name"],
                    "version": r["version"],
                    "publisher": r["publisher"],
                    "first_seen_at": r["first_seen_at"],
                    "last_seen_at": r["last_seen_at"],
                    "snapshot_count": r["snapshot_count"],
                    "fleet_prevalence_ratio": prevalence,
                    "installed_after_baseline": installed_after_baseline,
                    "finding": f"Software '{r['name']}' (v{r['version']}) flagged due to " + " and ".join(reasons) + ".",
                    "explanation": (
                        "Non-standard or newly introduced software outside the fleet baseline "
                        "may represent shadow IT or policy violations requiring review."
                    ),
                }
            )

    findings.sort(key=lambda x: (x["fleet_prevalence_ratio"], x["first_seen_at"]))
    return findings


def generate_fleet_insights(
    repo: TenantTelemetryRepository,
    *,
    window_days: int | None = None,
) -> dict[str, Any]:
    """Run all generalizable trend detectors for the active tenant."""
    return {
        "company_id": repo.company_id,
        "overview": repo.get_fleet_overview(),
        "battery_degradation": detect_battery_degradation(repo, window_days=window_days),
        "ram_constraints": detect_ram_constraints(repo, window_days=window_days),
        "storage_trends": detect_storage_trends(repo, window_days=window_days),
        "compliance_drift": detect_compliance_drift(repo, window_days=window_days),
        "software_anomalies": detect_software_anomalies(repo, window_days=window_days),
    }
