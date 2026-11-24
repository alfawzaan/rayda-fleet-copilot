"""Tests for generalizable time-series trend analytics on real and synthetic datasets."""

from __future__ import annotations

from pathlib import Path

from fleet_copilot.analytics.trends import (
    detect_battery_degradation,
    detect_compliance_drift,
    detect_ram_constraints,
    detect_software_anomalies,
    detect_storage_trends,
    generate_fleet_insights,
)
from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.ingestion import ingest_snapshot
from fleet_copilot.storage.repository import TenantTelemetryRepository


def test_real_dataset_analytics_acme(populated_db: FleetDatabase) -> None:
    repo = TenantTelemetryRepository(populated_db, "acme-001")

    # Zero high-severity compliance failures in the supplied dataset
    assert repo.query_compliance_failures(severity="high") == []

    # Battery degradation detects the 3 Acme devices transitioning to Service Recommended
    batt = detect_battery_degradation(repo)
    assert {b["device_id"] for b in batt} == {"7VP16KHV88LM", "8NM23J95R5I6", "M4XVHUV1MEPZ"}
    assert all(b["capacity_drop_pct"] > 3.0 for b in batt)

    # Sustained RAM pressure in Acme
    ram = detect_ram_constraints(repo)
    assert {r["device_id"] for r in ram} == {"M4XVHUV1MEPZ"}
    assert ram[0]["avg_memory_used_pct"] > 90.0

    # Low & declining disk space in Acme
    disk = detect_storage_trends(repo)
    assert {d["device_id"] for d in disk} == {"8NM23J95R5I6", "MT7PJB7N5LRE"}
    assert all(d["latest_available_pct"] < 5.0 for d in disk)

    # Compliance drift (screen_lock pass -> fail)
    drift = detect_compliance_drift(repo)
    assert len(drift) >= 4
    assert all(d["check_id"] == "screen_lock" for d in drift)

    # Statistical software anomalies (no hardcoded blocklist)
    sw = detect_software_anomalies(repo)
    assert len(sw) > 0
    flagged_names = {s["software_name"] for s in sw}
    assert flagged_names.issubset({"uTorrent", "TeamViewer", "CleanMyMac X"})

    # Combined fleet report
    report = generate_fleet_insights(repo)
    assert report["company_id"] == "acme-001"
    assert report["overview"]["device_count"] == 10


def test_analytics_generalize_to_unseen_synthetic_dataset(tmp_path: Path) -> None:
    """Verify trend detectors work on completely unseen check_ids, OSes, and rapid disk decline."""
    db = FleetDatabase(tmp_path / "synthetic.db")

    # Feed 5 days of synthetic telemetry for an unseen company & unseen compliance check
    for day in range(1, 6):
        ingest_snapshot(
            db,
            {
                "device_id": "SYN-01",
                "company_id": "nova-500",
                "employee_id": "emp-nova-1",
                "collected_at": f"2026-08-0{day}T10:00:00Z",
                "os": {"platform": "linux", "product_name": "Ubuntu", "product_version": "24.04"},
                "device_identity": {"model_name": "Framework Laptop 16", "total_memory": "16 GB"},
                "memory": {
                    "total_memory_bytes": 16_000_000_000,
                    "used_memory_bytes": 14_400_000_000,  # 90% every day
                },
                "disk_volumes": [
                    {
                        "volume_name": "nvme0",
                        "mount_point": "/",
                        "size_bytes": 100_000_000_000,
                        # Rapidly losing 5% per day: 40% -> 35% -> 30% -> 25% -> 20%
                        "available_bytes": (45 - 5 * day) * 1_000_000_000,
                    }
                ],
                "compliance_results": [
                    {
                        "check_id": "kernel_livepatch_enabled",
                        "status": "pass" if day <= 2 else "fail",
                        "severity": "high",
                    }
                ],
            },
        )

    repo = TenantTelemetryRepository(db, "nova-500")
    # Even though disk free is still 20% (> 15%), rapid slope (-5%/day) triggers alert + projection!
    disk = detect_storage_trends(repo)
    assert len(disk) == 1
    assert disk[0]["slope_pct_per_day"] <= -4.9
    assert disk[0]["projected_days_to_zero"] == 4.0

    # Unseen compliance check drift detected automatically
    drift = detect_compliance_drift(repo)
    assert len(drift) == 1
    assert drift[0]["check_id"] == "kernel_livepatch_enabled"
    assert drift[0]["severity"] == "high"

    # RAM constraint detected automatically
    ram = detect_ram_constraints(repo)
    assert len(ram) == 1
    assert ram[0]["avg_memory_used_pct"] == 90.0
