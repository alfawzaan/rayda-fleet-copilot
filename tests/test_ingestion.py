"""Tests for batch & realtime incremental telemetry ingestion and unseen-schema tolerance."""

from __future__ import annotations

import json
from pathlib import Path

from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.ingestion import ingest_ndjson_file, ingest_ndjson_text, ingest_snapshot
from fleet_copilot.storage.repository import TenantTelemetryRepository
from tests.conftest import DATASET_PATH


def test_batch_ingestion_counts_and_missing_batteries(populated_db: FleetDatabase) -> None:
    assert populated_db.get_snapshot_count() == 750
    assert populated_db.list_companies() == ["acme-001", "globex-002", "initech-003"]

    with populated_db.transaction() as conn:
        dev_count = conn.execute("SELECT COUNT(DISTINCT device_id) FROM snapshots").fetchone()[0]
        emp_count = conn.execute("SELECT COUNT(DISTINCT employee_id) FROM snapshots").fetchone()[0]
        null_batt = conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE battery_present IS NULL"
        ).fetchone()[0]

    assert dev_count == 25
    assert emp_count == 14
    # 90 Mac mini snapshots + 10 noisy laptop snapshots missing battery telemetry
    assert null_batt == 100


def test_idempotent_reingestion_does_not_duplicate_rows(tmp_path: Path) -> None:
    db = FleetDatabase(tmp_path / "idem.db")
    r1 = ingest_ndjson_file(db, DATASET_PATH)
    assert r1.inserted == 750
    assert r1.updated == 0

    r2 = ingest_ndjson_file(db, DATASET_PATH)
    assert r2.inserted == 0
    assert r2.updated == 750
    assert db.get_snapshot_count() == 750

    with db.transaction() as conn:
        disk_rows = conn.execute("SELECT COUNT(*) FROM disk_volumes").fetchone()[0]
        comp_rows = conn.execute("SELECT COUNT(*) FROM compliance_results").fetchone()[0]
    assert disk_rows == 750
    assert comp_rows == 750 * 3


def test_realtime_single_snapshot_with_unseen_company_and_extra_fields(
    populated_db: FleetDatabase,
) -> None:
    """Ensure ingestion generalizes to unseen companies, checks, OSes, and extra JSON fields."""
    live_payload = {
        "device_id": "ZEN-LINUX-01",
        "company_id": "zenith-999",
        "employee_id": "emp-zenith-01",
        "collected_at": "2026-07-01T12:00:00Z",
        "agent_version": "44.0.0",
        "os": {
            "platform": "linux",
            "product_name": "Ubuntu",
            "product_version": "24.04",
            "hostname": "zen-workstation-1",
        },
        "device_identity": {
            "model_name": "ThinkPad P16",
            "total_memory": "64 GB",
        },
        "memory": {
            "total_memory_bytes": 68719476736,
            "used_memory_bytes": 34359738368,
            "free_memory_bytes": 34359738368,
        },
        "disk_volumes": [
            {
                "volume_name": "root",
                "mount_point": "/",
                "size_bytes": 1000000000000,
                "available_bytes": 80000000000,
                "encrypted": False,
            }
        ],
        "compliance_results": [
            {"check_id": "edr_sensor_running", "status": "fail", "severity": "high"}
        ],
        "custom_telemetry_ext": {"gpu_temp_c": 91.5, "office_region": "ap-south-1"},
    }

    report = ingest_snapshot(populated_db, live_payload)
    assert report.inserted == 1
    assert not report.errors
    assert "zenith-999" in populated_db.list_companies()

    repo = TenantTelemetryRepository(populated_db, "zenith-999")
    high_fails = repo.query_compliance_failures(severity="high")
    assert len(high_fails) == 1
    assert high_fails[0]["check_id"] == "edr_sensor_running"

    # Verify unseen JSON attribute is immediately queryable via tenant-isolated SQL
    rows = repo.execute_readonly_sql(
        "SELECT device_id, json_extract(raw_payload, '$.custom_telemetry_ext.office_region') AS region "
        "FROM v_snapshots"
    )
    assert rows == [{"device_id": "ZEN-LINUX-01", "region": "ap-south-1"}]


def test_ingest_pretty_printed_single_json_and_array(populated_db: FleetDatabase) -> None:
    """Verify multi-line pretty-printed single JSON objects (from Streamlit UI) ingest cleanly."""
    payload = {
        "device_id": "LIVE-DEV-001",
        "company_id": "initech-003",
        "employee_id": "emp-initech-003-999",
        "collected_at": "2026-11-24T08:11:04.289345Z",
        "agent_version": "43.1.0",
        "os": {
            "platform": "darwin",
            "product_name": "macOS",
            "product_version": "15.4",
            "hostname": "initech-003-live-laptop-1",
        },
        "device_identity": {
            "serial_number": "LIVE-DEV-001",
            "model_name": "MacBook Pro",
            "processor": "Apple M4 Pro",
            "total_memory": "32 GB",
        },
        "memory": {
            "total_memory_bytes": 34359738368,
            "used_memory_bytes": 30000000000,
            "free_memory_bytes": 4359738368,
        },
        "disk_volumes": [
            {
                "volume_name": "Macintosh HD",
                "mount_point": "/",
                "size_bytes": 549755813888,
                "available_bytes": 21990232555,
                "encrypted": False,
            }
        ],
        "compliance_results": [
            {"check_id": "disk_encryption", "status": "fail", "severity": "high"},
            {"check_id": "edr_sensor_active", "status": "fail", "severity": "high"},
        ],
        "custom_telemetry_ext": {"gpu_temp_c": 88.4, "office_region": "eu-west-1"},
    }

    pretty_text = json.dumps(payload, indent=2)
    report = ingest_ndjson_text(populated_db, pretty_text)
    assert report.processed == 1
    assert report.inserted == 1
    assert not report.errors

    # Also verify pretty-printed JSON array
    arr_report = ingest_ndjson_text(populated_db, json.dumps([payload], indent=2))
    assert arr_report.processed == 1
    assert arr_report.updated == 1
    assert not arr_report.errors
