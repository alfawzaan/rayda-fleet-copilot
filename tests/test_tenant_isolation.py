"""Security tests verifying strict code-level & SQLite-authorizer tenant isolation."""

from __future__ import annotations

import pytest

from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.repository import (
    CrossTenantAccessError,
    TenantTelemetryRepository,
    UnsafeQueryError,
)


def test_repository_only_returns_active_tenant_devices(populated_db: FleetDatabase) -> None:
    acme_repo = TenantTelemetryRepository(populated_db, "acme-001")
    globex_repo = TenantTelemetryRepository(populated_db, "globex-002")

    acme_devices = acme_repo.list_latest_devices()
    globex_devices = globex_repo.list_latest_devices()

    assert len(acme_devices) == 10
    assert len(globex_devices) == 8
    assert all(d["company_id"] == "acme-001" for d in acme_devices)
    assert all(d["company_id"] == "globex-002" for d in globex_devices)
    assert {d["device_id"] for d in acme_devices}.isdisjoint(
        {d["device_id"] for d in globex_devices}
    )


def test_cross_tenant_device_and_employee_lookup_raises(populated_db: FleetDatabase) -> None:
    acme_repo = TenantTelemetryRepository(populated_db, "acme-001")
    globex_repo = TenantTelemetryRepository(populated_db, "globex-002")
    globex_dev = globex_repo.list_latest_devices()[0]

    with pytest.raises(CrossTenantAccessError):
        acme_repo.get_device_history(globex_dev["device_id"])

    with pytest.raises(CrossTenantAccessError):
        acme_repo.list_latest_devices(employee_id=globex_dev["employee_id"])

    # Nonexistent ID should return empty cleanly without leaking error
    assert acme_repo.get_device_history("NONEXISTENT-XYZ") == []


def test_sql_authorizer_enforces_tenant_views_and_blocks_base_tables(
    populated_db: FleetDatabase,
) -> None:
    acme_repo = TenantTelemetryRepository(populated_db, "acme-001")

    # Allowed: SELECT against tenant-filtered view returns only acme-001
    rows = acme_repo.execute_readonly_sql(
        "SELECT DISTINCT company_id, COUNT(DISTINCT device_id) AS dev_cnt FROM v_snapshots GROUP BY company_id"
    )
    assert rows == [{"company_id": "acme-001", "dev_cnt": 10}]

    # Blocked: attempting to query base table directly for another company
    with pytest.raises(UnsafeQueryError):
        acme_repo.execute_readonly_sql("SELECT * FROM snapshots WHERE company_id = 'globex-002'")

    # Blocked: attempting to read audit_logs or executed_actions directly
    with pytest.raises(UnsafeQueryError):
        acme_repo.execute_readonly_sql("SELECT * FROM audit_logs")

    # Blocked: write / DDL / stacked statements / PRAGMA
    for bad_sql in (
        "DELETE FROM v_snapshots",
        "DROP TABLE snapshots",
        "PRAGMA table_info(snapshots)",
        "SELECT * FROM v_snapshots; SELECT * FROM snapshots",
        "INSERT INTO snapshots (device_id) VALUES ('evil')",
    ):
        with pytest.raises(UnsafeQueryError):
            acme_repo.execute_readonly_sql(bad_sql)
