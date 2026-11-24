"""SQLite database initialization and connection management."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    employee_id TEXT,
    collected_at TEXT NOT NULL,
    agent_version TEXT,

    -- OS metadata
    os_platform TEXT NOT NULL,
    os_product_name TEXT NOT NULL,
    os_product_version TEXT NOT NULL,
    os_build_version TEXT,
    os_architecture TEXT,
    os_kernel_name TEXT,
    os_kernel_release TEXT,
    hostname TEXT,

    -- Device identity
    serial_number TEXT,
    model_name TEXT NOT NULL,
    model_identifier TEXT,
    processor TEXT,
    hardware_uuid TEXT,
    total_memory_label TEXT,

    -- Memory metrics
    ram_bytes INTEGER,
    total_memory_bytes INTEGER,
    used_memory_bytes INTEGER,
    free_memory_bytes INTEGER,
    memory_used_pct REAL,

    -- Battery metrics (nullable for desktops or missing telemetry)
    battery_present INTEGER,
    battery_charging_status TEXT,
    battery_percentage REAL,
    battery_condition TEXT,
    battery_cycle_count INTEGER,
    battery_full_charge_capacity INTEGER,

    -- Complete JSON payload for forward-compatible querying of unseen fields
    raw_payload TEXT NOT NULL,
    ingested_at TEXT NOT NULL,

    UNIQUE(device_id, collected_at)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_company_time
    ON snapshots(company_id, collected_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_company_device_time
    ON snapshots(company_id, device_id, collected_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_company_employee
    ON snapshots(company_id, employee_id);

CREATE TABLE IF NOT EXISTS disk_volumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    volume_name TEXT NOT NULL,
    file_system TEXT,
    mount_point TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    available_bytes INTEGER NOT NULL,
    available_pct REAL NOT NULL,
    encrypted INTEGER
);

CREATE INDEX IF NOT EXISTS idx_disk_company_device_time
    ON disk_volumes(company_id, device_id, collected_at);

CREATE TABLE IF NOT EXISTS compliance_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    check_id TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_compliance_company_device_time
    ON compliance_results(company_id, device_id, collected_at);
CREATE INDEX IF NOT EXISTS idx_compliance_company_status_sev
    ON compliance_results(company_id, status, severity);

CREATE TABLE IF NOT EXISTS installed_software (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT,
    publisher TEXT
);

CREATE INDEX IF NOT EXISTS idx_software_company_name
    ON installed_software(company_id, name);
CREATE INDEX IF NOT EXISTS idx_software_company_device_time
    ON installed_software(company_id, device_id, collected_at);

CREATE TABLE IF NOT EXISTS network_interfaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    address TEXT,
    family TEXT,
    interface_name TEXT,
    internal INTEGER,
    mac TEXT
);

CREATE INDEX IF NOT EXISTS idx_network_company_device
    ON network_interfaces(company_id, device_id);

-- Real transactional ITSM operational action tables
CREATE TABLE IF NOT EXISTS executed_actions (
    action_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    company_id TEXT NOT NULL,
    target_device_id TEXT,
    target_employee_id TEXT,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL,
    approved_by TEXT,
    approval_comment TEXT,
    evidence_ids_json TEXT NOT NULL,
    external_dispatch_json TEXT,
    idempotency_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_idempotency
    ON executed_actions(company_id, idempotency_key)
    WHERE status = 'executed';

-- Immutable audit log
CREATE TABLE IF NOT EXISTS audit_logs (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    target_entity TEXT,
    details_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_company_time
    ON audit_logs(company_id, timestamp DESC);
"""


class FleetDatabase:
    """Thread-safe SQLite database wrapper for telemetry, actions, and audit logs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.transaction() as conn:
            conn.executescript(SCHEMA_SQL)

    def list_companies(self) -> list[str]:
        """Dynamically discover all company_ids currently present in telemetry."""
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT DISTINCT company_id FROM snapshots ORDER BY company_id"
            ).fetchall()
            return [row["company_id"] for row in rows]

    def get_snapshot_count(self) -> int:
        with self.transaction() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM snapshots").fetchone()
            return int(row["cnt"]) if row else 0

    def get_entity_owner(
        self, *, device_id: str | None = None, employee_id: str | None = None
    ) -> str | None:
        """Internal governance lookup: determine which company owns a device or employee.

        Used ONLY by the policy/security layer to distinguish cross-tenant access
        attempts from genuinely nonexistent IDs. Never exposed directly to the LLM.
        """
        with self.transaction() as conn:
            if device_id:
                row = conn.execute(
                    "SELECT company_id FROM snapshots WHERE device_id = ? LIMIT 1",
                    (device_id,),
                ).fetchone()
                if row:
                    return str(row["company_id"])
            if employee_id:
                row = conn.execute(
                    "SELECT company_id FROM snapshots WHERE employee_id = ? LIMIT 1",
                    (employee_id,),
                ).fetchone()
                if row:
                    return str(row["company_id"])
        return None
