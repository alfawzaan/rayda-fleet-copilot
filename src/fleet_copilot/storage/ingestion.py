"""Batch and realtime incremental telemetry ingestion into SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from fleet_copilot.models.telemetry import TelemetrySnapshot
from fleet_copilot.storage.database import FleetDatabase


class IngestionReport(BaseModel):
    """Summary of a batch or single-record ingestion operation."""

    processed: int = 0
    inserted: int = 0
    updated: int = 0
    errors: list[str] = Field(default_factory=list)
    companies_seen: list[str] = Field(default_factory=list)
    devices_seen: list[str] = Field(default_factory=list)


def _upsert_snapshot_in_conn(
    conn: sqlite3.Connection, snapshot: TelemetrySnapshot
) -> tuple[int, bool]:
    """Upsert a single validated snapshot and its child records.

    Returns ``(snapshot_id, was_inserted)``.
    """
    collected_at = snapshot.normalized_timestamp()
    ingested_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    raw_payload = json.dumps(snapshot.raw_dict(), sort_keys=True)

    existing = conn.execute(
        "SELECT id FROM snapshots WHERE device_id = ? AND collected_at = ?",
        (snapshot.device_id, collected_at),
    ).fetchone()

    mem = snapshot.memory
    batt = snapshot.battery

    values = (
        snapshot.device_id,
        snapshot.company_id,
        snapshot.employee_id,
        collected_at,
        snapshot.agent_version,
        snapshot.os.platform,
        snapshot.os.product_name,
        snapshot.os.product_version,
        snapshot.os.build_version,
        snapshot.os.architecture,
        snapshot.os.kernel_name,
        snapshot.os.kernel_release,
        snapshot.os.hostname,
        snapshot.device_identity.serial_number,
        snapshot.device_identity.model_name,
        snapshot.device_identity.model_identifier,
        snapshot.device_identity.processor,
        snapshot.device_identity.hardware_uuid,
        snapshot.device_identity.total_memory,
        mem.ram_bytes if mem else None,
        mem.total_memory_bytes if mem else None,
        mem.used_memory_bytes if mem else None,
        mem.free_memory_bytes if mem else None,
        mem.used_pct if mem else None,
        (1 if batt.battery_present else 0) if (batt and batt.battery_present is not None) else None,
        batt.charging_status if batt else None,
        batt.percentage if batt else None,
        batt.condition if batt else None,
        batt.cycle_count if batt else None,
        batt.full_charge_capacity if batt else None,
        raw_payload,
        ingested_at,
    )

    if existing:
        snapshot_id = int(existing["id"])
        was_inserted = False
        conn.execute(
            """
            UPDATE snapshots SET
                device_id = ?, company_id = ?, employee_id = ?, collected_at = ?,
                agent_version = ?, os_platform = ?, os_product_name = ?,
                os_product_version = ?, os_build_version = ?, os_architecture = ?,
                os_kernel_name = ?, os_kernel_release = ?, hostname = ?,
                serial_number = ?, model_name = ?, model_identifier = ?,
                processor = ?, hardware_uuid = ?, total_memory_label = ?,
                ram_bytes = ?, total_memory_bytes = ?, used_memory_bytes = ?,
                free_memory_bytes = ?, memory_used_pct = ?,
                battery_present = ?, battery_charging_status = ?,
                battery_percentage = ?, battery_condition = ?,
                battery_cycle_count = ?, battery_full_charge_capacity = ?,
                raw_payload = ?, ingested_at = ?
            WHERE id = ?
            """,
            (*values, snapshot_id),
        )
        # Replace normalized child rows cleanly for idempotency
        for child_table in (
            "disk_volumes",
            "compliance_results",
            "installed_software",
            "network_interfaces",
        ):
            conn.execute(f"DELETE FROM {child_table} WHERE snapshot_id = ?", (snapshot_id,))
    else:
        was_inserted = True
        cursor = conn.execute(
            """
            INSERT INTO snapshots (
                device_id, company_id, employee_id, collected_at, agent_version,
                os_platform, os_product_name, os_product_version, os_build_version,
                os_architecture, os_kernel_name, os_kernel_release, hostname,
                serial_number, model_name, model_identifier, processor,
                hardware_uuid, total_memory_label,
                ram_bytes, total_memory_bytes, used_memory_bytes, free_memory_bytes,
                memory_used_pct, battery_present, battery_charging_status,
                battery_percentage, battery_condition, battery_cycle_count,
                battery_full_charge_capacity, raw_payload, ingested_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            """,
            values,
        )
        snapshot_id = int(cursor.lastrowid)

    if snapshot.disk_volumes:
        conn.executemany(
            """
            INSERT INTO disk_volumes (
                snapshot_id, company_id, device_id, collected_at,
                volume_name, file_system, mount_point,
                size_bytes, available_bytes, available_pct, encrypted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    snapshot.company_id,
                    snapshot.device_id,
                    collected_at,
                    v.volume_name,
                    v.file_system,
                    v.mount_point,
                    v.size_bytes,
                    v.available_bytes,
                    v.available_pct,
                    (1 if v.encrypted else 0) if v.encrypted is not None else None,
                )
                for v in snapshot.disk_volumes
            ],
        )

    if snapshot.compliance_results:
        conn.executemany(
            """
            INSERT INTO compliance_results (
                snapshot_id, company_id, device_id, collected_at,
                check_id, status, severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    snapshot.company_id,
                    snapshot.device_id,
                    collected_at,
                    c.check_id,
                    c.status.lower(),
                    c.severity.lower(),
                )
                for c in snapshot.compliance_results
            ],
        )

    if snapshot.installed_software:
        conn.executemany(
            """
            INSERT INTO installed_software (
                snapshot_id, company_id, device_id, collected_at,
                name, version, publisher
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    snapshot.company_id,
                    snapshot.device_id,
                    collected_at,
                    s.name,
                    s.version,
                    s.publisher,
                )
                for s in snapshot.installed_software
            ],
        )

    if snapshot.network:
        conn.executemany(
            """
            INSERT INTO network_interfaces (
                snapshot_id, company_id, device_id, collected_at,
                address, family, interface_name, internal, mac
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    snapshot.company_id,
                    snapshot.device_id,
                    collected_at,
                    n.address,
                    n.family,
                    n.interface_name,
                    (1 if n.internal else 0) if n.internal is not None else None,
                    n.mac,
                )
                for n in snapshot.network
            ],
        )

    return snapshot_id, was_inserted


def ingest_snapshot(db: FleetDatabase, payload: dict[str, Any]) -> IngestionReport:
    """Ingest a single realtime telemetry JSON object idempotently."""
    report = IngestionReport()
    try:
        snapshot = TelemetrySnapshot.model_validate(payload)
    except ValidationError as exc:
        report.errors.append(f"Validation error: {exc}")
        return report

    with db.transaction() as conn:
        _, inserted = _upsert_snapshot_in_conn(conn, snapshot)
        report.processed = 1
        if inserted:
            report.inserted = 1
        else:
            report.updated = 1
        report.companies_seen = [snapshot.company_id]
        report.devices_seen = [snapshot.device_id]
    return report


def ingest_ndjson_text(db: FleetDatabase, ndjson_text: str) -> IngestionReport:
    """Ingest newline-delimited JSON (or a single JSON object/array) transactionally."""
    report = IngestionReport()
    companies: set[str] = set()
    devices: set[str] = set()

    stripped = ndjson_text.strip()
    if not stripped:
        return report

    raw_items: list[tuple[int, dict[str, Any]]] = []
    # First attempt full-document JSON parse (handles pretty-printed single JSON objects and JSON arrays)
    parsed_whole = False
    try:
        doc = json.loads(stripped)
        parsed_whole = True
        if isinstance(doc, dict):
            raw_items.append((1, doc))
        elif isinstance(doc, list):
            for idx, obj in enumerate(doc, start=1):
                if isinstance(obj, dict):
                    raw_items.append((idx, obj))
                else:
                    report.errors.append(f"Item {idx}: expected JSON object")
        else:
            report.errors.append("Expected a JSON object, JSON array, or NDJSON stream.")
            return report
    except json.JSONDecodeError:
        parsed_whole = False

    if not parsed_whole:
        if stripped.startswith("["):
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                report.errors.append(f"Invalid JSON array: {exc}")
                return report
        # Fall back to newline-delimited JSON (NDJSON)
        for line_no, line in enumerate(stripped.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    raw_items.append((line_no, obj))
                else:
                    report.errors.append(f"Line {line_no}: expected JSON object")
            except json.JSONDecodeError as exc:
                report.errors.append(f"Line {line_no}: invalid JSON ({exc})")

    with db.transaction() as conn:
        for line_no, obj in raw_items:
            try:
                snapshot = TelemetrySnapshot.model_validate(obj)
            except ValidationError as exc:
                report.errors.append(f"Line {line_no}: validation failed ({exc.error_count()} errors)")
                continue

            _, inserted = _upsert_snapshot_in_conn(conn, snapshot)
            report.processed += 1
            if inserted:
                report.inserted += 1
            else:
                report.updated += 1
            companies.add(snapshot.company_id)
            devices.add(snapshot.device_id)

    report.companies_seen = sorted(companies)
    report.devices_seen = sorted(devices)
    return report


def ingest_ndjson_file(db: FleetDatabase, file_path: str | Path) -> IngestionReport:
    """Ingest an NDJSON telemetry file into the database."""
    path = Path(file_path)
    return ingest_ndjson_text(db, path.read_text(encoding="utf-8"))
