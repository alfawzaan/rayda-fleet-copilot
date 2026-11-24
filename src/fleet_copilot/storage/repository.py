"""Tenant-isolated telemetry repository and SQLite authorizer-enforced SQL reader.

All data access is strictly scoped to a single ``company_id`` bound at
construction time. Cross-tenant access is impossible at both the repository
query layer and the SQLite C authorizer layer.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from fleet_copilot.storage.database import FleetDatabase

ALLOWED_TENANT_VIEWS = frozenset(
    {
        "v_snapshots",
        "v_disk_volumes",
        "v_compliance_results",
        "v_installed_software",
        "v_network_interfaces",
    }
)

VIEW_TO_BASE_TABLE = {
    "v_snapshots": "snapshots",
    "v_disk_volumes": "disk_volumes",
    "v_compliance_results": "compliance_results",
    "v_installed_software": "installed_software",
    "v_network_interfaces": "network_interfaces",
}

_DESKTOP_KEYWORDS = ("mini", "desktop", "tower", "imac", "mac studio", "mac pro", "workstation", "nuc")
_LAPTOP_KEYWORDS = ("book", "laptop", "thinkpad", "xps", "latitude", "elitebook", "surface", "zenbook", "spectre", "ideapad")


class CrossTenantAccessError(PermissionError):
    """Raised when a query explicitly targets an entity owned by another tenant."""


class UnsafeQueryError(ValueError):
    """Raised when an ad-hoc SQL query violates read-only or tenant isolation rules."""


def infer_form_factor(model_name: str, ever_had_battery: bool) -> str:
    """Infer whether a device is a laptop or desktop dynamically from telemetry."""
    if ever_had_battery:
        return "laptop"
    lowered = (model_name or "").lower()
    if any(k in lowered for k in _DESKTOP_KEYWORDS):
        return "desktop"
    if any(k in lowered for k in _LAPTOP_KEYWORDS):
        return "laptop"
    return "unknown"


def parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Extract numeric version components (e.g. '14.5' -> (14, 5), '10 22H2' -> (10, 22, 2))."""
    nums = re.findall(r"\d+", version_str or "")
    return tuple(int(n) for n in nums) if nums else (0,)


class TenantTelemetryRepository:
    """Strictly tenant-scoped read interface over the fleet telemetry database."""

    def __init__(self, db: FleetDatabase, company_id: str) -> None:
        if not company_id or not company_id.strip():
            raise ValueError("company_id must be a non-empty string")
        self.db = db
        self.company_id = company_id.strip()

    def assert_device_in_tenant(self, device_id: str) -> bool:
        """Verify device ownership.

        Returns True if owned by this tenant, False if nonexistent everywhere,
        and raises ``CrossTenantAccessError`` if owned by a different company.
        """
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM snapshots WHERE company_id = ? AND device_id = ? LIMIT 1",
                (self.company_id, device_id),
            ).fetchone()
            if row:
                return True

        owner = self.db.get_entity_owner(device_id=device_id)
        if owner and owner != self.company_id:
            raise CrossTenantAccessError(
                f"Access denied: device '{device_id}' does not belong to tenant '{self.company_id}'."
            )
        return False

    def assert_employee_in_tenant(self, employee_id: str) -> bool:
        """Verify employee ownership within the active tenant."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM snapshots WHERE company_id = ? AND employee_id = ? LIMIT 1",
                (self.company_id, employee_id),
            ).fetchone()
            if row:
                return True

        owner = self.db.get_entity_owner(employee_id=employee_id)
        if owner and owner != self.company_id:
            raise CrossTenantAccessError(
                f"Access denied: employee '{employee_id}' does not belong to tenant '{self.company_id}'."
            )
        return False

    def _device_battery_history_map(self, conn: sqlite3.Connection) -> dict[str, bool]:
        rows = conn.execute(
            """
            SELECT device_id, MAX(CASE WHEN battery_present = 1 THEN 1 ELSE 0 END) AS has_batt
            FROM snapshots
            WHERE company_id = ?
            GROUP BY device_id
            """,
            (self.company_id,),
        ).fetchall()
        return {r["device_id"]: bool(r["has_batt"]) for r in rows}

    def get_fleet_overview(self) -> dict[str, Any]:
        """Return dynamic metadata about the tenant's fleet (no hardcoded catalogs)."""
        with self.db.transaction() as conn:
            summary = conn.execute(
                """
                SELECT
                    COUNT(*) AS snapshot_count,
                    COUNT(DISTINCT device_id) AS device_count,
                    COUNT(DISTINCT employee_id) AS employee_count,
                    MIN(collected_at) AS earliest_snapshot,
                    MAX(collected_at) AS latest_snapshot
                FROM snapshots
                WHERE company_id = ?
                """,
                (self.company_id,),
            ).fetchone()

            os_rows = conn.execute(
                """
                SELECT s.os_product_name, s.os_product_version, COUNT(DISTINCT s.device_id) AS device_count
                FROM snapshots s
                INNER JOIN (
                    SELECT device_id, MAX(collected_at) AS max_ts
                    FROM snapshots WHERE company_id = ? GROUP BY device_id
                ) latest ON s.device_id = latest.device_id AND s.collected_at = latest.max_ts
                WHERE s.company_id = ?
                GROUP BY s.os_product_name, s.os_product_version
                ORDER BY s.os_product_name, s.os_product_version
                """,
                (self.company_id, self.company_id),
            ).fetchall()

            check_rows = conn.execute(
                """
                SELECT DISTINCT check_id, severity
                FROM compliance_results
                WHERE company_id = ?
                ORDER BY severity, check_id
                """,
                (self.company_id,),
            ).fetchall()

            sw_rows = conn.execute(
                """
                SELECT name, COUNT(DISTINCT device_id) AS device_count
                FROM installed_software
                WHERE company_id = ?
                GROUP BY name
                ORDER BY device_count DESC, name ASC
                """,
                (self.company_id,),
            ).fetchall()

        return {
            "company_id": self.company_id,
            "snapshot_count": int(summary["snapshot_count"]),
            "device_count": int(summary["device_count"]),
            "employee_count": int(summary["employee_count"]),
            "earliest_snapshot": summary["earliest_snapshot"],
            "latest_snapshot": summary["latest_snapshot"],
            "os_distribution": [dict(r) for r in os_rows],
            "compliance_checks_observed": [dict(r) for r in check_rows],
            "software_observed": [dict(r) for r in sw_rows],
        }

    def list_latest_devices(
        self,
        *,
        os_name: str | None = None,
        device_type: str | None = None,
        employee_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List latest telemetry snapshot per device within this tenant."""
        if employee_id:
            self.assert_employee_in_tenant(employee_id)

        with self.db.transaction() as conn:
            batt_map = self._device_battery_history_map(conn)
            query = """
                SELECT s.*
                FROM snapshots s
                INNER JOIN (
                    SELECT device_id, MAX(collected_at) AS max_ts
                    FROM snapshots
                    WHERE company_id = ?
                    GROUP BY device_id
                ) latest
                ON s.device_id = latest.device_id AND s.collected_at = latest.max_ts
                WHERE s.company_id = ?
            """
            params: list[Any] = [self.company_id, self.company_id]
            if os_name:
                query += " AND LOWER(s.os_product_name) LIKE ?"
                params.append(f"%{os_name.lower()}%")
            if employee_id:
                query += " AND s.employee_id = ?"
                params.append(employee_id)
            query += " ORDER BY s.device_id"

            rows = conn.execute(query, params).fetchall()

            # Attach latest primary disk and latest failing compliance checks
            results: list[dict[str, Any]] = []
            for r in rows:
                form_factor = infer_form_factor(
                    r["model_name"], batt_map.get(r["device_id"], False)
                )
                if device_type and device_type.lower() != "all" and form_factor != device_type.lower():
                    continue

                disks = conn.execute(
                    """
                    SELECT volume_name, mount_point, size_bytes, available_bytes, available_pct, encrypted
                    FROM disk_volumes
                    WHERE snapshot_id = ? AND company_id = ?
                    """,
                    (r["id"], self.company_id),
                ).fetchall()

                failing_checks = conn.execute(
                    """
                    SELECT check_id, severity, status
                    FROM compliance_results
                    WHERE snapshot_id = ? AND company_id = ? AND status = 'fail'
                    """,
                    (r["id"], self.company_id),
                ).fetchall()

                min_disk_pct = (
                    min(float(d["available_pct"]) for d in disks) if disks else None
                )
                results.append(
                    {
                        "device_id": r["device_id"],
                        "company_id": r["company_id"],
                        "employee_id": r["employee_id"],
                        "collected_at": r["collected_at"],
                        "hostname": r["hostname"],
                        "model_name": r["model_name"],
                        "form_factor": form_factor,
                        "processor": r["processor"],
                        "total_memory_label": r["total_memory_label"],
                        "os_platform": r["os_platform"],
                        "os_product_name": r["os_product_name"],
                        "os_product_version": r["os_product_version"],
                        "memory_used_pct": r["memory_used_pct"],
                        "min_disk_available_pct": min_disk_pct,
                        "disk_volumes": [dict(d) for d in disks],
                        "battery_present": bool(r["battery_present"])
                        if r["battery_present"] is not None
                        else None,
                        "battery_condition": r["battery_condition"],
                        "battery_cycle_count": r["battery_cycle_count"],
                        "battery_full_charge_capacity": r["battery_full_charge_capacity"],
                        "failing_compliance_checks": [dict(c) for c in failing_checks],
                    }
                )
        return results

    def get_device_history(
        self,
        device_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        latest_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve time-ordered snapshots for a specific device in this tenant."""
        exists = self.assert_device_in_tenant(device_id)
        if not exists:
            return []

        with self.db.transaction() as conn:
            query = "SELECT * FROM snapshots WHERE company_id = ? AND device_id = ?"
            params: list[Any] = [self.company_id, device_id]
            if start_date:
                query += " AND collected_at >= ?"
                params.append(start_date)
            if end_date:
                query += " AND collected_at <= ?"
                params.append(end_date)
            query += " ORDER BY collected_at DESC" if latest_only else " ORDER BY collected_at ASC"
            if latest_only:
                query += " LIMIT 1"

            rows = conn.execute(query, params).fetchall()
            history: list[dict[str, Any]] = []
            for r in rows:
                disks = [
                    dict(d)
                    for d in conn.execute(
                        "SELECT volume_name, mount_point, size_bytes, available_bytes, available_pct, encrypted FROM disk_volumes WHERE snapshot_id = ? AND company_id = ?",
                        (r["id"], self.company_id),
                    ).fetchall()
                ]
                checks = [
                    dict(c)
                    for c in conn.execute(
                        "SELECT check_id, status, severity FROM compliance_results WHERE snapshot_id = ? AND company_id = ?",
                        (r["id"], self.company_id),
                    ).fetchall()
                ]
                software = [
                    dict(sw)
                    for sw in conn.execute(
                        "SELECT name, version, publisher FROM installed_software WHERE snapshot_id = ? AND company_id = ?",
                        (r["id"], self.company_id),
                    ).fetchall()
                ]
                history.append(
                    {
                        "device_id": r["device_id"],
                        "company_id": r["company_id"],
                        "employee_id": r["employee_id"],
                        "collected_at": r["collected_at"],
                        "hostname": r["hostname"],
                        "model_name": r["model_name"],
                        "os_product_name": r["os_product_name"],
                        "os_product_version": r["os_product_version"],
                        "memory_used_pct": r["memory_used_pct"],
                        "total_memory_bytes": r["total_memory_bytes"],
                        "used_memory_bytes": r["used_memory_bytes"],
                        "battery_present": bool(r["battery_present"])
                        if r["battery_present"] is not None
                        else None,
                        "battery_condition": r["battery_condition"],
                        "battery_cycle_count": r["battery_cycle_count"],
                        "battery_full_charge_capacity": r["battery_full_charge_capacity"],
                        "disk_volumes": disks,
                        "compliance_results": checks,
                        "installed_software": software,
                    }
                )
        return history

    def query_low_storage_devices(
        self,
        *,
        threshold_free_pct: float = 15.0,
        as_of_latest: bool = True,
        device_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find tenant devices with disk free space below ``threshold_free_pct``."""
        with self.db.transaction() as conn:
            batt_map = self._device_battery_history_map(conn)
            if as_of_latest:
                sql = """
                    SELECT d.*, s.employee_id, s.model_name, s.hostname, s.os_product_name, s.os_product_version
                    FROM disk_volumes d
                    INNER JOIN snapshots s ON d.snapshot_id = s.id
                    INNER JOIN (
                        SELECT device_id, MAX(collected_at) AS max_ts
                        FROM snapshots WHERE company_id = ? GROUP BY device_id
                    ) latest ON s.device_id = latest.device_id AND s.collected_at = latest.max_ts
                    WHERE d.company_id = ? AND d.available_pct < ?
                    ORDER BY d.available_pct ASC, d.device_id ASC
                """
                params = (self.company_id, self.company_id, threshold_free_pct)
            else:
                sql = """
                    SELECT
                        d.device_id,
                        d.company_id,
                        s.employee_id,
                        s.model_name,
                        s.hostname,
                        d.volume_name,
                        d.mount_point,
                        MIN(d.available_pct) AS available_pct,
                        MIN(d.available_bytes) AS available_bytes,
                        MAX(d.size_bytes) AS size_bytes,
                        COUNT(*) AS snapshots_below_threshold,
                        MIN(d.collected_at) AS first_seen_low,
                        MAX(d.collected_at) AS collected_at
                    FROM disk_volumes d
                    INNER JOIN snapshots s ON d.snapshot_id = s.id
                    WHERE d.company_id = ? AND d.available_pct < ?
                    GROUP BY d.device_id, d.volume_name, d.mount_point
                    ORDER BY available_pct ASC, d.device_id ASC
                """
                params = (self.company_id, threshold_free_pct)

            rows = conn.execute(sql, params).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                form_factor = infer_form_factor(
                    r["model_name"], batt_map.get(r["device_id"], False)
                )
                if device_type and device_type.lower() != "all" and form_factor != device_type.lower():
                    continue
                item = dict(r)
                item["form_factor"] = form_factor
                item["available_gb"] = round(item["available_bytes"] / (1024**3), 2)
                item["size_gb"] = round(item["size_bytes"] / (1024**3), 2)
                out.append(item)
        return out

    def query_compliance_failures(
        self,
        *,
        severity: str | None = None,
        check_id: str | None = None,
        as_of_latest: bool = True,
        device_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query compliance failures within the tenant."""
        with self.db.transaction() as conn:
            batt_map = self._device_battery_history_map(conn)
            if as_of_latest:
                sql = """
                    SELECT c.device_id, c.company_id, s.employee_id, s.model_name, s.hostname,
                           c.collected_at, c.check_id, c.status, c.severity
                    FROM compliance_results c
                    INNER JOIN snapshots s ON c.snapshot_id = s.id
                    INNER JOIN (
                        SELECT device_id, MAX(collected_at) AS max_ts
                        FROM snapshots WHERE company_id = ? GROUP BY device_id
                    ) latest ON s.device_id = latest.device_id AND s.collected_at = latest.max_ts
                    WHERE c.company_id = ? AND c.status = 'fail'
                """
                params: list[Any] = [self.company_id, self.company_id]
            else:
                sql = """
                    SELECT c.device_id, c.company_id, s.employee_id, s.model_name, s.hostname,
                           c.check_id, c.severity,
                           COUNT(*) AS failed_snapshots,
                           MIN(c.collected_at) AS first_failed_at,
                           MAX(c.collected_at) AS collected_at
                    FROM compliance_results c
                    INNER JOIN snapshots s ON c.snapshot_id = s.id
                    WHERE c.company_id = ? AND c.status = 'fail'
                """
                params = [self.company_id]

            if severity:
                sql += " AND LOWER(c.severity) = ?"
                params.append(severity.lower())
            if check_id:
                sql += " AND LOWER(c.check_id) = ?"
                params.append(check_id.lower())

            if not as_of_latest:
                sql += " GROUP BY c.device_id, c.check_id, c.severity"

            sql += " ORDER BY c.device_id ASC, c.check_id ASC"
            rows = conn.execute(sql, params).fetchall()

            results: list[dict[str, Any]] = []
            for r in rows:
                form_factor = infer_form_factor(
                    r["model_name"], batt_map.get(r["device_id"], False)
                )
                if device_type and device_type.lower() != "all" and form_factor != device_type.lower():
                    continue
                item = dict(r)
                item["form_factor"] = form_factor
                results.append(item)
        return results

    def query_os_versions(
        self,
        *,
        product_name: str | None = None,
        below_version: str | None = None,
        device_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query devices by OS product name and optional semantic version ceiling."""
        devices = self.list_latest_devices(os_name=product_name, device_type=device_type)
        if not below_version:
            return devices

        target_tuple = parse_version_tuple(below_version)
        matched: list[dict[str, Any]] = []
        for dev in devices:
            dev_tuple = parse_version_tuple(dev["os_product_version"])
            # Pad to equal length for safe numeric tuple comparison
            max_len = max(len(dev_tuple), len(target_tuple))
            padded_dev = dev_tuple + (0,) * (max_len - len(dev_tuple))
            padded_target = target_tuple + (0,) * (max_len - len(target_tuple))
            if padded_dev < padded_target:
                matched.append(dev)
        return matched

    def query_installed_software(
        self,
        *,
        software_name: str | None = None,
        as_of_latest: bool = True,
    ) -> list[dict[str, Any]]:
        """Search installed software across tenant devices (case-insensitive substring match)."""
        with self.db.transaction() as conn:
            if as_of_latest:
                sql = """
                    SELECT sw.device_id, sw.company_id, s.employee_id, s.model_name,
                           sw.collected_at, sw.name, sw.version, sw.publisher
                    FROM installed_software sw
                    INNER JOIN snapshots s ON sw.snapshot_id = s.id
                    INNER JOIN (
                        SELECT device_id, MAX(collected_at) AS max_ts
                        FROM snapshots WHERE company_id = ? GROUP BY device_id
                    ) latest ON s.device_id = latest.device_id AND s.collected_at = latest.max_ts
                    WHERE sw.company_id = ?
                """
                params: list[Any] = [self.company_id, self.company_id]
            else:
                sql = """
                    SELECT sw.device_id, sw.company_id, s.employee_id, s.model_name,
                           sw.name, sw.version, sw.publisher,
                           COUNT(*) AS snapshot_count,
                           MIN(sw.collected_at) AS first_seen_at,
                           MAX(sw.collected_at) AS collected_at
                    FROM installed_software sw
                    INNER JOIN snapshots s ON sw.snapshot_id = s.id
                    WHERE sw.company_id = ?
                """
                params = [self.company_id]

            if software_name:
                sql += " AND LOWER(sw.name) LIKE ?"
                params.append(f"%{software_name.lower()}%")

            if not as_of_latest:
                sql += " GROUP BY sw.device_id, sw.name, sw.version, sw.publisher"

            sql += " ORDER BY sw.name ASC, sw.device_id ASC"
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def execute_readonly_sql(self, sql: str, *, max_rows: int = 100) -> list[dict[str, Any]]:
        """Execute a safe read-only SQL query against tenant-isolated TEMP VIEWs.

        Security model (enforced at the SQLite C authorizer level):
        1. Only single SELECT / WITH statements are permitted.
        2. TEMP VIEWs ``v_snapshots``, ``v_disk_volumes``, ``v_compliance_results``,
           ``v_installed_software``, and ``v_network_interfaces`` are created with
           ``WHERE company_id = <tenant_id>`` baked in.
        3. ``PRAGMA query_only = ON`` and ``conn.set_authorizer(...)`` prohibit all
           writes, DDL, PRAGMAs, ATTACHes, and any direct reads of base tables
           (such as ``snapshots``, ``executed_actions``, or ``audit_logs``).
        """
        cleaned = (sql or "").strip().rstrip(";").strip()
        if not cleaned:
            raise UnsafeQueryError("SQL query cannot be empty.")
        if ";" in cleaned:
            raise UnsafeQueryError("Stacked SQL statements are prohibited.")
        first_token = cleaned.split(None, 1)[0].upper()
        if first_token not in {"SELECT", "WITH"}:
            raise UnsafeQueryError("Only read-only SELECT / WITH queries are permitted.")

        # Escape tenant_id safely for SQL literal inside TEMP VIEW DDL
        escaped_tenant = self.company_id.replace("'", "''")

        conn = self.db.connect()
        try:
            for view_name, base_table in VIEW_TO_BASE_TABLE.items():
                conn.execute(
                    f"CREATE TEMP VIEW {view_name} AS "
                    f"SELECT * FROM main.{base_table} WHERE company_id = '{escaped_tenant}'"
                )

            conn.execute("PRAGMA query_only = ON;")

            def _authorizer(
                action: int,
                arg1: str | None,
                arg2: str | None,
                dbname: str | None,
                source: str | None,
            ) -> int:
                if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ:
                    # Direct reads permitted ONLY on the tenant-filtered temp views
                    if arg1 in ALLOWED_TENANT_VIEWS:
                        return sqlite3.SQLITE_OK
                    # Underlying base-table reads permitted ONLY when originating from its matching view
                    if source in VIEW_TO_BASE_TABLE and VIEW_TO_BASE_TABLE[source] == arg1:
                        return sqlite3.SQLITE_OK
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_DENY

            conn.set_authorizer(_authorizer)
            cursor = conn.execute(cleaned)
            rows = cursor.fetchmany(max_rows)
            return [dict(r) for r in rows]
        except sqlite3.DatabaseError as exc:
            raise UnsafeQueryError(f"Query rejected by tenant-isolated SQL engine: {exc}") from exc
        finally:
            conn.set_authorizer(None)
            conn.close()
