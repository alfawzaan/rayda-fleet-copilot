"""Schema-tolerant Pydantic models for device telemetry snapshots.

Models intentionally use ``extra="allow"`` so realtime streams containing new
telemetry attributes or future agent versions are ingested without failure and
preserved inside ``raw_payload``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    """Base model allowing forward-compatible unknown fields."""

    model_config = ConfigDict(extra="allow")


class OSInfo(FlexibleModel):
    platform: str = "unknown"
    product_name: str = "unknown"
    product_version: str = "unknown"
    build_version: str | None = None
    architecture: str | None = None
    kernel_name: str | None = None
    kernel_release: str | None = None
    hostname: str | None = None


class DeviceIdentity(FlexibleModel):
    serial_number: str | None = None
    model_name: str = "unknown"
    model_identifier: str | None = None
    processor: str | None = None
    hardware_uuid: str | None = None
    total_memory: str | None = None


class MemoryInfo(FlexibleModel):
    ram_bytes: int | None = None
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int | None = None
    page_size_bytes: int | None = None

    @property
    def used_pct(self) -> float:
        if self.total_memory_bytes <= 0:
            return 0.0
        return round((self.used_memory_bytes / self.total_memory_bytes) * 100.0, 2)


class DiskVolume(FlexibleModel):
    volume_name: str = "unknown"
    file_system: str | None = None
    mount_point: str = "/"
    size_bytes: int
    available_bytes: int
    encrypted: bool | None = None

    @property
    def available_pct(self) -> float:
        if self.size_bytes <= 0:
            return 0.0
        return round((self.available_bytes / self.size_bytes) * 100.0, 2)


class BatteryInfo(FlexibleModel):
    battery_present: bool | None = None
    charging_status: str | None = None
    percentage: float | None = None
    condition: str | None = None
    cycle_count: int | None = None
    full_charge_capacity: int | None = None


class NetworkInterface(FlexibleModel):
    address: str | None = None
    family: str | None = None
    interface_name: str | None = None
    internal: bool | None = None
    mac: str | None = None


class InstalledSoftware(FlexibleModel):
    name: str
    version: str | None = None
    publisher: str | None = None


class ComplianceResult(FlexibleModel):
    check_id: str
    status: str
    severity: str = "unknown"


class TelemetrySnapshot(FlexibleModel):
    """Point-in-time telemetry snapshot for a single managed device."""

    device_id: str
    company_id: str
    employee_id: str | None = None
    collected_at: datetime
    agent_version: str | None = None
    os: OSInfo
    device_identity: DeviceIdentity
    memory: MemoryInfo | None = None
    disk_volumes: list[DiskVolume] = Field(default_factory=list)
    battery: BatteryInfo | None = None
    network: list[NetworkInterface] = Field(default_factory=list)
    installed_software: list[InstalledSoftware] = Field(default_factory=list)
    compliance_results: list[ComplianceResult] = Field(default_factory=list)

    def normalized_timestamp(self) -> str:
        """Return ISO-8601 UTC timestamp string for canonical storage & sorting."""
        ts = self.collected_at.isoformat()
        if ts.endswith("+00:00"):
            return ts[:-6] + "Z"
        return ts

    def raw_dict(self) -> dict[str, Any]:
        """Serialize full snapshot including any unseen extra fields."""
        return self.model_dump(mode="json")
