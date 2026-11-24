"""Generalizable time-series telemetry analytics and trend detection."""

from fleet_copilot.analytics.trends import (
    detect_battery_degradation,
    detect_compliance_drift,
    detect_ram_constraints,
    detect_software_anomalies,
    detect_storage_trends,
    generate_fleet_insights,
)

__all__ = [
    "detect_battery_degradation",
    "detect_compliance_drift",
    "detect_ram_constraints",
    "detect_software_anomalies",
    "detect_storage_trends",
    "generate_fleet_insights",
]
