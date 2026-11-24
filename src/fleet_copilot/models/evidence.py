"""Structured evidence citations and execution traces for grounding."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Atomic, traceable reference to a telemetry observation or computed aggregate."""

    evidence_id: str = Field(description="Unique ID for this evidence item within the run.")
    source_tool: str = Field(description="Tool that produced this evidence.")
    company_id: str
    device_id: str | None = None
    employee_id: str | None = None
    collected_at: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    metric: str
    value: Any
    summary: str

    def format_inline(self) -> str:
        """Return a concise citation tag suitable for grounded prose."""
        parts = [self.evidence_id]
        if self.device_id:
            parts.append(f"device={self.device_id}")
        if self.collected_at:
            parts.append(f"at={self.collected_at}")
        elif self.window_start and self.window_end:
            parts.append(f"window={self.window_start[:10]}..{self.window_end[:10]}")
        parts.append(f"{self.metric}={self.value}")
        return "[" + " | ".join(parts) + "]"


class ToolTrace(BaseModel):
    """Record of a single tool invocation executed during an agent run."""

    step: int
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_count: int | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    status: str = "ok"
    summary: str = ""
