"""Immutable audit logging persisted to SQLite and mirrored to append-only JSONL."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fleet_copilot.models.actions import AuditEvent
from fleet_copilot.storage.database import FleetDatabase


class AuditLogger:
    """Append-only audit trail for governance, security, and action execution."""

    def __init__(self, db: FleetDatabase, jsonl_path: str | Path | None = None) -> None:
        self.db = db
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        thread_id: str,
        company_id: str,
        actor_id: str,
        event_type: str,
        status: str,
        target_entity: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Persist an audit event to SQLite and append to JSONL mirror."""
        event = AuditEvent(
            event_id=f"AUD-{uuid.uuid4().hex[:12].upper()}",
            thread_id=thread_id,
            company_id=company_id,
            actor_id=actor_id,
            event_type=event_type,
            status=status,
            target_entity=target_entity,
            details=details or {},
        )
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (
                    event_id, timestamp, thread_id, company_id, actor_id,
                    event_type, status, target_entity, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp,
                    event.thread_id,
                    event.company_id,
                    event.actor_id,
                    event.event_type,
                    event.status,
                    event.target_entity,
                    json.dumps(event.details, sort_keys=True),
                ),
            )

        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(event.model_dump_json() + "\n")

        return event

    def list_events(self, company_id: str, *, limit: int = 100) -> list[AuditEvent]:
        """Return recent audit events scoped strictly to a single tenant."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT event_id, timestamp, thread_id, company_id, actor_id,
                       event_type, status, target_entity, details_json
                FROM audit_logs
                WHERE company_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (company_id, limit),
            ).fetchall()

        return [
            AuditEvent(
                event_id=r["event_id"],
                timestamp=r["timestamp"],
                thread_id=r["thread_id"],
                company_id=r["company_id"],
                actor_id=r["actor_id"],
                event_type=r["event_type"],
                status=r["status"],
                target_entity=r["target_entity"],
                details=json.loads(r["details_json"]),
            )
            for r in rows
        ]
