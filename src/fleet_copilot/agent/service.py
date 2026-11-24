"""High-level runtime service managing persistence, LangGraph checkpoints, and HITL resume."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from fleet_copilot.actions.audit import AuditLogger
from fleet_copilot.actions.backend import ActionExecutor
from fleet_copilot.agent.graph import build_fleet_copilot_graph, extract_text_content
from fleet_copilot.agent.llm import build_chat_model, build_fallback_models
from fleet_copilot.config import Settings, get_settings
from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.ingestion import IngestionReport, ingest_ndjson_file, ingest_ndjson_text
from fleet_copilot.storage.repository import TenantTelemetryRepository


class TurnResult(BaseModel):
    """Structured outcome returned to CLI, Streamlit UI, or evaluation harness."""

    status: str  # "completed" | "awaiting_approval"
    thread_id: str
    company_id: str
    answer: str = ""
    intent: str = ""
    plan: list[str] = Field(default_factory=list)
    tool_traces: list[dict[str, Any]] = Field(default_factory=list)
    evidence_store: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    pending_proposals: list[dict[str, Any]] = Field(default_factory=list)
    blocked_proposals: list[dict[str, Any]] = Field(default_factory=list)
    executed_actions: list[dict[str, Any]] = Field(default_factory=list)
    cross_tenant_blocked: bool = False


class FleetCopilotService:
    """Facade wrapping telemetry DB, LangGraph SqliteSaver checkpointer, and HITL workflows."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: BaseChatModel | None = None,
        auto_ingest_default_dataset: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_dirs()

        self.db = FleetDatabase(self.settings.telemetry_db_path)
        self.audit = AuditLogger(self.db, self.settings.audit_log_jsonl_path)
        self.action_executor = ActionExecutor(
            self.db,
            self.audit,
            action_webhook_url=self.settings.action_webhook_url,
            action_webhook_secret=self.settings.action_webhook_secret,
            slack_webhook_url=self.settings.slack_webhook_url,
            timeout_seconds=self.settings.webhook_timeout_seconds,
        )

        # Persistent SqliteSaver so HITL interrupts survive UI reloads & process restarts
        self._checkpoint_conn = sqlite3.connect(
            str(self.settings.checkpoint_db_path),
            check_same_thread=False,
        )
        self.checkpointer = SqliteSaver(self._checkpoint_conn)
        self.checkpointer.setup()

        self._llm = llm
        self._graph: Any = None

        if (
            auto_ingest_default_dataset
            and self.db.get_snapshot_count() == 0
            and self.settings.telemetry_dataset_path.exists()
        ):
            self.ingest_file(self.settings.telemetry_dataset_path, actor_id="system-bootstrap")

    @property
    def graph(self) -> Any:
        if self._graph is None:
            model = self._llm or build_chat_model(self.settings)
            fallbacks = [] if self._llm is not None else build_fallback_models(self.settings)
            self._graph = build_fleet_copilot_graph(
                db=self.db,
                audit_logger=self.audit,
                action_executor=self.action_executor,
                llm=model,
                fallback_llms=fallbacks,
                checkpointer=self.checkpointer,
                enable_llm_planner=self.settings.enable_llm_planner,
                max_steps=self.settings.max_agent_steps,
                max_sql_rows=self.settings.max_sql_rows,
            )
        return self._graph

    def close(self) -> None:
        if self._checkpoint_conn:
            self._checkpoint_conn.close()

    def list_companies(self) -> list[str]:
        return self.db.list_companies()

    def get_repository(self, company_id: str) -> TenantTelemetryRepository:
        return TenantTelemetryRepository(self.db, company_id)

    def ingest_file(
        self, file_path: str | Path, *, actor_id: str = "system", thread_id: str = "ingest"
    ) -> IngestionReport:
        report = ingest_ndjson_file(self.db, file_path)
        for cid in report.companies_seen:
            self.audit.record(
                thread_id=thread_id,
                company_id=cid,
                actor_id=actor_id,
                event_type="TELEMETRY_INGESTED",
                status="ok" if not report.errors else "partial",
                details={
                    "source": str(file_path),
                    "processed": report.processed,
                    "inserted": report.inserted,
                    "updated": report.updated,
                    "error_count": len(report.errors),
                },
            )
        return report

    def ingest_text(
        self, ndjson_text: str, *, actor_id: str = "it-admin", thread_id: str = "realtime-ingest"
    ) -> IngestionReport:
        report = ingest_ndjson_text(self.db, ndjson_text)
        for cid in report.companies_seen:
            self.audit.record(
                thread_id=thread_id,
                company_id=cid,
                actor_id=actor_id,
                event_type="TELEMETRY_INGESTED",
                status="ok" if not report.errors else "partial",
                details={
                    "source": "realtime_payload",
                    "processed": report.processed,
                    "inserted": report.inserted,
                    "updated": report.updated,
                    "error_count": len(report.errors),
                },
            )
        return report

    def run_turn(
        self,
        *,
        thread_id: str,
        company_id: str,
        user_message: str,
        actor_id: str = "it-admin",
    ) -> TurnResult:
        """Execute a conversational turn for a tenant-scoped thread."""
        config = {"configurable": {"thread_id": thread_id}}
        initial_input = {
            "messages": [HumanMessage(content=user_message)],
            "company_id": company_id,
            "actor_id": actor_id,
            "thread_id": thread_id,
        }
        self.graph.invoke(initial_input, config=config)
        return self._inspect_turn_result(thread_id=thread_id, company_id=company_id)

    def resume_approval(
        self,
        *,
        thread_id: str,
        company_id: str,
        decisions: list[dict[str, Any]],
    ) -> TurnResult:
        """Resume a paused HITL approval interrupt with explicit administrator decisions."""
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke(Command(resume={"decisions": decisions}), config=config)
        return self._inspect_turn_result(thread_id=thread_id, company_id=company_id)

    def get_pending_interrupt(self, thread_id: str) -> dict[str, Any] | None:
        """Inspect persisted checkpoint state for any pending HITL approval interrupt."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot or not snapshot.tasks:
            return None
        for task in snapshot.tasks:
            interrupts = getattr(task, "interrupts", None) or []
            if interrupts:
                return interrupts[0].value
        return None

    def list_threads(self, company_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        """List persisted conversation threads scoped strictly to a single tenant."""
        rows = self._checkpoint_conn.execute(
            """
            SELECT thread_id, MAX(checkpoint_id) AS latest_cp
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY latest_cp DESC
            LIMIT ?
            """,
            (limit * 3,),
        ).fetchall()

        threads: list[dict[str, Any]] = []
        for tid, _ in rows:
            tup = self.checkpointer.get_tuple({"configurable": {"thread_id": tid}})
            if not tup:
                continue
            ch = tup.checkpoint.get("channel_values") or {}
            if ch.get("company_id") != company_id:
                continue

            messages = ch.get("messages") or []
            user_msgs = [
                extract_text_content(m.content).strip()
                for m in messages
                if isinstance(m, HumanMessage) and extract_text_content(m.content).strip()
            ]
            preview = user_msgs[0][:56] if user_msgs else "(empty thread)"
            if user_msgs and len(user_msgs[0]) > 56:
                preview += "..."

            pending_writes = tup.pending_writes or []
            awaiting_approval = any(w[1] == "__interrupt__" for w in pending_writes)

            threads.append(
                {
                    "thread_id": tid,
                    "title": preview,
                    "turn_count": len(user_msgs),
                    "awaiting_approval": awaiting_approval,
                }
            )
            if len(threads) >= limit:
                break
        return threads

    def get_thread_history(self, thread_id: str, company_id: str) -> list[dict[str, Any]]:
        """Reconstruct tenant-scoped chat history with per-turn traces and citations."""
        tup = self.checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if not tup:
            return []
        ch = tup.checkpoint.get("channel_values") or {}
        if ch.get("company_id") != company_id:
            return []

        messages = ch.get("messages") or []
        full_evidence_store: dict[str, dict[str, Any]] = dict(ch.get("evidence_store") or {})
        cit_re = re.compile(r"\b(E\d+)\b")

        entries: list[dict[str, Any]] = []
        turn_traces: list[dict[str, Any]] = []
        turn_minted_eids: list[str] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        turn_step = 0

        for msg in messages:
            if isinstance(msg, HumanMessage):
                # Reset per-turn collectors at every new user message
                turn_traces = []
                turn_minted_eids = []
                pending_calls = {}
                turn_step = 0
                text = extract_text_content(msg.content).strip()
                if text:
                    entries.append({"role": "user", "content": text})
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    turn_step += 1
                    for tc in msg.tool_calls:
                        tc_id = str(tc.get("id") or "")
                        pending_calls[tc_id] = {
                            "step": turn_step,
                            "tool_name": tc.get("name", "unknown"),
                            "arguments": tc.get("args") or {},
                        }
                text = extract_text_content(msg.content).strip()
                # Final user-facing assistant text for this turn
                if text and not msg.tool_calls:
                    cited_in_text = [
                        eid
                        for eid in dict.fromkeys(cit_re.findall(text))
                        if eid in full_evidence_store
                    ]
                    turn_eids = list(dict.fromkeys([*cited_in_text, *turn_minted_eids]))
                    entries.append(
                        {
                            "role": "assistant",
                            "content": text,
                            "tool_traces": list(turn_traces),
                            "evidence_store": full_evidence_store,
                            "cited_evidence_ids": turn_eids,
                            "plan": [],
                            "intent": "",
                            "blocked_proposals": [],
                            "executed_actions": [],
                        }
                    )
            elif isinstance(msg, ToolMessage):
                tc_id = str(getattr(msg, "tool_call_id", "") or "")
                call_info = pending_calls.get(tc_id, {})
                tname = str(getattr(msg, "name", "") or call_info.get("tool_name") or "tool")
                raw_content = extract_text_content(msg.content)
                status = "ok"
                result_count: int | None = None
                eids: list[str] = []
                try:
                    payload = json.loads(raw_content)
                    if isinstance(payload, dict):
                        status = str(payload.get("status", "ok"))
                        if "count" in payload:
                            result_count = payload.get("count")
                        for ev_item in payload.get("evidence") or []:
                            if isinstance(ev_item, dict) and ev_item.get("evidence_id"):
                                eid = str(ev_item["evidence_id"])
                                eids.append(eid)
                                if eid not in turn_minted_eids:
                                    turn_minted_eids.append(eid)
                except (json.JSONDecodeError, TypeError):
                    pass

                turn_traces.append(
                    {
                        "step": call_info.get("step", max(turn_step, 1)),
                        "tool_name": tname,
                        "arguments": call_info.get("arguments") or {},
                        "result_count": result_count,
                        "evidence_ids": eids,
                        "status": status,
                        "summary": f"{tname} -> status={status}, count={result_count}, citations={len(eids)}",
                    }
                )

        # Attach latest turn's structured plan, intent, blocked/executed actions & final response
        final_resp = (ch.get("final_response") or "").strip()
        latest_traces = list(ch.get("tool_traces") or [])
        latest_cited = list(ch.get("cited_evidence_ids") or [])
        last_turn_meta = {
            "plan": list(ch.get("plan") or []),
            "intent": ch.get("intent", ""),
            "evidence_store": full_evidence_store,
            "blocked_proposals": list(ch.get("blocked_proposals") or []),
            "executed_actions": list(ch.get("executed_actions") or []),
        }
        if entries and entries[-1]["role"] == "assistant":
            if final_resp:
                entries[-1]["content"] = final_resp
            if latest_traces:
                entries[-1]["tool_traces"] = latest_traces
            if latest_cited:
                entries[-1]["cited_evidence_ids"] = list(
                    dict.fromkeys([*latest_cited, *entries[-1].get("cited_evidence_ids", [])])
                )
            entries[-1].update(last_turn_meta)
        elif final_resp:
            entries.append(
                {
                    "role": "assistant",
                    "content": final_resp,
                    "tool_traces": latest_traces,
                    "cited_evidence_ids": latest_cited,
                    **last_turn_meta,
                }
            )

        return entries

    def _inspect_turn_result(self, *, thread_id: str, company_id: str) -> TurnResult:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        values = snapshot.values or {}

        pending_interrupt = self.get_pending_interrupt(thread_id)
        if pending_interrupt is not None:
            return TurnResult(
                status="awaiting_approval",
                thread_id=thread_id,
                company_id=company_id,
                answer=values.get("draft_answer", ""),
                intent=values.get("intent", ""),
                plan=list(values.get("plan") or []),
                tool_traces=list(values.get("tool_traces") or []),
                evidence_store=dict(values.get("evidence_store") or {}),
                cited_evidence_ids=list(values.get("cited_evidence_ids") or []),
                pending_proposals=list(pending_interrupt.get("proposals") or []),
                blocked_proposals=list(values.get("blocked_proposals") or []),
                executed_actions=[],
                cross_tenant_blocked=bool(values.get("cross_tenant_blocked", False)),
            )

        return TurnResult(
            status="completed",
            thread_id=thread_id,
            company_id=company_id,
            answer=values.get("final_response") or values.get("draft_answer") or "",
            intent=values.get("intent", ""),
            plan=list(values.get("plan") or []),
            tool_traces=list(values.get("tool_traces") or []),
            evidence_store=dict(values.get("evidence_store") or {}),
            cited_evidence_ids=list(values.get("cited_evidence_ids") or []),
            pending_proposals=[],
            blocked_proposals=list(values.get("blocked_proposals") or []),
            executed_actions=list(values.get("executed_actions") or []),
            cross_tenant_blocked=bool(values.get("cross_tenant_blocked", False)),
        )
