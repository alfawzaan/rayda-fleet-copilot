"""Shared pytest fixtures and deterministic scripted ChatModel for LangGraph tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from fleet_copilot.actions.audit import AuditLogger
from fleet_copilot.actions.backend import ActionExecutor
from fleet_copilot.config import PROJECT_ROOT, Settings
from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.ingestion import ingest_ndjson_file

DATASET_PATH = PROJECT_ROOT / "data" / "device-telemetry-dataset.ndjson"


class ScriptedChatModel(BaseChatModel):
    """Deterministic BaseChatModel returning scripted AIMessages for offline graph tests."""

    responses: list[AIMessage]
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-test-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        # Return self so scripted tool_calls flow directly through the real LangGraph loop
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.call_index >= len(self.responses):
            msg = AIMessage(content="Fallback final response.")
        else:
            msg = self.responses[self.call_index]
            self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture()
def temp_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="gemini",
        llm_model="gemini-3.6-flash",
        google_api_key="test-key-not-used-offline",
        telemetry_dataset_path=DATASET_PATH,
        telemetry_db_path=tmp_path / "fleet.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        audit_log_jsonl_path=tmp_path / "audit.jsonl",
    )


@pytest.fixture()
def populated_db(tmp_path: Path) -> FleetDatabase:
    db = FleetDatabase(tmp_path / "telemetry.db")
    report = ingest_ndjson_file(db, DATASET_PATH)
    assert report.processed == 750
    assert not report.errors
    return db


@pytest.fixture()
def audit_logger(populated_db: FleetDatabase, tmp_path: Path) -> AuditLogger:
    return AuditLogger(populated_db, tmp_path / "audit.jsonl")


@pytest.fixture()
def action_executor(populated_db: FleetDatabase, audit_logger: AuditLogger) -> ActionExecutor:
    return ActionExecutor(populated_db, audit_logger)
