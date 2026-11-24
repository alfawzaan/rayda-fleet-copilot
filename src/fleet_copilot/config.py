"""Application configuration loaded from environment variables and defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for Fleet Copilot."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM configuration
    llm_provider: Literal["gemini", "openai", "anthropic"] = Field(default="gemini")
    llm_model: str = Field(default="gemini-3.6-flash")
    llm_fallback_models: str = Field(default="gemini-3.5-flash-lite,gemini-3.1-flash-lite")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=0, ge=0, le=5)
    llm_timeout_seconds: float = Field(default=45.0, gt=0.0)

    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)

    # Storage paths
    telemetry_dataset_path: Path = Field(
        default=PROJECT_ROOT / "data" / "device-telemetry-dataset.ndjson"
    )
    telemetry_db_path: Path = Field(default=PROJECT_ROOT / "runtime" / "fleet_copilot.db")
    checkpoint_db_path: Path = Field(default=PROJECT_ROOT / "runtime" / "langgraph_checkpoints.db")
    audit_log_jsonl_path: Path = Field(default=PROJECT_ROOT / "runtime" / "audit_log.jsonl")

    # Agent execution bounds
    enable_llm_planner: bool = Field(default=True)
    max_agent_steps: int = Field(default=8, ge=1, le=25)
    max_sql_rows: int = Field(default=100, ge=1, le=500)

    # Optional external action integrations (real HTTP dispatchers)
    action_webhook_url: str | None = Field(default=None)
    action_webhook_secret: str | None = Field(default=None)
    slack_webhook_url: str | None = Field(default=None)
    webhook_timeout_seconds: float = Field(default=10.0, gt=0.0)

    def ensure_runtime_dirs(self) -> None:
        """Create parent directories for runtime persistence files."""
        for path in (
            self.telemetry_db_path,
            self.checkpoint_db_path,
            self.audit_log_jsonl_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
