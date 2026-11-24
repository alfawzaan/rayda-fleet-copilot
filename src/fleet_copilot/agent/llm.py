"""Provider-agnostic LLM factory with resilient Gemini fallback support."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from fleet_copilot.config import Settings


def _instantiate_model(settings: Settings, model_name: str) -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is required when LLM_PROVIDER=gemini. "
                "Set it in your environment or .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_seconds,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Instantiate the primary ChatModel configured from Settings."""
    return _instantiate_model(settings, settings.llm_model)


def build_fallback_models(settings: Settings) -> list[BaseChatModel]:
    """Instantiate optional fallback models used automatically on transient 503/429 spikes."""
    if settings.llm_provider.lower() != "gemini" or not settings.llm_fallback_models:
        return []
    fallbacks: list[BaseChatModel] = []
    for raw_name in settings.llm_fallback_models.split(","):
        name = raw_name.strip()
        if name and name != settings.llm_model:
            fallbacks.append(_instantiate_model(settings, name))
    return fallbacks
