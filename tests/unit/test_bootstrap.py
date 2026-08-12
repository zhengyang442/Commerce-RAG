from __future__ import annotations

import logging

import pytest

from app.core.config import ConfigError, Settings
from app.core.logging import SecretRedactionFilter
from app.main import create_app


def test_app_boots_without_llm_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RAG_LLM_API_STYLE",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "RAG_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()
    app = create_app(settings)

    assert settings.llm_configured is False
    assert app.title == "CommerceRAG"


def test_invalid_api_style_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_API_STYLE", "unsupported")

    with pytest.raises(ConfigError, match="anthropic"):
        Settings.from_env()


def test_anthropic_configuration_can_use_default_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LLM_API_STYLE", "anthropic")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("RAG_LLM_MODEL", "configured-by-user")
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)

    assert Settings.from_env().llm_configured is True


def test_openai_configuration_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_API_STYLE", "openai")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("RAG_LLM_MODEL", "configured-by-user")
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)

    assert Settings.from_env().llm_configured is False


def test_invalid_structured_output_and_thinking_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LLM_STRUCTURED_OUTPUT", "xml")
    with pytest.raises(ConfigError, match="STRUCTURED_OUTPUT"):
        Settings.from_env()


def test_public_preview_settings_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PUBLIC_PREVIEW", "true")
    monkeypatch.setenv("RAG_ALLOWED_ORIGINS", "https://example.com,https://www.example.com")
    monkeypatch.setenv("RAG_ALLOWED_HOSTS", "example.com,www.example.com")
    monkeypatch.setenv("RAG_RATE_LIMIT_PER_MINUTE", "12")
    monkeypatch.setenv("RAG_ANSWER_RATE_LIMIT_PER_MINUTE", "4")
    monkeypatch.setenv("RAG_EXTERNAL_CALLS_PER_DAY", "25")

    settings = Settings.from_env()

    assert settings.public_preview is True
    assert settings.allowed_origins == ("https://example.com", "https://www.example.com")
    assert settings.allowed_hosts == ("example.com", "www.example.com")
    assert settings.rate_limit_per_minute == 12
    assert settings.answer_rate_limit_per_minute == 4
    assert settings.external_calls_per_day == 25


def test_invalid_public_preview_setting_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PUBLIC_PREVIEW", "maybe")
    with pytest.raises(ConfigError, match="PUBLIC_PREVIEW"):
        Settings.from_env()

    monkeypatch.setenv("RAG_LLM_STRUCTURED_OUTPUT", "auto")
    monkeypatch.setenv("RAG_LLM_THINKING", "sometimes")
    with pytest.raises(ConfigError, match="THINKING"):
        Settings.from_env()


def test_secret_filter_redacts_log_messages() -> None:
    record = logging.LogRecord("test", logging.ERROR, __file__, 1, "key=secret", (), None)

    assert SecretRedactionFilter(["secret"]).filter(record) is True
    assert record.getMessage() == "key=[REDACTED]"
