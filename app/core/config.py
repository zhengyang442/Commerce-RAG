from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "wands"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "catalog.sqlite"
DEFAULT_VECTOR_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "catalog_vectors.sqlite"
DEFAULT_EMBEDDING_CACHE_DIR = PROJECT_ROOT / "data" / "models" / "fastembed"
DEFAULT_RERANKER_CACHE_DIR = PROJECT_ROOT / "data" / "models" / "fastembed-reranker"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class ConfigError(ValueError):
    """Raised when an explicitly supplied setting is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    index_path: Path = DEFAULT_INDEX_PATH
    vector_index_path: Path = DEFAULT_VECTOR_INDEX_PATH
    embedding_cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR
    reranker_cache_dir: Path = DEFAULT_RERANKER_CACHE_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    llm_api_style: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_structured_output: str = "auto"
    llm_thinking: str = "auto"
    log_full_queries: bool = False
    query_rewrite_enabled: bool = True
    query_rewrite_timeout_seconds: float = 8.0
    public_preview: bool = False
    allowed_origins: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    rate_limit_per_minute: int = 30
    max_request_bytes: int = 16_384
    max_external_concurrency: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        api_style = _optional_env("RAG_LLM_API_STYLE")
        if api_style is not None:
            api_style = api_style.lower()
            if api_style not in {"anthropic", "openai"}:
                raise ConfigError("RAG_LLM_API_STYLE 必须是 anthropic 或 openai")

        timeout_raw = os.getenv("RAG_LLM_TIMEOUT_SECONDS", "30").strip()
        try:
            timeout = float(timeout_raw)
        except ValueError as error:
            raise ConfigError("RAG_LLM_TIMEOUT_SECONDS 必须是正数") from error
        if timeout <= 0:
            raise ConfigError("RAG_LLM_TIMEOUT_SECONDS 必须是正数")

        log_full_queries = os.getenv("RAG_LOG_FULL_QUERIES", "").strip().lower()
        if log_full_queries not in {"", "0", "1", "false", "true"}:
            raise ConfigError("RAG_LOG_FULL_QUERIES 必须是 true/false 或 1/0")

        structured_output = os.getenv("RAG_LLM_STRUCTURED_OUTPUT", "auto").strip().lower()
        if structured_output not in {"auto", "json_schema", "json_object"}:
            raise ConfigError("RAG_LLM_STRUCTURED_OUTPUT 必须是 auto、json_schema 或 json_object")

        thinking = os.getenv("RAG_LLM_THINKING", "auto").strip().lower()
        if thinking not in {"auto", "enabled", "disabled"}:
            raise ConfigError("RAG_LLM_THINKING 必须是 auto、enabled 或 disabled")

        rewrite_enabled = _boolean_env("RAG_QUERY_REWRITE_ENABLED", True)
        rewrite_timeout = _positive_float_env("RAG_QUERY_REWRITE_TIMEOUT_SECONDS", 8.0)
        public_preview = _boolean_env("RAG_PUBLIC_PREVIEW", False)
        rate_limit = _positive_int_env("RAG_RATE_LIMIT_PER_MINUTE", 30)
        max_request_bytes = _positive_int_env("RAG_MAX_REQUEST_BYTES", 16_384)
        max_external_concurrency = _positive_int_env("RAG_MAX_EXTERNAL_CONCURRENCY", 4)

        return cls(
            llm_api_style=api_style,
            llm_base_url=_optional_env("RAG_LLM_BASE_URL"),
            llm_api_key=_optional_env("RAG_LLM_API_KEY"),
            llm_model=_optional_env("RAG_LLM_MODEL"),
            llm_timeout_seconds=timeout,
            llm_structured_output=structured_output,
            llm_thinking=thinking,
            log_full_queries=log_full_queries in {"1", "true"},
            query_rewrite_enabled=rewrite_enabled,
            query_rewrite_timeout_seconds=rewrite_timeout,
            public_preview=public_preview,
            allowed_origins=_csv_env("RAG_ALLOWED_ORIGINS"),
            allowed_hosts=_csv_env("RAG_ALLOWED_HOSTS") or ("127.0.0.1", "localhost", "testserver"),
            rate_limit_per_minute=rate_limit,
            max_request_bytes=max_request_bytes,
            max_external_concurrency=max_external_concurrency,
        )

    @property
    def llm_configured(self) -> bool:
        if not self.llm_api_style or not self.llm_api_key or not self.llm_model:
            return False
        return self.llm_api_style == "anthropic" or bool(self.llm_base_url)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _boolean_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized not in {"0", "1", "false", "true"}:
        raise ConfigError(f"{name} 必须是 true/false 或 1/0")
    return normalized in {"1", "true"}


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigError(f"{name} 必须是正数") from error
    if value <= 0:
        raise ConfigError(f"{name} 必须是正数")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} 必须是正整数") from error
    if value <= 0:
        raise ConfigError(f"{name} 必须是正整数")
    return value


def _csv_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
