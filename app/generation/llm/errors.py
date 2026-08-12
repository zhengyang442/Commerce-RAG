from __future__ import annotations

from typing import Literal

LLMErrorKind = Literal[
    "not_configured",
    "quota_exhausted",
    "timeout",
    "provider_error",
    "model_error",
    "invalid_output",
]


class LLMError(Exception):
    def __init__(self, kind: LLMErrorKind, message: str, *, provider_request_id: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.provider_request_id = provider_request_id


class LLMNotConfiguredError(LLMError):
    def __init__(self, message: str = "生成模型未配置") -> None:
        super().__init__("not_configured", message)


class LLMQuotaExceededError(LLMError):
    def __init__(self, message: str = "今日生成额度已用完") -> None:
        super().__init__("quota_exhausted", message)


class LLMTimeoutError(LLMError):
    def __init__(self, message: str = "生成服务超时", **kwargs: str | None) -> None:
        super().__init__("timeout", message, **kwargs)


class LLMProviderError(LLMError):
    def __init__(self, message: str = "生成供应商请求失败", **kwargs: str | None) -> None:
        super().__init__("provider_error", message, **kwargs)


class LLMModelError(LLMError):
    def __init__(self, message: str = "生成模型不可用", **kwargs: str | None) -> None:
        super().__init__("model_error", message, **kwargs)


class LLMInvalidOutputError(LLMError):
    def __init__(self, message: str = "生成结果结构无效", **kwargs: str | None) -> None:
        super().__init__("invalid_output", message, **kwargs)
