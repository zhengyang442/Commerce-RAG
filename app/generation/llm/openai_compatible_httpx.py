from __future__ import annotations

import httpx

from app.core.config import Settings
from app.generation.evidence import EvidencePack
from app.generation.llm.base import (
    SYSTEM_PROMPT,
    parse_generated_answer,
    post_json,
    prompt_payload,
    raise_for_provider_status,
)
from app.generation.llm.contracts import OUTPUT_JSON_SCHEMA, LLMResult, LLMUsage
from app.generation.llm.errors import LLMInvalidOutputError, LLMNotConfiguredError


class OpenAICompatibleHTTPAdapter:
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
        self._owns_client = client is None

    async def generate(self, pack: EvidencePack) -> LLMResult:
        if (
            self.settings.llm_api_style != "openai"
            or not self.settings.llm_base_url
            or not self.settings.llm_api_key
            or not self.settings.llm_model
        ):
            raise LLMNotConfiguredError()
        payload = {
            "model": self.settings.llm_model,
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_payload(pack)},
            ],
            "response_format": self._response_format(),
        }
        thinking = self._thinking_mode()
        if thinking is not None:
            payload["thinking"] = {"type": thinking}
        response = await post_json(
            self.client,
            url=_chat_completions_url(self.settings.llm_base_url),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.settings.llm_api_key}",
            },
            payload=payload,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )
        raise_for_provider_status(response)
        provider_request_id = response.headers.get("x-request-id")
        try:
            body = response.json()
            choices = body.get("choices") or []
            if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                raise LLMInvalidOutputError(provider_request_id=provider_request_id)
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise LLMInvalidOutputError(provider_request_id=provider_request_id)
            usage = body.get("usage") or {}
            return LLMResult(
                answer=parse_generated_answer(content),
                model=str(body.get("model") or self.settings.llm_model),
                provider_request_id=provider_request_id,
                usage=LLMUsage(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                ),
            )
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            raise LLMInvalidOutputError(provider_request_id=provider_request_id) from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _response_format(self) -> dict[str, object]:
        mode = self.settings.llm_structured_output
        if mode == "auto":
            mode = "json_object" if self._is_deepseek() else "json_schema"
        if mode == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "commerce_rag_answer",
                "strict": True,
                "schema": OUTPUT_JSON_SCHEMA,
            },
        }

    def _thinking_mode(self) -> str | None:
        mode = self.settings.llm_thinking
        if mode == "auto":
            return "disabled" if self._is_deepseek() else None
        return mode

    def _is_deepseek(self) -> bool:
        base_url = (self.settings.llm_base_url or "").lower()
        model = (self.settings.llm_model or "").lower()
        return "deepseek.com" in base_url or model.startswith("deepseek-")


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"
