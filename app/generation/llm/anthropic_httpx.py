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

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"


class AnthropicHTTPAdapter:
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=_timeout(settings.llm_timeout_seconds))
        self._owns_client = client is None

    async def generate(self, pack: EvidencePack) -> LLMResult:
        if (
            self.settings.llm_api_style != "anthropic"
            or not self.settings.llm_api_key
            or not self.settings.llm_model
        ):
            raise LLMNotConfiguredError()
        base_url = (self.settings.llm_base_url or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
        response = await post_json(
            self.client,
            url=f"{base_url}/v1/messages",
            headers={
                "content-type": "application/json",
                "x-api-key": self.settings.llm_api_key,
                "anthropic-version": "2023-06-01",
            },
            payload={
                "model": self.settings.llm_model,
                "max_tokens": 2048,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt_payload(pack)}],
                "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_JSON_SCHEMA}},
            },
            timeout_seconds=self.settings.llm_timeout_seconds,
        )
        raise_for_provider_status(response)
        provider_request_id = response.headers.get("request-id")
        try:
            body = response.json()
            if body.get("stop_reason") != "end_turn":
                raise LLMInvalidOutputError(provider_request_id=provider_request_id)
            text_blocks = [
                block.get("text")
                for block in body.get("content", [])
                if block.get("type") == "text" and isinstance(block.get("text"), str)
            ]
            if len(text_blocks) != 1:
                raise LLMInvalidOutputError(provider_request_id=provider_request_id)
            usage = body.get("usage") or {}
            return LLMResult(
                answer=parse_generated_answer(text_blocks[0]),
                model=str(body.get("model") or self.settings.llm_model),
                provider_request_id=provider_request_id,
                usage=LLMUsage(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                ),
            )
        except (ValueError, TypeError, AttributeError) as error:
            raise LLMInvalidOutputError(provider_request_id=provider_request_id) from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


def _timeout(total: float) -> httpx.Timeout:
    return httpx.Timeout(timeout=total, connect=min(10.0, total), pool=min(10.0, total))
