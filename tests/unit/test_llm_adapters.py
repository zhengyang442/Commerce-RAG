from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.core.models import SearchResult
from app.generation.evidence import build_evidence_pack
from app.generation.llm.anthropic_httpx import AnthropicHTTPAdapter
from app.generation.llm.contracts import OUTPUT_JSON_SCHEMA
from app.generation.llm.errors import (
    LLMInvalidOutputError,
    LLMModelError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMTimeoutError,
)
from app.generation.llm.openai_compatible_httpx import OpenAICompatibleHTTPAdapter


def pack():
    result = SearchResult(
        rank=1,
        citation_id="P1",
        product_id=1,
        product_name="Blue Chair",
        product_features="Color: Blue",
        score=1.0,
        matched_fields=["product_name", "product_features"],
    )
    return build_evidence_pack(query="blue chair", normalized_query="blue chair", results=[result])


def generated_json() -> str:
    return json.dumps(
        {
            "recommendations": [
                {
                    "citation_id": "P1",
                    "reason": "The product is blue.",
                    "supporting_fields": ["product_features"],
                }
            ]
        }
    )


def test_system_prompt_forbids_even_negative_unavailable_claims() -> None:
    from app.generation.llm.base import SYSTEM_PROMPT

    assert "even to deny" in SYSTEM_PROMPT
    assert "application renders limitations outside your JSON" in SYSTEM_PROMPT


@pytest.mark.anyio
async def test_anthropic_payload_uses_structured_output_without_prompt_cache() -> None:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"request-id": "req_provider"},
            json={
                "model": "configured-model",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": generated_json()}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(
        llm_api_style="anthropic", llm_api_key="secret", llm_model="configured-model"
    )
    adapter = AnthropicHTTPAdapter(settings, client=client)

    result = await adapter.generate(pack())
    await adapter.generate(pack())

    assert result.answer.recommendations[0].citation_id == "P1"
    assert captured[0]["system"] == captured[1]["system"]
    assert "cache_control" not in json.dumps(captured[0])
    assert captured[0]["output_config"]["format"]["schema"] == OUTPUT_JSON_SCHEMA
    assert "secret" not in json.dumps(captured[0])
    await client.aclose()


@pytest.mark.anyio
async def test_openai_payload_normalizes_v1_url_and_uses_schema() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "configured-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": generated_json()},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(
        llm_api_style="openai",
        llm_base_url="https://gateway.example/v1",
        llm_api_key="secret",
        llm_model="configured-model",
    )

    result = await OpenAICompatibleHTTPAdapter(settings, client=client).generate(pack())

    assert seen["url"] == "https://gateway.example/v1/chat/completions"
    assert seen["body"]["response_format"]["json_schema"]["schema"] == OUTPUT_JSON_SCHEMA
    assert result.usage.input_tokens == 10
    await client.aclose()


@pytest.mark.anyio
async def test_deepseek_payload_uses_json_object_and_disables_thinking() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [{"finish_reason": "stop", "message": {"content": generated_json()}}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(
        llm_api_style="openai",
        llm_base_url="https://api.deepseek.com",
        llm_api_key="secret",
        llm_model="deepseek-v4-flash",
    )

    result = await OpenAICompatibleHTTPAdapter(settings, client=client).generate(pack())

    assert seen["response_format"] == {"type": "json_object"}
    assert seen["thinking"] == {"type": "disabled"}
    assert seen["temperature"] == 0
    assert "product_id" in seen["messages"][0]["content"]
    assert result.answer.recommendations[0].citation_id == "P1"
    await client.aclose()


@pytest.mark.anyio
async def test_unconfigured_adapter_never_makes_network_request() -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AnthropicHTTPAdapter(Settings(), client=client)

    with pytest.raises(LLMNotConfiguredError):
        await adapter.generate(pack())
    assert called is False
    await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "body", "error_type"),
    [
        (401, {"error": {"type": "authentication_error"}}, LLMProviderError),
        (429, {"error": {"type": "rate_limit_error"}}, LLMProviderError),
        (500, {"error": {"type": "api_error"}}, LLMProviderError),
        (404, {"error": {"message": "model not found"}}, LLMModelError),
    ],
)
async def test_provider_and_model_errors_are_typed(status, body, error_type) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body))
    )
    settings = Settings(
        llm_api_style="anthropic", llm_api_key="secret", llm_model="configured-model"
    )

    with pytest.raises(error_type) as captured:
        await AnthropicHTTPAdapter(settings, client=client).generate(pack())
    assert "secret" not in str(captured.value)
    await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response_body",
    [
        {"stop_reason": "max_tokens", "content": []},
        {"stop_reason": "refusal", "content": []},
        {"stop_reason": "end_turn", "content": []},
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "not json"}],
        },
        {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": '{"recommendations": [], "unknown": true}',
                }
            ],
        },
    ],
)
async def test_invalid_outputs_are_not_repaired(response_body) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response_body))
    )
    settings = Settings(
        llm_api_style="anthropic", llm_api_key="secret", llm_model="configured-model"
    )

    with pytest.raises(LLMInvalidOutputError):
        await AnthropicHTTPAdapter(settings, client=client).generate(pack())
    await client.aclose()


@pytest.mark.anyio
async def test_total_timeout_is_typed() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        import asyncio

        await asyncio.sleep(0.05)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(
        llm_api_style="anthropic",
        llm_api_key="secret",
        llm_model="configured-model",
        llm_timeout_seconds=0.001,
    )

    with pytest.raises(LLMTimeoutError):
        await AnthropicHTTPAdapter(settings, client=client).generate(pack())
    await client.aclose()
