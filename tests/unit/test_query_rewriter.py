from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.query_understanding.rewriter import ProviderQueryRewriter


@pytest.mark.anyio
async def test_deepseek_rewriter_uses_json_object_and_no_thinking() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "retrieval_query": "solid wood queen platform bed",
                                    "category_terms": ["platform bed"],
                                    "attributes": {"material": ["solid wood"]},
                                    "excluded_terms": ["box spring"],
                                }
                            )
                        },
                    }
                ],
            },
        )

    settings = Settings(
        llm_api_style="openai",
        llm_base_url="https://api.deepseek.com",
        llm_api_key="test",
        llm_model="deepseek-v4-flash",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    output = await ProviderQueryRewriter(settings, client=client).rewrite("实木大号平台床")
    await client.aclose()

    assert output.retrieval_query == "solid wood queen platform bed"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["temperature"] == 0
