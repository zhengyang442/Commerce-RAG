from __future__ import annotations

import pytest

from app.core.config import Settings
from app.generation.llm.errors import LLMTimeoutError
from app.query_understanding.models import RewriteOutput
from app.query_understanding.rules import analyze_with_rules, detect_language
from app.query_understanding.service import QueryUnderstandingService


class FakeRewriter:
    def __init__(self, *, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.closed = False

    async def rewrite(self, query: str) -> RewriteOutput:
        if self.error:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        self.closed = True


def test_rules_translate_common_chinese_and_remove_unavailable_intents() -> None:
    analysis = analyze_with_rules("我想找一张实木大号平台床，不需要弹簧床架，价格便宜一点")

    assert analysis.language == "zh"
    assert "solid wood" in analysis.output.retrieval_query
    assert "queen platform bed" in analysis.output.retrieval_query
    assert "platform bed" in analysis.output.retrieval_query
    assert "box spring" not in analysis.output.retrieval_query
    assert analysis.output.excluded_terms == ["box spring"]
    assert analysis.unsupported_intents == ["price"]
    assert "price" not in analysis.output.retrieval_query


def test_language_detection() -> None:
    assert detect_language("蓝色椅子") == "zh"
    assert detect_language("blue chair") == "en"
    assert detect_language("蓝色 velvet chair") == "mixed"


def test_glue_removal_does_not_consume_product_meaning() -> None:
    result = analyze_with_rules("大号床现在多少钱，有没有折扣")

    assert result.output.retrieval_query == "queen bed"
    assert result.has_untranslated_cjk is False
    assert result.unsupported_intents == ["price", "discount"]


@pytest.mark.anyio
async def test_english_query_keeps_original_without_calling_rewriter() -> None:
    called = False

    def factory(_):
        nonlocal called
        called = True
        return FakeRewriter()

    result = await QueryUnderstandingService(
        Settings(
            llm_api_style="openai",
            llm_base_url="https://example.test",
            llm_api_key="x",
            llm_model="m",
        ),
        rewriter_factory=factory,
    ).understand("solid wood queen bed price")

    assert called is False
    assert result.rewrite_source == "rules"
    assert result.retrieval_query == "solid wood queen bed"
    assert result.unsupported_intents == ["price"]


@pytest.mark.anyio
async def test_uncovered_chinese_query_uses_structured_rewriter() -> None:
    adapter = FakeRewriter(
        result=RewriteOutput(
            retrieval_query="blue velvet accent chair gold legs",
            category_terms=["accent chair"],
            attributes={"color": ["blue"], "material": ["velvet"]},
            excluded_terms=[],
        )
    )
    settings = Settings(
        llm_api_style="openai",
        llm_base_url="https://example.test",
        llm_api_key="x",
        llm_model="m",
    )
    result = await QueryUnderstandingService(
        settings, rewriter_factory=lambda _: adapter
    ).understand("给学龄儿童使用的人体工学座椅")

    assert result.rewrite_source == "llm"
    assert result.retrieval_query == "blue velvet accent chair gold legs"
    assert result.detected_language == "zh"
    assert adapter.closed is True


@pytest.mark.anyio
async def test_fully_covered_chinese_query_prefers_fast_rules() -> None:
    called = False

    def factory(_):
        nonlocal called
        called = True
        return FakeRewriter()

    settings = Settings(
        llm_api_style="openai",
        llm_base_url="https://example.test",
        llm_api_key="x",
        llm_model="m",
    )
    result = await QueryUnderstandingService(settings, rewriter_factory=factory).understand(
        "蓝色天鹅绒休闲椅，带金色椅腿"
    )

    assert called is False
    assert result.rewrite_source == "rules"
    assert result.retrieval_query == "blue velvet accent chair gold legs"


@pytest.mark.anyio
async def test_rewrite_failure_falls_back_to_rules() -> None:
    adapter = FakeRewriter(error=LLMTimeoutError())
    settings = Settings(
        llm_api_style="openai",
        llm_base_url="https://example.test",
        llm_api_key="x",
        llm_model="m",
    )
    result = await QueryUnderstandingService(
        settings, rewriter_factory=lambda _: adapter
    ).understand("给学龄儿童使用的人体工学座椅")

    assert result.rewrite_source == "rules_fallback"
    assert result.fallback_reason == "timeout"
    assert result.retrieval_query == "furniture"
    assert adapter.closed is True
