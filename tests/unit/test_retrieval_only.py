from __future__ import annotations

from app.core.models import SearchResult
from app.generation.retrieval_only import render_retrieval_only


def make_result() -> SearchResult:
    return SearchResult(
        rank=1,
        citation_id="P1",
        product_id=1,
        product_name="Chair",
        product_features="Blue",
        score=1.0,
    )


def test_not_configured_keeps_results_as_separate_evidence() -> None:
    content = render_retrieval_only([make_result()])

    assert "未配置生成模型" in content.answer
    assert "价格" in content.limitations[0]
    assert all("Chair" not in limitation for limitation in content.limitations)


def test_generation_failure_and_empty_results_are_explicit() -> None:
    failed = render_retrieval_only([make_result()], reason="timeout")
    empty = render_retrieval_only([])

    assert "已保留全部检索结果" in failed.answer
    assert "不足以判断" in empty.answer
