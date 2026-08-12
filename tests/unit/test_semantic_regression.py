from __future__ import annotations

from app.core.models import SearchResult
from scripts.evaluate_semantic_regression import has_expected_class


def result(product_class: str | None) -> SearchResult:
    return SearchResult(
        product_id=1,
        product_name="Product",
        product_class=product_class,
        product_features="features",
        rank=1,
        citation_id="P1",
        score=1.0,
    )


def test_expected_class_match_is_case_insensitive_and_respects_cutoff() -> None:
    results = [result("Accent Chairs"), result("End Tables|Nightstands")]

    assert has_expected_class(results, ["nightstands"], 1) is False
    assert has_expected_class(results, ["nightstands"], 3) is True
    assert has_expected_class(results, ["Sofas"], 3) is False
