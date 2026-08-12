from __future__ import annotations

from app.query_understanding.rules import analyze_with_rules


def test_chinese_cross_language_demo_story() -> None:
    result = analyze_with_rules("蓝色天鹅绒休闲椅，带金色椅腿")

    assert result.output.retrieval_query == "blue velvet accent chair gold legs"
    assert result.language == "zh"
    assert result.unsupported_intents == []


def test_multi_condition_exclusion_demo_story() -> None:
    result = analyze_with_rules("我想找一张实木大号平台床，不需要弹簧床架")

    assert "solid wood" in result.output.retrieval_query
    assert "queen platform bed" in result.output.retrieval_query
    assert result.output.excluded_terms == ["box spring"]


def test_safe_price_refusal_demo_story() -> None:
    result = analyze_with_rules("适合四个人的小型圆形餐桌，现在多少钱，有没有折扣")

    assert "small round dining table" in result.output.retrieval_query
    assert "seats four" in result.output.retrieval_query
    assert result.unsupported_intents == ["price", "discount"]
    assert "price" not in result.output.retrieval_query
