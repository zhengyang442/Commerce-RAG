from __future__ import annotations

import pytest

from app.core.models import SearchResult
from app.generation.citation_validator import validate_generated_answer
from app.generation.evidence import build_evidence_pack
from app.generation.llm.contracts import GeneratedAnswer, GeneratedRecommendation
from app.generation.llm.errors import LLMInvalidOutputError


def pack():
    result = SearchResult(
        rank=1,
        citation_id="P1",
        product_id=10,
        product_name="Blue Chair",
        product_class=None,
        product_features="Color: Blue",
        score=1.0,
    )
    return build_evidence_pack(query="blue chair", normalized_query="blue chair", results=[result])


def answer(citation_id="P1", reason="It is blue.", fields=None, *, duplicate=False):
    recommendation = GeneratedRecommendation(
        citation_id=citation_id,
        reason=reason,
        supporting_fields=fields or ["product_features"],
    )
    items = [recommendation, recommendation] if duplicate else [recommendation]
    return GeneratedAnswer(recommendations=items)


def test_valid_citation_maps_to_local_product_and_fields() -> None:
    citations = validate_generated_answer(answer(), pack())

    assert citations[0].citation_id == "P1"
    assert citations[0].product_id == 10
    assert citations[0].supporting_fields == ["product_features"]


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        (answer(citation_id="P9"), "不属于"),
        (answer(duplicate=True), "重复引用"),
        (answer(fields=["product_class"]), "缺失字段"),
        (answer(reason="It costs $99."), "禁区"),
        (answer(reason="现货库存充足。"), "禁区"),
    ],
)
def test_invalid_citations_and_forbidden_claims_are_rejected(generated, message) -> None:
    with pytest.raises(LLMInvalidOutputError, match=message):
        validate_generated_answer(generated, pack())
