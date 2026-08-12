from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from app.core.models import SearchResult, StrictModel
from app.generation.policy import EVIDENCE_FIELDS, UNAVAILABLE_FIELDS

EVIDENCE_SCHEMA_VERSION = "1"
TRUNCATION_LIMITS = {
    "product_name": 300,
    "product_class": 300,
    "category_hierarchy": 500,
    "product_description": 1500,
    "product_features": 2000,
}


class EvidenceItem(StrictModel):
    citation_id: str
    rank: int
    product_id: int
    product_name: str
    product_class: str | None = None
    category_hierarchy: str | None = None
    product_description: str | None = None
    product_features: str
    rating_count: int | None = None
    average_rating: float | None = None
    review_count: int | None = None
    score: float
    matched_fields: list[str]
    truncated_fields: list[str] = Field(default_factory=list)

    @field_validator("citation_id")
    @classmethod
    def citation_format(cls, value: str) -> str:
        if not value.startswith("P") or not value[1:].isdigit():
            raise ValueError("引用编号格式无效")
        return value


class EvidencePack(StrictModel):
    schema_version: str
    query: str
    normalized_query: str
    items: list[EvidenceItem]
    unavailable_fields: list[str]
    truncation_rules: dict[str, int]
    max_explained_items: int


def build_evidence_pack(
    *, query: str, normalized_query: str, results: list[SearchResult]
) -> EvidencePack:
    items: list[EvidenceItem] = []
    for expected_rank, result in enumerate(results, start=1):
        if result.rank != expected_rank or result.citation_id != f"P{expected_rank}":
            raise ValueError("检索结果的排名或引用编号不连续")
        result_data = result.model_dump()
        if not set(EVIDENCE_FIELDS).issubset(result_data):
            raise ValueError("检索结果字段不符合 EvidencePack 白名单")
        raw = {field: result_data[field] for field in EVIDENCE_FIELDS}
        truncated_fields: list[str] = []
        for field, limit in TRUNCATION_LIMITS.items():
            value = raw[field]
            if value is not None and len(value) > limit:
                raw[field] = value[:limit]
                truncated_fields.append(field)
        raw["truncated_fields"] = truncated_fields
        items.append(EvidenceItem.model_validate(raw))
    return EvidencePack(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        query=query,
        normalized_query=normalized_query,
        items=items,
        unavailable_fields=list(UNAVAILABLE_FIELDS),
        truncation_rules=dict(TRUNCATION_LIMITS),
        max_explained_items=min(5, len(items)),
    )


def evidence_as_prompt_data(pack: EvidencePack) -> dict[str, Any]:
    """Return deterministic plain data; product text remains untrusted evidence."""
    return pack.model_dump(mode="json")
