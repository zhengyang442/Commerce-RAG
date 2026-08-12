from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from app.core.models import FallbackReason, StrictModel

DetectedLanguage = Literal["zh", "en", "mixed", "other"]
RewriteSource = Literal["original", "rules", "llm", "rules_fallback"]
UnsupportedIntent = Literal[
    "price",
    "discount",
    "inventory",
    "delivery",
    "return_policy",
    "warranty",
    "review_text",
]


class RewriteOutput(StrictModel):
    retrieval_query: str = Field(min_length=1, max_length=500)
    category_terms: list[str] = Field(default_factory=list, max_length=8)
    attributes: dict[str, list[str]] = Field(default_factory=dict)
    excluded_terms: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("category_terms", "excluded_terms")
    @classmethod
    def bounded_terms(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            normalized = " ".join(value.split())
            if normalized and len(normalized) <= 80 and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    @field_validator("attributes")
    @classmethod
    def bounded_attributes(cls, values: dict[str, list[str]]) -> dict[str, list[str]]:
        allowed = {"category", "color", "material", "size", "capacity", "style", "feature"}
        cleaned: dict[str, list[str]] = {}
        for key, items in values.items():
            if key not in allowed:
                continue
            normalized_items = []
            for item in items[:12]:
                normalized = " ".join(item.split())
                if normalized and len(normalized) <= 80 and normalized not in normalized_items:
                    normalized_items.append(normalized)
            if normalized_items:
                cleaned[key] = normalized_items
        return cleaned


class QueryUnderstanding(StrictModel):
    detected_language: DetectedLanguage
    retrieval_query: str = Field(min_length=1, max_length=500)
    category_terms: list[str] = Field(default_factory=list)
    attributes: dict[str, list[str]] = Field(default_factory=dict)
    excluded_terms: list[str] = Field(default_factory=list)
    unsupported_intents: list[UnsupportedIntent] = Field(default_factory=list)
    rewrite_source: RewriteSource
    rewrite_latency_ms: float = 0.0
    fallback_reason: FallbackReason | None = None


REWRITE_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "retrieval_query": {"type": "string", "minLength": 1, "maxLength": 500},
        "category_terms": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "attributes": {
            "type": "object",
            "properties": {
                name: {"type": "array", "items": {"type": "string"}, "maxItems": 12}
                for name in (
                    "category",
                    "color",
                    "material",
                    "size",
                    "capacity",
                    "style",
                    "feature",
                )
            },
            "additionalProperties": False,
        },
        "excluded_terms": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string"},
        },
    },
    "required": ["retrieval_query", "category_terms", "attributes", "excluded_terms"],
    "additionalProperties": False,
}
