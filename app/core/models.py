from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Product(StrictModel):
    product_id: int
    product_name: str
    product_class: str | None = None
    category_hierarchy: str | None = None
    product_description: str | None = None
    product_features: str
    rating_count: int | None = None
    average_rating: float | None = None
    review_count: int | None = None


class SearchResult(Product):
    rank: int
    citation_id: str
    score: float
    matched_fields: list[str] = Field(default_factory=list)
    retrieval_sources: list[str] = Field(default_factory=list)
    source_ranks: dict[str, int] = Field(default_factory=dict)
    fusion_score: float | None = None
    reranker_score: float | None = None


class Timing(StrictModel):
    rewrite_ms: float = 0.0
    retrieval_ms: float = 0.0
    evidence_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class Citation(StrictModel):
    citation_id: str
    product_id: int
    supporting_fields: list[str]


AnswerMode = Literal["rag", "retrieval_only"]
FallbackReason = Literal[
    "not_configured",
    "timeout",
    "provider_error",
    "model_error",
    "invalid_output",
]
RetrievalStrategy = Literal["bm25", "vector", "hybrid", "rerank"]
