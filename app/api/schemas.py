from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.models import Citation, Product, RetrievalStrategy, SearchResult, Timing
from app.query_understanding.models import QueryUnderstanding
from app.retrieval.service import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K)
    retrieval_strategy: RetrievalStrategy = "bm25"

    @field_validator("query")
    @classmethod
    def query_must_be_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("查询不能为空")
        return value


class SearchResponse(APIModel):
    request_id: str
    query: str
    normalized_query: str
    results: list[SearchResult]
    latency_ms: float
    retrieval_strategy: RetrievalStrategy
    query_understanding: QueryUnderstanding


class HealthResponse(APIModel):
    request_id: str
    status: str
    data_version: str
    product_count: int
    index_ready: bool
    llm_configured: bool
    vector_index_ready: bool
    vector_count: int
    reranker_ready: bool


class ErrorBody(APIModel):
    request_id: str
    error: dict[str, str]


class ProductResponse(Product):
    pass


class AnswerResponse(APIModel):
    request_id: str
    query: str
    normalized_query: str
    answer: str
    citations: list[Citation]
    limitations: list[str]
    results: list[SearchResult]
    mode: str
    model: str | None = None
    fallback_reason: str | None = None
    timing: Timing
    retrieval_strategy: RetrievalStrategy
    query_understanding: QueryUnderstanding
