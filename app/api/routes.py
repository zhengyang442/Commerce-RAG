from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.schemas import (
    AnswerResponse,
    HealthResponse,
    ProductResponse,
    SearchRequest,
    SearchResponse,
)
from app.core.errors import (
    CommerceRAGError,
    EmptyQueryError,
    IndexNotReadyError,
    InvalidTopKError,
    NoSearchTokensError,
    ProductNotFoundError,
)
from app.core.request_id import new_request_id
from app.generation.orchestrator import AnswerOrchestrator
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.embedding import reranker_cache_ready
from app.retrieval.service import RetrievalService
from app.retrieval.sqlite_store import SQLiteStore
from app.retrieval.vector_store import VectorStore

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    status = SQLiteStore(settings.index_path).status()
    vector_status = VectorStore(
        settings.vector_index_path,
        embedding_cache_dir=settings.embedding_cache_dir,
    ).status()
    return HealthResponse(
        request_id=new_request_id(),
        status="ok" if status["index_ready"] else "degraded",
        data_version=str(status["data_version"]),
        product_count=int(status["product_count"]),
        index_ready=bool(status["index_ready"]),
        llm_configured=settings.llm_configured,
        vector_index_ready=bool(vector_status["vector_index_ready"]),
        vector_count=int(vector_status["vector_count"]),
        reranker_ready=reranker_cache_ready(settings.reranker_cache_dir),
    )


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse | JSONResponse:
    request_id = new_request_id()
    try:
        settings = request.app.state.settings
        understanding = await QueryUnderstandingService(
            settings,
            rewriter_factory=request.app.state.query_rewriter_factory,
            external_call_semaphore=request.app.state.external_call_semaphore,
        ).understand(payload.query)
        result = RetrievalService(
            settings.index_path,
            vector_index_path=settings.vector_index_path,
            embedding_cache_dir=settings.embedding_cache_dir,
            reranker_cache_dir=settings.reranker_cache_dir,
        ).search(
            payload.query,
            payload.top_k,
            payload.retrieval_strategy,
            retrieval_query=understanding.retrieval_query,
        )
    except CommerceRAGError as error:
        return api_error(error, request_id)
    return SearchResponse(
        request_id=request_id,
        query=result.query,
        normalized_query=result.normalized_query,
        results=result.results,
        latency_ms=result.latency_ms,
        retrieval_strategy=result.retrieval_strategy,
        query_understanding=understanding,
    )


@router.post("/answer", response_model=AnswerResponse)
async def answer(payload: SearchRequest, request: Request) -> AnswerResponse | JSONResponse:
    request_id = new_request_id()
    try:
        result = await AnswerOrchestrator(
            request.app.state.settings,
            adapter_factory=request.app.state.llm_adapter_factory,
            query_rewriter_factory=request.app.state.query_rewriter_factory,
            external_call_semaphore=request.app.state.external_call_semaphore,
        ).answer(payload.query, payload.top_k, payload.retrieval_strategy)
    except CommerceRAGError as error:
        return api_error(error, request_id)
    return AnswerResponse(
        request_id=request_id,
        query=result.query,
        normalized_query=result.normalized_query,
        answer=result.answer,
        citations=result.citations,
        limitations=result.limitations,
        results=result.results,
        mode=result.mode,
        model=result.model,
        fallback_reason=result.fallback_reason,
        timing=result.timing,
        retrieval_strategy=result.retrieval_strategy,
        query_understanding=result.query_understanding,
    )


@router.get("/products/{product_id}", response_model=ProductResponse)
def product(product_id: int, request: Request) -> ProductResponse | JSONResponse:
    request_id = new_request_id()
    try:
        value = SQLiteStore(request.app.state.settings.index_path).get_product(product_id)
    except CommerceRAGError as error:
        return api_error(error, request_id)
    return ProductResponse.model_validate(value.model_dump())


def api_error(error: CommerceRAGError, request_id: str) -> JSONResponse:
    if isinstance(error, (EmptyQueryError, NoSearchTokensError, InvalidTopKError)):
        status_code, code = 422, "invalid_query"
    elif isinstance(error, IndexNotReadyError):
        status_code, code = 503, "index_not_ready"
    elif isinstance(error, ProductNotFoundError):
        status_code, code = 404, "product_not_found"
    else:
        status_code, code = 500, "internal_error"
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "error": {"code": code, "message": str(error)},
        },
    )
