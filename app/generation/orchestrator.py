from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from app.core.config import Settings
from app.core.models import Citation, RetrievalStrategy, SearchResult, Timing
from app.core.usage_limits import DailyExternalCallBudget
from app.generation.answer_renderer import render_rag_answer
from app.generation.citation_validator import validate_generated_answer
from app.generation.evidence import build_evidence_pack
from app.generation.llm.anthropic_httpx import AnthropicHTTPAdapter
from app.generation.llm.contracts import LLMAdapter
from app.generation.llm.errors import LLMError, LLMInvalidOutputError, LLMQuotaExceededError
from app.generation.llm.openai_compatible_httpx import OpenAICompatibleHTTPAdapter
from app.generation.policy import FIXED_LIMITATIONS
from app.generation.retrieval_only import render_retrieval_only
from app.query_understanding.models import QueryUnderstanding
from app.query_understanding.service import QueryUnderstandingService, RewriterFactory
from app.retrieval.service import RetrievalService

AdapterFactory = Callable[[Settings], LLMAdapter]
LOGGER = logging.getLogger(__name__)


class AnswerOutcome:
    def __init__(
        self,
        *,
        query: str,
        normalized_query: str,
        answer: str,
        citations: list[Citation],
        limitations: list[str],
        results: list[SearchResult],
        mode: str,
        model: str | None,
        fallback_reason: str | None,
        timing: Timing,
        retrieval_strategy: RetrievalStrategy,
        query_understanding: QueryUnderstanding,
    ) -> None:
        self.query = query
        self.normalized_query = normalized_query
        self.answer = answer
        self.citations = citations
        self.limitations = limitations
        self.results = results
        self.mode = mode
        self.model = model
        self.fallback_reason = fallback_reason
        self.timing = timing
        self.retrieval_strategy = retrieval_strategy
        self.query_understanding = query_understanding


class AnswerOrchestrator:
    def __init__(
        self,
        settings: Settings,
        *,
        adapter_factory: AdapterFactory | None = None,
        query_rewriter_factory: RewriterFactory | None = None,
        external_call_semaphore: asyncio.Semaphore | None = None,
        external_call_budget: DailyExternalCallBudget | None = None,
    ) -> None:
        self.settings = settings
        self.adapter_factory = adapter_factory or build_adapter
        self.external_call_semaphore = external_call_semaphore
        self.external_call_budget = external_call_budget
        self.query_understanding = QueryUnderstandingService(
            settings,
            rewriter_factory=query_rewriter_factory,
            external_call_semaphore=external_call_semaphore,
            external_call_budget=external_call_budget,
        )
        self.retrieval = RetrievalService(
            settings.index_path,
            vector_index_path=settings.vector_index_path,
            embedding_cache_dir=settings.embedding_cache_dir,
            reranker_cache_dir=settings.reranker_cache_dir,
        )

    async def answer(
        self, query: str, top_k: int, strategy: RetrievalStrategy = "bm25"
    ) -> AnswerOutcome:
        total_started = time.perf_counter()
        understanding = await self.query_understanding.understand(query)
        search = await asyncio.to_thread(
            self.retrieval.search,
            query,
            top_k,
            strategy,
            retrieval_query=understanding.retrieval_query,
        )
        evidence_started = time.perf_counter()
        pack = build_evidence_pack(
            query=search.query,
            normalized_query=search.normalized_query,
            results=search.results,
        )
        evidence_ms = elapsed_ms(evidence_started)

        if not search.results:
            return self._retrieval_only(
                search.query,
                search.normalized_query,
                search.results,
                reason=None,
                retrieval_ms=search.latency_ms,
                evidence_ms=evidence_ms,
                total_started=total_started,
                retrieval_strategy=search.retrieval_strategy,
                query_understanding=understanding,
            )
        if not self.settings.llm_configured:
            return self._retrieval_only(
                search.query,
                search.normalized_query,
                search.results,
                reason="not_configured",
                retrieval_ms=search.latency_ms,
                evidence_ms=evidence_ms,
                total_started=total_started,
                retrieval_strategy=search.retrieval_strategy,
                query_understanding=understanding,
            )

        adapter = self.adapter_factory(self.settings)
        generation_started = time.perf_counter()
        try:
            if (
                self.external_call_budget is not None
                and not await self.external_call_budget.try_acquire()
            ):
                raise LLMQuotaExceededError()
            if self.external_call_semaphore is None:
                generated = await adapter.generate(pack)
            else:
                async with self.external_call_semaphore:
                    generated = await adapter.generate(pack)
            citations = validate_generated_answer(generated.answer, pack)
        except LLMError as error:
            LOGGER.warning(
                "LLM fallback kind=%s provider_request_id=%s error_type=%s",
                error.kind,
                error.provider_request_id or "unavailable",
                type(error).__name__,
            )
            return self._retrieval_only(
                search.query,
                search.normalized_query,
                search.results,
                reason=error.kind,
                retrieval_ms=search.latency_ms,
                evidence_ms=evidence_ms,
                generation_ms=elapsed_ms(generation_started),
                total_started=total_started,
                retrieval_strategy=search.retrieval_strategy,
                query_understanding=understanding,
            )
        finally:
            close = getattr(adapter, "aclose", None)
            if close is not None:
                await close()

        generation_ms = elapsed_ms(generation_started)
        return AnswerOutcome(
            query=search.query,
            normalized_query=search.normalized_query,
            answer=render_rag_answer(search.query, generated.answer),
            citations=citations,
            limitations=list(FIXED_LIMITATIONS),
            results=search.results,
            mode="rag",
            model=generated.model,
            fallback_reason=None,
            timing=Timing(
                rewrite_ms=understanding.rewrite_latency_ms,
                retrieval_ms=search.latency_ms,
                evidence_ms=evidence_ms,
                generation_ms=generation_ms,
                total_ms=elapsed_ms(total_started),
            ),
            retrieval_strategy=search.retrieval_strategy,
            query_understanding=understanding,
        )

    @staticmethod
    def _retrieval_only(
        query: str,
        normalized_query: str,
        results: list[SearchResult],
        *,
        reason: str | None,
        retrieval_ms: float,
        evidence_ms: float,
        total_started: float,
        query_understanding: QueryUnderstanding,
        generation_ms: float = 0.0,
        retrieval_strategy: RetrievalStrategy = "bm25",
    ) -> AnswerOutcome:
        content = render_retrieval_only(results, reason=reason or "not_configured")
        return AnswerOutcome(
            query=query,
            normalized_query=normalized_query,
            answer=content.answer,
            citations=[],
            limitations=content.limitations,
            results=results,
            mode="retrieval_only",
            model=None,
            fallback_reason=reason,
            timing=Timing(
                rewrite_ms=query_understanding.rewrite_latency_ms,
                retrieval_ms=retrieval_ms,
                evidence_ms=evidence_ms,
                generation_ms=generation_ms,
                total_ms=elapsed_ms(total_started),
            ),
            retrieval_strategy=retrieval_strategy,
            query_understanding=query_understanding,
        )


def build_adapter(settings: Settings) -> LLMAdapter:
    if settings.llm_api_style == "anthropic":
        return AnthropicHTTPAdapter(settings)
    if settings.llm_api_style == "openai":
        return OpenAICompatibleHTTPAdapter(settings)
    raise LLMInvalidOutputError("未知的 LLM API 风格")


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
