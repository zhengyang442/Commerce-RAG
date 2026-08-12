from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from app.core.config import Settings
from app.core.usage_limits import DailyExternalCallBudget
from app.generation.llm.errors import LLMError, LLMQuotaExceededError
from app.query_understanding.models import QueryUnderstanding, RewriteOutput
from app.query_understanding.rewriter import ProviderQueryRewriter, QueryRewriter
from app.query_understanding.rules import analyze_with_rules, detect_unsupported_intents

RewriterFactory = Callable[[Settings], QueryRewriter]
LOGGER = logging.getLogger(__name__)


class QueryUnderstandingService:
    def __init__(
        self,
        settings: Settings,
        *,
        rewriter_factory: RewriterFactory | None = None,
        external_call_semaphore: asyncio.Semaphore | None = None,
        external_call_budget: DailyExternalCallBudget | None = None,
    ) -> None:
        self.settings = settings
        self.rewriter_factory = rewriter_factory or ProviderQueryRewriter
        self.external_call_semaphore = external_call_semaphore
        self.external_call_budget = external_call_budget

    async def understand(self, query: str) -> QueryUnderstanding:
        started = time.perf_counter()
        rules = analyze_with_rules(query)
        should_rewrite = (
            self.settings.query_rewrite_enabled
            and self.settings.llm_configured
            and rules.language in {"zh", "mixed"}
            and rules.has_untranslated_cjk
        )
        if not should_rewrite:
            original = " ".join(query.split()).casefold()
            source = "original" if rules.output.retrieval_query == original else "rules"
            return self._result(rules.output, rules, source, started)

        rewriter = self.rewriter_factory(self.settings)
        try:
            if (
                self.external_call_budget is not None
                and not await self.external_call_budget.try_acquire()
            ):
                raise LLMQuotaExceededError()
            if self.external_call_semaphore is None:
                output = await rewriter.rewrite(query)
            else:
                async with self.external_call_semaphore:
                    output = await rewriter.rewrite(query)
            output = sanitize_llm_output(output, rules.output)
            return self._result(output, rules, "llm", started)
        except LLMError as error:
            LOGGER.warning(
                "Query rewrite fallback kind=%s provider_request_id=%s error_type=%s",
                error.kind,
                error.provider_request_id or "unavailable",
                type(error).__name__,
            )
            return self._result(
                rules.output,
                rules,
                "rules_fallback",
                started,
                fallback_reason=error.kind,
            )
        finally:
            close = getattr(rewriter, "aclose", None)
            if close is not None:
                await close()

    @staticmethod
    def _result(output, rules, source, started, *, fallback_reason=None) -> QueryUnderstanding:
        return QueryUnderstanding(
            detected_language=rules.language,
            retrieval_query=output.retrieval_query,
            category_terms=output.category_terms,
            attributes=output.attributes,
            excluded_terms=list(
                dict.fromkeys([*rules.output.excluded_terms, *output.excluded_terms])
            ),
            unsupported_intents=rules.unsupported_intents,
            rewrite_source=source,
            rewrite_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            fallback_reason=fallback_reason,
        )


def sanitize_llm_output(output: RewriteOutput, rules_output: RewriteOutput) -> RewriteOutput:
    # Deterministic rules own exclusions and commercial-intent removal. The model can add facets,
    # but cannot reintroduce a topic the application knows the dataset does not contain.
    retrieval_query = output.retrieval_query
    for intent in detect_unsupported_intents(retrieval_query):
        from app.query_understanding.rules import UNSUPPORTED_PATTERNS, remove_patterns

        retrieval_query = remove_patterns(retrieval_query, [UNSUPPORTED_PATTERNS[intent]])
    retrieval_query = " ".join(retrieval_query.split())
    from app.query_understanding.rules import CJK_RE

    if not retrieval_query or CJK_RE.search(retrieval_query):
        retrieval_query = rules_output.retrieval_query
    return output.model_copy(
        update={
            "retrieval_query": retrieval_query,
            "excluded_terms": list(
                dict.fromkeys([*rules_output.excluded_terms, *output.excluded_terms])
            ),
        }
    )
