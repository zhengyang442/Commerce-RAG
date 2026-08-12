from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from app.core.config import DEFAULT_ARTIFACTS_DIR, Settings
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.service import RetrievalService

MANIFEST_PATH = Path("data/evaluation/chinese_regression_v1.json")


def class_hit(results, expected: list[str], top_k: int) -> bool:
    terms = [item.casefold() for item in expected]
    return any(
        any(term in (result.product_class or "").casefold() for term in terms)
        for result in results[:top_k]
    )


async def evaluate(*, use_llm: bool) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    settings = Settings.from_env()
    if not use_llm:
        settings = Settings(
            index_path=settings.index_path,
            vector_index_path=settings.vector_index_path,
            embedding_cache_dir=settings.embedding_cache_dir,
            reranker_cache_dir=settings.reranker_cache_dir,
            query_rewrite_enabled=False,
        )
    understanding_service = QueryUnderstandingService(settings)
    retrieval = RetrievalService(
        settings.index_path,
        vector_index_path=settings.vector_index_path,
        embedding_cache_dir=settings.embedding_cache_dir,
        reranker_cache_dir=settings.reranker_cache_dir,
    )
    cases = []
    for case in manifest["cases"]:
        understanding = await understanding_service.understand(case["query"])
        response = await asyncio.to_thread(
            retrieval.search,
            case["query"],
            3,
            manifest["strategy"],
            retrieval_query=understanding.retrieval_query,
        )
        folded = understanding.retrieval_query.casefold()
        required = case.get("required_terms", [])
        expected_intents = case.get("unsupported_intents", [])
        expected_exclusions = case.get("excluded_terms", [])
        cases.append(
            {
                "id": case["id"],
                "query": case["query"],
                "retrieval_query": understanding.retrieval_query,
                "rewrite_source": understanding.rewrite_source,
                "fallback_reason": understanding.fallback_reason,
                "required_term_coverage": (
                    sum(term.casefold() in folded for term in required) / len(required)
                    if required
                    else 1.0
                ),
                "intent_exact": set(understanding.unsupported_intents) == set(expected_intents),
                "exclusion_hit": set(expected_exclusions).issubset(understanding.excluded_terms),
                "category_top1": class_hit(response.results, case["expected_classes"], 1),
                "category_top3": class_hit(response.results, case["expected_classes"], 3),
                "rewrite_latency_ms": understanding.rewrite_latency_ms,
                "retrieval_latency_ms": response.latency_ms,
                "top3": [
                    {"product_id": result.product_id, "product_class": result.product_class}
                    for result in response.results
                ],
            }
        )
    count = len(cases)
    rewrite_latencies = sorted(item["rewrite_latency_ms"] for item in cases)
    payload = {
        "evaluation_version": manifest["version"],
        "mode": "llm" if use_llm else "rules",
        "strategy": manifest["strategy"],
        "case_count": count,
        "metrics": {
            "query_rewrite_valid_rate": sum(bool(item["retrieval_query"]) for item in cases)
            / count,
            "required_term_coverage": statistics.fmean(
                item["required_term_coverage"] for item in cases
            ),
            "unsupported_intent_accuracy": sum(item["intent_exact"] for item in cases) / count,
            "exclusion_detection_rate": sum(item["exclusion_hit"] for item in cases) / count,
            "category_accuracy_at_1": sum(item["category_top1"] for item in cases) / count,
            "category_hit_at_3": sum(item["category_top3"] for item in cases) / count,
            "fallback_rate": sum(item["fallback_reason"] is not None for item in cases) / count,
            "rewrite_latency_ms": {
                "p50": statistics.median(rewrite_latencies),
                "p95": rewrite_latencies[max(0, int(0.95 * count) - 1)],
            },
        },
        "cases": cases,
    }
    output_dir = DEFAULT_ARTIFACTS_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "llm" if use_llm else "rules"
    path = output_dir / f"chinese_regression_v1_{suffix}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**payload, "cases": "omitted", "report": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="评测中文查询理解与 Hybrid 检索")
    parser.add_argument("--use-llm", action="store_true", help="调用已配置模型改写中文查询")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(use_llm=args.use_llm)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
