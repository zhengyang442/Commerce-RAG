from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path

from app.core.config import DEFAULT_ARTIFACTS_DIR, Settings
from app.generation.orchestrator import AnswerOrchestrator

DEFAULT_QUERIES = (
    "blue velvet accent chair gold legs",
    "solid wood queen platform bed no box spring",
    "small round dining table for four",
    "modern black metal bar stools set of two",
    "white dresser with six drawers",
    "gray sectional sofa with chaise",
    "natural wood coffee table with storage",
    "upholstered dining chair beige",
    "twin size daybed with trundle",
    "wooden nightstand with two drawers",
    "navy blue recliner chair",
    "farmhouse TV stand for 65 inch television",
    "king bed frame with headboard",
    "round glass side table gold frame",
    "leather office chair with arms",
    "three shelf bookcase walnut",
    "green velvet loveseat",
    "extendable dining table seats six",
    "shoe storage bench with cushion",
    "queen bed current price discount",
)

FORBIDDEN_ANSWER_PATTERNS = (
    re.compile(r"(?:\$|¥|￥)\s*\d"),
    re.compile(r"\b(?:in stock|out of stock|ships? in \d|\d+% off)\b", re.IGNORECASE),
    re.compile(r"(?:现价|售价|库存充足|无库存|折扣为|配送需要)"),
)


async def evaluate(settings: Settings) -> dict[str, object]:
    cases = []
    for query in DEFAULT_QUERIES:
        result = await AnswerOrchestrator(settings).answer(query, 5)
        valid_citations = all(
            citation.citation_id in {item.citation_id for item in result.results}
            for citation in result.citations
        )
        valid_fields = all(bool(citation.supporting_fields) for citation in result.citations)
        forbidden_claim = any(
            pattern.search(result.answer) for pattern in FORBIDDEN_ANSWER_PATTERNS
        )
        cases.append(
            {
                "query": query,
                "mode": result.mode,
                "model": result.model,
                "fallback_reason": result.fallback_reason,
                "result_count": len(result.results),
                "citation_count": len(result.citations),
                "valid_citations": valid_citations,
                "valid_supporting_fields": valid_fields,
                "forbidden_claim": forbidden_claim,
                "generation_ms": result.timing.generation_ms,
            }
        )
    latencies = sorted(float(case["generation_ms"]) for case in cases)
    p95_index = round(0.95 * (len(latencies) - 1))
    rag_count = sum(case["mode"] == "rag" for case in cases)
    return {
        "evaluation_version": "deepseek_real_v1",
        "model": settings.llm_model,
        "case_count": len(cases),
        "rag_count": rag_count,
        "rag_success_rate": rag_count / len(cases),
        "citation_validation_rate": sum(case["valid_citations"] for case in cases) / len(cases),
        "supporting_fields_validation_rate": sum(case["valid_supporting_fields"] for case in cases)
        / len(cases),
        "forbidden_claim_count": sum(case["forbidden_claim"] for case in cases),
        "generation_latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "p95": latencies[p95_index],
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 20 条 DeepSeek 真实 RAG 验收")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "llm_evaluation")
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.llm_configured:
        parser.exit(1, "LLM 尚未配置\n")
    payload = asyncio.run(evaluate(settings))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "deepseek_real_v1.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**payload, "cases": "omitted", "report": str(output_path)}, indent=2))
    passed = payload["rag_count"] >= 19 and payload["forbidden_claim_count"] == 0
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
