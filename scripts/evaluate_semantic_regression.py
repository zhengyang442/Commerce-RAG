from __future__ import annotations

import json
from pathlib import Path

from app.core.config import DEFAULT_ARTIFACTS_DIR, Settings
from app.retrieval.service import RetrievalService

MANIFEST_PATH = Path("data/evaluation/semantic_regression_v1.json")


def has_expected_class(results, expected_terms: list[str], top_k: int) -> bool:
    expected = [term.casefold() for term in expected_terms]
    return any(
        any(term in (result.product_class or "").casefold() for term in expected)
        for result in results[:top_k]
    )


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    settings = Settings.from_env()
    service = RetrievalService(
        settings.index_path,
        vector_index_path=settings.vector_index_path,
        embedding_cache_dir=settings.embedding_cache_dir,
    )
    reports = {}
    for strategy in ("bm25", "vector", "hybrid", "rerank"):
        cases = []
        for case in manifest["cases"]:
            response = service.search(case["query"], 3, strategy)
            cases.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "top1_hit": has_expected_class(
                        response.results, case["expected_product_class_terms"], 1
                    ),
                    "top3_hit": has_expected_class(
                        response.results, case["expected_product_class_terms"], 3
                    ),
                    "top3": [
                        {
                            "product_id": item.product_id,
                            "product_class": item.product_class,
                        }
                        for item in response.results
                    ],
                    "latency_ms": response.latency_ms,
                }
            )
        reports[strategy] = {
            "top1_accuracy": sum(item["top1_hit"] for item in cases) / len(cases),
            "top3_accuracy": sum(item["top3_hit"] for item in cases) / len(cases),
            "cases": cases,
        }
    payload = {
        "evaluation_version": manifest["version"],
        "case_count": len(manifest["cases"]),
        "results": reports,
        "comparison": {
            "vector_top1_beats_bm25": (
                reports["vector"]["top1_accuracy"] > reports["bm25"]["top1_accuracy"]
            ),
            "vector_top3_beats_bm25": (
                reports["vector"]["top3_accuracy"] > reports["bm25"]["top3_accuracy"]
            ),
            "hybrid_top1_beats_bm25": (
                reports["hybrid"]["top1_accuracy"] > reports["bm25"]["top1_accuracy"]
            ),
            "hybrid_top3_beats_bm25": (
                reports["hybrid"]["top3_accuracy"] > reports["bm25"]["top3_accuracy"]
            ),
            "rerank_top1_beats_hybrid": (
                reports["rerank"]["top1_accuracy"] > reports["hybrid"]["top1_accuracy"]
            ),
            "rerank_top3_beats_hybrid": (
                reports["rerank"]["top3_accuracy"] > reports["hybrid"]["top3_accuracy"]
            ),
        },
    }
    output_dir = DEFAULT_ARTIFACTS_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "semantic_regression_v1.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**payload, "results": "omitted", "report": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
