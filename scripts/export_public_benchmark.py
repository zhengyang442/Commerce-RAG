from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "artifacts" / "evaluation"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "releases" / "v0.4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出不含逐查询数据的公开 benchmark 快照")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_report(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def retrieval_summary(report: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    metrics = report["metrics"]["all"]
    retrieval = report["retrieval"]
    return {
        "strategy": retrieval["strategy"],
        "evaluation_version": report["evaluation_version"],
        "query_count": metrics["query_count"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "recall_at_10": metrics["recall_at_10"],
        "mrr_at_10": metrics["mrr_at_10"],
        "judged_at_10": metrics["judged_at_10"],
        "relevant_hit_at_10": metrics["relevant_hit_at_10"],
        "exact_hit_at_10": metrics["exact_hit_at_10"],
        "no_judged_top_10_query_count": metrics["no_judged_top_10_query_count"],
        "latency_ms": metrics["latency_ms"],
        "engine": retrieval["engine"],
        "source_report_sha256": source_sha256,
    }


def chinese_summary(report: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    metrics = report["metrics"]
    return {
        "mode": report["mode"],
        "evaluation_version": report["evaluation_version"],
        "case_count": report["case_count"],
        "query_rewrite_valid_rate": metrics["query_rewrite_valid_rate"],
        "required_term_coverage": metrics["required_term_coverage"],
        "unsupported_intent_accuracy": metrics["unsupported_intent_accuracy"],
        "exclusion_detection_rate": metrics["exclusion_detection_rate"],
        "category_accuracy_at_1": metrics["category_accuracy_at_1"],
        "category_hit_at_3": metrics["category_hit_at_3"],
        "fallback_rate": metrics["fallback_rate"],
        "rewrite_latency_ms": metrics["rewrite_latency_ms"],
        "source_report_sha256": source_sha256,
    }


def build_payload(source_root: Path) -> dict[str, Any]:
    reports: dict[str, tuple[dict[str, Any], str]] = {}
    for name in ("bm25", "vector", "hybrid", "rerank"):
        reports[name] = read_report(source_root / f"{name}_v1.json")
    for name in ("chinese_regression_v1_rules", "chinese_regression_v1_llm"):
        reports[name] = read_report(source_root / f"{name}.json")

    data_commits = {
        report["data_commit"] for report, _ in reports.values() if "data_commit" in report
    }
    if len(data_commits) != 1:
        raise ValueError("检索报告的数据版本不一致")

    return {
        "schema_version": "1",
        "release": "v0.4",
        "data": {
            "name": "Wayfair ANnotation Dataset (WANDS)",
            "commit": data_commits.pop(),
            "products": reports["bm25"][0]["raw_counts"]["products"],
            "queries": reports["bm25"][0]["raw_counts"]["queries"],
            "labels": reports["bm25"][0]["raw_counts"]["labels"],
            "split": {"dev": 384, "test": 96, "all": 480, "seed": 42},
        },
        "retrieval": [
            retrieval_summary(report, sha256)
            for report, sha256 in (reports[name] for name in ("bm25", "vector", "hybrid", "rerank"))
        ],
        "chinese_query_understanding": [
            chinese_summary(*reports["chinese_regression_v1_rules"]),
            chinese_summary(*reports["chinese_regression_v1_llm"]),
        ],
        "decisions": {
            "default_retrieval": "hybrid",
            "reranker": "experimental_latency_over_budget",
            "query_rewrite": "rules_first_llm_on_untranslated_cjk",
        },
        "notes": [
            (
                "Unjudged products receive zero gain only for benchmark calculation; "
                "they are not manually judged Irrelevant."
            ),
            "Latency is a local macOS ARM64 snapshot and must not be treated as a deployment SLA.",
            (
                "The LLM rewrite experiment used real provider calls; "
                "default tests do not call external models."
            ),
        ],
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
