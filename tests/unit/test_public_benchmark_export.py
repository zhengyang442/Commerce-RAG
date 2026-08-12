from __future__ import annotations

import json
from pathlib import Path

from scripts.export_public_benchmark import build_payload


def write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_public_export_contains_summaries_without_query_cases(tmp_path: Path) -> None:
    retrieval = {
        "data_commit": "fixed-data",
        "evaluation_version": "placeholder",
        "raw_counts": {"products": 2, "queries": 480, "labels": 3},
        "retrieval": {"strategy": "placeholder", "engine": "test"},
        "metrics": {
            "all": {
                "query_count": 480,
                "ndcg_at_10": 0.7,
                "recall_at_10": 0.1,
                "mrr_at_10": 0.6,
                "judged_at_10": 0.8,
                "relevant_hit_at_10": 0.9,
                "exact_hit_at_10": 0.5,
                "no_judged_top_10_query_count": 1,
                "latency_ms": {"p50": 10, "p95": 20},
            }
        },
        "cases": [{"query": "must not leak"}],
    }
    for name in ("bm25", "vector", "hybrid", "rerank"):
        report = json.loads(json.dumps(retrieval))
        report["evaluation_version"] = f"{name}_v1"
        report["retrieval"]["strategy"] = name
        write_report(tmp_path / f"{name}_v1.json", report)

    chinese = {
        "evaluation_version": "chinese_regression_v1",
        "mode": "rules",
        "case_count": 30,
        "metrics": {
            "query_rewrite_valid_rate": 1.0,
            "required_term_coverage": 0.98,
            "unsupported_intent_accuracy": 1.0,
            "exclusion_detection_rate": 1.0,
            "category_accuracy_at_1": 0.86,
            "category_hit_at_3": 0.96,
            "fallback_rate": 0.0,
            "rewrite_latency_ms": {"p50": 0.1, "p95": 0.2},
        },
        "cases": [{"query": "must not leak"}],
    }
    write_report(tmp_path / "chinese_regression_v1_rules.json", chinese)
    chinese["mode"] = "llm"
    write_report(tmp_path / "chinese_regression_v1_llm.json", chinese)

    payload = build_payload(tmp_path)

    assert payload["data"]["commit"] == "fixed-data"
    assert [item["strategy"] for item in payload["retrieval"]] == [
        "bm25",
        "vector",
        "hybrid",
        "rerank",
    ]
    assert "must not leak" not in json.dumps(payload)
