from __future__ import annotations

from app.evaluation.runner import render_markdown


def test_markdown_report_contains_frozen_metric_contract() -> None:
    payload = {
        "data_commit": "commit",
        "evaluation_version": "bm25_v1",
        "metrics": {
            name: {
                "query_count": count,
                "ndcg_at_10": 0.1,
                "recall_at_10": 0.2,
                "mrr_at_10": 0.3,
                "judged_at_10": 0.4,
                "relevant_hit_at_10": 0.5,
                "exact_hit_at_10": 0.25,
                "no_judged_top_10_query_count": 2,
                "latency_ms": {"p50": 1.0, "p95": 2.0},
            }
            for name, count in (("dev", 384), ("test", 96), ("all", 480))
        },
        "qrels": {
            "raw_count": 233448,
            "unique_count": 231873,
            "duplicate_pair_count": 1467,
            "conflict_pair_count": 14,
        },
    }

    markdown = render_markdown(payload)

    assert "nDCG@10" in markdown
    assert "Judged@10" in markdown
    assert "RelevantHit@10" in markdown
    assert "ExactHit@10" in markdown
    assert "不代表人工判定为 Irrelevant" in markdown
    assert "Exact=2" in markdown
    assert "384 dev / 96 test" in markdown
    assert "冻结 test 集不得用于 Phase 1 调参" in markdown

    payload["evaluation_version"] = "vector_v1"
    vector_markdown = render_markdown(payload)
    assert "CommerceRAG vector_v1 基线" in vector_markdown
    assert "冻结 test 集不得用于向量参数调优" in vector_markdown
