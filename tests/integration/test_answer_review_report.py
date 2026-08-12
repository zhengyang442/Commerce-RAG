from __future__ import annotations

from app.evaluation.answer_review import render_answer_review_markdown


def test_answer_review_report_states_live_provider_boundary() -> None:
    payload = {
        "case_count": 30,
        "passed_count": 30,
        "failed_count": 0,
        "cases": [
            {
                "case_id": "case-01",
                "scenario": "explicit_match",
                "passed": True,
                "checks": {"citation": True},
            }
        ],
    }

    markdown = render_answer_review_markdown(payload)

    assert "真实供应商：未使用" in markdown
    assert "不代表真实模型自然语言质量评测" in markdown
