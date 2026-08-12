from __future__ import annotations

from collections import Counter

from app.evaluation.answer_review import FIXTURE_PATH, load_cases


def test_answer_review_fixture_has_30_unique_cases_and_six_balanced_groups() -> None:
    cases = load_cases(FIXTURE_PATH)
    counts = Counter(case["scenario"] for case in cases)

    assert len(cases) == 30
    assert counts == {
        "explicit_match": 5,
        "multi_condition": 5,
        "partial_match": 5,
        "no_result": 5,
        "missing_information": 5,
        "generation_failure": 5,
    }
    assert all(case["required_checks"] for case in cases)
    assert all(case["forbidden_claims"] for case in cases)
