from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import calculate_query_metrics, discounted_cumulative_gain, percentile


def test_metrics_match_hand_calculation() -> None:
    relevance = {1: 2, 2: 1, 3: 0}

    metrics = calculate_query_metrics([2, 1, 9], relevance)

    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert metrics.ndcg_at_10 == pytest.approx(dcg / idcg)
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr_at_10 == 0.5
    assert metrics.judged_at_10 == 0.2
    assert metrics.relevant_hit_at_10 is True
    assert metrics.exact_hit_at_10 is True
    assert metrics.no_judged_top_10 is False


def test_zero_relevant_and_no_exact_are_scored_zero() -> None:
    metrics = calculate_query_metrics([1, 2], {1: 0, 2: 0})

    assert metrics.ndcg_at_10 == 0.0
    assert metrics.recall_at_10 == 0.0
    assert metrics.mrr_at_10 == 0.0
    assert metrics.no_relevant is True
    assert metrics.no_exact is True


def test_unjudged_results_are_distinct_from_labeled_irrelevant() -> None:
    metrics = calculate_query_metrics([99, 3], {3: 0})

    assert metrics.judged_at_10 == 0.1
    assert metrics.relevant_hit_at_10 is False
    assert metrics.exact_hit_at_10 is False
    assert metrics.no_judged_top_10 is False

    all_unjudged = calculate_query_metrics([98, 99], {3: 0})
    assert all_unjudged.judged_at_10 == 0.0
    assert all_unjudged.no_judged_top_10 is True


def test_discount_and_percentile_are_deterministic() -> None:
    assert discounted_cumulative_gain([2, 1]) == pytest.approx(2 + 1 / math.log2(3))
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    assert percentile([], 0.95) == 0.0
