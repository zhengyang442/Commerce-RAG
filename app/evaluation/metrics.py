from __future__ import annotations

import math
from dataclasses import dataclass

EVALUATION_K = 10
GAIN_MAPPING = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    judged_at_10: float
    relevant_hit_at_10: bool
    exact_hit_at_10: bool
    no_judged_top_10: bool
    no_relevant: bool
    no_exact: bool


def calculate_query_metrics(
    ranked_product_ids: list[int],
    relevance: dict[int, int],
    *,
    k: int = EVALUATION_K,
) -> QueryMetrics:
    ranked = ranked_product_ids[:k]
    gains = [relevance.get(product_id, 0) for product_id in ranked]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    dcg = discounted_cumulative_gain(gains)
    idcg = discounted_cumulative_gain(ideal)
    ndcg = dcg / idcg if idcg else 0.0

    relevant = {product_id for product_id, grade in relevance.items() if grade >= 1}
    recall = len(relevant.intersection(ranked)) / len(relevant) if relevant else 0.0
    judged_count = sum(product_id in relevance for product_id in ranked)
    judged_at_k = judged_count / k
    relevant_hit = any(relevance.get(product_id, 0) >= 1 for product_id in ranked)
    exact_hit = any(relevance.get(product_id, 0) == 2 for product_id in ranked)

    reciprocal_rank = 0.0
    for rank, product_id in enumerate(ranked, start=1):
        if relevance.get(product_id) == 2:
            reciprocal_rank = 1.0 / rank
            break

    return QueryMetrics(
        ndcg_at_10=ndcg,
        recall_at_10=recall,
        mrr_at_10=reciprocal_rank,
        judged_at_10=judged_at_k,
        relevant_hit_at_10=relevant_hit,
        exact_hit_at_10=exact_hit,
        no_judged_top_10=judged_count == 0,
        no_relevant=not relevant,
        no_exact=not any(grade == 2 for grade in relevance.values()),
    )


def discounted_cumulative_gain(gains: list[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
