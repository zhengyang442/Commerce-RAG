from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.ingestion.wands_models import LABEL_GRADES
from app.ingestion.wands_validator import WandsValidator


@dataclass(frozen=True, slots=True)
class Qrels:
    by_query: dict[int, dict[int, int]]
    raw_count: int
    unique_count: int
    duplicate_pair_count: int
    duplicate_annotation_count: int
    conflict_pair_count: int
    canonical_label_counts: dict[str, int]


def load_qrels(validator: WandsValidator) -> Qrels:
    grades_by_pair: dict[tuple[int, int], set[int]] = {}
    occurrences: Counter[tuple[int, int]] = Counter()
    raw_count = 0
    for label in validator.iter_labels():
        pair = (label.query_id, label.product_id)
        grades_by_pair.setdefault(pair, set()).add(LABEL_GRADES[label.label])
        occurrences[pair] += 1
        raw_count += 1

    by_query: dict[int, dict[int, int]] = {}
    canonical_counts: Counter[str] = Counter()
    grade_labels = {grade: label for label, grade in LABEL_GRADES.items()}
    for (query_id, product_id), grades in grades_by_pair.items():
        grade = max(grades)
        by_query.setdefault(query_id, {})[product_id] = grade
        canonical_counts[grade_labels[grade]] += 1

    return Qrels(
        by_query=by_query,
        raw_count=raw_count,
        unique_count=len(grades_by_pair),
        duplicate_pair_count=sum(count > 1 for count in occurrences.values()),
        duplicate_annotation_count=raw_count - len(grades_by_pair),
        conflict_pair_count=sum(len(grades) > 1 for grades in grades_by_pair.values()),
        canonical_label_counts={
            label: canonical_counts[label] for label in ("Exact", "Partial", "Irrelevant")
        },
    )
