from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from app.core.errors import DataValidationError
from app.core.models import Product

LABEL_GRADES = {"Irrelevant": 0, "Partial": 1, "Exact": 2}
PRODUCT_HEADER = (
    "product_id",
    "product_name",
    "product_class",
    "category hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
)
QUERY_HEADER = ("query_id", "query", "query_class")
LABEL_HEADER = ("id", "query_id", "product_id", "label")


@dataclass(frozen=True, slots=True)
class QueryRecord:
    query_id: int
    query: str
    query_class: str | None


@dataclass(frozen=True, slots=True)
class LabelRecord:
    label_id: int
    query_id: int
    product_id: int
    label: str


def parse_product(row: Mapping[str, str], *, location: str) -> Product:
    product_name = row["product_name"].strip()
    product_features = row["product_features"].strip()
    if not product_name:
        raise DataValidationError(f"{location}: product_name 不能为空")
    if not product_features:
        raise DataValidationError(f"{location}: product_features 不能为空")

    return Product(
        product_id=parse_non_negative_int(row["product_id"], "product_id", location),
        product_name=product_name,
        product_class=optional_text(row["product_class"]),
        category_hierarchy=optional_text(row["category hierarchy"]),
        product_description=optional_text(row["product_description"]),
        product_features=product_features,
        rating_count=parse_optional_count(row["rating_count"], "rating_count", location),
        average_rating=parse_optional_rating(row["average_rating"], location),
        review_count=parse_optional_count(row["review_count"], "review_count", location),
    )


def parse_query(row: Mapping[str, str], *, location: str) -> QueryRecord:
    query = row["query"].strip()
    if not query:
        raise DataValidationError(f"{location}: query 不能为空")
    return QueryRecord(
        query_id=parse_non_negative_int(row["query_id"], "query_id", location),
        query=query,
        query_class=optional_text(row["query_class"]),
    )


def parse_label(row: Mapping[str, str], *, location: str) -> LabelRecord:
    label = row["label"].strip()
    if label not in LABEL_GRADES:
        raise DataValidationError(f"{location}: label 值不在允许集合中")
    return LabelRecord(
        label_id=parse_non_negative_int(row["id"], "id", location),
        query_id=parse_non_negative_int(row["query_id"], "query_id", location),
        product_id=parse_non_negative_int(row["product_id"], "product_id", location),
        label=label,
    )


def optional_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def parse_non_negative_int(value: str, field: str, location: str) -> int:
    stripped = value.strip()
    try:
        parsed = int(stripped)
    except ValueError as error:
        raise DataValidationError(f"{location}: {field} 必须是非负整数") from error
    if parsed < 0:
        raise DataValidationError(f"{location}: {field} 必须是非负整数")
    return parsed


def parse_optional_count(value: str, field: str, location: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError as error:
        raise DataValidationError(f"{location}: {field} 必须是非负整数值") from error
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        raise DataValidationError(f"{location}: {field} 必须是非负整数值")
    return int(parsed)


def parse_optional_rating(value: str, location: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError as error:
        raise DataValidationError(f"{location}: average_rating 必须是 1 到 5 的有限数值") from error
    if not math.isfinite(parsed) or not 1 <= parsed <= 5:
        raise DataValidationError(f"{location}: average_rating 必须是 1 到 5 的有限数值")
    return parsed
