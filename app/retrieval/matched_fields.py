from __future__ import annotations

from collections.abc import Mapping

from app.retrieval.query_normalization import tokenize

SEARCH_FIELDS = (
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
)


def find_matched_fields(product: Mapping[str, object], query_tokens: tuple[str, ...]) -> list[str]:
    query_token_set = set(query_tokens)
    return [
        field
        for field in SEARCH_FIELDS
        if query_token_set.intersection(tokenize(str(product.get(field) or "")))
    ]
