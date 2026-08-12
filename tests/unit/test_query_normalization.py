from __future__ import annotations

import pytest

from app.core.errors import EmptyQueryError, NoSearchTokensError
from app.retrieval.query_normalization import normalize_query, tokenize


def test_normalization_collapses_unicode_whitespace_without_translation() -> None:
    normalized = normalize_query("  Solid\twood chair  ")

    assert normalized.normalized_query == "Solid wood chair"
    assert normalized.tokens == ("solid", "wood", "chair")
    assert normalized.match_expression == '"solid" OR "wood" OR "chair"'


def test_tokenizer_removes_diacritics_and_deduplicates_match_tokens() -> None:
    normalized = normalize_query("Café café")

    assert tokenize("Café café") == ("cafe", "cafe")
    assert normalized.tokens == ("cafe",)


def test_empty_and_punctuation_queries_are_rejected() -> None:
    with pytest.raises(EmptyQueryError):
        normalize_query(" \t  ")
    with pytest.raises(NoSearchTokensError):
        normalize_query('"() * !!!')
