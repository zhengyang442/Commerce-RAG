from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from app.core.errors import EmptyQueryError, NoSearchTokensError


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    original_query: str
    normalized_query: str
    tokens: tuple[str, ...]
    match_expression: str


def normalize_query(query: str) -> NormalizedQuery:
    if not isinstance(query, str):
        raise EmptyQueryError("查询必须是字符串")
    normalized = " ".join(query.split())
    if not normalized:
        raise EmptyQueryError("查询不能为空")
    tokens = tuple(dict.fromkeys(tokenize(normalized)))
    if not tokens:
        raise NoSearchTokensError("查询不包含可检索的字母或数字")
    expression = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
    return NormalizedQuery(
        original_query=query,
        normalized_query=normalized,
        tokens=tokens,
        match_expression=expression,
    )


def tokenize(text: str) -> tuple[str, ...]:
    """Match the FTS5 unicode61 word boundary closely without exposing MATCH syntax."""
    tokens: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKD", text).casefold():
        if unicodedata.combining(character):
            continue
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)
