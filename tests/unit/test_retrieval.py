from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.core.errors import (
    EmptyQueryError,
    IndexNotReadyError,
    InvalidTopKError,
    NoSearchTokensError,
)
from app.ingestion.sqlite_builder import SCHEMA_SQL
from app.retrieval.service import RetrievalService, validate_top_k


def make_index(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    0,
                    "Blue Chair",
                    "Accent Chairs",
                    "Furniture > Chairs",
                    "A seat",
                    "Color: Blue",
                    1,
                    4.0,
                    1,
                ),
                (
                    2,
                    "Wood Table",
                    "Dining Tables",
                    "Furniture > Tables",
                    None,
                    "Material: Wood",
                    None,
                    None,
                    None,
                ),
                (10, "Red Sofa", "Sofas", "Furniture > Seating", "A sofa", "Color: Red", 2, 3.0, 2),
            ],
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("schema_version", "1"), ("wands_commit", "test"), ("product_count", "3")],
        )
        connection.execute("INSERT INTO products_fts(products_fts) VALUES ('rebuild')")
        connection.commit()


def test_retrieval_returns_stable_results_and_matched_fields(tmp_path: Path) -> None:
    index_path = tmp_path / "catalog.sqlite"
    make_index(index_path)
    service = RetrievalService(index_path)

    first = service.search('"blue" OR NOT sofa', top_k=3)
    second = service.search("blue sofa", top_k=3)

    assert [result.product_id for result in first.results] == [10, 0]
    assert [result.product_id for result in second.results] == [10, 0]
    assert first.results[0].citation_id == "P1"
    assert first.results[0].matched_fields == ["product_name", "product_description"]
    assert first.results[0].score >= first.results[1].score


def test_product_id_zero_is_searchable(tmp_path: Path) -> None:
    index_path = tmp_path / "catalog.sqlite"
    make_index(index_path)

    results = RetrievalService(index_path).search("blue", top_k=1).results

    assert results[0].product_id == 0


def test_search_boundaries_are_rejected(tmp_path: Path) -> None:
    index_path = tmp_path / "catalog.sqlite"
    make_index(index_path)
    service = RetrievalService(index_path)

    with pytest.raises(EmptyQueryError):
        service.search(" ")
    with pytest.raises(NoSearchTokensError):
        service.search("!!!")
    with pytest.raises(InvalidTopKError):
        service.search("chair", 0)
    with pytest.raises(InvalidTopKError):
        service.search("chair", 21)
    with pytest.raises(InvalidTopKError):
        validate_top_k(True)


def test_missing_index_is_retrieval_error(tmp_path: Path) -> None:
    with pytest.raises(IndexNotReadyError):
        RetrievalService(tmp_path / "missing.sqlite").search("chair")

    with pytest.raises(IndexNotReadyError, match="向量索引"):
        RetrievalService(
            tmp_path / "missing.sqlite",
            vector_index_path=tmp_path / "missing-vectors.sqlite",
        ).search("chair", strategy="vector")
