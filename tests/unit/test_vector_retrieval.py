from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app.ingestion.sqlite_builder import SCHEMA_SQL
from app.ingestion.vector_builder import VectorIndexBuilder
from app.retrieval.embedding import EMBEDDING_DIMENSION, _validated_vector, product_as_document
from app.retrieval.service import RetrievalService
from app.retrieval.vector_store import VectorStore


class FakeEmbedder:
    model_name = "BAAI/bge-small-en-v1.5"
    dimension = EMBEDDING_DIMENSION

    def embed_passages(self, texts):
        for text in texts:
            yield self._vector(text)

    def embed_query(self, query):
        return self._vector(query)

    @staticmethod
    def _vector(text):
        vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        vector[0 if "blue" in text.lower() else 1] = 1.0
        return vector


def make_catalog(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "Blue Chair", "Chair", "Furniture", "Blue seat", "velvet", 1, 4.0, 1),
                (2, "Oak Table", "Table", "Furniture", "Wood table", "oak", 1, 4.0, 1),
            ],
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("wands_commit", "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"),
                ("product_count", "2"),
            ],
        )
        connection.execute("INSERT INTO products_fts(products_fts) VALUES ('rebuild')")
        connection.commit()


def test_product_document_keeps_labeled_fields() -> None:
    from app.core.models import Product

    text = product_as_document(
        Product(product_id=1, product_name="Blue Chair", product_features="Color: Blue")
    )

    assert text == "name: Blue Chair\nfeatures: Color: Blue"


def test_embedding_must_be_finite_normalized_and_expected_dimension() -> None:
    valid = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    valid[0] = 1.0
    assert _validated_vector(valid, EMBEDDING_DIMENSION).shape == (EMBEDDING_DIMENSION,)
    with pytest.raises(ValueError, match="维度"):
        _validated_vector(np.ones(2), EMBEDDING_DIMENSION)
    with pytest.raises(ValueError, match="非有限"):
        _validated_vector(np.full(EMBEDDING_DIMENSION, np.nan), EMBEDDING_DIMENSION)
    with pytest.raises(ValueError, match="未归一化"):
        _validated_vector(np.ones(EMBEDDING_DIMENSION), EMBEDDING_DIMENSION)


def test_vector_index_build_status_and_search(tmp_path: Path, monkeypatch) -> None:
    from app.retrieval import vector_store

    monkeypatch.setattr(vector_store, "EXPECTED_PRODUCT_COUNT", 2)
    catalog = tmp_path / "catalog.sqlite"
    vector_index = tmp_path / "vectors.sqlite"
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "model.onnx").write_bytes(b"fake-model")
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("a" * 40, encoding="utf-8")
    make_catalog(catalog)

    report = VectorIndexBuilder(
        catalog_path=catalog,
        vector_index_path=vector_index,
        embedding_cache_dir=cache,
        embedder=FakeEmbedder(),
        expected_product_count=2,
        batch_size=1,
    ).build()
    status = VectorStore(vector_index).status()
    result = RetrievalService(
        catalog,
        vector_index_path=vector_index,
        embedding_cache_dir=cache,
        embedder=FakeEmbedder(),
    ).search("blue furniture", 1, "vector")

    assert report.vector_count == 2
    assert report.failed_count == 0
    assert report.model_revision == "a" * 40
    assert report.peak_memory_mb > 0
    assert len(report.mapping_sha256) == 64
    assert status["vector_index_ready"] is True
    assert status["vector_count"] == 2
    assert result.retrieval_strategy == "vector"
    assert result.results[0].product_id == 1
    assert result.results[0].citation_id == "P1"

    second_report = VectorIndexBuilder(
        catalog_path=catalog,
        vector_index_path=tmp_path / "vectors-second.sqlite",
        embedding_cache_dir=cache,
        embedder=FakeEmbedder(),
        expected_product_count=2,
        batch_size=2,
    ).build()
    assert second_report.mapping_sha256 == report.mapping_sha256
    assert second_report.model_cache_sha256 == report.model_cache_sha256


def test_vector_builder_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="正整数"):
        VectorIndexBuilder(
            catalog_path=tmp_path / "catalog.sqlite",
            vector_index_path=tmp_path / "vectors.sqlite",
            embedding_cache_dir=tmp_path / "models",
            embedder=FakeEmbedder(),
            expected_product_count=0,
            batch_size=0,
        )
