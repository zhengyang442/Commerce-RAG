from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingestion.vector_builder import VectorIndexBuilder
from app.main import create_app
from app.retrieval.service import RetrievalService
from tests.unit.test_vector_retrieval import FakeEmbedder, make_catalog


def test_vector_search_api_returns_strategy_and_result(tmp_path, monkeypatch) -> None:
    from app.retrieval import vector_store

    monkeypatch.setattr(vector_store, "EXPECTED_PRODUCT_COUNT", 2)
    catalog = tmp_path / "catalog.sqlite"
    vectors = tmp_path / "vectors.sqlite"
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "model.onnx").write_bytes(b"fake-model")
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("a" * 40, encoding="utf-8")
    make_catalog(catalog)
    VectorIndexBuilder(
        catalog_path=catalog,
        vector_index_path=vectors,
        embedding_cache_dir=cache,
        embedder=FakeEmbedder(),
        expected_product_count=2,
    ).build()

    original_init = RetrievalService.__init__

    def init_with_fake(self, index_path, **kwargs):
        original_init(self, index_path, embedder=FakeEmbedder(), **kwargs)

    monkeypatch.setattr(RetrievalService, "__init__", init_with_fake)
    settings = Settings(
        index_path=catalog,
        vector_index_path=vectors,
        embedding_cache_dir=cache,
        reranker_cache_dir=tmp_path / "missing-reranker",
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        response = client.post(
            "/api/search",
            json={"query": "blue furniture", "top_k": 1, "retrieval_strategy": "vector"},
        )

    assert health.json()["vector_index_ready"] is True
    assert health.json()["vector_count"] == 2
    assert health.json()["reranker_ready"] is False
    assert response.status_code == 200
    assert response.json()["retrieval_strategy"] == "vector"
    assert response.json()["results"][0]["product_id"] == 1
