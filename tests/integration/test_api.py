from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingestion.sqlite_builder import SCHEMA_SQL
from app.main import create_app
from app.query_understanding.models import RewriteOutput
from app.retrieval import sqlite_store


def make_api_index(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("schema_version", "1"), ("wands_commit", "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5")],
        )
        connection.execute("INSERT INTO products_fts(products_fts) VALUES ('rebuild')")
        connection.commit()


def client_for(tmp_path: Path, monkeypatch) -> TestClient:
    index_path = tmp_path / "catalog.sqlite"
    make_api_index(index_path)
    monkeypatch.setattr(sqlite_store, "EXPECTED_PRODUCT_COUNT", 1)
    settings = Settings(
        index_path=index_path,
        vector_index_path=tmp_path / "missing-vectors.sqlite",
    )
    return TestClient(create_app(settings))


def test_health_search_and_product_contract(tmp_path: Path, monkeypatch) -> None:
    with client_for(tmp_path, monkeypatch) as client:
        health = client.get("/api/health")
        search = client.post(
            "/api/search",
            json={"query": "blue chair", "top_k": 1, "retrieval_strategy": "bm25"},
        )
        product = client.get("/api/products/0")

    assert health.status_code == 200
    assert health.json()["index_ready"] is True
    assert health.json()["product_count"] == 1
    assert health.json()["llm_configured"] is False
    assert health.json()["vector_index_ready"] is False
    assert health.json()["vector_count"] == 0
    assert search.status_code == 200
    assert search.json()["request_id"].startswith("req_")
    assert search.json()["results"][0]["category_hierarchy"] == "Furniture > Chairs"
    assert search.json()["results"][0]["citation_id"] == "P1"
    assert search.json()["retrieval_strategy"] == "bm25"
    assert product.status_code == 200
    assert product.json()["product_id"] == 0


def test_api_validation_and_missing_product_errors_include_request_id(
    tmp_path: Path, monkeypatch
) -> None:
    with client_for(tmp_path, monkeypatch) as client:
        empty = client.post("/api/search", json={"query": " ", "retrieval_strategy": "bm25"})
        out_of_range = client.post(
            "/api/search",
            json={"query": "chair", "top_k": 21, "retrieval_strategy": "bm25"},
        )
        missing = client.get("/api/products/999")

    for response in (empty, out_of_range, missing):
        assert response.status_code in {404, 422}
        assert response.json()["request_id"].startswith("req_")
        assert "RAG_LLM" not in response.text


def test_api_reports_index_not_ready(tmp_path: Path) -> None:
    settings = Settings(index_path=tmp_path / "missing.sqlite")
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        search = client.post("/api/search", json={"query": "chair", "retrieval_strategy": "bm25"})

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert search.status_code == 503
    assert search.json()["error"]["code"] == "index_not_ready"


class ChineseRewriter:
    async def rewrite(self, query: str) -> RewriteOutput:
        return RewriteOutput(
            retrieval_query="blue accent chair",
            category_terms=["accent chair"],
            attributes={"color": ["blue"]},
            excluded_terms=[],
        )

    async def aclose(self) -> None:
        pass


def test_chinese_search_uses_rewrite_and_returns_diagnostics(tmp_path: Path, monkeypatch) -> None:
    index_path = tmp_path / "catalog.sqlite"
    make_api_index(index_path)
    monkeypatch.setattr(sqlite_store, "EXPECTED_PRODUCT_COUNT", 1)
    settings = Settings(
        index_path=index_path,
        vector_index_path=tmp_path / "missing-vectors.sqlite",
        llm_api_style="openai",
        llm_base_url="https://example.test",
        llm_api_key="test",
        llm_model="test-model",
    )
    app = create_app(settings, query_rewriter_factory=lambda _: ChineseRewriter())

    with TestClient(app) as client:
        response = client.post(
            "/api/search",
            json={"query": "儿童人体工学座椅", "top_k": 1, "retrieval_strategy": "bm25"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["query"] == "儿童人体工学座椅"
    assert payload["normalized_query"] == "blue accent chair"
    assert payload["query_understanding"]["detected_language"] == "zh"
    assert payload["query_understanding"]["rewrite_source"] == "llm"
    assert payload["results"][0]["product_name"] == "Blue Chair"
