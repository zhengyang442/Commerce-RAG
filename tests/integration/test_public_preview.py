from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def preview_settings(tmp_path, **overrides) -> Settings:
    values = {
        "index_path": tmp_path / "missing.sqlite",
        "public_preview": True,
        "allowed_hosts": ("testserver",),
        "rate_limit_per_minute": 2,
        "answer_rate_limit_per_minute": 1,
        "max_request_bytes": 100,
    }
    values.update(overrides)
    return Settings(**values)


def test_public_preview_disables_api_docs_and_adds_security_headers(tmp_path) -> None:
    with TestClient(create_app(preview_settings(tmp_path))) as client:
        docs = client.get("/docs")
        page = client.get("/")

    assert docs.status_code == 404
    assert page.status_code == 200
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["referrer-policy"] == "no-referrer"
    assert page.headers["x-frame-options"] == "DENY"
    assert page.headers["cache-control"] == "no-store"


def test_public_preview_rejects_large_and_rate_limited_requests(tmp_path) -> None:
    with TestClient(create_app(preview_settings(tmp_path))) as client:
        large = client.post(
            "/api/search",
            content=b"x" * 101,
            headers={"content-type": "application/json", "content-length": "101"},
        )
    assert large.status_code == 413
    assert large.json()["error"]["code"] == "request_too_large"

    def streamed_body():
        yield b"x" * 60
        yield b"y" * 60

    with TestClient(create_app(preview_settings(tmp_path))) as client:
        streamed = client.post(
            "/api/search",
            content=streamed_body(),
            headers={"content-type": "application/json"},
        )
    assert streamed.status_code == 413

    settings = preview_settings(
        tmp_path,
        rate_limit_per_minute=1,
        answer_rate_limit_per_minute=1,
        max_request_bytes=1000,
    )
    with TestClient(create_app(settings)) as client:
        first_search = client.post("/api/search", json={"query": "chair"})
        first_answer = client.post("/api/answer", json={"query": "chair"})
        second_search = client.post("/api/search", json={"query": "chair"})
        second_answer = client.post("/api/answer", json={"query": "chair"})
    assert first_search.status_code == 503
    assert first_answer.status_code == 503
    assert second_search.status_code == 429
    assert second_search.json()["error"]["message"].startswith("搜索")
    assert second_answer.status_code == 429
    assert second_answer.json()["error"]["message"].startswith("回答")


def test_public_preview_rejects_unknown_host(tmp_path) -> None:
    with TestClient(create_app(preview_settings(tmp_path))) as client:
        response = client.get("/", headers={"host": "attacker.invalid"})

    assert response.status_code == 400
