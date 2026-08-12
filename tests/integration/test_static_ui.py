from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_static_ui_and_assets_are_served(tmp_path) -> None:
    with TestClient(create_app(Settings(index_path=tmp_path / "missing.sqlite"))) as client:
        page = client.get("/")
        styles = client.get("/static/styles.css")
        script = client.get("/static/app.js")

    assert page.status_code == 200
    assert 'lang="zh-CN"' in page.text
    assert "v0.4 支持中文查询" in page.text
    assert "支持中文或英文" in page.text
    assert "系统如何理解你的需求" in page.text
    assert "搜索并回答" in page.text
    assert "只检索" in page.text
    assert "BM25（v0.1）" in page.text
    assert "Vector（v0.2" in page.text
    assert "Hybrid RRF（v0.3" in page.text
    assert "Hybrid + Reranker（v0.3" in page.text
    assert 'option value="vector" disabled' in page.text
    assert styles.status_code == 200
    assert "@media (max-width: 760px)" in styles.text
    assert script.status_code == 200


def test_ui_uses_text_content_for_untrusted_product_and_model_text(tmp_path) -> None:
    with TestClient(create_app(Settings(index_path=tmp_path / "missing.sqlite"))) as client:
        script = client.get("/static/app.js").text

    assert ".innerHTML" not in script
    assert "textContent" in script
    assert "loading" in script.lower()
    assert "requestSubmit" in script
    assert "scrollIntoView" in script
    assert "fallback_reason" in script
    assert "retrieval_strategy" in script
    assert "vector_index_ready" in script
    assert "query_understanding" in script
