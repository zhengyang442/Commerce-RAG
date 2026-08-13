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
    assert page.headers["cache-control"] == "no-cache"
    assert 'lang="zh-CN"' in page.text
    assert "家具目录检索" in page.text
    assert "业务查询样例" in page.text
    assert "蓝色天鹅绒休闲椅，带金色椅腿" in page.text
    assert "实木大号平台床，不需要弹簧床架" in page.text
    assert "四人小型圆形餐桌，查询价格与折扣" in page.text
    assert "需求解析" in page.text
    assert "结果摘要" in page.text
    assert "候选商品" in page.text
    assert "把家具需求说清楚" not in page.text
    assert "让每个结果都有证据" not in page.text
    assert "没有证据就不承诺" not in page.text
    assert "搜索并回答" in page.text
    assert "只检索" in page.text
    assert 'value="5"' in page.text
    assert "开发者选项" in page.text
    assert "不保存完整查询" in page.text
    assert "BM25（v0.1）" in page.text
    assert "Vector（v0.2" in page.text
    assert "Hybrid RRF（v0.3" in page.text
    assert "Hybrid + Reranker（实验" in page.text
    assert 'option value="vector" disabled' in page.text
    assert "/static/styles.css?v=20260813-business-1" in page.text
    assert "/static/app.js?v=20260813-business-1" in page.text
    assert styles.status_code == 200
    assert "@media (max-width: 700px)" in styles.text
    assert "prefers-reduced-motion: reduce" in styles.text
    assert ".product-card { min-width: 0" in styles.text
    assert "overflow-wrap: anywhere" in styles.text
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
    assert "rawJson.textContent" in script
    assert "developer-mode" in script
    assert "dataset.query" in script
