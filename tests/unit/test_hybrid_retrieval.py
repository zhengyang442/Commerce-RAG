from __future__ import annotations

from pathlib import Path

from app.retrieval.embedding import RERANKER_MODEL_SHA256, reranker_cache_ready
from app.retrieval.service import RRF_K, RetrievalService
from tests.unit.test_vector_retrieval import make_catalog


class FakeReranker:
    model_name = "fake-reranker"

    def rerank(self, query, documents):
        return [float("blue" in text.casefold()) for text in documents]


def test_rrf_fuses_sources_with_stable_metadata(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "catalog.sqlite"
    make_catalog(catalog)
    service = RetrievalService(catalog)
    blue = service.search("blue", 1, "bm25").results[0]
    wood = service.search("wood", 1, "bm25").results[0]
    bm25 = [blue.model_copy(update={"rank": 1}), wood.model_copy(update={"rank": 2})]
    vector = [
        blue.model_copy(
            update={
                "rank": 1,
                "retrieval_sources": ["vector"],
                "source_ranks": {"vector": 1},
            }
        )
    ]
    monkeypatch.setattr(service, "search_connection", lambda connection, query, top_k: bm25)
    monkeypatch.setattr(service, "search_vector", lambda query, top_k: vector)

    hybrid = service.search("blue", 2, "hybrid").results

    assert hybrid[0].product_id == 1
    assert hybrid[0].retrieval_sources == ["bm25", "vector"]
    assert hybrid[0].source_ranks == {"bm25": 1, "vector": 1}
    assert hybrid[0].fusion_score == 2 / (RRF_K + 1)
    assert [item.citation_id for item in hybrid] == ["P1", "P2"]
    assert bm25[0].retrieval_sources == ["bm25"]


def test_reranker_reorders_only_hybrid_candidates(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "catalog.sqlite"
    make_catalog(catalog)
    service = RetrievalService(catalog, reranker=FakeReranker())
    candidates = [
        service.search("wood", 1, "bm25").results[0],
        service.search("blue", 1, "bm25").results[0],
    ]
    monkeypatch.setattr(
        service,
        "search_hybrid",
        lambda normalized, top_k, candidate_k=50: candidates,
    )

    results = service.search("blue", 2, "rerank").results

    assert results[0].product_id == 1
    assert results[0].reranker_score == 1.0
    assert results[0].citation_id == "P1"


def test_reranker_can_request_internal_top_50(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "catalog.sqlite"
    make_catalog(catalog)
    service = RetrievalService(catalog, reranker=FakeReranker())
    candidates = service.search("wood", 1, "bm25").results
    observed = {}

    def fake_hybrid(normalized, top_k, candidate_k=50):
        observed.update(top_k=top_k, candidate_k=candidate_k)
        return candidates

    monkeypatch.setattr(service, "search_hybrid", fake_hybrid)

    service.search("wood", 1, "rerank")

    assert observed == {"top_k": 50, "candidate_k": 50}


def test_reranker_cache_requires_expected_model_sha(tmp_path: Path, monkeypatch) -> None:
    from app.retrieval import embedding

    snapshot = (
        tmp_path
        / "models--Xenova--ms-marco-MiniLM-L-6-v2"
        / "snapshots"
        / embedding.RERANKER_MODEL_REVISION
    )
    (snapshot / "onnx").mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        (snapshot / name).write_text("{}", encoding="utf-8")
    model = snapshot / "onnx" / "model.onnx"
    model.write_bytes(b"known-model")
    import hashlib

    monkeypatch.setattr(
        embedding, "RERANKER_MODEL_SHA256", hashlib.sha256(b"known-model").hexdigest()
    )
    assert reranker_cache_ready(tmp_path) is True
    monkeypatch.setattr(embedding, "RERANKER_MODEL_SHA256", RERANKER_MODEL_SHA256)
    assert reranker_cache_ready(tmp_path) is False
