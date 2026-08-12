from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from app.core.config import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_RERANKER_CACHE_DIR,
    DEFAULT_VECTOR_INDEX_PATH,
)
from app.core.errors import IndexNotReadyError, InvalidTopKError
from app.core.models import RetrievalStrategy, SearchResult
from app.retrieval.embedding import (
    Embedder,
    Reranker,
    cached_fast_embedder,
    cached_fast_reranker,
    product_as_document,
    reranker_cache_ready,
)
from app.retrieval.matched_fields import find_matched_fields
from app.retrieval.query_normalization import NormalizedQuery, normalize_query
from app.retrieval.sqlite_store import SQLiteStore
from app.retrieval.vector_store import VectorStore

DEFAULT_TOP_K = 10
MIN_TOP_K = 1
MAX_TOP_K = 20
HYBRID_CANDIDATE_K = 50
RRF_K = 60


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    normalized_query: str
    results: list[SearchResult]
    latency_ms: float
    retrieval_strategy: RetrievalStrategy


class RetrievalService:
    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        *,
        vector_index_path: Path = DEFAULT_VECTOR_INDEX_PATH,
        embedding_cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        reranker_cache_dir: Path = DEFAULT_RERANKER_CACHE_DIR,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = SQLiteStore(index_path)
        self.vector_store = VectorStore(vector_index_path, embedding_cache_dir=embedding_cache_dir)
        self.embedding_cache_dir = embedding_cache_dir
        self.reranker_cache_dir = reranker_cache_dir
        self._embedder = embedder
        self._reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: RetrievalStrategy = "bm25",
        *,
        retrieval_query: str | None = None,
    ) -> SearchResponse:
        normalized = normalize_query(retrieval_query or query)
        validate_top_k(top_k)
        started = time.perf_counter()
        if strategy == "bm25":
            with self.store.connect() as connection:
                results = self.search_connection(connection, normalized, top_k)
        elif strategy == "vector":
            results = self.search_vector(normalized.normalized_query, top_k)
        elif strategy == "hybrid":
            results = self.search_hybrid(normalized, top_k)
        elif strategy == "rerank":
            results = self.search_rerank(normalized, top_k)
        else:
            raise ValueError(f"未知检索策略：{strategy}")
        return SearchResponse(
            query=query,
            normalized_query=normalized.normalized_query,
            results=results,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            retrieval_strategy=strategy,
        )

    def search_vector(self, normalized_query: str, top_k: int) -> list[SearchResult]:
        validate_candidate_k(top_k)
        if not self.vector_store.status()["vector_index_ready"]:
            raise IndexNotReadyError("本地向量索引未就绪或版本不兼容")
        try:
            with self.vector_store.connect() as vector_connection:
                embedder = self._embedder
                if embedder is None:
                    embedder = cached_fast_embedder(str(self.embedding_cache_dir.resolve()))
                    self._embedder = embedder
                query_vector = sqlite_vec.serialize_float32(embedder.embed_query(normalized_query))
                vector_rows = vector_connection.execute(
                    """
                    SELECT rowid AS product_id, distance
                    FROM product_vectors
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance ASC
                    """,
                    (query_vector, top_k),
                ).fetchall()
                vector_rows = sorted(
                    vector_rows,
                    key=lambda row: (float(row["distance"]), int(row["product_id"])),
                )
            product_ids = [int(row["product_id"]) for row in vector_rows]
            if not product_ids:
                return []
            placeholders = ",".join("?" for _ in product_ids)
            with self.store.connect() as catalog_connection:
                rows = catalog_connection.execute(
                    f"SELECT * FROM products WHERE product_id IN ({placeholders})", product_ids
                ).fetchall()
        except sqlite3.Error as error:
            raise IndexNotReadyError("SQLite 向量检索失败") from error
        products = {int(row["product_id"]): dict(row) for row in rows}
        results = []
        for rank, vector_row in enumerate(vector_rows, start=1):
            product_id = int(vector_row["product_id"])
            values = products[product_id]
            values["score"] = 1.0 - float(vector_row["distance"])
            values["rank"] = rank
            values["citation_id"] = f"P{rank}"
            values["matched_fields"] = []
            values["retrieval_sources"] = ["vector"]
            values["source_ranks"] = {"vector": rank}
            results.append(SearchResult.model_validate(values))
        return results

    def search_hybrid(
        self,
        normalized: NormalizedQuery,
        top_k: int,
        *,
        candidate_k: int = HYBRID_CANDIDATE_K,
    ) -> list[SearchResult]:
        validate_candidate_k(top_k)
        validate_candidate_k(candidate_k)
        with self.store.connect() as connection:
            bm25_results = self.search_connection(connection, normalized, candidate_k)
        vector_results = self.search_vector(normalized.normalized_query, candidate_k)
        by_product: dict[int, SearchResult] = {}
        source_ranks: dict[int, dict[str, int]] = {}
        fusion_scores: dict[int, float] = {}
        matched_fields: dict[int, list[str]] = {}
        for source, results in (("bm25", bm25_results), ("vector", vector_results)):
            for result in results:
                product_id = result.product_id
                by_product.setdefault(product_id, result)
                source_ranks.setdefault(product_id, {})[source] = result.rank
                fusion_scores[product_id] = fusion_scores.get(product_id, 0.0) + (
                    1.0 / (RRF_K + result.rank)
                )
                for field in result.matched_fields:
                    if field not in matched_fields.setdefault(product_id, []):
                        matched_fields[product_id].append(field)
        ordered = sorted(
            by_product,
            key=lambda product_id: (
                -fusion_scores[product_id],
                min(source_ranks[product_id].values()),
                product_id,
            ),
        )[:top_k]
        return [
            by_product[product_id].model_copy(
                update={
                    "rank": rank,
                    "citation_id": f"P{rank}",
                    "score": fusion_scores[product_id],
                    "matched_fields": matched_fields.get(product_id, []),
                    "retrieval_sources": list(source_ranks[product_id]),
                    "source_ranks": source_ranks[product_id],
                    "fusion_score": fusion_scores[product_id],
                    "reranker_score": None,
                }
            )
            for rank, product_id in enumerate(ordered, start=1)
        ]

    def search_rerank(self, normalized: NormalizedQuery, top_k: int) -> list[SearchResult]:
        validate_top_k(top_k)
        candidates = self.search_hybrid(
            normalized,
            HYBRID_CANDIDATE_K,
            candidate_k=HYBRID_CANDIDATE_K,
        )
        if not candidates:
            return []
        reranker = self._reranker
        if reranker is None:
            if not reranker_cache_ready(self.reranker_cache_dir):
                raise IndexNotReadyError("本地 Reranker 模型尚未下载或文件不完整")
            reranker = cached_fast_reranker(str(self.reranker_cache_dir.resolve()))
            self._reranker = reranker
        scores = reranker.rerank(
            normalized.normalized_query,
            [product_as_document(candidate) for candidate in candidates],
        )
        if len(scores) != len(candidates):
            raise IndexNotReadyError("Reranker 返回的分数数量与候选数量不一致")
        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank, item[0].product_id),
        )[:top_k]
        return [
            candidate.model_copy(
                update={
                    "rank": rank,
                    "citation_id": f"P{rank}",
                    "score": score,
                    "reranker_score": score,
                }
            )
            for rank, (candidate, score) in enumerate(scored, start=1)
        ]

    def search_connection(
        self,
        connection: sqlite3.Connection,
        normalized: NormalizedQuery,
        top_k: int,
    ) -> list[SearchResult]:
        validate_candidate_k(top_k)
        try:
            rows = connection.execute(
                """
                SELECT
                    p.product_id,
                    p.product_name,
                    p.product_class,
                    p.category_hierarchy,
                    p.product_description,
                    p.product_features,
                    p.rating_count,
                    p.average_rating,
                    p.review_count,
                    -bm25(products_fts) AS score
                FROM products_fts
                JOIN products AS p ON p.product_id = products_fts.rowid
                WHERE products_fts MATCH ?
                ORDER BY bm25(products_fts) ASC, p.product_id ASC
                LIMIT ?
                """,
                (normalized.match_expression, top_k),
            ).fetchall()
        except sqlite3.Error as error:
            raise IndexNotReadyError("SQLite FTS5 检索失败") from error

        results: list[SearchResult] = []
        for rank, row in enumerate(rows, start=1):
            values = dict(row)
            values["score"] = float(values["score"])
            values["rank"] = rank
            values["citation_id"] = f"P{rank}"
            values["matched_fields"] = find_matched_fields(values, normalized.tokens)
            values["retrieval_sources"] = ["bm25"]
            values["source_ranks"] = {"bm25": rank}
            results.append(SearchResult.model_validate(values))
        return results


def validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise InvalidTopKError("top_k 必须是整数")
    if not MIN_TOP_K <= top_k <= MAX_TOP_K:
        raise InvalidTopKError(f"top_k 必须在 {MIN_TOP_K} 到 {MAX_TOP_K} 之间")


def validate_candidate_k(candidate_k: int) -> None:
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or not 1 <= candidate_k <= 100
    ):
        raise InvalidTopKError("候选池大小必须在 1 到 100 之间")
