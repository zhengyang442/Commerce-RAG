from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import DEFAULT_EMBEDDING_CACHE_DIR, DEFAULT_RERANKER_CACHE_DIR
from app.core.models import Product

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
DOCUMENT_TEXT_VERSION = "product_text_v1"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"
RERANKER_MODEL_SHA256 = "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870"


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_passages(self, texts: Iterable[str]) -> Iterator[np.ndarray]: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, documents: Iterable[str]) -> list[float]: ...


class FastEmbedder:
    model_name = EMBEDDING_MODEL
    dimension = EMBEDDING_DIMENSION

    def __init__(
        self,
        *,
        cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        threads: int | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(cache_dir),
            threads=threads,
        )

    def embed_passages(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        for vector in self._model.passage_embed(texts):
            yield _validated_vector(vector, self.dimension)

    def embed_query(self, query: str) -> np.ndarray:
        return _validated_vector(next(self._model.query_embed(query)), self.dimension)


@lru_cache(maxsize=2)
def cached_fast_embedder(cache_dir: str) -> FastEmbedder:
    return FastEmbedder(cache_dir=Path(cache_dir))


class FastEmbedReranker:
    model_name = RERANKER_MODEL

    def __init__(
        self,
        *,
        cache_dir: Path = DEFAULT_RERANKER_CACHE_DIR,
        threads: int | None = None,
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextCrossEncoder(
            model_name=self.model_name,
            cache_dir=str(cache_dir),
            threads=threads,
        )

    def rerank(self, query: str, documents: Iterable[str]) -> list[float]:
        scores = [float(score) for score in self._model.rerank(query, documents)]
        if not all(np.isfinite(score) for score in scores):
            raise ValueError("Reranker 返回非有限分数")
        return scores


@lru_cache(maxsize=2)
def cached_fast_reranker(cache_dir: str) -> FastEmbedReranker:
    return FastEmbedReranker(cache_dir=Path(cache_dir))


def reranker_cache_ready(cache_dir: Path) -> bool:
    snapshot = (
        cache_dir / "models--Xenova--ms-marco-MiniLM-L-6-v2" / "snapshots" / RERANKER_MODEL_REVISION
    )
    required = (
        snapshot / "config.json",
        snapshot / "tokenizer.json",
        snapshot / "tokenizer_config.json",
        snapshot / "special_tokens_map.json",
        snapshot / "onnx" / "model.onnx",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    digest = hashlib.sha256()
    with (snapshot / "onnx" / "model.onnx").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == RERANKER_MODEL_SHA256


def product_as_document(product: Product) -> str:
    values = (
        ("name", product.product_name),
        ("class", product.product_class),
        ("category", product.category_hierarchy),
        ("description", product.product_description),
        ("features", product.product_features),
    )
    return "\n".join(f"{label}: {value}" for label, value in values if value)


def _validated_vector(value: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError(f"Embedding 维度必须是 {dimension}")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding 包含非有限数值")
    norm = float(np.linalg.norm(vector))
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ValueError(f"Embedding 未归一化：norm={norm:.6f}")
    return vector
