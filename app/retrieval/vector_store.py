from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from app.core.config import DEFAULT_VECTOR_INDEX_PATH
from app.core.errors import IndexNotReadyError
from app.ingestion.vector_builder import (
    VECTOR_INDEX_VERSION,
    VECTOR_SCHEMA_VERSION,
    model_cache_revision,
)
from app.retrieval.embedding import DOCUMENT_TEXT_VERSION, EMBEDDING_DIMENSION, EMBEDDING_MODEL
from app.retrieval.sqlite_store import EXPECTED_PRODUCT_COUNT, WANDS_COMMIT


class VectorStore:
    def __init__(
        self,
        path: Path = DEFAULT_VECTOR_INDEX_PATH,
        *,
        embedding_cache_dir: Path | None = None,
    ) -> None:
        self.path = path
        self.embedding_cache_dir = embedding_cache_dir

    def connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise IndexNotReadyError("本地向量索引尚未构建")
        connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        try:
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
        except Exception:
            connection.close()
            raise
        connection.row_factory = sqlite3.Row
        return connection

    def status(self) -> dict[str, object]:
        if not self.path.is_file():
            return {"vector_index_ready": False, "vector_count": 0}
        try:
            with self.connect() as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
                count = int(
                    connection.execute("SELECT count(*) FROM product_vectors").fetchone()[0]
                )
            ready = (
                metadata.get("schema_version") == VECTOR_SCHEMA_VERSION
                and metadata.get("index_version") == VECTOR_INDEX_VERSION
                and metadata.get("data_commit") == WANDS_COMMIT
                and metadata.get("model_name") == EMBEDDING_MODEL
                and len(metadata.get("model_revision", "")) == 40
                and int(metadata.get("dimension", 0)) == EMBEDDING_DIMENSION
                and metadata.get("normalized") == "true"
                and metadata.get("distance_metric") == "cosine"
                and metadata.get("document_text_version") == DOCUMENT_TEXT_VERSION
                and len(metadata.get("mapping_sha256", "")) == 64
                and len(metadata.get("model_cache_sha256", "")) == 64
                and int(metadata.get("product_count", 0)) == count
                and count == EXPECTED_PRODUCT_COUNT
            )
            if ready and self.embedding_cache_dir is not None:
                ready = metadata.get("model_revision") == model_cache_revision(
                    self.embedding_cache_dir
                )
        except (sqlite3.Error, ValueError, OSError):
            return {"vector_index_ready": False, "vector_count": 0}
        return {
            "vector_index_ready": ready,
            "vector_count": count,
            "vector_index_version": metadata.get("index_version"),
            "embedding_model": metadata.get("model_name"),
            "embedding_model_revision": metadata.get("model_revision"),
        }
