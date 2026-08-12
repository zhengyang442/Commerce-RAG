from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import sqlite_vec

from app.core.config import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_VECTOR_INDEX_PATH,
)
from app.core.errors import DataValidationError
from app.retrieval.embedding import (
    DOCUMENT_TEXT_VERSION,
    Embedder,
    FastEmbedder,
    product_as_document,
)
from app.retrieval.sqlite_store import EXPECTED_PRODUCT_COUNT, WANDS_COMMIT, SQLiteStore

VECTOR_SCHEMA_VERSION = "1"
VECTOR_INDEX_VERSION = "vector_v1"


@dataclass(frozen=True, slots=True)
class VectorBuildReport:
    vector_count: int
    failed_count: int
    dimension: int
    model_name: str
    model_revision: str
    document_text_version: str
    data_commit: str
    mapping_sha256: str
    model_cache_sha256: str
    index_sha256: str
    index_bytes: int
    peak_memory_mb: float
    duration_ms: float
    index_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VectorIndexBuilder:
    def __init__(
        self,
        *,
        catalog_path: Path = DEFAULT_INDEX_PATH,
        vector_index_path: Path = DEFAULT_VECTOR_INDEX_PATH,
        embedding_cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        embedder: Embedder | None = None,
        expected_product_count: int = EXPECTED_PRODUCT_COUNT,
        batch_size: int = 64,
    ) -> None:
        self.catalog_path = catalog_path
        self.vector_index_path = vector_index_path
        self.embedding_cache_dir = embedding_cache_dir
        self.embedder = embedder
        self.expected_product_count = expected_product_count
        if batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        self.batch_size = batch_size

    def build(self) -> VectorBuildReport:
        started = time.perf_counter()
        self._validate_catalog()
        if self.embedder is None:
            self.embedder = FastEmbedder(cache_dir=self.embedding_cache_dir)
        if model_cache_revision(self.embedding_cache_dir) == "unavailable":
            raise DataValidationError("Embedding 模型 revision 无法确定")
        if model_files_sha256(self.embedding_cache_dir) == "unavailable":
            raise DataValidationError("Embedding 模型文件校验值无法确定")
        self.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.vector_index_path.name}.",
            suffix=".tmp",
            dir=self.vector_index_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            vector_count, mapping_sha256 = self._build_temporary(temporary_path)
            self._assert_no_sidecars(temporary_path)
            os.replace(temporary_path, self.vector_index_path)
        except Exception:
            self._cleanup_temporary_files(temporary_path)
            raise
        index_sha256 = file_sha256(self.vector_index_path)
        embedder = self._require_embedder()
        return VectorBuildReport(
            vector_count=vector_count,
            failed_count=0,
            dimension=embedder.dimension,
            model_name=embedder.model_name,
            model_revision=model_cache_revision(self.embedding_cache_dir),
            document_text_version=DOCUMENT_TEXT_VERSION,
            data_commit=WANDS_COMMIT,
            mapping_sha256=mapping_sha256,
            model_cache_sha256=model_files_sha256(self.embedding_cache_dir),
            index_sha256=index_sha256,
            index_bytes=self.vector_index_path.stat().st_size,
            peak_memory_mb=peak_memory_mb(),
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            index_path=str(self.vector_index_path),
        )

    def _build_temporary(self, path: Path) -> tuple[int, str]:
        embedder = self._require_embedder()
        connection = sqlite3.connect(path)
        mapping_digest = hashlib.sha256()
        vector_count = 0
        try:
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                f"CREATE VIRTUAL TABLE product_vectors USING vec0("
                f"embedding float[{embedder.dimension}] distance_metric=cosine)"
            )
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT"
            )
            products = SQLiteStore(self.catalog_path).iter_products()
            while True:
                batch = []
                for _ in range(self.batch_size):
                    try:
                        batch.append(next(products))
                    except StopIteration:
                        break
                if not batch:
                    break
                vectors = embedder.embed_passages(product_as_document(item) for item in batch)
                rows = []
                for item, vector in zip(batch, vectors, strict=True):
                    rows.append((item.product_id, sqlite_vec.serialize_float32(vector)))
                    mapping_digest.update(f"{item.product_id}\n".encode())
                connection.executemany(
                    "INSERT INTO product_vectors(rowid, embedding) VALUES (?, ?)", rows
                )
                vector_count += len(rows)
                connection.commit()
            if vector_count != self.expected_product_count:
                raise DataValidationError(
                    f"向量数量不正确：expected={self.expected_product_count}, actual={vector_count}"
                )
            metadata = {
                "schema_version": VECTOR_SCHEMA_VERSION,
                "index_version": VECTOR_INDEX_VERSION,
                "data_commit": WANDS_COMMIT,
                "model_name": embedder.model_name,
                "model_revision": model_cache_revision(self.embedding_cache_dir),
                "dimension": str(embedder.dimension),
                "normalized": "true",
                "distance_metric": "cosine",
                "document_text_version": DOCUMENT_TEXT_VERSION,
                "mapping_sha256": mapping_digest.hexdigest(),
                "model_cache_sha256": model_files_sha256(self.embedding_cache_dir),
                "product_count": str(vector_count),
                "builder": json.dumps({"batch_size": self.batch_size}, sort_keys=True),
            }
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            stored_count = connection.execute("SELECT count(*) FROM product_vectors").fetchone()[0]
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if integrity != "ok" or stored_count != vector_count:
            raise DataValidationError("向量索引完整性检查失败")
        return vector_count, mapping_digest.hexdigest()

    def _validate_catalog(self) -> None:
        try:
            with SQLiteStore(self.catalog_path).connect() as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
                count = int(connection.execute("SELECT count(*) FROM products").fetchone()[0])
        except (sqlite3.Error, ValueError) as error:
            raise DataValidationError("必须先构建并校验商品 BM25 索引") from error
        if metadata.get("wands_commit") != WANDS_COMMIT or count != self.expected_product_count:
            raise DataValidationError("商品 BM25 索引版本或数量不正确")

    def _require_embedder(self) -> Embedder:
        if self.embedder is None:
            raise DataValidationError("Embedding 模型尚未初始化")
        return self.embedder

    @staticmethod
    def _sidecar_paths(database_path: Path) -> tuple[Path, Path]:
        return Path(f"{database_path}-wal"), Path(f"{database_path}-shm")

    def _assert_no_sidecars(self, database_path: Path) -> None:
        remaining = [path for path in self._sidecar_paths(database_path) if path.exists()]
        if remaining:
            names = ", ".join(path.name for path in remaining)
            raise DataValidationError(f"向量临时库关闭后仍存在 WAL/SHM：{names}")

    def _cleanup_temporary_files(self, database_path: Path) -> None:
        database_path.unlink(missing_ok=True)
        for path in (*self._sidecar_paths(database_path), Path(f"{database_path}-journal")):
            path.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("blobs/*") if item.is_file())
    if not files:
        files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        return "unavailable"
    for file_path in files:
        digest.update(file_path.name.encode())
        digest.update(b"\0")
        digest.update(file_sha256(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def model_cache_revision(path: Path) -> str:
    revisions = []
    for ref_path in sorted(path.rglob("refs/main")) if path.is_dir() else []:
        revision = ref_path.read_text(encoding="utf-8").strip()
        if revision:
            revisions.append(revision)
    return revisions[0] if len(revisions) == 1 else "unavailable"


def peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_used = value if platform.system() == "Darwin" else value * 1024
    return round(bytes_used / (1024 * 1024), 3)
