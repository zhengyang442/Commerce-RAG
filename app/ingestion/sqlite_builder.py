from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import DEFAULT_INDEX_PATH, DEFAULT_RAW_DATA_DIR
from app.core.errors import DataValidationError
from app.ingestion.wands_validator import WandsValidator
from app.retrieval.query_normalization import tokenize
from app.retrieval.sqlite_store import EXPECTED_PRODUCT_COUNT, SCHEMA_VERSION, WANDS_COMMIT

SCHEMA_SQL = """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    product_class TEXT,
    category_hierarchy TEXT,
    product_description TEXT,
    product_features TEXT NOT NULL,
    rating_count INTEGER,
    average_rating REAL,
    review_count INTEGER
) STRICT;

CREATE VIRTUAL TABLE products_fts USING fts5(
    product_name,
    product_class,
    category_hierarchy,
    product_description,
    product_features,
    content='products',
    content_rowid='product_id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;
"""


@dataclass(frozen=True, slots=True)
class BuildReport:
    product_count: int
    fts_count: int
    schema_version: str
    wands_commit: str
    missing_product_fields: dict[str, int]
    duration_ms: float
    index_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "product_count": self.product_count,
            "fts_count": self.fts_count,
            "schema_version": self.schema_version,
            "wands_commit": self.wands_commit,
            "missing_product_fields": self.missing_product_fields,
            "duration_ms": self.duration_ms,
            "index_path": self.index_path,
        }


class SQLiteIndexBuilder:
    def __init__(
        self,
        data_dir: Path = DEFAULT_RAW_DATA_DIR,
        index_path: Path = DEFAULT_INDEX_PATH,
        *,
        validator: WandsValidator | None = None,
        expected_product_count: int = EXPECTED_PRODUCT_COUNT,
    ) -> None:
        self.data_dir = data_dir
        self.index_path = index_path
        self.validator = validator or WandsValidator(data_dir)
        self.expected_product_count = expected_product_count

    def build(self) -> BuildReport:
        started = time.perf_counter()
        validation = self.validator.validate()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.index_path.name}.", suffix=".tmp", dir=self.index_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            product_count, fts_count = self._build_temporary(temporary_path)
            self._assert_no_sidecars(temporary_path)
            os.replace(temporary_path, self.index_path)
        except Exception:
            self._cleanup_temporary_files(temporary_path)
            raise

        return BuildReport(
            product_count=product_count,
            fts_count=fts_count,
            schema_version=SCHEMA_VERSION,
            wands_commit=WANDS_COMMIT,
            missing_product_fields=validation.missing_product_fields,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            index_path=str(self.index_path),
        )

    def _build_temporary(self, temporary_path: Path) -> tuple[int, int]:
        connection = sqlite3.connect(temporary_path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(SCHEMA_SQL)
            connection.execute("BEGIN")
            connection.executemany(
                """
                INSERT INTO products (
                    product_id, product_name, product_class, category_hierarchy,
                    product_description, product_features, rating_count,
                    average_rating, review_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        product.product_id,
                        product.product_name,
                        product.product_class,
                        product.category_hierarchy,
                        product.product_description,
                        product.product_features,
                        product.rating_count,
                        product.average_rating,
                        product.review_count,
                    )
                    for product in self.validator.iter_products()
                ),
            )
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                (
                    ("schema_version", SCHEMA_VERSION),
                    ("wands_commit", WANDS_COMMIT),
                    ("product_count", str(self.expected_product_count)),
                ),
            )
            connection.execute("INSERT INTO products_fts(products_fts) VALUES ('rebuild')")
            connection.commit()

            product_count = int(connection.execute("SELECT count(*) FROM products").fetchone()[0])
            fts_count = int(connection.execute("SELECT count(*) FROM products_fts").fetchone()[0])
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise DataValidationError("SQLite integrity_check 未通过")
            connection.execute(
                "INSERT INTO products_fts(products_fts, rank) VALUES ('integrity-check', 1)"
            )
            if (
                product_count != self.expected_product_count
                or fts_count != self.expected_product_count
            ):
                raise DataValidationError("SQLite 商品或 FTS 记录数量不正确")
            sentinel_product = connection.execute(
                "SELECT product_id, product_name FROM products ORDER BY product_id LIMIT 1"
            ).fetchone()
            sentinel_tokens = tokenize(sentinel_product[1]) if sentinel_product else ()
            sentinel = None
            if sentinel_tokens:
                sentinel = connection.execute(
                    """
                    SELECT rowid FROM products_fts
                    WHERE products_fts MATCH ? AND rowid = ?
                    LIMIT 1
                    """,
                    (f'"{sentinel_tokens[0]}"', sentinel_product[0]),
                ).fetchone()
            if sentinel is None:
                raise DataValidationError("SQLite FTS5 哨兵查询没有结果")
            return product_count, fts_count
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _sidecar_paths(database_path: Path) -> tuple[Path, Path]:
        return (Path(f"{database_path}-wal"), Path(f"{database_path}-shm"))

    def _assert_no_sidecars(self, database_path: Path) -> None:
        remaining = [path for path in self._sidecar_paths(database_path) if path.exists()]
        if remaining:
            names = ", ".join(path.name for path in remaining)
            raise DataValidationError(f"SQLite 临时库关闭后仍存在 WAL/SHM：{names}")

    def _cleanup_temporary_files(self, database_path: Path) -> None:
        database_path.unlink(missing_ok=True)
        for path in (*self._sidecar_paths(database_path), Path(f"{database_path}-journal")):
            path.unlink(missing_ok=True)
