from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from app.core.config import DEFAULT_INDEX_PATH
from app.core.errors import IndexNotReadyError, ProductNotFoundError
from app.core.models import Product

SCHEMA_VERSION = "1"
WANDS_COMMIT = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
EXPECTED_PRODUCT_COUNT = 42_994


class SQLiteStore:
    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH) -> None:
        self.index_path = index_path

    def connect(self, *, read_only: bool = True) -> sqlite3.Connection:
        if read_only:
            if not self.index_path.is_file():
                raise IndexNotReadyError("本地商品索引尚未构建")
            uri = f"file:{self.index_path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(self.index_path)
        connection.row_factory = sqlite3.Row
        return connection

    def status(self) -> dict[str, object]:
        if not self.index_path.is_file():
            return {
                "index_ready": False,
                "product_count": 0,
                "schema_version": None,
                "data_version": WANDS_COMMIT,
            }
        try:
            with self.connect() as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
                product_count = connection.execute("SELECT count(*) FROM products").fetchone()[0]
            ready = (
                metadata.get("schema_version") == SCHEMA_VERSION
                and metadata.get("wands_commit") == WANDS_COMMIT
                and int(product_count) == EXPECTED_PRODUCT_COUNT
            )
        except (sqlite3.Error, ValueError):
            return {
                "index_ready": False,
                "product_count": 0,
                "schema_version": None,
                "data_version": WANDS_COMMIT,
            }
        return {
            "index_ready": ready,
            "product_count": int(product_count),
            "schema_version": metadata.get("schema_version"),
            "data_version": metadata.get("wands_commit", WANDS_COMMIT),
        }

    def get_product(self, product_id: int) -> Product:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM products WHERE product_id = ?", (product_id,)
            ).fetchone()
        if row is None:
            raise ProductNotFoundError(f"商品 {product_id} 不存在")
        return product_from_row(row)

    def iter_products(self) -> Iterator[Product]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM products ORDER BY product_id")
            for row in rows:
                yield product_from_row(row)


def product_from_row(row: sqlite3.Row) -> Product:
    return Product(**{field: row[field] for field in Product.model_fields})
