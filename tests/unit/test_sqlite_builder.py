from __future__ import annotations

import sqlite3
from pathlib import Path

from app.ingestion.sqlite_builder import SCHEMA_SQL


def test_schema_is_strict_and_uses_expected_fts_tokenizer(tmp_path: Path) -> None:
    database = tmp_path / "schema.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(SCHEMA_SQL)
        products_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'products'"
        ).fetchone()[0]
        fts_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'products_fts'"
        ).fetchone()[0]

    assert "STRICT" in products_sql
    assert "unicode61 remove_diacritics 2" in fts_sql
    assert "content_rowid='product_id'" in fts_sql
