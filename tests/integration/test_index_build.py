from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.ingestion.sqlite_builder import SQLiteIndexBuilder
from app.ingestion.wands_validator import WandsValidator
from app.retrieval.sqlite_store import SQLiteStore


class TinyValidator(WandsValidator):
    def __init__(self, data_dir: Path) -> None:
        super().__init__(
            data_dir,
            expected_counts={"product.csv": 2, "query.csv": 2, "label.csv": 3},
            expected_label_counts=None,
            checksum_file=None,
        )


class TinyBuilder(SQLiteIndexBuilder):
    def __init__(self, data_dir: Path, index_path: Path, *, validator: WandsValidator) -> None:
        super().__init__(
            data_dir,
            index_path,
            validator=validator,
            expected_product_count=2,
        )


def test_atomic_builder_creates_searchable_database(tmp_path: Path) -> None:
    from tests.unit.test_wands_validator import write_tiny_wands

    data_dir = write_tiny_wands(tmp_path / "wands")
    index_path = tmp_path / "index" / "catalog.sqlite"
    builder = TinyBuilder(data_dir, index_path, validator=TinyValidator(data_dir))

    report = builder.build()

    assert report.product_count == 2
    with sqlite3.connect(index_path) as connection:
        assert connection.execute("SELECT count(*) FROM products").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM products_fts").fetchone()[0] == 2
        row = connection.execute(
            "SELECT rowid FROM products_fts WHERE products_fts MATCH ?", ("chair",)
        ).fetchone()
    assert row == (10,)


def test_failed_build_preserves_previous_index(tmp_path: Path) -> None:
    from tests.unit.test_wands_validator import write_tiny_wands

    data_dir = write_tiny_wands(tmp_path / "wands")
    index_path = tmp_path / "index" / "catalog.sqlite"
    index_path.parent.mkdir()
    index_path.write_bytes(b"known-good-index")

    class FailingBuilder(TinyBuilder):
        def _build_temporary(self, temporary_path: Path) -> tuple[int, int]:
            temporary_path.write_bytes(b"partial")
            raise sqlite3.OperationalError("injected failure")

    with pytest.raises(sqlite3.OperationalError, match="injected"):
        FailingBuilder(data_dir, index_path, validator=TinyValidator(data_dir)).build()

    assert index_path.read_bytes() == b"known-good-index"
    assert list(index_path.parent.glob(".catalog.sqlite.*.tmp")) == []


def test_sidecar_detection_preserves_previous_index(tmp_path: Path) -> None:
    from app.core.errors import DataValidationError
    from tests.unit.test_wands_validator import write_tiny_wands

    data_dir = write_tiny_wands(tmp_path / "wands")
    index_path = tmp_path / "index" / "catalog.sqlite"
    index_path.parent.mkdir()
    index_path.write_bytes(b"known-good-index")

    class SidecarBuilder(TinyBuilder):
        def _build_temporary(self, temporary_path: Path) -> tuple[int, int]:
            temporary_path.write_bytes(b"candidate")
            Path(f"{temporary_path}-wal").write_bytes(b"pending")
            return 2, 2

    with pytest.raises(DataValidationError, match="WAL/SHM"):
        SidecarBuilder(data_dir, index_path, validator=TinyValidator(data_dir)).build()

    assert index_path.read_bytes() == b"known-good-index"
    assert list(index_path.parent.glob(".catalog.sqlite.*.tmp*")) == []


def test_product_store_preserves_nulls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.retrieval import sqlite_store
    from tests.unit.test_wands_validator import write_tiny_wands

    data_dir = write_tiny_wands(tmp_path / "wands")
    index_path = tmp_path / "index" / "catalog.sqlite"
    TinyBuilder(data_dir, index_path, validator=TinyValidator(data_dir)).build()
    monkeypatch.setattr(sqlite_store, "EXPECTED_PRODUCT_COUNT", 2)
    store = SQLiteStore(index_path)

    product = store.get_product(20)

    assert product.product_class is None
    assert product.product_description is None
    assert product.average_rating is None
    assert store.status()["index_ready"] is True
