from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import IndexNotReadyError
from app.retrieval.sqlite_store import SQLiteStore


def test_missing_index_is_not_ready(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "missing.sqlite")

    assert store.status()["index_ready"] is False
    with pytest.raises(IndexNotReadyError):
        store.connect()
