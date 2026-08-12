from __future__ import annotations

import json

import pytest

from app.evaluation.split import DEFAULT_SPLIT_PATH, load_split


def test_version_controlled_split_has_expected_sizes() -> None:
    payload = json.loads(DEFAULT_SPLIT_PATH.read_text(encoding="utf-8"))
    query_ids = payload["dev_query_ids"] + payload["test_query_ids"]

    first = load_split(query_ids)
    second = load_split(list(reversed(query_ids)))

    assert first == second
    assert first.version == "split_v1"
    assert len(first.dev) == 384
    assert len(first.test) == 96
    assert len(first.all) == 480
    assert set(first.dev).isdisjoint(first.test)
    assert set(first.dev) | set(first.test) == set(first.all)


def test_split_rejects_query_set_drift() -> None:
    payload = json.loads(DEFAULT_SPLIT_PATH.read_text(encoding="utf-8"))
    query_ids = payload["dev_query_ids"] + payload["test_query_ids"]
    changed = query_ids.copy()
    changed[-1] = 999_999

    with pytest.raises(ValueError, match="当前 query_id 集合不一致"):
        load_split(changed)
