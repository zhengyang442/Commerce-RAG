from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import SearchRequest


def test_search_request_defaults_and_bounds() -> None:
    assert SearchRequest(query="chair").top_k == 10
    assert SearchRequest(query="chair").retrieval_strategy == "bm25"
    assert SearchRequest(query="chair", retrieval_strategy="vector").retrieval_strategy == "vector"
    assert SearchRequest(query="chair", retrieval_strategy="hybrid").retrieval_strategy == "hybrid"
    assert SearchRequest(query="chair", retrieval_strategy="rerank").retrieval_strategy == "rerank"
    assert SearchRequest(query="chair", top_k=20).top_k == 20
    with pytest.raises(ValidationError):
        SearchRequest(query="chair", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="chair", top_k=21)
    with pytest.raises(ValidationError):
        SearchRequest(query=" ")
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 501)
    with pytest.raises(ValidationError):
        SearchRequest(query="chair", retrieval_strategy="unknown")
