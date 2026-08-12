from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import DEFAULT_INDEX_PATH
from app.core.errors import CommerceRAGError
from app.retrieval.service import DEFAULT_TOP_K, RetrievalService


def main() -> int:
    parser = argparse.ArgumentParser(description="执行 CommerceRAG 商品检索")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--strategy", choices=("bm25", "vector", "hybrid", "rerank"), default="bm25"
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    try:
        response = RetrievalService(args.index_path).search(args.query, args.top_k, args.strategy)
    except CommerceRAGError as error:
        parser.exit(1, f"检索失败：{error}\n")
    payload = {
        "query": response.query,
        "normalized_query": response.normalized_query,
        "results": [result.model_dump() for result in response.results],
        "latency_ms": response.latency_ms,
        "retrieval_strategy": response.retrieval_strategy,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"查询：{response.normalized_query}（{response.latency_ms:.3f} ms）")
        for result in response.results:
            print(
                f"{result.citation_id} #{result.rank} {result.product_id} "
                f"{result.product_name} score={result.score:.6f} "
                f"fields={','.join(result.matched_fields)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
