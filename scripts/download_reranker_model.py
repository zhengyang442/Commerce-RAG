from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import DEFAULT_RERANKER_CACHE_DIR
from app.retrieval.embedding import (
    RERANKER_MODEL,
    RERANKER_MODEL_REVISION,
    RERANKER_MODEL_SHA256,
    FastEmbedReranker,
    reranker_cache_ready,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并校验 CommerceRAG v0.3 Reranker")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_RERANKER_CACHE_DIR)
    args = parser.parse_args()

    if not reranker_cache_ready(args.cache_dir):
        FastEmbedReranker(cache_dir=args.cache_dir)
    if not reranker_cache_ready(args.cache_dir):
        parser.exit(1, "Reranker 模型下载完成后仍未通过固定 SHA-256 校验\n")
    print(
        json.dumps(
            {
                "status": "ready",
                "model": RERANKER_MODEL,
                "revision": RERANKER_MODEL_REVISION,
                "model_sha256": RERANKER_MODEL_SHA256,
                "cache_dir": str(args.cache_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
