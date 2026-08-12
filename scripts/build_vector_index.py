from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_VECTOR_INDEX_PATH,
)
from app.core.errors import CommerceRAGError
from app.ingestion.vector_builder import VectorIndexBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 CommerceRAG vector_v1 向量索引")
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--vector-index-path", type=Path, default=DEFAULT_VECTOR_INDEX_PATH)
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "vector_index")
    args = parser.parse_args()
    try:
        report = VectorIndexBuilder(
            catalog_path=args.catalog_path,
            vector_index_path=args.vector_index_path,
            embedding_cache_dir=args.embedding_cache_dir,
            batch_size=args.batch_size,
        ).build()
    except (CommerceRAGError, OSError, ValueError) as error:
        parser.exit(1, f"向量索引构建失败：{error}\n")
    payload = report.to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "vector_v1_build.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**payload, "report_path": str(report_path)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
