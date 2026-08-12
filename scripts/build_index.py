from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import DEFAULT_INDEX_PATH, DEFAULT_RAW_DATA_DIR
from app.core.errors import CommerceRAGError
from app.ingestion.sqlite_builder import SQLiteIndexBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 CommerceRAG SQLite FTS5 索引")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    try:
        report = SQLiteIndexBuilder(args.data_dir, args.index_path).build()
    except (CommerceRAGError, OSError) as error:
        parser.exit(1, f"索引构建失败：{error}\n")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
