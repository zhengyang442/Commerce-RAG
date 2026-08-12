from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_VECTOR_INDEX_PATH,
)
from app.core.errors import CommerceRAGError
from app.evaluation.runner import EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="运行冻结的 CommerceRAG 检索评测")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "evaluation")
    parser.add_argument(
        "--strategy", choices=("bm25", "vector", "hybrid", "rerank"), default="bm25"
    )
    parser.add_argument("--vector-index-path", type=Path, default=DEFAULT_VECTOR_INDEX_PATH)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="冻结的 split_v1.json；默认读取仓库内受版本控制的清单",
    )
    args = parser.parse_args()

    try:
        runner_kwargs = {
            "data_dir": args.data_dir,
            "index_path": args.index_path,
            "output_dir": args.output_dir,
            "strategy": args.strategy,
            "vector_index_path": args.vector_index_path,
        }
        if args.split_path is not None:
            runner_kwargs["split_path"] = args.split_path
        payload = EvaluationRunner(**runner_kwargs).run()
    except (CommerceRAGError, OSError, ValueError) as error:
        parser.exit(1, f"评测失败：{error}\n")
    print(json.dumps({"metrics": payload["metrics"], "reports": payload["report_paths"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
