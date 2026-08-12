from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_INDEX_PATH
from app.evaluation.answer_review import FIXTURE_PATH, run_answer_review


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 CommerceRAG 30 条回答检查集")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "answer_review")
    args = parser.parse_args()
    try:
        payload = asyncio.run(
            run_answer_review(
                fixture_path=args.fixture,
                index_path=args.index_path,
                output_dir=args.output_dir,
            )
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"回答检查失败：{error}\n")
    print(
        json.dumps(
            {
                "case_count": payload["case_count"],
                "passed_count": payload["passed_count"],
                "failed_count": payload["failed_count"],
                "reports": payload["report_paths"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
