from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import DEFAULT_RAW_DATA_DIR
from app.core.errors import DataValidationError
from app.ingestion.wands_validator import WandsValidator


def main() -> int:
    parser = argparse.ArgumentParser(description="校验固定版本的 WANDS TSV 数据")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    args = parser.parse_args()

    try:
        report = WandsValidator(args.data_dir).validate()
    except DataValidationError as error:
        parser.exit(1, f"WANDS 校验失败：{error}\n")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
