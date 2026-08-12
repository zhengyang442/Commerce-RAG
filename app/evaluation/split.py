from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.config import PROJECT_ROOT

DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "evaluation" / "split_v1.json"


@dataclass(frozen=True, slots=True)
class EvaluationSplit:
    version: str
    data_commit: str
    seed: int
    algorithm: str
    dev: tuple[int, ...]
    test: tuple[int, ...]
    all: tuple[int, ...]
    manifest_path: str


def load_split(query_ids: list[int], manifest_path: Path = DEFAULT_SPLIT_PATH) -> EvaluationSplit:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"冻结评测清单不存在：{manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"冻结评测清单不是有效 JSON：{manifest_path}") from error

    if not isinstance(payload, dict):
        raise ValueError("冻结评测清单必须是 JSON 对象")
    dev = _read_query_ids(payload, "dev_query_ids")
    test = _read_query_ids(payload, "test_query_ids")
    observed = tuple(sorted(query_ids))

    if len(observed) != len(set(observed)):
        raise ValueError("输入评测 query_id 含重复值")
    if len(dev) != 384 or len(test) != 96:
        raise ValueError("冻结评测清单不是固定的 384/96 划分")
    if set(dev).intersection(test):
        raise ValueError("冻结评测清单的 dev/test 存在重叠")
    if set(dev).union(test) != set(observed):
        raise ValueError("冻结评测清单与当前 query_id 集合不一致")

    version = payload.get("version")
    data_commit = payload.get("data_commit")
    seed = payload.get("seed")
    algorithm = payload.get("algorithm")
    if version != "split_v1" or not isinstance(data_commit, str):
        raise ValueError("冻结评测清单版本或数据提交无效")
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(algorithm, str):
        raise ValueError("冻结评测清单的生成元数据无效")

    return EvaluationSplit(
        version=version,
        data_commit=data_commit,
        seed=seed,
        algorithm=algorithm,
        dev=dev,
        test=test,
        all=observed,
        manifest_path=str(manifest_path),
    )


def _read_query_ids(payload: dict[str, object], field: str) -> tuple[int, ...]:
    values = payload.get(field)
    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise ValueError(f"冻结评测清单的 {field} 必须是整数数组")
    if len(values) != len(set(values)):
        raise ValueError(f"冻结评测清单的 {field} 含重复值")
    return tuple(values)
