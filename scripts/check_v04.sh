#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

echo "[1/5] v0.4 代码、测试与 v0.1 基线回归"
bash scripts/check.sh

echo "[2/5] v0.3 默认 Hybrid 英文冻结回归"
uv run python -m scripts.evaluate --strategy hybrid

echo "[3/5] v0.4 中文规则保底评测"
uv run python -m scripts.evaluate_chinese

echo "[4/5] v0.4 报告门槛"
uv run python - <<'PY'
import json
from pathlib import Path

bm25 = json.loads(Path("artifacts/evaluation/bm25_v1.json").read_text(encoding="utf-8"))
hybrid = json.loads(Path("artifacts/evaluation/hybrid_v1.json").read_text(encoding="utf-8"))
chinese = json.loads(
    Path("artifacts/evaluation/chinese_regression_v1_rules.json").read_text(encoding="utf-8")
)
metrics = chinese["metrics"]
assert abs(bm25["metrics"]["all"]["ndcg_at_10"] - 0.666747) < 0.000001
assert hybrid["metrics"]["all"]["ndcg_at_10"] >= 0.740135  # v0.3 下降不超过 1%
assert metrics["query_rewrite_valid_rate"] == 1.0
assert metrics["unsupported_intent_accuracy"] == 1.0
assert metrics["exclusion_detection_rate"] == 1.0
assert metrics["category_accuracy_at_1"] >= 0.85
assert metrics["category_hit_at_3"] >= 0.80
print("中文规则保底、意图边界和英文 Hybrid 回归达到 v0.4 门槛")
PY

echo "[5/5] 可选真实 DeepSeek 中文改写评测"
if [[ "${RAG_RUN_LLM_EVAL:-0}" == "1" ]]; then
  uv run python -m scripts.evaluate_chinese --use-llm
else
  echo "未设置 RAG_RUN_LLM_EVAL=1；跳过会产生真实 API 调用的评测。"
fi

echo "CommerceRAG v0.4 自动验收通过"
