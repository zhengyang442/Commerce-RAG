#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

echo "[1/5] v0.1 回归与 v0.3 代码检查"
bash scripts/check.sh

echo "[2/5] Vector / Hybrid 冻结评测"
uv run python -m scripts.evaluate --strategy vector
uv run python -m scripts.evaluate --strategy hybrid

echo "[3/5] Reranker 模型完整性与冻结评测"
uv run python -m scripts.download_reranker_model
uv run python -m scripts.evaluate --strategy rerank

echo "[4/5] 四策略语义回归"
uv run python -m scripts.evaluate_semantic_regression

echo "[5/5] v0.3 报告门槛"
uv run python - <<'PY'
import json
from pathlib import Path

reports = {
    name: json.loads(Path(f"artifacts/evaluation/{name}_v1.json").read_text(encoding="utf-8"))
    for name in ("bm25", "vector", "hybrid", "rerank")
}
bm25 = reports["bm25"]["metrics"]["all"]
hybrid = reports["hybrid"]["metrics"]["all"]
rerank = reports["rerank"]["metrics"]["all"]
assert hybrid["ndcg_at_10"] >= bm25["ndcg_at_10"]
assert hybrid["relevant_hit_at_10"] >= 0.939583
assert hybrid["exact_hit_at_10"] >= 0.696667
assert hybrid["latency_ms"]["p95"] <= 200
assert rerank["ndcg_at_10"] >= hybrid["ndcg_at_10"]
assert rerank["latency_ms"]["p95"] > 500
print("Hybrid 达标；Reranker 质量提升但延迟不达默认门槛")
PY

echo "CommerceRAG v0.3 全量验收通过"
