#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

echo "[1/6] Python 版本"
uv run python -c 'import sys; assert sys.version_info >= (3, 11), sys.version; print(sys.version.split()[0])'

echo "[2/6] 格式与静态检查"
uv run ruff format --check app scripts tests
uv run ruff check app scripts tests
if command -v node >/dev/null 2>&1; then
  node --check app/static/app.js
else
  echo "未安装 Node.js；静态 UI 集成测试仍会运行。"
fi

echo "[3/6] 不依赖完整数据的自动测试"
uv run pytest -q

echo "[4/6] 公开 benchmark 契约"
uv run python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("benchmarks/releases/v0.4.json").read_text(encoding="utf-8"))
assert payload["release"] == "v0.4"
assert payload["data"]["commit"] == "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
assert payload["data"]["split"] == {"all": 480, "dev": 384, "seed": 42, "test": 96}
strategies = {item["strategy"]: item for item in payload["retrieval"]}
assert set(strategies) == {"bm25", "vector", "hybrid", "rerank"}
assert strategies["hybrid"]["ndcg_at_10"] >= strategies["bm25"]["ndcg_at_10"]
assert strategies["rerank"]["latency_ms"]["p95"] > 500
rules, llm = payload["chinese_query_understanding"]
assert rules["mode"] == "rules" and llm["mode"] == "llm"
assert rules["category_accuracy_at_1"] >= llm["category_accuracy_at_1"]
print("公开 benchmark 契约通过")
PY

echo "[5/6] Git 忽略边界"
for generated_path in \
  .env \
  data/raw/wands/product.csv \
  data/index/catalog.sqlite \
  data/index/catalog_vectors.sqlite \
  data/models/fastembed/CACHEDIR.TAG \
  artifacts/evaluation/bm25_v1.json \
  docs/internal/example.md \
  docs/AI审查的工程细节.md; do
  if ! git check-ignore -q "${generated_path}"; then
    echo "应被 Git 忽略但未忽略：${generated_path}" >&2
    exit 1
  fi
done
if git check-ignore -q .env.example; then
  echo ".env.example 不应被 Git 忽略" >&2
  exit 1
fi

echo "[6/6] 工作树密钥与内部文件扫描"
uv run python -m scripts.scan_public_repo --worktree-only

echo "CommerceRAG 公开仓库轻量检查通过"
