#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

echo "[1/5] v0.1 全量回归与 v0.2 代码检查"
bash scripts/check.sh

echo "[2/5] 42,994 商品 vector_v1 索引"
uv run python -m scripts.build_vector_index

echo "[3/5] 480 查询 vector_v1 冻结评测"
uv run python -m scripts.evaluate --strategy vector

echo "[4/5] BM25 / Vector 语义回归对照"
uv run python -m scripts.evaluate_semantic_regression

echo "[5/5] DeepSeek 20 条真实 RAG 验收"
if [[ -z "${RAG_LLM_API_STYLE:-}" || -z "${RAG_LLM_API_KEY:-}" || -z "${RAG_LLM_MODEL:-}" ]]; then
  echo "请先执行 set -a; source .env; set +a，再运行 v0.2 全量验收。" >&2
  exit 1
fi
uv run python -m scripts.evaluate_llm

echo "CommerceRAG v0.2 全量验收通过"
