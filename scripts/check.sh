#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

echo "[1/9] Python 版本"
uv run python -c 'import sys; assert sys.version_info >= (3, 11), sys.version; print(sys.version.split()[0])'

echo "[2/9] 格式与静态检查"
uv run ruff format --check app scripts tests
uv run ruff check app scripts tests
if command -v node >/dev/null 2>&1; then
  node --check app/static/app.js
else
  echo "未安装 Node.js；跳过可选 JavaScript 语法检查（静态 UI 集成测试仍会运行）。"
fi

echo "[3/9] 自动测试"
uv run pytest -q

echo "[4/9] WANDS 全量校验"
uv run python -m scripts.validate_wands >/tmp/commercerag-validate.json

echo "[5/9] 原子构建 SQLite FTS5 索引"
uv run python -m scripts.build_index >/tmp/commercerag-build.json

echo "[6/9] 480 查询 BM25 基线"
uv run python -m scripts.evaluate >/tmp/commercerag-evaluate.json

echo "[7/9] 30 条回答检查集"
uv run python -m scripts.review_answers >/tmp/commercerag-answer-review.json

echo "[8/9] 报告契约"
uv run python - <<'PY'
import json
from pathlib import Path

evaluation = json.loads(Path("artifacts/evaluation/bm25_v1.json").read_text(encoding="utf-8"))
review = json.loads(
    Path("artifacts/answer_review/answer_review_v1.json").read_text(encoding="utf-8")
)
assert evaluation["raw_counts"] == {"labels": 233448, "products": 42994, "queries": 480}
assert [evaluation["metrics"][name]["query_count"] for name in ("dev", "test", "all")] == [
    384,
    96,
    480,
]
assert evaluation["split"]["seed"] == 42
assert evaluation["split"]["version"] == "split_v1"
assert Path("data/evaluation/split_v1.json").is_file()
for name in ("dev", "test", "all"):
    metric = evaluation["metrics"][name]
    for field in (
        "judged_at_10",
        "relevant_hit_at_10",
        "exact_hit_at_10",
        "no_judged_top_10_query_count",
    ):
        assert field in metric
assert review["case_count"] == 30
assert review["passed_count"] == 30
assert review["failed_count"] == 0
print("评测与回答检查报告契约通过")
PY

echo "[9/9] Git 忽略与密钥边界"
for generated_path in \
  data/raw/wands/product.csv \
  data/raw/wands/query.csv \
  data/raw/wands/label.csv \
  data/index/catalog.sqlite \
  data/index/catalog_vectors.sqlite \
  data/models/fastembed/CACHEDIR.TAG \
  artifacts/evaluation/bm25_v1.json; do
  if ! git check-ignore -q "${generated_path}"; then
    echo "应被 Git 忽略但未忽略：${generated_path}" >&2
    exit 1
  fi
done
if git check-ignore -q .env.example; then
  echo ".env.example 不应被 Git 忽略" >&2
  exit 1
fi
uv run python - <<'PY'
import re
from pathlib import Path

excluded = {".git", ".venv", ".pytest_cache", ".ruff_cache", "artifacts"}
patterns = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9]{32,}"),
}
violations = []
for path in Path(".").rglob("*"):
    if not path.is_file() or any(part in excluded for part in path.parts):
        continue
    if path.parts[:3] == ("data", "raw", "wands"):
        continue
    if path.suffix not in {".py", ".js", ".html", ".css", ".md", ".toml", ".sh", ".example"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for label, pattern in patterns.items():
        if pattern.search(text):
            violations.append(f"{label}: {path}")
if violations:
    raise SystemExit("检测到疑似密钥：\n" + "\n".join(violations))
print("Git 忽略与密钥模式检查通过")
PY

echo "CommerceRAG v0.1 回归与当前代码检查通过"
