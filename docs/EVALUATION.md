# 公开评测说明

## 评测对象

- 数据：Wayfair ANnotation Dataset（WANDS）。
- 固定提交：`3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`。
- 商品：42,994；查询：480；原始人工标注：233,448。
- 固定划分：384 dev / 96 test，seed 42；同时报告全部 480 条。
- 检索 Top-K：10。

冻结清单位于 `data/evaluation/split_v1.json`。评测每次读取并校验该清单，不重新随机划分。

## 指标语义

| 指标 | 语义 |
| --- | --- |
| nDCG@10 | Exact=2、Partial=1、Irrelevant=0，考虑相关性等级和排名位置 |
| Recall@10 | Exact 与 Partial 均视为 relevant |
| MRR@10 | 首个 Exact 结果的倒数排名；没有 Exact 时为 0 |
| Judged@10 | Top-10 中存在人工标注的结果比例，分母固定为 10 |
| RelevantHit@10 | 至少命中一个 Exact 或 Partial 的查询比例 |
| ExactHit@10 | 至少命中一个 Exact 的查询比例 |

同一 query-product 的重复或冲突标注按最高增益聚合。未标注结果在基准计算中按 0 增益处理，只是评测约定，不能描述为人工判定的 Irrelevant。

WANDS 每条查询平均存在大量 relevant 商品，Top-10 的 Recall 上限受截断影响，因此不应脱离数据分布单独解读 Recall 数值。

## 冻结结果

| 策略 | nDCG@10 | Recall@10 | MRR@10 | Judged@10 | RelevantHit@10 | ExactHit@10 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.666747 | 0.056392 | 0.539732 | 0.782292 | 0.939583 | 0.666667 | 29.548ms |
| Vector | 0.731349 | 0.059982 | 0.555916 | 0.821667 | 0.962500 | 0.679167 | 28.511ms |
| Hybrid | **0.747611** | **0.061922** | **0.582816** | **0.844375** | **0.966667** | **0.702083** | **58.255ms** |
| Reranker | 0.759225 | 0.062096 | 0.604684 | 0.856458 | 0.966667 | 0.706250 | 1,416.331ms |

延迟来自 Python 3.11/macOS ARM64 的一次冻结本地运行，不是跨机器 benchmark 或部署 SLA。Hybrid 的质量明显高于 BM25，同时仍在项目的交互延迟预算内，因此成为默认。Reranker 只获得小幅质量提升，但 P95 超过 1.4 秒，未通过 500ms 默认门槛。

## 中文查询理解评测

30 条冻结中文需求覆盖商品类别、属性、否定条件和数据禁区。它是有预期类目的回归集，不是人工 qrels 排序评测，因此指标使用 CategoryAccuracy@1 和 CategoryHit@3，不能与 WANDS nDCG 混用。

| 模式 | 有效改写 | 必需词覆盖 | 禁区意图 | 排除条件 | Top-1 | Top-3 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules | 100% | 98.33% | 100% | 100% | **86.67%** | **96.67%** | **0.102ms** |
| Full LLM | 100% | 75.56% | 100% | 100% | 76.67% | 90.00% | 1,212.575ms |

因此 v0.4 使用规则优先，只在规则处理后仍存在未翻译中文时调用模型。真实 LLM 实验会产生供应商费用，默认测试与 CI 不执行它。

## 复现

完成 README 中的数据和索引准备后：

```bash
uv run python -m scripts.evaluate --strategy bm25
uv run python -m scripts.evaluate --strategy vector
uv run python -m scripts.evaluate --strategy hybrid
uv run python -m scripts.evaluate --strategy rerank
uv run python -m scripts.evaluate_chinese
```

报告生成到被 Git 忽略的 `artifacts/evaluation/`。导出公开摘要：

```bash
uv run python -m scripts.export_public_benchmark
git diff --exit-code -- benchmarks/releases/v0.4.json
```

公开 JSON 只保留汇总指标、配置、数据版本和源报告 SHA-256，不保留逐查询文本。当前机器可读证据是 [`benchmarks/releases/v0.4.json`](../benchmarks/releases/v0.4.json)。
