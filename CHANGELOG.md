# Changelog

本项目采用语义化版本的简化形式记录公开能力变化。

## [Unreleased] - v0.5

### Stage 1

- 建立与私有工程历史分离的公开仓库。
- 增加公开架构、数据治理、评测口径、演进记录和机器可读 benchmark。
- 增加公开轻量检查、GitHub Actions、贡献和安全文档。
- 本地与 GitHub Actions 通过 109 个测试、benchmark 契约和全部公开历史扫描。

### Stage 2

- 采用 MIT License。
- 将 Web UI 改为访客优先、可展开开发者诊断的双模式。
- 固定中文跨语言、多条件排除和价格/折扣安全拒答三个演示故事。
- 搜索与回答使用独立分钟限流，真实查询改写与回答生成共享每日额度。
- 额度耗尽、模型超时或输出无效时保留检索结果并明确降级。

## [0.4.0] - 2026-08-12

- 增加规则优先、LLM 按需的中文查询理解。
- 增加属性、否定条件和无数据商业意图识别。
- 增加中文回归集以及公网预览安全边界。
- 保持英文 Hybrid nDCG@10 `0.747611` 不回退。

## [0.3.0]

- 增加 BM25 + Vector 的 RRF Hybrid 并设为 Web 默认策略。
- 增加可选 Cross-encoder Reranker；因本机 P95 超过 1.4 秒保持实验状态。

## [0.2.0]

- 增加 FastEmbed + sqlite-vec 向量索引和独立 Vector 基线。
- 固定模型 revision、资产哈希和向量索引 metadata。

## [0.1.0]

- 建立固定 WANDS 数据、FTS5/BM25、FastAPI、EvidencePack、降级和冻结评测基线。
