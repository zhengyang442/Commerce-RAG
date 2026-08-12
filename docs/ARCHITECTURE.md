# 架构与信任边界

## 目标

CommerceRAG 将商品搜索拆成查询理解、候选检索、证据构造、可选生成和核验五层。任何模型输出都不能替代数据字段，外部商品文本也不能成为系统指令。

## 请求链路

```text
用户问题
  -> QueryUnderstandingService
       -> 中文规则词典与意图/否定条件解析
       -> 仅在仍含未翻译中文时调用可选 LLM Rewriter
  -> RetrievalService
       -> SQLite FTS5/BM25 Top-50
       -> BGE small English + sqlite-vec Top-50
       -> RRF(k=60) 融合
       -> 可选 Cross-encoder 重排
  -> EvidencePack
       -> 只包含 Top-K 商品白名单字段
  -> AnswerOrchestrator
       -> 未配置/超时/输出非法：retrieval_only
       -> 输出合法：结构化回答 + 引用验证
  -> FastAPI / Web UI
       -> /api/products/{id} 核验原始字段
```

## 主要组件

| 组件 | 目录 | 职责 |
| --- | --- | --- |
| 数据导入 | `app/ingestion/` | 固定 WANDS 数据契约、原子 FTS/向量索引构建 |
| 检索 | `app/retrieval/` | BM25、Vector、RRF Hybrid、可选 Reranker |
| 查询理解 | `app/query_understanding/` | 中文规则、商业禁区、排除条件、按需改写 |
| 证据与生成 | `app/generation/` | EvidencePack、模型适配、引用校验与降级 |
| API | `app/api/` | 类型化请求/响应、错误契约、公网预览边界 |
| 评测 | `app/evaluation/` | qrels 聚合、冻结 split、指标和报告生成 |
| Web | `app/static/` | 无构建链中文界面和安全 DOM 渲染 |

## 数据与索引

原始 WANDS 文件、SQLite 索引和模型缓存全部是可重建运行资产，不进入 Git：

```text
data/raw/wands/                 # 固定原始数据
data/index/catalog.sqlite       # FTS5 商品库
data/index/catalog_vectors.sqlite
data/models/fastembed/
data/models/fastembed-reranker/
```

FTS 索引在临时文件中完成事务、FTS rebuild、数量检查和 `PRAGMA integrity_check`，关闭连接并确认没有 WAL/SHM sidecar 后才原子替换正式文件。向量索引 metadata 绑定数据提交、模型 revision、模型文件哈希、维度和商品映射摘要。

## 信任边界

1. 商品名称、描述、特征、链接和 HTML 都是外部不可信数据，只能作为证据字段。
2. 模型只接收当前 Top-K EvidencePack，不接收整个数据库，也不能执行商品文本中的命令。
3. 模型输出必须符合结构化契约，引用必须指向当前 EvidencePack；失败立即退化为只检索。
4. 价格、库存、配送、促销和售后不在数据集中，系统必须陈述限制，不能生成承诺。
5. 默认日志记录请求 ID、阶段和耗时，不记录完整查询；API Key 在日志格式化层脱敏。

## 公网预览边界

`RAG_PUBLIC_PREVIEW=true` 时关闭 OpenAPI、ReDoc 和 schema 入口，启用：

- 精确 Host 白名单；
- 精确 CORS 来源；
- 搜索/回答请求体上限；
- 单进程每 IP 速率边界；
- 外部模型调用并发上限；
- `X-Content-Type-Options`、`Referrer-Policy`、`X-Frame-Options`。

这些能力只适合个人项目预览。阶段三仍需要 Cloudflare 提供 HTTPS、外层流量治理和隧道边界；Uvicorn 只监听 `127.0.0.1`。
