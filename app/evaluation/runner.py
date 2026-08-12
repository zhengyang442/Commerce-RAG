from __future__ import annotations

import importlib.metadata
import json
import platform
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_RERANKER_CACHE_DIR,
    DEFAULT_VECTOR_INDEX_PATH,
)
from app.core.models import RetrievalStrategy, SearchResult
from app.evaluation.metrics import GAIN_MAPPING, calculate_query_metrics, percentile
from app.evaluation.qrels import Qrels, load_qrels
from app.evaluation.split import DEFAULT_SPLIT_PATH, EvaluationSplit, load_split
from app.ingestion.wands_models import QueryRecord
from app.ingestion.wands_validator import WandsValidator
from app.retrieval.embedding import (
    DOCUMENT_TEXT_VERSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_MODEL_REVISION,
    RERANKER_MODEL_SHA256,
)
from app.retrieval.query_normalization import normalize_query
from app.retrieval.service import RetrievalService
from app.retrieval.sqlite_store import SCHEMA_VERSION, WANDS_COMMIT

EVALUATION_VERSIONS: dict[RetrievalStrategy, str] = {
    "bm25": "bm25_v1",
    "vector": "vector_v1",
    "hybrid": "hybrid_v1",
    "rerank": "rerank_v1",
}


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: int
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    judged_at_10: float
    relevant_hit_at_10: bool
    exact_hit_at_10: bool
    no_judged_top_10: bool
    latency_ms: float
    no_relevant: bool
    no_exact: bool


class EvaluationRunner:
    def __init__(
        self,
        *,
        data_dir: Path = DEFAULT_RAW_DATA_DIR,
        index_path: Path = DEFAULT_INDEX_PATH,
        output_dir: Path = DEFAULT_ARTIFACTS_DIR / "evaluation",
        split_path: Path = DEFAULT_SPLIT_PATH,
        strategy: RetrievalStrategy = "bm25",
        vector_index_path: Path = DEFAULT_VECTOR_INDEX_PATH,
        embedding_cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        reranker_cache_dir: Path = DEFAULT_RERANKER_CACHE_DIR,
    ) -> None:
        self.validator = WandsValidator(data_dir)
        self.strategy = strategy
        self.service = RetrievalService(
            index_path,
            vector_index_path=vector_index_path,
            embedding_cache_dir=embedding_cache_dir,
            reranker_cache_dir=reranker_cache_dir,
        )
        self.output_dir = output_dir
        self.split_path = split_path

    def run(self) -> dict[str, object]:
        validation = self.validator.validate()
        queries = {query.query_id: query for query in self.validator.iter_queries()}
        qrels = load_qrels(self.validator)
        split = load_split(list(queries), self.split_path)
        if (len(split.dev), len(split.test), len(split.all)) != (384, 96, 480):
            raise ValueError("评测划分不是固定的 384/96/480")
        if split.data_commit != WANDS_COMMIT:
            raise ValueError("冻结评测清单的数据提交与当前 WANDS 版本不一致")

        evaluations = self._evaluate_all(queries, qrels)
        payload: dict[str, object] = {
            "evaluation_version": EVALUATION_VERSIONS[self.strategy],
            "schema_version": SCHEMA_VERSION,
            "data_commit": WANDS_COMMIT,
            "retrieval": {
                "strategy": self.strategy,
                "engine": {
                    "bm25": "sqlite_fts5_bm25",
                    "vector": "fastembed_sqlite_vec",
                    "hybrid": "bm25_vector_rrf",
                    "rerank": "bm25_vector_rrf_cross_encoder",
                }[self.strategy],
                "fields": [
                    "product_name",
                    "product_class",
                    "category_hierarchy",
                    "product_description",
                    "product_features",
                ],
                "field_weights": "equal" if self.strategy == "bm25" else None,
                "query_operator": "quoted_tokens_or" if self.strategy == "bm25" else None,
                "top_k": 10,
                "warmup": "one fixed query before timing",
                "embedding_model": (EMBEDDING_MODEL if self.strategy != "bm25" else None),
                "embedding_dimension": EMBEDDING_DIMENSION if self.strategy != "bm25" else None,
                "distance_metric": "cosine" if self.strategy != "bm25" else None,
                "document_text_version": (
                    DOCUMENT_TEXT_VERSION if self.strategy != "bm25" else None
                ),
                "candidate_k_per_channel": 50 if self.strategy in {"hybrid", "rerank"} else None,
                "rrf_k": 60 if self.strategy in {"hybrid", "rerank"} else None,
                "reranker_model": RERANKER_MODEL if self.strategy == "rerank" else None,
                "reranker_model_revision": (
                    RERANKER_MODEL_REVISION if self.strategy == "rerank" else None
                ),
                "reranker_model_sha256": (
                    RERANKER_MODEL_SHA256 if self.strategy == "rerank" else None
                ),
            },
            "relevance": {
                "raw_mapping": GAIN_MAPPING,
                "aggregation": "max_gain",
                "recall_relevant": ["Exact", "Partial"],
                "mrr_high_relevance": ["Exact"],
                "no_relevant_score": 0.0,
                "no_exact_mrr": 0.0,
                "unjudged_metric_convention": (
                    "Unjudged products receive gain 0 only when computing benchmark metrics; "
                    "they were not manually judged Irrelevant."
                ),
                "judged_at_10_denominator": 10,
            },
            "raw_counts": {
                "products": validation.product_count,
                "queries": validation.query_count,
                "labels": validation.label_count,
            },
            "qrels": qrels_to_dict(qrels),
            "split": split_to_dict(split),
            "metrics": {
                name: summarize(evaluations, query_ids)
                for name, query_ids in (
                    ("dev", split.dev),
                    ("test", split.test),
                    ("all", split.all),
                )
            },
            "runtime_environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
                "fastembed": (
                    importlib.metadata.version("fastembed") if self.strategy != "bm25" else None
                ),
                "sqlite_vec": (
                    importlib.metadata.version("sqlite-vec") if self.strategy != "bm25" else None
                ),
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_version = EVALUATION_VERSIONS[self.strategy]
        json_path = self.output_dir / f"{evaluation_version}.json"
        markdown_path = self.output_dir / f"{evaluation_version}.md"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")
        payload["report_paths"] = {"json": str(json_path), "markdown": str(markdown_path)}
        return payload

    def _evaluate_all(
        self, queries: dict[int, QueryRecord], qrels: Qrels
    ) -> dict[int, QueryEvaluation]:
        output: dict[int, QueryEvaluation] = {}
        if self.strategy == "bm25":
            with self.service.store.connect() as connection:
                self.service.search_connection(connection, normalize_query("wood chair"), 10)
                for query_id in sorted(queries):
                    query = queries[query_id]
                    normalized = normalize_query(query.query)
                    started = time.perf_counter()
                    results = self.service.search_connection(connection, normalized, 10)
                    latency_ms = (time.perf_counter() - started) * 1000
                    output[query_id] = self._evaluate_query(query_id, results, latency_ms, qrels)
            return output

        self.service.search("wood chair", 10, self.strategy)
        for query_id in sorted(queries):
            query = queries[query_id]
            response = self.service.search(query.query, 10, self.strategy)
            output[query_id] = self._evaluate_query(
                query_id, response.results, response.latency_ms, qrels
            )
        return output

    @staticmethod
    def _evaluate_query(
        query_id: int,
        results: list[SearchResult],
        latency_ms: float,
        qrels: Qrels,
    ) -> QueryEvaluation:
        metrics = calculate_query_metrics(
            [result.product_id for result in results], qrels.by_query.get(query_id, {})
        )
        return QueryEvaluation(
            query_id=query_id,
            ndcg_at_10=metrics.ndcg_at_10,
            recall_at_10=metrics.recall_at_10,
            mrr_at_10=metrics.mrr_at_10,
            judged_at_10=metrics.judged_at_10,
            relevant_hit_at_10=metrics.relevant_hit_at_10,
            exact_hit_at_10=metrics.exact_hit_at_10,
            no_judged_top_10=metrics.no_judged_top_10,
            latency_ms=latency_ms,
            no_relevant=metrics.no_relevant,
            no_exact=metrics.no_exact,
        )


def summarize(
    evaluations: dict[int, QueryEvaluation], query_ids: tuple[int, ...]
) -> dict[str, object]:
    selected = [evaluations[query_id] for query_id in query_ids]
    count = len(selected)
    latencies = [item.latency_ms for item in selected]
    return {
        "query_count": count,
        "ndcg_at_10": mean([item.ndcg_at_10 for item in selected]),
        "recall_at_10": mean([item.recall_at_10 for item in selected]),
        "mrr_at_10": mean([item.mrr_at_10 for item in selected]),
        "judged_at_10": mean([item.judged_at_10 for item in selected]),
        "relevant_hit_at_10": mean([float(item.relevant_hit_at_10) for item in selected]),
        "exact_hit_at_10": mean([float(item.exact_hit_at_10) for item in selected]),
        "latency_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "no_relevant_query_count": sum(item.no_relevant for item in selected),
        "no_exact_query_count": sum(item.no_exact for item in selected),
        "no_judged_top_10_query_count": sum(item.no_judged_top_10 for item in selected),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def qrels_to_dict(qrels: Qrels) -> dict[str, object]:
    return {
        "raw_count": qrels.raw_count,
        "unique_count": qrels.unique_count,
        "duplicate_pair_count": qrels.duplicate_pair_count,
        "duplicate_annotation_count": qrels.duplicate_annotation_count,
        "conflict_pair_count": qrels.conflict_pair_count,
        "canonical_label_counts": qrels.canonical_label_counts,
    }


def split_to_dict(split: EvaluationSplit) -> dict[str, object]:
    return {
        "version": split.version,
        "data_commit": split.data_commit,
        "seed": split.seed,
        "algorithm": split.algorithm,
        "manifest_path": split.manifest_path,
        "dev_query_ids": list(split.dev),
        "test_query_ids": list(split.test),
        "all_query_ids": list(split.all),
    }


def render_markdown(payload: dict[str, object]) -> str:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    lines = [
        f"# CommerceRAG {payload['evaluation_version']} 基线",
        "",
        f"- 数据提交：`{payload['data_commit']}`",
        f"- 评测版本：`{payload['evaluation_version']}`",
        "- qrels 聚合：同一 query-product 对取最高增益（max_gain）",
        "- 增益：Exact=2、Partial=1、Irrelevant=0",
        "- 未标注（unjudged）结果仅在基准指标计算时按 0 增益处理，不代表人工判定为 Irrelevant",
        "- Recall relevant：Exact + Partial；MRR high relevance：Exact",
        "- 诊断指标：Judged@10 分母固定为 10；RelevantHit@10/ExactHit@10 为查询命中率",
        "- 划分：读取受版本控制的 data/evaluation/split_v1.json，384 dev / 96 test",
        "",
        "| split | queries | nDCG@10 | Recall@10 | MRR@10 | Judged@10 | "
        "RelevantHit@10 | ExactHit@10 | 无标注 Top-10 | P50 ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("dev", "test", "all"):
        item = metrics[name]
        latency = item["latency_ms"]
        lines.append(
            f"| {name} | {item['query_count']} | {item['ndcg_at_10']:.6f} | "
            f"{item['recall_at_10']:.6f} | {item['mrr_at_10']:.6f} | "
            f"{item['judged_at_10']:.6f} | {item['relevant_hit_at_10']:.6f} | "
            f"{item['exact_hit_at_10']:.6f} | {item['no_judged_top_10_query_count']} | "
            f"{latency['p50']:.3f} | {latency['p95']:.3f} |"
        )
    qrels = payload["qrels"]
    lines.extend(
        [
            "",
            "## 数据质量",
            "",
            f"- 原始标注：{qrels['raw_count']}",
            f"- 唯一 qrel：{qrels['unique_count']}",
            f"- 重复 query-product 对：{qrels['duplicate_pair_count']}",
            f"- 冲突对：{qrels['conflict_pair_count']}",
            "",
            (
                "报告不包含原始商品全文。冻结 test 集不得用于 Phase 1 调参。"
                if payload["evaluation_version"] == "bm25_v1"
                else "报告不包含原始商品全文。冻结 test 集不得用于向量参数调优。"
            ),
            "",
        ]
    )
    return "\n".join(lines)
