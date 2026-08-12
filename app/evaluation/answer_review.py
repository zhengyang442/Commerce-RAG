from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_INDEX_PATH, Settings
from app.generation.llm.contracts import GeneratedAnswer, GeneratedRecommendation, LLMResult
from app.generation.llm.errors import (
    LLMInvalidOutputError,
    LLMModelError,
    LLMProviderError,
    LLMTimeoutError,
)
from app.generation.orchestrator import AnswerOrchestrator

ANSWER_REVIEW_VERSION = "answer_review_v1"
FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "answer_review_cases.jsonl"
)
ERRORS = {
    "timeout": LLMTimeoutError,
    "provider_error": LLMProviderError,
    "model_error": LLMModelError,
    "invalid_output": LLMInvalidOutputError,
}


class ExactEvidenceAdapter:
    async def generate(self, pack) -> LLMResult:
        item = pack.items[0]
        return LLMResult(
            answer=GeneratedAnswer(
                recommendations=[
                    GeneratedRecommendation(
                        citation_id=item.citation_id,
                        reason=item.product_name,
                        supporting_fields=["product_name"],
                    )
                ]
            ),
            model="controlled-exact-evidence-fixture",
        )

    async def aclose(self) -> None:
        return None


class ErrorAdapter:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def generate(self, pack) -> LLMResult:
        raise self.error

    async def aclose(self) -> None:
        return None


async def run_answer_review(
    *,
    fixture_path: Path = FIXTURE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    output_dir: Path = DEFAULT_ARTIFACTS_DIR / "answer_review",
) -> dict[str, Any]:
    cases = load_cases(fixture_path)
    results = []
    for case in cases:
        checks = await evaluate_case(case, index_path)
        results.append(
            {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "passed": all(checks.values()),
                "checks": checks,
            }
        )
    scenario_counts = Counter(result["scenario"] for result in results)
    payload = {
        "review_version": ANSWER_REVIEW_VERSION,
        "case_count": len(results),
        "passed_count": sum(result["passed"] for result in results),
        "failed_count": sum(not result["passed"] for result in results),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "review_method": {
            "citation_and_support": "controlled output quotes product_name exactly",
            "missing_information": "retrieval_only fixed limitations",
            "generation_failures": "typed fake adapter errors",
            "live_provider_used": False,
        },
        "cases": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{ANSWER_REVIEW_VERSION}.json"
    markdown_path = output_dir / f"{ANSWER_REVIEW_VERSION}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_answer_review_markdown(payload), encoding="utf-8")
    payload["report_paths"] = {"json": str(json_path), "markdown": str(markdown_path)}
    return payload


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    case_ids = [case["case_id"] for case in cases]
    if len(cases) != 30 or len(set(case_ids)) != 30:
        raise ValueError("回答检查集必须正好包含 30 个唯一 case_id")
    return cases


async def evaluate_case(case: dict[str, Any], index_path: Path) -> dict[str, bool]:
    scenario = case["scenario"]
    if scenario in {"explicit_match", "multi_condition", "partial_match"}:
        settings = configured_settings(index_path)
        outcome = await AnswerOrchestrator(
            settings, adapter_factory=lambda _: ExactEvidenceAdapter()
        ).answer(case["query"], 10)
        citation = outcome.citations[0] if outcome.citations else None
        cited_result = next(
            (
                item
                for item in outcome.results
                if citation and item.citation_id == citation.citation_id
            ),
            None,
        )
        return {
            "results_present": bool(outcome.results),
            "mode_is_rag": outcome.mode == "rag",
            "citation_in_results": cited_result is not None,
            "field_support": bool(
                cited_result
                and citation
                and citation.supporting_fields == ["product_name"]
                and cited_result.product_name in outcome.answer
            ),
            "fixed_limitations_present": len(outcome.limitations) == 4,
        }
    if scenario == "no_result":
        outcome = await AnswerOrchestrator(Settings(index_path=index_path)).answer(
            case["query"], 10
        )
        return {
            "no_results": not outcome.results,
            "retrieval_only": outcome.mode == "retrieval_only",
            "conservative_language": "不足以判断" in outcome.answer,
            "no_citations": not outcome.citations,
        }
    if scenario == "missing_information":
        outcome = await AnswerOrchestrator(Settings(index_path=index_path)).answer(
            case["query"], 10
        )
        limitations = " ".join(outcome.limitations)
        return {
            "retrieval_only": outcome.mode == "retrieval_only",
            "results_present": bool(outcome.results),
            "commercial_limits_present": all(
                term in limitations for term in ("价格", "库存", "配送", "售后", "评论正文")
            ),
            "no_generated_citations": not outcome.citations,
        }
    if scenario == "generation_failure":
        reason = case["fallback_reason"]
        if reason == "not_configured":
            settings = Settings(index_path=index_path)
            factory = None
        else:
            settings = configured_settings(index_path)
            error = ERRORS[reason]()

            def factory(_: Settings) -> ErrorAdapter:
                return ErrorAdapter(error)

        baseline = await AnswerOrchestrator(Settings(index_path=index_path)).answer(
            case["query"], 10
        )
        outcome = await AnswerOrchestrator(settings, adapter_factory=factory).answer(
            case["query"], 10
        )
        return {
            "retrieval_only": outcome.mode == "retrieval_only",
            "fallback_reason_matches": outcome.fallback_reason == reason,
            "results_preserved": [item.product_id for item in outcome.results]
            == [item.product_id for item in baseline.results],
            "explicit_fallback": "检索" in outcome.answer,
        }
    raise ValueError(f"未知回答检查场景：{scenario}")


def configured_settings(index_path: Path) -> Settings:
    return Settings(
        index_path=index_path,
        llm_api_style="anthropic",
        llm_api_key="controlled-fixture-key",
        llm_model="controlled-fixture-model",
    )


def render_answer_review_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CommerceRAG 回答检查集 v1",
        "",
        f"- 案例：{payload['case_count']}",
        f"- 自动检查通过：{payload['passed_count']}",
        f"- 自动检查失败：{payload['failed_count']}",
        "- 真实供应商：未使用",
        "- 语义方法：受控回答逐字引用 product_name，验证引用、字段支持和降级不变量",
        "",
        "| case | scenario | status | checks |",
        "| --- | --- | --- | --- |",
    ]
    for result in payload["cases"]:
        checks = ", ".join(
            f"{name}={'pass' if passed else 'fail'}" for name, passed in result["checks"].items()
        )
        lines.append(
            f"| {result['case_id']} | {result['scenario']} | "
            f"{'pass' if result['passed'] else 'fail'} | {checks} |"
        )
    lines.extend(
        [
            "",
            "该报告验证第一版的引用、字段存在性、数据禁区和降级机制。",
            "未配置真实项目 LLM 密钥，因此不代表真实模型自然语言质量评测。",
            "",
        ]
    )
    return "\n".join(lines)
