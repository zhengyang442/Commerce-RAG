from __future__ import annotations

from dataclasses import dataclass

from app.core.models import SearchResult
from app.generation.policy import FIXED_LIMITATIONS


@dataclass(frozen=True, slots=True)
class RetrievalOnlyContent:
    answer: str
    limitations: list[str]


def render_retrieval_only(
    results: list[SearchResult], *, reason: str = "not_configured"
) -> RetrievalOnlyContent:
    if not results:
        answer = "没有检索到可用于回答的商品证据，当前不足以判断合适商品。"
    elif reason == "not_configured":
        answer = (
            f"已检索到 {len(results)} 个候选商品。当前未配置生成模型，"
            "请根据下方商品证据和引用编号核验结果。"
        )
    elif reason == "quota_exhausted":
        answer = (
            f"已检索到 {len(results)} 个候选商品。今日生成额度已用完，"
            "检索仍然可用，请直接核验下方商品证据。"
        )
    else:
        answer = f"已检索到 {len(results)} 个候选商品，但生成阶段不可用，已保留全部检索结果供核验。"
    return RetrievalOnlyContent(answer=answer, limitations=list(FIXED_LIMITATIONS))
