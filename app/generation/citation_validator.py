from __future__ import annotations

import re

from app.core.models import Citation
from app.generation.evidence import EvidencePack
from app.generation.llm.contracts import GeneratedAnswer
from app.generation.llm.errors import LLMInvalidOutputError
from app.generation.policy import SUPPORTING_FIELDS

FORBIDDEN_CLAIM_PATTERNS = (
    re.compile(r"\b(price|cost|discount|sale|promotion|stock|inventory)\b", re.IGNORECASE),
    re.compile(r"\b(delivery|shipping|arriv(?:e|al)|return|refund|warranty)\b", re.IGNORECASE),
    re.compile(r"价格|折扣|促销|库存|配送|到货|退换|退款|保修|售后"),
    re.compile(r"(?:\$|¥|￥)\s*\d"),
)


def validate_generated_answer(answer: GeneratedAnswer, pack: EvidencePack) -> list[Citation]:
    items = {item.citation_id: item for item in pack.items}
    seen: set[str] = set()
    citations: list[Citation] = []
    for recommendation in answer.recommendations:
        if recommendation.citation_id in seen:
            raise LLMInvalidOutputError("生成结果包含重复引用")
        seen.add(recommendation.citation_id)
        item = items.get(recommendation.citation_id)
        if item is None:
            raise LLMInvalidOutputError("生成引用不属于本次 Top-K")
        if any(pattern.search(recommendation.reason) for pattern in FORBIDDEN_CLAIM_PATTERNS):
            raise LLMInvalidOutputError("生成结果包含数据禁区陈述")
        fields = list(dict.fromkeys(recommendation.supporting_fields))
        if fields != recommendation.supporting_fields:
            raise LLMInvalidOutputError("生成结果包含重复支持字段")
        for field in fields:
            if field not in SUPPORTING_FIELDS:
                raise LLMInvalidOutputError("生成结果引用了非白名单字段")
            value = getattr(item, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise LLMInvalidOutputError("生成结果引用了缺失字段")
        citations.append(
            Citation(
                citation_id=item.citation_id,
                product_id=item.product_id,
                supporting_fields=fields,
            )
        )
    return citations
