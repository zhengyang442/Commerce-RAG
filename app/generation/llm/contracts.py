from __future__ import annotations

from typing import Protocol

from pydantic import Field

from app.core.models import StrictModel
from app.generation.evidence import EvidencePack
from app.generation.policy import SUPPORTING_FIELDS


class GeneratedRecommendation(StrictModel):
    citation_id: str = Field(min_length=2)
    reason: str = Field(min_length=1)
    supporting_fields: list[str] = Field(min_length=1)


class GeneratedAnswer(StrictModel):
    recommendations: list[GeneratedRecommendation] = Field(max_length=5)


class LLMUsage(StrictModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMResult(StrictModel):
    answer: GeneratedAnswer
    model: str
    provider_request_id: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class LLMAdapter(Protocol):
    async def generate(self, pack: EvidencePack) -> LLMResult: ...


OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "supporting_fields": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SUPPORTING_FIELDS)},
                    },
                },
                "required": ["citation_id", "reason", "supporting_fields"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}
