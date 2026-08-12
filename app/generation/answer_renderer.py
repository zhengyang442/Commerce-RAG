from __future__ import annotations

from app.generation.llm.contracts import GeneratedAnswer


def render_rag_answer(query: str, generated: GeneratedAnswer) -> str:
    lines = [f"需求理解：{query}"]
    if not generated.recommendations:
        lines.append("当前商品证据不足以支持明确推荐。")
    else:
        lines.append("基于本次检索证据，重点候选如下：")
        lines.extend(f"[{item.citation_id}] {item.reason}" for item in generated.recommendations)
    return "\n".join(lines)
