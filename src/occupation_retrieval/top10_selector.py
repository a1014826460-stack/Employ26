"""Shared Top10 second-stage selector for occupation detail retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.model_platform.llm import LLMClient, create_llm_client


@dataclass(frozen=True)
class Top10SelectionResult:
    """Structured final winner selected from a Top10 candidate list."""

    selected_rank: int | None
    selected_code: str
    selected_title: str
    reason: str
    confidence: float | None = None
    needs_review: bool = False
    selector_backend: str = "llm_over_top10"


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_lines(candidates: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            (
                f"{candidate.get('rank')}. {candidate.get('title')} "
                f"({candidate.get('code')}), score={candidate.get('score', 0)}"
            )
        )
    return "\n".join(lines) or "无候选"


def build_selection_prompts(
    *,
    job_title: str,
    job_description: str,
    top10_candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Build deterministic prompts for structured TopK winner selection."""
    candidate_count = len(top10_candidates)
    system_prompt = (
        f"你是职业细类选择器。你只能从给定 Top{candidate_count} 候选中选一个最合适的职业细类，"
        "并且必须输出 JSON。不要输出 markdown，不要输出额外解释。"
    )
    user_prompt = f"""
请从下面给定的职业细类候选中，选出最符合岗位的一个，并返回 JSON。

岗位名称：{job_title}
岗位描述：{job_description or "未提供"}

候选列表：
{_candidate_lines(top10_candidates)}

返回 JSON：
{{
  "selected_rank": 1,
  "selected_code": "职业代码",
  "selected_title": "职业名称",
  "reason": "一句中文理由",
  "confidence": 0.85,
  "needs_review": false
}}
""".strip()
    return system_prompt, user_prompt


def normalize_selector_payload(
    payload: Mapping[str, Any],
    *,
    top10_candidates: Sequence[Mapping[str, Any]],
) -> Top10SelectionResult:
    """Normalize raw selector JSON into a stable structured result."""
    candidates_by_rank = {}
    for candidate in top10_candidates:
        try:
            rank = int(candidate.get("rank"))
        except (TypeError, ValueError):
            continue
        candidates_by_rank[rank] = candidate

    try:
        selected_rank = int(payload.get("selected_rank")) if payload.get("selected_rank") is not None else None
    except (TypeError, ValueError):
        selected_rank = None

    selected_code = _safe_text(payload.get("selected_code", ""))
    selected_title = _safe_text(payload.get("selected_title", ""))

    if selected_rank is not None and selected_rank in candidates_by_rank:
        candidate = candidates_by_rank[selected_rank]
        if not selected_code:
            selected_code = _safe_text(candidate.get("code", ""))
        if not selected_title:
            selected_title = _safe_text(candidate.get("title", ""))

    if selected_rank not in candidates_by_rank or not selected_code:
        selected_rank = None
        selected_code = ""
        selected_title = ""

    return Top10SelectionResult(
        selected_rank=selected_rank,
        selected_code=selected_code,
        selected_title=selected_title,
        reason=_safe_text(payload.get("reason", "")),
        confidence=_safe_float(payload.get("confidence")),
        needs_review=bool(payload.get("needs_review", False)) or not bool(selected_code),
    )


def select_from_top10(
    *,
    job_title: str,
    job_description: str,
    top10_candidates: Sequence[Mapping[str, Any]],
    client: LLMClient | None = None,
    backend: str | None = None,
) -> Top10SelectionResult:
    """Use the project LLM runtime to select one final winner from TopK candidates."""
    runtime_client = client or create_llm_client(backend=backend)
    system_prompt, user_prompt = build_selection_prompts(
        job_title=job_title,
        job_description=job_description,
        top10_candidates=top10_candidates,
    )
    payload = runtime_client.complete_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        strength="cheap",
        max_output_tokens=256,
        temperature=0.0,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Top10 selector must return a JSON object")
    return normalize_selector_payload(payload, top10_candidates=top10_candidates)


def selection_to_dict(result: Top10SelectionResult) -> dict[str, Any]:
    """Expose a stable dict representation for reports and APIs."""
    return asdict(result)
