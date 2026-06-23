"""Occupation detail Top10 retrieval plus local LLM analysis service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.skill_extraction.config import SkillExtractionConfig, load_skill_extraction_config
from src.occupation_retrieval.top10_selector import selection_to_dict, select_from_top10
from src.utils.vllm_utils import (
    VLLMConfig,
    chat_completion,
    check_server,
    create_http_session,
    extract_message_parts,
    load_vllm_config,
)


DEFAULT_MODEL_RECIPE = "v1-bge-large"
DEFAULT_MODEL_PATH = "output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned"


@dataclass(frozen=True)
class OccupationAnalysisRequest:
    """Input contract for one occupation detail analysis request."""

    job_title: str
    job_description: str = ""
    top_k: int = 10
    include_llm_report: bool = True


def _safe_text(value: object) -> str:
    """Convert nullable values to stripped text."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _score_value(value: object) -> float:
    """Convert score-like values to float with a safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_top_candidates(matched_row: pd.Series, top_k: int = 10) -> list[dict[str, Any]]:
    """Extract ranked TopK candidates from the matcher output row."""
    candidates: list[dict[str, Any]] = []
    for rank in range(1, int(top_k) + 1):
        prefix = f"top{rank}"
        code = _safe_text(matched_row.get(f"{prefix}_code", ""))
        title = _safe_text(matched_row.get(f"{prefix}_title", ""))
        if not code and not title:
            continue
        candidates.append(
            {
                "rank": rank,
                "code": code,
                "title": title,
                "score": _score_value(matched_row.get(f"{prefix}_score", 0.0)),
                "detail_path": _safe_text(matched_row.get(f"{prefix}_detail_path", "")),
                "detail_name": _safe_text(matched_row.get(f"{prefix}_detail_name", "")),
            }
        )
    return candidates


def build_llm_messages(
    *,
    request: OccupationAnalysisRequest,
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build deterministic chat messages for occupation detail report generation."""
    candidate_lines = []
    for candidate in candidates:
        candidate_lines.append(
            (
                f"{candidate['rank']}. {candidate['title']} "
                f"({candidate['code']}), score={candidate['score']:.4f}, "
                f"path={candidate['detail_path']}"
            )
        )
    candidates_text = "\n".join(candidate_lines) or "无候选"
    user_prompt = f"""
请基于 BGE 检索返回的 Top10 职业细类候选，给出职业细类识别分析报告。

岗位名称：
{request.job_title}

岗位描述：
{request.job_description or "未提供"}

Top10 候选：
{candidates_text}

请严格按以下结构输出：
1. 最终建议：给出最推荐的职业细类 code 和 title。
2. 判断依据：结合岗位名称、岗位描述和候选层级解释为什么选择它。
3. 候选比较：说明 Top10 中最容易混淆的 2-3 个候选，以及它们为什么不如最终建议。
4. 风险提示：指出岗位描述不足、跨职业边界或需要人工复核的情况。
5. Agent 使用建议：说明是否可以直接采用 Top1，还是建议进入二阶段人工/LLM 复核。
""".strip()
    return [
        {
            "role": "system",
            "content": (
                "你是职业细类识别专家。你需要帮助招聘数据 Agent "
                "复核 BGE Top10 候选，并输出清晰、可审计的中文报告。"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


class OccupationAgentService:
    """Reusable service for API and future MCP wrappers."""

    def __init__(
        self,
        *,
        skill_config: SkillExtractionConfig | None = None,
        vllm_config: VLLMConfig | None = None,
        model_path: str | Path | None = None,
        force_rebuild_index: bool = False,
    ) -> None:
        base_config = skill_config or load_skill_extraction_config()
        resolved_model_path = Path(model_path) if model_path else base_config.occupation_detail_model_path
        cache_path = self._build_catalog_cache_path(base_config.cache_dir, resolved_model_path)
        self.skill_config = replace(
            base_config,
            embedding_model_path=resolved_model_path,
            catalog_embedding_cache_path=cache_path,
            match_top_k=max(10, int(base_config.occupation_detail_top_k)),
        )
        self.vllm_config = vllm_config or load_vllm_config()
        try:
            from src.skill_extraction.bge_matcher import OccupationBGEMatcher
        except ModuleNotFoundError as exc:
            if exc.name == "sentence_transformers":
                raise RuntimeError(
                    "当前 Python 环境缺少 sentence_transformers，无法加载 BGE 检索模型。\n"
                    "请优先使用项目内解释器启动：\n"
                    "  .\\.conda\\python.exe -m src.llm.occupation_agent_api --host 127.0.0.1 --port 8120\n"
                    "如果必须使用当前解释器，请先安装依赖：\n"
                    "  python -m pip install sentence-transformers"
                ) from exc
            raise
        self.matcher = OccupationBGEMatcher(self.skill_config)
        self.matcher.build_index(force_rebuild=force_rebuild_index)

    @staticmethod
    def _build_catalog_cache_path(cache_dir: Path, model_path: str | Path) -> Path:
        model_name = Path(str(model_path)).name or "occupation_detail_model"
        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in model_name)
        return cache_dir / f"occupation_agent_catalog_embeddings_{safe_name}.npy"

    def analyze(self, request: OccupationAnalysisRequest) -> dict[str, Any]:
        """Return BGE Top10 candidates and optional local LLM report."""
        if not _safe_text(request.job_title) and not _safe_text(request.job_description):
            raise ValueError("job_title 和 job_description 不能同时为空")

        top_k = max(1, min(int(request.top_k or 10), 50))
        jobs_df = pd.DataFrame(
            [
                {
                    "job_id": "ad-hoc",
                    "岗位名称": request.job_title,
                    "岗位描述": request.job_description,
                    "职业匹配来源": "occupation_agent_api",
                }
            ]
        )
        matched_df = self.matcher.match_jobs(jobs_df, top_k=top_k)
        if matched_df.empty:
            raise RuntimeError("BGE 职业细类检索没有返回结果")

        matched_row = matched_df.iloc[0]
        candidates = extract_top_candidates(matched_row, top_k=top_k)
        top1 = candidates[0] if candidates else None
        llm_report = ""
        llm_finish_reason = None
        llm_selected = None
        final_selection = top1

        if request.include_llm_report:
            llm_selected_result = select_from_top10(
                job_title=request.job_title,
                job_description=request.job_description,
                top10_candidates=candidates,
            )
            llm_selected = selection_to_dict(llm_selected_result)
            if llm_selected_result.selected_code:
                final_selection = {
                    "rank": llm_selected_result.selected_rank,
                    "code": llm_selected_result.selected_code,
                    "title": llm_selected_result.selected_title,
                    "selection_source": "llm_over_top10",
                }
            messages = build_llm_messages(request=request, candidates=candidates)
            session = create_http_session()
            check_server(self.vllm_config, session=session)
            response_data = chat_completion(
                config=self.vllm_config,
                messages=messages,
                session=session,
            )
            llm_report, _reasoning, llm_finish_reason = extract_message_parts(response_data)

        return {
            "model": DEFAULT_MODEL_RECIPE,
            "embedding_model_path": str(self.skill_config.embedding_model_path),
            "top_k": top_k,
            "query": {
                "job_title": request.job_title,
                "job_description": request.job_description,
            },
            "top1": top1,
            "top10_candidates": candidates,
            "llm_selected": llm_selected,
            "final_selection": final_selection,
            "llm_report": llm_report,
            "llm_finish_reason": llm_finish_reason,
        }


def parse_analysis_request(payload: dict[str, Any]) -> OccupationAnalysisRequest:
    """Parse and validate a JSON payload into an analysis request."""
    return OccupationAnalysisRequest(
        job_title=_safe_text(payload.get("job_title", "")),
        job_description=_safe_text(payload.get("job_description", "")),
        top_k=int(payload.get("top_k", 10) or 10),
        include_llm_report=bool(payload.get("include_llm_report", True)),
    )


def dumps_response(payload: dict[str, Any]) -> str:
    """Serialize API responses using project-friendly JSON defaults."""
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def request_to_dict(request: OccupationAnalysisRequest) -> dict[str, Any]:
    """Expose dataclass conversion for tests and thin wrappers."""
    return asdict(request)

