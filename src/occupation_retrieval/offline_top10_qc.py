#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline QC report for occupation-detail Top10 retrieval.

This script is intentionally report-only. Deterministic Top10 hit/miss metrics
come from exact occupation-code matching. The optional local LLM review only
adds advisory notes for miss samples and never changes the deterministic labels.

Usage:
    python -m src.occupation_retrieval.offline_top10_qc
    python -m src.occupation_retrieval.offline_top10_qc --use-llm-review --llm-limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy import bindparam, text

from src.db.occupation_detail_matches import DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE
from src.db.postgres import create_pg_engine
from src.model_platform.config import load_model_runtime_config
from src.model_platform.llm import LLMClient, create_llm_client

from .common import load_annotations_from_pg, resolve_output_file
from .datasets import build_anchor, build_candidate_records, get_majority_choice, get_task_choices

DEFAULT_OUTPUT_FILE = "offline_top10_qc_report.md"
QC_LABELS = ("HIT@10", "MISS@10", "GOLD_NONE", "MISSING_GOLD", "MISSING_MATCH")
LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure console logging for long-running offline QC jobs."""
    numeric_level = getattr(logging, str(level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


class StepTimer:
    """Tiny context manager for logging elapsed time per step."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.started_at = 0.0

    def __enter__(self) -> "StepTimer":
        self.started_at = time.perf_counter()
        LOGGER.info("开始: %s", self.label)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        elapsed = time.perf_counter() - self.started_at
        if exc is None:
            LOGGER.info("完成: %s, elapsed=%.2fs", self.label, elapsed)
        else:
            LOGGER.exception("失败: %s, elapsed=%.2fs", self.label, elapsed)


@dataclass
class QCSample:
    """One annotation-task sample in the offline Top10 QC report."""

    task_id: int
    recruitment_record_id: str
    job_title: str
    job_requirements: str
    gold_choice: str | None
    gold_code: str = ""
    gold_title: str = ""
    anchor: str = ""
    top10_candidates: list[dict[str, Any]] = field(default_factory=list)
    hit_rank: int | None = None
    qc_label: str = "PENDING"


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def _pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def parse_top10_candidates(value: object) -> list[dict[str, Any]]:
    """Parse the `top10_candidates` JSONB payload into a stable list."""
    if value is None:
        return []
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            continue
        rank = item.get("rank", index)
        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            rank_value = index
        candidates.append(
            {
                "rank": rank_value,
                "code": _safe_text(item.get("code", "")),
                "title": _safe_text(item.get("title", "")),
                "score": item.get("score", 0),
                "detail_path": _safe_text(item.get("detail_path", "")),
                "detail_name": _safe_text(item.get("detail_name", "")),
            }
        )
    candidates.sort(key=lambda candidate: int(candidate.get("rank", 999999)))
    return candidates


def compute_hit_rank(gold_code: str | None, candidates: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the 1-based Top10 rank for an exact gold-code match."""
    normalized_gold = _safe_text(gold_code)
    if not normalized_gold:
        return None
    for index, candidate in enumerate(candidates, start=1):
        if _safe_text(candidate.get("code")) == normalized_gold:
            try:
                return int(candidate.get("rank", index))
            except (TypeError, ValueError):
                return index
    return None


def build_gold_samples(tasks: Sequence[dict[str, Any]]) -> list[QCSample]:
    """Build task-level gold samples from Label Studio annotations."""
    samples: list[QCSample] = []
    for task in tasks:
        choices = get_task_choices(task, include_none=True)
        gold_choice, _ = get_majority_choice(choices, require_strict=True)
        data = task.get("data", {})
        job_title = _safe_text(data.get("job_title", ""))
        job_requirements = _safe_text(data.get("job_requirements_clean") or data.get("job_requirements", ""))
        anchor = build_anchor(job_title, job_requirements)
        sample = QCSample(
            task_id=int(task["task_id"]),
            recruitment_record_id=_safe_text(task.get("recruitment_record_id", "")),
            job_title=job_title,
            job_requirements=job_requirements,
            gold_choice=gold_choice,
            anchor=anchor,
        )

        if gold_choice == "NONE":
            sample.qc_label = "GOLD_NONE"
        elif gold_choice in {"A", "B", "C", "D", "E"}:
            candidate_by_letter = {
                candidate["letter"]: candidate
                for candidate in build_candidate_records(data)
            }
            candidate = candidate_by_letter.get(gold_choice, {})
            sample.gold_code = _safe_text(candidate.get("code", ""))
            sample.gold_title = _safe_text(candidate.get("title", ""))
            if not sample.gold_code:
                sample.qc_label = "MISSING_GOLD"
        else:
            sample.qc_label = "MISSING_GOLD"

        samples.append(sample)
    return samples


def load_occupation_detail_matches(
    *,
    table_name: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
    recruitment_record_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load canonical occupation-detail Top10 match rows by recruitment id.

    If `recruitment_record_ids` is provided, only those rows are fetched.
    """
    ids = sorted({_safe_text(value) for value in (recruitment_record_ids or []) if _safe_text(value)})
    if ids:
        LOGGER.info("连接 PostgreSQL，按 recruitment_record_id 子集读取 Top10 匹配表: %s (n=%s)", table_name, len(ids))
    else:
        LOGGER.info("连接 PostgreSQL，全量读取 Top10 匹配表: %s", table_name)
    engine = create_pg_engine(application_name="occupation_top10_qc")
    try:
        with engine.connect() as conn:
            base_sql = f"""
                select
                    recruitment_record_id,
                    occupation_code,
                    occupation_title,
                    top10_candidates,
                    model_recipe,
                    run_id,
                    updated_at
                from {table_name}
            """
            result: dict[str, dict[str, Any]] = {}
            if ids:
                chunk_size = 1000
                for start in range(0, len(ids), chunk_size):
                    chunk = ids[start : start + chunk_size]
                    stmt = text(base_sql + " where recruitment_record_id in :ids").bindparams(
                        bindparam("ids", expanding=True)
                    )
                    rows = conn.execute(stmt, {"ids": chunk}).mappings()
                    for row in rows:
                        key = _safe_text(row["recruitment_record_id"])
                        if key:
                            result[key] = dict(row)
            else:
                rows = conn.execute(text(base_sql)).mappings()
                for row in rows:
                    key = _safe_text(row["recruitment_record_id"])
                    if key:
                        result[key] = dict(row)
            LOGGER.info("Top10 匹配表读取完成: %s rows", len(result))
            return result
    finally:
        engine.dispose()


def summarize_qc_samples(
    samples: Sequence[QCSample],
    match_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply deterministic Top10 QC labels and return aggregate counts."""
    LOGGER.info("开始确定性 Top10 命中检查: samples=%s, match_rows=%s", len(samples), len(match_rows))
    counts: Counter[str] = Counter()
    hit_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    eligible_total = 0

    for sample in samples:
        if sample.qc_label in {"GOLD_NONE", "MISSING_GOLD"}:
            counts[sample.qc_label] += 1
            continue

        eligible_total += 1
        match_row = match_rows.get(sample.recruitment_record_id)
        if not match_row:
            sample.qc_label = "MISSING_MATCH"
            counts[sample.qc_label] += 1
            continue

        sample.top10_candidates = parse_top10_candidates(match_row.get("top10_candidates"))
        sample.hit_rank = compute_hit_rank(sample.gold_code, sample.top10_candidates)
        if sample.hit_rank is None:
            sample.qc_label = "MISS@10"
        else:
            sample.qc_label = "HIT@10"
            for k in hit_at_k:
                if sample.hit_rank <= k:
                    hit_at_k[k] += 1
        counts[sample.qc_label] += 1

    result = {
        "total_samples": len(samples),
        "eligible_total": eligible_total,
        "counts": {label: counts.get(label, 0) for label in QC_LABELS},
        "hit_at_k": hit_at_k,
    }
    LOGGER.info(
        "确定性检查完成: eligible=%s HIT@10=%s MISS@10=%s GOLD_NONE=%s MISSING_MATCH=%s",
        eligible_total,
        result["counts"].get("HIT@10", 0),
        result["counts"].get("MISS@10", 0),
        result["counts"].get("GOLD_NONE", 0),
        result["counts"].get("MISSING_MATCH", 0),
    )
    return result


def _gini(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(v for v in values if v >= 0)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    n = len(sorted_values)
    weighted = sum((index + 1) * value for index, value in enumerate(sorted_values))
    return (2 * weighted) / (n * total) - (n + 1) / n


def evaluate_distribution(code_counts: Mapping[str, int]) -> dict[str, Any]:
    """Evaluate occupation-detail distribution concentration."""
    counts = Counter({code: int(count) for code, count in code_counts.items() if count > 0})
    total = sum(counts.values())
    if total == 0:
        return {
            "total": 0,
            "num_classes": 0,
            "top_code": "",
            "max_share": 0.0,
            "top10_share": 0.0,
            "singleton_share": 0.0,
            "hhi": 0.0,
            "gini": 0.0,
            "assessment": "无有效样本",
        }

    top_items = counts.most_common()
    top_code, top_count = top_items[0]
    shares = [count / total for _, count in top_items]
    top10_share = sum(count for _, count in top_items[:10]) / total
    singleton_count = sum(1 for count in counts.values() if count == 1)
    hhi = sum(share * share for share in shares)
    gini = _gini(list(counts.values()))

    if _pct(top_count, total) >= 20 or top10_share >= 0.6 or gini >= 0.6:
        assessment = "明显不均匀"
    elif _pct(top_count, total) >= 10 or top10_share >= 0.4 or gini >= 0.4:
        assessment = "中度不均匀"
    else:
        assessment = "相对均匀"

    return {
        "total": total,
        "num_classes": len(counts),
        "top_code": top_code,
        "max_share": round(_pct(top_count, total), 2),
        "top10_share": round(top10_share * 100, 2),
        "singleton_share": round(_pct(singleton_count, len(counts)), 2),
        "hhi": round(hhi, 4),
        "gini": round(gini, 4),
        "assessment": assessment,
        "top_items": top_items,
    }


def build_distribution(samples: Sequence[QCSample]) -> tuple[Counter[str], dict[str, str]]:
    """Build gold occupation-detail code counts and titles."""
    code_counts: Counter[str] = Counter()
    code_titles: dict[str, str] = {}
    for sample in samples:
        if not sample.gold_code:
            continue
        code_counts[sample.gold_code] += 1
        code_titles.setdefault(sample.gold_code, sample.gold_title)
    return code_counts, code_titles


def _candidate_lines(candidates: Sequence[Mapping[str, Any]], limit: int = 10) -> str:
    lines = []
    for candidate in candidates[:limit]:
        lines.append(
            (
                f"{candidate.get('rank')}. {candidate.get('code')} "
                f"{candidate.get('title')} score={candidate.get('score')}"
            )
        )
    return "\n".join(lines) or "无候选"


def review_miss_sample_with_llm(sample: QCSample, client: LLMClient) -> dict[str, Any]:
    """Ask the local LLM for an advisory review of one deterministic miss."""
    system_prompt = (
        "你是职业细类检索离线质检助手。请严格基于输入判断 BGE Top10 "
        "是否可能包含正确职业细类；你的结论只用于人工质检，不得改写确定性指标。"
    )
    user_prompt = f"""
请复核一条 Top10 未命中的职业细类样本，并只返回 JSON。

岗位名称：{sample.job_title}
岗位要求：{sample.job_requirements}

人工标注 gold：
- choice: {sample.gold_choice}
- code: {sample.gold_code}
- title: {sample.gold_title}

BGE Top10：
{_candidate_lines(sample.top10_candidates)}

返回 JSON 字段：
{{
  "top10_contains_correct": false,
  "best_candidate_rank": null,
  "should_be_none": false,
  "near_miss_code": "",
  "reason": "一句中文原因",
  "review_needed": true
}}
""".strip()
    try:
        LOGGER.info(
            "LLM 复核 task_id=%s recruitment_record_id=%s gold_code=%s",
            sample.task_id,
            sample.recruitment_record_id,
            sample.gold_code,
        )
        started = time.perf_counter()
        payload = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            strength="cheap",
            max_output_tokens=512,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(payload, Mapping):
            raise ValueError("LLM JSON root must be an object")
        LOGGER.info(
            "LLM 复核完成 task_id=%s status=ok elapsed=%.2fs",
            sample.task_id,
            elapsed,
        )
        return {
            "task_id": sample.task_id,
            "recruitment_record_id": sample.recruitment_record_id,
            "gold_code": sample.gold_code,
            "gold_title": sample.gold_title,
            "llm_status": "ok",
            "top10_contains_correct": bool(payload.get("top10_contains_correct", False)),
            "best_candidate_rank": payload.get("best_candidate_rank"),
            "should_be_none": bool(payload.get("should_be_none", False)),
            "near_miss_code": _safe_text(payload.get("near_miss_code", "")),
            "reason": _safe_text(payload.get("reason", "")),
            "review_needed": bool(payload.get("review_needed", True)),
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "LLM 复核失败 task_id=%s recruitment_record_id=%s error=%s",
            sample.task_id,
            sample.recruitment_record_id,
            exc,
        )
        return {
            "task_id": sample.task_id,
            "recruitment_record_id": sample.recruitment_record_id,
            "gold_code": sample.gold_code,
            "gold_title": sample.gold_title,
            "llm_status": "failed",
            "reason": str(exc),
            "review_needed": True,
        }


def review_miss_samples(
    samples: Sequence[QCSample],
    *,
    limit: int,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """Review deterministic miss samples with the configured local LLM."""
    miss_samples = [sample for sample in samples if sample.qc_label == "MISS@10"]
    if limit > 0:
        miss_samples = miss_samples[:limit]
    if not miss_samples:
        LOGGER.info("没有需要复核的 MISS@10 样本")
        return []
    LOGGER.info("准备对 %s 个 MISS@10 样本做 LLM 复核", len(miss_samples))
    client = create_llm_client(backend=backend)
    reviews: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(miss_samples, start=1):
        LOGGER.info("LLM 复核进度 %s/%s task_id=%s", index, len(miss_samples), sample.task_id)
        reviews.append(review_miss_sample_with_llm(sample, client))
    LOGGER.info("LLM 复核批次完成: count=%s elapsed=%.2fs", len(reviews), time.perf_counter() - started)
    return reviews


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_safe_text(value).replace("\n", " ") for value in row) + " |")
    return lines


def render_report(
    *,
    samples: Sequence[QCSample],
    summary: Mapping[str, Any],
    distribution: Mapping[str, Any],
    code_titles: Mapping[str, str] | None = None,
    llm_reviews: Sequence[Mapping[str, Any]] | None = None,
    llm_backend_name: str = "",
) -> str:
    """Render the offline QC report as Markdown."""
    code_titles = code_titles or {}
    llm_reviews = llm_reviews or []
    counts = summary.get("counts", {})
    eligible_total = int(summary.get("eligible_total", 0))
    hit_at_k = summary.get("hit_at_k", {})

    lines = [
        "# Occupation Retrieval Offline Top10 QC Report",
        "",
        f"- Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Gold source: `annotations.label_studio_tasks_v2` majority choice mapped to candidate A-E code",
        f"- Match source: `{DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE}.top10_candidates`",
        "- LLM policy: advisory review for deterministic miss samples only; it does not change hit/miss metrics.",
        "",
        "## 确定性 Top10 命中统计",
        "",
    ]
    metric_rows = []
    for k in (1, 3, 5, 10):
        hits = int(hit_at_k.get(k, 0))
        metric_rows.append((f"Hit@{k}", hits, eligible_total, f"{_pct(hits, eligible_total):.2f}%"))
    metric_rows.extend(
        [
            ("HIT@10", counts.get("HIT@10", 0), len(samples), f"{_pct(counts.get('HIT@10', 0), len(samples)):.2f}%"),
            ("MISS@10", counts.get("MISS@10", 0), len(samples), f"{_pct(counts.get('MISS@10', 0), len(samples)):.2f}%"),
            ("GOLD_NONE", counts.get("GOLD_NONE", 0), len(samples), f"{_pct(counts.get('GOLD_NONE', 0), len(samples)):.2f}%"),
            ("MISSING_GOLD", counts.get("MISSING_GOLD", 0), len(samples), f"{_pct(counts.get('MISSING_GOLD', 0), len(samples)):.2f}%"),
            ("MISSING_MATCH", counts.get("MISSING_MATCH", 0), len(samples), f"{_pct(counts.get('MISSING_MATCH', 0), len(samples)):.2f}%"),
        ]
    )
    lines.extend(_markdown_table(["Metric", "Count", "Denominator", "Rate"], metric_rows))

    lines.extend(
        [
            "",
            "## 职业细类分布统计",
            "",
            f"- 有效 gold 样本数: {distribution.get('total', 0)}",
            f"- 职业细类数量: {distribution.get('num_classes', 0)}",
            f"- 最大单类占比: {distribution.get('max_share', 0.0):.2f}%",
            f"- Top10 类别占比: {distribution.get('top10_share', 0.0):.2f}%",
            f"- 单样本类别占比: {distribution.get('singleton_share', 0.0):.2f}%",
            f"- HHI: {distribution.get('hhi', 0.0)}",
            f"- Gini: {distribution.get('gini', 0.0)}",
            f"- 均匀性评估: **{distribution.get('assessment', '')}**",
            "",
        ]
    )
    top_rows = []
    for code, count in list(distribution.get("top_items", []))[:30]:
        top_rows.append((code, code_titles.get(code, ""), count, f"{_pct(count, distribution.get('total', 0)):.2f}%"))
    if top_rows:
        lines.extend(_markdown_table(["Code", "Title", "Count", "Share"], top_rows))
    else:
        lines.append("无有效 gold 职业细类。")

    miss_examples = [sample for sample in samples if sample.qc_label == "MISS@10"][:20]
    lines.extend(["", "## Top10 Miss 样本摘录", ""])
    if miss_examples:
        rows = [
            (
                sample.task_id,
                sample.recruitment_record_id,
                sample.job_title,
                sample.gold_code,
                sample.gold_title,
                _candidate_lines(sample.top10_candidates, limit=3).replace("\n", "<br>"),
            )
            for sample in miss_examples
        ]
        lines.extend(_markdown_table(["Task", "Recruitment", "Job Title", "Gold Code", "Gold Title", "Top3 Candidates"], rows))
    else:
        lines.append("没有确定性 MISS@10 样本。")

    lines.extend(["", "## 本地 LLM Miss 样本复核", ""])
    if llm_backend_name:
        lines.append(f"- LLM backend: `{llm_backend_name}`")
    if llm_reviews:
        rows = []
        for review in llm_reviews:
            rows.append(
                (
                    review.get("task_id", ""),
                    review.get("gold_code", ""),
                    review.get("llm_status", ""),
                    review.get("top10_contains_correct", ""),
                    review.get("should_be_none", ""),
                    review.get("near_miss_code", ""),
                    review.get("review_needed", ""),
                    review.get("reason", ""),
                )
            )
        lines.extend(
            _markdown_table(
                [
                    "Task",
                    "Gold Code",
                    "Status",
                    "LLM Says Contains",
                    "Should Be NONE",
                    "Near Miss",
                    "Review Needed",
                    "Reason",
                ],
                rows,
            )
        )
    else:
        lines.append("未运行 LLM 复核；如需复核确定性 MISS@10 样本，请加 `--use-llm-review`。")

    lines.extend(
        [
            "",
            "## 运行逻辑说明",
            "",
            "1. 对每条 Label Studio 任务取严格多数意见；若多数为 `NONE`，计入 `GOLD_NONE`。",
            "2. 对 A-E 多数意见，从任务候选字段取 gold occupation code。",
            "3. 通过 `recruitment_record_id` 连接 `public.occupation_detail_matches`。",
            "4. 若 gold code 精确出现在 `top10_candidates[*].code`，计为 `HIT@10`，否则计为 `MISS@10`。",
            "5. LLM 仅复核 `MISS@10` 样本并给出解释，主指标始终以第 4 步为准。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline Top10 QC Markdown report.")
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output Markdown file path or filename under output/occupation_retrieval.",
    )
    parser.add_argument(
        "--match-table",
        default=DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
        help="Occupation detail match table containing top10_candidates.",
    )
    parser.add_argument(
        "--use-llm-review",
        action="store_true",
        help="Review deterministic MISS@10 samples with the configured local LLM.",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=20,
        help="Maximum MISS@10 samples to review with LLM. 0 means all misses.",
    )
    parser.add_argument(
        "--llm-backend",
        default=None,
        help="Override LLM backend name. Defaults to config/model_runtime.yaml.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Console log level, e.g. DEBUG, INFO, WARNING.",
    )
    return parser.parse_args()


def _resolve_output_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return resolve_output_file(raw)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run_started = time.perf_counter()
    LOGGER.info("Offline Top10 QC started")
    LOGGER.info("输出文件: %s", args.output_file)
    LOGGER.info("匹配表: %s", args.match_table)
    LOGGER.info("LLM 复核: %s", "enabled" if args.use_llm_review else "disabled")

    with StepTimer("读取标注任务"):
        tasks = load_annotations_from_pg()
        LOGGER.info("任务数: %s", len(tasks))

    with StepTimer("构造 gold 样本"):
        samples = build_gold_samples(tasks)
        LOGGER.info("样本数: %s", len(samples))

    with StepTimer("确定性命中统计"):
        needed_recruitment_ids = [
            sample.recruitment_record_id
            for sample in samples
            if sample.recruitment_record_id and sample.gold_code
        ]
        LOGGER.info("本次需要查询的 recruitment_record_id 数: %s", len(needed_recruitment_ids))
        with StepTimer("读取 Top10 匹配结果"):
            matches = load_occupation_detail_matches(
                table_name=args.match_table,
                recruitment_record_ids=needed_recruitment_ids,
            )
        summary = summarize_qc_samples(samples, matches)

    with StepTimer("分布统计"):
        code_counts, code_titles = build_distribution(samples)
        distribution = evaluate_distribution(code_counts)
        LOGGER.info(
            "分布统计完成: total=%s classes=%s assessment=%s",
            distribution.get("total", 0),
            distribution.get("num_classes", 0),
            distribution.get("assessment", ""),
        )

    llm_reviews: list[dict[str, Any]] = []
    runtime = load_model_runtime_config()
    llm_backend_name = args.llm_backend or runtime.default_llm_backend
    if args.use_llm_review:
        with StepTimer("LLM miss 复核"):
            llm_reviews = review_miss_samples(
                samples,
                limit=max(0, int(args.llm_limit)),
                backend=args.llm_backend,
            )

    with StepTimer("渲染 Markdown 报告"):
        report = render_report(
            samples=samples,
            summary=summary,
            distribution=distribution,
            code_titles=code_titles,
            llm_reviews=llm_reviews,
            llm_backend_name=llm_backend_name if args.use_llm_review else "",
        )

    with StepTimer("写入报告文件"):
        output_path = _resolve_output_path(args.output_file)
        output_path.write_text(report, encoding="utf-8")

    LOGGER.info("Report saved to: %s", output_path)
    if args.use_llm_review:
        LOGGER.info("LLM backend: %s", llm_backend_name)
    else:
        LOGGER.info("LLM review skipped. Add --use-llm-review to review MISS@10 samples.")
    LOGGER.info(
        "Deterministic QC: HIT@10=%s MISS@10=%s GOLD_NONE=%s elapsed=%.2fs",
        summary["counts"].get("HIT@10", 0),
        summary["counts"].get("MISS@10", 0),
        summary["counts"].get("GOLD_NONE", 0),
        time.perf_counter() - run_started,
    )


if __name__ == "__main__":
    main()
