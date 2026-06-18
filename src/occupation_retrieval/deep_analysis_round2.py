#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二轮数据集深度分析：per-task majority 和 RAG TopK 命中率指标。

分析维度:
    1. 每任务标注员多数意见（majority vote）
    2. RAG 候选 TopK 命中率（Top-1 至 Top-5）
    3. 多数意见为 "NONE"（以上都不对）的任务比例

用法:
    python -m src.occupation_retrieval.deep_analysis_round2
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from .common import get_occupation_retrieval_output_dir, load_annotations_from_pg
from .datasets import get_majority_choice, parse_choice

TaskPayload = dict[str, Any]
TaskMap = dict[int, TaskPayload]
OUTPUT_FILE = Path(get_occupation_retrieval_output_dir()) / "deep_analysis_round2.md"


def load_task_annotations(raw_data: list[dict[str, Any]]) -> TaskMap:
    """将 PostgreSQL 读取出的任务列表转换为按任务聚合的结构。"""
    task_annotations: TaskMap = {}
    for item in raw_data:
        task_id = item["task_id"]
        ann_list: list[Optional[str]] = []
        for ann in item["annotations"]:
            ann_list.append(parse_choice(ann))

        task_annotations[task_id] = {
            "annotations": ann_list,
            "data": item["data"],
        }

    return task_annotations


def get_majority(task: TaskPayload) -> Optional[str]:
    """返回单个任务的多数意见。

    Args:
        task: 单个任务的聚合结果，需包含 `annotations` 字段。

    Returns:
        多数选择对应的字母，若没有有效选择则返回 `None`。
    """
    choices = [choice for choice in task["annotations"] if choice is not None]
    if not choices:
        return None

    majority, _ = get_majority_choice(choices, require_strict=False)
    return majority


def find_choice_topk(data: dict[str, Any], choice: str) -> Optional[int]:
    """查找指定选择对应的候选在 RAG 中的排名。

    Args:
        data: 任务原始数据字段。
        choice: 人类最终选择，取值为 `A-E`。

    Returns:
        候选对应的 TopK 排名；若未找到则返回 `None`。
    """
    for candidate in ["a", "b", "c", "d", "e"]:
        if choice != candidate.upper():
            continue
        source = str(data.get(f"candidate_{candidate}_source", ""))
        for topk in [1, 2, 3, 4, 5]:
            if f"top{topk}" in source:
                return topk
        return None
    return None


def build_cumulative_topk_stats_lines(
    title: str,
    topk_hits: dict[int, int],
    valid_total: int,
    none_total: int,
    valid_label: str,
    none_label: str,
) -> list[str]:
    """构造累计 TopK 命中率摘要文本。"""
    lines = [
        f"=== {title} ===",
        f"{valid_label}: {valid_total}",
        f"{none_label}: {none_total}",
    ]

    cumulative_hits = 0
    for topk in sorted(topk_hits.keys()):
        cumulative_hits += topk_hits[topk]
        pct = cumulative_hits / valid_total * 100 if valid_total > 0 else 0
        lines.append(
            f"  Top-{topk}: {topk_hits[topk]} -> cum Top{topk}="
            f"{cumulative_hits}/{valid_total} = {pct:.1f}%"
        )
    return lines


def main() -> None:
    """执行第二轮标注数据的 TopK 与多数意见分析。"""
    task_annotations = load_task_annotations(load_annotations_from_pg())
    output_lines: list[str] = []

    # 任务级 TopK 命中统计。
    topk_hits: defaultdict[int, int] = defaultdict(int)
    tasks_with_majority = 0
    tasks_with_none_maj = 0

    for task in task_annotations.values():
        majority = get_majority(task)
        if majority is None:
            continue
        if majority == "NONE":
            tasks_with_none_maj += 1
            continue

        tasks_with_majority += 1
        topk = find_choice_topk(task["data"], majority)
        if topk is not None:
            topk_hits[topk] += 1

    output_lines.extend(
        build_cumulative_topk_stats_lines(
        title="Per-Task (Majority Vote) RAG TopK Hit Rate",
        topk_hits=topk_hits,
        valid_total=tasks_with_majority,
        none_total=tasks_with_none_maj,
        valid_label="Tasks with majority (non-NONE)",
        none_label="Tasks with majority=NONE",
        )
    )

    # 单标注任务分析。
    single_tasks = {
        task_id: task
        for task_id, task in task_annotations.items()
        if len(task["annotations"]) == 1
    }
    single_topk: defaultdict[int, int] = defaultdict(int)
    single_valid = 0
    single_none = 0
    for task in single_tasks.values():
        choice = task["annotations"][0]
        if choice == "NONE":
            single_none += 1
            continue
        if choice is None:
            continue

        single_valid += 1
        topk = find_choice_topk(task["data"], choice)
        if topk is not None:
            single_topk[topk] += 1

    output_lines.append("")
    output_lines.extend(
        build_cumulative_topk_stats_lines(
        title="Single-Annotator Tasks RAG TopK",
        topk_hits=single_topk,
        valid_total=single_valid,
        none_total=single_none,
        valid_label="Valid choices",
        none_label="NONE",
        )
    )

    # 多标注任务按多数意见统计。
    multi_tasks = {
        task_id: task
        for task_id, task in task_annotations.items()
        if len(task["annotations"]) >= 2
    }
    multi_topk: defaultdict[int, int] = defaultdict(int)
    multi_valid = 0
    multi_none = 0
    for task in multi_tasks.values():
        majority = get_majority(task)
        if majority == "NONE":
            multi_none += 1
            continue
        if majority is None:
            continue

        multi_valid += 1
        topk = find_choice_topk(task["data"], majority)
        if topk is not None:
            multi_topk[topk] += 1

    output_lines.append("")
    output_lines.extend(
        build_cumulative_topk_stats_lines(
        title="Multi-Annotator Tasks (Majority) RAG TopK",
        topk_hits=multi_topk,
        valid_total=multi_valid,
        none_total=multi_none,
        valid_label="Valid majorities",
        none_label="NONE majorities",
        )
    )

    # 标注级 TopK 命中率。
    per_ann_topk: defaultdict[int, int] = defaultdict(int)
    per_ann_total = 0
    per_ann_none = 0
    for task in task_annotations.values():
        for annotation_choice in task["annotations"]:
            if annotation_choice is None:
                continue
            if annotation_choice == "NONE":
                per_ann_none += 1
                continue

            per_ann_total += 1
            topk = find_choice_topk(task["data"], annotation_choice)
            if topk is not None:
                per_ann_topk[topk] += 1

    output_lines.append("")
    output_lines.extend(
        build_cumulative_topk_stats_lines(
        title="Per-Annotation RAG TopK Hit Rate",
        topk_hits=per_ann_topk,
        valid_total=per_ann_total,
        none_total=per_ann_none,
        valid_label="Total valid annotations",
        none_label="NONE annotations",
        )
    )

    # 任务概览统计。
    total_tasks = len(task_annotations)
    output_lines.append("")
    output_lines.append("=== Task-Level Summary ===")
    output_lines.append(f"Total tasks: {total_tasks}")
    output_lines.append(f"  Multi-annotator: {len(multi_tasks)}")
    output_lines.append(f"  Single-annotator: {len(single_tasks)}")
    output_lines.append(
        f"Tasks with majority non-NONE: {tasks_with_majority} "
        f"({tasks_with_majority / total_tasks * 100:.1f}%)"
    )
    output_lines.append(
        f"Tasks with majority NONE: {tasks_with_none_maj} "
        f"({tasks_with_none_maj / total_tasks * 100:.1f}%)"
    )

    # 计算多标注任务中，多数意见被跟随的平均比例。
    agree_rates: list[float] = []
    for task in multi_tasks.values():
        choices = [choice for choice in task["annotations"] if choice is not None]
        if len(choices) < 2:
            continue
        counter = Counter(choices)
        _, top_count = counter.most_common(1)[0]
        agree_rates.append(top_count / len(choices))

    avg_agree = sum(agree_rates) / len(agree_rates) if agree_rates else 0
    output_lines.append("")
    output_lines.append(
        f"Avg majority agreement rate (multi-ann tasks): {avg_agree * 100:.1f}%"
    )

    # 枚举多个候选指标，帮助定位与 82.8% 最接近的解释口径。
    output_lines.append("")
    output_lines.append("=== CANDIDATES FOR 82.8% ===")
    candidates: dict[str, float] = {}

    candidates["A: Task-majority in RAG Top3"] = (
        sum(topk_hits[k] for k in [1, 2, 3]) / tasks_with_majority * 100
        if tasks_with_majority > 0
        else 0
    )
    candidates["B: Per-annotation in RAG Top3"] = (
        sum(per_ann_topk[k] for k in [1, 2, 3]) / per_ann_total * 100
        if per_ann_total > 0
        else 0
    )
    candidates["C: Single-ann tasks in RAG Top3"] = (
        sum(single_topk[k] for k in [1, 2, 3]) / single_valid * 100
        if single_valid > 0
        else 0
    )
    if multi_valid > 0:
        candidates["D: Multi-ann tasks majority in RAG Top3"] = (
            sum(multi_topk[k] for k in [1, 2, 3]) / multi_valid * 100
        )
    candidates["E: Avg majority agreement rate"] = avg_agree * 100
    candidates["F: Single-ann chose A-E (valid)"] = (
        single_valid / (single_valid + single_none) * 100
        if (single_valid + single_none) > 0
        else 0
    )
    total_annotations = per_ann_total + per_ann_none
    candidates["G: All annotations chose A-E"] = (
        per_ann_total / total_annotations * 100 if total_annotations > 0 else 0
    )

    for name, value in sorted(candidates.items(), key=lambda item: abs(item[1] - 82.8)):
        diff = abs(value - 82.8)
        marker = " *** CLOSEST ***" if diff < 1.0 else (" **" if diff < 5.0 else "")
        output_lines.append(f"  {name}: {value:.1f}% (diff={diff:.1f}pp){marker}")

    report = "\n".join(output_lines)
    print(report)
    OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

