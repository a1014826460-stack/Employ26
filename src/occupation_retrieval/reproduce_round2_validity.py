#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复现论文：第二轮数据集有效性检测。

对第二轮人工标注数据集进行有效性分析:
    1. 标注员间一致性（Inter-Annotator Agreement, Fleiss' Kappa）
    2. RAG 候选命中率（Top-1 ~ Top-5）
    3. DeepSeek-V4-Pro 标注准确率对比（与人类多数意见）
    4. 验证样本（is_validation_sample=1）专项分析
    5. 分歧样本特征统计

用法:
    python -m src.occupation_retrieval.reproduce_round2_validity
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from config.paths import get_project_paths
from .common import get_occupation_retrieval_output_dir, load_annotations_from_pg, load_deepseek_records

_project = get_project_paths()
BASE = str(_project.project_root)
OUTPUT_FILE = os.path.join(get_occupation_retrieval_output_dir(), "round2_validity_report.txt")


def extract_choice(annotation: dict[str, Any]) -> str | None:
    """从标注记录中提取最佳候选选择。

    Args:
        annotation: 单条 Label Studio annotation 记录。

    Returns:
        标准化后的 `A-E`、`NONE` 或 `None`。
    """
    for result in annotation.get("result", []):
        if result["from_name"] != "best_candidate_choice":
            continue
        choices = result["value"].get("choices", [])
        if not choices:
            return None

        raw_choice = choices[0]
        if len(raw_choice) >= 2 and raw_choice[-1] in "ABCDE":
            return raw_choice[-1]
        if "不" in raw_choice:
            return "NONE"
        return raw_choice
    return None


def make_summary(text: str) -> str:
    """给报告正文加上时间戳和统一抬头。"""
    lines = [
        "=" * 70,
        "Round 2 Dataset Validity Reproduction Analysis",
        f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        text,
    ]
    return "\n".join(lines)


def main() -> None:
    """执行第二轮数据集有效性复现实验。"""
    print("Loading data...")
    raw_data = load_annotations_from_pg()

    output_lines: list[str] = []

    def log(text: str) -> None:
        """同时打印并缓存报告文本。"""
        print(text)
        output_lines.append(text)

    log("=" * 70)
    log("Round 2 Dataset Validity Reproduction Analysis")
    log(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    log("\n[Dataset Overview]")
    log(f"  Total tasks: {len(raw_data)}")
    total_annotations = sum(len(item["annotations"]) for item in raw_data)
    log(f"  Total annotations: {total_annotations}")

    # 任务级聚合结果：每个任务保存标注员选择及原始 data 字段。
    task_annotations: dict[int, dict[str, Any]] = {}
    validation_tasks: list[int] = []
    for item in raw_data:
        task_id = item["task_id"]
        data_fields = item["data"]
        ann_list: list[dict[str, Any]] = []
        for ann in item["annotations"]:
            ann_list.append({
                "annotator": ann["completed_by"],
                "choice": extract_choice(ann),
            })
        task_annotations[task_id] = {
            "annotations": ann_list,
            "data": data_fields,
        }
        if data_fields.get("is_validation_sample") == "1":
            validation_tasks.append(task_id)

    multi_tasks = {
        tid: task
        for tid, task in task_annotations.items()
        if len(task["annotations"]) >= 2
    }
    single_tasks = {
        tid: task
        for tid, task in task_annotations.items()
        if len(task["annotations"]) == 1
    }

    log(f"  Tasks with annotations: {len(task_annotations)}")
    log(f"  Multi-annotator tasks: {len(multi_tasks)}")
    log(f"  Single-annotator tasks: {len(single_tasks)}")
    log(f"  Validation-sample tasks (is_validation_sample=1): {len(validation_tasks)}")

    all_choices: Counter[str | None] = Counter()
    for task in task_annotations.values():
        for annotation in task["annotations"]:
            all_choices[annotation["choice"]] += 1

    log("\n[Global Choice Distribution]")
    total = sum(all_choices.values())
    for choice in sorted(all_choices.keys(), key=lambda item: all_choices[item], reverse=True):
        log(f"  {choice}: {all_choices[choice]} ({all_choices[choice] / total * 100:.1f}%)")

    none_count = all_choices.get("NONE", 0)
    valid_count = total - none_count
    log(f"  Valid choices (A-E): {valid_count} ({valid_count / total * 100:.1f}%)")
    log(f"  NONE (以上选项都不属于): {none_count} ({none_count / total * 100:.1f}%)")

    single_choices: Counter[str | None] = Counter()
    for task in single_tasks.values():
        single_choices[task["annotations"][0]["choice"]] += 1
    log(f"\n[Single-Annotator Tasks Choice Distribution] ({len(single_tasks)} tasks)")
    single_total = sum(single_choices.values())
    for choice in sorted(single_choices.keys(), key=lambda item: single_choices[item], reverse=True):
        log(
            f"  {choice}: {single_choices[choice]} "
            f"({single_choices[choice] / single_total * 100:.1f}%)"
        )
    single_valid = single_total - single_choices.get("NONE", 0)
    log(f"  Chose A-E: {single_valid} ({single_valid / single_total * 100:.1f}%)")
    log(
        "  Chose NONE: "
        f"{single_choices.get('NONE', 0)} "
        f"({single_choices.get('NONE', 0) / single_total * 100:.1f}%)"
    )

    log(f"\n{'=' * 70}")
    log("[Analysis 1: Inter-Annotator Agreement]")
    log(f"{'=' * 70}")

    majority_votes: dict[int, str | None]
    full_agree = 0
    avg_pairwise = 0.0
    majority_exists = 0

    if len(multi_tasks) > 0:
        for task in multi_tasks.values():
            choices = [annotation["choice"] for annotation in task["annotations"]]
            if len(set(choices)) == 1:
                full_agree += 1
        log(f"\n  Full agreement: {full_agree}/{len(multi_tasks)} = {full_agree / len(multi_tasks) * 100:.1f}%")

        pairwise_rates: list[float] = []
        for task in multi_tasks.values():
            choices = [annotation["choice"] for annotation in task["annotations"]]
            n_choices = len(choices)
            if n_choices < 2:
                continue
            agrees = sum(
                1
                for i in range(n_choices)
                for j in range(i + 1, n_choices)
                if choices[i] == choices[j]
            )
            pairwise_rates.append(agrees / (n_choices * (n_choices - 1) // 2))
        avg_pairwise = sum(pairwise_rates) / len(pairwise_rates) if pairwise_rates else 0
        log(f"  Avg pairwise agreement: {avg_pairwise * 100:.1f}%")

        majority_votes = {}
        for tid, task in multi_tasks.items():
            choices = [annotation["choice"] for annotation in task["annotations"]]
            counter = Counter(choices)
            top_choice, top_count = counter.most_common(1)[0]
            if top_count > len(choices) / 2:
                majority_exists += 1
                majority_votes[tid] = top_choice
            else:
                majority_votes[tid] = None
        log(
            f"  Majority exists (>50%): {majority_exists}/{len(multi_tasks)} = "
            f"{majority_exists / len(multi_tasks) * 100:.1f}%"
        )

        log("\n  [Breakdown by Annotation Count]")
        log(f"  {'Count':<8} {'Tasks':<8} {'FullAgree':<12} {'Majority':<12} {'PairwiseAvg':<12}")
        log(f"  {'-' * 52}")
        by_count: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for task in multi_tasks.values():
            by_count[len(task["annotations"])].append(task)
        for n_ann in sorted(by_count.keys()):
            tasks = by_count[n_ann]
            full = sum(
                1
                for task in tasks
                if len(set(annotation["choice"] for annotation in task["annotations"])) == 1
            )
            maj = sum(
                1
                for task in tasks
                if Counter(
                    annotation["choice"] for annotation in task["annotations"]
                ).most_common(1)[0][1] > len(task["annotations"]) / 2
            )
            pw_list: list[float] = []
            for task in tasks:
                choices = [annotation["choice"] for annotation in task["annotations"]]
                n_choices = len(choices)
                if n_choices < 2:
                    pw_list.append(1.0)
                else:
                    agrees = sum(
                        1
                        for i in range(n_choices)
                        for j in range(i + 1, n_choices)
                        if choices[i] == choices[j]
                    )
                    pw_list.append(agrees / (n_choices * (n_choices - 1) // 2))
            avg = sum(pw_list) / len(pw_list)
            log(f"  {n_ann:<8} {len(tasks):<8} {full:<12} {maj:<12} {avg * 100:.1f}%")

        val_multi = {tid: task for tid, task in multi_tasks.items() if tid in validation_tasks}
        if val_multi:
            log(f"\n  [Validation-Sample Multi-Tasks Only] ({len(val_multi)} tasks)")
            val_full = sum(
                1
                for task in val_multi.values()
                if len(set(annotation["choice"] for annotation in task["annotations"])) == 1
            )
            pw_vals: list[float] = []
            for task in val_multi.values():
                choices = [annotation["choice"] for annotation in task["annotations"]]
                n_choices = len(choices)
                if n_choices >= 2:
                    agrees = sum(
                        1
                        for i in range(n_choices)
                        for j in range(i + 1, n_choices)
                        if choices[i] == choices[j]
                    )
                    pw_vals.append(agrees / (n_choices * (n_choices - 1) // 2))
            avg_val_pw = sum(pw_vals) / len(pw_vals) if pw_vals else 0
            log(f"  Full agreement: {val_full}/{len(val_multi)} = {val_full / len(val_multi) * 100:.1f}%")
            log(f"  Avg pairwise: {avg_val_pw * 100:.1f}%")
    else:
        majority_votes = {}
        log("  No multi-annotator tasks found!")

    log(f"\n{'=' * 70}")
    log("[Analysis 2: RAG Candidate Source Hit Rate]")
    log(f"{'=' * 70}")

    source_hits: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"appears": 0, "chosen": 0})
    for task in task_annotations.values():
        data = task["data"]
        cand_sources: dict[str, str] = {}
        for cand in ["a", "b", "c", "d", "e"]:
            source = data.get(f"candidate_{cand}_source", "unknown")
            cand_sources[cand.upper()] = source
            source_hits[source]["appears"] += 1
        for annotation in task["annotations"]:
            if annotation["choice"] in cand_sources:
                source = cand_sources[annotation["choice"]]
                source_hits[source]["chosen"] += 1

    log(f"  {'Source':<35} {'Appears':<10} {'Chosen':<10} {'HitRate':<10}")
    log(f"  {'-' * 65}")
    for source in sorted(source_hits.keys()):
        stats = source_hits[source]
        rate = stats["chosen"] / stats["appears"] * 100 if stats["appears"] > 0 else 0
        log(f"  {source:<35} {stats['appears']:<10} {stats['chosen']:<10} {rate:.1f}%")

    log("\n[Analysis 2b: Human vs RAG Top1 & Top5]")
    human_chose_top1 = 0
    human_chose_topn: defaultdict[int, int] = defaultdict(int)
    human_chose_valid = 0

    for task in task_annotations.values():
        data = task["data"]
        rank_to_candidate: dict[int, str] = {}
        for cand in ["a", "b", "c", "d", "e"]:
            source = data.get(f"candidate_{cand}_source", "")
            for rank in [1, 2, 3, 4, 5]:
                if f"top{rank}" in source:
                    rank_to_candidate[rank] = cand.upper()
                    break

        for annotation in task["annotations"]:
            if annotation["choice"] not in ("A", "B", "C", "D", "E"):
                continue
            human_chose_valid += 1
            for rank, candidate in rank_to_candidate.items():
                if annotation["choice"] == candidate:
                    human_chose_topn[rank] += 1
                    if rank == 1:
                        human_chose_top1 += 1
                    break

    log(
        f"  Human chose RAG Top1: {human_chose_top1}/{human_chose_valid} = "
        f"{human_chose_top1 / human_chose_valid * 100:.1f}%"
    )
    cumulative = 0
    for rank in sorted(human_chose_topn.keys()):
        cumulative += human_chose_topn[rank]
        log(
            f"  Human chose RAG Top1-{rank}: {cumulative}/{human_chose_valid} = "
            f"{cumulative / human_chose_valid * 100:.1f}%"
        )

    log(f"\n{'=' * 70}")
    log("[Analysis 3: Deepseek-v4-pro vs Human]")
    log(f"{'=' * 70}")

    deepseek_data: dict[int, dict[str, Any]] = load_deepseek_records()
    log(f"  Deepseek records loaded: {len(deepseek_data)}")

    ds_vs_single_total = 0
    ds_vs_single_agree = 0
    ds_vs_maj_total = 0
    ds_vs_maj_agree = 0
    ds_all_total = 0
    ds_all_agree = 0

    if deepseek_data:
        for tid, task in single_tasks.items():
            if tid not in deepseek_data:
                continue
            human_choice = task["annotations"][0]["choice"]
            if human_choice == "NONE":
                continue
            ds_choice = deepseek_data[tid]["deepseek_choice"]
            ds_vs_single_total += 1
            if ds_choice == human_choice:
                ds_vs_single_agree += 1

        if ds_vs_single_total > 0:
            log("\n  [Deepseek vs Single-Annotator Human]")
            log(f"  Compared: {ds_vs_single_total}")
            log(
                f"  Agreement: {ds_vs_single_agree}/{ds_vs_single_total} = "
                f"{ds_vs_single_agree / ds_vs_single_total * 100:.1f}%"
            )

        for tid, majority in majority_votes.items():
            if tid not in deepseek_data or majority in (None, "NONE"):
                continue
            ds_choice = deepseek_data[tid]["deepseek_choice"]
            ds_vs_maj_total += 1
            if ds_choice == majority:
                ds_vs_maj_agree += 1

        if ds_vs_maj_total > 0:
            log("\n  [Deepseek vs Majority-Vote]")
            log(f"  Compared: {ds_vs_maj_total}")
            log(
                f"  Agreement: {ds_vs_maj_agree}/{ds_vs_maj_total} = "
                f"{ds_vs_maj_agree / ds_vs_maj_total * 100:.1f}%"
            )

        for tid, task in task_annotations.items():
            if tid not in deepseek_data:
                continue
            if tid in majority_votes and majority_votes[tid] is not None:
                ref = majority_votes[tid]
            elif len(task["annotations"]) == 1:
                ref = task["annotations"][0]["choice"]
            else:
                continue
            if ref == "NONE":
                continue
            ds_choice = deepseek_data[tid]["deepseek_choice"]
            ds_all_total += 1
            if ds_choice == ref:
                ds_all_agree += 1

        if ds_all_total > 0:
            log("\n  [Deepseek vs Human (All Tasks)]")
            log(f"  Compared: {ds_all_total}")
            log(
                f"  Agreement: {ds_all_agree}/{ds_all_total} = "
                f"{ds_all_agree / ds_all_total * 100:.1f}%"
            )

        log("\n  [Deepseek Accuracy by Confidence]")
        log(f"  {'Confidence':<14} {'Tasks':<8} {'Correct':<8} {'Accuracy':<10}")
        log(f"  {'-' * 40}")
        ds_by_conf: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for tid, task in task_annotations.items():
            if tid not in deepseek_data:
                continue
            if tid in majority_votes and majority_votes[tid] is not None:
                ref = majority_votes[tid]
            elif len(task["annotations"]) == 1:
                ref = task["annotations"][0]["choice"]
            else:
                continue
            if ref == "NONE":
                continue
            ds_rec = deepseek_data[tid]
            conf = ds_rec["deepseek_confidence"]
            bucket = f"{conf:.1f}"
            ds_by_conf[bucket]["total"] += 1
            if ds_rec["deepseek_choice"] == ref:
                ds_by_conf[bucket]["correct"] += 1

        for bucket in sorted(ds_by_conf.keys()):
            stats = ds_by_conf[bucket]
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            log(f"  {bucket:<14} {stats['total']:<8} {stats['correct']:<8} {acc:.1f}%")

        ds_choice_dist: Counter[str] = Counter()
        for rec in deepseek_data.values():
            ds_choice_dist[rec["deepseek_choice"]] += 1
        log("\n  [Deepseek Choice Distribution]")
        for choice in sorted(ds_choice_dist.keys(), key=lambda item: ds_choice_dist[item], reverse=True):
            log(
                f"  {choice}: {ds_choice_dist[choice]} "
                f"({ds_choice_dist[choice] / len(deepseek_data) * 100:.1f}%)"
            )

    log(f"\n{'=' * 70}")
    log("[Analysis 4: Per-Annotator Quality]")
    log(f"{'=' * 70}")

    ann_stats: defaultdict[Any, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "multi": 0, "agree_maj": 0}
    )
    for tid, task in task_annotations.items():
        majority = majority_votes.get(tid)
        for annotation in task["annotations"]:
            aid = annotation["annotator"]
            ann_stats[aid]["total"] += 1
            if len(task["annotations"]) >= 2:
                ann_stats[aid]["multi"] += 1
                if majority is not None and annotation["choice"] == majority:
                    ann_stats[aid]["agree_maj"] += 1

    log(f"  {'Ann':<8} {'Total':<8} {'Multi':<8} {'AgreeMaj':<10} {'Rate':<8}")
    log(f"  {'-' * 42}")
    for aid in sorted(ann_stats.keys()):
        stats = ann_stats[aid]
        if stats["multi"] > 0:
            rate = stats["agree_maj"] / stats["multi"] * 100
            log(
                f"  {aid:<8} {stats['total']:<8} {stats['multi']:<8} "
                f"{stats['agree_maj']:<10} {rate:.1f}%"
            )
        else:
            log(
                f"  {aid:<8} {stats['total']:<8} {stats['multi']:<8} "
                f"{stats['agree_maj']:<10} N/A"
            )

    log(f"\n{'=' * 70}")
    log("[Analysis 5: 'NONE' (以上选项都不属于) Analysis]")
    log(f"{'=' * 70}")

    none_majority_tasks = [tid for tid, majority in majority_votes.items() if majority == "NONE"]
    log(f"  Tasks with majority=NONE: {len(none_majority_tasks)}")

    none_single = sum(
        1
        for task in single_tasks.values()
        if task["annotations"][0]["choice"] == "NONE"
    )
    log(
        f"  Single-annotator tasks chose NONE: {none_single}/{len(single_tasks)} "
        f"({none_single / len(single_tasks) * 100:.1f}%)"
    )

    log(f"\n{'=' * 70}")
    log("[Analysis 6: Validation Samples (is_validation_sample=1) Deep Dive]")
    log(f"{'=' * 70}")

    val_tasks_data = {
        tid: task_annotations[tid]
        for tid in validation_tasks
        if tid in task_annotations
    }
    log(f"  Validation sample tasks: {len(val_tasks_data)}")

    if val_tasks_data:
        val_choices: Counter[str | None] = Counter()
        for task in val_tasks_data.values():
            for annotation in task["annotations"]:
                val_choices[annotation["choice"]] += 1
        log("\n  [Validation Set Choice Distribution]")
        val_total = sum(val_choices.values())
        for choice in sorted(val_choices.keys(), key=lambda item: val_choices[item], reverse=True):
            log(
                f"  {choice}: {val_choices[choice]} "
                f"({val_choices[choice] / val_total * 100:.1f}%)"
            )

        val_full = sum(
            1
            for task in val_tasks_data.values()
            if len(task["annotations"]) >= 2
            and len(set(annotation["choice"] for annotation in task["annotations"])) == 1
        )
        val_multi_count = sum(
            1 for task in val_tasks_data.values() if len(task["annotations"]) >= 2
        )
        if val_multi_count > 0:
            log(
                f"\n  Full agreement (validation, multi-ann): {val_full}/{val_multi_count} = "
                f"{val_full / val_multi_count * 100:.1f}%"
            )

        if deepseek_data:
            ds_val_total = 0
            ds_val_agree = 0
            for tid in validation_tasks:
                if tid not in deepseek_data or tid not in task_annotations:
                    continue
                task = task_annotations[tid]
                if tid in majority_votes and majority_votes[tid] is not None:
                    ref = majority_votes[tid]
                elif len(task["annotations"]) == 1:
                    ref = task["annotations"][0]["choice"]
                else:
                    continue
                if ref == "NONE":
                    continue
                ds_val_total += 1
                if deepseek_data[tid]["deepseek_choice"] == ref:
                    ds_val_agree += 1
            if ds_val_total > 0:
                log(
                    f"  Deepseek accuracy on validation: {ds_val_agree}/{ds_val_total} = "
                    f"{ds_val_agree / ds_val_total * 100:.1f}%"
                )

    log(f"\n{'=' * 70}")
    log("[SUMMARY: All Metrics]")
    log(f"{'=' * 70}")

    metrics: list[tuple[str, float]] = []
    metrics.append(("Human chose A-E (not NONE) [all tasks]", valid_count / total * 100))
    metrics.append(("Human chose A-E (single-ann tasks)", single_valid / single_total * 100))

    if human_chose_valid > 0:
        metrics.append(("Human agree with RAG Top1", human_chose_top1 / human_chose_valid * 100))
        metrics.append(("Human agree with RAG Top1-5", cumulative / human_chose_valid * 100))

    if len(multi_tasks) > 0:
        metrics.append(("Multi-ann full agreement", full_agree / len(multi_tasks) * 100))
        metrics.append(("Multi-ann pairwise agreement", avg_pairwise * 100))
        metrics.append(("Multi-ann majority exists", majority_exists / len(multi_tasks) * 100))

        non_none_multi = 0
        non_none_full = 0
        for task in multi_tasks.values():
            non_none_choices = [
                annotation["choice"]
                for annotation in task["annotations"]
                if annotation["choice"] != "NONE"
            ]
            if len(non_none_choices) >= 2:
                non_none_multi += 1
                if len(set(non_none_choices)) == 1:
                    non_none_full += 1
        if non_none_multi > 0:
            metrics.append(("Multi-ann full agreement (excl NONE)", non_none_full / non_none_multi * 100))

    if ds_all_total > 0:
        metrics.append(("Deepseek agree with human (all tasks)", ds_all_agree / ds_all_total * 100))
    if ds_vs_single_total > 0:
        metrics.append(("Deepseek agree with human (single-ann)", ds_vs_single_agree / ds_vs_single_total * 100))
    if ds_vs_maj_total > 0:
        metrics.append(("Deepseek agree with human (majority)", ds_vs_maj_agree / ds_vs_maj_total * 100))

    if val_tasks_data:
        val_multi_count = sum(1 for task in val_tasks_data.values() if len(task["annotations"]) >= 2)
        if val_multi_count > 0:
            pw_vals: list[float] = []
            for task in val_tasks_data.values():
                if len(task["annotations"]) < 2:
                    continue
                choices = [annotation["choice"] for annotation in task["annotations"]]
                n_choices = len(choices)
                agrees = sum(
                    1
                    for i in range(n_choices)
                    for j in range(i + 1, n_choices)
                    if choices[i] == choices[j]
                )
                pw_vals.append(agrees / (n_choices * (n_choices - 1) // 2))
            avg_val_pw = sum(pw_vals) / len(pw_vals) if pw_vals else 0
            metrics.append(("Validation set pairwise agreement", avg_val_pw * 100))

    log(f"\n  {'Metric':<55} {'Value':<10}")
    log(f"  {'-' * 65}")
    for name, value in metrics:
        marker = " <-- CLOSE TO 82.8%" if abs(value - 82.8) < 3.0 else ""
        log(f"  {name:<55} {value:.1f}%{marker}")

    log("\n  Looking for metric closest to 82.8%...")
    closest = min(metrics, key=lambda item: abs(item[1] - 82.8))
    log(f"  Closest: {closest[0]} = {closest[1]:.1f}% (diff={abs(closest[1] - 82.8):.1f}pp)")

    log(f"\n{'=' * 70}")
    log("Analysis complete.")
    log(f"{'=' * 70}")

    report_body = "\n".join(output_lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file_obj:
        file_obj.write(report_body)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

