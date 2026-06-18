#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多指标模型评估：超越 Top-1 准确率的综合评测。

评估维度:
    1. 候选选择准确率 — 从 5 个候选（A-E）中选对的比例（实际标注场景）
    2. 候选排序质量 — 人类选择在 5 个候选中排第几
    3. 分歧仲裁准确率 — 320 条分歧中，模型更接近人还是 DeepSeek
    4. 小类/中类/大类级准确率 — 不要求精确到细类
    5. MRR (Mean Reciprocal Rank) — 全量检索的平均倒数排名
    6. NDCG@5 — 归一化折损累积增益

用法:
    python -m src.occupation_retrieval.eval_models_multimetric

前置条件:
    1. 各版本微调模型已保存到 OUTPUT_DIR
    2. BGE baseline 模型路径通过 config/paths.py 或环境变量配置
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

import numpy as np

from config.paths import get_project_paths
from .common import (
    get_occupation_retrieval_output_dir,
    get_runtime_device,
    get_training_output_dir,
    load_annotations_from_pg,
    load_deepseek_records,
    load_occupation_dict_df,
    resolve_base_model_path,
    resolve_existing_model_path,
    resolve_model_dir,
)
from .datasets import build_anchor, build_candidate_records, get_task_choices
from .metrics import (
    compute_candidate_accuracy,
    reciprocal_rank,
    same_hierarchy_level,
    summarize_model_metrics,
)

_project = get_project_paths()
BASE_DIR = str(_project.project_root)
OUTPUT_DIR = str(get_training_output_dir())
OUTPUT_FILE = os.path.join(get_occupation_retrieval_output_dir(), "model_comparison.md")

DEFAULT_MODEL_PATHS = {
    "v1 (全量)":             resolve_model_dir("bge-large-round2-finetuned"),
    "v3 (Silver/Gold)":      resolve_model_dir("bge-large-round2-finetuned-v3"),
    "v4 (Medium分歧)":       resolve_model_dir("bge-large-round2-finetuned-v4"),
    "baseline (bge-large)":  resolve_base_model_path(),
}


def parse_args() -> argparse.Namespace:
    """解析统一评估脚本参数。"""
    parser = argparse.ArgumentParser(
        description="运行 occupation_retrieval 检索模型统一评估，支持显式传入待评估模型列表。",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "显式添加一个待评估模型，可重复传入。"
            "例如 --model v1-bge-m3=output/occupation_retrieval/rag_round2_training/v1-bge-m3"
        ),
    )
    return parser.parse_args()


def build_model_paths(model_args: list[str]) -> dict[str, str]:
    """根据命令行参数构造待评估模型列表。"""
    if not model_args:
        return dict(DEFAULT_MODEL_PATHS)

    model_paths = dict(DEFAULT_MODEL_PATHS)
    for raw in model_args:
        if "=" not in raw:
            raise SystemExit(f"--model 参数格式错误，应为 NAME=PATH，收到: {raw}")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(f"--model 参数格式错误，应为 NAME=PATH，收到: {raw}")
        model_paths[name] = path
    return model_paths


def load_dict() -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """加载职业大典并返回文本与层级映射。"""
    df = load_occupation_dict_df()
    c2text: dict[str, str] = {}
    c2title: dict[str, str] = {}
    c2subclass: dict[str, str] = {}
    c2midclass: dict[str, str] = {}
    c2major: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        title = str(row["title"]).strip()
        desc = str(row.get("desc", ""))
        tasks = str(row.get("tasks", ""))
        if not code or not title:
            continue
        c2title[code] = title
        parts = [title]
        if desc and desc != "nan":
            parts.append(f"定义：{desc}")
        if tasks and tasks != "nan":
            parts.append(f"任务：{tasks}")
        c2text[code] = "。".join(parts)
        parts = code.split("-")
        c2subclass[code] = "-".join(parts[:3]) if len(parts) >= 3 else code
        c2midclass[code] = "-".join(parts[:2]) if len(parts) >= 2 else code
        c2major[code] = parts[0] if parts else code
    return c2text, c2title, c2subclass, c2midclass, c2major


def main() -> None:
    """执行多版本嵌入模型的综合评估。"""
    args = parse_args()
    model_paths = build_model_paths(args.model)

    import torch
    from sentence_transformers import SentenceTransformer

    print("Loading data...")
    raw_data = load_annotations_from_pg()
    ds_records = load_deepseek_records()
    c2text, c2title, c2subclass, c2midclass, c2major = load_dict()

    print("Building eval samples...")
    eval_samples: list[dict[str, Any]] = []
    for item in raw_data:
        tid = item["task_id"]
        recruitment_record_id = item["recruitment_record_id"]
        data = item["data"]
        jt = str(data.get("job_title", "")).strip()
        jr = str(data.get("job_requirements_clean", "")).strip()
        if not jr:
            continue
        anchor = build_anchor(jt, jr)

        anns = item["annotations"]
        choices = get_task_choices(item, include_none=False)
        if not choices:
            continue
        if len(anns) >= 2:
            ctr = Counter(choices)
            hum_choice, _ = ctr.most_common(1)[0]
        else:
            hum_choice = choices[0]

        candidates: list[dict[str, str]] = []
        for candidate in build_candidate_records(data):
            title = candidate["title"]
            code = candidate["code"]
            source = candidate["source"]
            if not code:
                continue
            text = c2text.get(code, title)
            candidates.append({
                "letter": candidate["letter"],
                "code": code,
                "title": c2title.get(code, title),
                "text": text,
                "source": source,
            })

        if len(candidates) < 3:
            continue

        ds = ds_records.get(tid)
        ds_choice = ds["deepseek_choice"] if ds else None

        eval_samples.append({
            "task_id": tid,
            "recruitment_record_id": recruitment_record_id,
            "anchor": anchor,
            "candidates": candidates,
            "human_choice": hum_choice,
            "ds_choice": ds_choice,
            "n_ann": len(anns),
        })

    print(f"  Eval samples: {len(eval_samples)}")

    occ_codes = sorted(c2text.keys())
    occ_texts = [c2text[c] for c in occ_codes]

    all_results: dict[str, dict[str, Any]] = {}

    for model_name, model_path in model_paths.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        device = get_runtime_device()
        resolved_model_path = resolve_existing_model_path(model_path, label=model_name)
        model = SentenceTransformer(str(resolved_model_path), device=device)
        model.max_seq_length = 256

        with torch.no_grad():
            occ_emb = model.encode(
                occ_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=True,
            )

        results = {
            "candidate_hit": 0,
            "candidate_total": 0,
            "human_rank_in_candidates": [],
            "subclass_hit": 0,
            "midclass_hit": 0,
            "major_hit": 0,
            "full_total": 0,
            "reciprocal_ranks": [],
            "ds_side_human": 0,
            "ds_side_ds": 0,
            "ds_total": 0,
        }

        candidate_anchor_texts = [sample["anchor"] for sample in eval_samples]
        with torch.no_grad():
            candidate_anchor_emb = model.encode(
                candidate_anchor_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=True,
            )

        for sample_index, sample in enumerate(eval_samples):
            candidates = sample["candidates"]
            cand_texts = [c["text"] for c in candidates]

            with torch.no_grad():
                cand_emb = model.encode(
                    cand_texts,
                    batch_size=len(cand_texts),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                )
                anc_emb = candidate_anchor_emb[sample_index].unsqueeze(0)
                sims = torch.mm(anc_emb, cand_emb.T).squeeze(0)
                sorted_idx = torch.argsort(sims, descending=True).cpu().tolist()

            model_pick = candidates[sorted_idx[0]]["letter"]

            results["candidate_total"] += 1
            results["candidate_hit"] += compute_candidate_accuracy(
                model_pick,
                sample["human_choice"],
            )

            human_letter = sample["human_choice"]
            human_idx_in_cands = next(
                (i for i, c in enumerate(candidates) if c["letter"] == human_letter),
                None,
            )
            if human_idx_in_cands is not None:
                human_rank = sorted_idx.index(human_idx_in_cands) + 1
                results["human_rank_in_candidates"].append(human_rank)

            if sample["ds_choice"] and sample["ds_choice"] != sample["human_choice"]:
                results["ds_total"] += 1
                model_rank_human = (
                    sorted_idx.index(human_idx_in_cands) + 1
                    if human_idx_in_cands is not None
                    else 99
                )
                ds_letter = sample["ds_choice"]
                ds_idx = next(
                    (i for i, c in enumerate(candidates) if c["letter"] == ds_letter),
                    None,
                )
                model_rank_ds = sorted_idx.index(ds_idx) + 1 if ds_idx is not None else 99
                if model_rank_human < model_rank_ds:
                    results["ds_side_human"] += 1
                elif model_rank_ds < model_rank_human:
                    results["ds_side_ds"] += 1

        np.random.seed(42)
        mrr_samples = np.random.choice(
            len(eval_samples),
            min(3000, len(eval_samples)),
            replace=False,
        )
        mrr_anchors = [eval_samples[i]["anchor"] for i in mrr_samples]

        mrr_human_codes: list[str | None] = []
        for i in mrr_samples:
            s = eval_samples[i]
            hum_letter = s["human_choice"]
            hum_code = next((c["code"] for c in s["candidates"] if c["letter"] == hum_letter), None)
            mrr_human_codes.append(hum_code)

        with torch.no_grad():
            mrr_anc_emb = model.encode(
                mrr_anchors,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
            mrr_sims = torch.mm(mrr_anc_emb, occ_emb.T)
            _, mrr_ranked = torch.topk(mrr_sims, k=50, dim=1)
        mrr_ranked = mrr_ranked.cpu().tolist()

        for i, (code, ranked) in enumerate(zip(mrr_human_codes, mrr_ranked)):
            if not code or code not in occ_codes: continue
            target_idx = occ_codes.index(code)
            results["full_total"] += 1
            try:
                rank = ranked.index(target_idx) + 1
            except ValueError:
                rank = 51  # not in top 50
            results["reciprocal_ranks"].append(reciprocal_rank(rank))

            pred_code = occ_codes[ranked[0]]
            if same_hierarchy_level(code, pred_code, c2subclass):
                results["subclass_hit"] += 1
            if same_hierarchy_level(code, pred_code, c2midclass):
                results["midclass_hit"] += 1
            if same_hierarchy_level(code, pred_code, c2major):
                results["major_hit"] += 1

        results.update(summarize_model_metrics(results))

        all_results[model_name] = results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print comparison ──
    print(f"\n\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")

    metrics = [
        ("候选选择准确率", "candidate_acc", "%", "↑ 从5个候选里选对的概率"),
        ("人类选择在候选中的平均排位", "mean_human_rank", "", "↓ 越低越好, 1=完美"),
        ("分歧中偏向人类的比例", "ds_side_human_pct", "%", "↑ 模型更接近人类判断"),
        ("全量检索 MRR", "mrr", "", "↑ 平均倒数排名"),
        ("细类准确率 (Subclass)", "subclass_acc", "%", "↑ 前3位代码匹配"),
        ("中类准确率 (Midclass)", "midclass_acc", "%", "↑ 前2位代码匹配"),
        ("大类准确率 (Major)", "major_acc", "%", "↑ 第1位代码匹配"),
    ]

    for metric_name, key, unit, desc in metrics:
        print(f"\n  [{metric_name}] {desc}")
        print(f"  {'Model':<30} {'Value':>10}")
        print(f"  {'-'*40}")
        values = [(name, r[key]) for name, r in all_results.items()]
        best = max(values, key=lambda x: x[1]) if "↑" in desc else min(values, key=lambda x: x[1])
        for name, val in values:
            marker = " <-- best" if name == best[0] else ""
            if unit == "%":
                print(f"  {name:<30} {val:>9.1f}%{marker}")
            else:
                print(f"  {name:<30} {val:>9.3f}{marker}")

    print(f"\n\n  [候选排序分布] (人类选择在5个候选中的排位)")
    print(f"  {'Model':<30} {'Rank#1':>8} {'Rank#2':>8} {'Rank#3':>8} {'Rank#4':>8} {'Rank#5':>8}")
    for name in all_results:
        r = all_results[name]
        ranks = r["human_rank_in_candidates"]
        dist = {i: sum(1 for x in ranks if int(x) == i) for i in range(1, 6)}
        total = len(ranks)
        print(f"  {name:<30} " +
              " ".join(f"{dist[i]/total*100:>7.1f}%" for i in range(1, 6)))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for metric_name, key, unit, desc in metrics:
            f.write(f"\n[{metric_name}] {desc}\n")
            values = [(name, r[key]) for name, r in all_results.items()]
            best = max(values, key=lambda x: x[1]) if "↑" in desc else min(values, key=lambda x: x[1])
            for name, val in values:
                marker = " <-- best" if name == best[0] else ""
                if unit == "%":
                    f.write(f"  {name:<30} {val:>9.1f}%{marker}\n")
                else:
                    f.write(f"  {name:<30} {val:>9.3f}{marker}\n")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


