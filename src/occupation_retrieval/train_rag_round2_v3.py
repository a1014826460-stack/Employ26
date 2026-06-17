#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG 训练 v3：Silver/Gold 标签集（BGE-M3 fine-tuning with label quality filtering）。

策略:
    Gold — 多标注任务中 DeepSeek 与人类多数意见一致的样本
    Silver — 单标注任务中 DeepSeek 与人类选择一致的样本
    Excluded — DeepSeek 与人类不一致（两者可能都有噪声）

仅使用 Gold + Silver 进行训练，在保留的测试集上评估。

用法:
    python -m src.occupation_retrieval.train_rag_round2_v3

前置条件:
    1. 已有标注 JSON 和 DeepSeek relabel 结果
    2. BGE 模型路径通过 config/paths.py 或环境变量 EMPLOYDATA_BGE_MODEL_PATH 配置
"""

from __future__ import annotations

import json
import os
import time
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from config.paths import get_project_paths
from .common import (
    get_runtime_device,
    get_training_output_dir,
    load_annotations_from_pg,
    load_deepseek_records,
    load_occupation_dict_df,
    resolve_base_model_path,
    resolve_model_dir,
)
from .datasets import parse_choice as parse_human_choice

_project = get_project_paths()
BASE_DIR = str(_project.project_root)
OUTPUT_DIR = str(get_training_output_dir())

BASE_MODEL_PATH = resolve_base_model_path()
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "bge-large-round2-finetuned-v3")


# ── 配置 ────────────────────────────────────────
@dataclass
class Config:
    """v3 训练方案配置。"""

    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    max_seq_length: int = 256
    warmup_ratio: float = 0.1
    random_seed: int = 42
    test_ratio: float = 0.15  # Silver 中留出多少做测试


def load_occupation_dict(dict_path: str) -> dict[str, str]:
    """加载职业大典并构造训练用的正样本文本。"""
    df = load_occupation_dict_df()
    code_to_text: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        title = str(row["title"]).strip()
        desc = str(row["desc"]).strip()
        tasks = str(row.get("tasks", "")).strip()
        if not code or not title: continue
        parts = [title]
        if desc and desc.lower() != "nan": parts.append(f"定义：{desc}")
        if tasks and tasks.lower() != "nan": parts.append(f"任务：{tasks}")
        code_to_text[code] = "。".join(parts)
    return code_to_text


def build_pair(
    task_id: int,
    anchor: str,
    code: str,
    code_to_text: dict[str, str],
    n_annotators: int,
    label_type: str,
) -> dict[str, Any] | None:
    """构建单条训练样本对。"""
    if not code or code not in code_to_text:
        return None
    return {
        "task_id": task_id,
        "anchor": anchor,
        "positive": code_to_text[code],
        "code": code,
        "n_annotators": n_annotators,
        "label_type": label_type,
    }


def main() -> None:
    """执行 v3 Silver/Gold 方案训练与评估。"""
    config = Config()
    print("=" * 70)
    print("RAG Training v3: Silver/Gold Label Set")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 加载数据 ──
    print("\n[Step 1] Loading data...")
    raw_data = load_annotations_from_pg()
    print(f"  Human annotations: {len(raw_data)} tasks")

    ds_records = load_deepseek_records()
    print(f"  Deepseek records: {len(ds_records)}")

    code_to_text = load_occupation_dict("")
    print(f"  Occupation dict: {len(code_to_text)} entries")

    # ── 按 task_id 索引人工标注 ──
    human_data: dict[int, dict[str, Any]] = {}
    for item in raw_data:
        tid = item["task_id"]
        choices = []
        for ann in item["annotations"]:
            c = parse_human_choice(ann)
            choices.append(c)
        human_data[tid] = {
            "annotations": choices,
            "data": item["data"],
            "recruitment_record_id": item["recruitment_record_id"],
            "n_annotators": len(item["annotations"]),
        }

    # ── 分类：Gold / Silver / Excluded ──
    gold_pairs: list[dict[str, Any]] = []
    silver_pairs: list[dict[str, Any]] = []
    excluded_disagree = 0
    excluded_none = 0
    excluded_no_ds = 0

    # 处理有 Deepseek 标注的任务
    ds_overlap = set(ds_records.keys()) & set(human_data.keys())
    print(f"\n  Tasks with both human & Deepseek: {len(ds_overlap)}")

    for tid in ds_overlap:
        h = human_data[tid]
        ds = ds_records[tid]
        ds_choice = ds["deepseek_choice"]
        data = h["data"]

        # 获取人类多数意见
        valid_choices = [c for c in h["annotations"] if c and c != "NONE"]
        if not valid_choices:
            excluded_none += 1
            continue

        if h["n_annotators"] >= 2:
            counter = Counter(valid_choices)
            human_choice, cnt = counter.most_common(1)[0]
            if cnt <= h["n_annotators"] / 2:
                # 无明显多数
                continue
        else:
            human_choice = valid_choices[0]

        if human_choice == "NONE":
            excluded_none += 1
            continue

        # 人类选择的候选对应的职业代码
        human_code = str(data.get(f"candidate_{human_choice.lower()}_code", "")).strip()

        # 构建 anchor
        job_title = str(data.get("job_title", "")).strip()
        job_reqs = str(data.get("job_requirements_clean", "")).strip()
        if not job_reqs:
            continue
        anchor = f"{job_title} {job_reqs}" if job_title else job_reqs

        # 分类
        if ds_choice == human_choice:
            # Deepseek 与人类一致
            if h["n_annotators"] >= 2:
                # Gold: 多标注 + Deepseek 一致
                pair = build_pair(tid, anchor, human_code, code_to_text,
                                  h["n_annotators"], "gold")
                if pair: gold_pairs.append(pair)
            else:
                # Silver: 单标注 + Deepseek 一致
                pair = build_pair(tid, anchor, human_code, code_to_text,
                                  h["n_annotators"], "silver")
                if pair: silver_pairs.append(pair)
        else:
            # Deepseek 与人类不一致 → 排除
            excluded_disagree += 1

    print(f"\n  [Label Distribution]")
    print(f"    Gold   (multi-ann + DS agree):  {len(gold_pairs)}")
    print(f"    Silver (single-ann + DS agree): {len(silver_pairs)}")
    print(f"    Total training candidates:      {len(gold_pairs) + len(silver_pairs)}")
    print(f"\n    Excluded - DS disagrees: {excluded_disagree}")
    print(f"    Excluded - human=NONE:   {excluded_none}")

    if len(gold_pairs) + len(silver_pairs) == 0:
        print("ERROR: No training data!")
        return

    # ── 划分 train/test ──
    # Gold + Silver 全部用于训练
    # 测试集: 所有不在 Gold/Silver 中的有效数据
    print(f"\n[Step 2] Split train/test...")
    random.seed(config.random_seed)

    train_pairs = list(gold_pairs) + list(silver_pairs)
    train_ids = {p["task_id"] for p in train_pairs}

    # 测试集: 排除 train_ids 后的所有有效数据
    test_pairs: list[dict[str, Any]] = []
    for tid, h in human_data.items():
        if tid in train_ids:
            continue
        valid = [c for c in h["annotations"] if c and c != "NONE"]
        if not valid:
            continue
        if h["n_annotators"] >= 2:
            counter = Counter(valid)
            choice, cnt = counter.most_common(1)[0]
            if cnt <= h["n_annotators"] / 2:
                continue
        else:
            choice = valid[0]

        data = h["data"]
        code = str(data.get(f"candidate_{choice.lower()}_code", "")).strip()
        if not code or code not in code_to_text:
            continue
        job_title = str(data.get("job_title", "")).strip()
        job_reqs = str(data.get("job_requirements_clean", "")).strip()
        if not job_reqs:
            continue
        anchor = f"{job_title} {job_reqs}" if job_title else job_reqs
        test_pairs.append({
            "task_id": tid, "anchor": anchor,
            "positive": code_to_text[code], "code": code,
            "n_annotators": h["n_annotators"],
            "label_type": "test",
        })

    print(f"  Train: {len(train_pairs)} (gold={len(gold_pairs)}, silver={len(silver_pairs)})")
    print(f"  Test:  {len(test_pairs)} (all non-gold/silver valid data)")

    # 统计测试集构成
    test_multi = sum(1 for p in test_pairs if p["n_annotators"] >= 2)
    test_single = len(test_pairs) - test_multi
    print(f"    Multi-ann in test: {test_multi}, Single-ann in test: {test_single}")

    # ── 构建 InputExample ──
    train_examples = [InputExample(texts=[p["anchor"], p["positive"]]) for p in train_pairs]

    # ── 训练 ──
    print(f"\n[Step 3] Training...")
    device = get_runtime_device()
    model = SentenceTransformer(BASE_MODEL_PATH, device=device)
    model.max_seq_length = config.max_seq_length
    print(f"  Device: {device}, Dim: {model.get_sentence_embedding_dimension()}")
    print(f"  Batches/epoch: {len(train_examples)//config.batch_size + 1}")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_dataloader) * config.epochs * config.warmup_ratio)

    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=OUTPUT_MODEL_PATH,
        show_progress_bar=True,
        save_best_model=True,
        use_amp=True,
        evaluator=None,
    )
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")

    # ── 评估（分块计算避免 OOM）──
    print(f"\n[Step 4] Evaluation...")
    codes = sorted(code_to_text.keys())
    occ_texts = [code_to_text[c] for c in codes]
    code_to_idx = {c: i for i, c in enumerate(codes)}
    test_anchors = [p["anchor"] for p in test_pairs]

    with torch.no_grad():
        occ_emb = model.encode(occ_texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_tensor=True)

    # 分块编码 + 分块计算相似度
    EVAL_CHUNK = 1000
    topk = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    for start in range(0, len(test_anchors), EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, len(test_anchors))
        chunk = test_anchors[start:end]
        with torch.no_grad():
            chunk_emb = model.encode(chunk, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_tensor=True)
            chunk_sim = torch.mm(chunk_emb, occ_emb.T)
            _, chunk_ranked = torch.topk(chunk_sim, k=10, dim=1)
        chunk_ranked = chunk_ranked.cpu().tolist()

        for i, rk in enumerate(chunk_ranked):
            pair = test_pairs[start + i]
            gt = code_to_idx.get(pair["code"])
            if gt is None: continue
            total += 1
            for rank, pred in enumerate(rk, 1):
                if pred == gt:
                    for k in topk:
                        if rank <= k: topk[k] += 1
                    break
        print(f"  Eval progress: {end}/{len(test_anchors)}")

    print(f"\n  Test N={total}")
    for k in [1, 3, 5, 10]:
        print(f"  Top-{k:>2}: {topk[k]:>5}/{total} = {topk[k]/total*100:.1f}%")

    # ── 对比 v1（释放当前模型，加载v1）──
    print(f"\n  [Comparison with v1 (all data)]")
    del model
    torch.cuda.empty_cache()
    v1_path = resolve_model_dir("bge-large-round2-finetuned")
    v1_model = SentenceTransformer(v1_path, device=device)
    v1_model.max_seq_length = 256
    v1_topk = {1: 0, 3: 0, 5: 0}
    v1_total = 0
    with torch.no_grad():
        v1_occ = v1_model.encode(occ_texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=True, convert_to_tensor=True)
    for start in range(0, len(test_anchors), EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, len(test_anchors))
        chunk = test_anchors[start:end]
        with torch.no_grad():
            chunk_emb = v1_model.encode(chunk, batch_size=64, normalize_embeddings=True,
                                        show_progress_bar=False, convert_to_tensor=True)
            chunk_sim = torch.mm(chunk_emb, v1_occ.T)
            _, chunk_ranked = torch.topk(chunk_sim, k=5, dim=1)
        chunk_ranked = chunk_ranked.cpu().tolist()
        for i, rk in enumerate(chunk_ranked):
            pair = test_pairs[start + i]
            gt = code_to_idx.get(pair["code"])
            if gt is None: continue
            v1_total += 1
            for rank, pred in enumerate(rk, 1):
                if pred == gt:
                    for k in v1_topk:
                        if rank <= k: v1_topk[k] += 1
                    break

    print(f"  {'':<20} {'v1 (all data)':<15} {'v3 (silver/gold)':<15}")
    print(f"  {'-'*50}")
    for k in [1, 3, 5]:
        v1_pct = v1_topk[k]/v1_total*100
        v3_pct = topk[k]/total*100
        diff = v3_pct - v1_pct
        sign = "+" if diff > 0 else ""
        print(f"  Top-{k:<18} {v1_pct:.1f}%{'':8} {v3_pct:.1f}% ({sign}{diff:.1f}pp)")

    # 保存
    results = {
        "train_gold": len(gold_pairs),
        "train_silver": len(silver_pairs),
        "test_total": total,
        "v3_top1": topk[1]/total*100 if total else 0,
        "v3_top3": topk[3]/total*100 if total else 0,
        "v3_top5": topk[5]/total*100 if total else 0,
        "v3_top10": topk[10]/total*100 if total else 0,
        "v1_top1": v1_topk[1]/v1_total*100 if v1_total else 0,
        "v1_top3": v1_topk[3]/v1_total*100 if v1_total else 0,
        "v1_top5": v1_topk[5]/v1_total*100 if v1_total else 0,
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_v3.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Model: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()

