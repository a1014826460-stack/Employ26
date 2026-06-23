#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG 训练 v5：全量人标 + DeepSeek 高置信度验证过采样（BGE-large fine-tuning）。

策略:
    v1 冠军模型的问题：完全没有使用 DeepSeek 标签，浪费了宝贵的第二意见。
    v3/v4 的问题：用 DeepSeek 做减法（过滤掉不一致样本），减少了训练数据量。

    v5 策略 —— 用 DeepSeek 做加法：
    1. 保留全部 v1 人标训练对（基线，已被证明有效）
    2. DeepSeek 与人类一致的样本 → 过采样（2x-3x），强化高置信数据
    3. DeepSeek 与人类不一致但 DS 极高置信度（>= 0.95）→ 也加入（低权重），
       因为可能人类标错了
    4. 有 DeepSeek 标签但无有效人标的任务 → DS 高置信（>= 0.9）直接作为训练对

用法:
    python -m src.occupation_retrieval.train_rag_round2_v5

前置条件:
    1. annotations.label_studio_tasks_v2 已有人工标注
    2. annotations.deepseek_relabel_raw 已有 DeepSeek 重标结果
    3. BGE 模型路径通过 config/paths.py 或环境变量 EMPLOYDATA_BGE_MODEL_PATH 配置
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

from config.paths import get_project_paths
from .common import (
    get_runtime_device,
    get_training_output_dir,
    load_annotations_from_pg,
    load_deepseek_records,
    load_occupation_dict_df,
    resolve_base_model_path,
    resolve_existing_model_path,
)
from .datasets import build_anchor, get_majority_choice, get_task_choices

_project = get_project_paths()
BASE_DIR = str(_project.project_root)

OUTPUT_DIR = str(get_training_output_dir())

DEFAULT_BASE_MODEL_PATH = resolve_base_model_path()
DEFAULT_OUTPUT_MODEL_NAME = "bge-large-round2-finetuned-v5"
DEFAULT_RESULT_FILE_NAME = "evaluation_v5.json"


# ── 配置 ────────────────────────────────────────
@dataclass
class V5Config:
    """v5 训练方案配置。

    DeepSeek 采样权重：
        ds_agree_weight: DS 与人类一致时，额外复制的倍数（默认 2x → 最终 3 份）
        ds_disagree_weight: DS 与人类不一致但极高置信度时，加入的倍率（默认 0 → 不加入）
        ds_only_conf_threshold: 仅 DS 标注任务的置信度阈值（默认 0.90）
        ds_disagree_conf_threshold: DS 与人标不一致但仍采纳 DS 的置信度阈值（默认 0.95）
    """

    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    max_seq_length: int = 256
    warmup_ratio: float = 0.1
    test_ratio: float = 0.15
    random_seed: int = 42
    use_multi_ann_as_test: bool = True
    # DeepSeek 过采样参数
    ds_agree_conf_threshold: float = 0.80  # DS 一致时的最低置信度才过采样
    ds_agree_oversample: int = 2  # DS 一致且高置信 → 额外复制 2 份
    ds_disagree_conf_threshold: float = 0.95  # DS 不一致但极高置信 → 也加入
    ds_disagree_oversample: int = 0  # DS 不一致时默认不加入（0 表示跳过）
    ds_only_conf_threshold: float = 0.90  # 仅 DS 标注时的最低置信度


def _slugify_name(value: str) -> str:
    """将运行标签转换为适合文件名使用的形式。"""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return cleaned or "run"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行 occupation_retrieval v5 训练（人标全量 + DS 过采样）。",
    )
    parser.add_argument(
        "--base-model-path",
        help="覆盖基础 embedding 模型路径。",
    )
    parser.add_argument(
        "--output-model-name",
        help="输出模型目录名（默认: bge-large-round2-finetuned-v5）。",
    )
    parser.add_argument(
        "--run-label",
        help="运行标签，用于结果文件命名。",
    )
    parser.add_argument(
        "--ds-agree-oversample",
        type=int,
        default=2,
        help="DS 一致时的额外过采样倍数（默认: 2）。",
    )
    parser.add_argument(
        "--ds-disagree-oversample",
        type=int,
        default=0,
        help="DS 不一致但极高置信时的过采样倍数（默认: 0，不加入）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批次大小（默认: 32）。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数（默认: 3）。",
    )
    return parser.parse_args()


def load_occupation_dict() -> dict[str, str]:
    """加载《中国职业大典》，返回 {code: full_text} 映射。"""
    df = load_occupation_dict_df()
    col_map = {}
    for target in ["code", "title", "desc", "tasks"]:
        for col in df.columns:
            if target in str(col).lower() or (target == "code" and "代码" in str(col)):
                col_map[target] = col
                break
        else:
            col_map[target] = target

    code_to_text: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row[col_map["code"]]).strip()
        title = str(row[col_map["title"]]).strip()
        desc = str(row[col_map["desc"]]).strip()
        tasks = str(row[col_map.get("tasks", "tasks")]).strip()
        if not code or not title:
            continue
        parts = [title]
        if desc and desc.lower() != "nan":
            parts.append(f"定义：{desc}")
        if tasks and tasks.lower() != "nan":
            parts.append(f"任务：{tasks}")
        code_to_text[code] = "。".join(parts)
    return code_to_text


def make_pair(
    task_id: int,
    recruitment_record_id: str,
    anchor: str,
    code: str,
    code_to_text: dict[str, str],
    n_annotators: int,
    source: str,
    weight: int = 1,
) -> dict[str, Any]:
    """构造单条训练样本对（含来源标签）。"""
    return {
        "task_id": task_id,
        "recruitment_record_id": recruitment_record_id,
        "anchor": anchor,
        "positive": code_to_text[code],
        "code": code,
        "n_annotators": n_annotators,
        "source": source,  # "human" | "ds_agree" | "ds_disagree" | "ds_only"
        "weight": weight,
    }


# ── 1. 提取训练数据 ─────────────────────────────────
def extract_training_pairs(
    config: V5Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """提取 v5 训练正样本对（人标全量 + DS 高置信度验证过采样）。

    Returns:
        single_pairs: 单标注任务的正样本
        multi_pairs: 多标注任务的正样本
    """
    print("=" * 70)
    print("[Step 1] 提取 v5 训练正样本对")
    print(f"  策略: 人标全量 + DS一致(conf>={config.ds_agree_conf_threshold})过采样{config.ds_agree_oversample}x")
    print("=" * 70)

    # 加载数据
    raw_data = load_annotations_from_pg()
    print(f"  加载人标数据: {len(raw_data)} 条任务")

    ds_records = load_deepseek_records()
    print(f"  加载 DeepSeek 记录: {len(ds_records)} 条")

    code_to_text = load_occupation_dict()
    print(f"  加载职业大典: {len(code_to_text)} 个职业细类")

    # 统计
    single_pairs: list[dict[str, Any]] = []
    multi_pairs: list[dict[str, Any]] = []
    skipped_none = 0
    skipped_no_code = 0
    skipped_no_text = 0
    ds_agree_count = 0
    ds_disagree_added = 0
    ds_only_added = 0

    for item in raw_data:
        task_id = item["task_id"]
        recruitment_record_id = item["recruitment_record_id"]
        data = item["data"]
        job_title = str(data.get("job_title", "")).strip()
        job_reqs = str(data.get("job_requirements_clean", "")).strip()

        if not job_reqs:
            continue

        anchor = build_anchor(job_title, job_reqs)

        # ── 人类标注信息 ──
        choices = get_task_choices(item, include_none=False)

        if not choices:
            skipped_none += 1
            # 尝试仅用 DS 标注
            ds = ds_records.get(task_id)
            if ds and ds.get("deepseek_choice") not in (None, "NONE"):
                ds_conf = float(ds.get("deepseek_confidence", 0))
                if ds_conf >= config.ds_only_conf_threshold:
                    ds_choice = ds["deepseek_choice"]
                    code_key = f"candidate_{ds_choice.lower()}_code"
                    code = str(data.get(code_key, "")).strip()
                    if code and code in code_to_text:
                        pair = make_pair(
                            task_id, recruitment_record_id, anchor, code,
                            code_to_text, 0, "ds_only", weight=1,
                        )
                        single_pairs.append(pair)
                        ds_only_added += 1
            continue

        # 确定人类参考答案
        if len(item["annotations"]) >= 2:
            ref_choice, count = get_majority_choice(choices, require_strict=False)
            if not ref_choice or count <= len(item["annotations"]) / 2:
                continue
            is_multi = True
        else:
            ref_choice = choices[0]
            is_multi = False

        code_key = f"candidate_{ref_choice.lower()}_code"
        human_code = str(data.get(code_key, "")).strip()
        if not human_code:
            skipped_no_code += 1
            continue
        if human_code not in code_to_text:
            skipped_no_text += 1
            continue

        # ── 基础人类标注对（v1 逻辑）──
        pair = make_pair(
            task_id, recruitment_record_id, anchor, human_code,
            code_to_text, len(item["annotations"]), "human", weight=1,
        )
        if is_multi:
            multi_pairs.append(pair)
        else:
            single_pairs.append(pair)

        # ── DeepSeek 过采样逻辑 ──
        ds = ds_records.get(task_id)
        if not ds:
            continue

        ds_choice = ds.get("deepseek_choice")
        if ds_choice is None or ds_choice == "NONE":
            continue

        ds_conf = float(ds.get("deepseek_confidence", 0))

        if ds_choice == ref_choice and ds_conf >= config.ds_agree_conf_threshold:
            # DS 与人类一致且高置信 → 过采样
            for _ in range(config.ds_agree_oversample):
                extra_pair = make_pair(
                    task_id, recruitment_record_id, anchor, human_code,
                    code_to_text, len(item["annotations"]), "ds_agree", weight=2,
                )
                if is_multi:
                    multi_pairs.append(extra_pair)
                else:
                    single_pairs.append(extra_pair)
            ds_agree_count += 1

        elif (
            ds_choice != ref_choice
            and ds_conf >= config.ds_disagree_conf_threshold
            and config.ds_disagree_oversample > 0
        ):
            # DS 与人类不一致但极高置信 → 也加入（低权重）
            ds_code_key = f"candidate_{ds_choice.lower()}_code"
            ds_code = str(data.get(ds_code_key, "")).strip()
            if ds_code and ds_code in code_to_text:
                for _ in range(config.ds_disagree_oversample):
                    extra_pair = make_pair(
                        task_id, recruitment_record_id, anchor, ds_code,
                        code_to_text, len(item["annotations"]), "ds_disagree", weight=1,
                    )
                    if is_multi:
                        multi_pairs.append(extra_pair)
                    else:
                        single_pairs.append(extra_pair)
                ds_disagree_added += 1

    # ── 输出统计 ──
    n_human_pairs = (
        sum(1 for p in single_pairs if p["source"] == "human")
        + sum(1 for p in multi_pairs if p["source"] == "human")
    )
    n_ds_agree_pairs = (
        sum(1 for p in single_pairs if p["source"] == "ds_agree")
        + sum(1 for p in multi_pairs if p["source"] == "ds_agree")
    )

    print(f"\n  单标注正样本: {len(single_pairs)} 对")
    print(f"  多标注正样本: {len(multi_pairs)} 对")
    print(f"  合计: {len(single_pairs) + len(multi_pairs)} 对")
    print("\n  来源分布:")
    print(f"    人类标注: {n_human_pairs}")
    print(f"    DS一致过采样: {n_ds_agree_pairs} (来自 {ds_agree_count} 个任务)")
    print(f"    DS不一致但加入: {ds_disagree_added}")
    print(f"    仅DS标注: {ds_only_added}")
    print("\n  跳过原因:")
    print(f"    选了 NONE: {skipped_none}")
    print(f"    查不到代码: {skipped_no_code}")
    print(f"    代码不在大典中: {skipped_no_text}")

    # 统计职业覆盖
    all_codes = [p["code"] for p in single_pairs] + [p["code"] for p in multi_pairs]
    code_counts = Counter(all_codes)
    print(f"\n  覆盖职业细类数: {len(code_counts)}")
    print("  最频繁职业 Top5:")
    for ccode, cnt in code_counts.most_common(5):
        print(f"    {ccode}: {cnt} 条")

    return single_pairs, multi_pairs


# ── 2. 划分训练/测试集（按 task_id 分组，杜绝泄露）──
def split_train_test(
    single_pairs: list[dict[str, Any]],
    multi_pairs: list[dict[str, Any]],
    config: V5Config,
) -> tuple[
    list[Any],
    list[dict[str, Any]],
    list[Any],
    list[dict[str, Any]],
]:
    """划分训练集和测试集。

    策略: 多标注任务全部作为测试集（高质量评估基准），
          单标注任务按 task_id 分组后按比例划分，
          保证同一 task 的所有副本（人标+DS过采样）不会同时出现在 train 和 test。
    """
    print("\n" + "=" * 70)
    print("[Step 2] 划分训练/测试集")
    print("=" * 70)

    random.seed(config.random_seed)

    if config.use_multi_ann_as_test:
        # 多标注任务的 task_id 全部进 test
        multi_task_ids: set[int] = set()
        test_pairs = list(multi_pairs)
        for p in multi_pairs:
            multi_task_ids.add(p["task_id"])

        # 单标注任务按 task_id 分组
        single_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for p in single_pairs:
            single_by_task[p["task_id"]].append(p)

        # 随机打乱 task_id 列表，按比例分成 train/test
        task_ids = list(single_by_task.keys())
        random.shuffle(task_ids)
        n_test_tasks = int(len(task_ids) * config.test_ratio)
        test_task_ids = set(task_ids[:n_test_tasks])

        # 按 task_id 分配到 train/test（确保同一 task 的所有副本同侧）
        train_pairs: list[dict[str, Any]] = []
        for tid, pairs in single_by_task.items():
            if tid in test_task_ids:
                test_pairs.extend(pairs)
            else:
                train_pairs.extend(pairs)

        print(f"  策略: 多标注({len(multi_pairs)}对, {len(multi_task_ids)}个task) → test")
        print(f"        单标注 {len(single_by_task)} 个task → {n_test_tasks} test / {len(task_ids) - n_test_tasks} train")
    else:
        all_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for p in single_pairs + multi_pairs:
            all_by_task[p["task_id"]].append(p)
        task_ids = list(all_by_task.keys())
        random.shuffle(task_ids)
        n_test_tasks = int(len(task_ids) * config.test_ratio)
        test_task_ids = set(task_ids[:n_test_tasks])

        train_pairs = []
        test_pairs = []
        for tid, pairs in all_by_task.items():
            if tid in test_task_ids:
                test_pairs.extend(pairs)
            else:
                train_pairs.extend(pairs)

    def to_example(pair: dict[str, Any]) -> InputExample:
        return InputExample(texts=[pair["anchor"], pair["positive"]])

    train_examples = [to_example(p) for p in train_pairs]
    test_examples_ie = [to_example(p) for p in test_pairs]

    print(f"  训练集: {len(train_examples)} 对")
    print(f"  测试集: {len(test_pairs)} 对")

    # 验证无泄露：train 和 test 的 task_id 不应有交集
    train_tids = {p["task_id"] for p in train_pairs}
    test_tids = {p["task_id"] for p in test_pairs}
    leak = train_tids & test_tids
    if leak:
        print(f"  WARNING: {len(leak)} task_ids appear in BOTH train and test!")
    else:
        print("  OK: train/test task_ids are fully isolated, no data leak")

    # 测试集职业覆盖
    test_codes = set(p["code"] for p in test_pairs)
    train_codes = set(p["code"] for p in train_pairs)
    print(f"  测试集覆盖职业: {len(test_codes)} 个细类")
    print(f"  仅在测试集的职业: {len(test_codes - train_codes)} 个")

    return train_examples, train_pairs, test_examples_ie, test_pairs


# ── 3. 微调模型 ─────────────────────────────────────
def train_model(
    train_examples: list[Any],
    config: V5Config,
    output_model_path: str,
    base_model_path: str,
) -> Any:
    """使用 MultipleNegativesRankingLoss 微调 BGE 模型。"""
    print("\n" + "=" * 70)
    print("[Step 3] 微调检索模型")
    print("=" * 70)

    device = get_runtime_device()
    print(f"  设备: {device}")
    print(f"  基础模型: {base_model_path}")

    resolved_path = resolve_existing_model_path(base_model_path, label="基础 embedding")
    model = SentenceTransformer(str(resolved_path), device=device)
    model.max_seq_length = config.max_seq_length
    print(f"  最大序列长度: {config.max_seq_length}")
    print(f"  嵌入维度: {model.get_embedding_dimension()}")

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=config.batch_size,
    )
    train_loss = MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_dataloader) * config.epochs * config.warmup_ratio)

    print("\n  训练配置:")
    print(f"    Batch Size: {config.batch_size}")
    print(f"    Epochs: {config.epochs}")
    print(f"    Learning Rate: {config.learning_rate}")
    print(f"    Warmup Steps: {warmup_steps}")
    print(f"    Batches per epoch: {len(train_dataloader)}")

    os.makedirs(output_model_path, exist_ok=True)

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=output_model_path,
        show_progress_bar=True,
        save_best_model=True,
        use_amp=True,
        evaluator=None,
    )
    elapsed = time.time() - t0
    print(f"\n  训练完成，耗时: {elapsed/60:.1f} 分钟")
    print(f"  模型保存至: {output_model_path}")

    return model


# ── 4. 评估 ─────────────────────────────────────────
def evaluate_model(
    model: Any,
    test_pairs: list[dict[str, Any]],
    code_to_text: dict[str, str],
    config: V5Config,
    base_model_path: str,
) -> dict[str, Any]:
    """在测试集上评估检索准确率（Top1/Top3/Top5/Top10）。"""
    print("\n" + "=" * 70)
    print("[Step 4] 评估检索准确率")
    print("=" * 70)

    device = get_runtime_device()

    codes = sorted(code_to_text.keys())
    occ_texts = [code_to_text[c] for c in codes]
    code_to_idx = {c: i for i, c in enumerate(codes)}

    print(f"  职业细类总数: {len(codes)}")
    print(f"  测试样本数: {len(test_pairs)}")

    with torch.no_grad():
        occ_embeddings = model.encode(
            occ_texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=True, convert_to_tensor=True,
        )

    test_anchors = [p["anchor"] for p in test_pairs]
    test_codes = [p["code"] for p in test_pairs]

    # 分块编码避免 OOM
    EVAL_CHUNK = 1000
    topk_hits: dict[int, int] = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0

    for start in range(0, len(test_anchors), EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, len(test_anchors))
        chunk = test_anchors[start:end]
        with torch.no_grad():
            chunk_emb = model.encode(
                chunk, batch_size=64, normalize_embeddings=True,
                show_progress_bar=False, convert_to_tensor=True,
            )
            chunk_sim = torch.mm(chunk_emb, occ_embeddings.T)
            _, chunk_ranked = torch.topk(chunk_sim, k=10, dim=1)
        chunk_ranked = chunk_ranked.cpu().tolist()

        for i, rk in enumerate(chunk_ranked):
            pair = test_pairs[start + i]
            gt_idx = code_to_idx.get(pair["code"])
            if gt_idx is None:
                continue
            total += 1
            for rank, pred_idx in enumerate(rk, 1):
                if pred_idx == gt_idx:
                    for k in topk_hits:
                        if rank <= k:
                            topk_hits[k] += 1
                    break
        print(f"  评估进度: {end}/{len(test_anchors)}")

    # 输出结果
    print(f"\n  检索准确率 (N={total}):")
    print(f"  {'指标':<20} {'命中数':<10} {'准确率':<10}")
    print(f"  {'-'*40}")
    for k in [1, 3, 5, 10]:
        hit = topk_hits[k]
        pct = hit / total * 100 if total > 0 else 0
        print(f"  Top-{k:<18} {hit:<10} {pct:.1f}%")

    # 多标注测试子集
    multi_test = [p for p in test_pairs if p["n_annotators"] >= 2]
    multi_top1 = 0
    n_multi = 0
    if multi_test:
        multi_anchors = [p["anchor"] for p in multi_test]
        with torch.no_grad():
            multi_emb = model.encode(
                multi_anchors, batch_size=64, normalize_embeddings=True,
                show_progress_bar=False, convert_to_tensor=True,
            )
        multi_sim = torch.mm(multi_emb, occ_embeddings.T)
        _, multi_ranked = torch.topk(multi_sim, k=1, dim=1)
        multi_ranked = multi_ranked.cpu().tolist()

        for true_code, (rk,) in zip(
            [p["code"] for p in multi_test], multi_ranked
        ):
            gt_idx = code_to_idx.get(true_code)
            if gt_idx is None:
                continue
            n_multi += 1
            if rk == gt_idx:
                multi_top1 += 1

        if n_multi > 0:
            print(f"\n  多标注测试子集 Top1: {multi_top1}/{n_multi} = {multi_top1/n_multi*100:.1f}%")

    # 对比基准模型
    print("\n  [对比基准] 未微调基础模型的准确率...")
    resolved_path = resolve_existing_model_path(base_model_path, label="基础 embedding")
    base_model = SentenceTransformer(str(resolved_path), device=device)
    base_model.max_seq_length = config.max_seq_length

    with torch.no_grad():
        base_occ_emb = base_model.encode(
            occ_texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=False, convert_to_tensor=True,
        )
        base_anchor_emb = base_model.encode(
            test_anchors, batch_size=64, normalize_embeddings=True,
            show_progress_bar=False, convert_to_tensor=True,
        )
    base_sim = torch.mm(base_anchor_emb, base_occ_emb.T)
    _, base_ranked = torch.topk(base_sim, k=5, dim=1)
    base_ranked = base_ranked.cpu().tolist()

    base_topk = {1: 0, 3: 0, 5: 0}
    base_total = 0
    for true_code, ranked in zip(test_codes, base_ranked):
        gt_idx = code_to_idx.get(true_code)
        if gt_idx is None:
            continue
        base_total += 1
        for rank, pred_idx in enumerate(ranked, 1):
            if pred_idx == gt_idx:
                for k in base_topk:
                    if rank <= k:
                        base_topk[k] += 1
                break

    print(f"  {'基准 Top-1':<20} {base_topk[1]:<10} {base_topk[1]/base_total*100:.1f}%")
    print(f"  {'基准 Top-3':<20} {base_topk[3]:<10} {base_topk[3]/base_total*100:.1f}%")
    print(f"  {'基准 Top-5':<20} {base_topk[5]:<10} {base_topk[5]/base_total*100:.1f}%")

    return {
        "top1": topk_hits[1] / total * 100 if total > 0 else 0,
        "top3": topk_hits[3] / total * 100 if total > 0 else 0,
        "top5": topk_hits[5] / total * 100 if total > 0 else 0,
        "top10": topk_hits[10] / total * 100 if total > 0 else 0,
        "base_top1": base_topk[1] / base_total * 100 if base_total > 0 else 0,
        "base_top3": base_topk[3] / base_total * 100 if base_total > 0 else 0,
        "base_top5": base_topk[5] / base_total * 100 if base_total > 0 else 0,
        "test_total": total,
        "multi_top1": multi_top1 / n_multi * 100 if n_multi > 0 else None,
    }


# ── 主流程 ──────────────────────────────────────────
def main() -> None:
    """执行 v5 训练、评估与结果落盘。"""
    args = parse_args()
    config = V5Config()
    # 命令行覆盖
    config.ds_agree_oversample = args.ds_agree_oversample
    config.ds_disagree_oversample = args.ds_disagree_oversample
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    base_model_path = args.base_model_path or DEFAULT_BASE_MODEL_PATH
    output_model_name = args.output_model_name or DEFAULT_OUTPUT_MODEL_NAME
    run_label = args.run_label or output_model_name
    result_file_name = f"evaluation_{_slugify_name(run_label)}.json"

    output_model_path = os.path.join(OUTPUT_DIR, output_model_name)
    result_file_path = os.path.join(OUTPUT_DIR, result_file_name)

    print("=" * 70)
    print("v5 训练：人标全量 + DeepSeek 高置信度验证过采样")
    print(f"运行标签: {run_label}")
    print(f"基础模型: {base_model_path}")
    print(f"输出模型: {output_model_path}")
    print(f"DS 一致过采样: {config.ds_agree_oversample}x (conf >= {config.ds_agree_conf_threshold})")
    print(f"DS 不一致加入: {config.ds_disagree_oversample}x (conf >= {config.ds_disagree_conf_threshold})")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 提取训练数据
    single_pairs, multi_pairs = extract_training_pairs(config)

    if len(single_pairs) + len(multi_pairs) == 0:
        print("ERROR: 无有效训练数据！")
        return

    # Step 2: 划分 train/test
    train_examples, train_pairs, test_examples_ie, test_pairs = split_train_test(
        single_pairs, multi_pairs, config
    )

    # Step 3: 微调
    model = train_model(train_examples, config, output_model_path, base_model_path)

    # Step 4: 评估
    code_to_text = load_occupation_dict()
    results = evaluate_model(model, test_pairs, code_to_text, config, base_model_path)

    # ── 保存结果 ──
    results["config"] = {
        "ds_agree_conf_threshold": config.ds_agree_conf_threshold,
        "ds_agree_oversample": config.ds_agree_oversample,
        "ds_disagree_conf_threshold": config.ds_disagree_conf_threshold,
        "ds_disagree_oversample": config.ds_disagree_oversample,
        "ds_only_conf_threshold": config.ds_only_conf_threshold,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "train_pairs": len(train_pairs),
    }
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存至: {result_file_path}")

    # ── 最终报告 ──
    print("\n" + "=" * 70)
    print("[最终报告] v5 训练完成")
    print("=" * 70)
    print(f"""
  训练样本数: {len(train_examples)}
  测试样本数: {results['test_total']}
  基础模型: {base_model_path}
  微调后模型: {output_model_path}

  检索准确率 (v5):
    Top-1: {results['top1']:.1f}%
    Top-3: {results['top3']:.1f}%
    Top-5: {results['top5']:.1f}%
    Top-10: {results['top10']:.1f}%

  基准模型 (未微调):
    Top-1: {results['base_top1']:.1f}%
    Top-3: {results['base_top3']:.1f}%
    Top-5: {results['base_top5']:.1f}%
""")
    if results.get("multi_top1"):
        print(f"  多标注子集 Top1: {results['multi_top1']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
