#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二轮标注数据集训练 RAG 检索模型（BGE-M3 fine-tuning）。

使用第二轮人工标注数据微调 BGE-M3，提升职业细类匹配的 Top1 准确率。

流程:
    1. 从 Label-Studio JSON 提取训练正样本对
    2. 加载《中国职业大典》获取职业完整文本
    3. 划分 train/test（多标注任务作为 test）
    4. 使用 MultipleNegativesRankingLoss 微调 BGE-M3
    5. 在 test 集上评估 Top1/Top3 检索准确率

用法:
    python -m src.occupation_retrieval.train_rag_round2

前置条件:
    1. 已有 Label-Studio 标注 JSON 文件
    2. BGE 模型路径通过 config/paths.py 或环境变量 EMPLOYDATA_BGE_MODEL_PATH 配置
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.paths import get_project_paths
from .common import (
    get_runtime_device,
    get_training_output_dir,
    load_annotations_from_pg,
    load_occupation_dict_df,
    resolve_base_model_path,
    resolve_existing_model_path,
)
from .datasets import build_anchor, get_majority_choice, get_task_choices, parse_choice

_project = get_project_paths()
BASE_DIR = str(_project.project_root)

OUTPUT_DIR = str(get_training_output_dir())

DEFAULT_BASE_MODEL_PATH = resolve_base_model_path()
DEFAULT_OUTPUT_MODEL_NAME = "bge-large-round2-finetuned"
DEFAULT_RESULT_FILE_NAME = "evaluation_results.json"


@dataclass(frozen=True)
class RunSettings:
    """单次训练运行的路径与命名配置。"""

    base_model_path: str
    output_model_name: str
    output_model_path: str
    run_label: str | None
    result_file_name: str
    result_file_path: str

# ── 训练参数 ────────────────────────────────────────
@dataclass
class TrainConfig:
    """RAG Round2 微调与评估配置。"""

    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 2e-5
    max_seq_length: int = 256
    warmup_ratio: float = 0.1
    test_ratio: float = 0.15
    random_seed: int = 42
    # 是否仅用多标注任务作为测试集（更严格的评估）
    use_multi_ann_as_test: bool = True


def _slugify_name(value: str) -> str:
    """将运行标签转换为适合文件名使用的形式。"""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return cleaned or "run"


def _same_path(left: str, right: str) -> bool:
    """判断两个路径是否指向同一路径。"""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return left == right


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行 occupation_retrieval v1 检索基线训练，支持单变量底座挑战。",
    )
    parser.add_argument(
        "--base-model-path",
        help="覆盖基础 embedding 模型路径；第一轮挑战时仅允许替换这一项。",
    )
    parser.add_argument(
        "--output-model-name",
        help=(
            "输出模型目录名。默认保持历史基线目录；"
            "若切换基础模型，建议显式传入新目录名，例如 v1-bge-m3。"
        ),
    )
    parser.add_argument(
        "--run-label",
        help=(
            "运行标签，用于结果文件命名。"
            "例如 v1-bge-m3 会生成 evaluation_v1_bge_m3.json。"
        ),
    )
    return parser.parse_args()


def build_run_settings(args: argparse.Namespace) -> RunSettings:
    """根据命令行参数构造单次运行设置。"""
    base_model_path = args.base_model_path or DEFAULT_BASE_MODEL_PATH
    output_model_name = args.output_model_name or DEFAULT_OUTPUT_MODEL_NAME

    if args.base_model_path and not args.output_model_name and not _same_path(
        base_model_path,
        DEFAULT_BASE_MODEL_PATH,
    ):
        raise SystemExit(
            "使用新的 --base-model-path 时，必须同时传入 --output-model-name，"
            "以避免覆盖已冻结的基线产物。"
        )

    derived_run_label = args.run_label
    if not derived_run_label and output_model_name != DEFAULT_OUTPUT_MODEL_NAME:
        derived_run_label = output_model_name

    if derived_run_label:
        result_file_name = f"evaluation_{_slugify_name(derived_run_label)}.json"
    else:
        result_file_name = DEFAULT_RESULT_FILE_NAME

    output_model_path = os.path.join(OUTPUT_DIR, output_model_name)
    result_file_path = os.path.join(OUTPUT_DIR, result_file_name)
    return RunSettings(
        base_model_path=base_model_path,
        output_model_name=output_model_name,
        output_model_path=output_model_path,
        run_label=derived_run_label,
        result_file_name=result_file_name,
        result_file_path=result_file_path,
    )


def load_occupation_dict(dict_path: str) -> dict[str, str]:
    """加载《中国职业大典》，返回 {code: full_text} 映射。"""
    df = load_occupation_dict_df()

    # 自动识别列名
    col_map = {}
    for target in ["code", "title", "desc", "tasks"]:
        for col in df.columns:
            if target in str(col).lower() or (
                target == "code" and "代码" in str(col)
            ):
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

        # 构建检索正样本文本（与现有 D4_T2match.py 格式对齐）
        parts = [title]
        if desc and desc.lower() != "nan":
            parts.append(f"定义：{desc}")
        if tasks and tasks.lower() != "nan":
            parts.append(f"任务：{tasks}")
        code_to_text[code] = "。".join(parts)

    return code_to_text


# ── 1. 提取训练数据 ─────────────────────────────────
def extract_training_pairs(
    annotation_file: str,
    dict_file: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    从标注 JSON 提取训练正样本对。

    Returns:
        single_pairs: 单标注任务的正样本 [(anchor, positive, code, task_id), ...]
        multi_pairs:  多标注任务（多数意见）的正样本
    """
    print("=" * 70)
    print("[Step 1] 提取训练正样本对")
    print("=" * 70)

    # 加载数据
    raw_data = load_annotations_from_pg()
    print(f"  加载标注数据: {len(raw_data)} 条任务")

    code_to_text = load_occupation_dict(dict_file)
    print(f"  加载职业大典: {len(code_to_text)} 个职业细类")

    # 解析所有任务
    single_pairs: list[dict[str, Any]] = []
    multi_pairs: list[dict[str, Any]] = []
    skipped_none = 0
    skipped_no_code = 0
    skipped_no_text = 0

    for item in raw_data:
        task_id = item["task_id"]
        recruitment_record_id = item["recruitment_record_id"]
        data = item["data"]
        job_title = str(data.get("job_title", "")).strip()
        job_reqs = str(data.get("job_requirements_clean", "")).strip()

        if not job_reqs:
            continue

        # 构建 anchor 文本
        anchor = build_anchor(job_title, job_reqs)

        # 获取所有标注的选择
        choices = get_task_choices(item, include_none=False)

        if not choices:
            skipped_none += 1
            continue

        # 确定参考答案（多数意见或单标注）
        if len(item["annotations"]) >= 2:
            ref_choice, count = get_majority_choice(choices, require_strict=False)
            if not ref_choice or count <= len(item["annotations"]) / 2:
                # 无明显多数，跳过
                continue
            # 多标注任务
            is_multi = True
        else:
            ref_choice = choices[0]
            is_multi = False

        # 查候选对应的职业代码
        code_key = f"candidate_{ref_choice.lower()}_code"
        code = str(data.get(code_key, "")).strip()

        if not code:
            skipped_no_code += 1
            continue

        # 查职业大典获取完整文本
        positive_text = code_to_text.get(code)
        if not positive_text:
            skipped_no_text += 1
            continue

        pair = {
            "task_id": task_id,
            "recruitment_record_id": recruitment_record_id,
            "anchor": anchor,
            "positive": positive_text,
            "code": code,
            "job_title": job_title,
            "ref_choice": ref_choice,
            "n_annotators": len(item["annotations"]),
        }

        if is_multi:
            multi_pairs.append(pair)
        else:
            single_pairs.append(pair)

    print(f"\n  单标注正样本: {len(single_pairs)} 对")
    print(f"  多标注正样本: {len(multi_pairs)} 对")
    print(f"  合计: {len(single_pairs) + len(multi_pairs)} 对")
    print(f"\n  跳过原因:")
    print(f"    选了 NONE: {skipped_none}")
    print(f"    查不到代码: {skipped_no_code}")
    print(f"    代码不在大典中: {skipped_no_text}")

    # 统计职业分布
    all_codes = [p["code"] for p in single_pairs] + [p["code"] for p in multi_pairs]
    code_counts = Counter(all_codes)
    print(f"\n  覆盖职业细类数: {len(code_counts)}")
    print(f"  最频繁职业 Top5:")
    for code, count in code_counts.most_common(5):
        print(f"    {code}: {count} 条")

    return single_pairs, multi_pairs


# ── 2. 划分训练/测试集 ──────────────────────────────
def split_train_test(
    single_pairs: list[dict[str, Any]],
    multi_pairs: list[dict[str, Any]],
    config: TrainConfig,
) -> tuple[
    list[Any],
    list[dict[str, Any]],
    list[Any],
    list[dict[str, Any]],
]:
    """
    划分训练集和测试集。

    策略: 多标注任务全部作为测试集（高质量评估基准），
          单标注任务按比例划分。

    Returns:
        train_examples, train_metadata, test_examples, test_metadata
    """
    print("\n" + "=" * 70)
    print("[Step 2] 划分训练/测试集")
    print("=" * 70)

    from sentence_transformers import InputExample

    random.seed(config.random_seed)

    if config.use_multi_ann_as_test:
        # 多标注全部 → test
        test_pairs = list(multi_pairs)
        # 从单标注中再分出一些作为 test
        random.shuffle(single_pairs)
        n_extra_test = int(len(single_pairs) * config.test_ratio)
        test_pairs.extend(single_pairs[:n_extra_test])
        train_pairs = single_pairs[n_extra_test:]
        print(f"  策略: 多标注({len(multi_pairs)}) + 单标注抽样({n_extra_test}) → test")
    else:
        # 全部随机划分
        all_pairs = single_pairs + multi_pairs
        random.shuffle(all_pairs)
        n_test = int(len(all_pairs) * config.test_ratio)
        test_pairs = all_pairs[:n_test]
        train_pairs = all_pairs[n_test:]

    # 构建 InputExample
    def to_example(pair: dict[str, Any]) -> InputExample:
        """将训练对转换为 SentenceTransformer 输入样本。"""
        return InputExample(texts=[pair["anchor"], pair["positive"]])

    train_examples = [to_example(p) for p in train_pairs]
    test_examples_ie = [to_example(p) for p in test_pairs]  # not used for training

    print(f"  训练集: {len(train_examples)} 对")
    print(f"  测试集: {len(test_pairs)} 对")

    # 测试集职业覆盖
    test_codes = set(p["code"] for p in test_pairs)
    train_codes = set(p["code"] for p in train_pairs)
    print(f"  测试集覆盖职业: {len(test_codes)} 个细类")
    print(f"  仅在测试集的职业: {len(test_codes - train_codes)} 个")

    return train_examples, train_pairs, test_examples_ie, test_pairs


# ── 3. 微调模型 ─────────────────────────────────────
def train_model(
    train_examples: list[Any],
    config: TrainConfig,
    run_settings: RunSettings,
) -> Any:
    """使用 MultipleNegativesRankingLoss 微调当前基础模型。"""
    print("\n" + "=" * 70)
    print("[Step 3] 微调检索模型")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    device = get_runtime_device()
    print(f"  设备: {device}")
    print(f"  基础模型: {run_settings.base_model_path}")

    # 加载模型
    base_model_path = resolve_existing_model_path(
        run_settings.base_model_path,
        label="基础 embedding",
    )
    model = SentenceTransformer(str(base_model_path), device=device)
    model.max_seq_length = config.max_seq_length
    print(f"  最大序列长度: {config.max_seq_length}")
    print(f"  嵌入维度: {model.get_sentence_embedding_dimension()}")

    # DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
    )

    # 损失函数: MultipleNegativesRankingLoss
    # 每个 batch 内，anchor 与对应的 positive 是正样本对，
    # 其他 positive 自动成为负样本 (in-batch negatives)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 训练
    warmup_steps = int(len(train_dataloader) * config.epochs * config.warmup_ratio)

    print(f"\n  训练配置:")
    print(f"    Batch Size: {config.batch_size}")
    print(f"    Epochs: {config.epochs}")
    print(f"    Learning Rate: {config.learning_rate}")
    print(f"    Warmup Steps: {warmup_steps}")
    print(f"    Batches per epoch: {len(train_dataloader)}")

    os.makedirs(run_settings.output_model_path, exist_ok=True)

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=run_settings.output_model_path,
        show_progress_bar=True,
        save_best_model=True,
        use_amp=True,               # 混合精度加速
        evaluator=None,             # 禁用评估器避免每epoch的慢速评估
    )
    elapsed = time.time() - t0
    print(f"\n  训练完成，耗时: {elapsed/60:.1f} 分钟")
    print(f"  模型保存至: {run_settings.output_model_path}")

    return model


# ── 4. 评估 ─────────────────────────────────────────
def evaluate_model(
    model: Any,
    test_pairs: list[dict[str, Any]],
    code_to_text: dict[str, str],
    config: TrainConfig,
    run_settings: RunSettings,
) -> dict[str, Any]:
    """在测试集上评估检索准确率（Top1/Top3/Top5）。"""
    print("\n" + "=" * 70)
    print("[Step 4] 评估检索准确率")
    print("=" * 70)

    import torch
    from sentence_transformers import SentenceTransformer

    device = get_runtime_device()

    # 构建职业大典向量库
    codes = sorted(code_to_text.keys())
    occ_texts = [code_to_text[c] for c in codes]
    code_to_idx = {c: i for i, c in enumerate(codes)}

    print(f"  职业细类总数: {len(codes)}")
    print(f"  测试样本数: {len(test_pairs)}")

    # 编码所有职业文本
    with torch.no_grad():
        occ_embeddings = model.encode(
            occ_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    # 编码所有测试 anchor
    test_anchors = [p["anchor"] for p in test_pairs]
    test_codes = [p["code"] for p in test_pairs]

    with torch.no_grad():
        anchor_embeddings = model.encode(
            test_anchors,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    # 计算余弦相似度并排序
    similarity = torch.mm(anchor_embeddings, occ_embeddings.T)
    _, ranked_indices = torch.topk(similarity, k=5, dim=1)
    ranked_indices = ranked_indices.cpu().tolist()

    # 统计命中率
    topk_hits: dict[int, int] = {1: 0, 2: 0, 3: 0, 5: 0}
    total = len(test_codes)
    per_class_correct = defaultdict(lambda: {"total": 0, "hit": 0})

    for i, (true_code, ranked) in enumerate(zip(test_codes, ranked_indices)):
        gt_idx = code_to_idx.get(true_code)
        if gt_idx is None:
            total -= 1
            continue

        per_class_correct[true_code]["total"] += 1
        for rank, pred_idx in enumerate(ranked, 1):
            if pred_idx == gt_idx:
                for k in topk_hits:
                    if rank <= k:
                        topk_hits[k] += 1
                per_class_correct[true_code]["hit"] += 1
                break

    # 输出结果
    print(f"\n  检索准确率 (N={total}):")
    print(f"  {'指标':<20} {'命中数':<10} {'准确率':<10}")
    print(f"  {'-'*40}")
    for k in [1, 3, 5]:
        hit = topk_hits[k]
        pct = hit / total * 100 if total > 0 else 0
        marker = " <-- 对标论文 ~78%" if k == 1 else ""
        print(f"  Top-{k:<18} {hit:<10} {pct:.1f}%{marker}")

    # 按测试集来源分组（多标注 vs 单标注）
    multi_test = [p for p in test_pairs if p["n_annotators"] >= 2]
    if multi_test:
        multi_anchors = [p["anchor"] for p in multi_test]
        multi_codes_set = [p["code"] for p in multi_test]
        with torch.no_grad():
            multi_emb = model.encode(
                multi_anchors, batch_size=64, normalize_embeddings=True,
                show_progress_bar=False, convert_to_tensor=True,
            )
        multi_sim = torch.mm(multi_emb, occ_embeddings.T)
        _, multi_ranked = torch.topk(multi_sim, k=3, dim=1)
        multi_ranked = multi_ranked.cpu().tolist()

        multi_top1 = 0
        multi_top3 = 0
        for true_code, ranked in zip(multi_codes_set, multi_ranked):
            gt_idx = code_to_idx.get(true_code)
            if gt_idx is None:
                continue
            if ranked[0] == gt_idx:
                multi_top1 += 1
                multi_top3 += 1
            elif gt_idx in ranked[:3]:
                multi_top3 += 1

        n_multi = len(multi_test)
        if n_multi > 0:
            print(f"\n  多标注测试子集 (N={n_multi}, 高质量基准):")
            print(f"    Top1: {multi_top1}/{n_multi} = {multi_top1/n_multi*100:.1f}%")
            print(f"    Top3: {multi_top3}/{n_multi} = {multi_top3/n_multi*100:.1f}%")

    # 对比基准模型（未微调）
    print(f"\n  [对比基准] 未微调基础模型的准确率...")
    base_model_path = resolve_existing_model_path(
        run_settings.base_model_path,
        label="基础 embedding",
    )
    base_model = SentenceTransformer(str(base_model_path), device=device)
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

    # 提升幅度
    improvement = (topk_hits[1]/total - base_topk[1]/base_total) * 100
    print(f"\n  Top1 提升: +{improvement:.1f}pp")

    return {
        "top1": topk_hits[1] / total * 100 if total > 0 else 0,
        "top3": topk_hits[3] / total * 100 if total > 0 else 0,
        "top5": topk_hits[5] / total * 100 if total > 0 else 0,
        "base_top1": base_topk[1] / base_total * 100 if base_total > 0 else 0,
        "base_top3": base_topk[3] / base_total * 100 if base_total > 0 else 0,
        "base_top5": base_topk[5] / base_total * 100 if base_total > 0 else 0,
        "test_total": total,
        "multi_top1": multi_top1 / n_multi * 100 if (multi_test and n_multi > 0) else None,
    }


# ── 主流程 ──────────────────────────────────────────
def main() -> None:
    """执行 Round2 数据训练、评估与结果落盘。"""
    args = parse_args()
    config = TrainConfig()
    run_settings = build_run_settings(args)

    print("=" * 70)
    print("第二轮标注数据集 → RAG 检索模型训练")
    if run_settings.run_label:
        print(f"运行标签: {run_settings.run_label}")
    print(f"基础模型: {run_settings.base_model_path}")
    print(f"输出模型目录: {run_settings.output_model_path}")
    print(f"评估结果文件: {run_settings.result_file_path}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 提取训练数据
    single_pairs, multi_pairs = extract_training_pairs("", "")

    if len(single_pairs) + len(multi_pairs) == 0:
        print("ERROR: 无有效训练数据！")
        return

    # Step 2: 划分 train/test
    train_examples, train_pairs, test_examples_ie, test_pairs = split_train_test(
        single_pairs, multi_pairs, config
    )

    # Step 3: 微调
    model = train_model(train_examples, config, run_settings)

    # Step 4: 评估
    code_to_text = load_occupation_dict("")
    results = evaluate_model(model, test_pairs, code_to_text, config, run_settings)

    # ── 保存结果 ──
    with open(run_settings.result_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存至: {run_settings.result_file_path}")

    # ── 最终报告 ──
    print("\n" + "=" * 70)
    print("[最终报告]")
    print("=" * 70)
    print(f"""
  训练样本数: {len(train_examples)}
  测试样本数: {results['test_total']}
  基础模型: {run_settings.base_model_path}
  微调后模型: {run_settings.output_model_path}

  检索准确率:
    微调前 Top1: {results['base_top1']:.1f}%
    微调后 Top1: {results['top1']:.1f}%
    微调前 Top3: {results['base_top3']:.1f}%
    微调后 Top3: {results['top3']:.1f}%
    微调前 Top5: {results['base_top5']:.1f}%
    微调后 Top5: {results['top5']:.1f}%
""")
    if results.get("multi_top1"):
        print(f"  多标注子集 Top1: {results['multi_top1']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()


