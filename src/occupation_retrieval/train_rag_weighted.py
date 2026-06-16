#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案A：置信加权训练（RAG retrieval model fine-tuning with confidence-weighted oversampling）。

用多维度质量分给每条训练样本赋权，通过 oversampling 实现加权
MultipleNegativesRankingLoss：

    S级（极高置信）— 权重 1.0，重复 10x
    A级（高置信）  — 权重 0.7，重复 7x
    B级（中等）    — 权重 0.3，重复 3x
    C级（低置信）  — 权重 0.1，重复 1x
    D级（疑似错误）— 丢弃

用法:
    python -m src.occupation_retrieval.train_rag_weighted

前置条件:
    1. 已有 Label-Studio 标注 JSON 文件
    2. 已有《中国职业大典》xlsx 词典
    3. BGE 模型路径通过 config/paths.py 或环境变量 EMPLOYDATA_BGE_MODEL_PATH 配置
"""

from __future__ import annotations

import json
import os
import time
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    safe_empty_cuda_cache,
)

_project = get_project_paths()
BASE_DIR = str(_project.project_root)
OUTPUT_DIR = str(get_training_output_dir())
BASE_MODEL_PATH = resolve_base_model_path()
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "bge-large-round2-finetuned-weighted")

# ── 大类关键词 ──
MAJOR_CLASS_KEYWORDS = {
    "专业技术人员": ["技术","工程","开发","设计","分析","研发","算法","架构","测试",
                    "运维","前端","后端","数据","软件","硬件","网络","安全","系统",
                    "编程","代码","Java","Python","医学","医师","护士","药剂","检验",
                    "律师","会计","审计","教师","教授","翻译","记者","编辑"],
    "办事人员和有关人员": ["行政","人事","财务","出纳","法务","合规","秘书","助理",
                        "文员","前台","档案","后勤","统计"],
    "社会生产服务和生活服务人员": ["服务","销售","客服","运营","配送","物流","餐饮",
                              "司机","保安","保洁","快递","导游","美容","美发",
                              "厨师","房地产","保险","中介","店员","收银","导购"],
    "生产制造及有关人员": ["生产","制造","加工","装配","质检","操作","车工","钳工",
                        "焊工","电工","维修","安装","调试"],
    "企事业单位负责人": ["经理","总监","主管","总裁","CEO","负责人","主任","部长",
                       "校长","董事长","总经理","项目经理"],
}

@dataclass
class Config:
    """加权 oversampling 训练配置。"""

    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    max_seq_length: int = 256
    warmup_ratio: float = 0.1
    random_seed: int = 42
    # oversample multipliers
    oversample: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """初始化每个质量层级对应的过采样倍数。"""
        self.oversample = {"S": 10, "A": 7, "B": 3, "C": 1, "D": 0}


def parse_choice(ann: dict[str, Any]) -> str | None:
    """从单条人工标注中提取规范化选择。"""
    for r in ann.get("result", []):
        if r["from_name"] == "best_candidate_choice":
            choices = r["value"].get("choices", [])
            if not choices:
                return None
            raw = choices[0]
            if len(raw) >= 2 and raw[-1] in "ABCDE":
                return raw[-1]
            if "不" in raw:
                return "NONE"
    return None


def load_dict() -> tuple[dict[str, str], dict[str, str]]:
    """加载职业文本和职业大类映射。"""
    df = load_occupation_dict_df()
    c2text: dict[str, str] = {}
    c2major: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        title = str(row["title"]).strip()
        if not code or not title:
            continue
        parts = [title]
        for key, prefix in [("desc", "定义："), ("tasks", "任务：")]:
            value = str(row.get(key, ""))
            if value and value != "nan":
                parts.append(f"{prefix}{value}")
        c2text[code] = "。".join(parts)
        c2major[code] = str(row.get("大类", "")).strip()
    return c2text, c2major


def guess_major_class(job_title: str) -> str | None:
    """根据岗位名称关键词猜测职业大类。"""
    if not job_title:
        return None
    scores: dict[str, int] = {}
    for m, info in MAJOR_CLASS_KEYWORDS.items():
        score = sum(1 for kw in info if kw in job_title)
        if score > 0:
            scores[m] = score
    return max(scores, key=scores.get) if scores else None


def class_match(title_guess: str | None, occ_major: str | None) -> float:
    """判断岗位名称猜测的大类与职业大典大类是否匹配。"""
    if not title_guess or not occ_major:
        return 0.5
    for major_class in MAJOR_CLASS_KEYWORDS:
        if major_class == title_guess:
            return 1.0 if occ_major == major_class else 0.0
    return 0.5


def compute_tier(record: dict[str, Any]) -> str:
    """复用多维信号规则，为样本打质量层级。"""
    score = 0
    if record["n_annotators"] >= 2 and record["has_majority"]:
        score += 3 if record["pairwise"] >= 0.8 else (2 if record["pairwise"] >= 0.5 else 1)
    elif record["n_annotators"] >= 2:
        score += 1
    if record["ds_agrees"] is True:
        score += 3 if (record["ds_conf"] and record["ds_conf"] >= 0.9) else 2
    elif record["ds_agrees"] is False:
        score -= 2
    if record["sem_rank"] is not None and record["sem_rank"] > 0:
        if record["sem_rank"] <= 5:
            score += 3
        elif record["sem_rank"] <= 20:
            score += 2
        elif record["sem_rank"] <= 50:
            score += 1
        else:
            score -= 1
    if record["kw_match"] == 1.0:
        score += 2
    elif record["kw_match"] == 0.0:
        score -= 1
    if record["ann_quality"] >= 0.7:
        score += 1
    elif record["ann_quality"] < 0.5:
        score -= 1
    if record["none_rate"] > 0.3:
        score -= 1
    if record["top1_pick"]:
        score += 1

    if score >= 9:
        return "S"
    if score >= 6:
        return "A"
    if score >= 3:
        return "B"
    if score >= 0:
        return "C"
    return "D"


def main() -> None:
    """执行基于多维质量层级的加权微调方案。"""
    config = Config()
    print("=" * 60)
    print("方案A：置信加权训练 (Confidence-Weighted)")
    print(f"  oversample: {config.oversample}")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load ──
    print("\n[1] Loading...")
    raw_data = load_annotations_from_pg()
    ds_records = load_deepseek_records()
    c2text, c2major = load_dict()

    # ── Compute annotator quality first ──
    ann_stats = defaultdict(lambda: {"agree":0,"total":0})
    for item in raw_data:
        anns = item["annotations"]
        if len(anns) < 2: continue
        choices = [(a["completed_by"], parse_choice(a)) for a in anns]
        valid = [(aid, c) for aid, c in choices if c and c != "NONE"]
        if len(valid) < 2: continue
        ctr = Counter([c for _,c in valid])
        majority, _ = ctr.most_common(1)[0]
        for aid, c in valid:
            ann_stats[aid]["total"] += 1
            if c == majority: ann_stats[aid]["agree"] += 1
    ann_quality = {
        aid: stats["agree"] / stats["total"] if stats["total"] > 0 else 0.5
        for aid, stats in ann_stats.items()
    }

    # ── Load BGE for semantic rank ──
    print("\n[2] Encoding occupations for semantic ranking...")
    occ_codes = sorted(c2text.keys()); occ_texts = [c2text[c] for c in occ_codes]
    model = SentenceTransformer(BASE_MODEL_PATH, device=get_runtime_device()); model.max_seq_length = 256
    with torch.no_grad():
        occ_emb = model.encode(occ_texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_tensor=True)

    # ── Score each sample ──
    print("\n[3] Computing quality tiers...")
    scored_pairs: list[dict[str, Any]] = []
    for item in raw_data:
        tid = item["task_id"]; recruitment_record_id = item["recruitment_record_id"]; data = item["data"]; anns = item["annotations"]
        jt = str(data.get("job_title","")).strip()
        jr = str(data.get("job_requirements_clean","")).strip()
        if not jr: continue
        anchor = f"{jt} {jr}"
        n_ann = len(anns)

        choices = [c for c in [parse_choice(a) for a in anns] if c and c != "NONE"]
        if not choices: continue

        if n_ann >= 2:
            ctr = Counter(choices); majority, cnt = ctr.most_common(1)[0]
            has_maj = cnt > n_ann/2
            pw = sum(1 for i in range(len(choices)) for j in range(i+1,len(choices))
                     if choices[i]==choices[j]) / (len(choices)*(len(choices)-1)//2) if len(choices)>1 else 1.0
        else:
            majority = choices[0]; has_maj = False; pw = 1.0

        hum_code = str(data.get(f"candidate_{majority.lower()}_code","")).strip()
        if not hum_code or hum_code not in c2text: continue

        # DS
        ds = ds_records.get(tid)
        ds_agrees = (ds["deepseek_choice"]==majority) if ds else None
        ds_conf = ds.get("deepseek_confidence") if ds else None

        # Semantic rank
        sem_rank = None
        if hum_code in occ_codes:
            with torch.no_grad():
                anc = model.encode([anchor], batch_size=1, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_tensor=True)
                sims = torch.mm(anc, occ_emb.T).squeeze(0)
                sorted_idx = torch.argsort(sims, descending=True).cpu().tolist()
                sem_rank = sorted_idx.index(occ_codes.index(hum_code)) + 1

        kw_match = class_match(guess_major_class(jt), c2major.get(hum_code,""))
        none_count = sum(1 for a in anns if parse_choice(a)=="NONE")
        top1 = any("top1" in str(data.get(f"candidate_{c}_source",""))
                   and majority==c.upper() for c in "abcde")

        aa = [ann_quality.get(a["completed_by"],0.5) for a in anns
              if parse_choice(a) and parse_choice(a)!="NONE"]
        avg_aq = np.mean(aa) if aa else 0.5

        r = dict(task_id=tid, n_annotators=n_ann, has_majority=has_maj,
                pairwise=pw, ds_agrees=ds_agrees, ds_conf=ds_conf,
                sem_rank=sem_rank, kw_match=kw_match, ann_quality=avg_aq,
                none_rate=none_count/n_ann, top1_pick=top1)
        tier = compute_tier(r)

        scored_pairs.append(dict(
            task_id=tid, recruitment_record_id=recruitment_record_id, anchor=anchor,
            positive=c2text[hum_code], code=hum_code,
            tier=tier, score=r,
        ))

    # ── Split ──
    print("\n[4] Split train/test & oversample...")
    tier_counts = Counter(p["tier"] for p in scored_pairs)
    for t in ["S","A","B","C","D"]:
        print(f"  {t}: {tier_counts[t]}")

    random.seed(config.random_seed)
    random.shuffle(scored_pairs)

    # Test: 多标注 + DS一致 保留作为干净测试
    test_set = [p for p in scored_pairs
                if p["tier"] in ("S","A") and p["score"]["n_annotators"]>=2 and p["score"]["ds_agrees"]]
    # 补充随机测试样本
    remaining = [p for p in scored_pairs if p not in test_set]
    random.shuffle(remaining)
    test_set += remaining[:2500]
    test_ids = {p["task_id"] for p in test_set}

    train_pool = [p for p in scored_pairs if p["task_id"] not in test_ids]

    # Oversample
    train_examples = []
    tier_train_counts = defaultdict(int)
    for p in train_pool:
        mult = config.oversample.get(p["tier"], 1)
        for _ in range(mult):
            train_examples.append(InputExample(texts=[p["anchor"], p["positive"]]))
        tier_train_counts[p["tier"]] += 1

    print(f"\n  Original train pool: {len(train_pool)}")
    print(f"  After oversample: {len(train_examples)}")
    for t in ["S","A","B","C","D"]:
        print(f"    {t}: {tier_train_counts[t]} unique -> {tier_train_counts[t]*config.oversample.get(t,0)} oversampled")
    print(f"  Test: {len(test_set)}")

    del model; safe_empty_cuda_cache()

    # ── Train ──
    print(f"\n[5] Training...")
    model = SentenceTransformer(BASE_MODEL_PATH, device=get_runtime_device()); model.max_seq_length = 256
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_dataloader) * config.epochs * config.warmup_ratio)
    print(f"  Batches/epoch: {len(train_dataloader)}, Total: {len(train_dataloader)*config.epochs}")

    t0 = time.time()
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.epochs, warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=OUTPUT_MODEL_PATH, show_progress_bar=True,
        save_best_model=True, use_amp=True, evaluator=None,
    )
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")

    # ── Evaluate ──
    print(f"\n[6] Evaluation...")
    test_anchors = [p["anchor"] for p in test_set]
    with torch.no_grad():
        occ_emb2 = model.encode(occ_texts, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=True, convert_to_tensor=True)

    topk = {1:0,3:0,5:0,10:0}; total = 0
    CHUNK = 1000
    for start in range(0, len(test_anchors), CHUNK):
        end = min(start+CHUNK, len(test_anchors))
        with torch.no_grad():
            ce = model.encode(test_anchors[start:end], batch_size=64, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_tensor=True)
            cs = torch.mm(ce, occ_emb2.T)
            _, cr = torch.topk(cs, k=10, dim=1)
        cr = cr.cpu().tolist()
        for i, rk in enumerate(cr):
            gt_code = test_set[start+i]["code"]
            if gt_code not in occ_codes: continue
            gt = occ_codes.index(gt_code)
            total += 1
            for rank, pred in enumerate(rk, 1):
                if pred == gt:
                    for k in topk:
                        if rank <= k: topk[k] += 1
                    break
        print(f"  Eval: {end}/{len(test_anchors)}")

    print(f"\n  Test N={total}")
    for k in [1,3,5,10]:
        print(f"  Top-{k:>2}: {topk[k]:>5}/{total} = {topk[k]/total*100:.1f}%")

    # ── Compare v1 ──
    print(f"\n[7] Comparing with v1...")
    del model; safe_empty_cuda_cache()
    v1 = SentenceTransformer(resolve_model_dir("bge-large-round2-finetuned"),
                             device=get_runtime_device()); v1.max_seq_length = 256
    v1tk = {1:0,3:0,5:0}; v1t = 0
    with torch.no_grad():
        v1occ = v1.encode(occ_texts, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_tensor=True)
    for start in range(0, len(test_anchors), CHUNK):
        end = min(start+CHUNK, len(test_anchors))
        with torch.no_grad():
            ce = v1.encode(test_anchors[start:end], batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_tensor=True)
            cs = torch.mm(ce, v1occ.T)
            _, cr = torch.topk(cs, k=5, dim=1)
        cr = cr.cpu().tolist()
        for i, rk in enumerate(cr):
            gt_code = test_set[start+i]["code"]
            if gt_code not in occ_codes: continue
            gt = occ_codes.index(gt_code)
            v1t += 1
            for rank, pred in enumerate(rk, 1):
                if pred == gt:
                    for k in v1tk:
                        if rank <= k: v1tk[k] += 1
                    break

    print(f"  {'':<12} {'v1':<12} {'weighted':<12}")
    for k in [1,3,5]:
        print(f"  Top-{k:<9} {v1tk[k]/v1t*100:.1f}%       {topk[k]/total*100:.1f}%")

    # Save
    results = {f"top{k}": topk[k]/total*100 for k in [1,3,5,10]}
    results.update({f"v1_top{k}": v1tk[k]/v1t*100 for k in [1,3,5]})
    results["tiers"] = dict(tier_counts)
    with open(os.path.join(OUTPUT_DIR, "evaluation_weighted.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone! Model: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()

