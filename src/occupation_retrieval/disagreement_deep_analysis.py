#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ occupation_retrieval 2026-06-07
分歧深度分析：挖掘标注数据中的错误模式。

按标注分歧来划分正负样本，从以下维度寻找判别信号:
    1. 标注员间分歧程度与语义相似度的关联
    2. DeepSeek 预测与人类标注的偏差模式
    3. 职业大类关键词与细类匹配的冲突
    4. 候选排名（rank）对分歧的影响

用法:
    python -m src.occupation_retrieval.disagreement_deep_analysis

前置条件:
    BGE 模型路径通过 config/paths.py 或环境变量 EMPLOYDATA_BGE_MODEL_PATH 配置。
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.paths import get_project_paths
from .common import (
    get_occupation_retrieval_output_dir,
    get_runtime_device,
    load_annotations_from_pg,
    load_deepseek_records,
    load_occupation_dict_df,
    resolve_base_model_path,
)
from .datasets import parse_choice

_project = get_project_paths()
BASE_DIR = str(_project.project_root)
OUTPUT_FILE = os.path.join(get_occupation_retrieval_output_dir(), "disagreement_analysis.txt")
MODEL_PATH = resolve_base_model_path()


def load_dict() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """加载职业大典并构建分析所需的多种映射。"""
    df = load_occupation_dict_df()
    c2text: dict[str, str] = {}
    c2title: dict[str, str] = {}
    c2major: dict[str, str] = {}
    c2subclass: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        title = str(row["title"]).strip()
        desc = str(row.get("desc", "")).strip()
        tasks = str(row.get("tasks", "")).strip()
        major = str(row.get("大类", "")).strip()
        subclass = "-".join(code.split("-")[:3]) if code else ""
        if not code or not title:
            continue
        c2title[code] = title
        c2major[code] = major
        c2subclass[code] = subclass
        parts = [title]
        if desc and desc.lower() != "nan":
            parts.append(f"定义：{desc}")
        if tasks and tasks.lower() != "nan":
            parts.append(f"任务：{tasks}")
        c2text[code] = "。".join(parts)
    return c2text, c2title, c2major, c2subclass


def hierarchy_distance(
    code_a: Optional[str],
    code_b: Optional[str],
    c2subclass: dict[str, str],
) -> int:
    """计算两个职业代码的层级距离。

    Args:
        code_a: 职业代码 A。
        code_b: 职业代码 B。
        c2subclass: 代码到小类层级编码的映射。

    Returns:
        0 表示同细类，1 表示同小类，2 表示同中类，3 表示同大类，4 表示完全不同。
    """
    if not code_a or not code_b:
        return 4
    if code_a == code_b:
        return 0
    sa = c2subclass.get(code_a, "")
    sb = c2subclass.get(code_b, "")
    if sa == sb:
        return 1
    if sa[:2] == sb[:2]:
        return 2
    if sa[:1] == sb[:1]:
        return 3
    return 4


def main() -> None:
    """执行人类与 DeepSeek 分歧样本的深度误差分析。"""
    print("Loading...")
    raw_data = load_annotations_from_pg()
    ds_records = load_deepseek_records()
    c2text, c2title, c2major, c2subclass = load_dict()

    device = get_runtime_device()
    model = SentenceTransformer(MODEL_PATH, device=device)
    model.max_seq_length = 256

    # 预编码全部职业文本，后续每条任务只需要编码 anchor。
    occ_codes = sorted(c2text.keys())
    occ_texts = [c2text[c] for c in occ_codes]
    with torch.no_grad():
        occ_emb = model.encode(
            occ_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    # 逐任务计算分歧分析特征。
    records: list[dict[str, Any]] = []
    for item in raw_data:
        tid = item["task_id"]
        recruitment_record_id = item["recruitment_record_id"]
        data = item["data"]
        anns = item["annotations"]
        n_ann = len(anns)
        jt = str(data.get("job_title", "")).strip()
        jr = str(data.get("job_requirements_clean", "")).strip()
        if not jr:
            continue
        anchor = f"{jt} {jr}"

        choices = [parse_choice(a) for a in anns]
        valid = [c for c in choices if c and c != "NONE"]
        if not valid:
            continue

        if n_ann >= 2:
            counter = Counter(valid)
            hum_choice, cnt = counter.most_common(1)[0]
            has_majority = cnt > len(anns) / 2
            pairwise = sum(
                1
                for i in range(len(valid))
                for j in range(i + 1, len(valid))
                if valid[i] == valid[j]
            )
            pairwise = (
                pairwise / (len(valid) * (len(valid) - 1) // 2)
                if len(valid) > 1
                else 1.0
            )
        else:
            hum_choice = valid[0]
            has_majority = False
            pairwise = 1.0

        hum_code = str(data.get(f"candidate_{hum_choice.lower()}_code", "")).strip()
        hum_title = c2title.get(hum_code, "")

        ds = ds_records.get(tid)
        ds_choice = ds["deepseek_choice"] if ds else None
        ds_conf = ds.get("deepseek_confidence") if ds else None
        ds_code = (
            str(data.get(f"candidate_{ds_choice.lower()}_code", ""))
            if ds and ds_choice and ds_choice != "NONE"
            else None
        )
        ds_title = c2title.get(ds_code, "") if ds_code else ""

        # 语义排名用于衡量“人类所选职业”在全量检索中的相对位置。
        sem_rank = None
        sorted_idx: list[int] | None = None
        if hum_code in occ_codes:
            with torch.no_grad():
                anc_emb = model.encode(
                    [anchor],
                    batch_size=1,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                )
                sims = torch.mm(anc_emb, occ_emb.T).squeeze(0)
                sorted_idx = torch.argsort(sims, descending=True).cpu().tolist()
                target_idx = occ_codes.index(hum_code)
                sem_rank = sorted_idx.index(target_idx) + 1

        # 对比 DeepSeek 所选职业的语义排名，帮助判断谁更接近检索模型。
        ds_sem_rank = None
        if ds_code and ds_code in occ_codes:
            target_idx = occ_codes.index(ds_code)
            if sorted_idx is None:
                with torch.no_grad():
                    anc_emb = model.encode(
                        [anchor],
                        batch_size=1,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                    )
                    sims = torch.mm(anc_emb, occ_emb.T).squeeze(0)
                    sorted_idx = torch.argsort(sims, descending=True).cpu().tolist()
            ds_sem_rank = sorted_idx.index(target_idx) + 1

        # 层级距离体现“分歧”是细粒度误差还是跨大类误差。
        hdist = hierarchy_distance(hum_code, ds_code, c2subclass) if ds_code else None

        # 记录人类是否直接选择了 RAG Top1 候选。
        hum_picked_top1 = False
        for cand, src in [
            ("a", "candidate_a_source"),
            ("b", "candidate_b_source"),
            ("c", "candidate_c_source"),
            ("d", "candidate_d_source"),
            ("e", "candidate_e_source"),
        ]:
            if "top1" in str(data.get(src, "")) and hum_choice == cand.upper():
                hum_picked_top1 = True

        records.append({
            "task_id": tid, "recruitment_record_id": recruitment_record_id, "job_title": jt, "job_reqs": jr[:150],
            "n_ann": n_ann, "has_majority": has_majority,
            "hum_choice": hum_choice, "hum_code": hum_code, "hum_title": hum_title,
            "ds_choice": ds_choice, "ds_code": ds_code, "ds_title": ds_title,
            "ds_conf": ds_conf, "ds_agrees": (ds_choice == hum_choice) if ds else None,
            "sem_rank": sem_rank, "ds_sem_rank": ds_sem_rank,
            "hierarchy_dist": hdist,
            "pairwise": pairwise,
            "hum_picked_top1": hum_picked_top1,
        })

    def w(text: str) -> None:
        """同时写入报告文件并尝试打印到控制台。"""
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        try:
            print(text)
        except UnicodeEncodeError:
            pass

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    w("=" * 70)
    w("分歧深度分析报告")
    w("=" * 70)

    # Split records
    agree_recs = [r for r in records if r["ds_agrees"] is True]
    disagree_recs = [r for r in records if r["ds_agrees"] is False]
    no_ds_recs = [r for r in records if r["ds_agrees"] is None]

    w(f"\n总记录: {len(records)}")
    w(f"DS一致: {len(agree_recs)} ({len(agree_recs)/len(records)*100:.1f}%)")
    w(f"DS分歧: {len(disagree_recs)} ({len(disagree_recs)/len(records)*100:.1f}%)")
    w(f"无DS:   {len(no_ds_recs)} ({len(no_ds_recs)/len(records)*100:.1f}%)")

    # === 分析1: 语义排名分布 ===
    w(f"\n{'='*70}")
    w(f"[分析1] 语义排名分布: DS一致 vs DS分歧")
    w(f"{'='*70}")

    bins = [(1,3), (4,10), (11,30), (31,100), (101,300), (301,1698)]
    for label, subset in [("DS一致", agree_recs), ("DS分歧", disagree_recs)]:
        w(f"\n  {label} (N={len(subset)}):")
        for lo, hi in bins:
            cnt = sum(1 for r in subset if r["sem_rank"] and lo <= r["sem_rank"] <= hi)
            pct = cnt/len(subset)*100 if subset else 0
            bar = "█" * int(pct/2)
            w(f"    语义Top{lo}-{hi:>4}: {cnt:>5} ({pct:5.1f}%) {bar}")

    # === 分析2: 层级距离分布 ===
    w(f"\n{'='*70}")
    w(f"[分析2] 职业层级距离: 人类 vs Deepseek 选择的距离")
    w(f"{'='*70}")
    dist_labels = {0:"同细类", 1:"同小类(不同细类)", 2:"同中类(不同小类)",
                   3:"同大类(不同中类)", 4:"完全不同大类"}
    dist_cnt = Counter(r["hierarchy_dist"] for r in disagree_recs if r["hierarchy_dist"] is not None)
    for d in sorted(dist_cnt.keys()):
        cnt = dist_cnt[d]
        pct = cnt/sum(dist_cnt.values())*100
        w(f"  {dist_labels.get(d, f'L{d}')}: {cnt} ({pct:.1f}%)")

    # === 分析3: DS分歧时，谁的选择语义排名更高 ===
    w(f"\n{'='*70}")
    w(f"[分析3] DS分歧时，谁的语义排名更高")
    w(f"{'='*70}")
    human_better = 0
    ds_better = 0
    both_bad = 0
    for r in disagree_recs:
        hr = r["sem_rank"] or 9999
        dr = r["ds_sem_rank"] or 9999
        if hr <= 10 and dr <= 10:
            pass  # both good
        if hr < dr: human_better += 1
        elif dr < hr: ds_better += 1
        else: both_bad += 1

    n_dis = len(disagree_recs)
    w(f"  人类选择语义排名更高: {human_better} ({human_better/n_dis*100:.1f}%)")
    w(f"  DS选择语义排名更高:   {ds_better} ({ds_better/n_dis*100:.1f}%)")
    w(f"  持平:                 {both_bad} ({both_bad/n_dis*100:.1f}%)")

    # === 分析4: 分歧 by DS置信度 ===
    w(f"\n{'='*70}")
    w(f"[分析4] DS分歧按DS置信度分组")
    w(f"{'='*70}")
    conf_bins = [(0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
    for lo, hi in conf_bins:
        cnt = sum(1 for r in disagree_recs if r["ds_conf"] and lo <= r["ds_conf"] < hi)
        pct = cnt/n_dis*100
        w(f"  DS置信度 [{lo:.2f}-{hi:.2f}): {cnt:>5} ({pct:.1f}%)")

    # === 分析5: 分歧职业热力图 ===
    w(f"\n{'='*70}")
    w(f"[分析5] DS分歧最高频的职业 (Top 20)")
    w(f"{'='*70}")
    disagree_occ = Counter(r["hum_title"] for r in disagree_recs if r["hum_title"])
    for title, cnt in disagree_occ.most_common(20):
        w(f"  {title}: {cnt} 条分歧")

    # === 分析6: 分歧岗位名称关键词 ===
    w(f"\n{'='*70}")
    w(f"[分析6] DS分歧的岗位名称常见词")
    w(f"{'='*70}")
    import re
    title_words = Counter()
    for r in disagree_recs:
        words = re.findall(r'[一-龥]{2,}', r["job_title"])
        for wd in words:
            if wd not in ("工程师", "经理", "主管", "专员", "助理", "顾问", "代表", "总监"):
                title_words[wd] += 1
    for wd, cnt in title_words.most_common(40):
        w(f"  {wd}: {cnt}")

    # === 分析7: 多标注 vs 单标注分歧率 ===
    w(f"\n{'='*70}")
    w(f"[分析7] 按标注人数看分歧率")
    w(f"{'='*70}")
    by_ann = defaultdict(lambda: {"total": 0, "disagree": 0})
    for r in records:
        if r["ds_agrees"] is None: continue
        n = min(r["n_ann"], 5) if r["n_ann"] <= 5 else "6+"
        by_ann[n]["total"] += 1
        if r["ds_agrees"] is False:
            by_ann[n]["disagree"] += 1
    for n in sorted(by_ann.keys(), key=lambda x: x if isinstance(x, int) else 99):
        s = by_ann[n]
        rate = s["disagree"]/s["total"]*100 if s["total"]>0 else 0
        label = f"{n}人标注" if isinstance(n, int) else "多人标注"
        w(f"  {label}: 分歧率={s['disagree']}/{s['total']} = {rate:.1f}%")

    # === 分析8: 人类选了RAG Top1 vs 没选Top1时的分歧率 ===
    w(f"\n{'='*70}")
    w(f"[分析8] 人类选RAG Top1 vs 没选Top1 的分歧率")
    w(f"{'='*70}")
    for label, cond in [("选了Top1", lambda r: r["hum_picked_top1"]),
                         ("没选Top1", lambda r: not r["hum_picked_top1"])]:
        subset = [r for r in records if cond(r) and r["ds_agrees"] is not None]
        disag = sum(1 for r in subset if r["ds_agrees"] is False)
        w(f"  {label}: {disag}/{len(subset)} = {disag/len(subset)*100:.1f}% 分歧")

    # === 分析9: 构建正负样本划分方案 ===
    w(f"\n{'='*70}")
    w(f"[分析9] 基于分歧的正负样本划分方案")
    w(f"{'='*70}")

    # Positive: multi-ann + DS agree + semantic top10
    pos_strict = [r for r in records if r["ds_agrees"] is True
                  and r["has_majority"] and r["sem_rank"] and r["sem_rank"] <= 10]
    pos_moderate = [r for r in records if r["ds_agrees"] is True
                    and r["sem_rank"] and r["sem_rank"] <= 10]
    pos_lenient = [r for r in records if r["ds_agrees"] is True]

    # Negative: multi-ann + DS disagree + semantic rank bad
    neg_strict = [r for r in records if r["ds_agrees"] is False
                  and r["has_majority"] and r["sem_rank"] and r["sem_rank"] >= 30]
    neg_moderate = [r for r in records if r["ds_agrees"] is False
                    and r["sem_rank"] and r["sem_rank"] >= 30]
    neg_lenient = [r for r in records if r["ds_agrees"] is False
                   and r["sem_rank"] and r["sem_rank"] >= 100]

    w(f"\n  正样本方案:")
    w(f"    严格 (多标注+DS一致+语义Top10):  {len(pos_strict)}")
    w(f"    中等 (DS一致+语义Top10):         {len(pos_moderate)}")
    w(f"    宽松 (DS一致):                   {len(pos_lenient)}")
    w(f"\n  负样本方案:")
    w(f"    严格 (多标注+DS分歧+语义≥#30):   {len(neg_strict)}")
    w(f"    中等 (DS分歧+语义≥#30):          {len(neg_moderate)}")
    w(f"    宽松 (DS分歧+语义≥#100):         {len(neg_lenient)}")

    # === 分析10: 边界案例展示 ===
    w(f"\n{'='*70}")
    w(f"[分析10] 各类边界案例")
    w(f"{'='*70}")

    # (a) DS分歧但人类大概率对
    ds_wrong_but_human_right = [r for r in disagree_recs
                                if r["sem_rank"] and r["sem_rank"] <= 5
                                and r["has_majority"]]
    ds_wrong_but_human_right.sort(key=lambda r: r["sem_rank"])
    w(f"\n  [a] DS可能错了(人类语义Top5+多标注一致): {len(ds_wrong_but_human_right)}条")
    for r in ds_wrong_but_human_right[:5]:
        w(f"\n    task_id={r['task_id']} 岗位: {r['job_title']}")
        w(f"    人类选: {r['hum_title']} (语义#{r['sem_rank']}, {r['n_ann']}人pairwise={r['pairwise']:.0%})")
        w(f"    DS选:   {r['ds_title']} (conf={r['ds_conf']}, 语义#{r['ds_sem_rank']})")
        w(f"    层级距离: {r['hierarchy_dist']}")

    # (b) DS分歧且人类大概率错
    human_wrong = [r for r in disagree_recs
                   if r["sem_rank"] and r["sem_rank"] >= 100
                   and r["ds_sem_rank"] and r["ds_sem_rank"] <= 20]
    human_wrong.sort(key=lambda r: r["sem_rank"], reverse=True)
    w(f"\n  [b] 人类可能错了(语义极低+DS选择语义高): {len(human_wrong)}条")
    for r in human_wrong[:5]:
        w(f"\n    task_id={r['task_id']} 岗位: {r['job_title']}")
        w(f"    人类选: {r['hum_title']} (语义#{r['sem_rank']})")
        w(f"    DS选:   {r['ds_title']} (语义#{r['ds_sem_rank']}, conf={r['ds_conf']})")
        w(f"    层级距离: {r['hierarchy_dist']}")

    # (c) 双方都不对(都排名低)
    both_wrong = [r for r in disagree_recs
                  if r["sem_rank"] and r["sem_rank"] >= 50
                  and r["ds_sem_rank"] and r["ds_sem_rank"] >= 50]
    w(f"\n  [c] 双方都可能错了(都排名≥#50): {len(both_wrong)}条")
    for r in both_wrong[:5]:
        w(f"\n    task_id={r['task_id']} 岗位: {r['job_title']}")
        w(f"    人类选: {r['hum_title']} (语义#{r['sem_rank']})")
        w(f"    DS选:   {r['ds_title']} (语义#{r['ds_sem_rank']}, conf={r['ds_conf']})")
        w(f"    层级距离: {r['hierarchy_dist']}")

    # (d) 层级距离大(跨领域分歧)
    cross_domain = [r for r in disagree_recs if r["hierarchy_dist"] is not None and r["hierarchy_dist"] >= 3]
    w(f"\n  [d] 跨大类分歧(层级距离≥3): {len(cross_domain)}条")
    for r in cross_domain[:5]:
        w(f"\n    task_id={r['task_id']} 岗位: {r['job_title']}")
        w(f"    人类选: {r['hum_title']} (语义#{r['sem_rank']})")
        w(f"    DS选:   {r['ds_title']} (语义#{r['ds_sem_rank']}, conf={r['ds_conf']})")
        w(f"    层级距离: {r['hierarchy_dist']}")

    # (e) 语义排名接近但有分歧(边界模糊)
    borderline = [r for r in disagree_recs
                  if r["sem_rank"] and r["ds_sem_rank"]
                  and r["sem_rank"] <= 10 and r["ds_sem_rank"] <= 10]
    w(f"\n  [e] 双方语义都Top10但仍有分歧: {len(borderline)}条")
    for r in borderline[:5]:
        w(f"\n    task_id={r['task_id']} 岗位: {r['job_title']}")
        w(f"    人类选: {r['hum_title']} (语义#{r['sem_rank']})")
        w(f"    DS选:   {r['ds_title']} (语义#{r['ds_sem_rank']}, conf={r['ds_conf']})")

    print(f"\nDone! Report: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


