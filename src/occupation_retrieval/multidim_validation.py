#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多维度交叉验证：检测第二轮标注数据集中可能存在的标注错误。

通过 5 个独立信号维度对每条标注进行质量评分，
组合不同信号形成质量层级，挑出典型案例。

五个信号维度:
    1. RAG 语义相似度排名（候选在检索结果中的位置）
    2. 职业大类关键词匹配（岗位名称 vs 候选大类）
    3. 标注员间一致性（标注员对同一任务的投票分布）
    4. DeepSeek 预测一致性（模型预测 vs 人类标注）
    5. 候选文本相似度（多个候选间的语义距离）

策略:
    - 不依赖单一维度，多信号交叉验证
    - 对每个质量层输出"疑似正确"和"疑似错误"的案例

用法:
    python -m src.occupation_retrieval.multidim_validation

前置条件:
    BGE 模型路径通过 config/paths.py 或环境变量 EMPLOYDATA_BGE_MODEL_PATH 配置。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any

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
OUTPUT_FILE = os.path.join(get_occupation_retrieval_output_dir(), "multidim_validation_report.txt")
OUTPUT_JSON = os.path.join(get_occupation_retrieval_output_dir(), "multidim_validation_results.json")
MODEL_PATH = resolve_base_model_path()


# ── 大类关键词映射 ──────────────────────────────
MAJOR_CLASS_KEYWORDS = {
    "专业技术人员": {
        "kw": ["技术", "工程", "开发", "设计", "分析", "研发", "算法", "架构",
               "测试", "运维", "前端", "后端", "数据", "软件", "硬件", "网络",
               "安全", "系统", "编程", "代码", "AI", "Java", "Python", "医学",
               "医师", "护士", "药剂", "检验", "影像", "法律", "律师", "会计",
               "审计", "教师", "教授", "翻译", "记者", "编辑"],
        "label": "专业技术人员"
    },
    "办事人员和有关人员": {
        "kw": ["行政", "人事", "财务", "会计", "出纳", "法务", "合规", "秘书",
               "助理", "文员", "前台", "档案", "后勤", "统计", "审计"],
        "label": "办事人员"
    },
    "社会生产服务和生活服务人员": {
        "kw": ["服务", "销售", "客服", "运营", "配送", "物流", "餐饮", "司机",
               "保安", "保洁", "快递", "外卖", "导游", "美容", "美发", "厨师",
               "房地产", "保险", "中介", "店员", "收银", "导购"],
        "label": "服务人员"
    },
    "生产制造及有关人员": {
        "kw": ["生产", "制造", "加工", "装配", "质检", "操作", "车工", "钳工",
               "焊工", "电工", "维修", "安装", "调试", "采矿", "冶炼"],
        "label": "制造人员"
    },
    "企事业单位负责人": {
        "kw": ["经理", "总监", "主管", "总裁", "CEO", "负责人", "主任", "部长",
               "院长", "校长", "董事长", "总经理", "副总", "项目经理"],
        "label": "管理人员"
    },
}


def load_occupation_dict(
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """加载职业大典，返回 {code: text}, {code: title}, {code: major_class}。"""
    df = load_occupation_dict_df()

    code_to_text: dict[str, str] = {}
    code_to_title: dict[str, str] = {}
    code_to_major: dict[str, str] = {}

    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        title = str(row["title"]).strip()
        desc = str(row.get("desc", "")).strip()
        tasks = str(row.get("tasks", "")).strip()
        major = str(row.get("大类", "")).strip()

        if not code or not title:
            continue
        code_to_title[code] = title

        parts = [title]
        if desc and desc.lower() != "nan":
            parts.append(f"定义：{desc}")
        if tasks and tasks.lower() != "nan":
            parts.append(f"任务：{tasks}")
        code_to_text[code] = "。".join(parts)

        if major and major.lower() != "nan":
            code_to_major[code] = major

    return code_to_text, code_to_title, code_to_major


def guess_major_class(job_title: str) -> str | None:
    """根据岗位名称中的关键词猜测职业大类。"""
    if not job_title:
        return None
    scores: dict[str, int] = {}
    for mclass, info in MAJOR_CLASS_KEYWORDS.items():
        score = sum(1 for kw in info["kw"] if kw.lower() in job_title.lower())
        if score > 0:
            scores[mclass] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def class_match_score(title_guess: str | None, occ_major: str | None) -> float:
    """
    返回岗位名称关键词猜测的大类与职业大典大类是否匹配。
    1.0 = 完全匹配, 0.0 = 不匹配, 0.5 = 无法判断
    """
    if not title_guess or not occ_major:
        return 0.5
    occ_major = occ_major.strip()
    for mclass, info in MAJOR_CLASS_KEYWORDS.items():
        if info["label"] == title_guess or title_guess == mclass:
            return 1.0 if (occ_major == mclass or
                           info["label"] in occ_major or
                           occ_major.startswith(info["label"])) else 0.0
    return 0.5


def main() -> None:
    """执行多信号交叉验证并输出质量分层报告。"""
    print("=" * 70)
    print("多维度交叉验证：检测标注数据集错误")
    print("=" * 70)

    # ── 加载数据 ──
    print("\n[1] 加载数据...")
    raw_data = load_annotations_from_pg()
    print(f"    人工标注: {len(raw_data)} 条任务")

    ds_records = load_deepseek_records()
    print(f"    Deepseek: {len(ds_records)} 条记录")

    code_to_text, code_to_title, code_to_major = load_occupation_dict()
    print(f"    职业大典: {len(code_to_text)} 个细类")

    # ── 加载 BGE 模型用于语义相似度 ──
    print("\n[2] 加载语义模型...")
    device = get_runtime_device()
    model = SentenceTransformer(MODEL_PATH, device=device)
    model.max_seq_length = 256

    # 预编码所有职业文本
    occ_codes = sorted(code_to_text.keys())
    occ_texts = [code_to_text[c] for c in occ_codes]
    with torch.no_grad():
        occ_embeddings = model.encode(occ_texts, batch_size=64,
                                      normalize_embeddings=True,
                                      show_progress_bar=True,
                                      convert_to_tensor=True)

    # ── 解析每个任务的信号 ──
    print("\n[3] 计算多维度信号...")
    results: list[dict[str, Any]] = []

    # 先计算标注员质量
    ann_quality = defaultdict(lambda: {"agree": 0, "total": 0})
    for item in raw_data:
        anns = item["annotations"]
        if len(anns) < 2: continue
        choices = []
        for ann in anns:
            c = parse_choice(ann)
            if c and c != "NONE":
                choices.append((ann["completed_by"], c))
        if len(choices) < 2: continue
        counter = Counter([c for _, c in choices])
        majority, _ = counter.most_common(1)[0]
        for aid, c in choices:
            ann_quality[aid]["total"] += 1
            if c == majority:
                ann_quality[aid]["agree"] += 1

    ann_score: dict[Any, float] = {}
    for aid in ann_quality:
        s = ann_quality[aid]
        ann_score[aid] = s["agree"] / s["total"] if s["total"] > 0 else 0.5

    # 主循环：计算每条任务的所有信号
    for item in raw_data:
        task_id = item["task_id"]
        recruitment_record_id = item["recruitment_record_id"]
        data = item["data"]
        anns = item["annotations"]
        n_annotators = len(anns)

        job_title = str(data.get("job_title", "")).strip()
        job_reqs = str(data.get("job_requirements_clean", "")).strip()
        if not job_reqs:
            continue

        # ---- 信号1: 标注员间一致性 ----
        choices = []
        annotators = []
        for ann in anns:
            c = parse_choice(ann)
            if c and c != "NONE":
                choices.append(c)
                annotators.append(ann["completed_by"])

        if not choices:
            continue

        if n_annotators >= 2 and len(choices) >= 2:
            counter = Counter(choices)
            majority, maj_count = counter.most_common(1)[0]
            pairwise_agree = sum(1 for i in range(len(choices))
                                 for j in range(i+1, len(choices))
                                 if choices[i] == choices[j])
            pairwise_rate = pairwise_agree / (len(choices)*(len(choices)-1)//2) if len(choices) > 1 else 1.0
            has_majority = maj_count > len(choices) / 2
        else:
            majority = choices[0]
            maj_count = 1
            pairwise_rate = 1.0
            has_majority = False

        # ---- 信号2: Deepseek 一致性 ----
        ds_record = ds_records.get(task_id)
        ds_agrees = None
        ds_confidence = None
        if ds_record:
            ds_choice = ds_record["deepseek_choice"]
            ds_confidence = ds_record.get("deepseek_confidence")
            ds_agrees = (ds_choice == majority)

        # ---- 信号3: 语义相似度 ----
        chosen_code = str(data.get(f"candidate_{majority.lower()}_code", "")).strip()
        chosen_title_in_list = str(data.get(f"candidate_{majority.lower()}_title", "")).strip()

        anchor = f"{job_title} {job_reqs}"
        with torch.no_grad():
            anchor_emb = model.encode([anchor], batch_size=1,
                                      normalize_embeddings=True,
                                      show_progress_bar=False,
                                      convert_to_tensor=True)

        semantic_sim = None
        chosen_idx = None
        if chosen_code in code_to_text:
            # 这里关心的不是候选间排序，而是该职业在全量职业库里的真实语义名次。
            sims = torch.mm(anchor_emb, occ_embeddings.T).squeeze(0)
            sorted_indices = torch.argsort(sims, descending=True).cpu().tolist()
            chosen_code_idx = occ_codes.index(chosen_code) if chosen_code in occ_codes else -1
            if chosen_code_idx >= 0:
                try:
                    rank = sorted_indices.index(chosen_code_idx) + 1
                except ValueError:
                    rank = -1
                semantic_rank = rank
                semantic_sim = float(sims[chosen_code_idx].item())
            else:
                semantic_rank = -1
        else:
            semantic_rank = -1

        # ---- 信号4: 岗位名称-大类匹配 ----
        title_guess = guess_major_class(job_title)
        occ_major = code_to_major.get(chosen_code, "")
        kw_match = class_match_score(title_guess, occ_major)

        # ---- 信号5: 标注员历史质量 ----
        avg_ann_quality = (
            np.mean([ann_score.get(a, 0.5) for a in annotators])
            if annotators else 0.5
        )

        # ---- 信号6: 有多大比例标注员选了NONE (备选都不合适) ----
        none_count = sum(1 for ann in anns
                         if parse_choice(ann) == "NONE")
        none_rate = none_count / n_annotators if n_annotators > 0 else 0

        # ---- Top5候选质量 ----
        top1_rank = None
        for cand in ["a", "b", "c", "d", "e"]:
            src = str(data.get(f"candidate_{cand}_source", ""))
            if "top1" in src and majority == cand.upper():
                top1_rank = True
                break
        else:
            top1_rank = False  # 人类选的不是RAG Top1

        # ---- 收集所有候选的职业大类 ----
        candidate_classes = set()
        for cand in ["a", "b", "c", "d", "e"]:
            code = str(data.get(f"candidate_{cand}_code", "")).strip()
            major = code_to_major.get(code, "")
            if major:
                candidate_classes.add(major)

        results.append({
            "task_id": task_id,
            "recruitment_record_id": recruitment_record_id,
            "job_title": job_title,
            "job_reqs": job_reqs[:200],
            "anchor": anchor[:300],
            "n_annotators": n_annotators,
            "majority": majority,
            "chosen_code": chosen_code,
            "chosen_title": code_to_title.get(chosen_code, ""),
            # 信号
            "pairwise_agree": pairwise_rate,
            "has_majority": has_majority,
            "ds_agrees": ds_agrees,
            "ds_confidence": ds_confidence,
            "semantic_rank": semantic_rank,
            "semantic_sim": semantic_sim,
            "kw_match": kw_match,
            "title_guess": title_guess,
            "occ_major": occ_major,
            "ann_quality": avg_ann_quality,
            "none_rate": none_rate,
            "top1_rank": top1_rank,
            "candidate_classes_count": len(candidate_classes),
            "maj_count": maj_count,
        })

    print(f"    有效任务: {len(results)}")

    # ── 定义质量层级 ──
    print("\n[4] 质量层级分析...")

    tiers: list[str] = []
    # 逐条打分
    for r in results:
        score = 0
        reasons = []

        # 维度1: 多标注一致性 (0-3分)
        if r["n_annotators"] >= 2 and r["has_majority"]:
            if r["pairwise_agree"] >= 0.8:
                score += 3
                reasons.append("多标注高度一致")
            elif r["pairwise_agree"] >= 0.5:
                score += 2
                reasons.append("多标注基本一致")
            else:
                score += 1
                reasons.append("多标注分歧")
        elif r["n_annotators"] >= 2:
            score += 1
            reasons.append("多标注无明显多数")
        else:
            reasons.append("单标注(无法交叉验证)")

        # 维度2: Deepseek 一致性 (0-3分)
        if r["ds_agrees"] is True:
            if r["ds_confidence"] is not None and r["ds_confidence"] >= 0.9:
                score += 3
                reasons.append("Deepseek高置信一致")
            else:
                score += 2
                reasons.append("Deepseek一致")
        elif r["ds_agrees"] is False:
            score -= 2
            reasons.append("Deepseek分歧(扣分)")

        # 维度3: 语义排名 (0-3分)
        if r["semantic_rank"] is not None and r["semantic_rank"] > 0:
            if r["semantic_rank"] <= 5:
                score += 3
                reasons.append(f"语义排名Top{r['semantic_rank']}")
            elif r["semantic_rank"] <= 20:
                score += 2
                reasons.append(f"语义排名Top{r['semantic_rank']}")
            elif r["semantic_rank"] <= 50:
                score += 1
                reasons.append(f"语义排名Top{r['semantic_rank']}")
            else:
                score -= 1
                reasons.append(f"语义排名低(#{r['semantic_rank']})(扣分)")
        else:
            reasons.append("无语义排名")

        # 维度4: 大类关键词匹配 (0-2分)
        if r["kw_match"] == 1.0:
            score += 2
            reasons.append("岗位名称与职业大类匹配")
        elif r["kw_match"] == 0.0:
            score -= 1
            reasons.append("岗位名称与职业大类不匹配(扣分)")

        # 维度5: 标注员质量 (0-1分)
        if r["ann_quality"] >= 0.7:
            score += 1
        elif r["ann_quality"] < 0.5:
            score -= 1
            reasons.append(f"标注员质量低({r['ann_quality']:.0%})")

        # 维度6: NONE比例 (如果很多人认为都不合适)
        if r["none_rate"] > 0.3:
            score -= 1
            reasons.append(f"{r['none_rate']:.0%}标注员选了NONE")

        # 维度7: 人类是否选了RAG Top1 (选了Top1说明匹配明确)
        if r["top1_rank"] is True:
            score += 1
            reasons.append("人类选了RAG Top1")

        tier = "C"  # 默认
        if score >= 9:
            tier = "S"   # Super: 极高置信
        elif score >= 6:
            tier = "A"   # 高置信
        elif score >= 3:
            tier = "B"   # 中等置信
        elif score >= 0:
            tier = "C"   # 低置信，可能有问题
        else:
            tier = "D"   # 极低，很可能错误

        r["tier"] = tier
        r["score"] = score
        r["reasons"] = reasons
        tiers.append(tier)

    # ── 统计 ──
    tier_counts = Counter(tiers)
    total = len(results)
    print(f"\n    质量层级分布 (N={total}):")
    tier_descriptions = {
        "S": "极高置信 (多标注一致 + DS一致 + 高语义匹配 + 大类匹配)",
        "A": "高置信   (多数信号正面)",
        "B": "中等置信 (部分信号正面，部分缺失)",
        "C": "低置信   (信号不足或部分信号负面)",
        "D": "疑似错误 (多个负面信号)",
    }
    for t in ["S", "A", "B", "C", "D"]:
        cnt = tier_counts.get(t, 0)
        print(f"    {t}: {cnt:>5} ({cnt/total*100:5.1f}%)  {tier_descriptions[t]}")

    # ── 挑选案例 ──
    print("\n[5] 挑选典型案例...")

    # 按 tier 分组
    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)

    output_lines: list[str] = []

    def w(text: str) -> None:
        """向报告缓存和控制台同时输出一行文本。"""
        output_lines.append(text)
        # Only try to print; silently ignore encoding errors on Windows
        try:
            print(text)
        except UnicodeEncodeError:
            pass

    w("=" * 70)
    w("多维度交叉验证报告：标注数据集质量分析")
    w("=" * 70)
    w(f"\n总样本数: {len(results)}")
    w(f"\n## 质量层级分布")
    for t in ["S", "A", "B", "C", "D"]:
        cnt = tier_counts.get(t, 0)
        w(f"  {t} ({tier_descriptions[t]}): {cnt} ({cnt/total*100:.1f}%)")

    # ── 维度组合分析 ──
    w(f"\n\n## 维度组合交叉分析")

    combos = [
        {
            "name": "多标注+Deepseek双确认 (最强组合)",
            "cond": lambda r: (r["n_annotators"] >= 2 and r["has_majority"]
                               and r["ds_agrees"] is True and r["pairwise_agree"] >= 0.8),
            "n_samples": 5,
        },
        {
            "name": "多标注+Deepseek分歧-标注员可能错 (标注争议)",
            "cond": lambda r: (r["n_annotators"] >= 2 and r["has_majority"]
                               and r["ds_agrees"] is False and r["pairwise_agree"] < 0.8),
            "n_samples": 5,
        },
        {
            "name": "单标注+Deepseek一致+语义Top3 (Silver高质)",
            "cond": lambda r: (r["n_annotators"] == 1 and r["ds_agrees"] is True
                               and r["semantic_rank"] is not None and r["semantic_rank"] <= 3),
            "n_samples": 5,
        },
        {
            "name": "单标注+Deepseek分歧+语义排名低 (疑似错误)",
            "cond": lambda r: (r["n_annotators"] == 1 and r["ds_agrees"] is False
                               and r["semantic_rank"] is not None and r["semantic_rank"] > 50),
            "n_samples": 5,
        },
        {
            "name": "语义排名高但Deepseek分歧 (DS可能幻觉)",
            "cond": lambda r: (r["ds_agrees"] is False
                               and r["semantic_rank"] is not None and r["semantic_rank"] <= 5),
            "n_samples": 5,
        },
        {
            "name": "大类不匹配+Deepseek分歧 (跨领域错误)",
            "cond": lambda r: (r["kw_match"] == 0.0 and r["ds_agrees"] is False),
            "n_samples": 5,
        },
        {
            "name": "人类选NONERate高 (>30%的人认为候选都不合适)",
            "cond": lambda r: r["none_rate"] > 0.3,
            "n_samples": 5,
        },
        {
            "name": "语义排名极低 (#1-100之外) - 候选质量差",
            "cond": lambda r: (r["semantic_rank"] is not None and r["semantic_rank"] > 100),
            "n_samples": 5,
        },
    ]

    for combo in combos:
        matched = [r for r in results if combo["cond"](r)]
        if not matched:
            w(f"\n### {combo['name']}")
            w(f"  匹配数量: 0 (无样本)")
            continue

        # 按score排序，取top和bottom
        matched.sort(key=lambda r: r["score"], reverse=True)
        n = min(combo["n_samples"], len(matched))
        examples = matched[:n]  # 取score最高的几个

        w(f"\n### {combo['name']}")
        w(f"  匹配数量: {len(matched)} 条")
        w(f"  展示 {n} 条高分案例:")

        for i, r in enumerate(examples, 1):
            w(f"\n  --- 案例 {i} (task_id={r['task_id']}, score={r['score']}, tier={r['tier']}) ---")
            w(f"  岗位名称: {r['job_title']}")
            w(f"  岗位描述: {r['job_reqs'][:120]}...")
            w(f"  人类选择: {r['majority']} → {r['chosen_code']} {r['chosen_title']}")
            w(f"  信号: {', '.join(r['reasons'])}")
            w(f"  详情: n_ann={r['n_annotators']}, pairwise={r['pairwise_agree']:.1%}, "
              f"DS={r['ds_agrees']}, sem_rank=#{r['semantic_rank']}, "
              f"kw_match={r['kw_match']:.0f}, ann_q={r['ann_quality']:.0%}, "
              f"none_rate={r['none_rate']:.0%}, top1_pick={r['top1_rank']}")

    # ── S/D 两级各取5个 ──
    w(f"\n\n## S级 (极高置信) 抽样展示")
    s_samples = sorted(by_tier.get("S", []), key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(s_samples[:10], 1):
        w(f"\n  [{i}] task_id={r['task_id']} score={r['score']}")
        w(f"  岗位: {r['job_title']}")
        w(f"  选择: {r['majority']} → {r['chosen_code']} {r['chosen_title']}")
        w(f"  信号: {'; '.join(r['reasons'])}")
        w(f"  描述: {r['job_reqs'][:100]}...")

    w(f"\n\n## D级 (疑似错误) 抽样展示")
    d_samples = sorted(by_tier.get("D", []), key=lambda r: r["score"])
    for i, r in enumerate(d_samples[:10], 1):
        w(f"\n  [{i}] task_id={r['task_id']} score={r['score']}")
        w(f"  岗位: {r['job_title']}")
        w(f"  选择: {r['majority']} → {r['chosen_code']} {r['chosen_title']}")
        w(f"  信号: {'; '.join(r['reasons'])}")
        w(f"  描述: {r['job_reqs'][:100]}...")

    # ── 保存 ──
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    # 保存 JSON
    json_data = {
        "total": len(results),
        "tier_distribution": {t: tier_counts.get(t, 0) for t in ["S", "A", "B", "C", "D"]},
        "tier_descriptions": tier_descriptions,
        "results": [{k: v for k, v in r.items() if k != "anchor"}
                    for r in sorted(results, key=lambda r: r["score"], reverse=True)],
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    w(f"\n\n报告已保存至: {OUTPUT_FILE}")
    w(f"JSON 已保存至: {OUTPUT_JSON}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()

