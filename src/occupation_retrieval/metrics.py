"""occupation_retrieval 的评估指标工具。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def compute_candidate_accuracy(predicted_letter: str | None, human_letter: str | None) -> int:
    """判断候选选择是否命中人工选择。

    Args:
        predicted_letter: 模型预测的候选字母。
        human_letter: 人工参考候选字母。

    Returns:
        int: 命中返回 1，否则返回 0。
    """
    if not predicted_letter or not human_letter:
        return 0
    return int(str(predicted_letter).upper() == str(human_letter).upper())


def reciprocal_rank(rank: int | None) -> float:
    """计算单条样本的倒数排名。

    Args:
        rank: 从 1 开始的排名；为空或非正数时视为未命中。

    Returns:
        float: 倒数排名。
    """
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank


def same_hierarchy_level(
    code_a: str | None,
    code_b: str | None,
    code_to_level: Mapping[str, str],
) -> bool:
    """判断两个职业代码是否处在同一层级映射值下。"""
    if not code_a or not code_b:
        return False
    left = code_to_level.get(str(code_a))
    right = code_to_level.get(str(code_b))
    return bool(left and right and left == right)


def _pct(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def summarize_model_metrics(raw_counts: dict[str, Any]) -> dict[str, float]:
    """将评估原始计数汇总为百分比和均值指标。

    Args:
        raw_counts: `eval_models_multimetric` 中累计的原始计数字典。

    Returns:
        dict[str, float]: 可用于报告输出的指标。
    """
    ranks = raw_counts.get("human_rank_in_candidates", [])
    reciprocal_ranks = raw_counts.get("reciprocal_ranks", [])
    ds_total = float(raw_counts.get("ds_total", 0))
    return {
        "candidate_acc": _pct(
            float(raw_counts.get("candidate_hit", 0)),
            float(raw_counts.get("candidate_total", 0)),
        ),
        "mean_human_rank": float(np.mean(ranks)) if ranks else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "subclass_acc": _pct(
            float(raw_counts.get("subclass_hit", 0)),
            float(raw_counts.get("full_total", 0)),
        ),
        "midclass_acc": _pct(
            float(raw_counts.get("midclass_hit", 0)),
            float(raw_counts.get("full_total", 0)),
        ),
        "major_acc": _pct(
            float(raw_counts.get("major_hit", 0)),
            float(raw_counts.get("full_total", 0)),
        ),
        "ds_side_human_pct": _pct(float(raw_counts.get("ds_side_human", 0)), ds_total),
        "ds_side_ds_pct": _pct(float(raw_counts.get("ds_side_ds", 0)), ds_total),
    }
