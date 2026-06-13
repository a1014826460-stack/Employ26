"""LLM-as-Judge 评估：独立于 gold 标签的预测质量评判。

对每个样本的预测结果，让 LLM 判断每个预测是否在原文中有依据，
产出 second-opinion 精确率，不受 gold 标注稀疏性影响。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """你是HR领域专家。判断以下软技能预测是否在岗位描述中有明确依据。

规则：
1. 如果文本中明确提到了该技能（或同义表述），判定为 valid
2. 如果文本中未涉及该技能，判定为 invalid
3. 维度标注合理性仅作参考，不作为主要判定依据
4. 只输出 JSON 对象

输出格式: {"技能名": "valid", ...}"""


def judge_predictions(
    text: str,
    predictions: List[Dict[str, str]],
    llm_client: Any,
) -> Dict[str, str]:
    """让 LLM 评判每个预测是否在文本中有依据。

    参数:
        text: 岗位描述原文。
        predictions: 预测的软技能列表。
        llm_client: LLM 客户端。

    返回:
        dict: 技能名 → "valid"|"invalid" 的映射。
    """
    if not predictions:
        return {}

    pred_list = "\n".join(
        f"- {p['name']} [{p.get('dimension', '')}]" for p in predictions
    )

    try:
        resp = llm_client.complete_json(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=f"岗位描述:\n{text[:2000]}\n\n预测技能:\n{pred_list}",
            strength="cheap",
            max_output_tokens=1000,
        )
        if isinstance(resp, dict):
            return {k: str(v).lower() for k, v in resp.items()}
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)

    return {}


def compute_judge_metrics(
    eval_results_path: str | Path,
    llm_client: Any,
    max_samples: int = 50,
) -> Dict[str, float]:
    """对评估结果中的预测进行 LLM 独立评判。

    参数:
        eval_results_path: v7 error CSV 路径。
        llm_client: LLM 客户端。
        max_samples: 最多评判的样本数。

    返回:
        dict: {"llm_judge_precision": ..., "judged_samples": ...}
    """
    import csv

    path = Path(eval_results_path)
    if not path.exists():
        return {"error": f"file not found: {path}"}

    rows = list(csv.DictReader(open(path, "r", encoding="utf-8-sig")))

    total_pred = 0
    total_valid = 0

    for i, row in enumerate(rows[:max_samples]):
        text = row.get("text", "")
        try:
            pred = json.loads(row.get("predicted_skills", "[]"))
        except json.JSONDecodeError:
            continue

        if not pred or not text:
            continue

        results = judge_predictions(text, pred, llm_client)
        for p in pred:
            name = p if isinstance(p, str) else p.get("name", "")
            if name:
                total_pred += 1
                if results.get(name, "invalid") == "valid":
                    total_valid += 1

        if i % 10 == 0:
            logger.info("judge progress: %d/%d", i, min(len(rows), max_samples))

    return {
        "llm_judge_precision": total_valid / max(total_pred, 1),
        "total_predicted": total_pred,
        "total_valid": total_valid,
        "judged_samples": min(len(rows), max_samples),
    }
