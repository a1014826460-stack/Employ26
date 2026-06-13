"""Automatically build a multiclass context-classifier dataset with local LLM labeling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from math import ceil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .context_labels import LABEL_TIE_BREAKER, LABEL_TO_ID
from .llm_labeling_utils import (
    extract_json_from_response,
    normalize_skill_key,
    run_prompt_pairs,
    safe_text,
    write_jsonl,
)
from .hard_skill_matcher import FlatHardSkillMatcher, load_flat_dictionary
from .regression_eval import load_regression_dataset


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_OUTPUT_PATH = "output/skill_extraction/context_classifier/context_dataset_llm.jsonl"
MAX_CANDIDATES_PER_PROMPT = 12

SYSTEM_PROMPT = """\
你是招聘硬技能候选审核专家。你的任务是判断“词典召回出的候选技能”在当前岗位文本里是否真的成立。

标签定义：
1. valid_hard_skill: 候选是明确的硬技能，且语义匹配岗位文本。
2. too_generic: 文本里出现的是泛化容器词，粒度太粗，不适合直接入词典。
3. wrong_alias_mapping: 文本中命中了某个 alias，但实际语义对应的是别的概念，不应映射到当前技能名。
4. not_skill: 文本里的命中不是硬技能，只是普通词、职责描述、软技能或其他无关内容。

判别规则：
1. 必须基于岗位原文判断，不要脑补。
2. 对“测试、办公软件、资格证书、专业知识、数据分析工具”这类泛词优先判为 too_generic。
3. 对 alias 错吸附的情况优先判为 wrong_alias_mapping。
4. 只有非常明确时才判为 valid_hard_skill。

只输出 JSON：
{"labels":[{"candidate_index":0,"label":"valid_hard_skill","reason":"简短原因"}]}
"""

USER_TEMPLATE = """\
样本ID: {sample_id}
岗位文本:
{text}

参考硬技能清单（弱参考，仍需以原文为准）:
{reference_skills}

候选技能列表:
{candidate_block}
"""


def _extract_label_items(payload: object) -> List[Dict]:
    """从 LLM 解析结果中提取候选标签列表。

    参数:
        payload: `extract_json_from_response()` 的返回值，可能是字典、列表或其他类型。

    返回:
        List[Dict]: 规范化后的标签对象列表。

    说明:
        为兼容不同输出风格，当前支持：
        - `{"labels": [...]}`
        - `{"data": [...]}`
        - `{"items": [...]}`
        - 直接返回 `[...]`
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for field_name in ("labels", "data", "items", "results"):
            field_value = payload.get(field_name)
            if isinstance(field_value, list):
                return [item for item in field_value if isinstance(item, dict)]
    return []


def _chunk_candidates(candidates: Sequence[Dict], chunk_size: int) -> List[List[Dict]]:
    """按 prompt 容量切分候选技能列表。

    参数:
        candidates: 单条样本的候选技能列表。
        chunk_size: 每个 prompt 最多容纳的候选数。

    返回:
        List[List[Dict]]: 切分后的候选块列表。
    """
    return [
        list(candidates[start : start + chunk_size])
        for start in range(0, len(candidates), chunk_size)
    ]


def _render_candidate_block(candidates: Sequence[Dict]) -> str:
    """将候选技能渲染成可直接放入 prompt 的文本块。

    参数:
        candidates: 候选技能对象列表。

    返回:
        str: 带编号的多行文本。
    """
    lines: List[str] = []
    for index, candidate in enumerate(candidates):
        lines.append(
            f"{index}. skill_name={candidate['skill_name']} | "
            f"matched_term={candidate['matched_term']} | "
            f"term_role={candidate['term_role']}"
        )
    return "\n".join(lines)


def _aggregate_label_votes(
    labels_by_vote: Sequence[Dict[int, Dict]],
    candidates: Sequence[Dict],
    min_vote_support: int,
) -> List[Dict]:
    """将多轮 LLM 标签投票聚合为稳定的多分类结果。

    参数:
        labels_by_vote: 同一批候选在多轮推理中的标签结果。
        candidates: 当前候选技能列表。
        min_vote_support: 标签被保留的最小支持票数。

    返回:
        List[Dict]: 聚合后的候选标签列表。
    """
    output_rows: List[Dict] = []
    for candidate_index, candidate in enumerate(candidates):
        label_counter: Counter[str] = Counter()
        reason_counter: Counter[str] = Counter()
        for vote in labels_by_vote:
            label_info = vote.get(candidate_index)
            if not label_info:
                continue
            label_counter[label_info["label"]] += 1
            reason = safe_text(label_info.get("reason", ""))
            if reason:
                reason_counter[reason] += 1

        if not label_counter:
            continue

        best_label = sorted(
            label_counter.items(),
            key=lambda item: (-item[1], LABEL_TIE_BREAKER.get(item[0], 99), item[0]),
        )[0][0]
        vote_count = int(label_counter[best_label])
        if vote_count < min_vote_support:
            continue

        output_rows.append(
            {
                "skill_name": candidate["skill_name"],
                "matched_term": candidate["matched_term"],
                "term_role": candidate["term_role"],
                "label": best_label,
                "label_reason": reason_counter.most_common(1)[0][0] if reason_counter else "",
                "label_vote_count": vote_count,
            }
        )
    return output_rows


def build_context_dataset_with_llm(
    regression_dataset_path: str | Path,
    dictionary_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    model_path: str | None = None,
    batch_size: int = 16,
    num_votes: int = 1,
    min_vote_support: int | None = None,
    max_candidates_per_prompt: int = MAX_CANDIDATES_PER_PROMPT,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
) -> Dict:
    """构建自动标注版上下文判别训练集。

    参数:
        regression_dataset_path: 回归集路径。
        dictionary_path: 平面技能词典路径。
        output_path: 输出 JSONL 路径。
        model_path: LLM 模型路径；为空时使用配置默认值。
        batch_size: vLLM 批量推理大小。
        num_votes: 每个 prompt 的投票轮数。
        min_vote_support: 标签被保留的最小支持票数。
        max_candidates_per_prompt: 单个 prompt 容纳的最大候选数。
        gpu_memory_utilization: 显存占比。
        max_model_len: 最大上下文长度。
        max_num_seqs: 最大并发序列数。

    返回:
        Dict: 数据集摘要，包括记录数、样本数、解析成功数和标签分布。
    """
    from .config import load_skill_extraction_config

    config = load_skill_extraction_config()
    resolved_model_path = str(model_path or config.llm_model_path)
    regression_rows = load_regression_dataset(regression_dataset_path)
    matcher = FlatHardSkillMatcher(load_flat_dictionary(dictionary_path))
    vote_support = int(min_vote_support or ceil(max(1, int(num_votes)) / 2))

    task_rows: List[Tuple[str, int, List[Dict]]] = []
    prompt_pairs: List[Tuple[str, str]] = []
    sample_lookup: Dict[str, Dict] = {}

    for row in regression_rows:
        raw_candidates = matcher.match_candidates(row.text)
        deduped_candidates: List[Dict] = []
        seen: set[Tuple[str, str, str]] = set()
        for candidate in raw_candidates:
            candidate_key = (
                normalize_skill_key(candidate["skill_name"]),
                normalize_skill_key(candidate["matched_term"]),
                normalize_skill_key(candidate["term_role"]),
            )
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            deduped_candidates.append(candidate)

        if not deduped_candidates:
            continue

        sample_lookup[row.sample_id] = {
            "sample_id": row.sample_id,
            "text": row.text,
            "gold_skills": row.gold_skills,
        }
        for chunk in _chunk_candidates(deduped_candidates, max_candidates_per_prompt):
            for vote_index in range(max(1, int(num_votes))):
                prompt_pairs.append(
                    (
                        SYSTEM_PROMPT,
                        USER_TEMPLATE.format(
                            sample_id=row.sample_id,
                            text=row.text,
                            reference_skills=json.dumps(row.gold_skills, ensure_ascii=False),
                            candidate_block=_render_candidate_block(chunk),
                        ),
                    )
                )
                task_rows.append((row.sample_id, vote_index, chunk))

    if not prompt_pairs:
        raise ValueError("No matcher candidates were produced from the regression dataset.")

    llm_outputs = run_prompt_pairs(
        model_path=resolved_model_path,
        prompt_pairs=prompt_pairs,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    grouped_votes: Dict[Tuple[str, str], List[Dict[int, Dict]]] = defaultdict(list)
    parse_success = 0
    for (sample_id, _vote_index, chunk), output_text in zip(task_rows, llm_outputs):
        parsed = extract_json_from_response(output_text)
        label_items = _extract_label_items(parsed)
        vote_map: Dict[int, Dict] = {}
        for item in label_items:
            try:
                candidate_index = int(item.get("candidate_index"))
            except (TypeError, ValueError):
                continue
            if candidate_index < 0 or candidate_index >= len(chunk):
                continue
            label = safe_text(item.get("label", ""))
            if label not in LABEL_TO_ID:
                continue
            vote_map[candidate_index] = {
                "label": label,
                "reason": safe_text(item.get("reason", "")),
            }
        if parsed is not None:
            parse_success += 1
        chunk_key = (
            sample_id,
            json.dumps(chunk, ensure_ascii=False, sort_keys=True),
        )
        grouped_votes[chunk_key].append(vote_map)

    output_rows: List[Dict] = []
    label_counter: Counter[str] = Counter()
    for (sample_id, chunk_key_json), vote_maps in grouped_votes.items():
        chunk = json.loads(chunk_key_json)
        merged_labels = _aggregate_label_votes(
            vote_maps,
            candidates=chunk,
            min_vote_support=vote_support,
        )
        sample_info = sample_lookup[sample_id]
        for item in merged_labels:
            label_counter[item["label"]] += 1
            output_rows.append(
                {
                    "sample_id": sample_id,
                    "text": sample_info["text"],
                    "job_title": "",
                    "skill_name": item["skill_name"],
                    "matched_term": item["matched_term"],
                    "term_role": item["term_role"],
                    "label": item["label"],
                    "label_reason": item["label_reason"],
                    "label_vote_count": item["label_vote_count"],
                    "reference_gold_skills": sample_info["gold_skills"],
                }
            )

    write_jsonl(output_path, output_rows)
    summary = {
        "output_path": str(output_path),
        "sample_count": len({row["sample_id"] for row in output_rows}),
        "record_count": len(output_rows),
        "parse_success": parse_success,
        "prompt_count": len(prompt_pairs),
        "label_distribution": dict(label_counter),
        "model_path": resolved_model_path,
        "num_votes": int(num_votes),
        "min_vote_support": vote_support,
    }
    logger.info(
        "Context dataset written to %s (%d records)",
        output_path,
        len(output_rows),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    返回:
        argparse.ArgumentParser: 已注册参数的 CLI 解析器。
    """
    parser = argparse.ArgumentParser(description="Use local LLM to build a context-classifier dataset.")
    parser.add_argument("--regression-dataset", required=True, help="Regression JSONL/CSV dataset path.")
    parser.add_argument("--dictionary", default="dicts/flat_skill_dictionary.json", help="Flat dictionary path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--model", default=None, help="LLM model path; defaults to config/database.yaml.")
    parser.add_argument("--batch-size", type=int, default=16, help="vLLM prompt batch size.")
    parser.add_argument("--num-votes", type=int, default=1, help="Number of LLM votes per prompt.")
    parser.add_argument("--min-vote-support", type=int, default=None, help="Minimum votes required to keep a label.")
    parser.add_argument("--max-candidates-per-prompt", type=int, default=MAX_CANDIDATES_PER_PROMPT, help="Candidate count per prompt.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="vLLM GPU memory utilization.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=48, help="vLLM max concurrent sequences.")
    return parser


def main() -> None:
    """命令行入口函数。

    负责解析参数、执行上下文训练集构建流程，并打印摘要结果。
    """
    args = build_parser().parse_args()
    summary = build_context_dataset_with_llm(
        regression_dataset_path=args.regression_dataset,
        dictionary_path=args.dictionary,
        output_path=args.output,
        model_path=args.model,
        batch_size=args.batch_size,
        num_votes=args.num_votes,
        min_vote_support=args.min_vote_support,
        max_candidates_per_prompt=args.max_candidates_per_prompt,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
