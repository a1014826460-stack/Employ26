"""Automatically build a regression dataset with Qwen3 + GPT final review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from math import ceil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .config import load_skill_extraction_config
from .llm_labeling_utils import (
    build_chat_prompts,
    extract_json_from_response,
    init_labeling_llm,
    load_requirement_match_rows,
    normalize_skill_key,
    prepare_labeling_frame,
    run_openai_prompt_pairs,
    run_prompt_pairs,
    safe_text,
    stratified_sample_frame,
    write_jsonl,
)
from .hard_skill_matcher import FlatHardSkillMatcher, load_flat_dictionary


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_OUTPUT_PATH = "output/skill_extraction/regression/flat_skill_regression_dataset.jsonl"
DEFAULT_SAMPLE_SIZE = 400
DEFAULT_DICTIONARY_PATH = "dicts/flat_skill_dictionary.json"
DEFAULT_MAX_FALLBACK_CANDIDATES = 8
DEFAULT_PROMPT_OUTPUT_DIR = "output/skill_extraction/regression/prompts"
DEFAULT_OPENAI_REVIEW_MODEL = "openai/gpt-5.4-mini"
GENERIC_SKILL_NAMES = {
    "财务软件",
    "办公软件",
    "数据分析工具",
    "专业知识",
    "理论基础",
    "知识基础",
    "测试",
    "测试仪器",
}
GENERIC_CERTIFICATE_NAMES = {
    "英语4级",
    "英语六级",
    "大学英语四级",
    "大学英语六级",
    "CET4",
    "CET6",
}

SYSTEM_PROMPT = """\
你是招聘硬技能标注专家。请从岗位文本中抽取可以直接落入技能词典的硬技能。
标注规则：
1. 只保留硬技能：编程语言、框架、数据库、工具软件、办公软件、设备/仪器、工艺方法、具体证书/资质。
2. 排除软技能、学历、年限、岗位职责、福利待遇、泛化容器词。
3. 不要输出“测试”“办公软件”“数据分析工具”“资格证书”“专业知识”这类过泛名称。
4. evidence 必须是岗位原文中的连续子串，不能编造，也不要跨句拼接。
5. name 需要标准化；如果文本里出现别名，也要统一成更规范的技能名。
6. skill_type 只能从以下枚举中选择：编程语言、框架、数据库、工具软件、办公软件、设备/仪器、工艺方法、证书/资质。
7. 如果文本中没有明确硬技能，返回空列表。
只输出 JSON：{"skills":[{"name":"标准技能名","evidence":"原文证据","skill_type":"技能类别"}]}
"""

USER_TEMPLATE = """\
样本ID: {sample_id}
岗位名称: {job_title}
职业中类: {occupation_title}
职业编码: {occupation_code}

岗位文本：{text}
"""

SECOND_REVIEW_SYSTEM_PROMPT = """\
你是招聘硬技能复核专家。上一轮抽取结果为空，请重新检查岗位文本。
要求：
1. 只抽取明确、可落词典的硬技能。
2. 不要输出软技能、职责描述、学历年限和泛化容器词。
3. 如果候选提示里有具体工具、证书、设备、框架或工艺，请重点复核。
4. evidence 必须是原文中的连续子串。
5. skill_type 只能从以下枚举中选择：编程语言、框架、数据库、工具软件、办公软件、设备/仪器、工艺方法、证书/资质。
6. 如果文本中确实没有明确硬技能，返回空列表。
只输出 JSON：{"skills":[{"name":"标准技能名","evidence":"原文证据","skill_type":"技能类别"}]}
"""

SECOND_REVIEW_USER_TEMPLATE = """\
样本ID: {sample_id}
岗位名称: {job_title}
职业中类: {occupation_title}
职业编码: {occupation_code}

岗位文本：{text}

高置信候选提示（仅作辅助参考，不代表一定正确）：{candidate_hints}
"""

FINAL_REVIEW_SYSTEM_PROMPT = """\
你是招聘硬技能终审专家。现在你会看到岗位原文，以及 Qwen3 和规则链路给出的候选技能。
你的任务是输出这条岗位样本的最终硬技能金标。
要求：
1. 你可以删除错误候选，也可以补充少量原文中非常明确但候选漏掉的硬技能。
2. 只保留可直接落词典的硬技能：编程语言、框架、数据库、工具软件、办公软件、设备/仪器、工艺方法、证书/资质。
3. 不要保留软技能、职责描述、泛词、学科名称、专业知识、管理能力、业务流程。
4. evidence 必须是岗位原文中的最小连续子串，不要跨句拼接，不要复用过大的整段文本。
5. name 必须尽量规范化，例如 springboot -> Spring Boot，mongoDB -> MongoDB。
6. skill_type 只能从以下枚举中选择：编程语言、框架、数据库、工具软件、办公软件、设备/仪器、工艺方法、证书/资质。
7. 如果最终没有明确硬技能，返回空列表。
只输出 JSON：{"skills":[{"name":"标准技能名","evidence":"原文证据","skill_type":"技能类别"}]}
"""

FINAL_REVIEW_USER_TEMPLATE = """\
样本ID: {sample_id}
岗位名称: {job_title}
职业中类: {occupation_title}
职业编码: {occupation_code}

岗位文本：{text}

候选技能（来自 Qwen3 与规则链路，仅供参考）：
{candidate_block}
"""


def _extract_skill_items(payload: object) -> List[Dict]:
    """Extract skill dictionaries from several response JSON layouts.

    Different LLM stages may emit either a bare list or an object whose
    payload is stored under fields such as ``skills`` or ``items``.
    This helper normalizes those variants into a plain ``List[Dict]`` so
    downstream cleaning and vote aggregation logic can stay stage-agnostic.
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for field_name in ("skills", "data", "items", "results"):
            field_value = payload.get(field_name)
            if isinstance(field_value, list):
                return [item for item in field_value if isinstance(item, dict)]
    return []


def _clean_skill_item(item: Dict, text: str) -> Dict | None:
    """Validate and normalize one skill item against the source JD text.

    The labeling pipeline only keeps grounded hard-skill annotations. A record
    is accepted when:
    1. ``name`` is non-empty.
    2. ``evidence`` is non-empty.
    3. ``evidence`` appears verbatim in the original job-description text.

    Returning ``None`` signals that the model output is unusable for gold-label
    construction and should be dropped from the current stage result.
    """
    name = safe_text(item.get("name", ""))
    evidence = safe_text(item.get("evidence", ""))
    skill_type = safe_text(item.get("skill_type", ""))
    if not name or not evidence or evidence not in text:
        return None
    if name in GENERIC_SKILL_NAMES:
        return None
    if name in GENERIC_CERTIFICATE_NAMES:
        return None
    if skill_type == "证书/资质" and name.endswith("英语四级"):
        return None
    if skill_type == "证书/资质" and name.endswith("英语六级"):
        return None
    return {"name": name, "evidence": evidence, "skill_type": skill_type}


def _aggregate_votes(
    vote_results: Sequence[Sequence[Dict]],
    min_vote_support: int,
) -> List[Dict]:
    """Merge repeated initial-pass votes into one stable candidate list.

    Each vote is first de-duplicated by normalized skill name so one noisy
    completion cannot vote for the same skill multiple times. Skills whose
    support is below ``min_vote_support`` are removed. For retained skills, the
    function keeps the most common ``name``, ``evidence``, and ``skill_type``
    observed across votes and appends ``vote_count`` for later debugging and
    final-review prompts.
    """
    counts: Counter[str] = Counter()
    items_by_key: Dict[str, List[Dict]] = defaultdict(list)

    for vote in vote_results:
        seen_in_vote: set[str] = set()
        for item in vote:
            key = normalize_skill_key(item.get("name", ""))
            if not key or key in seen_in_vote:
                continue
            seen_in_vote.add(key)
            counts[key] += 1
            items_by_key[key].append(item)

    merged: List[Dict] = []
    for key, vote_count in counts.items():
        if vote_count < min_vote_support:
            continue
        candidates = items_by_key[key]
        name_counter = Counter(safe_text(item.get("name", "")) for item in candidates)
        evidence_counter = Counter(safe_text(item.get("evidence", "")) for item in candidates)
        type_counter = Counter(safe_text(item.get("skill_type", "")) for item in candidates)
        merged.append(
            {
                "name": name_counter.most_common(1)[0][0],
                "evidence": evidence_counter.most_common(1)[0][0],
                "skill_type": type_counter.most_common(1)[0][0],
                "vote_count": vote_count,
            }
        )

    merged.sort(key=lambda item: (-int(item["vote_count"]), item["name"].casefold()))
    return merged


def _score_fallback_candidate(candidate: Dict) -> int:
    """Assign a conservative heuristic score to a matcher candidate.

    The score intentionally favors candidates that look like concrete,
    dictionary-ready hard skills, such as exact name matches, longer terms, and
    strings containing ASCII letters or digits. Low-confidence matches are
    filtered out before they are exposed to the second-review prompt.
    """
    skill_name = safe_text(candidate.get("skill_name", ""))
    matched_term = safe_text(candidate.get("matched_term", ""))
    term_role = safe_text(candidate.get("term_role", ""))

    score = 0
    if term_role == "name":
        score += 3
    if matched_term.casefold() == skill_name.casefold():
        score += 2
    if len(matched_term) >= 4:
        score += 1
    if any(char.isascii() and char.isalpha() for char in matched_term):
        score += 1
    if any(char.isdigit() for char in matched_term):
        score += 1
    if any(token in matched_term for token in ("证", "证书", "资格", "软件", "系统")):
        score += 1
    return score


def _select_high_confidence_candidates(
    matcher: FlatHardSkillMatcher,
    text: str,
    max_candidates: int = DEFAULT_MAX_FALLBACK_CANDIDATES,
) -> List[Dict]:
    """Pick a compact, high-confidence matcher hint list for second review.

    The flat-skill matcher may return many raw matches. This function scores and
    ranks them, enriches each surviving match with dictionary ``skill_type``,
    and returns only the top ``max_candidates`` records. The goal is to give
    the LLM a few precise reminders without overwhelming the review prompt.
    """
    raw_candidates = matcher.match_candidates(text)
    ranked_candidates: List[Tuple[int, Dict]] = []
    for candidate in raw_candidates:
        score = _score_fallback_candidate(candidate)
        if score < 4:
            continue
        skill = matcher.find_skill_by_name(candidate["skill_name"]) or {}
        ranked_candidates.append(
            (
                score,
                {
                    "skill_name": candidate["skill_name"],
                    "matched_term": candidate["matched_term"],
                    "term_role": candidate["term_role"],
                    "skill_type": safe_text(skill.get("skill_type", "")),
                },
            )
        )

    ranked_candidates.sort(
        key=lambda item: (
            -item[0],
            item[1]["skill_name"].casefold(),
            item[1]["matched_term"].casefold(),
        )
    )
    return [item[1] for item in ranked_candidates[: max(1, int(max_candidates))]]


def _build_candidate_hint_text(candidates: Sequence[Dict]) -> str:
    """Render matcher candidates into a compact second-review hint block."""
    if not candidates:
        return "无"
    lines: List[str] = []
    for index, candidate in enumerate(candidates, start=1):
        skill_type = safe_text(candidate.get("skill_type", "")) or "未知类别"
        lines.append(
            f"{index}. 技能名={candidate['skill_name']} | 命中词={candidate['matched_term']} | "
            f"来源={candidate['term_role']} | 类型={skill_type}"
        )
    return "\n".join(lines)


def _build_final_candidate_block(candidates: Sequence[Dict]) -> str:
    """Render the draft candidate list that is shown to the final-review model."""
    if not candidates:
        return "无"
    lines: List[str] = []
    for index, item in enumerate(candidates, start=1):
        vote_count = item.get("vote_count")
        vote_text = f" | 支持票数={vote_count}" if vote_count else ""
        lines.append(
            f"{index}. 技能名={safe_text(item.get('name', ''))} | "
            f"证据={safe_text(item.get('evidence', ''))} | "
            f"类型={safe_text(item.get('skill_type', ''))}{vote_text}"
        )
    return "\n".join(lines)


def _build_prompt_export_paths(
    output_path: str | Path,
    prompt_output_dir: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Build JSONL paths used to persist prompt dumps for all LLM stages.

    Files are named from the final dataset stem so one dataset run produces a
    predictable trio of prompt artifacts: initial-pass prompts, second-review
    prompts, and OpenAI final-review prompts.
    """
    dataset_path = Path(output_path)
    export_dir = Path(prompt_output_dir) if prompt_output_dir else dataset_path.parent / "prompts"
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    return (
        export_dir / f"{stem}.initial_prompts.jsonl",
        export_dir / f"{stem}.second_review_prompts.jsonl",
        export_dir / f"{stem}.openai_final_review_prompts.jsonl",
    )


def _build_stage_output_paths(
    output_path: str | Path,
    prompt_output_dir: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Build JSONL paths used to persist raw stage outputs for debugging.

    The output directory intentionally mirrors the prompt-dump directory so a
    single run's prompts and responses stay side by side. Each file stores the
    raw model text plus parsed and cleaned intermediate results for one stage.
    """
    dataset_path = Path(output_path)
    export_dir = Path(prompt_output_dir) if prompt_output_dir else dataset_path.parent / "prompts"
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    return (
        export_dir / f"{stem}.initial_outputs.jsonl",
        export_dir / f"{stem}.second_review_outputs.jsonl",
        export_dir / f"{stem}.openai_final_review_outputs.jsonl",
    )


def _write_qwen_prompt_dump(
    path: Path,
    stage: str,
    prompt_records: Sequence[Dict],
    llm=None,
) -> None:
    """Persist Qwen prompt records and rendered prompts for offline replay.

    When a live vLLM engine is available, the helper also stores the tokenizer's
    rendered chat template so the exact generation input can be reproduced
    during manual debugging.
    """
    rows = [dict(record) for record in prompt_records]
    if llm is not None and rows:
        rendered_prompts = build_chat_prompts(
            llm,
            [(row["system_prompt"], row["user_prompt"]) for row in rows],
        )
        for row, rendered_prompt in zip(rows, rendered_prompts):
            row["rendered_prompt"] = rendered_prompt
    write_jsonl(path, rows)
    logger.info("Saved %s prompts to %s (%d rows)", stage, path, len(rows))


def _write_prompt_dump(path: Path, stage: str, prompt_records: Sequence[Dict]) -> None:
    """Persist prompt records for stages that do not need chat-template rendering."""
    write_jsonl(path, prompt_records)
    logger.info("Saved %s prompts to %s (%d rows)", stage, path, len(prompt_records))


def _build_stage_output_record(
    *,
    stage: str,
    row: Dict,
    raw_output_text: str,
    parsed_payload: Dict | List | None,
    cleaned_skill_items: Sequence[Dict],
    vote_index: int | None = None,
    fallback_candidates: Sequence[Dict] | None = None,
    draft_skill_items: Sequence[Dict] | None = None,
) -> Dict:
    """Build one normalized debug row shared by all stage-output JSONL files.

    The three labeling stages expose slightly different context, but for manual
    inspection it is much easier when the common sample metadata and parse
    result fields always appear in the same order. Stage-specific context is
    appended at the end and defaults to empty values so every JSONL file keeps
    a stable, easy-to-scan schema.
    """
    return {
        "stage": stage,
        "sample_id": row["sample_id"],
        "vote_index": vote_index,
        "job_title": row["job_title"],
        "occupation_title": row["occupation_title"],
        "occupation_code": row["occupation_code"],
        "text": row["text"],
        "parse_success": parsed_payload is not None,
        "extracted_skill_items": _extract_skill_items(parsed_payload),
        "cleaned_skill_items": list(cleaned_skill_items),
        "raw_output_text": raw_output_text,
        "parsed_payload": parsed_payload,
        "fallback_candidates": list(fallback_candidates or []),
        "draft_skill_items": list(draft_skill_items or []),
    }


def _write_stage_output_dump(path: Path, stage: str, output_records: Sequence[Dict]) -> None:
    """Persist per-stage model outputs for manual inspection.

    Each row is designed to answer the common debugging questions:
    what prompt instance produced this completion, what the raw model text was,
    whether JSON parsing succeeded, and what cleaned skill items survived.
    Records are normalized through ``_build_stage_output_record`` so all stage
    output files share the same core field order.
    """
    write_jsonl(path, output_records)
    logger.info("Saved %s outputs to %s (%d rows)", stage, path, len(output_records))


def _run_initial_labeling(
    sampled_rows: Sequence[Dict],
    llm,
    resolved_model_path: str,
    batch_size: int,
    num_votes: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
) -> tuple[Dict[str, List[List[Dict]]], int, int, List[Dict], List[Dict]]:
    """Run the main Qwen pass and capture prompt/output artifacts.

    For every sampled row, the function sends ``num_votes`` independent prompt
    instances to Qwen, parses the responses, cleans grounded skill items, and
    groups vote results by ``sample_id``. It returns both the structured vote
    map used by downstream aggregation and a row-wise debug log containing raw
    completions from the initial stage.
    """
    prompt_pairs: List[Tuple[str, str]] = []
    prompt_records: List[Dict] = []
    output_records: List[Dict] = []
    task_rows: List[Tuple[str, int]] = []
    sample_lookup = {row["sample_id"]: row for row in sampled_rows}

    for row in sampled_rows:
        for vote_index in range(max(1, int(num_votes))):
            system_prompt = SYSTEM_PROMPT
            user_prompt = USER_TEMPLATE.format(
                sample_id=row["sample_id"],
                job_title=row["job_title"] or "未知岗位",
                occupation_title=row["occupation_title"] or "未知职业",
                occupation_code=row["occupation_code"] or "",
                text=row["text"],
            )
            prompt_pairs.append((system_prompt, user_prompt))
            prompt_records.append(
                {
                    "stage": "initial_vote",
                    "sample_id": row["sample_id"],
                    "vote_index": vote_index,
                    "job_title": row["job_title"],
                    "occupation_title": row["occupation_title"],
                    "occupation_code": row["occupation_code"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            task_rows.append((row["sample_id"], vote_index))

    llm_outputs = run_prompt_pairs(
        model_path=resolved_model_path,
        prompt_pairs=prompt_pairs,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        llm=llm,
    )

    votes_by_sample: Dict[str, List[List[Dict]]] = defaultdict(list)
    parse_success = 0
    for (sample_id, _vote_index), output_text in zip(task_rows, llm_outputs):
        parsed = extract_json_from_response(output_text)
        skill_items = _extract_skill_items(parsed)
        cleaned_items = [
            cleaned
            for cleaned in (
                _clean_skill_item(item, sample_lookup[sample_id]["text"])
                for item in skill_items
            )
            if cleaned is not None
        ]
        votes_by_sample[sample_id].append(cleaned_items)
        output_records.append(
            _build_stage_output_record(
                stage="initial_vote",
                row=sample_lookup[sample_id],
                vote_index=_vote_index,
                raw_output_text=output_text,
                parsed_payload=parsed,
                cleaned_skill_items=cleaned_items,
            )
        )
        if parsed is not None:
            parse_success += 1

    return votes_by_sample, parse_success, len(prompt_pairs), prompt_records, output_records


def _run_second_review(
    empty_rows: Sequence[Dict],
    matcher: FlatHardSkillMatcher,
    llm,
    resolved_model_path: str,
    batch_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_fallback_candidates: int,
) -> tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], int, int, List[Dict], List[Dict]]:
    """Run the second-pass Qwen review for samples that remained empty.

    Only rows with no skills after vote aggregation reach this stage. The
    function injects a conservative matcher-derived hint block, parses the new
    completions, and returns both cleaned skill candidates and detailed debug
    rows so we can inspect how the second review behaved for each sample.
    """
    if not empty_rows:
        return {}, {}, 0, 0, [], []

    prompt_pairs: List[Tuple[str, str]] = []
    prompt_records: List[Dict] = []
    output_records: List[Dict] = []
    task_rows: List[str] = []
    fallback_candidates_by_sample: Dict[str, List[Dict]] = {}

    for row in empty_rows:
        fallback_candidates = _select_high_confidence_candidates(
            matcher=matcher,
            text=row["text"],
            max_candidates=max_fallback_candidates,
        )
        candidate_hints = _build_candidate_hint_text(fallback_candidates)
        fallback_candidates_by_sample[row["sample_id"]] = fallback_candidates
        system_prompt = SECOND_REVIEW_SYSTEM_PROMPT
        user_prompt = SECOND_REVIEW_USER_TEMPLATE.format(
            sample_id=row["sample_id"],
            job_title=row["job_title"] or "未知岗位",
            occupation_title=row["occupation_title"] or "未知职业",
            occupation_code=row["occupation_code"] or "",
            text=row["text"],
            candidate_hints=candidate_hints,
        )
        prompt_pairs.append((system_prompt, user_prompt))
        prompt_records.append(
            {
                "stage": "second_review",
                "sample_id": row["sample_id"],
                "job_title": row["job_title"],
                "occupation_title": row["occupation_title"],
                "occupation_code": row["occupation_code"],
                "candidate_hints": candidate_hints,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        task_rows.append(row["sample_id"])

    llm_outputs = run_prompt_pairs(
        model_path=resolved_model_path,
        prompt_pairs=prompt_pairs,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        llm=llm,
    )

    cleaned_by_sample: Dict[str, List[Dict]] = {}
    parse_success = 0
    row_lookup = {row["sample_id"]: row for row in empty_rows}
    for sample_id, output_text in zip(task_rows, llm_outputs):
        parsed = extract_json_from_response(output_text)
        skill_items = _extract_skill_items(parsed)
        cleaned_items = [
            cleaned
            for cleaned in (
                _clean_skill_item(item, row_lookup[sample_id]["text"])
                for item in skill_items
            )
            if cleaned is not None
        ]
        cleaned_by_sample[sample_id] = cleaned_items
        output_records.append(
            _build_stage_output_record(
                stage="second_review",
                row=row_lookup[sample_id],
                raw_output_text=output_text,
                parsed_payload=parsed,
                cleaned_skill_items=cleaned_items,
                fallback_candidates=fallback_candidates_by_sample.get(sample_id, []),
            )
        )
        if parsed is not None:
            parse_success += 1

    return (
        cleaned_by_sample,
        fallback_candidates_by_sample,
        parse_success,
        len(prompt_pairs),
        prompt_records,
        output_records,
    )


def _build_fallback_skill_items(candidates: Sequence[Dict]) -> List[Dict]:
    """Convert conservative matcher hits into the same shape as LLM skill items.

    This path is only used when both the initial vote aggregation and optional
    second review fail to recover any skills. The function keeps one entry per
    normalized skill name and marks each item with ``vote_count=1`` so the
    downstream final-review stage can treat it like a lightweight draft label.
    """
    items: List[Dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        skill_name = safe_text(candidate.get("skill_name", ""))
        evidence = safe_text(candidate.get("matched_term", ""))
        if not skill_name or not evidence:
            continue
        key = normalize_skill_key(skill_name)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "name": skill_name,
                "evidence": evidence,
                "skill_type": safe_text(candidate.get("skill_type", "")),
                "vote_count": 1,
            }
        )
    return items


def _run_openai_final_review(
    sampled_rows: Sequence[Dict],
    draft_items_by_sample: Dict[str, List[Dict]],
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> tuple[Dict[str, List[Dict]], int, List[Dict], List[Dict]]:
    """Run the OpenAI final-review stage over draft labels from prior stages.

    The prompt contains the original JD text plus the current draft candidate
    list generated by Qwen and rule fallback. The function returns cleaned final
    labels per sample, the number of JSON-parse successes, the exported prompt
    records, and a raw per-sample output log for manual auditing.
    """
    prompt_pairs: List[Tuple[str, str]] = []
    prompt_records: List[Dict] = []
    output_records: List[Dict] = []

    for row in sampled_rows:
        sample_id = row["sample_id"]
        candidate_items = draft_items_by_sample.get(sample_id, [])
        candidate_block = _build_final_candidate_block(candidate_items)
        system_prompt = FINAL_REVIEW_SYSTEM_PROMPT
        user_prompt = FINAL_REVIEW_USER_TEMPLATE.format(
            sample_id=sample_id,
            job_title=row["job_title"] or "未知岗位",
            occupation_title=row["occupation_title"] or "未知职业",
            occupation_code=row["occupation_code"] or "",
            text=row["text"],
            candidate_block=candidate_block,
        )
        prompt_pairs.append((system_prompt, user_prompt))
        prompt_records.append(
            {
                "stage": "openai_final_review",
                "sample_id": sample_id,
                "job_title": row["job_title"],
                "occupation_title": row["occupation_title"],
                "occupation_code": row["occupation_code"],
                "candidate_block": candidate_block,
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    outputs = run_openai_prompt_pairs(
        prompt_pairs=prompt_pairs,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )

    reviewed_by_sample: Dict[str, List[Dict]] = {}
    parse_success = 0
    row_lookup = {row["sample_id"]: row for row in sampled_rows}
    for row, output_text in zip(sampled_rows, outputs):
        sample_id = row["sample_id"]
        parsed = extract_json_from_response(output_text)
        skill_items = _extract_skill_items(parsed)
        cleaned_items = [
            cleaned
            for cleaned in (
                _clean_skill_item(item, row_lookup[sample_id]["text"])
                for item in skill_items
            )
            if cleaned is not None
        ]
        reviewed_by_sample[sample_id] = cleaned_items
        output_records.append(
            _build_stage_output_record(
                stage="openai_final_review",
                row=row_lookup[sample_id],
                raw_output_text=output_text,
                parsed_payload=parsed,
                cleaned_skill_items=cleaned_items,
                draft_skill_items=draft_items_by_sample.get(sample_id, []),
            )
        )
        if parsed is not None:
            parse_success += 1

    return reviewed_by_sample, parse_success, prompt_records, output_records


def build_regression_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    source_table: str | None = None,
    dictionary_path: str | Path = DEFAULT_DICTIONARY_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    source_limit_rows: int | None = None,
    limit: int | None = None,
    seed: int = 42,
    model_path: str | None = None,
    batch_size: int = 16,
    max_text_chars: int = 900,
    min_text_chars: int = 20,
    num_votes: int = 1,
    min_vote_support: int | None = None,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
    enable_second_review: bool = True,
    enable_high_confidence_fallback: bool = True,
    max_fallback_candidates: int = DEFAULT_MAX_FALLBACK_CANDIDATES,
    prompt_output_dir: str | Path | None = DEFAULT_PROMPT_OUTPUT_DIR,
    save_stage_outputs: bool = False,
    use_openai_final_review: bool = False,
    openai_review_model: str = DEFAULT_OPENAI_REVIEW_MODEL,
    openai_max_output_tokens: int = 2048,
    openai_reasoning_effort: str = "low",
) -> Dict:
    """Build a regression dataset with multi-stage LLM labeling.

    The pipeline is:
    1. Load and normalize JD rows from DuckDB.
    2. Stratified-sample rows for labeling.
    3. Run the initial Qwen pass, optionally with multiple votes.
    4. Re-review still-empty rows with matcher hints.
    5. Optionally recover a few labels from conservative matcher fallback.
    6. Optionally ask an OpenAI model to perform final review.
    7. Write the final dataset plus optional prompt/output debug artifacts.

    When ``save_stage_outputs`` is enabled, each LLM stage also writes a JSONL
    file containing the raw completion text, parsed JSON payload, and cleaned
    skill items for every prompt instance. This is intended for manual error
    analysis rather than for model replay.
    """
    config = load_skill_extraction_config()
    resolved_model_path = str(model_path or config.llm_model_path)
    resolved_source_limit = source_limit_rows if source_limit_rows is not None else limit

    raw_df = load_requirement_match_rows(
        config=config,
        source_table=source_table,
        limit=resolved_source_limit,
    )
    prepared_df = prepare_labeling_frame(
        raw_df,
        max_text_chars=max_text_chars,
        min_text_chars=min_text_chars,
    )
    sampled_df = stratified_sample_frame(prepared_df, sample_size=sample_size, seed=seed)
    sampled_rows = sampled_df.to_dict(orient="records")
    if not sampled_rows:
        raise ValueError("No valid rows were found for regression dataset labeling.")

    matcher = FlatHardSkillMatcher(load_flat_dictionary(dictionary_path))
    vote_support = int(min_vote_support or ceil(max(1, int(num_votes)) / 2))
    initial_prompt_path, second_prompt_path, openai_prompt_path = _build_prompt_export_paths(
        output_path=output_path,
        prompt_output_dir=prompt_output_dir,
    )
    initial_output_path: Path | None = None
    second_output_path: Path | None = None
    openai_output_path: Path | None = None
    if save_stage_outputs:
        (
            initial_output_path,
            second_output_path,
            openai_output_path,
        ) = _build_stage_output_paths(
            output_path=output_path,
            prompt_output_dir=prompt_output_dir,
        )

    qwen_llm = init_labeling_llm(
        model_path=resolved_model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    (
        initial_votes_by_sample,
        parse_success,
        prompt_count,
        initial_prompt_records,
        initial_output_records,
    ) = _run_initial_labeling(
        sampled_rows=sampled_rows,
        llm=qwen_llm,
        resolved_model_path=resolved_model_path,
        batch_size=batch_size,
        num_votes=num_votes,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    _write_qwen_prompt_dump(initial_prompt_path, "initial_vote", initial_prompt_records, llm=qwen_llm)
    if save_stage_outputs and initial_output_path is not None:
        _write_stage_output_dump(initial_output_path, "initial_vote", initial_output_records)

    aggregated_by_sample: Dict[str, List[Dict]] = {
        row["sample_id"]: _aggregate_votes(
            initial_votes_by_sample.get(row["sample_id"], []),
            min_vote_support=vote_support,
        )
        for row in sampled_rows
    }

    empty_rows = [row for row in sampled_rows if not aggregated_by_sample.get(row["sample_id"], [])]
    logger.info(
        "Initial labeling finished: %d/%d samples are still empty after vote aggregation",
        len(empty_rows),
        len(sampled_rows),
    )

    second_review_items_by_sample: Dict[str, List[Dict]] = {}
    fallback_candidates_by_sample: Dict[str, List[Dict]] = {}
    second_review_parse_success = 0
    second_review_prompt_count = 0
    second_review_prompt_records: List[Dict] = []
    second_review_output_records: List[Dict] = []
    if enable_second_review and empty_rows:
        (
            second_review_items_by_sample,
            fallback_candidates_by_sample,
            second_review_parse_success,
            second_review_prompt_count,
            second_review_prompt_records,
            second_review_output_records,
        ) = _run_second_review(
            empty_rows=empty_rows,
            matcher=matcher,
            llm=qwen_llm,
            resolved_model_path=resolved_model_path,
            batch_size=batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_fallback_candidates=max_fallback_candidates,
        )
    _write_qwen_prompt_dump(second_prompt_path, "second_review", second_review_prompt_records, llm=qwen_llm)
    if save_stage_outputs and second_output_path is not None:
        _write_stage_output_dump(second_output_path, "second_review", second_review_output_records)

    draft_items_by_sample: Dict[str, List[Dict]] = {}
    draft_source_by_sample: Dict[str, str] = {}
    second_review_recovered_count = 0
    fallback_recovered_count = 0

    for row in sampled_rows:
        sample_id = row["sample_id"]
        aggregated_items = aggregated_by_sample.get(sample_id, [])
        draft_source = "initial_vote"

        if not aggregated_items and enable_second_review:
            reviewed_items = second_review_items_by_sample.get(sample_id, [])
            if reviewed_items:
                aggregated_items = reviewed_items
                second_review_recovered_count += 1
                draft_source = "second_review"

        if not aggregated_items and enable_high_confidence_fallback:
            fallback_items = _build_fallback_skill_items(
                fallback_candidates_by_sample.get(sample_id, [])
            )
            if fallback_items:
                aggregated_items = fallback_items
                fallback_recovered_count += 1
                draft_source = "matcher_fallback"

        draft_items_by_sample[sample_id] = aggregated_items
        draft_source_by_sample[sample_id] = draft_source

    openai_reviewed_by_sample: Dict[str, List[Dict]] = {}
    openai_review_parse_success = 0
    openai_prompt_records: List[Dict] = []
    openai_output_records: List[Dict] = []
    if use_openai_final_review:
        (
            openai_reviewed_by_sample,
            openai_review_parse_success,
            openai_prompt_records,
            openai_output_records,
        ) = _run_openai_final_review(
            sampled_rows=sampled_rows,
            draft_items_by_sample=draft_items_by_sample,
            model=openai_review_model,
            max_output_tokens=openai_max_output_tokens,
            reasoning_effort=openai_reasoning_effort,
        )
    _write_prompt_dump(openai_prompt_path, "openai_final_review", openai_prompt_records)
    if save_stage_outputs and openai_output_path is not None:
        _write_stage_output_dump(openai_output_path, "openai_final_review", openai_output_records)

    output_rows: List[Dict] = []
    total_gold_skills = 0
    openai_final_review_changed_count = 0

    for row in sampled_rows:
        sample_id = row["sample_id"]
        draft_items = draft_items_by_sample.get(sample_id, [])
        final_items = draft_items
        final_source = draft_source_by_sample.get(sample_id, "initial_vote")
        if use_openai_final_review:
            reviewed_items = openai_reviewed_by_sample.get(sample_id, [])
            final_items = reviewed_items
            final_source = "openai_final_review"
            if json.dumps(draft_items, ensure_ascii=False, sort_keys=True) != json.dumps(
                reviewed_items,
                ensure_ascii=False,
                sort_keys=True,
            ):
                openai_final_review_changed_count += 1

        gold_skills = [item["name"] for item in final_items]
        total_gold_skills += len(gold_skills)
        output_rows.append(
            {
                "sample_id": sample_id,
                "text": row["text"],
                "job_title": row["job_title"],
                "occupation_title": row["occupation_title"],
                "occupation_code": row["occupation_code"],
                "gold_skills": gold_skills,
                "gold_skill_items": final_items,
                "llm_vote_count": int(num_votes),
                "llm_min_vote_support": vote_support,
                "label_source": final_source,
                "draft_label_source": draft_source_by_sample.get(sample_id, "initial_vote"),
            }
        )

    write_jsonl(output_path, output_rows)
    summary = {
        "output_path": str(output_path),
        "source_table": source_table or config.requirement_match_table,
        "dictionary_path": str(dictionary_path),
        "source_limit_rows": resolved_source_limit,
        "sample_count": len(output_rows),
        "parse_success": parse_success + second_review_parse_success + openai_review_parse_success,
        "prompt_count": prompt_count + second_review_prompt_count + len(openai_prompt_records),
        "avg_gold_skills_per_sample": total_gold_skills / max(len(output_rows), 1),
        "model_path": resolved_model_path,
        "num_votes": int(num_votes),
        "min_vote_support": vote_support,
        "initial_empty_count": len(empty_rows),
        "second_review_recovered_count": second_review_recovered_count,
        "fallback_recovered_count": fallback_recovered_count,
        "save_stage_outputs": save_stage_outputs,
        "use_openai_final_review": use_openai_final_review,
        "openai_review_model": openai_review_model if use_openai_final_review else "",
        "openai_final_review_changed_count": openai_final_review_changed_count,
        "initial_prompt_dump_path": str(initial_prompt_path),
        "second_review_prompt_dump_path": str(second_prompt_path),
        "openai_final_review_prompt_dump_path": str(openai_prompt_path),
        "initial_output_dump_path": str(initial_output_path) if initial_output_path else "",
        "second_review_output_dump_path": str(second_output_path) if second_output_path else "",
        "openai_final_review_output_dump_path": str(openai_output_path) if openai_output_path else "",
    }
    logger.info(
        "Regression dataset written to %s (%d samples, avg skills %.2f, second-review recovered=%d, fallback recovered=%d, openai changed=%d)",
        output_path,
        len(output_rows),
        summary["avg_gold_skills_per_sample"],
        second_review_recovered_count,
        fallback_recovered_count,
        openai_final_review_changed_count,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for regression-dataset generation.

    The parser exposes switches for the three-stage labeling workflow:
    initial Qwen voting, optional second review and matcher fallback, and
    optional OpenAI final review. It also exposes debug-export controls so the
    same script can be used for both production data generation and manual
    pipeline inspection.
    """
    parser = argparse.ArgumentParser(description="Use Qwen3 and GPT final review to build a regression dataset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--source-table", default=None, help="DuckDB source table.")
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY_PATH,
        help="Flat skill dictionary path used for fallback.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Final number of rows to label after sampling.",
    )
    parser.add_argument(
        "--source-limit-rows",
        "--limit",
        dest="source_limit_rows",
        type=int,
        default=None,
        help="Limit the number of rows read from DuckDB before sampling. Useful for quick small-sample tests.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model", default=None, help="Qwen model path; defaults to config/database.yaml.")
    parser.add_argument("--batch-size", type=int, default=16, help="Qwen vLLM prompt batch size.")
    parser.add_argument("--max-text-chars", type=int, default=900, help="Maximum chars kept per sample.")
    parser.add_argument("--min-text-chars", type=int, default=20, help="Minimum chars required per sample.")
    parser.add_argument("--num-votes", type=int, default=1, help="Number of Qwen votes per sample in the initial pass.")
    parser.add_argument("--min-vote-support", type=int, default=None, help="Minimum votes required to keep a skill.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="Qwen vLLM GPU memory utilization.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Qwen vLLM max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Qwen vLLM max concurrent sequences.")
    parser.add_argument("--disable-second-review", action="store_true", help="Disable the Qwen second review for empty samples.")
    parser.add_argument("--disable-high-confidence-fallback", action="store_true", help="Disable conservative matcher fallback.")
    parser.add_argument("--max-fallback-candidates", type=int, default=DEFAULT_MAX_FALLBACK_CANDIDATES, help="Maximum number of fallback candidates.")
    parser.add_argument(
        "--prompt-output-dir",
        default=DEFAULT_PROMPT_OUTPUT_DIR,
        help="Directory used to save exported initial, second-review, and OpenAI final-review prompts.",
    )
    parser.add_argument(
        "--save-stage-outputs",
        action="store_true",
        help="Persist each stage's raw LLM outputs, parsed payloads, and cleaned results for debugging.",
    )
    parser.add_argument(
        "--use-openai-final-review",
        action="store_true",
        default=True,
        help="Use GPT-5.4-mini to perform final review over local-LLM draft labels.",
    )
    parser.add_argument(
        "--openai-review-model",
        default=DEFAULT_OPENAI_REVIEW_MODEL,
        help="OpenAI model used for final review.",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens for the OpenAI final review.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        default="low",
        help="Reasoning effort for the OpenAI final review.",
    )
    return parser


def main() -> None:
    """Parse CLI args, run the pipeline, and print a JSON summary."""
    args = build_parser().parse_args()
    summary = build_regression_dataset(
        output_path=args.output,
        source_table=args.source_table,
        dictionary_path=args.dictionary,
        sample_size=args.sample_size,
        source_limit_rows=args.source_limit_rows,
        seed=args.seed,
        model_path=args.model,
        batch_size=args.batch_size,
        max_text_chars=args.max_text_chars,
        min_text_chars=args.min_text_chars,
        num_votes=args.num_votes,
        min_vote_support=args.min_vote_support,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_second_review=not args.disable_second_review,
        enable_high_confidence_fallback=not args.disable_high_confidence_fallback,
        max_fallback_candidates=args.max_fallback_candidates,
        prompt_output_dir=args.prompt_output_dir,
        save_stage_outputs=args.save_stage_outputs,
        use_openai_final_review=args.use_openai_final_review,
        openai_review_model=args.openai_review_model,
        openai_max_output_tokens=args.openai_max_output_tokens,
        openai_reasoning_effort=args.openai_reasoning_effort,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
