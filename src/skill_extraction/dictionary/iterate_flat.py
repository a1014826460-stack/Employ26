from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import json
import logging
from pathlib import Path
import shutil
from typing import Dict, Iterable, List

from .iteration_rules import DEFAULT_RULES_PATH, load_iteration_rules
from ..labeling.utils import extract_json_from_response, run_openai_prompt_pairs
from ..hard.pg_matcher import run_match_pg
from ..evaluation.regression import evaluate_regression_dataset


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_DICT_PATH = "dicts/flat_skill_dictionary.json"
DEFAULT_DATASET_PATH = "output/skill_extraction/regression/flat_skill_regression_dataset.jsonl"
DEFAULT_REPORT_DIR = "output/skill_extraction/reports/dictionary_iteration"
DEFAULT_REVIEW_MODEL = "openai/gpt-5.4-mini"

REVIEW_SYSTEM_PROMPT = """你负责审核招聘技能词典的增量候选。只根据给定JD片段判断该候选是否适合并入硬技能词典。保守原则：软技能、泛能力、业务容器词一律拒绝；明确工具、框架、设备、工艺、证书、具体系统可保留。只输出JSON。"""
REVIEW_USER_TEMPLATE = """候选技能: {name}
候选类别: {skill_type}
证据片段: {evidence}
样本ID: {sample_id}
JD片段: {text}

输出:
{{"decision":"keep|review|reject","reason":"简短原因"}}"""


def _load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: str | Path, payload: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _latest_path(directory: str | Path, pattern: str) -> Path | None:
    candidates = list(Path(directory).glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _collect_error_patterns(error_csv_path: str | Path) -> Dict[str, List[Dict]]:
    import csv

    false_positive_counter: Counter[str] = Counter()
    false_negative_counter: Counter[str] = Counter()
    samples: List[Dict] = []

    with open(error_csv_path, "r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            false_positives = json.loads(row.get("false_positives", "[]") or "[]")
            false_negatives = json.loads(row.get("false_negatives", "[]") or "[]")
            for item in false_positives:
                false_positive_counter[str(item)] += 1
            for item in false_negatives:
                false_negative_counter[str(item)] += 1
            samples.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "text": row.get("text", "")[:320],
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                }
            )

    return {
        "top_false_positives": [
            {"name": name, "count": count}
            for name, count in false_positive_counter.most_common(20)
        ],
        "top_false_negatives": [
            {"name": name, "count": count}
            for name, count in false_negative_counter.most_common(20)
        ],
        "sample_errors": samples[:10],
    }


def _detect_local_llm_status() -> Dict[str, object]:
    status = {
        "vllm_installed": importlib.util.find_spec("vllm") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "cuda_available": False,
        "blocked": True,
        "reason": "",
    }
    if status["torch_installed"]:
        try:
            import torch  # type: ignore

            status["cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001
            status["reason"] = f"torch check failed: {exc}"
    if status["vllm_installed"] and status["cuda_available"]:
        status["blocked"] = False
        status["reason"] = ""
        return status

    reasons: List[str] = []
    if not status["vllm_installed"]:
        reasons.append("vllm not installed")
    if not status["cuda_available"]:
        reasons.append("CUDA unavailable")
    status["reason"] = ", ".join(reasons)
    return status


def _load_dataset_rows(dataset_path: str | Path) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    with open(dataset_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                rows[sample_id] = row
    return rows


def _review_candidates(
    rules: Dict,
    dataset_rows: Dict[str, Dict],
    model: str,
) -> List[Dict]:
    candidate_additions = list(rules.get("candidate_additions", []))
    max_reviews = max(0, int(rules.get("max_api_reviews", 10)))
    prompt_pairs = []
    review_records: List[Dict] = []

    for candidate in candidate_additions[:max_reviews]:
        sample_id = str(candidate.get("sample_id", "")).strip()
        dataset_row = dataset_rows.get(sample_id, {})
        text = str(dataset_row.get("text", "")).strip()[:420]
        user_prompt = REVIEW_USER_TEMPLATE.format(
            name=str(candidate.get("name", "")).strip(),
            skill_type=str(candidate.get("skill_type", "")).strip(),
            evidence=str(candidate.get("evidence", "")).strip(),
            sample_id=sample_id,
            text=text,
        )
        prompt_pairs.append((REVIEW_SYSTEM_PROMPT, user_prompt))
        review_records.append(
            {
                "candidate": candidate,
                "sample_id": sample_id,
                "prompt_user": user_prompt,
            }
        )

    if not prompt_pairs:
        return []

    outputs = run_openai_prompt_pairs(
        prompt_pairs=prompt_pairs,
        model=model,
        max_output_tokens=256,
        reasoning_effort="low",
    )

    reviewed: List[Dict] = []
    for record, output_text in zip(review_records, outputs):
        parsed = extract_json_from_response(output_text) or {}
        reviewed.append(
            {
                "candidate": record["candidate"],
                "sample_id": record["sample_id"],
                "decision": str(parsed.get("decision", "review")).strip().lower() or "review",
                "reason": str(parsed.get("reason", "")).strip(),
                "raw_output": output_text,
            }
        )
    return reviewed


def _upsert_skill(skill_lookup: Dict[str, Dict], skill_data: Dict) -> None:
    name = str(skill_data.get("name", "")).strip()
    if not name:
        return
    existing = skill_lookup.get(name.casefold())
    if existing is None:
        skill_lookup[name.casefold()] = {
            "name": name,
            "aliases": list(dict.fromkeys(str(alias).strip() for alias in skill_data.get("aliases", []) if str(alias).strip())),
            "skill_type": str(skill_data.get("skill_type", "")).strip(),
            "notes": str(skill_data.get("notes", "")).strip(),
        }
        return

    alias_pool = list(existing.get("aliases", []) or [])
    alias_pool.extend(skill_data.get("aliases", []) or [])
    existing["aliases"] = list(dict.fromkeys(str(alias).strip() for alias in alias_pool if str(alias).strip()))
    if not existing.get("skill_type"):
        existing["skill_type"] = str(skill_data.get("skill_type", "")).strip()
    if skill_data.get("notes"):
        existing["notes"] = str(existing.get("notes", "")).strip() or str(skill_data.get("notes", "")).strip()


def _apply_dictionary_updates(
    dict_path: str | Path,
    rules: Dict,
    approved_candidate_names: Iterable[str],
) -> Dict[str, object]:
    dictionary = _load_json(dict_path)
    original_skills = dictionary.get("skills", [])
    skill_lookup: Dict[str, Dict] = {
        str(skill.get("name", "")).strip().casefold(): dict(skill)
        for skill in original_skills
        if str(skill.get("name", "")).strip()
    }
    approved_name_set = {str(name).strip().casefold() for name in approved_candidate_names if str(name).strip()}

    added_names: List[str] = []
    merged_names: List[Dict[str, str]] = []

    for candidate in rules.get("candidate_additions", []):
        candidate_name = str(candidate.get("name", "")).strip()
        if candidate_name.casefold() not in approved_name_set:
            continue
        existed = candidate_name.casefold() in skill_lookup
        _upsert_skill(skill_lookup, candidate)
        if not existed:
            added_names.append(candidate_name)

    for operation in rules.get("merge_operations", []):
        canonical_name = str(operation.get("canonical_name", "")).strip()
        if not canonical_name:
            continue
        canonical = skill_lookup.get(canonical_name.casefold())
        if canonical is None:
            continue

        extra_aliases = [str(alias).strip() for alias in operation.get("append_aliases", []) if str(alias).strip()]
        if extra_aliases:
            canonical["aliases"] = list(
                dict.fromkeys([*(canonical.get("aliases", []) or []), *extra_aliases, canonical_name])
            )

        for merge_name in operation.get("merge_names", []):
            merge_name = str(merge_name).strip()
            if not merge_name or merge_name.casefold() == canonical_name.casefold():
                continue
            merged = skill_lookup.get(merge_name.casefold())
            if merged is None:
                continue
            alias_pool = [
                *(canonical.get("aliases", []) or []),
                merge_name,
                *(merged.get("aliases", []) or []),
            ]
            canonical["aliases"] = list(dict.fromkeys(str(alias).strip() for alias in alias_pool if str(alias).strip()))
            del skill_lookup[merge_name.casefold()]
            merged_names.append({"from": merge_name, "to": canonical_name})

    new_skills = sorted(skill_lookup.values(), key=lambda item: item["name"].casefold())
    dictionary["skills"] = new_skills
    dictionary.setdefault("metadata", {})
    dictionary["metadata"]["skill_count"] = len(new_skills)
    dictionary["metadata"]["alias_count"] = sum(len(item.get("aliases", []) or []) for item in new_skills)
    dictionary["metadata"]["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    history = list(dictionary["metadata"].get("iteration_history", []) or [])
    history.append(
        {
            "updated_at": dictionary["metadata"]["updated_at"],
            "added_skills": added_names,
            "merged_skills": merged_names,
        }
    )
    dictionary["metadata"]["iteration_history"] = history[-10:]

    dict_file = Path(dict_path)
    backup_path = dict_file.with_name(f"{dict_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{dict_file.suffix}")
    shutil.copy2(dict_file, backup_path)
    with open(dict_file, "w", encoding="utf-8") as file_obj:
        json.dump(dictionary, file_obj, ensure_ascii=False, indent=2)

    return {
        "backup_path": str(backup_path),
        "added_skills": added_names,
        "merged_skills": merged_names,
        "final_skill_count": len(new_skills),
    }


def run_iteration(
    dict_path: str | Path = DEFAULT_DICT_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    review_model: str = DEFAULT_REVIEW_MODEL,
) -> Dict:
    rules = load_iteration_rules(rules_path)
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Running baseline match and regression evaluation")
    run_match_pg(dict_path=str(dict_path), write=True)
    baseline_summary = evaluate_regression_dataset(dataset_path=dataset_path, dict_path=dict_path)
    baseline_error_path = _latest_path(Path("output/skill_extraction/reports/regression_eval"), "regression_errors_*.csv")
    baseline_patterns = _collect_error_patterns(baseline_error_path) if baseline_error_path else {}

    local_llm_status = _detect_local_llm_status()
    precision_target = float(rules.get("precision_target", 0.65))
    if baseline_summary["precision"] >= precision_target:
        result = {
            "status": "baseline_sufficient",
            "local_llm_status": local_llm_status,
            "baseline_summary": baseline_summary,
            "baseline_error_patterns": baseline_patterns,
        }
        _write_json(report_root / f"iteration_{timestamp}.json", result)
        return result

    dataset_rows = _load_dataset_rows(dataset_path)
    review_records = _review_candidates(rules=rules, dataset_rows=dataset_rows, model=review_model)
    approved_candidate_names = [
        record["candidate"]["name"]
        for record in review_records
        if record["decision"] == "keep"
    ]

    update_summary = _apply_dictionary_updates(
        dict_path=dict_path,
        rules=rules,
        approved_candidate_names=approved_candidate_names,
    )

    logger.info("Re-running match and evaluation after conservative updates")
    run_match_pg(dict_path=str(dict_path), write=True)
    updated_summary = evaluate_regression_dataset(dataset_path=dataset_path, dict_path=dict_path)
    updated_error_path = _latest_path(Path("output/skill_extraction/reports/regression_eval"), "regression_errors_*.csv")
    updated_patterns = _collect_error_patterns(updated_error_path) if updated_error_path else {}

    result = {
        "status": "completed",
        "local_llm_status": local_llm_status,
        "rules_path": str(Path(rules_path)),
        "baseline_summary": baseline_summary,
        "baseline_error_patterns": baseline_patterns,
        "api_review_model": review_model,
        "api_reviews": review_records,
        "dictionary_update_summary": update_summary,
        "updated_summary": updated_summary,
        "updated_error_patterns": updated_patterns,
    }
    _write_json(report_root / f"iteration_{timestamp}.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative flat skill dictionary iteration")
    parser.add_argument("--dictionary", default=DEFAULT_DICT_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--review-model", default=DEFAULT_REVIEW_MODEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_iteration(
        dict_path=args.dictionary,
        dataset_path=args.dataset,
        rules_path=args.rules,
        report_dir=args.report_dir,
        review_model=args.review_model,
    )
    logger.info(
        "Iteration finished: baseline P=%.4f -> updated P=%.4f",
        float(result["baseline_summary"]["precision"]),
        float(result.get("updated_summary", result["baseline_summary"])["precision"]),
    )


if __name__ == "__main__":
    main()
