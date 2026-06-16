from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List

from .iterate_flat import run_iteration
from .iteration_rules import DEFAULT_RULES_PATH, load_iteration_rules
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
DEFAULT_STATE_PATH = f"{DEFAULT_REPORT_DIR}/workflow_state.json"
DEFAULT_REVIEW_MODEL = "openai/gpt-5.4-mini"


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _load_json(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _latest_path(directory: str | Path, pattern: str) -> Path | None:
    candidates = list(Path(directory).glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _collect_error_patterns(error_csv_path: str | Path | None) -> Dict[str, Any]:
    if not error_csv_path:
        return {}

    false_positive_counter: Counter[str] = Counter()
    false_negative_counter: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []

    with open(error_csv_path, "r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            false_positives = json.loads(row.get("false_positives", "[]") or "[]")
            false_negatives = json.loads(row.get("false_negatives", "[]") or "[]")
            for item in false_positives:
                false_positive_counter[str(item)] += 1
            for item in false_negatives:
                false_negative_counter[str(item)] += 1
            if false_positives or false_negatives:
                samples.append(
                    {
                        "sample_id": row.get("sample_id", ""),
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                        "text": row.get("text", "")[:320],
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


def _extract_workflow_config(rules: Dict[str, Any], max_rounds_override: int | None) -> Dict[str, Any]:
    workflow = dict(rules.get("workflow", {}) or {})
    workflow["max_rounds"] = int(max_rounds_override or workflow.get("max_rounds", 3))
    workflow["min_precision_gain"] = float(workflow.get("min_precision_gain", 0.002))
    workflow["min_recall_gain"] = float(workflow.get("min_recall_gain", 0.005))
    workflow["stop_if_no_candidate_kept"] = bool(workflow.get("stop_if_no_candidate_kept", True))
    workflow["stop_if_no_metric_gain"] = bool(workflow.get("stop_if_no_metric_gain", True))
    workflow["precision_target"] = float(rules.get("precision_target", 0.8))
    workflow["recall_target"] = float(rules.get("recall_target", 0.9))
    workflow["f1_target"] = float(rules.get("f1_target", 0.9))
    workflow["max_api_reviews"] = int(rules.get("max_api_reviews", 10))
    return workflow


def _runtime_info() -> Dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }


def _determine_error_mode(patterns: Dict[str, Any]) -> str:
    fp_count = sum(int(item.get("count", 0) or 0) for item in patterns.get("top_false_positives", []))
    fn_count = sum(int(item.get("count", 0) or 0) for item in patterns.get("top_false_negatives", []))
    if fn_count > fp_count:
        return "recall_first"
    if fp_count > fn_count:
        return "precision_first"
    return "balanced"


def _decide_next_action(
    result: Dict[str, Any],
    workflow: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = result["baseline_summary"]
    final_summary = result.get("updated_summary", baseline)
    api_reviews = result.get("api_reviews", []) or []
    kept_count = sum(1 for item in api_reviews if item.get("decision") == "keep")
    delta_precision = float(final_summary["precision"]) - float(baseline["precision"])
    delta_recall = float(final_summary["recall"]) - float(baseline["recall"])
    delta_f1 = float(final_summary["f1"]) - float(baseline["f1"])

    targets_met = (
        float(final_summary["precision"]) >= workflow["precision_target"]
        and float(final_summary["recall"]) >= workflow["recall_target"]
        and float(final_summary["f1"]) >= workflow["f1_target"]
    )
    if targets_met:
        return {
            "action": "stop_sufficient",
            "reason": "precision/recall/f1 all meet workflow targets",
            "delta_precision": delta_precision,
            "delta_recall": delta_recall,
            "delta_f1": delta_f1,
        }

    if result.get("status") == "baseline_sufficient":
        return {
            "action": "stop_sufficient",
            "reason": "baseline metrics already meet precision target",
            "delta_precision": 0.0,
            "delta_recall": 0.0,
            "delta_f1": 0.0,
        }

    if workflow["stop_if_no_candidate_kept"] and api_reviews and kept_count == 0:
        return {
            "action": "stop_blocked",
            "reason": "api review did not keep any candidates",
            "delta_precision": delta_precision,
            "delta_recall": delta_recall,
            "delta_f1": delta_f1,
        }

    if workflow["stop_if_no_metric_gain"]:
        no_precision_gain = delta_precision < workflow["min_precision_gain"]
        no_recall_gain = delta_recall < workflow["min_recall_gain"]
        if no_precision_gain and no_recall_gain:
            return {
                "action": "stop_blocked",
                "reason": "metric gains are below workflow thresholds",
                "delta_precision": delta_precision,
                "delta_recall": delta_recall,
                "delta_f1": delta_f1,
            }

    return {
        "action": "continue",
        "reason": "metrics improved but targets are not fully met",
        "delta_precision": delta_precision,
        "delta_recall": delta_recall,
        "delta_f1": delta_f1,
    }


def _build_round_record(
    round_index: int,
    result: Dict[str, Any],
    workflow: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = result["baseline_summary"]
    final_summary = result.get("updated_summary", baseline)
    updated_patterns = result.get("updated_error_patterns") or result.get("baseline_error_patterns", {})
    decision = _decide_next_action(result, workflow)
    api_reviews = result.get("api_reviews", []) or []

    return {
        "round_index": round_index,
        "executed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "status": result.get("status", ""),
        "baseline_summary": baseline,
        "final_summary": final_summary,
        "local_llm_status": result.get("local_llm_status", {}),
        "api_review_count": len(api_reviews),
        "api_keep_count": sum(1 for item in api_reviews if item.get("decision") == "keep"),
        "dictionary_update_summary": result.get("dictionary_update_summary", {}),
        "error_mode": _determine_error_mode(updated_patterns),
        "top_false_positives": updated_patterns.get("top_false_positives", [])[:10],
        "top_false_negatives": updated_patterns.get("top_false_negatives", [])[:10],
        "next_action": decision,
    }


def run_baseline(
    dict_path: str | Path,
    dataset_path: str | Path,
    rules_path: str | Path,
    state_path: str | Path,
) -> Dict[str, Any]:
    rules = load_iteration_rules(rules_path)
    workflow = _extract_workflow_config(rules, max_rounds_override=None)

    logger.info("Running normalized workflow baseline")
    run_match_pg(dict_path=str(dict_path), write=True)
    summary = evaluate_regression_dataset(dataset_path=dataset_path, dict_path=dict_path)
    error_path = _latest_path(
        Path("output/skill_extraction/reports/regression_eval"),
        "regression_errors_*.csv",
    )
    patterns = _collect_error_patterns(error_path)
    decision = {
        "action": "continue",
        "reason": "baseline requires iterative improvement",
    }
    if (
        float(summary["precision"]) >= workflow["precision_target"]
        and float(summary["recall"]) >= workflow["recall_target"]
        and float(summary["f1"]) >= workflow["f1_target"]
    ):
        decision = {
            "action": "stop_sufficient",
            "reason": "baseline metrics already meet workflow targets",
        }

    state = {
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime": _runtime_info(),
        "workflow": workflow,
        "current_stage": "baseline_evaluated",
        "latest_summary": summary,
        "latest_error_mode": _determine_error_mode(patterns),
        "latest_error_patterns": patterns,
        "latest_decision": decision,
        "history": [],
    }
    _write_json(state_path, state)
    return state


def run_workflow(
    dict_path: str | Path = DEFAULT_DICT_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    state_path: str | Path = DEFAULT_STATE_PATH,
    review_model: str = DEFAULT_REVIEW_MODEL,
    max_rounds: int | None = None,
) -> Dict[str, Any]:
    rules = load_iteration_rules(rules_path)
    workflow = _extract_workflow_config(rules, max_rounds_override=max_rounds)
    state = _load_json(state_path)
    history = list(state.get("history", []) or [])

    for round_index in range(1, workflow["max_rounds"] + 1):
        logger.info("Starting normalized workflow round %d/%d", round_index, workflow["max_rounds"])
        result = run_iteration(
            dict_path=dict_path,
            dataset_path=dataset_path,
            rules_path=rules_path,
            report_dir=report_dir,
            review_model=review_model,
        )
        round_record = _build_round_record(round_index, result, workflow)
        history.append(round_record)

        state = {
            "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "runtime": _runtime_info(),
            "workflow": workflow,
            "current_stage": "iteration_completed",
            "latest_summary": round_record["final_summary"],
            "latest_error_mode": round_record["error_mode"],
            "latest_decision": round_record["next_action"],
            "latest_round": round_record,
            "history": history,
        }
        _write_json(state_path, state)

        if round_record["next_action"]["action"].startswith("stop"):
            return state

    state["current_stage"] = "max_rounds_reached"
    state["latest_decision"] = {
        "action": "stop_max_rounds",
        "reason": "workflow reached max_rounds without meeting stop criteria",
    }
    _write_json(state_path, state)
    return state


def print_status(state_path: str | Path) -> Dict[str, Any]:
    state = _load_json(state_path)
    if not state:
        logger.info("No workflow state found at %s", state_path)
        return {}

    latest_summary = state.get("latest_summary", {})
    latest_decision = state.get("latest_decision", {})
    logger.info(
        "Workflow status: stage=%s P=%.4f R=%.4f F1=%.4f action=%s",
        state.get("current_stage", ""),
        float(latest_summary.get("precision", 0.0) or 0.0),
        float(latest_summary.get("recall", 0.0) or 0.0),
        float(latest_summary.get("f1", 0.0) or 0.0),
        latest_decision.get("action", ""),
    )
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalized flat skill dictionary workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dictionary", default=DEFAULT_DICT_PATH)
        subparser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
        subparser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
        subparser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
        subparser.add_argument("--state-path", default=DEFAULT_STATE_PATH)

    baseline = subparsers.add_parser("baseline", help="Run baseline match + evaluation and write workflow state.")
    add_common_args(baseline)

    run = subparsers.add_parser("run", help="Run the normalized iterative workflow.")
    add_common_args(run)
    run.add_argument("--review-model", default=DEFAULT_REVIEW_MODEL)
    run.add_argument("--max-rounds", type=int, default=None)

    status = subparsers.add_parser("status", help="Show the latest workflow state.")
    status.add_argument("--state-path", default=DEFAULT_STATE_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "baseline":
        run_baseline(
            dict_path=args.dictionary,
            dataset_path=args.dataset,
            rules_path=args.rules,
            state_path=args.state_path,
        )
        return

    if args.command == "run":
        run_workflow(
            dict_path=args.dictionary,
            dataset_path=args.dataset,
            rules_path=args.rules,
            report_dir=args.report_dir,
            state_path=args.state_path,
            review_model=args.review_model,
            max_rounds=args.max_rounds,
        )
        return

    if args.command == "status":
        print_status(args.state_path)


if __name__ == "__main__":
    main()
