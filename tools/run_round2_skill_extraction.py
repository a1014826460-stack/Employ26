"""Run V3 skill extraction for all round-2 Label Studio tasks.

The round-2 annotation table contains occupational-choice labels and soft-skill
span labels.  It does not contain human hard-skill span labels, so this script
reports hard-skill extraction coverage/descriptive rates and soft-skill
precision/coverage against the available span gold.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import get_project_paths
from src.skill_extraction.evaluation.v3 import _normalize_skill_name
from src.skill_extraction.pipeline import create_v3_pipeline

LOGGER = logging.getLogger("round2_skill_extraction")
ROUND2_TABLE = "annotations.label_studio_tasks_v2"
OUTPUT_TABLE = "public.round2_skill_extraction_v3"

DIMENSION_LABELS = {
    "开放性": "openness",
    "尽责性": "conscientiousness",
    "外向性": "extraversion",
    "宜人性": "agreeableness",
    "神经质": "neuroticism",
    "情绪稳定性": "neuroticism",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _parse_annotations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _parse_gold_soft_skills(annotations: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract soft-skill span labels from Label Studio annotation payloads."""
    gold: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for annotation in annotations:
        for result in annotation.get("result", []) or []:
            if not isinstance(result, dict) or result.get("type") != "labels":
                continue
            from_name = _safe_text(result.get("from_name"))
            if not from_name.startswith("softskill_"):
                continue
            value = result.get("value") or {}
            name = _safe_text(value.get("text"))
            if not name:
                continue
            labels = value.get("labels") or []
            label = _safe_text(labels[0]) if labels else ""
            dimension = DIMENSION_LABELS.get(label, label)
            key = (_normalize_skill_name(name), _normalize_skill_name(dimension))
            if key in seen:
                continue
            seen.add(key)
            gold.append({"name": name, "dimension": dimension})
    return gold


def fetch_round2_records(limit: int | None = None) -> list[dict[str, Any]]:
    """Fetch round-2 tasks and adapt them to V3 pipeline records."""
    paths = get_project_paths()
    limit_sql = " LIMIT %s" if limit else ""
    query = f"""
        SELECT
            id,
            recruitment_record_id,
            job_title,
            job_requirements,
            annotations_completed_jsonb,
            annotations_completed
        FROM {ROUND2_TABLE}
        ORDER BY id
        {limit_sql}
    """
    conn = psycopg2.connect(**paths.pg_connection_params)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (limit,)) if limit else cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    for (
        task_id,
        recruitment_record_id,
        job_title,
        job_requirements,
        annotations_jsonb,
        annotations_text,
    ) in rows:
        annotations = _parse_annotations(annotations_jsonb or annotations_text)
        records.append(
            {
                "task_id": int(task_id),
                "recruitment_record_id": _safe_text(recruitment_record_id)
                or f"round2_task_{task_id}",
                "source_table": ROUND2_TABLE,
                "source_row_number": int(task_id),
                "job_title": _safe_text(job_title),
                "requirements_text": _safe_text(job_requirements),
                "gold_soft_skills": _parse_gold_soft_skills(annotations),
            }
        )
    return records


def _soft_hit(pred_name: str, gold_name: str) -> bool:
    pred = _normalize_skill_name(pred_name)
    gold = _normalize_skill_name(gold_name)
    return bool(pred and gold and (pred == gold or pred in gold or gold in pred))


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute hard descriptive stats and soft gold metrics."""
    total = len(rows)
    hard_records_with_hits = 0
    soft_records_with_hits = 0
    hard_pred_total = 0
    soft_pred_total = 0
    soft_pred_matched_total = 0
    soft_gold_total = 0
    soft_matched_total = 0
    soft_dim_correct = 0
    soft_dim_total = 0
    soft_error_rows: list[dict[str, Any]] = []

    for row in rows:
        hard_skills = row["hard_skills"]
        soft_skills = row["soft_skills"]
        gold_soft = row["gold_soft_skills"]
        if hard_skills:
            hard_records_with_hits += 1
        if soft_skills:
            soft_records_with_hits += 1
        hard_pred_total += len({_normalize_skill_name(s["name"]) for s in hard_skills})
        predicted_names = {_normalize_skill_name(s["name"]) for s in soft_skills}
        gold_names = {_normalize_skill_name(s["name"]) for s in gold_soft}
        soft_pred_total += len(predicted_names)
        soft_gold_total += len(gold_names)

        matched_gold: set[str] = set()
        pred_to_gold: dict[str, str] = {}
        for gold in gold_soft:
            for pred in soft_skills:
                if _soft_hit(pred["name"], gold["name"]):
                    norm_gold = _normalize_skill_name(gold["name"])
                    matched_gold.add(norm_gold)
                    pred_to_gold[_normalize_skill_name(pred["name"])] = norm_gold
                    break

        soft_matched_total += len(matched_gold)
        pred_hit_names = {
            pred_name
            for pred_name in predicted_names
            if any(_soft_hit(pred_name, gold_name) for gold_name in gold_names)
        }
        soft_pred_matched_total += len(pred_hit_names)

        gold_by_norm = {_normalize_skill_name(item["name"]): item for item in gold_soft}
        for pred in soft_skills:
            norm_pred = _normalize_skill_name(pred["name"])
            norm_gold = pred_to_gold.get(norm_pred)
            if not norm_gold:
                continue
            gold = gold_by_norm.get(norm_gold)
            if not gold:
                continue
            soft_dim_total += 1
            if _normalize_skill_name(pred.get("dimension", "")) == _normalize_skill_name(
                gold.get("dimension", "")
            ):
                soft_dim_correct += 1

        missing = gold_names - matched_gold
        extra = {
            name
            for name in predicted_names
            if not any(_soft_hit(name, gold_name) for gold_name in gold_names)
        }
        if missing or extra:
            soft_error_rows.append(
                {
                    "task_id": row["task_id"],
                    "job_title": row["job_title"],
                    "predicted_soft_skills": json.dumps(
                        [s["name"] for s in soft_skills], ensure_ascii=False
                    ),
                    "gold_soft_skills": json.dumps(
                        [s["name"] for s in gold_soft], ensure_ascii=False
                    ),
                    "missing_soft_skills": json.dumps(
                        [s["name"] for s in gold_soft if _normalize_skill_name(s["name"]) in missing],
                        ensure_ascii=False,
                    ),
                    "extra_soft_skills": json.dumps(
                        [s["name"] for s in soft_skills if _normalize_skill_name(s["name"]) in extra],
                        ensure_ascii=False,
                    ),
                }
            )

    return {
        "record_count": total,
        "hard_skill_metrics": {
            "record_coverage": hard_records_with_hits / max(total, 1),
            "records_with_hits": hard_records_with_hits,
            "predicted_count": hard_pred_total,
            "avg_predicted_per_record": hard_pred_total / max(total, 1),
            "precision": None,
            "recall": None,
            "note": "第二轮标注数据没有逐词 hard skill gold；此处仅报告提取覆盖率。",
        },
        "soft_skill_metrics": {
            "coverage": soft_matched_total / max(soft_gold_total, 1),
            "precision": soft_pred_matched_total / max(soft_pred_total, 1),
            "dimension_accuracy": soft_dim_correct / max(soft_dim_total, 1),
            "records_with_hits": soft_records_with_hits,
            "record_coverage": soft_records_with_hits / max(total, 1),
            "predicted_count": soft_pred_total,
            "predicted_matched_count": soft_pred_matched_total,
            "gold_count": soft_gold_total,
            "matched_count": soft_matched_total,
        },
        "soft_error_rows": soft_error_rows,
    }


def write_outputs(rows: list[dict[str, Any]], metrics: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"round2_v3_skill_extraction_{timestamp}.jsonl"
    summary_path = out_dir / f"round2_v3_skill_metrics_{timestamp}.json"
    report_path = out_dir / f"round2_v3_skill_report_{timestamp}.md"
    soft_errors_path = out_dir / f"round2_v3_soft_errors_{timestamp}.csv"

    with jsonl_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "soft_error_rows"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with soft_errors_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        fieldnames = [
            "task_id",
            "job_title",
            "predicted_soft_skills",
            "gold_soft_skills",
            "missing_soft_skills",
            "extra_soft_skills",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics["soft_error_rows"])

    hard = metrics["hard_skill_metrics"]
    soft = metrics["soft_skill_metrics"]
    report_path.write_text(
        "\n".join(
            [
                "# Round2 V3 Skill Extraction Report",
                "",
                f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Source table: `{ROUND2_TABLE}`",
                f"- Records: {metrics['record_count']}",
                "",
                "## Hard Skills",
                "",
                f"- Record coverage: {hard['record_coverage']:.4f}",
                f"- Records with hits: {hard['records_with_hits']}",
                f"- Predicted hard skills: {hard['predicted_count']}",
                f"- Avg predicted per record: {hard['avg_predicted_per_record']:.4f}",
                f"- Note: {hard['note']}",
                "",
                "## Soft Skills",
                "",
                f"- Precision: {soft['precision']:.4f}",
                f"- Coverage: {soft['coverage']:.4f}",
                f"- Dimension accuracy: {soft['dimension_accuracy']:.4f}",
                f"- Record coverage: {soft['record_coverage']:.4f}",
                f"- Gold count: {soft['gold_count']}",
                f"- Predicted count: {soft['predicted_count']}",
                f"- Matched count: {soft['matched_count']}",
                "",
                "## Outputs",
                "",
                f"- Results JSONL: `{jsonl_path.name}`",
                f"- Summary JSON: `{summary_path.name}`",
                f"- Soft errors CSV: `{soft_errors_path.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "jsonl": jsonl_path,
        "summary": summary_path,
        "report": report_path,
        "soft_errors": soft_errors_path,
    }


def write_pg_results(rows: list[dict[str, Any]], output_table: str = OUTPUT_TABLE) -> int:
    paths = get_project_paths()
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
            task_id INTEGER PRIMARY KEY,
            recruitment_record_id TEXT,
            job_title TEXT,
            source_table TEXT,
            source_row_number INTEGER,
            hard_skills JSONB NOT NULL,
            hard_skill_count INTEGER NOT NULL,
            soft_skills JSONB NOT NULL,
            soft_skill_count INTEGER NOT NULL,
            gold_soft_skills JSONB NOT NULL,
            extracted_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """
    upsert_sql = f"""
        INSERT INTO {output_table} (
            task_id, recruitment_record_id, job_title, source_table, source_row_number,
            hard_skills, hard_skill_count, soft_skills, soft_skill_count,
            gold_soft_skills, extracted_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s::jsonb, now()
        )
        ON CONFLICT (task_id) DO UPDATE SET
            recruitment_record_id = EXCLUDED.recruitment_record_id,
            job_title = EXCLUDED.job_title,
            source_table = EXCLUDED.source_table,
            source_row_number = EXCLUDED.source_row_number,
            hard_skills = EXCLUDED.hard_skills,
            hard_skill_count = EXCLUDED.hard_skill_count,
            soft_skills = EXCLUDED.soft_skills,
            soft_skill_count = EXCLUDED.soft_skill_count,
            gold_soft_skills = EXCLUDED.gold_soft_skills,
            extracted_at = now()
    """
    payload = [
        (
            row["task_id"],
            row["recruitment_record_id"],
            row["job_title"],
            row["source_table"],
            row["source_row_number"],
            json.dumps(row["hard_skills"], ensure_ascii=False),
            len(row["hard_skills"]),
            json.dumps(row["soft_skills"], ensure_ascii=False),
            len(row["soft_skills"]),
            json.dumps(row["gold_soft_skills"], ensure_ascii=False),
        )
        for row in rows
    ]
    conn = psycopg2.connect(**paths.pg_connection_params)
    try:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            for item in payload:
                cur.execute(upsert_sql, item)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(payload)


def run(limit: int | None = None, write_pg: bool = True) -> dict[str, Any]:
    paths = get_project_paths()
    out_dir = paths.skill_extraction_output_dir / "round2_v3"
    pipeline = create_v3_pipeline(use_llm=False)
    records = fetch_round2_records(limit=limit)
    LOGGER.info("Loaded round2 records: %d", len(records))
    results = pipeline.process_records(records)

    rows: list[dict[str, Any]] = []
    gold_by_id = {record["task_id"]: record["gold_soft_skills"] for record in records}
    task_by_id = {record["task_id"]: record for record in records}
    for result in results:
        base = task_by_id[int(result.source_row_number)]
        rows.append(
            {
                "task_id": base["task_id"],
                "recruitment_record_id": result.recruitment_record_id,
                "job_title": result.job_title,
                "source_table": result.source_table,
                "source_row_number": result.source_row_number,
                "hard_skills": result.hard_skills,
                "soft_skills": result.soft_skills,
                "gold_soft_skills": gold_by_id[base["task_id"]],
            }
        )

    metrics = compute_metrics(rows)
    if write_pg:
        written = write_pg_results(rows)
        LOGGER.info("Wrote PostgreSQL results: %d", written)
    paths_map = write_outputs(rows, metrics, out_dir)
    LOGGER.info("Report: %s", paths_map["report"])
    return {"metrics": metrics, "paths": {k: str(v) for k, v in paths_map.items()}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run V3 skill extraction for round2 annotations")
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--no-write-pg", action="store_true", default=False)
    args = parser.parse_args()
    outcome = run(limit=args.limit or None, write_pg=not args.no_write_pg)
    summary = {k: v for k, v in outcome["metrics"].items() if k != "soft_error_rows"}
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
