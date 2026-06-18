"""Repair Label Studio task recruitment_record_id using task payload text.

The historical backfill audit can be wrong when historical row_id no longer
points to the same task payload. This repair uses the current task `data_raw`
as the source of truth and remaps tasks to `public.recruitment_jobs_normalized`
through sample-table source locators.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, text

from src.db.postgres import create_pg_engine
from src.utils.backfill_label_studio_recruitment_record_ids import load_sample_tables


LOGGER = logging.getLogger(__name__)
TASK_TABLE = "annotations.label_studio_tasks_v2"
NORMALIZED_TABLE = "public.recruitment_jobs_normalized"
BACKUP_TABLE_PREFIX = "annotations.label_studio_tasks_v2_rrid_backup"
AUDIT_TABLE = "annotations.label_studio_task_rrid_repair_audit"
AUTO_THRESHOLD = 0.86
AUTO_MARGIN = 0.03


@dataclass(frozen=True)
class MatchCandidate:
    """A candidate source row for one Label Studio task."""

    source_table: str
    source_row_number: int
    recruitment_record_id: str
    title: str
    description: str
    company_name: str
    score: float


def normalize_text(value: object) -> str:
    """Normalize text for stable Chinese job-title matching."""
    if value is None:
        return ""
    text_value = str(value).strip()
    if not text_value or text_value.lower() == "nan":
        return ""
    text_value = text_value.replace("\xa0", " ")
    text_value = re.sub(r"\s+", "", text_value)
    return text_value.casefold()


def compact_text(value: object) -> str:
    """Normalize text for containment checks by removing punctuation too."""
    text_value = normalize_text(value)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text_value)


def split_requirement_segments(value: object) -> list[str]:
    """Split cleaned requirements into evidence segments."""
    text_value = display_text(value)
    raw_segments = re.split(r"[|。；;！!？?\n\r]+|\d+[、.．)]", text_value)
    segments: list[str] = []
    for segment in raw_segments:
        compact = compact_text(segment)
        if len(compact) < 8:
            continue
        if compact in {"任职要求", "岗位要求", "技能要求", "职位要求"}:
            continue
        segments.append(compact)
    return segments


def display_text(value: object) -> str:
    """Normalize display text without removing internal spaces."""
    if value is None:
        return ""
    text_value = str(value).strip()
    if not text_value or text_value.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text_value)


def parse_data_raw(raw: object) -> dict[str, Any]:
    """Parse task data_raw safely."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_task_payload(row: dict[str, Any]) -> dict[str, str]:
    """Extract the current task payload used as repair source of truth."""
    data = parse_data_raw(row.get("data_raw"))
    return {
        "task_id": str(row["id"]),
        "job_title": display_text(data.get("job_title") or row.get("job_title")),
        "job_requirements": display_text(
            data.get("job_requirements_clean")
            or data.get("job_requirements")
            or row.get("job_requirements")
        ),
        "current_recruitment_record_id": display_text(row.get("recruitment_record_id")),
    }


def build_source_index(
    sample_tables: dict[str, pd.DataFrame],
    locator_to_rrid: dict[tuple[str, int], str],
) -> dict[str, list[MatchCandidate]]:
    """Index source sample rows by normalized job title."""
    index: dict[str, list[MatchCandidate]] = defaultdict(list)
    for source_table, frame in sample_tables.items():
        for row_number, (_, row) in enumerate(frame.reset_index(drop=True).iterrows(), start=1):
            rrid = locator_to_rrid.get((source_table, row_number), "")
            if not rrid:
                continue
            title = display_text(row.get("岗位名称", ""))
            if not title:
                continue
            index[normalize_text(title)].append(
                MatchCandidate(
                    source_table=source_table,
                    source_row_number=row_number,
                    recruitment_record_id=rrid,
                    title=title,
                    description=display_text(row.get("岗位描述", "")),
                    company_name=display_text(row.get("公司名称", "")),
                    score=0.0,
                )
            )
    return index


def load_locator_to_rrid() -> dict[tuple[str, int], str]:
    """Load normalized-table IDs by source locator for sample tables only."""
    engine = create_pg_engine(application_name="repair_label_studio_rrid_locator")
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    select recruitment_record_id, source_table, source_row_number
                    from {NORMALIZED_TABLE}
                    where source_table in :source_tables
                    """
                ).bindparams(bindparam("source_tables", expanding=True)),
                {
                    "source_tables": [
                        '"51job".sample',
                        '"Liepin".sample',
                        '"Zhilian".sample',
                    ]
                },
            ).mappings()
            return {
                (str(row["source_table"]), int(row["source_row_number"])): str(row["recruitment_record_id"])
                for row in rows
            }
    finally:
        engine.dispose()


def score_candidate(task_payload: dict[str, str], candidate: MatchCandidate) -> float:
    """Score a candidate using title equality plus requirements-in-description evidence."""
    task_req = compact_text(task_payload["job_requirements"])
    cand_desc = compact_text(candidate.description)
    if not task_req:
        return 0.0
    if task_req and task_req in cand_desc:
        return 1.0
    if cand_desc and cand_desc in task_req:
        return 0.98
    sequence_score = float(SequenceMatcher(None, task_req, cand_desc).ratio())

    segments = split_requirement_segments(task_payload["job_requirements"])
    if not segments:
        return sequence_score
    total_weight = sum(len(segment) for segment in segments)
    covered_weight = sum(len(segment) for segment in segments if segment in cand_desc)
    coverage_score = covered_weight / total_weight if total_weight else 0.0

    # Reward highly distinctive partial coverage even when the source JD has
    # long responsibility text before the requirement section.
    return max(sequence_score, coverage_score)


def choose_candidate(
    task_payload: dict[str, str],
    candidates: list[MatchCandidate],
) -> tuple[str, str, MatchCandidate | None, float | None, float | None, int]:
    """Choose one high-confidence candidate for a task."""
    if not task_payload["job_title"]:
        return "MISSING_TASK_TITLE", "no_title", None, None, None, 0
    if not candidates:
        return "UNMATCHED", "no_same_title_candidate", None, None, None, 0
    scored = sorted(
        [
            MatchCandidate(
                source_table=c.source_table,
                source_row_number=c.source_row_number,
                recruitment_record_id=c.recruitment_record_id,
                title=c.title,
                description=c.description,
                company_name=c.company_name,
                score=score_candidate(task_payload, c),
            )
            for c in candidates
        ],
        key=lambda item: item.score,
        reverse=True,
    )
    best = scored[0]
    second_score = scored[1].score if len(scored) > 1 else None
    if best.score >= 0.999:
        same_best = [c for c in scored if abs(c.score - best.score) < 1e-9]
        if len(same_best) == 1:
            return "AUTO_CONFIRMED", "requirements_exact_unique", best, best.score, second_score, len(scored)
        return "REVIEW_REQUIRED", "requirements_exact_ambiguous", best, best.score, second_score, len(scored)
    if best.score >= AUTO_THRESHOLD and (
        second_score is None or best.score - second_score >= AUTO_MARGIN
    ):
        return "AUTO_CONFIRMED", "requirements_similarity_unique", best, best.score, second_score, len(scored)
    return "REVIEW_REQUIRED", "requirements_similarity_ambiguous", best, best.score, second_score, len(scored)


def load_task_rows() -> list[dict[str, Any]]:
    """Load task rows with current payloads."""
    engine = create_pg_engine(application_name="repair_label_studio_rrid_tasks")
    try:
        with engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    text(
                        f"""
                        select id, job_title, job_requirements, recruitment_record_id, data_raw
                        from {TASK_TABLE}
                        order by id
                        """
                    )
                ).mappings()
            ]
    finally:
        engine.dispose()


def ensure_repair_audit_table(conn) -> None:
    """Create repair audit table."""
    conn.execute(
        text(
            f"""
            create table if not exists {AUDIT_TABLE} (
                task_id integer primary key,
                old_recruitment_record_id text,
                new_recruitment_record_id text,
                mapping_status text not null,
                mapping_rule text not null,
                candidate_count integer not null,
                best_similarity_score double precision,
                second_similarity_score double precision,
                job_title text,
                matched_title text,
                matched_company_name text,
                source_table text,
                source_row_number bigint,
                repaired_at timestamptz not null default now()
            )
            """
        )
    )


def backup_current_ids(conn) -> str:
    """Backup current task IDs before overwriting."""
    suffix = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
    backup_table = f"{BACKUP_TABLE_PREFIX}_{suffix}"
    conn.execute(
        text(
            f"""
            create table {backup_table} as
            select id as task_id, recruitment_record_id, now() as backed_up_at
            from {TASK_TABLE}
            """
        )
    )
    return backup_table


def repair_recruitment_record_ids(*, dry_run: bool = False) -> dict[str, Any]:
    """Repair task recruitment_record_id values from task payload text."""
    LOGGER.info("加载 normalized source locator...")
    locator_to_rrid = load_locator_to_rrid()
    LOGGER.info("normalized source locator rows: %s", len(locator_to_rrid))
    LOGGER.info("加载 sample 表...")
    source_index = build_source_index(load_sample_tables(), locator_to_rrid)
    LOGGER.info("source title index keys: %s", len(source_index))
    tasks = load_task_rows()

    audit_rows: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    changed = 0
    for task in tasks:
        payload = build_task_payload(task)
        title_key = normalize_text(payload["job_title"])
        status, rule, best, best_score, second_score, candidate_count = choose_candidate(
            payload,
            source_index.get(title_key, []),
        )
        status_counts[status] += 1
        old_rrid = payload["current_recruitment_record_id"]
        new_rrid = best.recruitment_record_id if best and status == "AUTO_CONFIRMED" else ""
        if new_rrid and new_rrid != old_rrid:
            changed += 1
            updates.append({"task_id": int(payload["task_id"]), "recruitment_record_id": new_rrid})

        audit_rows.append(
            {
                "task_id": int(payload["task_id"]),
                "old_recruitment_record_id": old_rrid or None,
                "new_recruitment_record_id": new_rrid or None,
                "mapping_status": status,
                "mapping_rule": rule,
                "candidate_count": candidate_count,
                "best_similarity_score": best_score,
                "second_similarity_score": second_score,
                "job_title": payload["job_title"],
                "matched_title": best.title if best else None,
                "matched_company_name": best.company_name if best else None,
                "source_table": best.source_table if best else None,
                "source_row_number": best.source_row_number if best else None,
            }
        )

    summary = {
        "total_tasks": len(tasks),
        "status_counts": dict(status_counts),
        "updates_needed": changed,
        "dry_run": dry_run,
    }
    if dry_run:
        LOGGER.info("dry-run summary: %s", summary)
        return summary

    engine = create_pg_engine(application_name="repair_label_studio_rrid_apply")
    try:
        with engine.begin() as conn:
            backup_table = backup_current_ids(conn)
            ensure_repair_audit_table(conn)
            conn.execute(text(f"truncate table {AUDIT_TABLE}"))
            conn.execute(
                text(
                    f"""
                    insert into {AUDIT_TABLE} (
                        task_id,
                        old_recruitment_record_id,
                        new_recruitment_record_id,
                        mapping_status,
                        mapping_rule,
                        candidate_count,
                        best_similarity_score,
                        second_similarity_score,
                        job_title,
                        matched_title,
                        matched_company_name,
                        source_table,
                        source_row_number
                    ) values (
                        :task_id,
                        :old_recruitment_record_id,
                        :new_recruitment_record_id,
                        :mapping_status,
                        :mapping_rule,
                        :candidate_count,
                        :best_similarity_score,
                        :second_similarity_score,
                        :job_title,
                        :matched_title,
                        :matched_company_name,
                        :source_table,
                        :source_row_number
                    )
                    """
                ),
                audit_rows,
            )
            if updates:
                conn.execute(
                    text(
                        f"""
                        update {TASK_TABLE}
                        set recruitment_record_id = :recruitment_record_id
                        where id = :task_id
                        """
                    ),
                    updates,
                )
        summary["backup_table"] = backup_table
    finally:
        engine.dispose()
    LOGGER.info("repair summary: %s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Repair Label Studio recruitment_record_id from task payload text")
    parser.add_argument("--dry-run", action="store_true", help="Only compute mappings; do not write database")
    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    summary = repair_recruitment_record_ids(dry_run=bool(args.dry_run))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
