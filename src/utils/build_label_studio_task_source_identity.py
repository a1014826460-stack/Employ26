"""Build full task -> source-record identity tables for Label Studio tasks."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy import bindparam, text

from src.db.label_studio_task_source_identity import (
    DEFAULT_CANDIDATE_TABLE,
    DEFAULT_IDENTITY_TABLE,
    build_task_payload_fingerprint,
    ensure_task_source_identity_tables,
    insert_task_source_identity_candidate_rows,
    quote_table_name,
    upsert_task_source_identity_rows,
)
from src.db.postgres import create_pg_engine
from src.utils.repair_label_studio_recruitment_record_ids import (
    MatchCandidate,
    build_source_index,
    build_task_payload,
    choose_candidate,
    display_text,
    load_locator_to_rrid,
    load_sample_tables,
    normalize_text,
    rank_candidates,
)


LOGGER = logging.getLogger(__name__)
TASK_TABLE = "annotations.label_studio_tasks_v2"
NORMALIZED_TABLE = "public.recruitment_jobs_normalized"
RESOLVER_VERSION = "label_studio_task_source_identity_v1"
DEFAULT_TOP_N = 10


def _parse_publish_date(value: object) -> tuple[int, int]:
    """Return a sortable publish-date key; newer dates rank higher."""
    raw = display_text(value)
    if not raw:
        return (0, 0)
    normalized = raw.replace("/", "-").replace(".", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m%d"):
        try:
            dt = datetime.strptime(normalized, fmt)
            return (1, int(dt.strftime("%Y%m%d")))
        except ValueError:
            continue
    digits = "".join(ch for ch in normalized if ch.isdigit())
    if len(digits) >= 8:
        try:
            dt = datetime.strptime(digits[:8], "%Y%m%d")
            return (1, int(dt.strftime("%Y%m%d")))
        except ValueError:
            pass
    return (0, 0)


def pick_forced_candidate(ranked: list[MatchCandidate]) -> tuple[MatchCandidate | None, int | None]:
    """Pick one candidate heuristically when we no longer want REVIEW_REQUIRED."""
    if not ranked:
        return None, None
    decorated = []
    for idx, candidate in enumerate(ranked, start=1):
        has_date, date_value = _parse_publish_date(candidate.publish_date)
        decorated.append(
            (
                -float(candidate.score),
                -int(has_date),
                -int(date_value),
                idx,
                candidate,
            )
        )
    decorated.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=False)
    chosen = decorated[0][4]
    chosen_rank = decorated[0][3]
    return chosen, chosen_rank


def load_normalized_source_index() -> dict[str, list[MatchCandidate]]:
    """Load source candidates using the proven sample-table title/description path.

    We intentionally preserve the current repair-script evidence path:
    sample tables provide the matching text, while normalized rows only enrich
    metadata by recruitment_record_id.
    """
    engine = create_pg_engine(application_name="task_source_identity_candidates")
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    select
                        recruitment_record_id,
                        source_platform,
                        source_table,
                        source_row_number,
                        source_native_job_id,
                        dedupe_fingerprint,
                        job_title,
                        job_description_raw,
                        work_city,
                        company_name,
                        publish_date,
                        salary_raw,
                        education_requirement_raw,
                        experience_requirement_raw,
                        company_industry_raw
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
            metadata_by_rrid = {
                display_text(row.get("recruitment_record_id")): dict(row)
                for row in rows
                if display_text(row.get("recruitment_record_id"))
            }
    finally:
        engine.dispose()

    base_index = build_source_index(load_sample_tables(), load_locator_to_rrid())
    enriched: dict[str, list[MatchCandidate]] = defaultdict(list)
    for title_key, candidates in base_index.items():
        for candidate in candidates:
            meta = metadata_by_rrid.get(candidate.recruitment_record_id, {})
            enriched[title_key].append(
                MatchCandidate(
                    source_table=candidate.source_table,
                    source_platform=display_text(meta.get("source_platform")) or candidate.source_platform,
                    source_row_number=candidate.source_row_number,
                    recruitment_record_id=candidate.recruitment_record_id,
                    source_native_job_id=display_text(meta.get("source_native_job_id")),
                    dedupe_fingerprint=display_text(meta.get("dedupe_fingerprint")),
                    title=candidate.title,
                    description=candidate.description,
                    company_name=candidate.company_name,
                    work_city=display_text(meta.get("work_city")),
                    salary_raw=display_text(meta.get("salary_raw")),
                    publish_date=display_text(meta.get("publish_date")),
                    education_requirement_raw=display_text(meta.get("education_requirement_raw")),
                    experience_requirement_raw=display_text(meta.get("experience_requirement_raw")),
                    company_industry_raw=display_text(meta.get("company_industry_raw")),
                    score=0.0,
                )
            )
    return enriched


def load_task_rows() -> list[dict[str, Any]]:
    engine = create_pg_engine(application_name="task_source_identity_tasks")
    try:
        with engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    text(
                        f"""
                        select
                            id,
                            row_id,
                            sample_source,
                            job_title,
                            job_requirements,
                            recruitment_record_id,
                            data_raw
                        from {TASK_TABLE}
                        order by id
                        """
                    )
                ).mappings()
            ]
    finally:
        engine.dispose()


def resolve_task_source_identity_rows(
    tasks: list[dict[str, Any]],
    source_index: dict[str, list[MatchCandidate]],
    *,
    top_n: int = DEFAULT_TOP_N,
    force_pick_review_required: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Resolve main identity rows and candidate rows for all tasks."""
    identity_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for task in tasks:
        payload = build_task_payload(task)
        title_key = normalize_text(payload["job_title"])
        candidates = source_index.get(title_key, [])
        ranked = rank_candidates(payload, candidates) if candidates else []
        status, rule, best, best_score, _second_score, candidate_count = choose_candidate(payload, candidates)
        forced_pick_count = 0
        if force_pick_review_required and status == "REVIEW_REQUIRED":
            forced_candidate, forced_rank = pick_forced_candidate(ranked)
            if forced_candidate is not None:
                status = "AUTO_CONFIRMED"
                rule = f"{rule}+forced_latest"
                best = forced_candidate
                best_score = forced_candidate.score
                forced_pick_count = 1
            else:
                forced_rank = None
        else:
            forced_rank = None
        status_counts[status] += 1

        selected_rank = None
        if best is not None:
            if forced_rank is not None:
                selected_rank = forced_rank
            else:
                for rank, candidate in enumerate(ranked, start=1):
                    if (
                        candidate.recruitment_record_id == best.recruitment_record_id
                        and candidate.source_table == best.source_table
                        and candidate.source_row_number == best.source_row_number
                    ):
                        selected_rank = rank
                        break

        payload_fingerprint = build_task_payload_fingerprint(
            task_job_title=payload["job_title"],
            task_job_requirements=payload["job_requirements"],
            task_sample_source=payload.get("task_sample_source", ""),
            task_snapshot_row_id=payload.get("task_snapshot_row_id", ""),
        )

        identity_rows.append(
            {
                "task_id": int(payload["task_id"]),
                "identity_status": status,
                "identity_rule": rule,
                "selected_rank": selected_rank,
                "selected_score": best_score,
                "selected_recruitment_record_id": best.recruitment_record_id if best else "",
                "selected_source_platform": best.source_platform if best else "",
                "selected_source_table": best.source_table if best else "",
                "selected_source_row_number": best.source_row_number if best else None,
                "selected_source_native_job_id": best.source_native_job_id if best else "",
                "selected_dedupe_fingerprint": best.dedupe_fingerprint if best else "",
                "selected_company_name": best.company_name if best else "",
                "selected_work_city": best.work_city if best else "",
                "selected_salary_raw": best.salary_raw if best else "",
                "selected_publish_date": best.publish_date if best else "",
                "selected_job_title": best.title if best else "",
                "selected_job_description_raw": best.description if best else "",
                "task_job_title": payload["job_title"],
                "task_job_requirements": payload["job_requirements"],
                "task_sample_source": payload.get("task_sample_source", ""),
                "task_snapshot_row_id": payload.get("task_snapshot_row_id", ""),
                "task_payload_fingerprint": payload_fingerprint,
                "resolver_version": RESOLVER_VERSION,
                "review_decision": "",
                "review_notes": "",
                "reviewed_by": "",
                "reviewed_at": None,
            }
        )

        for rank, candidate in enumerate(ranked[: max(0, int(top_n))], start=1):
            candidate_rows.append(
                {
                    "task_id": int(payload["task_id"]),
                    "candidate_rank": rank,
                    "candidate_score": candidate.score,
                    "is_selected": rank == selected_rank,
                    "selection_reason": rule if rank == selected_rank else "",
                    "recruitment_record_id": candidate.recruitment_record_id,
                    "source_platform": candidate.source_platform,
                    "source_table": candidate.source_table,
                    "source_row_number": candidate.source_row_number,
                    "source_native_job_id": candidate.source_native_job_id,
                    "dedupe_fingerprint": candidate.dedupe_fingerprint,
                    "job_title": candidate.title,
                    "company_name": candidate.company_name,
                    "work_city": candidate.work_city,
                    "salary_raw": candidate.salary_raw,
                    "publish_date": candidate.publish_date,
                    "job_description_raw": candidate.description,
                    "education_requirement_raw": candidate.education_requirement_raw,
                    "experience_requirement_raw": candidate.experience_requirement_raw,
                    "company_industry_raw": candidate.company_industry_raw,
                }
            )

    summary = {
        "total_tasks": len(tasks),
        "status_counts": dict(status_counts),
        "identity_rows": len(identity_rows),
        "candidate_rows": len(candidate_rows),
        "review_required": status_counts.get("REVIEW_REQUIRED", 0),
        "forced_auto_confirmed": sum(
            1 for row in identity_rows if str(row["identity_rule"]).endswith("+forced_latest")
        ),
    }
    return identity_rows, candidate_rows, summary


def build_task_source_identity(
    *,
    dry_run: bool = False,
    top_n: int = DEFAULT_TOP_N,
    identity_table: str = DEFAULT_IDENTITY_TABLE,
    candidate_table: str = DEFAULT_CANDIDATE_TABLE,
    force_pick_review_required: bool = False,
) -> dict[str, Any]:
    LOGGER.info("加载 normalized sample source index ...")
    source_index = load_normalized_source_index()
    LOGGER.info("source title index keys: %s", len(source_index))
    LOGGER.info("加载 Label Studio tasks ...")
    tasks = load_task_rows()
    identity_rows, candidate_rows, summary = resolve_task_source_identity_rows(
        tasks,
        source_index,
        top_n=top_n,
        force_pick_review_required=force_pick_review_required,
    )
    summary = {
        **summary,
        "dry_run": dry_run,
        "identity_table": identity_table,
        "candidate_table": candidate_table,
        "resolver_version": RESOLVER_VERSION,
        "force_pick_review_required": force_pick_review_required,
    }
    if dry_run:
        LOGGER.info("dry-run summary: %s", summary)
        return summary

    engine = create_pg_engine(application_name="task_source_identity_apply")
    try:
        with engine.begin() as conn:
            ensure_task_source_identity_tables(
                conn,
                identity_table=identity_table,
                candidate_table=candidate_table,
            )
            conn.execute(text(f"truncate table {quote_table_name(candidate_table)}"))
            upsert_task_source_identity_rows(conn, identity_rows, identity_table=identity_table)
            insert_task_source_identity_candidate_rows(conn, candidate_rows, candidate_table=candidate_table)
    finally:
        engine.dispose()
    LOGGER.info("build summary: %s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full Label Studio task source identity tables")
    parser.add_argument("--dry-run", action="store_true", help="Only compute mappings; do not write database")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Candidate rows to persist per task")
    parser.add_argument("--identity-table", default=DEFAULT_IDENTITY_TABLE, help="Main identity table name")
    parser.add_argument("--candidate-table", default=DEFAULT_CANDIDATE_TABLE, help="Candidate evidence table name")
    parser.add_argument(
        "--force-pick-review-required",
        action="store_true",
        help="Heuristically pick one candidate for REVIEW_REQUIRED tasks using score first, then newer publish_date.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    summary = build_task_source_identity(
        dry_run=bool(args.dry_run),
        top_n=max(0, int(args.top_n)),
        identity_table=str(args.identity_table),
        candidate_table=str(args.candidate_table),
        force_pick_review_required=bool(args.force_pick_review_required),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
