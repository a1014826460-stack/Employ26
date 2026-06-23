"""Helpers for Label Studio task -> source-record identity tables."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any

from sqlalchemy import text

from src.db.postgres import ensure_schema


DEFAULT_IDENTITY_TABLE = "annotations.label_studio_task_source_identity"
DEFAULT_CANDIDATE_TABLE = "annotations.label_studio_task_source_identity_candidates"

ACTIVE_IDENTITY_STATUSES = ("AUTO_CONFIRMED", "MANUAL_CONFIRMED")
PROTECTED_IDENTITY_STATUSES = ("MANUAL_CONFIRMED", "MANUAL_REJECTED")


def split_table_name(table_name: str) -> tuple[str, str]:
    normalized = str(table_name).strip()
    if "." not in normalized:
        return "public", normalized.strip('"')
    schema, table = normalized.split(".", 1)
    return schema.strip().strip('"'), table.strip().strip('"')


def quote_table_name(table_name: str) -> str:
    schema_name, raw_table_name = split_table_name(table_name)
    return f'"{schema_name}"."{raw_table_name}"'


def safe_text(value: object) -> str:
    if value is None:
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def build_task_payload_fingerprint(
    *,
    task_job_title: str,
    task_job_requirements: str,
    task_sample_source: str,
    task_snapshot_row_id: str,
) -> str:
    """Build a stable fingerprint for the current task payload."""
    parts = [
        safe_text(task_job_title).casefold(),
        safe_text(task_job_requirements).casefold(),
        safe_text(task_sample_source).casefold(),
        safe_text(task_snapshot_row_id).casefold(),
    ]
    canonical = "||".join(parts)
    return sha1(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSourceIdentityRow:
    task_id: int
    identity_status: str
    identity_rule: str
    selected_rank: int | None
    selected_score: float | None
    selected_recruitment_record_id: str
    selected_source_platform: str
    selected_source_table: str
    selected_source_row_number: int | None
    selected_source_native_job_id: str
    selected_dedupe_fingerprint: str
    selected_company_name: str
    selected_work_city: str
    selected_salary_raw: str
    selected_publish_date: str
    selected_job_title: str
    selected_job_description_raw: str
    task_job_title: str
    task_job_requirements: str
    task_sample_source: str
    task_snapshot_row_id: str
    task_payload_fingerprint: str
    resolver_version: str
    review_decision: str
    review_notes: str
    reviewed_by: str
    reviewed_at: str | None


@dataclass(frozen=True)
class TaskSourceIdentityCandidateRow:
    task_id: int
    candidate_rank: int
    candidate_score: float | None
    is_selected: bool
    selection_reason: str
    recruitment_record_id: str
    source_platform: str
    source_table: str
    source_row_number: int | None
    source_native_job_id: str
    dedupe_fingerprint: str
    job_title: str
    company_name: str
    work_city: str
    salary_raw: str
    publish_date: str
    job_description_raw: str
    education_requirement_raw: str
    experience_requirement_raw: str
    company_industry_raw: str


def ensure_task_source_identity_tables(
    connection,
    *,
    identity_table: str = DEFAULT_IDENTITY_TABLE,
    candidate_table: str = DEFAULT_CANDIDATE_TABLE,
) -> None:
    """Create the identity main table and candidate evidence table."""
    for table_name in (identity_table, candidate_table):
        schema_name, _ = split_table_name(table_name)
        ensure_schema(connection, schema_name)

    qualified_identity = quote_table_name(identity_table)
    qualified_candidate = quote_table_name(candidate_table)
    identity_prefix = split_table_name(identity_table)[1].replace('"', "").replace(".", "_")
    candidate_prefix = split_table_name(candidate_table)[1].replace('"', "").replace(".", "_")

    connection.execute(
        text(
            f"""
            create table if not exists {qualified_identity} (
                task_id integer primary key,
                identity_status text not null,
                identity_rule text not null,
                selected_rank integer,
                selected_score double precision,
                selected_recruitment_record_id text,
                selected_source_platform text,
                selected_source_table text,
                selected_source_row_number bigint,
                selected_source_native_job_id text not null default '',
                selected_dedupe_fingerprint text not null default '',
                selected_company_name text,
                selected_work_city text,
                selected_salary_raw text,
                selected_publish_date text,
                selected_job_title text,
                selected_job_description_raw text,
                task_job_title text,
                task_job_requirements text,
                task_sample_source text,
                task_snapshot_row_id text,
                task_payload_fingerprint text not null,
                resolver_version text not null,
                review_decision text not null default '',
                review_notes text not null default '',
                reviewed_by text not null default '',
                reviewed_at timestamptz,
                created_at timestamptz not null default now(),
                updated_at timestamptz not null default now(),
                constraint fk_label_studio_task_source_identity_task
                    foreign key (task_id)
                    references annotations.label_studio_tasks_v2 (id)
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            create table if not exists {qualified_candidate} (
                task_id integer not null,
                candidate_rank integer not null,
                candidate_score double precision,
                is_selected boolean not null default false,
                selection_reason text not null default '',
                recruitment_record_id text,
                source_platform text,
                source_table text,
                source_row_number bigint,
                source_native_job_id text not null default '',
                dedupe_fingerprint text not null default '',
                job_title text,
                company_name text,
                work_city text,
                salary_raw text,
                publish_date text,
                job_description_raw text,
                education_requirement_raw text,
                experience_requirement_raw text,
                company_industry_raw text,
                created_at timestamptz not null default now(),
                primary key (task_id, candidate_rank),
                constraint fk_label_studio_task_source_identity_candidates_task
                    foreign key (task_id)
                    references annotations.label_studio_tasks_v2 (id)
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            create index if not exists idx_{identity_prefix}_status
            on {qualified_identity} (identity_status)
            """
        )
    )
    connection.execute(
        text(
            f"""
            create index if not exists idx_{identity_prefix}_selected_rrid
            on {qualified_identity} (selected_recruitment_record_id)
            """
        )
    )
    connection.execute(
        text(
            f"""
            create index if not exists idx_{identity_prefix}_resolver_version
            on {qualified_identity} (resolver_version)
            """
        )
    )
    connection.execute(
        text(
            f"""
            create index if not exists idx_{candidate_prefix}_rrid
            on {qualified_candidate} (recruitment_record_id)
            """
        )
    )
    connection.execute(
        text(
            f"""
            create index if not exists idx_{candidate_prefix}_source_locator
            on {qualified_candidate} (source_table, source_row_number)
            """
        )
    )


def truncate_task_source_identity_tables(
    connection,
    *,
    identity_table: str = DEFAULT_IDENTITY_TABLE,
    candidate_table: str = DEFAULT_CANDIDATE_TABLE,
) -> None:
    connection.execute(text(f"truncate table {quote_table_name(candidate_table)}"))
    connection.execute(text(f"truncate table {quote_table_name(identity_table)}"))


def upsert_task_source_identity_rows(
    connection,
    rows: list[dict[str, Any]],
    *,
    identity_table: str = DEFAULT_IDENTITY_TABLE,
) -> None:
    if not rows:
        return
    qualified_table = quote_table_name(identity_table)
    connection.execute(
        text(
            f"""
            insert into {qualified_table} (
                task_id,
                identity_status,
                identity_rule,
                selected_rank,
                selected_score,
                selected_recruitment_record_id,
                selected_source_platform,
                selected_source_table,
                selected_source_row_number,
                selected_source_native_job_id,
                selected_dedupe_fingerprint,
                selected_company_name,
                selected_work_city,
                selected_salary_raw,
                selected_publish_date,
                selected_job_title,
                selected_job_description_raw,
                task_job_title,
                task_job_requirements,
                task_sample_source,
                task_snapshot_row_id,
                task_payload_fingerprint,
                resolver_version,
                review_decision,
                review_notes,
                reviewed_by,
                reviewed_at
            ) values (
                :task_id,
                :identity_status,
                :identity_rule,
                :selected_rank,
                :selected_score,
                :selected_recruitment_record_id,
                :selected_source_platform,
                :selected_source_table,
                :selected_source_row_number,
                :selected_source_native_job_id,
                :selected_dedupe_fingerprint,
                :selected_company_name,
                :selected_work_city,
                :selected_salary_raw,
                :selected_publish_date,
                :selected_job_title,
                :selected_job_description_raw,
                :task_job_title,
                :task_job_requirements,
                :task_sample_source,
                :task_snapshot_row_id,
                :task_payload_fingerprint,
                :resolver_version,
                :review_decision,
                :review_notes,
                :reviewed_by,
                :reviewed_at
            )
            on conflict (task_id) do update set
                identity_status = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.identity_status
                    else excluded.identity_status
                end,
                identity_rule = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.identity_rule
                    else excluded.identity_rule
                end,
                selected_rank = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_rank
                    else excluded.selected_rank
                end,
                selected_score = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_score
                    else excluded.selected_score
                end,
                selected_recruitment_record_id = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_recruitment_record_id
                    else excluded.selected_recruitment_record_id
                end,
                selected_source_platform = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_source_platform
                    else excluded.selected_source_platform
                end,
                selected_source_table = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_source_table
                    else excluded.selected_source_table
                end,
                selected_source_row_number = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_source_row_number
                    else excluded.selected_source_row_number
                end,
                selected_source_native_job_id = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_source_native_job_id
                    else excluded.selected_source_native_job_id
                end,
                selected_dedupe_fingerprint = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_dedupe_fingerprint
                    else excluded.selected_dedupe_fingerprint
                end,
                selected_company_name = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_company_name
                    else excluded.selected_company_name
                end,
                selected_work_city = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_work_city
                    else excluded.selected_work_city
                end,
                selected_salary_raw = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_salary_raw
                    else excluded.selected_salary_raw
                end,
                selected_publish_date = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_publish_date
                    else excluded.selected_publish_date
                end,
                selected_job_title = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_job_title
                    else excluded.selected_job_title
                end,
                selected_job_description_raw = case
                    when {qualified_table}.identity_status in ('MANUAL_CONFIRMED', 'MANUAL_REJECTED')
                    then {qualified_table}.selected_job_description_raw
                    else excluded.selected_job_description_raw
                end,
                task_job_title = excluded.task_job_title,
                task_job_requirements = excluded.task_job_requirements,
                task_sample_source = excluded.task_sample_source,
                task_snapshot_row_id = excluded.task_snapshot_row_id,
                task_payload_fingerprint = excluded.task_payload_fingerprint,
                resolver_version = excluded.resolver_version,
                updated_at = now()
            """
        ),
        rows,
    )


def insert_task_source_identity_candidate_rows(
    connection,
    rows: list[dict[str, Any]],
    *,
    candidate_table: str = DEFAULT_CANDIDATE_TABLE,
) -> None:
    if not rows:
        return
    qualified_table = quote_table_name(candidate_table)
    connection.execute(
        text(
            f"""
            insert into {qualified_table} (
                task_id,
                candidate_rank,
                candidate_score,
                is_selected,
                selection_reason,
                recruitment_record_id,
                source_platform,
                source_table,
                source_row_number,
                source_native_job_id,
                dedupe_fingerprint,
                job_title,
                company_name,
                work_city,
                salary_raw,
                publish_date,
                job_description_raw,
                education_requirement_raw,
                experience_requirement_raw,
                company_industry_raw
            ) values (
                :task_id,
                :candidate_rank,
                :candidate_score,
                :is_selected,
                :selection_reason,
                :recruitment_record_id,
                :source_platform,
                :source_table,
                :source_row_number,
                :source_native_job_id,
                :dedupe_fingerprint,
                :job_title,
                :company_name,
                :work_city,
                :salary_raw,
                :publish_date,
                :job_description_raw,
                :education_requirement_raw,
                :experience_requirement_raw,
                :company_industry_raw
            )
            on conflict (task_id, candidate_rank) do update set
                candidate_score = excluded.candidate_score,
                is_selected = excluded.is_selected,
                selection_reason = excluded.selection_reason,
                recruitment_record_id = excluded.recruitment_record_id,
                source_platform = excluded.source_platform,
                source_table = excluded.source_table,
                source_row_number = excluded.source_row_number,
                source_native_job_id = excluded.source_native_job_id,
                dedupe_fingerprint = excluded.dedupe_fingerprint,
                job_title = excluded.job_title,
                company_name = excluded.company_name,
                work_city = excluded.work_city,
                salary_raw = excluded.salary_raw,
                publish_date = excluded.publish_date,
                job_description_raw = excluded.job_description_raw,
                education_requirement_raw = excluded.education_requirement_raw,
                experience_requirement_raw = excluded.experience_requirement_raw,
                company_industry_raw = excluded.company_industry_raw
            """
        ),
        rows,
    )


def load_active_task_identity_map(
    connection,
    *,
    identity_table: str = DEFAULT_IDENTITY_TABLE,
) -> dict[int, dict[str, Any]]:
    """Load task identities usable by downstream consumers."""
    qualified_table = quote_table_name(identity_table)
    rows = connection.execute(
        text(
            f"""
            select *
            from {qualified_table}
            where identity_status in ('AUTO_CONFIRMED', 'MANUAL_CONFIRMED')
            """
        )
    ).mappings()
    return {int(row["task_id"]): dict(row) for row in rows}


def load_all_task_identity_map(
    connection,
    *,
    identity_table: str = DEFAULT_IDENTITY_TABLE,
) -> dict[int, dict[str, Any]]:
    """Load all task identities regardless of status."""
    qualified_table = quote_table_name(identity_table)
    rows = connection.execute(text(f"select * from {qualified_table}")).mappings()
    return {int(row["task_id"]): dict(row) for row in rows}
