"""将三家招聘平台招聘表补充写入统一规范层。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
from sqlalchemy import bindparam, text

from src.db.postgres import create_pg_engine
from src.db.recruitment_jobs_normalized import (
    DEFAULT_BACKFILL_CHUNK_STATE_TABLE,
    DEFAULT_BACKFILL_LOCATOR_TABLE,
    DEFAULT_NORMALIZED_TABLE,
    build_normalized_rows_from_dataframe,
    ensure_recruitment_jobs_normalized_backfill_runtime_tables,
    ensure_recruitment_jobs_normalized_table,
    infer_source_platform,
    quote_table_name,
    split_table_name,
    upsert_recruitment_jobs_normalized,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_SOURCE_TABLES = [
    '"51job".sample',
    '"Liepin".sample',
    '"Zhilian".sample',
]

DEFAULT_FULL_SOURCE_TABLES = [
    '"51job".cleaned_data',
    '"Liepin".cleaned_data',
    '"Zhilian".cleaned_data',
]

REPAIRABLE_SOURCE_TABLE_MAPPINGS = {
    '"51job"."cleaned_data"': '"51job".cleaned_data',
    '"Liepin"."cleaned_data"': '"Liepin".cleaned_data',
    '"Zhilian"."cleaned_data"': '"Zhilian".cleaned_data',
    '"51job"."sample"': '"51job".sample',
    '"Liepin"."sample"': '"Liepin".sample',
    '"Zhilian"."sample"': '"Zhilian".sample',
}

DEFAULT_WORKERS = 2
DEFAULT_CHUNK_SIZE = 25000
DEFAULT_MAX_RETRIES = 2
DEFAULT_POOL_RECYCLE_SECONDS = 1800


@dataclass(frozen=True)
class ChunkPlan:
    """单个 chunk 的执行计划。"""

    chunk_id: int
    range_start: int
    range_end: int
    planned_rows: int
    source_table: str = ""


@dataclass(frozen=True)
class ChunkResult:
    """单个 chunk 的执行结果。"""

    run_id: str
    source_table: str
    chunk_id: int
    range_start: int
    range_end: int
    planned_rows: int
    written_rows: int
    duration_seconds: float
    attempt: int
    status: str
    error_message: str = ""

    @property
    def rows_per_second(self) -> float:
        """返回当前 chunk 的吞吐。"""
        if self.duration_seconds <= 0:
            return 0.0
        return float(self.written_rows) / float(self.duration_seconds)


def canonicalize_source_table_name(table_name: str) -> str:
    """把用户输入的 source 表名标准化为可直接用于 SQL 的格式。"""
    schema_name, raw_table_name = split_table_name(str(table_name).strip())
    if not schema_name or not raw_table_name:
        raise ValueError(f"无效 source_table: {table_name!r}")
    return f'"{schema_name}".{raw_table_name}'


def _load_source_dataframe(
    source_table: str,
    normalized_table: str,
    only_missing: bool = False,
    offset: int = 0,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    """读取单张 PostgreSQL source 表，并生成稳定 source_row_number。"""
    limit_sql = ""
    params: dict[str, object] = {
        "source_table": source_table,
        "offset_rows": int(offset),
    }
    if limit_rows is not None:
        limit_sql = "LIMIT :limit_rows"
        params["limit_rows"] = int(limit_rows)
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            where_sql = "WHERE n.recruitment_record_id IS NULL" if only_missing else ""
            return pd.read_sql_query(
                text(
                    f"""
                    WITH src AS (
                        SELECT
                            row_number() OVER (ORDER BY ctid) AS __source_row_number,
                            *
                        FROM {source_table}
                    )
                    SELECT src.*
                    FROM src
                    LEFT JOIN {quote_table_name(normalized_table)} n
                      ON n.source_table = :source_table
                     AND n.source_row_number = src.__source_row_number
                    {where_sql}
                    ORDER BY src.__source_row_number
                    {limit_sql}
                    OFFSET :offset_rows
                    """
                ),
                connection,
                params=params,
            )
    finally:
        engine.dispose()


def _count_missing_source_rows(source_table: str, normalized_table: str) -> dict[str, int]:
    """统计指定 source_table 中尚未进入统一规范层的记录数。"""
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    f"""
                    WITH src AS (
                        SELECT row_number() OVER (ORDER BY ctid) AS source_row_number
                        FROM {source_table}
                    )
                    SELECT
                        count(*) AS sample_rows,
                        sum(CASE WHEN n.recruitment_record_id IS NULL THEN 1 ELSE 0 END) AS missing_in_normalized,
                        sum(CASE WHEN n.recruitment_record_id IS NOT NULL THEN 1 ELSE 0 END) AS present_in_normalized
                    FROM src s
                    LEFT JOIN {quote_table_name(normalized_table)} n
                      ON n.source_table = :source_table
                     AND n.source_row_number = s.source_row_number
                    """
                ),
                {"source_table": source_table},
            ).mappings().one()
            return {key: int(value or 0) for key, value in row.items()}
    finally:
        engine.dispose()


def _sql_safe_text(column_name: str) -> str:
    """生成 PostgreSQL 侧安全转文本表达式。"""
    return f"COALESCE(NULLIF(BTRIM(src.{column_name}::text), 'nan'), '')"


def plan_chunks(
    *,
    total_rows: int,
    chunk_size: int,
    limit_rows: int | None = None,
    source_table: str = "",
) -> list[ChunkPlan]:
    """根据总行数生成稳定的 chunk 列表。"""
    effective_total_rows = max(0, int(total_rows))
    if limit_rows is not None:
        effective_total_rows = min(effective_total_rows, max(0, int(limit_rows)))
    resolved_chunk_size = max(1, int(chunk_size))
    chunks: list[ChunkPlan] = []
    for chunk_id, start in enumerate(range(1, effective_total_rows + 1, resolved_chunk_size), start=1):
        end = min(start + resolved_chunk_size - 1, effective_total_rows)
        chunks.append(
            ChunkPlan(
                chunk_id=chunk_id,
                range_start=start,
                range_end=end,
                planned_rows=end - start + 1,
                source_table=source_table,
            )
        )
    return chunks


def _calculate_percentile(values: list[float], percentile: float) -> float:
    """计算简单分位数。"""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(percentile))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_run_summary(
    *,
    chunk_results: list[ChunkResult],
    total_duration_seconds: float | None = None,
) -> dict[str, object]:
    """按 chunk 结果构建运行汇总。"""
    durations = [float(result.duration_seconds) for result in chunk_results]
    duration_base = float(total_duration_seconds) if total_duration_seconds is not None else sum(durations)
    written_rows = int(sum(int(result.written_rows) for result in chunk_results))
    planned_rows = int(sum(int(result.planned_rows) for result in chunk_results))
    failed_chunks = int(sum(1 for result in chunk_results if result.status != "succeeded"))
    table_summary: dict[str, dict[str, float | int]] = {}
    for result in chunk_results:
        current = table_summary.setdefault(
            result.source_table,
            {
                "chunks": 0,
                "written_rows": 0,
                "planned_rows": 0,
                "duration_seconds": 0.0,
            },
        )
        current["chunks"] = int(current["chunks"]) + 1
        current["written_rows"] = int(current["written_rows"]) + int(result.written_rows)
        current["planned_rows"] = int(current["planned_rows"]) + int(result.planned_rows)
        current["duration_seconds"] = float(current["duration_seconds"]) + float(result.duration_seconds)
    for current in table_summary.values():
        duration_value = float(current["duration_seconds"])
        current["rows_per_second"] = (
            float(current["written_rows"]) / duration_value if duration_value > 0 else 0.0
        )
    return {
        "planned_rows": planned_rows,
        "written_rows": written_rows,
        "duration_seconds": duration_base,
        "rows_per_second": (float(written_rows) / duration_base) if duration_base > 0 else 0.0,
        "failed_chunks": failed_chunks,
        "chunk_duration_p50": _calculate_percentile(durations, 0.50),
        "chunk_duration_p95": _calculate_percentile(durations, 0.95),
        "chunk_duration_max": max(durations) if durations else 0.0,
        "tables": table_summary,
    }


def build_chunk_insert_sql(
    *,
    locator_table: str,
    normalized_table: str,
    source_table: str,
    only_missing: bool,
) -> str:
    """构建单个 chunk 的 SQL 回填语句。"""
    qualified_locator = quote_table_name(locator_table)
    qualified_normalized = quote_table_name(normalized_table)
    update_columns = [
        "source_platform",
        "source_native_job_id",
        "dedupe_fingerprint",
        "job_title",
        "job_description_raw",
        "work_city",
        "company_name",
        "publish_date",
        "salary_raw",
        "education_requirement_raw",
        "experience_requirement_raw",
        "company_size_raw",
        "company_industry_raw",
    ]
    conflict_sql = "ON CONFLICT (source_table, source_row_number) DO NOTHING"
    if not only_missing:
        conflict_sql = f"""
        ON CONFLICT (source_table, source_row_number)
        DO UPDATE SET
            source_platform = EXCLUDED.source_platform,
            source_native_job_id = EXCLUDED.source_native_job_id,
            dedupe_fingerprint = EXCLUDED.dedupe_fingerprint,
            job_title = EXCLUDED.job_title,
            job_description_raw = EXCLUDED.job_description_raw,
            work_city = EXCLUDED.work_city,
            company_name = EXCLUDED.company_name,
            publish_date = EXCLUDED.publish_date,
            salary_raw = EXCLUDED.salary_raw,
            education_requirement_raw = EXCLUDED.education_requirement_raw,
            experience_requirement_raw = EXCLUDED.experience_requirement_raw,
            company_size_raw = EXCLUDED.company_size_raw,
            company_industry_raw = EXCLUDED.company_industry_raw,
            updated_at = now()
        WHERE
            {" OR ".join(f"{qualified_normalized}.{column_name} IS DISTINCT FROM EXCLUDED.{column_name}" for column_name in update_columns)}
        """
    return f"""
        WITH selected AS (
            SELECT
                loc.source_row_number,
                src.*
            FROM {qualified_locator} loc
            JOIN {source_table} src
              ON src.ctid = loc.source_ctid
            LEFT JOIN {qualified_normalized} existing
              ON existing.source_table = :source_table
             AND existing.source_row_number = loc.source_row_number
            WHERE loc.run_id = :run_id
              AND loc.source_table = :source_table
              AND loc.chunk_id = :chunk_id
              {"AND existing.recruitment_record_id IS NULL" if only_missing else ""}
            ORDER BY loc.source_row_number
        ),
        prepared AS (
            SELECT
                gen_random_uuid()::text AS recruitment_record_id,
                :source_platform AS source_platform,
                :source_table AS source_table,
                src.source_row_number AS source_row_number,
                '' AS source_native_job_id,
                md5(
                    lower(:source_platform) || '||' ||
                    lower({_sql_safe_text('"公司名称"')}) || '||' ||
                    lower({_sql_safe_text('"岗位名称"')}) || '||' ||
                    lower({_sql_safe_text('"岗位描述"')}) || '||' ||
                    lower({_sql_safe_text('"发布时间"')}) || '||' ||
                    lower({_sql_safe_text('"工作城市"')})
                ) AS dedupe_fingerprint,
                {_sql_safe_text('"岗位名称"')} AS job_title,
                {_sql_safe_text('"岗位描述"')} AS job_description_raw,
                {_sql_safe_text('"工作城市"')} AS work_city,
                {_sql_safe_text('"公司名称"')} AS company_name,
                {_sql_safe_text('"发布时间"')} AS publish_date,
                {_sql_safe_text('"薪资水平"')} AS salary_raw,
                {_sql_safe_text('"学历要求"')} AS education_requirement_raw,
                {_sql_safe_text('"经验要求"')} AS experience_requirement_raw,
                {_sql_safe_text('"公司规模"')} AS company_size_raw,
                {_sql_safe_text('"公司行业"')} AS company_industry_raw
            FROM selected src
        )
        , inserted AS (
            INSERT INTO {qualified_normalized} (
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
                company_size_raw,
                company_industry_raw
            )
            SELECT
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
                company_size_raw,
                company_industry_raw
            FROM prepared
            {conflict_sql}
            RETURNING 1
        )
        SELECT count(*) AS inserted_rows
        FROM inserted
    """


def repair_source_table_values(
    *,
    normalized_table: str = DEFAULT_NORMALIZED_TABLE,
    locator_table: str = DEFAULT_BACKFILL_LOCATOR_TABLE,
    chunk_state_table: str = DEFAULT_BACKFILL_CHUNK_STATE_TABLE,
    mappings: dict[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """修复历史错误的 source_table 值，并清理对应 runtime 行。

    说明:
        该函数只处理已知坏值 -> 正确值的映射。
        若目标表中已存在相同 source_row_number 的正确值，会先删除正确值，再把坏值改名，
        以避免 `(source_table, source_row_number)` 唯一键冲突。
    """
    mapping_items = list((mappings or REPAIRABLE_SOURCE_TABLE_MAPPINGS).items())
    if not mapping_items:
        return {}

    normalized_table_name = quote_table_name(normalized_table)
    locator_table_name = quote_table_name(locator_table)
    chunk_state_table_name = quote_table_name(chunk_state_table)
    repair_summary: dict[str, dict[str, int]] = {}

    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            for bad_source_table, good_source_table in mapping_items:
                deleted_conflicts = connection.execute(
                    text(
                        f"""
                        DELETE FROM {normalized_table_name} target
                        USING {normalized_table_name} bad
                        WHERE bad.source_table = :bad_source_table
                          AND target.source_table = :good_source_table
                          AND target.source_row_number = bad.source_row_number
                        """
                    ),
                    {
                        "bad_source_table": bad_source_table,
                        "good_source_table": good_source_table,
                    },
                ).rowcount or 0
                updated_rows = connection.execute(
                    text(
                        f"""
                        UPDATE {normalized_table_name}
                        SET source_table = :good_source_table
                        WHERE source_table = :bad_source_table
                        """
                    ),
                    {
                        "bad_source_table": bad_source_table,
                        "good_source_table": good_source_table,
                    },
                ).rowcount or 0
                deleted_chunk_rows = connection.execute(
                    text(
                        f"""
                        DELETE FROM {chunk_state_table_name}
                        WHERE source_table = :bad_source_table
                        """
                    ),
                    {"bad_source_table": bad_source_table},
                ).rowcount or 0
                deleted_locator_rows = connection.execute(
                    text(
                        f"""
                        DELETE FROM {locator_table_name}
                        WHERE source_table = :bad_source_table
                        """
                    ),
                    {"bad_source_table": bad_source_table},
                ).rowcount or 0
                if any((deleted_conflicts, updated_rows, deleted_chunk_rows, deleted_locator_rows)):
                    repair_summary[bad_source_table] = {
                        "deleted_conflicting_rows": int(deleted_conflicts),
                        "updated_rows": int(updated_rows),
                        "deleted_chunk_rows": int(deleted_chunk_rows),
                        "deleted_locator_rows": int(deleted_locator_rows),
                        "target_source_table": good_source_table,
                    }
            connection.execute(text(f"ANALYZE {normalized_table_name}"))
    finally:
        engine.dispose()

    return repair_summary


def _effective_locator_limit_rows(limit_rows: int | None, max_chunks: int | None, chunk_size: int) -> int | None:
    """计算 locator 初始化需要覆盖的总行数。"""
    if limit_rows is not None:
        return max(0, int(limit_rows))
    if max_chunks is not None:
        return max(0, int(max_chunks)) * max(1, int(chunk_size))
    return None


def _locator_exists(connection, *, run_id: str, source_table: str, locator_table: str) -> bool:
    """判断给定 run 是否已存在 locator 快照。"""
    qualified_locator = quote_table_name(locator_table)
    existing = connection.execute(
        text(
            f"""
            SELECT 1
            FROM {qualified_locator}
            WHERE run_id = :run_id
              AND source_table = :source_table
            LIMIT 1
            """
        ),
        {
            "run_id": run_id,
            "source_table": source_table,
        },
    ).scalar_one_or_none()
    return existing is not None


def _prepare_source_locator(
    connection,
    *,
    run_id: str,
    source_table: str,
    chunk_size: int,
    locator_table: str,
    chunk_state_table: str,
    limit_rows: int | None,
) -> int:
    """构建单张 source 表的 locator 快照和 chunk 状态。"""
    qualified_locator = quote_table_name(locator_table)
    qualified_chunk_state = quote_table_name(chunk_state_table)
    params: dict[str, object] = {
        "run_id": run_id,
        "source_table": source_table,
        "chunk_size": max(1, int(chunk_size)),
    }
    limit_sql = ""
    if limit_rows is not None:
        params["limit_rows"] = max(0, int(limit_rows))
        limit_sql = "LIMIT :limit_rows"
    connection.execute(
        text(f"DELETE FROM {qualified_chunk_state} WHERE run_id = :run_id AND source_table = :source_table"),
        {
            "run_id": run_id,
            "source_table": source_table,
        },
    )
    connection.execute(
        text(f"DELETE FROM {qualified_locator} WHERE run_id = :run_id AND source_table = :source_table"),
        {
            "run_id": run_id,
            "source_table": source_table,
        },
    )
    connection.execute(
        text(
            f"""
            WITH ordered_source AS (
                SELECT ctid AS source_ctid
                FROM {source_table}
                ORDER BY ctid
                {limit_sql}
            ),
            numbered AS (
                SELECT
                    row_number() OVER (ORDER BY source_ctid) AS source_row_number,
                    source_ctid
                FROM ordered_source
            )
            INSERT INTO {qualified_locator} (
                run_id,
                source_table,
                source_row_number,
                source_ctid,
                chunk_id
            )
            SELECT
                :run_id,
                :source_table,
                source_row_number,
                source_ctid,
                ((source_row_number - 1) / :chunk_size) + 1 AS chunk_id
            FROM numbered
            ORDER BY source_row_number
            """
        ),
        params,
    )
    connection.execute(
        text(
            f"""
            INSERT INTO {qualified_chunk_state} (
                run_id,
                source_table,
                chunk_id,
                range_start,
                range_end,
                planned_rows,
                attempt,
                status
            )
            SELECT
                :run_id,
                :source_table,
                chunk_id,
                MIN(source_row_number) AS range_start,
                MAX(source_row_number) AS range_end,
                COUNT(*) AS planned_rows,
                0 AS attempt,
                'pending' AS status
            FROM {qualified_locator}
            WHERE run_id = :run_id
              AND source_table = :source_table
            GROUP BY chunk_id
            ORDER BY chunk_id
            """
        ),
        {
            "run_id": run_id,
            "source_table": source_table,
        },
    )
    row = connection.execute(
        text(
            f"""
            SELECT COUNT(*) AS planned_rows
            FROM {qualified_locator}
            WHERE run_id = :run_id
              AND source_table = :source_table
            """
        ),
        {
            "run_id": run_id,
            "source_table": source_table,
        },
    ).mappings().one()
    return int(row["planned_rows"] or 0)


def _load_chunk_plans(
    connection,
    *,
    run_id: str,
    source_table: str,
    chunk_state_table: str,
    max_chunks: int | None,
    include_failed: bool,
) -> list[ChunkPlan]:
    """读取待执行 chunk 列表。"""
    qualified_chunk_state = quote_table_name(chunk_state_table)
    statuses = ["pending", "running"]
    if include_failed:
        statuses.append("failed")
    query = text(
        f"""
        SELECT
            chunk_id,
            range_start,
            range_end,
            planned_rows
        FROM {qualified_chunk_state}
        WHERE run_id = :run_id
          AND source_table = :source_table
          AND status IN :statuses
        ORDER BY chunk_id
        """
    ).bindparams(bindparam("statuses", expanding=True))
    rows = connection.execute(
        query,
        {
            "run_id": run_id,
            "source_table": source_table,
            "statuses": statuses,
        },
    ).mappings().all()
    chunks = [
        ChunkPlan(
            chunk_id=int(row["chunk_id"]),
            range_start=int(row["range_start"]),
            range_end=int(row["range_end"]),
            planned_rows=int(row["planned_rows"]),
            source_table=source_table,
        )
        for row in rows
    ]
    if max_chunks is not None:
        return chunks[: max(0, int(max_chunks))]
    return chunks


def _start_chunk_attempt(
    connection,
    *,
    run_id: str,
    chunk_plan: ChunkPlan,
    chunk_state_table: str,
) -> int:
    """将 chunk 状态切换为 running，并返回当前 attempt。"""
    qualified_chunk_state = quote_table_name(chunk_state_table)
    row = connection.execute(
        text(
            f"""
            SELECT attempt
            FROM {qualified_chunk_state}
            WHERE run_id = :run_id
              AND source_table = :source_table
              AND chunk_id = :chunk_id
            """
        ),
        {
            "run_id": run_id,
            "source_table": chunk_plan.source_table,
            "chunk_id": chunk_plan.chunk_id,
        },
    ).mappings().one()
    attempt = int(row["attempt"] or 0) + 1
    connection.execute(
        text(
            f"""
            UPDATE {qualified_chunk_state}
            SET attempt = :attempt,
                status = 'running',
                started_at = now(),
                finished_at = NULL,
                duration_seconds = NULL,
                written_rows = 0,
                error_message = '',
                updated_at = now()
            WHERE run_id = :run_id
              AND source_table = :source_table
              AND chunk_id = :chunk_id
            """
        ),
        {
            "attempt": attempt,
            "run_id": run_id,
            "source_table": chunk_plan.source_table,
            "chunk_id": chunk_plan.chunk_id,
        },
    )
    return attempt


def _finish_chunk_attempt(
    connection,
    *,
    chunk_result: ChunkResult,
    chunk_state_table: str,
) -> None:
    """写回 chunk 执行结果。"""
    qualified_chunk_state = quote_table_name(chunk_state_table)
    connection.execute(
        text(
            f"""
            UPDATE {qualified_chunk_state}
            SET attempt = :attempt,
                status = :status,
                finished_at = now(),
                duration_seconds = :duration_seconds,
                written_rows = :written_rows,
                error_message = :error_message,
                updated_at = now()
            WHERE run_id = :run_id
              AND source_table = :source_table
              AND chunk_id = :chunk_id
            """
        ),
        {
            "attempt": chunk_result.attempt,
            "status": chunk_result.status,
            "duration_seconds": float(chunk_result.duration_seconds),
            "written_rows": int(chunk_result.written_rows),
            "error_message": chunk_result.error_message,
            "run_id": chunk_result.run_id,
            "source_table": chunk_result.source_table,
            "chunk_id": chunk_result.chunk_id,
        },
    )


def _execute_backfill_chunk(
    connection,
    *,
    run_id: str,
    chunk_plan: ChunkPlan,
    normalized_table: str,
    locator_table: str,
    only_missing: bool,
) -> int:
    """执行单个 chunk 的 SQL 回填。"""
    source_platform = infer_source_platform(chunk_plan.source_table)
    inserted_rows = connection.execute(
        text(
            build_chunk_insert_sql(
                locator_table=locator_table,
                normalized_table=normalized_table,
                source_table=chunk_plan.source_table,
                only_missing=only_missing,
            )
        ),
        {
            "run_id": run_id,
            "source_table": chunk_plan.source_table,
            "source_platform": source_platform,
            "chunk_id": int(chunk_plan.chunk_id),
        },
    ).scalar_one()
    return int(inserted_rows or 0)


def _count_locator_backfill_rows(
    connection,
    *,
    run_id: str,
    source_table: str,
    locator_table: str,
    normalized_table: str,
) -> dict[str, int]:
    """基于 locator 范围统计当前 run 的回填覆盖情况。"""
    qualified_locator = quote_table_name(locator_table)
    qualified_normalized = quote_table_name(normalized_table)
    row = connection.execute(
        text(
            f"""
            SELECT
                count(*) AS sample_rows,
                sum(CASE WHEN n.recruitment_record_id IS NULL THEN 1 ELSE 0 END) AS missing_in_normalized,
                sum(CASE WHEN n.recruitment_record_id IS NOT NULL THEN 1 ELSE 0 END) AS present_in_normalized
            FROM {qualified_locator} loc
            LEFT JOIN {qualified_normalized} n
              ON n.source_table = loc.source_table
             AND n.source_row_number = loc.source_row_number
            WHERE loc.run_id = :run_id
              AND loc.source_table = :source_table
            """
        ),
        {
            "run_id": run_id,
            "source_table": source_table,
        },
    ).mappings().one()
    return {key: int(value or 0) for key, value in row.items()}


def _log_chunk_result(chunk_result: ChunkResult) -> None:
    """输出单个 chunk 的结构化日志。"""
    logger.info(
        "chunk_result=%s",
        json.dumps(
            {
                **asdict(chunk_result),
                "rows_per_second": round(chunk_result.rows_per_second, 4),
            },
            ensure_ascii=False,
        ),
    )


def _write_benchmark_payload(path: str, payload: dict[str, object]) -> None:
    """将压测结果写入 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_chunk_with_retries(
    *,
    engine,
    run_id: str,
    chunk_plan: ChunkPlan,
    normalized_table: str,
    locator_table: str,
    chunk_state_table: str,
    only_missing: bool,
    max_retries: int,
    benchmark: bool,
) -> ChunkResult:
    """执行单个 chunk，并在失败时自动重试。"""
    final_result: ChunkResult | None = None
    for _ in range(max(0, int(max_retries)) + 1):
        with engine.begin() as connection:
            attempt = _start_chunk_attempt(
                connection,
                run_id=run_id,
                chunk_plan=chunk_plan,
                chunk_state_table=chunk_state_table,
            )
        started_at = perf_counter()
        try:
            with engine.begin() as connection:
                written_rows = _execute_backfill_chunk(
                    connection,
                    run_id=run_id,
                    chunk_plan=chunk_plan,
                    normalized_table=normalized_table,
                    locator_table=locator_table,
                    only_missing=only_missing,
                )
            final_result = ChunkResult(
                run_id=run_id,
                source_table=chunk_plan.source_table,
                chunk_id=chunk_plan.chunk_id,
                range_start=chunk_plan.range_start,
                range_end=chunk_plan.range_end,
                planned_rows=chunk_plan.planned_rows,
                written_rows=written_rows,
                duration_seconds=perf_counter() - started_at,
                attempt=attempt,
                status="succeeded",
            )
            with engine.begin() as connection:
                _finish_chunk_attempt(
                    connection,
                    chunk_result=final_result,
                    chunk_state_table=chunk_state_table,
                )
            if benchmark:
                _log_chunk_result(final_result)
            return final_result
        except Exception as exc:  # pragma: no cover - 异常路径依赖真实数据库
            final_result = ChunkResult(
                run_id=run_id,
                source_table=chunk_plan.source_table,
                chunk_id=chunk_plan.chunk_id,
                range_start=chunk_plan.range_start,
                range_end=chunk_plan.range_end,
                planned_rows=chunk_plan.planned_rows,
                written_rows=0,
                duration_seconds=perf_counter() - started_at,
                attempt=attempt,
                status="failed",
                error_message=str(exc),
            )
            with engine.begin() as connection:
                _finish_chunk_attempt(
                    connection,
                    chunk_result=final_result,
                    chunk_state_table=chunk_state_table,
                )
            logger.warning(
                "chunk_failed source_table=%s chunk_id=%s attempt=%s error=%s",
                chunk_plan.source_table,
                chunk_plan.chunk_id,
                attempt,
                exc,
            )
            if attempt > int(max_retries):
                if benchmark:
                    _log_chunk_result(final_result)
                return final_result
    assert final_result is not None
    return final_result


def backfill_recruitment_jobs_normalized(
    source_tables: list[str] | None = None,
    normalized_table: str = DEFAULT_NORMALIZED_TABLE,
    dry_run: bool = False,
    only_missing: bool = True,
    limit_rows: int | None = None,
    batch_size: int = 5000,
    use_sql_bulk: bool = True,
    workers: int = DEFAULT_WORKERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int | None = None,
    benchmark: bool = False,
    benchmark_json: str | None = None,
    resume_run_id: str | None = None,
    retry_failed_chunks: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    db_pool_size: int | None = None,
    db_max_overflow: int | None = None,
    db_pool_recycle: int = DEFAULT_POOL_RECYCLE_SECONDS,
    locator_table: str = DEFAULT_BACKFILL_LOCATOR_TABLE,
    chunk_state_table: str = DEFAULT_BACKFILL_CHUNK_STATE_TABLE,
) -> dict[str, object]:
    """将三家平台招聘表补充写入统一规范层。"""
    source_tables = [canonicalize_source_table_name(name) for name in list(source_tables or DEFAULT_SOURCE_TABLES)]
    summary: dict[str, object] = {}
    resolved_batch_size = max(1, int(batch_size))

    if dry_run:
        for source_table in source_tables:
            summary[source_table] = _count_missing_source_rows(
                source_table=source_table,
                normalized_table=normalized_table,
            )
        return summary

    bootstrap_engine = create_pg_engine()
    try:
        with bootstrap_engine.begin() as connection:
            ensure_recruitment_jobs_normalized_table(connection, table_name=normalized_table)
            ensure_recruitment_jobs_normalized_backfill_runtime_tables(
                connection,
                locator_table=locator_table,
                chunk_state_table=chunk_state_table,
            )
    finally:
        bootstrap_engine.dispose()

    if use_sql_bulk:
        run_id = str(resume_run_id or uuid4())
        resolved_workers = max(1, int(workers))
        resolved_chunk_size = max(1, int(chunk_size))
        pool_size = int(db_pool_size) if db_pool_size is not None else max(resolved_workers, 1)
        max_overflow = int(db_max_overflow) if db_max_overflow is not None else max(1, resolved_workers)
        locator_limit_rows = _effective_locator_limit_rows(limit_rows, max_chunks, resolved_chunk_size)
        chunk_results: list[ChunkResult] = []
        planned_chunks: list[ChunkPlan] = []
        run_started_at = perf_counter()
        engine = create_pg_engine(
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=int(db_pool_recycle),
            application_name="backfill_recruitment_jobs_normalized",
        )
        try:
            with engine.begin() as connection:
                for source_table in source_tables:
                    if not resume_run_id or not _locator_exists(
                        connection,
                        run_id=run_id,
                        source_table=source_table,
                        locator_table=locator_table,
                    ):
                        planned_rows = _prepare_source_locator(
                            connection,
                            run_id=run_id,
                            source_table=source_table,
                            chunk_size=resolved_chunk_size,
                            locator_table=locator_table,
                            chunk_state_table=chunk_state_table,
                            limit_rows=locator_limit_rows,
                        )
                        logger.info(
                            "已准备 locator %s: run_id=%s planned_rows=%s chunk_size=%s",
                            source_table,
                            run_id,
                            planned_rows,
                            resolved_chunk_size,
                        )
                    planned_chunks.extend(
                        _load_chunk_plans(
                            connection,
                            run_id=run_id,
                            source_table=source_table,
                            chunk_state_table=chunk_state_table,
                            max_chunks=max_chunks,
                            include_failed=bool(retry_failed_chunks),
                        )
                    )
            if planned_chunks:
                with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
                    futures = [
                        executor.submit(
                            _run_chunk_with_retries,
                            engine=engine,
                            run_id=run_id,
                            chunk_plan=chunk_plan,
                            normalized_table=normalized_table,
                            locator_table=locator_table,
                            chunk_state_table=chunk_state_table,
                            only_missing=only_missing,
                            max_retries=max_retries,
                            benchmark=benchmark,
                        )
                        for chunk_plan in planned_chunks
                    ]
                    for future in as_completed(futures):
                        chunk_results.append(future.result())
                chunk_results.sort(key=lambda item: (item.source_table, item.chunk_id))
            with engine.begin() as connection:
                for source_table in source_tables:
                    counts = _count_locator_backfill_rows(
                        connection,
                        run_id=run_id,
                        source_table=source_table,
                        locator_table=locator_table,
                        normalized_table=normalized_table,
                    )
                    table_results = [result for result in chunk_results if result.source_table == source_table]
                    counts["written_rows"] = int(sum(result.written_rows for result in table_results))
                    counts["chunk_count"] = len(table_results)
                    counts["failed_chunks"] = int(sum(1 for result in table_results if result.status != "succeeded"))
                    summary[source_table] = counts
                connection.execute(text(f"ANALYZE {quote_table_name(normalized_table)}"))
        finally:
            engine.dispose()
        run_summary = build_run_summary(
            chunk_results=chunk_results,
            total_duration_seconds=perf_counter() - run_started_at,
        )
        summary["__meta__"] = {
            "run_id": run_id,
            "workers": resolved_workers,
            "chunk_size": resolved_chunk_size,
            "benchmark": bool(benchmark),
            "retry_failed_chunks": bool(retry_failed_chunks),
            "max_retries": int(max_retries),
            "db_pool_size": pool_size,
            "db_max_overflow": max_overflow,
            "run_summary": run_summary,
        }
        if benchmark_json:
            _write_benchmark_payload(
                benchmark_json,
                {
                    "meta": summary["__meta__"],
                    "tables": {
                        source_table: summary[source_table]
                        for source_table in source_tables
                    },
                    "chunks": [
                        {
                            **asdict(chunk_result),
                            "rows_per_second": chunk_result.rows_per_second,
                        }
                        for chunk_result in chunk_results
                    ],
                },
            )
        return summary

    for source_table in source_tables:
        total_written_count = 0
        offset = 0
        remaining_limit = int(limit_rows) if limit_rows is not None else None
        while True:
            current_limit = resolved_batch_size
            if remaining_limit is not None:
                if remaining_limit <= 0:
                    break
                current_limit = min(current_limit, remaining_limit)
            source_df = _load_source_dataframe(
                source_table=source_table,
                normalized_table=normalized_table,
                only_missing=only_missing,
                offset=offset,
                limit_rows=current_limit,
            )
            if source_df.empty:
                break
            rows = build_normalized_rows_from_dataframe(
                source_df,
                source_table=source_table,
            )
            written_count = upsert_recruitment_jobs_normalized(
                rows,
                table_name=normalized_table,
            )
            total_written_count += int(written_count)
            logger.info(
                "已写入 %s: batch_rows=%s, total_written=%s",
                source_table,
                len(source_df),
                total_written_count,
            )
            offset += len(source_df)
            if remaining_limit is not None:
                remaining_limit -= len(source_df)
            if len(source_df) < current_limit:
                break
        counts = _count_missing_source_rows(
            source_table=source_table,
            normalized_table=normalized_table,
        )
        counts["written_rows"] = int(total_written_count)
        summary[source_table] = counts

    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="补齐三家招聘平台表到 recruitment_jobs_normalized")
    parser.add_argument(
        "--source-table",
        nargs="*",
        default=None,
        help="可选：覆盖默认 source 表列表",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="使用三家 cleaned_data 总体表作为 source；默认使用三家 sample 表",
    )
    parser.add_argument(
        "--normalized-table",
        default=DEFAULT_NORMALIZED_TABLE,
        help="目标 recruitment_jobs_normalized 表名",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出缺口统计，不写数据库",
    )
    parser.add_argument(
        "--all-source-rows",
        action="store_true",
        help="默认仅补 normalized 缺口；传入后改为重扫 source 全表并执行 upsert",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="每张 source 表最多读取多少行；用于试跑或分批调试",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Python DataFrame 回退路径每批读取并写入的 source 行数",
    )
    parser.add_argument(
        "--python-batches",
        action="store_true",
        help="使用旧的 Python DataFrame 分批写入路径；默认使用 PostgreSQL 端批量 INSERT ... SELECT",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="SQL chunk 并发 worker 数；默认 2，当前压测推荐值",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="每个 SQL chunk 覆盖的 source_row_number 行数；默认 25000，当前压测推荐值",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="仅执行前 N 个 chunk；适合压测",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="输出逐 chunk 压测日志",
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="可选：将压测汇总和 chunk 明细写入 JSON 文件",
    )
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help="复用指定 run_id 的 locator 和 chunk 状态继续执行",
    )
    parser.add_argument(
        "--retry-failed-chunks",
        action="store_true",
        help="resume 时额外重试 failed 状态的 chunk",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="单个 chunk 失败后的最大重试次数",
    )
    parser.add_argument(
        "--db-pool-size",
        type=int,
        default=None,
        help="SQLAlchemy 连接池 pool_size；默认跟 workers 对齐",
    )
    parser.add_argument(
        "--db-max-overflow",
        type=int,
        default=None,
        help="SQLAlchemy 连接池 max_overflow；默认跟 workers 对齐",
    )
    parser.add_argument(
        "--repair-source-table-values",
        action="store_true",
        help="先修复历史错误的 source_table 值，再执行回填",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    if bool(args.repair_source_table_values):
        repair_summary = repair_source_table_values(normalized_table=args.normalized_table)
        logger.info("repair summary: %s", json.dumps(repair_summary, ensure_ascii=False))
    source_tables = args.source_table
    if source_tables is None and bool(args.full):
        source_tables = DEFAULT_FULL_SOURCE_TABLES
    summary = backfill_recruitment_jobs_normalized(
        source_tables=source_tables,
        normalized_table=args.normalized_table,
        dry_run=bool(args.dry_run),
        only_missing=not bool(args.all_source_rows),
        limit_rows=args.limit_rows,
        batch_size=int(args.batch_size),
        use_sql_bulk=not bool(args.python_batches),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        max_chunks=args.max_chunks,
        benchmark=bool(args.benchmark),
        benchmark_json=args.benchmark_json,
        resume_run_id=args.resume_run_id,
        retry_failed_chunks=bool(args.retry_failed_chunks),
        max_retries=int(args.max_retries),
        db_pool_size=args.db_pool_size,
        db_max_overflow=args.db_max_overflow,
    )
    logger.info("backfill summary: %s", json.dumps(summary, ensure_ascii=False))
    meta = summary.get("__meta__", {}) if isinstance(summary, dict) else {}
    run_summary = meta.get("run_summary", {}) if isinstance(meta, dict) else {}
    if int(run_summary.get("failed_chunks", 0) or 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
