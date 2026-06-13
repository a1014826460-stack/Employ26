"""`public.job_description_parsed` 的建表与写入逻辑。"""

from __future__ import annotations

import json
import math
import re
from time import sleep
from collections.abc import Sequence

import pandas as pd
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

from src.data_pipeline.description_schema import (
    DEFAULT_PARSED_TABLE,
    DESCRIPTION_SECTIONS_JSON_COL,
    DUTIES_TEXT_COL,
    JOB_DESCRIPTION_CLEAN_COL,
    PARSER_VERSION,
    PG_COLUMN_ORDER,
    RAG_QUERY_SOURCE_COL,
    RAG_QUERY_TEXT_COL,
    REQUIREMENTS_TEXT_COL,
    SECTIONS_BRIEF_COL,
    UNCLASSIFIED_TEXT_COL,
)
from src.db.postgres import create_pg_engine, ensure_schema


def split_table_name(table_name: str) -> tuple[str, str]:
    """拆分 PostgreSQL 表名，支持 schema.table 与带双引号 schema。"""
    normalized = str(table_name).strip()
    if "." not in normalized:
        return "public", normalized.strip('"')
    schema, table = normalized.split(".", 1)
    return schema.strip().strip('"'), table.strip().strip('"')


def quote_table_name(table_name: str) -> str:
    """返回安全双引号包裹的 PostgreSQL 表名。"""
    schema_name, raw_table_name = split_table_name(table_name)
    return f'"{schema_name}"."{raw_table_name}"'


def ensure_job_description_parsed_table(
    connection,
    table_name: str = DEFAULT_PARSED_TABLE,
) -> None:
    """确保岗位描述解析结果表存在，并补齐推荐索引。"""
    schema_name, raw_table_name = split_table_name(table_name)
    ensure_schema(connection, schema_name)
    qualified_table = f'"{schema_name}"."{raw_table_name}"'
    object_prefix = re.sub(r"[^A-Za-z0-9_]+", "_", raw_table_name).strip("_") or "job_description_parsed"

    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                recruitment_record_id text PRIMARY KEY,
                source_platform text NOT NULL,
                source_table text NOT NULL,
                source_row_number bigint NOT NULL,
                job_title text,
                job_description_raw text,
                job_description_clean text,
                description_sections jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                requirements_text text,
                duties_text text,
                unclassified_text text,
                sections_brief text,
                rag_query_text text,
                rag_query_source text,
                parser_version text NOT NULL,
                parsed_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_source_platform
            ON {qualified_table} (source_platform)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_source_row
            ON {qualified_table} (source_table, source_row_number)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_sections_gin
            ON {qualified_table} USING gin (description_sections)
            """
        )
    )


def infer_source_platform(source_table: str) -> str:
    """根据来源表名推断招聘平台。"""
    lowered = str(source_table).lower()
    if "51job" in lowered or "qcwy" in lowered:
        return "51job"
    if "liepin" in lowered:
        return "Liepin"
    if "zhilian" in lowered:
        return "Zhilian"
    return ""


def _safe_int(value: object, fallback: int) -> int:
    """安全转换整数，失败时使用 fallback。"""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_text(value: object, fallback: str = "") -> str:
    """安全转换文本，避免 pandas 缺失值落库为字符串 nan。"""
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text_value = str(value)
    return text_value if text_value else fallback


def build_parsed_pg_rows(
    parsed_df: pd.DataFrame,
    source_table: str,
    source_platform: str | None = None,
    parser_version: str = PARSER_VERSION,
    title_col: str = "岗位名称",
    desc_col: str = "岗位描述",
) -> list[dict[str, object]]:
    """将 `parse_desc_df` 结果转换为英文列名的 PostgreSQL 写入行。"""
    required_columns = {
        title_col,
        desc_col,
        JOB_DESCRIPTION_CLEAN_COL,
        DESCRIPTION_SECTIONS_JSON_COL,
        REQUIREMENTS_TEXT_COL,
        DUTIES_TEXT_COL,
        UNCLASSIFIED_TEXT_COL,
        SECTIONS_BRIEF_COL,
        RAG_QUERY_TEXT_COL,
        RAG_QUERY_SOURCE_COL,
    }
    missing_columns = required_columns.difference(parsed_df.columns)
    if missing_columns:
        raise KeyError(f"缺少转换 PostgreSQL 解析结果所需列: {sorted(missing_columns)}")

    platform = source_platform or infer_source_platform(source_table)
    rows: list[dict[str, object]] = []
    for fallback_index, (_, row) in enumerate(parsed_df.iterrows(), start=1):
        recruitment_record_id = _safe_text(row.get("recruitment_record_id"))
        if not recruitment_record_id:
            raise KeyError("缺少 recruitment_record_id，无法写入岗位描述解析结果。")
        row_source_table = _safe_text(row.get("source_table"), source_table)
        row_source_platform = _safe_text(row.get("source_platform"), platform or infer_source_platform(row_source_table))
        source_row_number = _safe_int(
            row.get("source_row_number"),
            _safe_int(row.get("__source_row_number"), fallback_index),
        )
        sections_json = _safe_text(row.get(DESCRIPTION_SECTIONS_JSON_COL), "{}")
        rows.append(
            {
                "recruitment_record_id": recruitment_record_id,
                "source_platform": row_source_platform,
                "source_table": row_source_table,
                "source_row_number": source_row_number,
                "job_title": _safe_text(row.get(title_col)),
                "job_description_raw": _safe_text(row.get(desc_col)),
                "job_description_clean": _safe_text(row.get(JOB_DESCRIPTION_CLEAN_COL)),
                "description_sections": sections_json,
                "requirements_text": _safe_text(row.get(REQUIREMENTS_TEXT_COL)),
                "duties_text": _safe_text(row.get(DUTIES_TEXT_COL)),
                "unclassified_text": _safe_text(row.get(UNCLASSIFIED_TEXT_COL)),
                "sections_brief": _safe_text(row.get(SECTIONS_BRIEF_COL)),
                "rag_query_text": _safe_text(row.get(RAG_QUERY_TEXT_COL)),
                "rag_query_source": _safe_text(row.get(RAG_QUERY_SOURCE_COL)),
                "parser_version": parser_version,
            }
        )
    return rows


def write_parsed_rows_to_postgres(
    rows: Sequence[dict[str, object]],
    table_name: str = DEFAULT_PARSED_TABLE,
) -> int:
    """把岗位描述解析结果 upsert 到 PostgreSQL。"""
    if not rows:
        return 0

    schema_name, raw_table_name = split_table_name(table_name)
    qualified_table = f'"{schema_name}"."{raw_table_name}"'
    columns_sql = ", ".join(PG_COLUMN_ORDER)
    values_sql = ", ".join(
        "CAST(:description_sections AS jsonb)" if column == "description_sections" else f":{column}"
        for column in PG_COLUMN_ORDER
    )
    update_sql = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in PG_COLUMN_ORDER
        if column not in {"recruitment_record_id"}
    )

    prepared_rows = [
        {
            **row,
            "description_sections": json.dumps(
                json.loads(str(row["description_sections"])),
                ensure_ascii=False,
            ),
        }
        for row in rows
    ]
    statement = text(
        f"""
        INSERT INTO {qualified_table} ({columns_sql})
        VALUES ({values_sql})
        ON CONFLICT (recruitment_record_id)
        DO UPDATE SET
            {update_sql},
            parsed_at = now()
        """
    )

    engine = create_pg_engine()
    try:
        for attempt in range(1, 5):
            try:
                with engine.begin() as connection:
                    ensure_job_description_parsed_table(connection, table_name=table_name)
                    connection.execute(statement, prepared_rows)
                return len(rows)
            except OperationalError as exc:
                if "deadlock detected" not in str(exc).lower() or attempt >= 4:
                    raise
                sleep(2 * attempt)
    finally:
        engine.dispose()
    return len(rows)
