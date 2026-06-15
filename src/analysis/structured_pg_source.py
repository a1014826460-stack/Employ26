"""结构化统计的 PostgreSQL 主数据源。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.analysis.analysis_common import (
    build_mapped_source_table_expr,
    enrich_common_dimension_columns,
)
from src.db.postgres import create_pg_engine, get_table_columns
from src.db.recruitment_jobs_normalized import (
    DEFAULT_NORMALIZED_TABLE,
    quote_table_name,
    split_table_name,
)
from src.skill_extraction.config import load_skill_extraction_config


DEFAULT_MATCH_TABLE = "public.skill_extraction_requirement_matches"

STRUCTURED_SOURCE_COLUMNS = [
    "recruitment_record_id",
    "source_platform",
    "source_table",
    "source_row_number",
    "job_title",
    "work_city",
    "company_name",
    "publish_date",
    "publish_month",
    "salary_raw",
    "education_requirement_raw",
    "experience_requirement_raw",
    "company_size_raw",
    "company_industry_raw",
    "city_normalized",
    "industry_normalized",
    "company_size_normalized",
    "occupation_code",
    "occupation_title",
    "occupation_core",
    "occupation_category",
    "occupation_major_category",
    "occupation_middle_category",
    "occupation_minor_category",
    "occupation_detail_category",
    "occupation_confidence",
    "occupation_is_matched",
]


@dataclass(frozen=True)
class StructuredSourceConfig:
    """结构化统计 PostgreSQL 源配置。"""

    normalized_table: str = DEFAULT_NORMALIZED_TABLE
    occupation_match_table: str = DEFAULT_MATCH_TABLE


def load_default_structured_source_config() -> StructuredSourceConfig:
    """从项目配置读取结构化统计主输入表。"""
    skill_config = load_skill_extraction_config()
    return StructuredSourceConfig(
        normalized_table=skill_config.recruitment_normalized_table,
        occupation_match_table=getattr(
            skill_config,
            "occupation_detail_match_table",
            skill_config.requirement_match_table,
        ),
    )


def build_structured_source_query(
    config: StructuredSourceConfig,
    *,
    match_columns: set[str] | None = None,
) -> str:
    """构建结构化统计主输入查询。"""
    normalized_table = quote_table_name(config.normalized_table)
    match_table = quote_table_name(config.occupation_match_table)
    available_match_columns = match_columns or set()
    has_record_id = "recruitment_record_id" in available_match_columns
    has_source_locator = {"__source_table", "__source_row_number"}.issubset(available_match_columns)
    if not has_record_id and not has_source_locator:
        raise ValueError(
            "职业匹配结果表缺少可用关联键：需要 recruitment_record_id "
            "或 __source_table + __source_row_number"
        )

    match_record_id_select = "m.recruitment_record_id" if has_record_id else "NULL::text AS recruitment_record_id"
    match_source_table_expr = build_mapped_source_table_expr('m."__source_table"')
    match_source_table_select = (
        f"{match_source_table_expr} AS match_source_table"
        if has_source_locator
        else "NULL::text AS match_source_table"
    )
    match_source_row_number_select = (
        'm."__source_row_number" AS match_source_row_number'
        if has_source_locator
        else "NULL::bigint AS match_source_row_number"
    )
    partition_expr = (
        "m.recruitment_record_id"
        if has_record_id
        else 'm."__source_table", m."__source_row_number"'
    )
    where_expr = (
        "COALESCE(m.recruitment_record_id, '') <> ''"
        if has_record_id
        else 'COALESCE(m."__source_table", \'\') <> \'\' AND m."__source_row_number" IS NOT NULL'
    )
    join_expr = (
        "m.recruitment_record_id = n.recruitment_record_id"
        if has_record_id
        else "m.match_source_table = n.source_table AND m.match_source_row_number = n.source_row_number"
    )
    return f"""
        WITH latest_occupation_match AS (
            SELECT *
            FROM (
                SELECT
                    {match_record_id_select},
                    {match_source_table_select},
                    {match_source_row_number_select},
                    m.occupation_code,
                    m.occupation_title,
                    m."大类" AS occupation_major_category,
                    m."中类" AS occupation_middle_category,
                    m."小类" AS occupation_minor_category,
                    m."细类" AS occupation_detail_category,
                    m.is_matched AS occupation_is_matched,
                    COALESCE(m.top1_score, 0) AS occupation_confidence,
                    row_number() OVER (
                        PARTITION BY {partition_expr}
                        ORDER BY
                            CASE WHEN COALESCE(m.is_matched, false) THEN 0 ELSE 1 END,
                            COALESCE(m.top1_score, 0) DESC,
                            COALESCE(m.occupation_code, '') ASC
                    ) AS rn
                FROM {match_table} m
                WHERE {where_expr}
            ) ranked
            WHERE rn = 1
        )
        SELECT
            n.recruitment_record_id,
            n.source_platform,
            n.source_table,
            n.source_row_number,
            n.job_title,
            n.work_city,
            n.company_name,
            n.publish_date,
            n.salary_raw,
            n.education_requirement_raw,
            n.experience_requirement_raw,
            n.company_size_raw,
            n.company_industry_raw,
            m.occupation_code,
            m.occupation_title,
            m.occupation_major_category,
            m.occupation_middle_category,
            m.occupation_minor_category,
            m.occupation_detail_category,
            m.occupation_confidence,
            m.occupation_is_matched
        FROM {normalized_table} n
        LEFT JOIN latest_occupation_match m
          ON {join_expr}
    """


def load_structured_analysis_dataframe(
    config: StructuredSourceConfig | None = None,
) -> pd.DataFrame:
    """读取结构化统计主输入，并补齐分析兼容字段。"""
    resolved_config = config or load_default_structured_source_config()
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            match_schema, match_table = split_table_name(resolved_config.occupation_match_table)
            match_columns = set(get_table_columns(connection, match_schema, match_table))
            source_df = pd.read_sql_query(
                text(build_structured_source_query(resolved_config, match_columns=match_columns)),
                connection,
            )
    finally:
        engine.dispose()

    return normalize_structured_source_dataframe(source_df)


def build_structured_source_coverage(
    config: StructuredSourceConfig | None = None,
) -> dict[str, float | int | str]:
    """统计结构化主输入覆盖率。"""
    resolved_config = config or load_default_structured_source_config()
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            match_schema, match_table = split_table_name(resolved_config.occupation_match_table)
            match_columns = set(get_table_columns(connection, match_schema, match_table))
            has_record_id = "recruitment_record_id" in match_columns
            has_source_locator = {"__source_table", "__source_row_number"}.issubset(match_columns)
            if not has_record_id and not has_source_locator:
                raise ValueError(
                    "职业匹配结果表缺少可用关联键：需要 recruitment_record_id "
                    "或 __source_table + __source_row_number"
                )

            normalized_table = quote_table_name(resolved_config.normalized_table)
            match_table_name = quote_table_name(resolved_config.occupation_match_table)
            if has_record_id:
                match_keys_cte = f"""
                    match_keys AS (
                        SELECT DISTINCT recruitment_record_id
                        FROM {match_table_name}
                        WHERE COALESCE(recruitment_record_id, '') <> ''
                    )
                """
                coverage_query = f"""
                    WITH {match_keys_cte}
                    SELECT
                        count(*) AS normalized_rows,
                        count(*) FILTER (WHERE mk.recruitment_record_id IS NOT NULL) AS matched_rows,
                        count(*) FILTER (WHERE COALESCE(n.salary_raw, '') <> '') AS salary_nonempty_rows,
                        count(*) FILTER (WHERE COALESCE(n.education_requirement_raw, '') <> '') AS education_nonempty_rows,
                        count(*) FILTER (WHERE COALESCE(n.publish_date, '') <> '') AS publish_date_nonempty_rows
                    FROM {normalized_table} n
                    LEFT JOIN match_keys mk
                      ON mk.recruitment_record_id = n.recruitment_record_id
                """
            else:
                coverage_query = f"""
                    WITH match_keys AS (
                        SELECT DISTINCT
                            {build_mapped_source_table_expr('"__source_table"')} AS source_table,
                            "__source_row_number" AS source_row_number
                        FROM {match_table_name}
                        WHERE COALESCE("__source_table", '') <> ''
                          AND "__source_row_number" IS NOT NULL
                    )
                    SELECT
                        count(*) AS normalized_rows,
                        count(*) FILTER (WHERE mk.source_table IS NOT NULL) AS matched_rows,
                        count(*) FILTER (WHERE COALESCE(n.salary_raw, '') <> '') AS salary_nonempty_rows,
                        count(*) FILTER (WHERE COALESCE(n.education_requirement_raw, '') <> '') AS education_nonempty_rows,
                        count(*) FILTER (WHERE COALESCE(n.publish_date, '') <> '') AS publish_date_nonempty_rows
                    FROM {normalized_table} n
                    LEFT JOIN match_keys mk
                      ON mk.source_table = n.source_table
                     AND mk.source_row_number = n.source_row_number
                """
            row = connection.execute(text(coverage_query)).mappings().one()
    finally:
        engine.dispose()

    normalized_rows = int(row["normalized_rows"] or 0)
    matched_rows = int(row["matched_rows"] or 0)
    salary_nonempty_rows = int(row["salary_nonempty_rows"] or 0)
    education_nonempty_rows = int(row["education_nonempty_rows"] or 0)
    publish_date_nonempty_rows = int(row["publish_date_nonempty_rows"] or 0)
    return {
        "normalized_table": resolved_config.normalized_table,
        "occupation_match_table": resolved_config.occupation_match_table,
        "normalized_rows": normalized_rows,
        "matched_rows": matched_rows,
        "matched_share": round(matched_rows / normalized_rows, 6) if normalized_rows else 0.0,
        "salary_nonempty_rows": salary_nonempty_rows,
        "salary_nonempty_share": round(salary_nonempty_rows / normalized_rows, 6) if normalized_rows else 0.0,
        "education_nonempty_rows": education_nonempty_rows,
        "education_nonempty_share": round(education_nonempty_rows / normalized_rows, 6) if normalized_rows else 0.0,
        "publish_date_nonempty_rows": publish_date_nonempty_rows,
        "publish_date_nonempty_share": round(publish_date_nonempty_rows / normalized_rows, 6) if normalized_rows else 0.0,
        "match_join_key": "recruitment_record_id" if has_record_id else "__source_table+__source_row_number(mapped)",
    }


def write_structured_source_coverage(
    output_dir: Path,
    config: StructuredSourceConfig | None = None,
) -> Path:
    """写入结构化主输入覆盖率摘要。"""
    coverage = build_structured_source_coverage(config=config)
    output_path = output_dir / "input_coverage_summary.json"
    output_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
def normalize_structured_source_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    """补齐结构化统计需要的规范列与历史兼容列。"""
    df = enrich_common_dimension_columns(source_df)
    for column in STRUCTURED_SOURCE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    occupation_title = df["occupation_title"].fillna("").astype(str).str.strip()
    detail_category = df["occupation_detail_category"].fillna("").astype(str).str.strip()
    middle_category = df["occupation_middle_category"].fillna("").astype(str).str.strip()
    major_category = df["occupation_major_category"].fillna("").astype(str).str.strip()

    df["occupation_core"] = occupation_title.where(occupation_title != "", detail_category)
    df["occupation_category"] = middle_category.where(middle_category != "", major_category)
    df["occupation_core"] = df["occupation_core"].replace("", pd.NA)
    df["occupation_category"] = df["occupation_category"].replace("", pd.NA)

    # 兼容现有统计脚本的旧字段名；字段来源仍是 PostgreSQL 规范层。
    df["岗位名称"] = df["job_title"]
    df["工作城市"] = df["work_city"]
    df["公司名称"] = df["company_name"]
    df["发布时间"] = df["publish_date"]
    df["薪资水平"] = df["salary_raw"]
    df["学历要求"] = df["education_requirement_raw"]
    df["经验要求"] = df["experience_requirement_raw"]
    df["公司规模"] = df["company_size_raw"]
    df["公司行业"] = df["company_industry_raw"]
    df["city_clean"] = df["city_normalized"]
    df["industry_clean"] = df["industry_normalized"]
    df["occupation_confidence"] = df["occupation_confidence"]
    return df
