"""第二阶段 requirement text 约束抽取与聚合分析。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.analysis.analysis_common import (
    build_mapped_source_table_expr,
    build_requirement_output_dir,
    enrich_common_dimension_columns,
    write_run_manifest,
)
from src.analysis.requirement_constraint_extraction import (
    DEFAULT_EXTRACTOR_VERSION,
    convert_constraints_to_fact_rows,
    extract_requirement_constraints,
    normalize_item_text,
    parse_requirement_rules,
    split_requirement_items,
)
from src.db.analysis_lexicon import (
    build_lexicon_summary_frames,
    load_current_lexicon_resources,
)
from src.db.postgres import create_pg_engine, get_table_columns
from src.db.recruitment_jobs_normalized import (
    DEFAULT_NORMALIZED_TABLE,
    quote_table_name,
    split_table_name,
)
from src.db.requirement_constraint_facts import (
    load_requirement_constraint_facts_dataframe,
    replace_requirement_constraint_facts,
)


DEFAULT_TOP_N = 20
DEFAULT_GROUP_SIZE = 50
DEFAULT_MONTHLY_GROUP_SIZE = 20
DEFAULT_PARSED_TABLE = "public.job_description_parsed"

DIMENSION_COLUMNS = [
    "dimension_name",
    "constraint_type",
    "record_count",
    "record_share",
    "fact_count",
]
VALUE_COLUMNS = [
    "dimension_name",
    "normalized_value",
    "operator",
    "unit",
    "constraint_type",
    "record_count",
    "record_share",
]
BREAKDOWN_COLUMNS = [
    "group_name",
    "group_value",
    "dimension_name",
    "normalized_value",
    "record_count",
    "record_share",
]
NOISE_COLUMNS = [
    "noise_type",
    "noise_text",
    "rule_source",
    "record_count",
    "item_count",
    "record_share",
]
STRINGENCY_COLUMNS = [
    "recruitment_record_id",
    "source_platform",
    "job_title",
    "city_normalized",
    "industry_normalized",
    "company_size_normalized",
    "constraint_fact_count",
    "dimension_count",
    "stringency_score",
    "has_experience_gate",
    "has_education_gate",
    "has_demographic_gate",
    "has_certificate_gate",
    "has_work_condition_gate",
    "noise_only_item_count",
]
DIAGNOSTIC_COLUMNS = [
    "total_normalized_records",
    "parsed_records_available",
    "requirements_nonempty_records",
    "duties_fallback_records",
    "publish_date_parseable_records",
    "records_with_reliable_itemization",
    "records_with_constraints",
    "constraint_fact_rows",
    "template_noise_records",
]
LEXICON_SUMMARY_COLUMNS = [
    "resource_name",
    "dimension_a",
    "dimension_b",
    "enabled",
    "row_count",
]
LEGACY_OUTPUT_FILES = (
    "overall_frequency.csv",
    "term_type_frequency.csv",
    "category_frequency.csv",
    "dimension_breakdown_frequency.csv",
    "dedup_vs_record_weight.csv",
    "noise_terms_filtered.csv",
)

@dataclass
class AnalysisParams:
    """Phase 2 统计参数。"""

    top_n: int = DEFAULT_TOP_N
    min_group_size: int = DEFAULT_GROUP_SIZE
    min_monthly_group_size: int = DEFAULT_MONTHLY_GROUP_SIZE
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION


def build_current_output_dir(run_date: datetime | None = None) -> Path:
    """构建 req_analysis 输出目录。"""
    return build_requirement_output_dir(run_date=run_date)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _sort_if_not_empty(df: pd.DataFrame, by: list[str], ascending: list[bool]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(by, ascending=ascending)


def _reshape_lexicon_summary_frames(summary_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把不同资源摘要展平成统一 CSV。"""
    rows: list[dict[str, Any]] = []
    for resource_name, frame in summary_frames.items():
        if frame.empty:
            continue
        if resource_name == "user_dictionary":
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "resource_name": resource_name,
                        "dimension_a": str(row.get("term_type", "")),
                        "dimension_b": str(row.get("category", "")),
                        "enabled": bool(row.get("enabled", False)),
                        "row_count": int(row.get("row_count", 0)),
                    }
                )
        elif resource_name == "stopwords":
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "resource_name": resource_name,
                        "dimension_a": str(row.get("scope", "")),
                        "dimension_b": str(row.get("stop_strength", "")),
                        "enabled": bool(row.get("enabled", False)),
                        "row_count": int(row.get("row_count", 0)),
                    }
                )
        elif resource_name == "phrase_rules":
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "resource_name": resource_name,
                        "dimension_a": str(row.get("rule_type", "")),
                        "dimension_b": str(row.get("source", "")),
                        "enabled": bool(row.get("enabled", False)),
                        "row_count": int(row.get("row_count", 0)),
                    }
                )
        elif resource_name == "requirement_rules":
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "resource_name": resource_name,
                        "dimension_a": str(row.get("rule_type", "")),
                        "dimension_b": str(row.get("dimension_name", "")),
                        "enabled": bool(row.get("enabled", False)),
                        "row_count": int(row.get("row_count", 0)),
                    }
                )
    if not rows:
        return _empty_frame(LEXICON_SUMMARY_COLUMNS)
    return _sort_if_not_empty(
        pd.DataFrame(rows, columns=LEXICON_SUMMARY_COLUMNS),
        ["resource_name", "dimension_a", "dimension_b", "enabled"],
        [True, True, True, True],
    )


def _cleanup_legacy_outputs(output_dir: Path) -> None:
    """删除同目录下的 Phase 1 旧产物，避免混淆。"""
    for filename in LEGACY_OUTPUT_FILES:
        target = output_dir / filename
        if target.exists():
            target.unlink()


def build_requirement_analysis_query(
    *,
    normalized_table: str = DEFAULT_NORMALIZED_TABLE,
    parsed_table: str = DEFAULT_PARSED_TABLE,
    parsed_columns: set[str] | None = None,
) -> str:
    """构建 requirement text 分析主输入查询。"""
    available_columns = parsed_columns or set()
    has_record_id = "recruitment_record_id" in available_columns
    has_source_locator = {"source_table", "source_row_number"}.issubset(available_columns)
    has_legacy_source_locator = {"__source_table", "__source_row_number"}.issubset(available_columns)

    if not has_record_id and not has_source_locator and not has_legacy_source_locator:
        raise ValueError(
            "岗位描述解析表缺少可用关联键：需要 recruitment_record_id，"
            "或 source_table + source_row_number，或 __source_table + __source_row_number"
        )

    normalized_table_name = quote_table_name(normalized_table)
    parsed_table_name = quote_table_name(parsed_table)
    parsed_record_id_select = "p.recruitment_record_id" if has_record_id else "NULL::text AS recruitment_record_id"

    if has_record_id:
        parsed_source_table_select = "NULL::text AS parsed_source_table"
        parsed_source_row_number_select = "NULL::bigint AS parsed_source_row_number"
        partition_expr = "p.recruitment_record_id"
        where_expr = "COALESCE(p.recruitment_record_id, '') <> ''"
        join_expr = "p.recruitment_record_id = n.recruitment_record_id"
    elif has_source_locator:
        parsed_source_table_select = (
            f"{build_mapped_source_table_expr('p.source_table')} AS parsed_source_table"
        )
        parsed_source_row_number_select = "p.source_row_number AS parsed_source_row_number"
        partition_expr = "p.source_table, p.source_row_number"
        where_expr = "COALESCE(p.source_table, '') <> '' AND p.source_row_number IS NOT NULL"
        join_expr = "p.parsed_source_table = n.source_table AND p.parsed_source_row_number = n.source_row_number"
    else:
        parsed_source_table_expr = build_mapped_source_table_expr('p."__source_table"')
        parsed_source_table_select = f"{parsed_source_table_expr} AS parsed_source_table"
        parsed_source_row_number_select = 'p."__source_row_number" AS parsed_source_row_number'
        partition_expr = 'p."__source_table", p."__source_row_number"'
        where_expr = 'COALESCE(p."__source_table", \'\') <> \'\' AND p."__source_row_number" IS NOT NULL'
        join_expr = "p.parsed_source_table = n.source_table AND p.parsed_source_row_number = n.source_row_number"

    return f"""
        WITH latest_parsed AS (
            SELECT *
            FROM (
                SELECT
                    {parsed_record_id_select},
                    {parsed_source_table_select},
                    {parsed_source_row_number_select},
                    p.requirements_text,
                    p.duties_text,
                    p.sections_brief,
                    p.parser_version,
                    p.parsed_at,
                    row_number() OVER (
                        PARTITION BY {partition_expr}
                        ORDER BY
                            p.parsed_at DESC NULLS LAST,
                            COALESCE(
                                NULLIF(regexp_replace(COALESCE(p.parser_version, ''), '\\D', '', 'g'), ''),
                                '0'
                            )::integer DESC,
                            COALESCE(p.requirements_text, '') DESC
                    ) AS rn
                FROM {parsed_table_name} p
                WHERE {where_expr}
            ) ranked
            WHERE rn = 1
        )
        SELECT
            n.recruitment_record_id,
            n.source_platform,
            n.source_table,
            n.source_row_number,
            n.source_native_job_id,
            n.job_title,
            n.work_city,
            n.company_name,
            n.publish_date,
            n.salary_raw,
            n.education_requirement_raw,
            n.experience_requirement_raw,
            n.company_size_raw,
            n.company_industry_raw,
            p.requirements_text,
            p.duties_text,
            p.sections_brief,
            p.parser_version,
            p.parsed_at
        FROM {normalized_table_name} n
        LEFT JOIN latest_parsed p
          ON {join_expr}
    """


def load_requirement_analysis_dataframe() -> pd.DataFrame:
    """读取 requirement text 分析主输入。"""
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            parsed_schema, parsed_table = split_table_name(DEFAULT_PARSED_TABLE)
            parsed_columns = set(get_table_columns(connection, parsed_schema, parsed_table))
            return pd.read_sql_query(
                text(build_requirement_analysis_query(parsed_columns=parsed_columns)),
                connection,
            )
    finally:
        engine.dispose()


def _build_constraint_dimension_frequency(
    facts_df: pd.DataFrame,
    requirement_record_count: int,
) -> pd.DataFrame:
    if facts_df.empty:
        return _empty_frame(DIMENSION_COLUMNS)
    grouped = (
        facts_df.groupby(["dimension_name", "constraint_type"], dropna=False)
        .agg(
            record_count=("recruitment_record_id", "nunique"),
            fact_count=("fact_id", "count"),
        )
        .reset_index()
    )
    grouped["record_share"] = grouped["record_count"].map(
        lambda value: round(float(value) / float(max(1, requirement_record_count)), 6)
    )
    return _sort_if_not_empty(
        grouped[DIMENSION_COLUMNS],
        ["record_count", "fact_count", "dimension_name"],
        [False, False, True],
    )


def _build_constraint_value_distribution(
    facts_df: pd.DataFrame,
    requirement_record_count: int,
) -> pd.DataFrame:
    if facts_df.empty:
        return _empty_frame(VALUE_COLUMNS)
    grouped = (
        facts_df.groupby(
            ["dimension_name", "normalized_value", "operator", "unit", "constraint_type"],
            dropna=False,
        )
        .agg(record_count=("recruitment_record_id", "nunique"))
        .reset_index()
    )
    grouped["record_share"] = grouped["record_count"].map(
        lambda value: round(float(value) / float(max(1, requirement_record_count)), 6)
    )
    return _sort_if_not_empty(
        grouped[VALUE_COLUMNS],
        ["record_count", "dimension_name", "normalized_value"],
        [False, True, True],
    )


def _build_constraint_breakdowns(
    facts_df: pd.DataFrame,
    source_df: pd.DataFrame,
    params: AnalysisParams,
) -> pd.DataFrame:
    if facts_df.empty:
        return _empty_frame(BREAKDOWN_COLUMNS)
    meta_df = source_df[
        [
            "recruitment_record_id",
            "city_normalized",
            "industry_normalized",
            "company_size_normalized",
            "publish_month",
        ]
    ].copy()
    merged = facts_df.merge(meta_df, on="recruitment_record_id", how="left")
    rows: list[dict[str, Any]] = []
    group_configs = (
        ("city", "city_normalized", params.min_group_size),
        ("industry", "industry_normalized", params.min_group_size),
        ("company_size", "company_size_normalized", params.min_group_size),
        ("publish_month", "publish_month", params.min_monthly_group_size),
    )
    for group_name, column_name, min_group_size in group_configs:
        counts = (
            merged[column_name]
            .fillna("未知")
            .astype(str)
            .replace("", "未知")
            .value_counts()
        )
        eligible_values = {key for key, value in counts.items() if int(value) >= min_group_size}
        if not eligible_values:
            continue
        scoped = merged[merged[column_name].fillna("未知").astype(str).isin(eligible_values)].copy()
        grouped = (
            scoped.groupby([column_name, "dimension_name", "normalized_value"], dropna=False)
            .agg(record_count=("recruitment_record_id", "nunique"))
            .reset_index()
        )
        for _, row in grouped.iterrows():
            group_value = str(row[column_name] or "未知")
            denominator = int(counts.get(group_value, 0)) or 1
            rows.append(
                {
                    "group_name": group_name,
                    "group_value": group_value,
                    "dimension_name": str(row["dimension_name"]),
                    "normalized_value": str(row["normalized_value"]),
                    "record_count": int(row["record_count"]),
                    "record_share": round(float(row["record_count"]) / float(denominator), 6),
                }
            )
    if not rows:
        return _empty_frame(BREAKDOWN_COLUMNS)
    return _sort_if_not_empty(
        pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS),
        ["group_name", "group_value", "record_count", "dimension_name", "normalized_value"],
        [True, True, False, True, True],
    )


def _build_template_noise_report(
    noise_hits_df: pd.DataFrame,
    requirement_record_count: int,
) -> pd.DataFrame:
    if noise_hits_df.empty:
        return _empty_frame(NOISE_COLUMNS)
    grouped = (
        noise_hits_df.groupby(["noise_type", "noise_text", "rule_source"], dropna=False)
        .agg(
            record_count=("recruitment_record_id", "nunique"),
            item_count=("item_key", "nunique"),
        )
        .reset_index()
    )
    grouped["record_share"] = grouped["record_count"].map(
        lambda value: round(float(value) / float(max(1, requirement_record_count)), 6)
    )
    return _sort_if_not_empty(
        grouped[NOISE_COLUMNS],
        ["record_count", "item_count", "noise_type", "noise_text"],
        [False, False, True, True],
    )


def _build_requirement_stringency_index(
    facts_df: pd.DataFrame,
    noise_hits_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    base_df = source_df[
        [
            "recruitment_record_id",
            "source_platform",
            "job_title",
            "city_normalized",
            "industry_normalized",
            "company_size_normalized",
        ]
    ].drop_duplicates(subset=["recruitment_record_id"]).copy()

    fact_groups: dict[str, dict[str, Any]] = {}
    for recruitment_record_id, group in facts_df.groupby("recruitment_record_id") if not facts_df.empty else []:
        dimensions = {str(value) for value in group["dimension_name"].dropna().tolist()}
        score = 0
        if "experience" in dimensions:
            score += 2
        if "education" in dimensions:
            score += 1
        for dimension_name in ("age", "gender", "physical_condition", "certificate", "language"):
            if dimension_name in dimensions:
                score += 1
        for dimension_name in ("travel", "shift", "availability", "work_condition"):
            if dimension_name in dimensions:
                score += 1
        if len(dimensions) >= 3:
            score += 1
        fact_groups[str(recruitment_record_id)] = {
            "constraint_fact_count": int(len(group)),
            "dimension_count": int(len(dimensions)),
            "stringency_score": int(score),
            "has_experience_gate": "experience" in dimensions,
            "has_education_gate": "education" in dimensions,
            "has_demographic_gate": any(
                item in dimensions for item in ("age", "gender", "physical_condition")
            ),
            "has_certificate_gate": any(item in dimensions for item in ("certificate", "language")),
            "has_work_condition_gate": any(
                item in dimensions for item in ("travel", "shift", "availability", "work_condition")
            ),
        }

    noise_counts = (
        noise_hits_df.groupby("recruitment_record_id").agg(noise_only_item_count=("item_key", "nunique")).to_dict("index")
        if not noise_hits_df.empty
        else {}
    )

    rows: list[dict[str, Any]] = []
    for _, row in base_df.iterrows():
        recruitment_record_id = str(row["recruitment_record_id"])
        fact_meta = fact_groups.get(
            recruitment_record_id,
            {
                "constraint_fact_count": 0,
                "dimension_count": 0,
                "stringency_score": 0,
                "has_experience_gate": False,
                "has_education_gate": False,
                "has_demographic_gate": False,
                "has_certificate_gate": False,
                "has_work_condition_gate": False,
            },
        )
        noise_only_item_count = int(noise_counts.get(recruitment_record_id, {}).get("noise_only_item_count", 0))
        stringency_score = int(fact_meta["stringency_score"])
        if int(fact_meta["constraint_fact_count"]) == 0 and noise_only_item_count > 0:
            stringency_score = max(0, stringency_score - 1)
        rows.append(
            {
                "recruitment_record_id": recruitment_record_id,
                "source_platform": str(row["source_platform"]),
                "job_title": str(row["job_title"]),
                "city_normalized": str(row["city_normalized"]),
                "industry_normalized": str(row["industry_normalized"]),
                "company_size_normalized": str(row["company_size_normalized"]),
                "constraint_fact_count": int(fact_meta["constraint_fact_count"]),
                "dimension_count": int(fact_meta["dimension_count"]),
                "stringency_score": int(stringency_score),
                "has_experience_gate": bool(fact_meta["has_experience_gate"]),
                "has_education_gate": bool(fact_meta["has_education_gate"]),
                "has_demographic_gate": bool(fact_meta["has_demographic_gate"]),
                "has_certificate_gate": bool(fact_meta["has_certificate_gate"]),
                "has_work_condition_gate": bool(fact_meta["has_work_condition_gate"]),
                "noise_only_item_count": noise_only_item_count,
            }
        )
    return _sort_if_not_empty(
        pd.DataFrame(rows, columns=STRINGENCY_COLUMNS),
        ["stringency_score", "constraint_fact_count", "recruitment_record_id"],
        [False, False, True],
    )


def _build_report_text(
    *,
    manifest: dict[str, Any],
    diagnostics_df: pd.DataFrame,
    dimension_df: pd.DataFrame,
    value_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    stringency_df: pd.DataFrame,
    params: AnalysisParams,
) -> str:
    top_city_industry = breakdown_df[breakdown_df["group_name"].isin(["city", "industry", "company_size"])].head(params.top_n)
    avg_stringency = (
        stringency_df.groupby("city_normalized", dropna=False)
        .agg(avg_stringency_score=("stringency_score", "mean"))
        .reset_index()
        .sort_values(["avg_stringency_score", "city_normalized"], ascending=[False, True])
        .head(params.top_n)
        if not stringency_df.empty
        else pd.DataFrame()
    )
    lines = [
        "# Requirement Text Constraint Extraction and Analysis Report",
        "",
        "## 1. Run Summary",
        f"- extractor_version: {manifest['extractor_version']}",
        f"- lexicon_version: {manifest['lexicon_version']}",
        f"- total_normalized_records: {manifest['total_normalized_records']}",
        f"- requirements_nonempty_records: {manifest['requirements_nonempty_records']}",
        f"- records_with_constraints: {manifest['records_with_constraints']}",
        f"- constraint_fact_rows: {manifest['constraint_fact_rows']}",
        "",
        "## Key Notes",
        "- 本期正式结论聚焦 requirement 约束、模板噪声与招聘门槛强度，不包含 hard skill / soft skill 分类研究。",
        "- hard skill / soft skill 继续列为 TODO，后续单独做更细的词典治理与标注验证。",
        "",
        "## 2. Sample Coverage and Extraction Diagnostics",
        diagnostics_df.to_string(index=False),
        "",
        "## 3. Constraint Dimension Frequency",
        dimension_df.head(params.top_n).to_string(index=False) if not dimension_df.empty else "无结果",
        "",
        "## 4. Constraint Value Distribution",
        value_df.head(params.top_n).to_string(index=False) if not value_df.empty else "无结果",
        "",
        "## 5. City / Industry / Company Size Differences",
        top_city_industry.to_string(index=False) if not top_city_industry.empty else "无结果",
        "",
        "## 6. Template Noise Report",
        noise_df.head(params.top_n).to_string(index=False) if not noise_df.empty else "无结果",
        "",
        "## 7. Hiring Stringency Index",
        avg_stringency.to_string(index=False) if not avg_stringency.empty else "无结果",
        "",
        "## 8. Notes",
        "- PostgreSQL `public.requirement_constraint_facts` 已作为 Phase 2 正式中间层。",
        "- `duties_text` 本期继续只用于诊断，不并入正式约束事实。",
    ]
    return "\n".join(lines)


def analyze_requirement_texts(
    output_dir: Path,
    params: AnalysisParams,
) -> dict[str, Any]:
    """运行第二阶段 requirement text 约束抽取与分析。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_outputs(output_dir)

    resources = load_current_lexicon_resources()
    rules_by_type = parse_requirement_rules(resources["requirement_rules"])
    source_df = load_requirement_analysis_dataframe()
    if source_df.empty:
        source_df = pd.DataFrame(
            columns=[
                "recruitment_record_id",
                "source_platform",
                "source_table",
                "source_row_number",
                "job_title",
                "work_city",
                "company_name",
                "publish_date",
                "salary_raw",
                "education_requirement_raw",
                "experience_requirement_raw",
                "company_size_raw",
                "company_industry_raw",
                "requirements_text",
                "duties_text",
                "sections_brief",
                "parser_version",
                "parsed_at",
            ]
        )

    total_records = len(source_df)
    source_df = enrich_common_dimension_columns(source_df)

    requirements_nonempty_mask = source_df["requirements_text"].fillna("").astype(str).str.strip() != ""
    duties_fallback_mask = (~requirements_nonempty_mask) & (
        source_df["duties_text"].fillna("").astype(str).str.strip() != ""
    )
    publish_parseable_mask = source_df["publish_month"].astype(str).str.strip() != ""
    parsed_available_mask = source_df["parser_version"].fillna("").astype(str).str.strip() != ""

    analysis_df = source_df.loc[requirements_nonempty_mask].copy()
    fact_rows = []
    noise_rows: list[dict[str, Any]] = []
    reliable_itemization_records = 0
    records_with_constraints = 0

    for _, row in analysis_df.iterrows():
        extraction = extract_requirement_constraints(
            str(row.get("requirements_text", "")),
            rules_by_type=rules_by_type,
        )
        if extraction.reliable_itemization:
            reliable_itemization_records += 1
        fact_rows.extend(
            convert_constraints_to_fact_rows(
                recruitment_record_id=str(row["recruitment_record_id"]),
                source_table=str(row["source_table"]),
                source_row_number=int(row["source_row_number"]),
                constraints=extraction.constraints,
                extractor_version=params.extractor_version,
            )
        )
        if extraction.constraints:
            records_with_constraints += 1
        for hit in extraction.template_noise_hits:
            noise_rows.append(
                {
                    "recruitment_record_id": str(row["recruitment_record_id"]),
                    "item_key": f"{row['recruitment_record_id']}::{hit.item_index}::{hit.noise_text}",
                    "noise_type": hit.noise_type,
                    "noise_text": hit.noise_text,
                    "rule_source": hit.rule_source,
                    "item_index": hit.item_index,
                    "item_text_raw": hit.item_text_raw,
                    "item_text_normalized": hit.item_text_normalized,
                }
            )

    replace_requirement_constraint_facts(
        fact_rows,
        extractor_version=params.extractor_version,
    )
    facts_df = load_requirement_constraint_facts_dataframe(extractor_version=params.extractor_version)
    noise_hits_df = pd.DataFrame(noise_rows)

    dimension_df = _build_constraint_dimension_frequency(facts_df, len(analysis_df))
    value_df = _build_constraint_value_distribution(facts_df, len(analysis_df))
    breakdown_df = _build_constraint_breakdowns(facts_df, analysis_df, params)
    noise_df = _build_template_noise_report(noise_hits_df, len(analysis_df))
    stringency_df = _build_requirement_stringency_index(facts_df, noise_hits_df, analysis_df)

    diagnostics_df = pd.DataFrame(
        [
            {
                "total_normalized_records": int(total_records),
                "parsed_records_available": int(parsed_available_mask.sum()),
                "requirements_nonempty_records": int(requirements_nonempty_mask.sum()),
                "duties_fallback_records": int(duties_fallback_mask.sum()),
                "publish_date_parseable_records": int(publish_parseable_mask.sum()),
                "records_with_reliable_itemization": int(reliable_itemization_records),
                "records_with_constraints": int(records_with_constraints),
                "constraint_fact_rows": int(len(facts_df)),
                "template_noise_records": int(noise_hits_df["recruitment_record_id"].nunique()) if not noise_hits_df.empty else 0,
            }
        ],
        columns=DIAGNOSTIC_COLUMNS,
    )
    lexicon_summary_df = _reshape_lexicon_summary_frames(build_lexicon_summary_frames(resources))

    diagnostics_df.to_csv(output_dir / "coverage_diagnostics.csv", index=False, encoding="utf-8-sig")
    lexicon_summary_df.to_csv(output_dir / "lexicon_summary.csv", index=False, encoding="utf-8-sig")
    dimension_df.to_csv(output_dir / "constraint_dimension_frequency.csv", index=False, encoding="utf-8-sig")
    value_df.to_csv(output_dir / "constraint_value_distribution.csv", index=False, encoding="utf-8-sig")
    breakdown_df.to_csv(output_dir / "constraint_by_city_industry.csv", index=False, encoding="utf-8-sig")
    noise_df.to_csv(output_dir / "template_noise_report.csv", index=False, encoding="utf-8-sig")
    stringency_df.to_csv(output_dir / "requirement_stringency_index.csv", index=False, encoding="utf-8-sig")
    (output_dir / "report.md").write_text(
        _build_report_text(
            manifest={
                "extractor_version": params.extractor_version,
                "lexicon_version": resources["release"]["version"],
                "total_normalized_records": int(total_records),
                "requirements_nonempty_records": int(requirements_nonempty_mask.sum()),
                "records_with_constraints": int(records_with_constraints),
                "constraint_fact_rows": int(len(facts_df)),
            },
            diagnostics_df=diagnostics_df,
            dimension_df=dimension_df,
            value_df=value_df,
            breakdown_df=breakdown_df,
            noise_df=noise_df,
            stringency_df=stringency_df,
            params=params,
        ),
        encoding="utf-8",
    )

    manifest_steps = [
        "load_requirement_analysis_dataframe",
        "extract_requirement_constraints",
        "replace_requirement_constraint_facts",
        "aggregate_requirement_reports",
    ]
    manifest_params = {
        "top_n": int(params.top_n),
        "min_group_size": int(params.min_group_size),
        "min_monthly_group_size": int(params.min_monthly_group_size),
        "extractor_version": params.extractor_version,
        "normalized_table": DEFAULT_NORMALIZED_TABLE,
        "parsed_table": DEFAULT_PARSED_TABLE,
        "fact_table": "public.requirement_constraint_facts",
    }
    manifest_input_files = [
        f"postgres:{DEFAULT_NORMALIZED_TABLE}",
        f"postgres:{DEFAULT_PARSED_TABLE}",
        "postgres:analysis_lexicon.current_release",
    ]
    manifest_output_files = [
        "run_manifest.json",
        "coverage_diagnostics.csv",
        "lexicon_summary.csv",
        "constraint_dimension_frequency.csv",
        "constraint_value_distribution.csv",
        "constraint_by_city_industry.csv",
        "template_noise_report.csv",
        "requirement_stringency_index.csv",
        "report.md",
        "postgres:public.requirement_constraint_facts",
    ]
    manifest_extra_fields = {
        "extractor_version": params.extractor_version,
        "lexicon_version": resources["release"]["version"],
        "total_normalized_records": int(total_records),
        "parsed_records_available": int(parsed_available_mask.sum()),
        "requirements_nonempty_records": int(requirements_nonempty_mask.sum()),
        "duties_fallback_records": int(duties_fallback_mask.sum()),
        "records_with_reliable_itemization": int(reliable_itemization_records),
        "records_with_constraints": int(records_with_constraints),
        "constraint_fact_rows": int(len(facts_df)),
    }
    write_run_manifest(
        output_dir,
        workflow="requirement_text_analysis",
        steps=manifest_steps,
        params=manifest_params,
        input_files=manifest_input_files,
        output_files=manifest_output_files,
        extra_fields=manifest_extra_fields,
    )

    manifest = {
        "workflow": "requirement_text_analysis",
        "steps": manifest_steps,
        "params": manifest_params,
        "input_files": manifest_input_files,
        "output_files": manifest_output_files,
        **manifest_extra_fields,
    }

    return {
        "manifest": manifest,
        "output_dir": str(output_dir),
        "fact_rows": len(facts_df),
        "constraint_dimension_rows": len(dimension_df),
        "constraint_value_rows": len(value_df),
        "breakdown_rows": len(breakdown_df),
        "noise_rows": len(noise_df),
        "stringency_rows": len(stringency_df),
    }


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="第二阶段 requirement text 约束抽取与分析")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--min-monthly-group-size", type=int, default=DEFAULT_MONTHLY_GROUP_SIZE)
    parser.add_argument("--extractor-version", default=DEFAULT_EXTRACTOR_VERSION)
    parser.add_argument("--output-dir", default="", help="显式指定输出目录")
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else build_current_output_dir()
    analyze_requirement_texts(
        output_dir=output_dir,
        params=AnalysisParams(
            top_n=int(args.top_n),
            min_group_size=int(args.min_group_size),
            min_monthly_group_size=int(args.min_monthly_group_size),
            extractor_version=str(args.extractor_version).strip() or DEFAULT_EXTRACTOR_VERSION,
        ),
    )


if __name__ == "__main__":
    main()
