"""PostgreSQL 硬技能匹配入口。

本模块承接旧 DuckDB 匹配脚本中仍有价值的生产职责：从 PostgreSQL
读取岗位描述解析结果，运行平面硬技能词典匹配，并可将结果写入调试表。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from config.paths import get_project_paths

from .matcher import FlatHardSkillMatcher, load_flat_dictionary

logger = logging.getLogger(__name__)

STRONG_REVALIDATION_PRECISION_THRESHOLD: float = 0.95
STRONG_REVALIDATION_PARSE_SUCCESS_THRESHOLD: float = 0.90
STRONG_REVALIDATION_MIN_SAMPLES: int = 20


@dataclass(frozen=True)
class HardSkillMatchRecord:
    """单条 PostgreSQL 硬技能匹配结果。"""

    recruitment_record_id: str
    source_table: str | None
    source_row_number: int | None
    job_title: str
    skills: list[dict[str, Any]]
    matched_at: datetime

    def to_row(self) -> tuple[Any, ...]:
        """转换为 PostgreSQL 写入行。"""
        return (
            self.recruitment_record_id,
            self.source_table,
            self.source_row_number,
            self.job_title,
            json.dumps(self.skills, ensure_ascii=False),
            len(self.skills),
            self.matched_at,
        )


def should_trigger_strong_revalidation(summary: dict[str, Any]) -> bool:
    """判断是否需要触发强复核。

    当抽样精度异常高且解析成功率也很高时，可能意味着验证样本过于简单，
    需要提升复核强度来避免发布门禁失真。
    """
    estimated_precision = float(summary.get("estimated_precision", 0.0) or 0.0)
    parse_success_rate = float(summary.get("parse_success_rate", 0.0) or 0.0)
    total_samples = int(summary.get("total_samples", 0) or 0)
    if total_samples < STRONG_REVALIDATION_MIN_SAMPLES:
        return False
    return (
        estimated_precision >= STRONG_REVALIDATION_PRECISION_THRESHOLD
        and parse_success_rate >= STRONG_REVALIDATION_PARSE_SUCCESS_THRESHOLD
    )


def _safe_text(value: Any) -> str:
    """安全转换文本字段。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan"} else text


def _match_text_from_row(row: dict[str, Any]) -> str:
    """按统一优先级选择匹配文本。"""
    for key in ("requirements_text", "duties_text", "job_description_clean"):
        text = _safe_text(row.get(key))
        if text:
            return text
    return ""


def fetch_latest_parsed_records(
    source_table: str | None = None,
    *,
    limit: int | None = None,
    pg_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """从 PostgreSQL 读取最新岗位描述解析记录。

    Args:
        source_table: 源表名，为空时从配置读取 `processing_results.job_description_parsed`。
        limit: 可选读取上限。
        pg_params: PostgreSQL 连接参数；为空时使用集中配置。

    Returns:
        list[dict[str, Any]]: 可直接传给匹配器的记录。
    """
    import psycopg2

    paths = get_project_paths()
    table_name = source_table or paths.get_table_name(
        "processing_results",
        "job_description_parsed",
        "public.job_description_parsed",
    )
    params = pg_params or paths.pg_connection_params
    limit_sql = " LIMIT %s" if limit is not None else ""
    query = f"""
        SELECT DISTINCT ON (source_table, source_row_number)
            recruitment_record_id,
            source_table,
            source_row_number,
            job_title,
            requirements_text,
            duties_text,
            job_description_clean
        FROM {table_name}
        WHERE recruitment_record_id IS NOT NULL
        ORDER BY source_table, source_row_number, parsed_at DESC
        {limit_sql}
    """

    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            if limit is None:
                cur.execute(query)
            else:
                cur.execute(query, (limit,))
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def match_records(
    records: Sequence[dict[str, Any]],
    matcher: FlatHardSkillMatcher,
) -> list[HardSkillMatchRecord]:
    """对内存记录执行硬技能匹配。"""
    matched: list[HardSkillMatchRecord] = []
    now = datetime.now()
    for row in records:
        text = _match_text_from_row(row)
        skills = matcher.match_text(text) if text else []
        matched.append(
            HardSkillMatchRecord(
                recruitment_record_id=_safe_text(row.get("recruitment_record_id")),
                source_table=row.get("source_table"),
                source_row_number=row.get("source_row_number"),
                job_title=_safe_text(row.get("job_title")),
                skills=skills,
                matched_at=now,
            )
        )
    return matched


def write_debug_results(
    results: Iterable[HardSkillMatchRecord],
    *,
    output_table: str | None = None,
    pg_params: dict[str, Any] | None = None,
) -> int:
    """将硬技能匹配结果写入 PostgreSQL 调试表。"""
    import psycopg2

    paths = get_project_paths()
    table_name = output_table or paths.get_table_name(
        "processing_results",
        "hard_skill_match_results_dev",
        "public.hard_skill_match_results_dev",
    )
    params = pg_params or paths.pg_connection_params
    rows = [record.to_row() for record in results]
    if not rows:
        return 0

    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            recruitment_record_id TEXT PRIMARY KEY,
            source_table TEXT,
            source_row_number INTEGER,
            job_title TEXT,
            skills JSONB,
            skill_count INTEGER,
            matched_at TIMESTAMP
        )
    """
    upsert_sql = f"""
        INSERT INTO {table_name} (
            recruitment_record_id,
            source_table,
            source_row_number,
            job_title,
            skills,
            skill_count,
            matched_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (recruitment_record_id) DO UPDATE SET
            source_table = EXCLUDED.source_table,
            source_row_number = EXCLUDED.source_row_number,
            job_title = EXCLUDED.job_title,
            skills = EXCLUDED.skills,
            skill_count = EXCLUDED.skill_count,
            matched_at = EXCLUDED.matched_at
    """

    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            for row in rows:
                cur.execute(upsert_sql, row)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(rows)


def run_match_pg(
    *,
    dict_path: str | Path | None = None,
    source_table: str | None = None,
    output_table: str | None = None,
    limit: int | None = None,
    write: bool = True,
) -> list[HardSkillMatchRecord]:
    """运行 PostgreSQL 硬技能匹配流程。"""
    paths = get_project_paths()
    resolved_dict = dict_path or paths.dict_dir / "flat_skill_dictionary.json"
    matcher = FlatHardSkillMatcher(load_flat_dictionary(resolved_dict))
    records = fetch_latest_parsed_records(source_table, limit=limit)
    results = match_records(records, matcher)
    logger.info("硬技能匹配完成: %d 条记录", len(results))
    if write:
        written = write_debug_results(results, output_table=output_table)
        logger.info("硬技能调试结果已写入 PostgreSQL: %d 条", written)
    return results
