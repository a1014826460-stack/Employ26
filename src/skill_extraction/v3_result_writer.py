"""V3 技能抽取结果的 PostgreSQL 输出模块。

提供表创建（含索引）和批量 upsert 写入功能。
目标表: ``public.skill_extraction_v3_results``。

用法::

    from src.skill_extraction.v3_result_writer import (
        create_v3_results_table,
        write_v3_results,
    )

    create_v3_results_table()
    write_v3_results(results)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ─── SQL 定义 ───────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS public.skill_extraction_v3_results (
    id SERIAL PRIMARY KEY,
    recruitment_record_id TEXT NOT NULL,
    source_table TEXT,
    source_row_number INTEGER,
    job_title TEXT,
    hard_skills JSONB,
    hard_skill_count INTEGER,
    soft_skills JSONB,
    soft_skill_count INTEGER,
    pipeline_version TEXT DEFAULT 'v3',
    extracted_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(recruitment_record_id)
);
"""

_CREATE_INDEX_RID_SQL = """\
CREATE INDEX IF NOT EXISTS idx_v3_results_rid
ON public.skill_extraction_v3_results(recruitment_record_id);
"""

_CREATE_INDEX_HARD_SKILLS_SQL = """\
CREATE INDEX IF NOT EXISTS idx_v3_results_hard_skills
ON public.skill_extraction_v3_results USING GIN(hard_skills);
"""

_CREATE_INDEX_SOFT_SKILLS_SQL = """\
CREATE INDEX IF NOT EXISTS idx_v3_results_soft_skills
ON public.skill_extraction_v3_results USING GIN(soft_skills);
"""

_UPSERT_SQL = """\
INSERT INTO public.skill_extraction_v3_results (
    recruitment_record_id,
    source_table,
    source_row_number,
    job_title,
    hard_skills,
    hard_skill_count,
    soft_skills,
    soft_skill_count,
    pipeline_version,
    extracted_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (recruitment_record_id) DO UPDATE SET
    source_table = EXCLUDED.source_table,
    source_row_number = EXCLUDED.source_row_number,
    job_title = EXCLUDED.job_title,
    hard_skills = EXCLUDED.hard_skills,
    hard_skill_count = EXCLUDED.hard_skill_count,
    soft_skills = EXCLUDED.soft_skills,
    soft_skill_count = EXCLUDED.soft_skill_count,
    pipeline_version = EXCLUDED.pipeline_version,
    extracted_at = EXCLUDED.extracted_at;
"""


# ─── 连接辅助 ───────────────────────────────────────────────────────────────


def _get_connection(pg_params: Optional[Dict[str, Any]] = None):
    """获取 psycopg2 数据库连接。

    参数:
        pg_params: PostgreSQL 连接参数字典，为 None 时从 config.paths 获取。

    返回:
        psycopg2.connection: 数据库连接对象。
    """
    if pg_params is None:
        from config.paths import get_project_paths

        paths = get_project_paths()
        pg_params = paths.pg_connection_params

    import psycopg2

    return psycopg2.connect(**pg_params)


# ─── 公开接口 ───────────────────────────────────────────────────────────────


def create_v3_results_table(
    pg_params: Optional[Dict[str, Any]] = None,
) -> None:
    """创建 ``public.skill_extraction_v3_results`` 表及索引。

    使用 ``CREATE TABLE IF NOT EXISTS`` 和 ``CREATE INDEX IF NOT EXISTS``，
    可安全重复调用。

    参数:
        pg_params: PostgreSQL 连接参数字典，为 None 时从 config.paths 获取。
    """
    conn = _get_connection(pg_params)
    try:
        with conn.cursor() as cur:
            logger.info("创建表 public.skill_extraction_v3_results ...")
            cur.execute(_CREATE_TABLE_SQL)
            logger.info("创建索引 idx_v3_results_rid ...")
            cur.execute(_CREATE_INDEX_RID_SQL)
            logger.info("创建索引 idx_v3_results_hard_skills (GIN) ...")
            cur.execute(_CREATE_INDEX_HARD_SKILLS_SQL)
            logger.info("创建索引 idx_v3_results_soft_skills (GIN) ...")
            cur.execute(_CREATE_INDEX_SOFT_SKILLS_SQL)
        conn.commit()
        logger.info("表和索引创建完成")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _prepare_row(result: Dict[str, Any]) -> tuple:
    """将单条结果字典转换为 upsert 行元组。

    参数:
        result: 单条抽取结果字典，通常来自 ``RecordResult.to_dict()``。

    返回:
        tuple: 对应 UPSERT_SQL 占位符顺序的值元组。
    """
    hard_skills = result.get("hard_skills", [])
    soft_skills = result.get("soft_skills", [])

    # JSONB 字段需要序列化为 JSON 字符串
    hard_skills_json = json.dumps(hard_skills, ensure_ascii=False)
    soft_skills_json = json.dumps(soft_skills, ensure_ascii=False)

    return (
        result.get("recruitment_record_id", ""),
        result.get("source_table"),
        result.get("source_row_number"),
        result.get("job_title", ""),
        hard_skills_json,
        result.get("hard_skill_count", len(hard_skills)),
        soft_skills_json,
        result.get("soft_skill_count", len(soft_skills)),
        result.get("pipeline_version", "v3"),
        result.get("extracted_at") or datetime.now(),
    )


def write_v3_results(
    results: Sequence[Dict[str, Any]],
    pg_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 500,
) -> int:
    """批量写入 V3 技能抽取结果，使用 ``recruitment_record_id`` 唯一键 upsert。

    参数:
        results: 结果字典序列，每项通常来自 ``RecordResult.to_dict()``。
        pg_params: PostgreSQL 连接参数字典，为 None 时从 config.paths 获取。
        batch_size: 每次提交的行数，默认 500。

    返回:
        int: 成功写入的记录数。
    """
    if not results:
        logger.info("无结果需要写入")
        return 0

    conn = _get_connection(pg_params)
    written = 0
    try:
        with conn.cursor() as cur:
            for idx, result in enumerate(results, 1):
                row = _prepare_row(result)
                cur.execute(_UPSERT_SQL, row)
                written += 1

                if idx % batch_size == 0:
                    conn.commit()
                    logger.info("已写入 %d / %d 条", idx, len(results))

            conn.commit()
        logger.info("写入完成，共 %d 条", written)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return written
