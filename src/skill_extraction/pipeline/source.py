"""PostgreSQL 技能抽取数据源。"""

from __future__ import annotations

from typing import Any

from config.paths import get_project_paths


def fetch_latest_job_description_records(
    source_table: str | None = None,
    *,
    pg_params: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """读取每条来源记录的最新岗位描述解析结果。

    Args:
        source_table: PostgreSQL 源表；为空时读取配置中的
            `processing_results.job_description_parsed`。
        pg_params: PostgreSQL 连接参数；为空时使用集中配置。
        limit: 可选读取上限，主要用于 smoke test。

    Returns:
        list[dict[str, Any]]: 岗位描述解析记录。
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
