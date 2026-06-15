"""PostgreSQL helpers for full-dataset occupation detail match results."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.db.postgres import create_pg_engine, ensure_schema
from src.db.recruitment_jobs_normalized import quote_table_name, split_table_name


DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE = "public.occupation_detail_matches"


def _safe_text(value: object) -> str:
    """Convert nullable values to stable stripped text."""
    if value is None:
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def _safe_float(value: object) -> float:
    """Convert nullable score values to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ensure_occupation_detail_matches_table(
    connection,
    table_name: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
) -> None:
    """Ensure the canonical occupation detail result table exists."""
    schema_name, raw_table_name = split_table_name(table_name)
    ensure_schema(connection, schema_name)
    qualified_table = quote_table_name(table_name)
    object_prefix = raw_table_name.replace('"', "").replace(".", "_").strip("_")

    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                recruitment_record_id text PRIMARY KEY,
                source_platform text NOT NULL DEFAULT '',
                source_table text NOT NULL DEFAULT '',
                source_row_number bigint,
                job_title text NOT NULL DEFAULT '',
                query_text text NOT NULL DEFAULT '',
                query_source text NOT NULL DEFAULT '',
                occupation_code text NOT NULL DEFAULT '',
                occupation_title text NOT NULL DEFAULT '',
                "大类" text NOT NULL DEFAULT '',
                "中类" text NOT NULL DEFAULT '',
                "小类" text NOT NULL DEFAULT '',
                "细类" text NOT NULL DEFAULT '',
                top1_score double precision NOT NULL DEFAULT 0,
                is_matched boolean NOT NULL DEFAULT false,
                selected_candidate_rank integer,
                top_k integer NOT NULL DEFAULT 10,
                top10_candidates jsonb NOT NULL DEFAULT '[]'::jsonb,
                model_recipe text NOT NULL DEFAULT '',
                base_model text NOT NULL DEFAULT '',
                model_path text NOT NULL DEFAULT '',
                run_id text NOT NULL DEFAULT '',
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    for column_sql in (
        "source_platform text NOT NULL DEFAULT ''",
        "source_table text NOT NULL DEFAULT ''",
        "source_row_number bigint",
        "job_title text NOT NULL DEFAULT ''",
        "query_text text NOT NULL DEFAULT ''",
        "query_source text NOT NULL DEFAULT ''",
        "occupation_code text NOT NULL DEFAULT ''",
        "occupation_title text NOT NULL DEFAULT ''",
        '"大类" text NOT NULL DEFAULT \'\'',
        '"中类" text NOT NULL DEFAULT \'\'',
        '"小类" text NOT NULL DEFAULT \'\'',
        '"细类" text NOT NULL DEFAULT \'\'',
        "top1_score double precision NOT NULL DEFAULT 0",
        "is_matched boolean NOT NULL DEFAULT false",
        "selected_candidate_rank integer",
        "top_k integer NOT NULL DEFAULT 10",
        "top10_candidates jsonb NOT NULL DEFAULT '[]'::jsonb",
        "model_recipe text NOT NULL DEFAULT ''",
        "base_model text NOT NULL DEFAULT ''",
        "model_path text NOT NULL DEFAULT ''",
        "run_id text NOT NULL DEFAULT ''",
        "created_at timestamptz NOT NULL DEFAULT now()",
        "updated_at timestamptz NOT NULL DEFAULT now()",
    ):
        connection.execute(
            text(
                f"""
                ALTER TABLE {qualified_table}
                ADD COLUMN IF NOT EXISTS {column_sql}
                """
            )
        )

    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_occupation_code
            ON {qualified_table} (occupation_code)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_is_matched
            ON {qualified_table} (is_matched)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_updated_at
            ON {qualified_table} (updated_at)
            """
        )
    )


def build_top_candidates(
    matched_row: dict[str, Any] | pd.Series,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Build ordered Top-K candidate JSON payload from matcher wide columns."""
    row = dict(matched_row)
    candidates: list[dict[str, Any]] = []
    for rank in range(1, int(top_k) + 1):
        prefix = f"top{rank}"
        candidates.append(
            {
                "rank": rank,
                "code": _safe_text(row.get(f"{prefix}_code", "")),
                "title": _safe_text(row.get(f"{prefix}_title", "")),
                "score": _safe_float(row.get(f"{prefix}_score", 0)),
                "detail_path": _safe_text(row.get(f"{prefix}_detail_path", "")),
                "detail_name": _safe_text(row.get(f"{prefix}_detail_name", "")),
            }
        )
    return candidates


def build_occupation_detail_match_records(
    *,
    source_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    run_id: str,
    model_recipe: str,
    base_model: str,
    model_path: str,
    top_k: int = 10,
    target_table: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
) -> list[dict[str, Any]]:
    """Convert source rows and matcher outputs into upsert-ready records."""
    if len(source_df) != len(matched_df):
        raise ValueError("source_df and matched_df must have the same row count")

    records: list[dict[str, Any]] = []
    for source_row, matched_row in zip(
        source_df.to_dict(orient="records"),
        matched_df.to_dict(orient="records"),
    ):
        top1_code = _safe_text(matched_row.get("top1_code", ""))
        top1_title = _safe_text(matched_row.get("top1_title", ""))
        selected_rank = matched_row.get("selected_candidate_rank", None)
        records.append(
            {
                "target_table": target_table,
                "recruitment_record_id": _safe_text(source_row.get("recruitment_record_id", "")),
                "source_platform": _safe_text(source_row.get("source_platform", "")),
                "source_table": _safe_text(source_row.get("source_table", "")),
                "source_row_number": source_row.get("source_row_number", None),
                "job_title": _safe_text(source_row.get("job_title", "")),
                "query_text": _safe_text(matched_row.get("query_text", "")),
                "query_source": _safe_text(
                    matched_row.get("query_source", "job_title+job_description_raw")
                ),
                "occupation_code": top1_code,
                "occupation_title": top1_title,
                "大类": _safe_text(matched_row.get("大类", "")),
                "中类": _safe_text(matched_row.get("中类", "")),
                "小类": _safe_text(matched_row.get("小类", "")),
                "细类": _safe_text(matched_row.get("细类", "")),
                "top1_score": _safe_float(matched_row.get("top1_score", 0)),
                "is_matched": bool(top1_code and top1_title),
                "selected_candidate_rank": int(selected_rank) if selected_rank else None,
                "top_k": int(top_k),
                "top10_candidates": json.dumps(
                    build_top_candidates(matched_row, top_k=top_k),
                    ensure_ascii=False,
                ),
                "model_recipe": _safe_text(model_recipe),
                "base_model": _safe_text(base_model),
                "model_path": _safe_text(model_path),
                "run_id": _safe_text(run_id),
            }
        )
    return records


def upsert_occupation_detail_match_records(
    records: list[dict[str, Any]],
    table_name: str = DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
) -> int:
    """Batch upsert occupation detail match records."""
    if not records:
        return 0

    qualified_table = quote_table_name(table_name)
    engine = create_pg_engine(application_name="occupation_detail_matches_upsert")
    try:
        with engine.begin() as connection:
            ensure_occupation_detail_matches_table(connection, table_name=table_name)
            connection.execute(
                text(
                    f"""
                    INSERT INTO {qualified_table} (
                        recruitment_record_id,
                        source_platform,
                        source_table,
                        source_row_number,
                        job_title,
                        query_text,
                        query_source,
                        occupation_code,
                        occupation_title,
                        "大类",
                        "中类",
                        "小类",
                        "细类",
                        top1_score,
                        is_matched,
                        selected_candidate_rank,
                        top_k,
                        top10_candidates,
                        model_recipe,
                        base_model,
                        model_path,
                        run_id
                    )
                    VALUES (
                        :recruitment_record_id,
                        :source_platform,
                        :source_table,
                        :source_row_number,
                        :job_title,
                        :query_text,
                        :query_source,
                        :occupation_code,
                        :occupation_title,
                        :大类,
                        :中类,
                        :小类,
                        :细类,
                        :top1_score,
                        :is_matched,
                        :selected_candidate_rank,
                        :top_k,
                        CAST(:top10_candidates AS jsonb),
                        :model_recipe,
                        :base_model,
                        :model_path,
                        :run_id
                    )
                    ON CONFLICT (recruitment_record_id)
                    DO UPDATE SET
                        source_platform = EXCLUDED.source_platform,
                        source_table = EXCLUDED.source_table,
                        source_row_number = EXCLUDED.source_row_number,
                        job_title = EXCLUDED.job_title,
                        query_text = EXCLUDED.query_text,
                        query_source = EXCLUDED.query_source,
                        occupation_code = EXCLUDED.occupation_code,
                        occupation_title = EXCLUDED.occupation_title,
                        "大类" = EXCLUDED."大类",
                        "中类" = EXCLUDED."中类",
                        "小类" = EXCLUDED."小类",
                        "细类" = EXCLUDED."细类",
                        top1_score = EXCLUDED.top1_score,
                        is_matched = EXCLUDED.is_matched,
                        selected_candidate_rank = EXCLUDED.selected_candidate_rank,
                        top_k = EXCLUDED.top_k,
                        top10_candidates = EXCLUDED.top10_candidates,
                        model_recipe = EXCLUDED.model_recipe,
                        base_model = EXCLUDED.base_model,
                        model_path = EXCLUDED.model_path,
                        run_id = EXCLUDED.run_id,
                        updated_at = now()
                    """
                ),
                records,
            )
    finally:
        engine.dispose()
    return len(records)
