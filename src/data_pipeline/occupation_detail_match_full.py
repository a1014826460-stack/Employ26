"""Full-dataset occupation detail matching into PostgreSQL."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db.occupation_detail_matches import (
    DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
    build_occupation_detail_match_records,
    ensure_occupation_detail_matches_table,
    upsert_occupation_detail_match_records,
)
from src.db.postgres import create_pg_engine
from src.db.recruitment_jobs_normalized import (
    DEFAULT_NORMALIZED_TABLE,
    quote_table_name,
)
from src.skill_extraction.config import load_skill_extraction_config


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_TOP_K = 10
DEFAULT_MODEL_RECIPE = "v1"
DEFAULT_BASE_MODEL = "bge-large-zh-v1.5"
DEFAULT_MODEL_PATH = "output/penghui/rag_round2_training/bge-large-round2-finetuned"


def resolve_runtime_defaults(config) -> dict[str, object]:
    """Resolve runtime defaults from project config."""
    return {
        "normalized_table": config.recruitment_normalized_table,
        "target_table": config.occupation_detail_match_table,
        "model_path": str(config.occupation_detail_model_path),
        "top_k": int(config.occupation_detail_top_k),
    }


def build_match_input_dataframe(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized jobs into the existing matcher input contract."""
    work_df = jobs_df.copy()
    work_df["岗位名称"] = work_df["job_title"].fillna("").astype(str)
    work_df["岗位描述"] = work_df["job_description_raw"].fillna("").astype(str)
    work_df["职业匹配来源"] = "job_title+job_description_raw"
    return work_df


def build_unmatched_batch_query(
    *,
    normalized_table: str,
    target_table: str,
    resume: bool,
    last_seen_record_id: str | None = None,
) -> str:
    """Build stable batch query for normalized jobs."""
    normalized = quote_table_name(normalized_table)
    target = quote_table_name(target_table)
    resume_filter = (
        f"""
        AND NOT EXISTS (
            SELECT 1
            FROM {target} m
            WHERE m.recruitment_record_id = n.recruitment_record_id
        )
        """
        if resume and not last_seen_record_id
        else ""
    )
    cursor_filter = (
        "AND n.recruitment_record_id > :last_seen_record_id"
        if last_seen_record_id
        else ""
    )
    return f"""
        SELECT
            n.recruitment_record_id,
            n.source_platform,
            n.source_table,
            n.source_row_number,
            n.job_title,
            n.job_description_raw
        FROM {normalized} n
        WHERE COALESCE(n.recruitment_record_id, '') <> ''
          AND COALESCE(n.job_title, '') <> ''
          {cursor_filter}
          {resume_filter}
        ORDER BY n.recruitment_record_id
        LIMIT :batch_size
    """


def load_job_batch(
    *,
    connection,
    normalized_table: str,
    target_table: str,
    batch_size: int,
    resume: bool,
    last_seen_record_id: str | None = None,
) -> pd.DataFrame:
    """Load one candidate batch from PostgreSQL."""
    query = build_unmatched_batch_query(
        normalized_table=normalized_table,
        target_table=target_table,
        resume=resume,
        last_seen_record_id=last_seen_record_id,
    )
    params = {"batch_size": int(batch_size)}
    if last_seen_record_id:
        params["last_seen_record_id"] = str(last_seen_record_id)
    return pd.read_sql_query(text(query), connection, params=params)


def load_resume_cursor(
    *,
    connection,
    target_table: str,
) -> str | None:
    """Load current max processed record id from target table."""
    target = quote_table_name(target_table)
    value = connection.execute(
        text(
            f"""
            SELECT max(recruitment_record_id)
            FROM {target}
            WHERE COALESCE(recruitment_record_id, '') <> ''
            """
        )
    ).scalar_one_or_none()
    return str(value) if value else None


def _build_catalog_cache_path(cache_dir: Path, model_path: str | Path) -> Path:
    """Build a model-specific catalog embedding cache path."""
    model_name = Path(str(model_path)).name or "occupation_detail_model"
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in model_name)
    return cache_dir / f"occupation_catalog_embeddings_{safe_name}.npy"


def run_full_occupation_detail_matching(
    *,
    database_config_path: str | Path | None = None,
    normalized_table: str | None = None,
    target_table: str | None = None,
    model_path: str | None = None,
    model_recipe: str = DEFAULT_MODEL_RECIPE,
    base_model: str = DEFAULT_BASE_MODEL,
    top_k: int | None = None,
    batch_size: int = 20000,
    limit_rows: int | None = None,
    resume: bool = True,
    run_id: str | None = None,
) -> int:
    """Run resumable full-dataset occupation detail matching."""
    config = load_skill_extraction_config(database_config_path=database_config_path)
    defaults = resolve_runtime_defaults(config)
    normalized_table = normalized_table or str(defaults["normalized_table"])
    target_table = target_table or str(defaults["target_table"])
    model_path = model_path or str(defaults["model_path"])
    top_k = int(top_k or defaults["top_k"])

    matcher_config = replace(
        config,
        embedding_model_path=Path(model_path),
        catalog_embedding_cache_path=_build_catalog_cache_path(config.cache_dir, model_path),
    )

    from src.skill_extraction.bge_matcher import OccupationBGEMatcher

    run_id = run_id or datetime.now().strftime("occupation-detail-%Y%m%d-%H%M%S")
    logger.info(
        "启动全量职业细类识别: normalized=%s, target=%s, model=%s, top_k=%s, batch_size=%s, resume=%s",
        normalized_table,
        target_table,
        model_path,
        top_k,
        batch_size,
        resume,
    )

    matcher = OccupationBGEMatcher(matcher_config)
    matcher.build_index()

    total_written = 0
    engine = create_pg_engine(application_name="occupation_detail_match_full")
    try:
        with engine.begin() as connection:
            ensure_occupation_detail_matches_table(connection, table_name=target_table)
            last_seen_record_id = load_resume_cursor(
                connection=connection,
                target_table=target_table,
            ) if resume else None

        while True:
            if limit_rows is not None and total_written >= int(limit_rows):
                break

            effective_batch_size = int(batch_size)
            if limit_rows is not None:
                effective_batch_size = min(effective_batch_size, int(limit_rows) - total_written)
            if effective_batch_size <= 0:
                break

            with engine.connect() as connection:
                jobs_df = load_job_batch(
                    connection=connection,
                    normalized_table=normalized_table,
                    target_table=target_table,
                    batch_size=effective_batch_size,
                    resume=resume,
                    last_seen_record_id=last_seen_record_id,
                )

            if jobs_df.empty:
                logger.info("没有更多待匹配招聘记录，结束。")
                break

            match_input_df = build_match_input_dataframe(jobs_df)
            matched_df = matcher.match_jobs(match_input_df, top_k=top_k)
            records = build_occupation_detail_match_records(
                source_df=jobs_df,
                matched_df=matched_df,
                run_id=run_id,
                model_recipe=model_recipe,
                base_model=base_model,
                model_path=model_path,
                top_k=top_k,
                target_table=target_table,
            )
            written = upsert_occupation_detail_match_records(records, table_name=target_table)
            total_written += written
            last_seen_record_id = str(jobs_df["recruitment_record_id"].iloc[-1])
            logger.info("本批写入 %s 行，累计写入 %s 行", written, total_written)
    finally:
        engine.dispose()
    return total_written


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="全量职业细类识别: Top10 检索 + Top1 默认输出 + PostgreSQL 入库")
    parser.add_argument("--database-config", default=None, help="数据库配置文件路径，默认 config/database.yaml")
    parser.add_argument("--normalized-table", default=None, help="招聘统一规范层表")
    parser.add_argument("--target-table", default=None, help="职业细类识别结果表")
    parser.add_argument("--model-path", default=None, help="finetuned embedding 模型路径")
    parser.add_argument("--model-recipe", default=DEFAULT_MODEL_RECIPE, help="模型配方名")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="底座模型名")
    parser.add_argument("--top-k", type=int, default=None, help="底层检索候选数，默认读取配置")
    parser.add_argument("--batch-size", type=int, default=20000, help="每批招聘记录数")
    parser.add_argument("--limit-rows", type=int, default=None, help="调试/试跑时限制总处理行数")
    parser.add_argument("--run-id", default=None, help="本次运行 ID，默认自动生成")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已有 recruitment_record_id，强制重算覆盖")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    run_full_occupation_detail_matching(
        database_config_path=args.database_config,
        normalized_table=args.normalized_table,
        target_table=args.target_table,
        model_path=args.model_path,
        model_recipe=args.model_recipe,
        base_model=args.base_model,
        top_k=args.top_k,
        batch_size=args.batch_size,
        limit_rows=args.limit_rows,
        resume=not args.no_resume,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
