from src.data_pipeline.backfill_recruitment_jobs_normalized import (
    ChunkResult,
    REPAIRABLE_SOURCE_TABLE_MAPPINGS,
    build_chunk_insert_sql,
    build_parser,
    build_run_summary,
    canonicalize_source_table_name,
    plan_chunks,
)
from src.db.postgres import build_pg_engine_options


def test_build_pg_engine_options_include_pooling_defaults():
    options = build_pg_engine_options(
        pool_size=4,
        max_overflow=2,
        pool_recycle=1800,
        pool_pre_ping=True,
        application_name="backfill-test",
    )

    assert options["pool_size"] == 4
    assert options["max_overflow"] == 2
    assert options["pool_recycle"] == 1800
    assert options["pool_pre_ping"] is True
    assert options["connect_args"]["application_name"] == "backfill-test"


def test_plan_chunks_splits_ranges_deterministically():
    chunks = plan_chunks(total_rows=5, chunk_size=2, limit_rows=None)

    assert [(chunk.range_start, chunk.range_end) for chunk in chunks] == [(1, 2), (3, 4), (5, 5)]


def test_build_run_summary_computes_rows_per_second():
    summary = build_run_summary(
        chunk_results=[
            ChunkResult("run-1", '"51job".sample', 1, 1, 2, 2, 2, 2.0, 1, "succeeded", ""),
            ChunkResult("run-1", '"51job".sample', 2, 3, 4, 2, 2, 1.0, 1, "succeeded", ""),
        ]
    )

    assert summary["written_rows"] == 4
    assert summary["rows_per_second"] == 4 / 3


def test_build_chunk_insert_sql_uses_conflict_do_nothing_for_missing_only():
    sql = build_chunk_insert_sql(
        locator_table="public.recruitment_jobs_normalized_backfill_locator",
        normalized_table="public.recruitment_jobs_normalized",
        source_table='"51job".sample',
        only_missing=True,
    )

    assert "ON CONFLICT (source_table, source_row_number) DO NOTHING" in sql
    assert "RETURNING 1" in sql
    assert "SELECT count(*) AS inserted_rows" in sql


def test_build_chunk_insert_sql_uses_conditional_update_for_full_refresh():
    sql = build_chunk_insert_sql(
        locator_table="public.recruitment_jobs_normalized_backfill_locator",
        normalized_table="public.recruitment_jobs_normalized",
        source_table='"51job".sample',
        only_missing=False,
    )

    assert "IS DISTINCT FROM" in sql


def test_build_parser_exposes_concurrency_flags():
    parser = build_parser()
    args = parser.parse_args(["--workers", "4", "--chunk-size", "200000", "--benchmark"])

    assert args.workers == 4
    assert args.chunk_size == 200000
    assert args.benchmark is True


def test_build_parser_uses_benchmarked_default_concurrency_values():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.workers == 2
    assert args.chunk_size == 25000


def test_canonicalize_source_table_name_quotes_mixed_case_schema():
    assert canonicalize_source_table_name("Liepin.cleaned_data") == '"Liepin".cleaned_data'
    assert canonicalize_source_table_name('"51job".sample') == '"51job".sample'


def test_build_parser_exposes_repair_flag():
    parser = build_parser()
    args = parser.parse_args(["--repair-source-table-values"])

    assert args.repair_source_table_values is True


def test_repairable_source_table_mappings_cover_known_bad_cleaned_tables():
    assert REPAIRABLE_SOURCE_TABLE_MAPPINGS['"Liepin"."cleaned_data"'] == '"Liepin".cleaned_data'
    assert REPAIRABLE_SOURCE_TABLE_MAPPINGS['"51job"."cleaned_data"'] == '"51job".cleaned_data'
    assert REPAIRABLE_SOURCE_TABLE_MAPPINGS['"Zhilian"."cleaned_data"'] == '"Zhilian".cleaned_data'
