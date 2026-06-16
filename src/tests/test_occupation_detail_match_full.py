from pathlib import Path

import pandas as pd

from src.data_pipeline.occupation_detail_match_full import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MODEL_RECIPE,
    DEFAULT_TOP_K,
    _build_catalog_cache_path,
    build_match_input_dataframe,
    build_unmatched_batch_query,
    resolve_runtime_defaults,
)


def test_build_match_input_dataframe_uses_normalized_columns():
    jobs_df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "rid-1",
                "source_platform": "Liepin",
                "source_table": '"Liepin".raw_data',
                "source_row_number": 1,
                "job_title": "算法工程师",
                "job_description_raw": "负责推荐系统模型研发。",
            }
        ]
    )

    result = build_match_input_dataframe(jobs_df)

    assert result.loc[0, "岗位名称"] == "算法工程师"
    assert result.loc[0, "岗位描述"] == "负责推荐系统模型研发。"
    assert result.loc[0, "职业匹配来源"] == "job_title+job_description_raw"
    assert DEFAULT_TOP_K == 10
    assert DEFAULT_MODEL_RECIPE == "v1"
    assert DEFAULT_BASE_MODEL == "bge-large-zh-v1.5"


def test_build_unmatched_batch_query_skips_existing_rows_when_resume_enabled():
    query = build_unmatched_batch_query(
        normalized_table="public.recruitment_jobs_normalized",
        target_table="public.occupation_detail_matches",
        resume=True,
    )

    assert '"public"."recruitment_jobs_normalized"' in query
    assert '"public"."occupation_detail_matches"' in query
    assert "not exists" in query.lower()
    assert "limit :batch_size" in query.lower()


def test_build_unmatched_batch_query_can_use_record_id_cursor():
    query = build_unmatched_batch_query(
        normalized_table="public.recruitment_jobs_normalized",
        target_table="public.occupation_detail_matches",
        resume=True,
        last_seen_record_id="rid-1",
    )

    assert "n.recruitment_record_id > :last_seen_record_id" in query
    assert "not exists" not in query.lower()


class _FakeConfig:
    recruitment_normalized_table = "public.recruitment_jobs_normalized"
    occupation_detail_match_table = "public.occupation_detail_matches"
    occupation_detail_model_path = Path("output/penghui/rag_round2_training/bge-large-round2-finetuned")
    occupation_detail_top_k = 10


def test_resolve_runtime_defaults_uses_config_when_args_are_empty():
    defaults = resolve_runtime_defaults(_FakeConfig())

    assert defaults["normalized_table"] == "public.recruitment_jobs_normalized"
    assert defaults["target_table"] == "public.occupation_detail_matches"
    assert defaults["top_k"] == 10
    assert defaults["model_path"].endswith("bge-large-round2-finetuned")


def test_build_catalog_cache_path_versions_canonical_catalog_embeddings():
    cache_path = _build_catalog_cache_path(
        Path("output/cache"),
        "output/penghui/rag_round2_training/bge-large-round2-finetuned",
    )

    assert cache_path.name == (
        "occupation_catalog_embeddings_occupation-detail-v4-canonical-pro_"
        "bge-large-round2-finetuned.npy"
    )
