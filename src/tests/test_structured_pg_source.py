import pandas as pd

from src.analysis.structured_pg_source import (
    StructuredSourceConfig,
    build_structured_source_coverage,
    build_structured_source_query,
    load_default_structured_source_config,
    normalize_structured_source_dataframe,
)


def test_normalize_structured_source_dataframe_adds_analysis_columns():
    source_df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "r1",
                "source_platform": "51job",
                "source_table": '"51job".sample',
                "source_row_number": 1,
                "job_title": "算法工程师",
                "work_city": "广东省深圳市",
                "company_name": "A公司",
                "publish_date": "2026-06-01",
                "salary_raw": "20-30K",
                "education_requirement_raw": "本科",
                "experience_requirement_raw": "3-5年",
                "company_size_raw": "100-499人",
                "company_industry_raw": "互联网/人工智能",
                "occupation_code": "2-02-10-09",
                "occupation_title": "人工智能工程技术人员",
                "occupation_major_category": "专业技术人员",
                "occupation_middle_category": "工程技术人员",
                "occupation_minor_category": "信息和通信工程技术人员",
                "occupation_detail_category": "人工智能工程技术人员",
                "occupation_confidence": 0.91,
                "occupation_is_matched": True,
            }
        ]
    )

    result_df = normalize_structured_source_dataframe(source_df)
    row = result_df.iloc[0]

    assert row["publish_month"] == "2026-06"
    assert row["city_normalized"] == "深圳"
    assert row["industry_normalized"] == "互联网"
    assert row["occupation_core"] == "人工智能工程技术人员"
    assert row["occupation_category"] == "工程技术人员"
    assert row["薪资水平"] == "20-30K"
    assert row["学历要求"] == "本科"
    assert row["city_clean"] == "深圳"
    assert row["industry_clean"] == "互联网"


def test_build_structured_source_query_falls_back_to_source_locator_join():
    query = build_structured_source_query(
        StructuredSourceConfig(),
        match_columns={
            "__source_table",
            "__source_row_number",
            "occupation_code",
            "occupation_title",
        },
    )

    assert "m.recruitment_record_id" not in query
    assert "m.match_source_table = n.source_table" in query
    assert "m.match_source_row_number = n.source_row_number" in query


def test_build_structured_source_coverage_uses_source_locator_mapping(monkeypatch):
    class FakeResult:
        def mappings(self):
            return self

        def one(self):
            return {
                "normalized_rows": 100,
                "matched_rows": 80,
                "salary_nonempty_rows": 90,
                "education_nonempty_rows": 70,
                "publish_date_nonempty_rows": 95,
            }

    class FakeConnection:
        def execute(self, _query):
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def connect(self):
            return FakeConnection()

        def dispose(self):
            return None

    monkeypatch.setattr(
        "src.analysis.structured_pg_source.create_pg_engine",
        lambda: FakeEngine(),
    )
    monkeypatch.setattr(
        "src.analysis.structured_pg_source.get_table_columns",
        lambda connection, schema, table: ["__source_table", "__source_row_number"],
    )

    coverage = build_structured_source_coverage(StructuredSourceConfig())

    assert coverage["normalized_rows"] == 100
    assert coverage["matched_rows"] == 80
    assert coverage["matched_share"] == 0.8
    assert coverage["match_join_key"] == "__source_table+__source_row_number(mapped)"


class _FakeSkillConfig:
    recruitment_normalized_table = "public.recruitment_jobs_normalized"
    requirement_match_table = "public.skill_extraction_requirement_matches"
    occupation_detail_match_table = "public.occupation_detail_matches"


def test_structured_source_prefers_occupation_detail_match_table(monkeypatch):
    monkeypatch.setattr(
        "src.analysis.structured_pg_source.load_skill_extraction_config",
        lambda: _FakeSkillConfig(),
    )

    config = load_default_structured_source_config()

    assert config.normalized_table == "public.recruitment_jobs_normalized"
    assert config.occupation_match_table == "public.occupation_detail_matches"
