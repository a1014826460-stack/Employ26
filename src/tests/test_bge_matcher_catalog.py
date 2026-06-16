from pathlib import Path

import pandas as pd

from src.skill_extraction.bge_matcher import OccupationBGEMatcher, _format_occupation_code


class _FakeConfig:
    embedding_model_path = Path("unused-model")
    embedding_device = "cpu"
    catalog_preprocessed_table = "public.occ_dict_unified"
    db_path = Path("unused.duckdb")
    duckdb_threads = 1
    catalog_embedding_cache_path = Path("unused.npy")
    embedding_batch_size = 32
    match_top_k = 10


def test_format_occupation_code_normalizes_plain_digits():
    assert _format_occupation_code("4050301") == "4-05-03-01"
    assert _format_occupation_code("4-05-03-01") == "4-05-03-01"


def test_load_catalog_uses_hierarchy_detail_as_canonical(monkeypatch):
    catalog_df = pd.DataFrame(
        [
            {
                "code": "4-05-03-01",
                "title": "保险代理人",
                "desc": "",
                "tasks": "",
                "大类": "商业、服务业人员",
                "中类": "运输服务人员",
                "小类": "航空运输服务人员",
                "细类": "航空运输飞行服务员",
                "task_text_joined": "",
                "title_clean": "",
                "desc_clean": "",
                "hierarchy_text": "商业、服务业人员 运输服务人员 航空运输服务人员 航空运输飞行服务员",
            }
        ]
    )

    class _FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeEngine:
        def connect(self):
            return _FakeConnection()

        def dispose(self):
            return None

    monkeypatch.setattr("src.skill_extraction.bge_matcher.create_pg_engine", lambda **_: _FakeEngine())
    monkeypatch.setattr("src.skill_extraction.bge_matcher.pd.read_sql_query", lambda *_args, **_kwargs: catalog_df)
    monkeypatch.setattr(OccupationBGEMatcher, "__init__", lambda self, config: setattr(self, "config", config))

    matcher = OccupationBGEMatcher(_FakeConfig())
    matcher.catalog_df = pd.DataFrame()
    matcher.catalog_embeddings = None

    result = matcher.load_catalog()

    assert result.loc[0, "code"] == "4-05-03-01"
    assert result.loc[0, "title"] == "保险代理人"
    assert result.loc[0, "细类"] == "保险代理人"
    assert result.loc[0, "detail_name"] == "保险代理人"
    assert result.loc[0, "detail_path"].endswith("保险代理人")
