import json

import pandas as pd

from src.db.occupation_detail_matches import (
    DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE,
    build_occupation_detail_match_records,
    build_top_candidates,
)


def test_build_top_candidates_keeps_top10_order_and_fields():
    row = {
        "top1_code": "1",
        "top1_title": "人工智能工程技术人员",
        "top1_score": 0.91,
        "top1_detail_path": "工程技术 > 人工智能工程技术人员",
        "top1_detail_name": "人工智能工程技术人员",
        "top10_code": "10",
        "top10_title": "其他工程技术人员",
        "top10_score": 0.51,
        "top10_detail_path": "工程技术 > 其他工程技术人员",
        "top10_detail_name": "其他工程技术人员",
    }

    candidates = build_top_candidates(row, top_k=10)

    assert len(candidates) == 10
    assert candidates[0] == {
        "rank": 1,
        "code": "1",
        "title": "人工智能工程技术人员",
        "score": 0.91,
        "detail_path": "工程技术 > 人工智能工程技术人员",
        "detail_name": "人工智能工程技术人员",
    }
    assert candidates[-1]["rank"] == 10
    assert candidates[-1]["code"] == "10"


def test_build_occupation_detail_match_records_uses_top1_as_final_output():
    source_df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "rid-1",
                "source_platform": "Liepin",
                "source_table": '"Liepin".raw_data',
                "source_row_number": 7,
                "job_title": "算法工程师",
            }
        ]
    )
    matched_df = pd.DataFrame(
        [
            {
                "query_text": "算法工程师。负责机器学习模型研发",
                "query_source": "job_title+job_description_raw",
                "selected_candidate_rank": 1,
                "top1_code": "2-02-10-09",
                "top1_title": "人工智能工程技术人员",
                "top1_score": 0.91,
                "大类": "专业技术人员",
                "中类": "工程技术人员",
                "小类": "信息和通信工程技术人员",
                "细类": "人工智能工程技术人员",
                "top1_detail_path": "专业技术人员 > 工程技术人员 > 信息和通信工程技术人员 > 人工智能工程技术人员",
                "top1_detail_name": "人工智能工程技术人员",
            }
        ]
    )

    records = build_occupation_detail_match_records(
        source_df=source_df,
        matched_df=matched_df,
        run_id="test-run",
        model_recipe="v1",
        base_model="bge-large-zh-v1.5",
        model_path="output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned",
        top_k=10,
    )

    assert len(records) == 1
    record = records[0]
    assert record["target_table"] == DEFAULT_OCCUPATION_DETAIL_MATCH_TABLE
    assert record["recruitment_record_id"] == "rid-1"
    assert record["occupation_code"] == "2-02-10-09"
    assert record["occupation_title"] == "人工智能工程技术人员"
    assert record["selected_candidate_rank"] == 1
    assert record["top_k"] == 10
    assert json.loads(record["top10_candidates"])[0]["rank"] == 1

