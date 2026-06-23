import pandas as pd

from src.llm.occupation_agent_service import (
    OccupationAnalysisRequest,
    build_llm_messages,
    extract_top_candidates,
    request_to_dict,
)
from src.occupation_retrieval.top10_selector import Top10SelectionResult


def test_extract_top_candidates_keeps_ranked_top10_fields():
    matched_row = pd.Series(
        {
            "top1_code": "2-02-13-05",
            "top1_title": "算法工程技术人员",
            "top1_score": 0.91,
            "top1_detail_path": "专业技术人员 > 工程技术人员 > 信息工程技术人员 > 算法工程技术人员",
            "top1_detail_name": "算法工程技术人员",
            "top2_code": "2-02-10-03",
            "top2_title": "计算机软件工程技术人员",
            "top2_score": 0.84,
            "top2_detail_path": "专业技术人员 > 工程技术人员 > 信息和通信工程技术人员 > 计算机软件工程技术人员",
            "top2_detail_name": "计算机软件工程技术人员",
        }
    )

    candidates = extract_top_candidates(matched_row, top_k=10)

    assert len(candidates) == 2
    assert candidates[0]["rank"] == 1
    assert candidates[0]["code"] == "2-02-13-05"
    assert candidates[0]["title"] == "算法工程技术人员"
    assert candidates[0]["score"] == 0.91
    assert candidates[1]["rank"] == 2


def test_build_llm_messages_contains_job_candidates_and_report_contract():
    request = OccupationAnalysisRequest(
        job_title="算法工程师",
        job_description="负责推荐算法、用户画像、模型训练和效果评估",
        top_k=10,
    )
    candidates = [
        {
            "rank": 1,
            "code": "2-02-13-05",
            "title": "算法工程技术人员",
            "score": 0.91,
            "detail_path": "专业技术人员 > 工程技术人员 > 信息工程技术人员 > 算法工程技术人员",
            "detail_name": "算法工程技术人员",
        }
    ]

    messages = build_llm_messages(request=request, candidates=candidates)

    assert messages[0]["role"] == "system"
    assert "职业细类识别" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "算法工程师" in messages[1]["content"]
    assert "推荐算法" in messages[1]["content"]
    assert "Top10" in messages[1]["content"]
    assert "算法工程技术人员" in messages[1]["content"]
    assert "最终建议" in messages[1]["content"]


def test_request_to_dict_keeps_boolean_and_topk_fields():
    payload = request_to_dict(
        OccupationAnalysisRequest(
            job_title="算法工程师",
            job_description="负责推荐算法",
            top_k=10,
            include_llm_report=True,
        )
    )
    assert payload["job_title"] == "算法工程师"
    assert payload["top_k"] == 10
    assert payload["include_llm_report"] is True


def test_top10_selection_result_has_stable_fields():
    result = Top10SelectionResult(
        selected_rank=2,
        selected_code="2-02-10-09",
        selected_title="人工智能工程技术人员",
        reason="更贴近职责",
        confidence=0.82,
        needs_review=False,
    )
    assert result.selected_rank == 2
    assert result.selected_code == "2-02-10-09"
