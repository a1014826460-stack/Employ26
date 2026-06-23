from src.occupation_retrieval.top10_selector import (
    Top10SelectionResult,
    normalize_selector_payload,
)


def test_normalize_selector_payload_extracts_ranked_winner() -> None:
    payload = {
        "selected_rank": 2,
        "selected_code": "2-02-10-09",
        "selected_title": "人工智能工程技术人员",
        "reason": "更贴近岗位职责",
        "confidence": 0.82,
        "needs_review": False,
    }
    result = normalize_selector_payload(
        payload,
        top10_candidates=[
            {"rank": 1, "code": "2-02-10-03", "title": "计算机软件工程技术人员"},
            {"rank": 2, "code": "2-02-10-09", "title": "人工智能工程技术人员"},
        ],
    )
    assert isinstance(result, Top10SelectionResult)
    assert result.selected_rank == 2
    assert result.selected_code == "2-02-10-09"
    assert result.selected_title == "人工智能工程技术人员"
    assert result.reason == "更贴近岗位职责"


def test_normalize_selector_payload_falls_back_to_rank_lookup() -> None:
    result = normalize_selector_payload(
        {
            "selected_rank": 1,
            "reason": "Top1 已经最合适",
            "confidence": 0.9,
        },
        top10_candidates=[
            {"rank": 1, "code": "2-02-10-03", "title": "计算机软件工程技术人员"},
        ],
    )
    assert result.selected_rank == 1
    assert result.selected_code == "2-02-10-03"
    assert result.selected_title == "计算机软件工程技术人员"
    assert result.needs_review is False


def test_normalize_selector_payload_marks_review_for_invalid_winner() -> None:
    result = normalize_selector_payload(
        {
            "selected_rank": 99,
            "selected_code": "",
            "selected_title": "",
            "reason": "候选都不够好",
            "confidence": 0.2,
            "needs_review": True,
        },
        top10_candidates=[
            {"rank": 1, "code": "2-02-10-03", "title": "计算机软件工程技术人员"},
        ],
    )
    assert result.selected_rank is None
    assert result.selected_code == ""
    assert result.needs_review is True
