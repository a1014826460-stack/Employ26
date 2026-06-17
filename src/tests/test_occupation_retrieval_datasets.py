from src.occupation_retrieval.datasets import (
    build_anchor,
    build_candidate_records,
    get_majority_choice,
    get_task_choices,
    parse_choice,
)
from src.occupation_retrieval.common import resolve_existing_model_path

import pytest


def _annotation(choice: str | None) -> dict:
    if choice is None:
        return {"result": [{"from_name": "best_candidate_choice", "value": {"choices": []}}]}
    return {
        "result": [
            {
                "from_name": "best_candidate_choice",
                "value": {"choices": [choice]},
            }
        ]
    }


def test_parse_choice_extracts_candidate_letter() -> None:
    assert parse_choice(_annotation("候选 A")) == "A"
    assert parse_choice(_annotation("候选 E")) == "E"


def test_parse_choice_handles_none_empty_and_missing_fields() -> None:
    assert parse_choice(_annotation("以上选项都不属于")) == "NONE"
    assert parse_choice(_annotation(None)) is None
    assert parse_choice({"result": []}) is None
    assert parse_choice({}) is None


def test_get_task_choices_can_filter_none() -> None:
    task = {
        "annotations": [
            _annotation("候选 A"),
            _annotation("以上选项都不属于"),
            _annotation(None),
        ]
    }
    assert get_task_choices(task) == ["A", "NONE"]
    assert get_task_choices(task, include_none=False) == ["A"]


def test_get_majority_choice_supports_strict_majority_and_ties() -> None:
    assert get_majority_choice(["A", "A", "B"]) == ("A", 2)
    assert get_majority_choice(["A", "B"]) == (None, 1)
    assert get_majority_choice(["A", "B"], require_strict=False) == ("A", 1)
    assert get_majority_choice(["NONE", "NONE"]) == ("NONE", 2)
    assert get_majority_choice([]) == (None, 0)


def test_build_candidate_records_is_stable_with_missing_fields() -> None:
    records = build_candidate_records(
        {
            "candidate_a_code": "1-01-01",
            "candidate_a_title": "测试职业",
            "candidate_a_source": "top1",
            "candidate_c_title": "缺代码职业",
        }
    )
    assert len(records) == 5
    assert records[0] == {
        "letter": "A",
        "code": "1-01-01",
        "title": "测试职业",
        "source": "top1",
    }
    assert records[2]["letter"] == "C"
    assert records[2]["code"] == ""
    assert records[2]["title"] == "缺代码职业"


def test_build_anchor_preserves_historical_spacing() -> None:
    assert build_anchor("算法工程师", "负责模型训练") == "算法工程师 负责模型训练"
    assert build_anchor("", "负责模型训练") == "负责模型训练"
    assert build_anchor("算法工程师", "") == "算法工程师"
    assert build_anchor("", "") == ""


def test_resolve_existing_model_path_allows_huggingface_model_name() -> None:
    assert str(resolve_existing_model_path("BAAI/bge-large-zh-v1.5", label="base")) == "BAAI/bge-large-zh-v1.5"


def test_resolve_existing_model_path_rejects_missing_local_path() -> None:
    with pytest.raises(FileNotFoundError, match="模型路径不存在"):
        resolve_existing_model_path(
            "output/occupation_retrieval/missing-model",
            label="base",
        )
