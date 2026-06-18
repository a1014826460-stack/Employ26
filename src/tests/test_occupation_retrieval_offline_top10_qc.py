import json

from src.occupation_retrieval.offline_top10_qc import (
    QCSample,
    build_gold_samples,
    compute_hit_rank,
    evaluate_distribution,
    parse_top10_candidates,
    render_report,
    review_miss_sample_with_llm,
    summarize_qc_samples,
)


def _task(
    *,
    task_id: int = 1,
    recruitment_record_id: str = "rid-1",
    choices: list[str] | None = None,
) -> dict:
    annotations = []
    for choice in choices or ["候选 A"]:
        annotations.append(
            {
                "result": [
                    {
                        "from_name": "best_candidate_choice",
                        "value": {"choices": [choice]},
                    }
                ]
            }
        )
    return {
        "task_id": task_id,
        "recruitment_record_id": recruitment_record_id,
        "data": {
            "job_title": "算法工程师",
            "job_requirements_clean": "负责推荐算法和机器学习模型",
            "candidate_a_code": "2-02-10-09",
            "candidate_a_title": "人工智能工程技术人员",
            "candidate_b_code": "2-02-10-03",
            "candidate_b_title": "计算机软件工程技术人员",
        },
        "annotations": annotations,
    }


def test_parse_top10_candidates_handles_list_json_and_empty_values() -> None:
    assert parse_top10_candidates(None) == []
    assert parse_top10_candidates([]) == []
    assert parse_top10_candidates([{"rank": 1, "code": "A"}])[0]["code"] == "A"
    assert parse_top10_candidates('[{"rank": 1, "code": "A"}]')[0]["rank"] == 1
    assert parse_top10_candidates("not-json") == []


def test_compute_hit_rank_detects_ranked_code_match() -> None:
    candidates = [
        {"rank": 1, "code": "A"},
        {"rank": 2, "code": "B"},
    ]
    assert compute_hit_rank("B", candidates) == 2
    assert compute_hit_rank("C", candidates) is None
    assert compute_hit_rank("", candidates) is None


def test_build_gold_samples_maps_majority_choice_to_candidate_code() -> None:
    samples = build_gold_samples([_task(choices=["候选 A", "候选 A", "候选 B"])])

    assert len(samples) == 1
    assert samples[0].gold_choice == "A"
    assert samples[0].gold_code == "2-02-10-09"
    assert samples[0].qc_label == "PENDING"


def test_build_gold_samples_keeps_none_and_tie_cases() -> None:
    samples = build_gold_samples(
        [
            _task(task_id=1, recruitment_record_id="rid-1", choices=["以上选项都不属于"]),
            _task(task_id=2, recruitment_record_id="rid-2", choices=["候选 A", "候选 B"]),
        ]
    )

    assert samples[0].qc_label == "GOLD_NONE"
    assert samples[1].qc_label == "MISSING_GOLD"


def test_summarize_qc_samples_labels_hits_misses_and_missing_matches() -> None:
    samples = [
        QCSample(
            task_id=1,
            recruitment_record_id="rid-hit",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="A",
            gold_title="职业A",
        ),
        QCSample(
            task_id=2,
            recruitment_record_id="rid-miss",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="C",
            gold_title="职业C",
        ),
        QCSample(
            task_id=3,
            recruitment_record_id="rid-none",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="NONE",
            qc_label="GOLD_NONE",
        ),
        QCSample(
            task_id=4,
            recruitment_record_id="rid-missing",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="D",
            gold_title="职业D",
        ),
    ]
    matches = {
        "rid-hit": {"top10_candidates": [{"rank": 1, "code": "A", "title": "职业A"}]},
        "rid-miss": {"top10_candidates": [{"rank": 1, "code": "B", "title": "职业B"}]},
    }

    summary = summarize_qc_samples(samples, matches)

    assert summary["counts"]["HIT@10"] == 1
    assert summary["counts"]["MISS@10"] == 1
    assert summary["counts"]["GOLD_NONE"] == 1
    assert summary["counts"]["MISSING_MATCH"] == 1
    assert samples[0].hit_rank == 1
    assert samples[1].qc_label == "MISS@10"


def test_evaluate_distribution_reports_concentration_and_uniformity() -> None:
    summary = evaluate_distribution({"A": 3, "B": 1})

    assert summary["total"] == 4
    assert summary["num_classes"] == 2
    assert summary["top_code"] == "A"
    assert summary["max_share"] == 75.0
    assert summary["assessment"] == "明显不均匀"


class FakeLLMClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, **kwargs):
        self.last_kwargs = kwargs
        return self.payload


def test_review_miss_sample_with_llm_returns_advisory_result_only() -> None:
    sample = QCSample(
        task_id=1,
        recruitment_record_id="rid-1",
        job_title="算法工程师",
        job_requirements="负责推荐算法",
        gold_choice="A",
        gold_code="A",
        gold_title="职业A",
        top10_candidates=[{"rank": 1, "code": "B", "title": "职业B", "score": 0.9}],
        qc_label="MISS@10",
    )
    client = FakeLLMClient(
        {
            "top10_contains_correct": False,
            "best_candidate_rank": None,
            "should_be_none": False,
            "near_miss_code": "B",
            "reason": "Top10 中只有相近工程类岗位。",
            "review_needed": True,
        }
    )

    result = review_miss_sample_with_llm(sample, client)

    assert result["llm_status"] == "ok"
    assert result["review_needed"] is True
    assert result["near_miss_code"] == "B"


def test_render_report_contains_deterministic_and_llm_sections() -> None:
    sample = QCSample(
        task_id=1,
        recruitment_record_id="rid-1",
        job_title="算法工程师",
        job_requirements="负责推荐算法",
        gold_choice="A",
        gold_code="A",
        gold_title="职业A",
        qc_label="HIT@10",
        hit_rank=1,
    )
    report = render_report(
        samples=[sample],
        summary={"total_samples": 1, "eligible_total": 1, "counts": {"HIT@10": 1}},
        distribution=evaluate_distribution({"A": 1}),
        llm_reviews=[{"task_id": 2, "llm_status": "ok", "reason": "需要人工复核"}],
    )

    assert "Occupation Retrieval Offline Top10 QC Report" in report
    assert "确定性 Top10 命中统计" in report
    assert "本地 LLM Miss 样本复核" in report
    assert "需要人工复核" in report
    assert json.dumps({"sanity": True})
