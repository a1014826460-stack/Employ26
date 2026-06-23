import json
import csv

from src.occupation_retrieval.offline_top10_qc import (
    QCSample,
    apply_deepseek_arbitration,
    apply_llm_top10_selection,
    build_deepseek_arbitration_prompts,
    build_gold_samples,
    compute_hit_rank,
    evaluate_distribution,
    parse_top10_candidates,
    render_report,
    review_miss_sample_with_llm,
    summarize_selector_metrics,
    summarize_qc_samples,
    write_llm_outputs_csv,
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


def test_summarize_qc_samples_excludes_identity_review_required_from_hit_denominator() -> None:
    samples = [
        QCSample(
            task_id=1,
            recruitment_record_id="rid-review",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="A",
            gold_title="职业A",
            task_source_identity_status="REVIEW_REQUIRED",
        ),
        QCSample(
            task_id=2,
            recruitment_record_id="rid-hit",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="A",
            gold_title="职业A",
            task_source_identity_status="AUTO_CONFIRMED",
        ),
    ]
    matches = {
        "rid-hit": {"top10_candidates": [{"rank": 1, "code": "A", "title": "职业A"}]},
    }

    summary = summarize_qc_samples(samples, matches)

    assert summary["eligible_total"] == 1
    assert summary["counts"]["IDENTITY_REVIEW_REQUIRED"] == 1
    assert summary["counts"]["HIT@10"] == 1
    assert samples[0].qc_label == "IDENTITY_REVIEW_REQUIRED"


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


def test_apply_llm_top10_selection_writes_structured_winner_back_to_sample(monkeypatch) -> None:
    sample = QCSample(
        task_id=1,
        recruitment_record_id="rid-1",
        job_title="算法工程师",
        job_requirements="负责推荐算法",
        gold_choice="A",
        gold_code="A",
        gold_title="职业A",
        top10_candidates=[{"rank": 1, "code": "B", "title": "职业B", "score": 0.9}],
    )

    monkeypatch.setattr(
        "src.occupation_retrieval.offline_top10_qc.select_from_top10",
        lambda **kwargs: __import__("src.occupation_retrieval.top10_selector", fromlist=["Top10SelectionResult"]).Top10SelectionResult(
            selected_rank=1,
            selected_code="B",
            selected_title="职业B",
            reason="更贴近职责",
            confidence=0.8,
            needs_review=False,
        ),
    )
    monkeypatch.setattr(
        "src.occupation_retrieval.offline_top10_qc.create_llm_client",
        lambda backend=None: object(),
    )

    outputs = apply_llm_top10_selection([sample], limit=20)

    assert outputs[0]["selected_code"] == "B"
    assert sample.llm_selection_attempted is True
    assert sample.llm_selected_code == "B"
    assert sample.llm_selected_rank == 1


def test_summarize_selector_metrics_only_counts_attempted_llm_samples() -> None:
    samples = [
        QCSample(
            task_id=1,
            recruitment_record_id="rid-1",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="A",
            top10_candidates=[{"rank": 1, "code": "B", "title": "职业B"}],
            top1_code="B",
            llm_selected_code="A",
            llm_selection_attempted=True,
        ),
        QCSample(
            task_id=2,
            recruitment_record_id="rid-2",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="A",
            top10_candidates=[{"rank": 1, "code": "A", "title": "职业A"}],
            top1_code="A",
            llm_selected_code="B",
            llm_selection_attempted=True,
        ),
        QCSample(
            task_id=3,
            recruitment_record_id="rid-3",
            job_title="算法工程师",
            job_requirements="负责推荐算法",
            gold_choice="A",
            gold_code="A",
            top10_candidates=[{"rank": 1, "code": "A", "title": "职业A"}],
            top1_code="A",
            llm_selected_code="",
            llm_selection_attempted=False,
        ),
    ]

    metrics = summarize_selector_metrics(samples)

    assert metrics["selector_total"] == 2
    assert metrics["top1_hit"] == 1
    assert metrics["llm_hit"] == 1
    assert metrics["llm_better_than_top1"] == 1
    assert metrics["llm_worse_than_top1"] == 1


def test_build_deepseek_arbitration_prompts_include_full_top10_and_both_sides() -> None:
    sample = QCSample(
        task_id=1,
        recruitment_record_id="rid-1",
        job_title="算法工程师",
        job_requirements="负责推荐算法",
        gold_choice="A",
        gold_code="A",
        gold_title="职业A",
        top1_code="B",
        top1_title="职业B",
        llm_selected_code="B",
        llm_selected_title="职业B",
        llm_selected_rank=2,
        top10_candidates=[
            {"rank": 1, "code": "B", "title": "职业B", "score": 0.95},
            {"rank": 2, "code": "C", "title": "职业C", "score": 0.91},
        ],
    )
    system_prompt, user_prompt = build_deepseek_arbitration_prompts(sample)

    assert "support_llm、support_gold、support_neither" in system_prompt
    assert "完整 Top10 候选" in user_prompt
    assert "LLM 从 Top10 中选中的结果" in user_prompt
    assert "人工 gold" in user_prompt
    assert "1. B 职业B" in user_prompt
    assert "2. C 职业C" in user_prompt


def test_apply_deepseek_arbitration_only_runs_for_llm_gold_disagreement(monkeypatch) -> None:
    sample_support_gold = QCSample(
        task_id=1,
        recruitment_record_id="rid-1",
        job_title="算法工程师",
        job_requirements="负责推荐算法",
        gold_choice="A",
        gold_code="A",
        gold_title="职业A",
        top1_code="B",
        top1_title="职业B",
        llm_selected_code="B",
        llm_selected_title="职业B",
        llm_selected_rank=1,
        top10_candidates=[{"rank": 1, "code": "B", "title": "职业B", "score": 0.95}],
    )
    sample_same = QCSample(
        task_id=2,
        recruitment_record_id="rid-2",
        job_title="算法工程师",
        job_requirements="负责推荐算法",
        gold_choice="A",
        gold_code="A",
        gold_title="职业A",
        top1_code="A",
        top1_title="职业A",
        llm_selected_code="A",
        llm_selected_title="职业A",
        llm_selected_rank=1,
        top10_candidates=[{"rank": 1, "code": "A", "title": "职业A", "score": 0.98}],
    )

    class FakeDeepSeekClient:
        def complete_json(self, **kwargs):
            self.last_kwargs = kwargs
            return {
                "support": "support_gold",
                "reason": "更支持人工标注结果",
                "review_needed": False,
            }

    fake_client = FakeDeepSeekClient()
    monkeypatch.setattr(
        "src.occupation_retrieval.offline_top10_qc.build_deepseek_client",
        lambda **kwargs: fake_client,
    )

    outputs = apply_deepseek_arbitration([sample_support_gold, sample_same], limit=20)

    assert len(outputs) == 1
    assert outputs[0]["support"] == "support_gold"
    assert sample_support_gold.arbitration_support == "support_gold"
    assert sample_support_gold.arbitration_review_needed is False
    assert sample_same.arbitration_support == ""


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
        top1_code="A",
        llm_selected_code="A",
    )
    report = render_report(
        samples=[sample],
        summary={"total_samples": 1, "eligible_total": 1, "counts": {"HIT@10": 1}},
        distribution=evaluate_distribution({"A": 1}),
        selector_metrics={"selector_total": 1, "top1_hit": 1, "llm_hit": 1, "llm_same_as_top1": 1, "llm_better_than_top1": 0, "llm_worse_than_top1": 0},
        arbitration_outputs=[{"task_id": 2, "top1_code": "B", "llm_selected_code": "C", "gold_code": "A", "support": "support_gold", "review_needed": True, "reason": "更支持人工"}],
        llm_reviews=[{"task_id": 2, "llm_status": "ok", "reason": "需要人工复核"}],
    )

    assert "Occupation Retrieval Offline Top10 QC Report" in report
    assert "确定性 Top10 命中统计" in report
    assert "Top1 与 LLM-over-Top10 对比" in report
    assert "DeepSeek 实时二裁" in report
    assert "本地 LLM Miss 样本复核" in report
    assert "需要人工复核" in report
    assert json.dumps({"sanity": True})


def test_write_llm_outputs_csv_exports_attempted_samples(tmp_path) -> None:
    sample = QCSample(
        task_id=1,
        recruitment_record_id="rid-1",
        job_title="算法工程师",
        job_requirements="负责推荐算法",
        gold_choice="A",
        gold_code="A",
        gold_title="职业A",
        qc_label="HIT@10",
        hit_rank=2,
        top1_code="B",
        top1_title="职业B",
        llm_selection_attempted=True,
        llm_selected_rank=2,
        llm_selected_code="A",
        llm_selected_title="职业A",
        arbitration_support="support_gold",
        arbitration_review_needed=True,
        arbitration_reason="人工 gold 更合理",
        top10_candidates=[
            {"rank": 1, "code": "B", "title": "职业B", "score": 0.9},
            {"rank": 2, "code": "A", "title": "职业A", "score": 0.8},
        ],
    )
    skipped = QCSample(
        task_id=2,
        recruitment_record_id="rid-2",
        job_title="未跑 LLM",
        job_requirements="",
        gold_choice="A",
        gold_code="A",
    )
    output_path = tmp_path / "llm_outputs.csv"

    write_llm_outputs_csv(
        [sample, skipped],
        output_path=output_path,
        llm_backend_name="external_api",
        deepseek_model="deepseek-v4-pro",
    )

    with output_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert len(rows) == 1
    assert rows[0]["task_id"] == "1"
    assert rows[0]["llm_selected_code"] == "A"
    assert rows[0]["llm_top10_correct"] == "True"
    assert rows[0]["llm_better_than_top1"] == "True"
    assert rows[0]["arbitration_support"] == "support_gold"
    assert rows[0]["candidate_1_code"] == "B"
    assert rows[0]["candidate_1_score"] == "0.9"
    assert rows[0]["candidate_2_title"] == "职业A"
