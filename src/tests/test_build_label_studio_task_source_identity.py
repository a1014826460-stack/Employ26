from src.utils.build_label_studio_task_source_identity import (
    pick_forced_candidate,
    resolve_task_source_identity_rows,
)
from src.utils.repair_label_studio_recruitment_record_ids import MatchCandidate, normalize_text


def _task(task_id: int, title: str, requirements: str) -> dict:
    return {
        "id": task_id,
        "row_id": task_id,
        "sample_source": "tier3_main",
        "job_title": title,
        "job_requirements": requirements,
        "recruitment_record_id": "",
        "data_raw": {
            "job_title": title,
            "job_requirements_clean": requirements,
            "sample_source": "tier3_main",
            "row_id": str(task_id),
        },
    }


def _candidate(
    *,
    rrid: str,
    title: str,
    desc: str,
    score: float = 0.0,
    source_row_number: int = 1,
) -> MatchCandidate:
    return MatchCandidate(
        source_table='"51job".sample',
        source_platform="51job",
        source_row_number=source_row_number,
        recruitment_record_id=rrid,
        source_native_job_id="",
        dedupe_fingerprint=f"fp-{rrid}",
        title=title,
        description=desc,
        company_name="公司A",
        work_city="广州",
        salary_raw="10-20K",
        publish_date="2026-06-01",
        education_requirement_raw="本科",
        experience_requirement_raw="3年",
        company_industry_raw="制造业",
        score=score,
    )


def test_resolve_task_source_identity_rows_marks_auto_confirmed_unique_match() -> None:
    tasks = [_task(1, "销售经理", "负责粉末冶金产品销售经验和市场开拓")]
    source_index = {
        normalize_text("销售经理"): [
            _candidate(
                rrid="rid-1",
                title="销售经理",
                desc="负责粉末冶金产品销售经验和市场开拓以及客户维护",
            )
        ]
    }

    identity_rows, candidate_rows, summary = resolve_task_source_identity_rows(tasks, source_index, top_n=3)

    assert summary["status_counts"]["AUTO_CONFIRMED"] == 1
    assert identity_rows[0]["selected_recruitment_record_id"] == "rid-1"
    assert identity_rows[0]["selected_rank"] == 1
    assert candidate_rows[0]["is_selected"] is True


def test_resolve_task_source_identity_rows_marks_review_required_when_tied() -> None:
    tasks = [_task(2, "销售经理", "负责客户开拓和销售管理")]
    source_index = {
        normalize_text("销售经理"): [
            _candidate(
                rrid="rid-1",
                title="销售经理",
                desc="负责客户开拓和销售管理",
                source_row_number=1,
            ),
            _candidate(
                rrid="rid-2",
                title="销售经理",
                desc="负责客户开拓和销售管理",
                source_row_number=2,
            ),
        ]
    }

    identity_rows, candidate_rows, summary = resolve_task_source_identity_rows(tasks, source_index, top_n=10)

    assert summary["status_counts"]["REVIEW_REQUIRED"] == 1
    assert identity_rows[0]["identity_status"] == "REVIEW_REQUIRED"
    assert identity_rows[0]["selected_recruitment_record_id"] in {"rid-1", "rid-2"}
    assert len(candidate_rows) == 2


def test_pick_forced_candidate_prefers_newer_publish_date_when_scores_tie() -> None:
    ranked = [
        _candidate(
            rrid="rid-old",
            title="销售经理",
            desc="负责客户开拓和销售管理",
            source_row_number=1,
        ),
        _candidate(
            rrid="rid-new",
            title="销售经理",
            desc="负责客户开拓和销售管理",
            source_row_number=2,
        ),
    ]
    ranked[0] = ranked[0].__class__(**{**ranked[0].__dict__, "publish_date": "2024-01-01"})
    ranked[1] = ranked[1].__class__(**{**ranked[1].__dict__, "publish_date": "2025-01-01"})

    chosen, chosen_rank = pick_forced_candidate(ranked)

    assert chosen is not None
    assert chosen.recruitment_record_id == "rid-new"
    assert chosen_rank == 2


def test_resolve_task_source_identity_rows_can_force_pick_review_required() -> None:
    tasks = [_task(3, "销售经理", "负责客户开拓和销售管理")]
    source_index = {
        normalize_text("销售经理"): [
            _candidate(
                rrid="rid-1",
                title="销售经理",
                desc="负责客户开拓和销售管理",
                source_row_number=1,
            ),
            _candidate(
                rrid="rid-2",
                title="销售经理",
                desc="负责客户开拓和销售管理",
                source_row_number=2,
            ),
        ]
    }
    source_index[normalize_text("销售经理")][0] = source_index[normalize_text("销售经理")][0].__class__(
        **{**source_index[normalize_text("销售经理")][0].__dict__, "publish_date": "2024-01-01"}
    )
    source_index[normalize_text("销售经理")][1] = source_index[normalize_text("销售经理")][1].__class__(
        **{**source_index[normalize_text("销售经理")][1].__dict__, "publish_date": "2025-01-01"}
    )

    identity_rows, candidate_rows, summary = resolve_task_source_identity_rows(
        tasks,
        source_index,
        top_n=10,
        force_pick_review_required=True,
    )

    assert summary["status_counts"]["AUTO_CONFIRMED"] == 1
    assert summary["forced_auto_confirmed"] == 1
    assert identity_rows[0]["identity_rule"].endswith("+forced_latest")
    assert identity_rows[0]["selected_recruitment_record_id"] == "rid-2"
    assert any(row["is_selected"] and row["recruitment_record_id"] == "rid-2" for row in candidate_rows)
