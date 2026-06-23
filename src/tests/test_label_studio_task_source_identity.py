from src.db.label_studio_task_source_identity import (
    build_task_payload_fingerprint,
    split_table_name,
)


def test_build_task_payload_fingerprint_is_stable() -> None:
    first = build_task_payload_fingerprint(
        task_job_title="销售经理",
        task_job_requirements="负责销售管理",
        task_sample_source="tier3_main",
        task_snapshot_row_id="1662",
    )
    second = build_task_payload_fingerprint(
        task_job_title="销售经理",
        task_job_requirements="负责销售管理",
        task_sample_source="tier3_main",
        task_snapshot_row_id="1662",
    )
    assert first == second


def test_build_task_payload_fingerprint_changes_when_payload_changes() -> None:
    first = build_task_payload_fingerprint(
        task_job_title="销售经理",
        task_job_requirements="负责销售管理",
        task_sample_source="tier3_main",
        task_snapshot_row_id="1662",
    )
    second = build_task_payload_fingerprint(
        task_job_title="销售经理",
        task_job_requirements="负责区域销售管理",
        task_sample_source="tier3_main",
        task_snapshot_row_id="1662",
    )
    assert first != second


def test_split_table_name_supports_schema_table() -> None:
    assert split_table_name("annotations.label_studio_task_source_identity") == (
        "annotations",
        "label_studio_task_source_identity",
    )
