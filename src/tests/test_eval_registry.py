"""测试评估注册表读写。"""
import json
from pathlib import Path
from src.skill_extraction._eval_registry import (
    load_registry,
    append_eval_record,
    get_record_by_version,
    list_records,
)


SAMPLE_RECORD = {
    "dict_version": "v1",
    "evaluated_at": "2026-06-12T14:00:00",
    "soft_skill_metrics": {
        "coverage": 0.1141,
        "precision": 0.0876,
        "dimension_accuracy": 0.8495,
    },
    "hard_skill_metrics": {
        "precision": 0.7018,
        "recall": 0.9053,
        "f1": 0.7907,
        "category_accuracy": 1.0,
    },
    "gold_source": "annotations.label_studio_tasks_v2",
    "sample_count": 300,
}


def test_load_registry_creates_if_missing(tmp_path):
    """registry.json 不存在时自动创建空注册表。"""
    result = load_registry(tmp_path)
    assert result == {"evaluations": []}
    assert (tmp_path / "registry.json").exists()


def test_load_registry_reads_existing(tmp_path):
    """读取已有注册表。"""
    existing = {"evaluations": [SAMPLE_RECORD]}
    (tmp_path / "registry.json").write_text(
        json.dumps(existing, ensure_ascii=False)
    )
    result = load_registry(tmp_path)
    assert len(result["evaluations"]) == 1


def test_append_eval_record(tmp_path):
    """追加一条评估记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    with open(tmp_path / "registry.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["evaluations"]) == 1
    assert data["evaluations"][0]["dict_version"] == "v1"


def test_append_multiple_records(tmp_path):
    """追加多条记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    record2 = dict(SAMPLE_RECORD, dict_version="v2")
    append_eval_record(tmp_path, record2)
    with open(tmp_path / "registry.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["evaluations"]) == 2


def test_get_record_by_version(tmp_path):
    """按版本号获取最新记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    record2 = dict(
        SAMPLE_RECORD,
        dict_version="v1",
        evaluated_at="2026-06-12T15:00:00",
    )
    append_eval_record(tmp_path, record2)
    result = get_record_by_version(tmp_path, "v1")
    # 返回最新一条
    assert result["evaluated_at"] == "2026-06-12T15:00:00"


def test_get_record_by_version_not_found(tmp_path):
    """版本不存在时返回 None。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    result = get_record_by_version(tmp_path, "v99")
    assert result is None


def test_list_records(tmp_path):
    """列出所有版本的最新评估记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    record2 = dict(SAMPLE_RECORD, dict_version="v2")
    append_eval_record(tmp_path, record2)
    records = list_records(tmp_path)
    assert len(records) == 2
    versions = {r["dict_version"] for r in records}
    assert versions == {"v1", "v2"}
