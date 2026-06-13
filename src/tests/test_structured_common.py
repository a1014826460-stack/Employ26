from datetime import datetime

import pandas as pd
import pytest

from src.analysis.structured_common import (
    build_structured_output_dir,
    load_integrated_data,
    set_write_legacy_csv_copies,
    write_csv_with_legacy_copy,
)


def test_build_structured_output_dir_uses_batch_folder(tmp_path):
    output_dir = build_structured_output_dir(
        datetime(2026, 6, 11),
        base_output_dir=tmp_path,
    )

    assert output_dir == tmp_path / "structured_analysis_06-11"


def test_load_integrated_data_validates_required_columns(tmp_path):
    integrated_dir = tmp_path / "integrated"
    integrated_dir.mkdir()
    pd.DataFrame({"岗位名称": ["工程师"]}).to_csv(
        integrated_dir / "sample_整合_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError, match="薪资水平"):
        load_integrated_data(integrated_dir, required_columns={"岗位名称", "薪资水平"})


def test_write_csv_with_legacy_copy_writes_only_canonical_by_default(tmp_path):
    df = pd.DataFrame({"job_count": [1]})
    legacy_path = tmp_path / "行业月度数据.csv"
    legacy_path.write_text("old", encoding="utf-8")

    output_files = write_csv_with_legacy_copy(
        df,
        tmp_path,
        canonical_filename="jobs_by_industry.csv",
        legacy_filename="行业月度数据.csv",
    )

    assert output_files == ["jobs_by_industry.csv"]
    assert (tmp_path / "jobs_by_industry.csv").exists()
    assert not legacy_path.exists()


def test_write_csv_with_legacy_copy_writes_both_names_when_requested(tmp_path):
    df = pd.DataFrame({"job_count": [1]})

    output_files = write_csv_with_legacy_copy(
        df,
        tmp_path,
        canonical_filename="jobs_by_industry.csv",
        legacy_filename="行业月度数据.csv",
        write_legacy_copy=True,
    )

    assert output_files == ["jobs_by_industry.csv", "行业月度数据.csv"]
    assert (tmp_path / "jobs_by_industry.csv").exists()
    assert (tmp_path / "行业月度数据.csv").exists()


def test_write_csv_with_legacy_copy_uses_global_legacy_setting(tmp_path):
    df = pd.DataFrame({"job_count": [1]})
    set_write_legacy_csv_copies(True)
    try:
        output_files = write_csv_with_legacy_copy(
            df,
            tmp_path,
            canonical_filename="jobs_by_industry.csv",
            legacy_filename="行业月度数据.csv",
        )
    finally:
        set_write_legacy_csv_copies(False)

    assert output_files == ["jobs_by_industry.csv", "行业月度数据.csv"]
    assert (tmp_path / "jobs_by_industry.csv").exists()
    assert (tmp_path / "行业月度数据.csv").exists()
