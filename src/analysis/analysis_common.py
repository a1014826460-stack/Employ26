"""分析链路共享的输入规范、输出目录和清单工具。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import get_project_paths


RUN_DATE_FORMAT = "%m-%d"

SOURCE_TABLE_ALIASES = {
    "recruit.main.gd_recruit_qcwy_sample": '"51job".sample',
    "recruit.main.gd_recruit_liepin_sample": '"Liepin".sample',
    "recruit.main.zhilian_guangdong_sample": '"Zhilian".sample',
    "recruit.main.gd_recruit_qcwy_cleaned": '"51job".cleaned_data',
    "recruit.main.gd_recruit_liepin_cleaned": '"Liepin".cleaned_data',
    "recruit.main.zhilian_guangdong_cleaned": '"Zhilian".cleaned_data',
}

GUANGDONG_CITIES = [
    "深圳",
    "广州",
    "佛山",
    "东莞",
    "惠州",
    "珠海",
    "中山",
    "江门",
    "肇庆",
    "汕头",
    "湛江",
    "茂名",
    "韶关",
    "梅州",
    "清远",
    "阳江",
    "河源",
    "云浮",
    "潮州",
    "揭阳",
    "汕尾",
]


def build_analysis_output_dir(
    workflow_prefix: str,
    run_date: datetime | None = None,
    *,
    base_output_dir: Path | None = None,
) -> Path:
    """构建分析批次输出目录。"""
    current = run_date or datetime.now()
    reports_root = base_output_dir or (get_project_paths().output_dir / "reports")
    return reports_root / f"{workflow_prefix}_{current.strftime(RUN_DATE_FORMAT)}"


def build_structured_output_dir(
    run_date: datetime | None = None,
    *,
    base_output_dir: Path | None = None,
) -> Path:
    """构建结构化统计批次输出目录。"""
    return build_analysis_output_dir(
        "structured_analysis",
        run_date=run_date,
        base_output_dir=base_output_dir,
    )


def build_requirement_output_dir(
    run_date: datetime | None = None,
    *,
    base_output_dir: Path | None = None,
) -> Path:
    """构建 requirement text 批次输出目录。"""
    return build_analysis_output_dir(
        "req_analysis",
        run_date=run_date,
        base_output_dir=base_output_dir,
    )


def collect_output_files(
    output_dir: Path,
    *,
    extra_outputs: list[str] | None = None,
) -> list[str]:
    """收集输出目录内文件，并合并额外的数据库产物引用。"""
    filenames = sorted(path.name for path in output_dir.iterdir() if path.is_file()) if output_dir.exists() else []
    merged = set(filenames)
    for item in extra_outputs or []:
        merged.add(item)
    return sorted(merged)


def write_run_manifest(
    output_dir: Path,
    *,
    workflow: str,
    steps: list[str],
    params: dict[str, Any],
    input_files: list[str],
    output_files: list[str],
    extra_fields: dict[str, Any] | None = None,
) -> Path:
    """写入统一运行清单。"""
    manifest = {
        "workflow": workflow,
        "run_timestamp": datetime.now().isoformat(),
        "steps": steps,
        "params": params,
        "input_files": input_files,
        "output_files": output_files,
    }
    if extra_fields:
        manifest.update(extra_fields)

    output_path = output_dir / "run_manifest.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_mapped_source_table_expr(column_ref: str) -> str:
    """把历史来源表名映射成统一规范层使用的 PostgreSQL 表名。"""
    branches = "\n".join(
        f"        WHEN {old_name!r} THEN {new_name!r}"
        for old_name, new_name in SOURCE_TABLE_ALIASES.items()
    )
    return (
        f"CASE {column_ref}\n"
        f"{branches}\n"
        f"        ELSE {column_ref}\n"
        "    END"
    )


def safe_string(value: object) -> str:
    """把空值安全转成去空白字符串。"""
    if value is None or pd.isna(value):
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def parse_publish_month(value: object) -> str:
    """把发布时间解析到 YYYY-MM。"""
    text_value = safe_string(value)
    if not text_value:
        return ""
    parsed = pd.to_datetime(text_value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m")


def normalize_city(value: object) -> str:
    """轻量标准化广东城市。"""
    text_value = safe_string(value)
    if not text_value:
        return "未知"
    for city in GUANGDONG_CITIES:
        if city in text_value:
            return city
    return "其他"


def normalize_industry(value: object) -> str:
    """轻量标准化行业。"""
    text_value = safe_string(value)
    if not text_value:
        return "未知"
    normalized = re.sub(r"[,，/、]+", ",", text_value)
    return normalized.split(",")[0].strip() or "未知"


def normalize_company_size(value: object) -> str:
    """轻量标准化公司规模。"""
    text_value = safe_string(value)
    return text_value or "未知"


def enrich_common_dimension_columns(
    source_df: pd.DataFrame,
    *,
    publish_date_column: str = "publish_date",
    city_column: str = "work_city",
    industry_column: str = "company_industry_raw",
    company_size_column: str = "company_size_raw",
) -> pd.DataFrame:
    """补齐两条分析链路共用的标准维度列。"""
    # 直接在原 DataFrame 上添加新列，避免 copy() 导致内存翻倍
    df = source_df
    if publish_date_column in df.columns:
        # 向量化 pd.to_datetime 替代逐行 .map(parse_publish_month)，860 万行从数十分钟降至数秒
        dates = pd.to_datetime(df[publish_date_column], errors="coerce")
        df["publish_month"] = dates.dt.strftime("%Y-%m").where(dates.notna(), "")
    else:
        df["publish_month"] = pd.NA
    if city_column in df.columns:
        df["city_normalized"] = df[city_column].map(normalize_city)
    else:
        df["city_normalized"] = pd.NA
    if industry_column in df.columns:
        df["industry_normalized"] = df[industry_column].map(normalize_industry)
    else:
        df["industry_normalized"] = pd.NA
    if company_size_column in df.columns:
        df["company_size_normalized"] = df[company_size_column].map(normalize_company_size)
    else:
        df["company_size_normalized"] = pd.NA
    return df
