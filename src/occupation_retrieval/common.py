"""occupation_retrieval 实验脚本共用的数据访问与路径工具。

统一职责：
1. 从 PostgreSQL 读取第二轮 Label Studio 标注任务
2. 从 PostgreSQL 或 JSONL 读取 DeepSeek 重标结果
3. 从 PostgreSQL 读取职业词典
4. 统一 occupation_retrieval 目录下的输出路径与模型路径兜底
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from config.paths import get_project_paths
from src.db.label_studio_task_source_identity import (
    DEFAULT_IDENTITY_TABLE,
    load_all_task_identity_map,
)
from src.db.postgres import create_pg_engine, get_table_columns, table_exists

PROJECT_PATHS = get_project_paths()
PROJECT_ROOT = PROJECT_PATHS.project_root
OCCUPATION_RETRIEVAL_OUTPUT_DIR = PROJECT_PATHS.output_dir / "occupation_retrieval"
TRAINING_OUTPUT_DIR = OCCUPATION_RETRIEVAL_OUTPUT_DIR / "rag_round2_training"
LEGACY_TRAINING_OUTPUT_DIR = PROJECT_ROOT / "output" / "rag_round2_training"

DEFAULT_TASK_TABLE = "annotations.label_studio_tasks_v2"
DEFAULT_DEEPSEEK_TABLE = "annotations.deepseek_relabel_raw"
DEFAULT_OCC_TABLE = "public.occ_dict_unified"
DEEPSEEK_REQUIRED_COLUMNS = {
    "recruitment_record_id": "text",
    "job_title": "text",
    "deepseek_choice": "text",
    "deepseek_confidence": "double precision",
    "deepseek_reasoning": "text",
    "deepseek_raw_response": "text",
    "candidates": "jsonb",
    "payload": "jsonb",
}


def ensure_deepseek_table(
    *,
    table_name: str = DEFAULT_DEEPSEEK_TABLE,
    jsonl_path: Path | None = None,
) -> str:
    """确保 DeepSeek 重标结果已同步到 PostgreSQL。"""
    if jsonl_path is None:
        candidate_paths = [
            PROJECT_ROOT
            / "output"
            / "deepseek_relabel"
            / "round2"
            / "round2_deepseek_relabel_raw.jsonl",
            PROJECT_ROOT / "output" / "deepseek_relabel" / "deepseek_relabel_raw.jsonl",
        ]
        jsonl_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
    schema, table = table_name.split(".", 1)

    engine = create_pg_engine()
    with engine.begin() as conn:
        if not table_exists(conn, schema, table):
            conn.execute(text(f'create schema if not exists "{schema}"'))
            conn.execute(
                text(
                    f"""
                    create table if not exists {table_name} (
                        task_id integer primary key,
                        recruitment_record_id text,
                        job_title text,
                        deepseek_choice text,
                        deepseek_confidence double precision,
                        deepseek_reasoning text,
                        deepseek_raw_response text,
                        candidates jsonb,
                        payload jsonb
                    )
                    """
                )
            )
        else:
            existing_columns = set(get_table_columns(conn, schema, table))
            for column_name, column_type in DEEPSEEK_REQUIRED_COLUMNS.items():
                if column_name not in existing_columns:
                    conn.execute(
                        text(
                            f'alter table {table_name} add column if not exists {column_name} {column_type}'
                        )
                    )

        row_count = conn.execute(text(f"select count(*) from {table_name}")).scalar_one()
        if row_count > 0:
            return table_name

        if not jsonl_path.exists():
            raise FileNotFoundError(f"DeepSeek JSONL 不存在: {jsonl_path}")

        records: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(
                    {
                        "task_id": int(payload["task_id"]),
                        "recruitment_record_id": str(payload.get("recruitment_record_id", "")),
                        "job_title": str(payload.get("job_title", "")),
                        "deepseek_choice": payload.get("deepseek_choice"),
                        "deepseek_confidence": payload.get("deepseek_confidence"),
                        "deepseek_reasoning": payload.get("deepseek_reasoning"),
                        "deepseek_raw_response": payload.get("deepseek_raw_response"),
                        "candidates": json.dumps(payload.get("candidates", {}), ensure_ascii=False),
                        "payload": json.dumps(payload, ensure_ascii=False),
                    }
                )

        insert_sql = text(
            f"""
            insert into {table_name} (
                task_id,
                recruitment_record_id,
                job_title,
                deepseek_choice,
                deepseek_confidence,
                deepseek_reasoning,
                deepseek_raw_response,
                candidates,
                payload
            ) values (
                :task_id,
                :recruitment_record_id,
                :job_title,
                :deepseek_choice,
                :deepseek_confidence,
                :deepseek_reasoning,
                :deepseek_raw_response,
                cast(:candidates as jsonb),
                cast(:payload as jsonb)
            )
            on conflict (task_id) do update set
                recruitment_record_id = excluded.recruitment_record_id,
                job_title = excluded.job_title,
                deepseek_choice = excluded.deepseek_choice,
                deepseek_confidence = excluded.deepseek_confidence,
                deepseek_reasoning = excluded.deepseek_reasoning,
                deepseek_raw_response = excluded.deepseek_raw_response,
                candidates = excluded.candidates,
                payload = excluded.payload
            """
        )
        conn.execute(insert_sql, records)
    return table_name


def load_annotations_from_pg(
    *,
    table_name: str = DEFAULT_TASK_TABLE,
    require_recruitment_record_id: bool = True,
    use_task_source_identity: bool = False,
    identity_table: str = DEFAULT_IDENTITY_TABLE,
) -> list[dict[str, Any]]:
    """从 PostgreSQL 读取任务表，并重建为实验脚本使用的结构。

    返回结果中的身份字段约束：
    - `task_id` 只表示标注任务身份
    - `recruitment_record_id` 表示正式招聘记录身份
    - 不再暴露模糊的 `id` 字段，避免被误读为招聘记录主键
    """
    engine = create_pg_engine()
    with engine.connect() as conn:
        identity_map: dict[int, dict[str, Any]] = {}
        if use_task_source_identity:
            identity_map = load_all_task_identity_map(conn, identity_table=identity_table)
        available_columns = {
            str(row[0])
            for row in conn.execute(
                text(
                    """
                    select column_name
                    from information_schema.columns
                    where table_schema = 'annotations'
                      and table_name = 'label_studio_tasks_v2'
                    """
                )
            ).fetchall()
        }
        recruitment_record_sql = (
            "coalesce(recruitment_record_id, '') as recruitment_record_id,"
            if "recruitment_record_id" in available_columns
            else "'' as recruitment_record_id,"
        )
        rows = conn.execute(
            text(
                f"""
                select
                    id as task_id,
                    {recruitment_record_sql}
                    annotations_completed,
                    data_raw
                from {table_name}
                order by task_id
                """
            )
        ).mappings()
        raw_data: list[dict[str, Any]] = []
        for row in rows:
            annotations_raw = row["annotations_completed"] or "[]"
            data_raw = row["data_raw"] or "{}"
            recruitment_record_id = str(row["recruitment_record_id"] or "")
            identity_row = identity_map.get(int(row["task_id"]))
            identity_status = ""
            if identity_row:
                identity_status = str(identity_row.get("identity_status") or "")
                if identity_status in {"AUTO_CONFIRMED", "MANUAL_CONFIRMED"}:
                    recruitment_record_id = str(identity_row.get("selected_recruitment_record_id") or "")
            if require_recruitment_record_id and not recruitment_record_id:
                raise ValueError(
                    "检测到缺失 recruitment_record_id 的标注任务，"
                    "请先完成历史任务回填后再运行 occupation_retrieval 实验脚本。"
                )
            raw_data.append(
                {
                    "task_id": int(row["task_id"]),
                    "recruitment_record_id": recruitment_record_id,
                    "task_source_identity_status": identity_status,
                    "task_source_identity": identity_row or {},
                    "annotations": json.loads(annotations_raw),
                    "data": json.loads(data_raw),
                }
            )
    return raw_data


def load_deepseek_records(
    *,
    table_name: str = DEFAULT_DEEPSEEK_TABLE,
) -> dict[int, dict[str, Any]]:
    """从 PostgreSQL 读取 DeepSeek 重标结果。"""
    ensure_deepseek_table(table_name=table_name)
    engine = create_pg_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                select
                    task_id,
                    deepseek_choice,
                    deepseek_confidence,
                    deepseek_reasoning,
                    deepseek_raw_response,
                    candidates,
                    payload
                from {table_name}
                """
            )
        ).mappings()
        result: dict[int, dict[str, Any]] = {}
        for row in rows:
            payload = row["payload"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload.setdefault("task_id", int(row["task_id"]))
            payload.setdefault("deepseek_choice", row["deepseek_choice"])
            payload.setdefault("deepseek_confidence", row["deepseek_confidence"])
            payload.setdefault("deepseek_reasoning", row["deepseek_reasoning"])
            payload.setdefault("deepseek_raw_response", row["deepseek_raw_response"])
            result[int(row["task_id"])] = payload
    return result


def load_occupation_dict_df(
    *,
    table_name: str = DEFAULT_OCC_TABLE,
) -> pd.DataFrame:
    """从 PostgreSQL 读取职业词典表。"""
    engine = create_pg_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text(f"select * from {table_name}"), conn)
    df.fillna("", inplace=True)
    return df


def get_occupation_retrieval_output_dir() -> Path:
    """返回统一实验输出目录，并确保存在。"""
    OCCUPATION_RETRIEVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OCCUPATION_RETRIEVAL_OUTPUT_DIR


def get_training_output_dir() -> Path:
    """返回统一训练输出目录，并确保存在。"""
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TRAINING_OUTPUT_DIR


def resolve_output_file(name: str) -> Path:
    """解析统一实验输出文件路径，并确保输出目录存在。

    Args:
        name: 输出文件名，不允许为空。

    Returns:
        Path: 位于 `output/occupation_retrieval` 下的文件路径。
    """
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("输出文件名不能为空")
    return get_occupation_retrieval_output_dir() / normalized


def resolve_existing_model_path(path: str | Path, *, label: str) -> Path | str:
    """解析并校验模型路径，缺失时给出清晰错误。

    Args:
        path: 模型目录或 HuggingFace 模型名。
        label: 错误提示中的模型标签。

    Returns:
        Path | str: 本地模型路径，或保持原样的 HuggingFace 模型名。
    """
    raw = str(path or "").strip()
    if not raw:
        raise FileNotFoundError(f"{label} 模型路径为空")
    target = Path(raw)
    if target.exists():
        return target
    normalized = raw.replace("\\", "/")
    if (
        normalized.count("/") == 1
        and not normalized.startswith((".", "/", "output/", "models/", "checkpoints/"))
        and ":" not in normalized
    ):
        return raw
    raise FileNotFoundError(
        f"{label} 模型路径不存在: {target}. "
        "请检查 config/database.yaml、EMPLOYDATA_BGE_MODEL_PATH 或命令行 --model 参数。"
    )


def resolve_model_dir(model_name: str) -> str:
    """解析模型目录，优先新目录，其次兼容历史输出目录。"""
    candidates = [
        get_training_output_dir() / model_name,
        LEGACY_TRAINING_OUTPUT_DIR / model_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(get_training_output_dir() / model_name)


def resolve_base_model_path() -> str:
    """解析可用的基础 embedding 模型路径。"""
    candidates = [
        os.getenv("EMPLOYDATA_BGE_MODEL_PATH"),
        str(PROJECT_PATHS.bge_model_path),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "BAAI/bge-large-zh-v1.5"


def get_runtime_device() -> str:
    """返回当前推荐的运行设备。"""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def safe_empty_cuda_cache() -> None:
    """仅在 torch 可用且存在 CUDA 时清理显存。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
