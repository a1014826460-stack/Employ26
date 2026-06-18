"""用 DeepSeek 对第二轮 Label Studio 任务做检验性标注。

本脚本面向 `annotations.label_studio_tasks_v2` 的全量第二轮任务，
以 `annotations.deepseek_relabel_raw` 作为正式结果表，支持断点续跑。

运行示例:
    python -m src.anno_analysis.deepseek_round2_relabel --dry-run
    python -m src.anno_analysis.deepseek_round2_relabel --limit 100 --workers 2
    python -m src.anno_analysis.deepseek_round2_relabel --workers 8

环境变量:
    DEEPSEEK_API_KEY: DeepSeek API key。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text

from config.paths import get_project_paths
from src.db.postgres import create_pg_engine, get_table_columns, table_exists
from src.occupation_retrieval.datasets import get_majority_choice, parse_choice

PROJECT_PATHS = get_project_paths()
PROJECT_ROOT = PROJECT_PATHS.project_root
OUTPUT_DIR = PROJECT_PATHS.output_dir / "deepseek_relabel" / "round2"
LOG_DIR = PROJECT_ROOT / "logs"

TASK_TABLE = "annotations.label_studio_tasks_v2"
DEEPSEEK_TABLE = "annotations.deepseek_relabel_raw"

RAW_JSONL = OUTPUT_DIR / "round2_deepseek_relabel_raw.jsonl"
ERROR_JSONL = OUTPUT_DIR / "round2_deepseek_relabel_errors.jsonl"
DIFF_CSV = OUTPUT_DIR / "round2_deepseek_relabel_diff.csv"
PROGRESS_JSON = OUTPUT_DIR / "round2_deepseek_relabel_progress.json"
COVERAGE_MD = OUTPUT_DIR / "round2_deepseek_relabel_coverage.md"
MISSING_JSON = OUTPUT_DIR / "round2_deepseek_relabel_missing_task_ids.json"
RUN_LOG = LOG_DIR / "deepseek_round2_relabel.log"
ERROR_LOG = LOG_DIR / "deepseek_round2_relabel_errors.log"

VALID_CHOICES = {"A", "B", "C", "D", "E", "NONE"}
REQUIRED_KEYS = {"best_candidate", "confidence", "reasoning"}
REQUIRED_TABLE_COLUMNS = {
    "recruitment_record_id": "text",
    "job_title": "text",
    "deepseek_choice": "text",
    "deepseek_confidence": "double precision",
    "deepseek_reasoning": "text",
    "deepseek_raw_response": "text",
    "candidates": "jsonb",
    "payload": "jsonb",
}

JUDGE_SYSTEM_PROMPT = """你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。
你的任务是根据招聘岗位的实际工作内容，从 5 个候选职业中选择最匹配的一个。

评判原则：
1. 以岗位描述和任职要求中的实际工作内容为主，不要只看岗位名称。
2. 英文缩写、行业术语和技术词应保留原意并参与判断。
3. 如果 5 个候选都不合适，选择 "NONE"。
4. 只输出严格 JSON，不要输出 Markdown 或额外解释。"""

JUDGE_USER_TEMPLATE = """请从以下 5 个候选职业中，选择与招聘岗位最匹配的一个。

【招聘岗位】
岗位名称：{job_title}
岗位描述/要求：
{job_requirements}

【候选职业】
候选A: [{code_a}] {title_a}
候选B: [{code_b}] {title_b}
候选C: [{code_c}] {title_c}
候选D: [{code_d}] {title_d}
候选E: [{code_e}] {title_e}

请输出 JSON：
{{"best_candidate":"A"|"B"|"C"|"D"|"E"|"NONE","confidence":0.0,"reasoning":"50字内说明"}}"""


@dataclass(frozen=True)
class Round2Task:
    """第二轮任务的 DeepSeek 标注输入。"""

    task_id: int
    recruitment_record_id: str
    job_title: str
    job_requirements: str
    candidates: dict[str, dict[str, str]]
    human_choices: list[str]
    human_majority: str | None


class ThreadSafeJsonlWriter:
    """线程安全 JSONL 追加写入器。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


class ProgressTracker:
    """断点续跑进度文件，仅记录成功写入 PG 的任务。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done_ids: set[int] = set()

    def load(self) -> set[int]:
        if not self.path.exists():
            return set()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.done_ids = {int(task_id) for task_id in data.get("done_task_ids", [])}
        except Exception:
            self.done_ids = set()
        return set(self.done_ids)

    def mark_done(self, task_id: int) -> None:
        with self._lock:
            self.done_ids.add(int(task_id))
            self.flush()

    def flush(self) -> None:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "done_count": len(self.done_ids),
            "done_task_ids": sorted(self.done_ids),
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_logging() -> logging.Logger:
    """配置控制台日志和文件日志。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("deepseek_round2_relabel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(RUN_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    error_handler = logging.FileHandler(ERROR_LOG, encoding="utf-8")
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.WARNING)
    logger.addHandler(error_handler)
    return logger


def ensure_deepseek_table() -> None:
    """确保正式 DeepSeek 结果表存在。"""
    engine = create_pg_engine()
    with engine.begin() as conn:
        if table_exists(conn, "annotations", "deepseek_relabel_raw"):
            existing_columns = set(get_table_columns(conn, "annotations", "deepseek_relabel_raw"))
            for column_name, column_type in REQUIRED_TABLE_COLUMNS.items():
                if column_name not in existing_columns:
                    conn.execute(
                        text(
                            f'alter table {DEEPSEEK_TABLE} add column if not exists {column_name} {column_type}'
                        )
                    )
            return
        conn.execute(text('create schema if not exists "annotations"'))
        conn.execute(
            text(
                """
                create table if not exists annotations.deepseek_relabel_raw (
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


def load_existing_deepseek_ids() -> set[int]:
    """读取 PG 中已有 DeepSeek 成功记录，用于断点续跑。"""
    ensure_deepseek_table()
    engine = create_pg_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"select task_id from {DEEPSEEK_TABLE}")).fetchall()
    return {int(row[0]) for row in rows}


def safe_json_loads(value: Any, fallback: Any) -> Any:
    """安全解析 JSON 字段。"""
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def extract_task_text(data: dict[str, Any]) -> tuple[str, str]:
    """从 Label Studio data_raw 中提取岗位名称和岗位要求。"""
    job_title = str(
        data.get("job_title")
        or data.get("title")
        or data.get("岗位名称")
        or ""
    ).strip()
    requirements = str(
        data.get("job_requirements_clean")
        or data.get("job_requirements")
        or data.get("requirements")
        or data.get("岗位要求")
        or data.get("岗位描述")
        or ""
    ).strip()
    return job_title, requirements


def extract_candidates(data: dict[str, Any]) -> dict[str, dict[str, str]]:
    """从 Label Studio data_raw 中提取 A-E 候选。"""
    candidates: dict[str, dict[str, str]] = {}
    for choice in "ABCDE":
        key = choice.lower()
        code = str(data.get(f"candidate_{key}_code") or data.get(f"cand_{key}_code") or "").strip()
        title = str(data.get(f"candidate_{key}_title") or data.get(f"cand_{key}_title") or "").strip()
        candidates[choice] = {"code": code, "title": title}
    return candidates


def parse_human_choices(annotations: list[dict[str, Any]]) -> tuple[list[str], str | None]:
    """解析人工选择和宽松多数票。"""
    choices = [choice for choice in (parse_choice(item) for item in annotations) if choice]
    if not choices:
        return [], None
    majority, _ = get_majority_choice(choices, require_strict=False)
    return choices, majority


def load_round2_tasks(*, limit: int = 0) -> list[Round2Task]:
    """从 PostgreSQL 读取第二轮任务。"""
    engine = create_pg_engine()
    sql = f"""
        select
            id as task_id,
            coalesce(recruitment_record_id, '') as recruitment_record_id,
            coalesce(annotations_completed_jsonb::text, annotations_completed) as annotations_completed,
            coalesce(data_raw_jsonb::text, data_raw) as data_raw
        from {TASK_TABLE}
        order by id
    """
    if limit > 0:
        sql += " limit :limit"

    tasks: list[Round2Task] = []
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": limit} if limit > 0 else {}).mappings()
        for row in rows:
            data = safe_json_loads(row["data_raw"], {})
            annotations = safe_json_loads(row["annotations_completed"], [])
            job_title, job_requirements = extract_task_text(data)
            candidates = extract_candidates(data)
            human_choices, human_majority = parse_human_choices(annotations)
            tasks.append(
                Round2Task(
                    task_id=int(row["task_id"]),
                    recruitment_record_id=str(row["recruitment_record_id"] or ""),
                    job_title=job_title,
                    job_requirements=job_requirements,
                    candidates=candidates,
                    human_choices=human_choices,
                    human_majority=human_majority,
                )
            )
    return tasks


def validate_task(task: Round2Task) -> str | None:
    """校验任务是否足以提交 DeepSeek。"""
    if not task.job_title and not task.job_requirements:
        return "岗位名称和岗位要求均为空"
    missing = [
        choice
        for choice, candidate in task.candidates.items()
        if not candidate["code"] or not candidate["title"]
    ]
    if missing:
        return f"候选字段缺失: {','.join(missing)}"
    return None


def validate_response(parsed: dict[str, Any], task_id: int) -> str | None:
    """校验 DeepSeek JSON 响应。"""
    if not isinstance(parsed, dict):
        return f"task_id={task_id}: 响应不是 JSON object"
    missing = REQUIRED_KEYS - set(parsed)
    if missing:
        return f"task_id={task_id}: 缺少字段 {sorted(missing)}"
    if parsed.get("best_candidate") not in VALID_CHOICES:
        return f"task_id={task_id}: best_candidate 无效: {parsed.get('best_candidate')}"
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        return f"task_id={task_id}: confidence 无效: {confidence}"
    if not str(parsed.get("reasoning", "")).strip():
        return f"task_id={task_id}: reasoning 为空"
    return None


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """从模型原始响应中解析 JSON。"""
    text_value = raw_text.strip()
    text_value = text_value.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text_value)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text_value, flags=re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


class DeepSeekClient:
    """DeepSeek 检验性标注客户端。"""

    def __init__(self, *, model: str, timeout: int):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self.model = model
        self.timeout = timeout
        self._api_key = api_key

    def _client(self) -> OpenAI:
        return OpenAI(api_key=self._api_key, base_url="https://api.deepseek.com")

    def relabel(self, task: Round2Task) -> dict[str, Any]:
        """调用 DeepSeek 并返回已校验的记录。"""
        user_prompt = JUDGE_USER_TEMPLATE.format(
            job_title=task.job_title,
            job_requirements=task.job_requirements[:3000],
            code_a=task.candidates["A"]["code"],
            title_a=task.candidates["A"]["title"],
            code_b=task.candidates["B"]["code"],
            title_b=task.candidates["B"]["title"],
            code_c=task.candidates["C"]["code"],
            title_c=task.candidates["C"]["title"],
            code_d=task.candidates["D"]["code"],
            title_d=task.candidates["D"]["title"],
            code_e=task.candidates["E"]["code"],
            title_e=task.candidates["E"]["title"],
        )
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
            timeout=self.timeout,
        )
        choice = response.choices[0]
        message = choice.message
        raw = (getattr(message, "content", None) or "").strip()
        if not raw:
            raw = (getattr(message, "reasoning_content", None) or "").strip()
        parsed = parse_json_response(raw)
        error = validate_response(parsed, task.task_id)
        if error:
            raise ValueError(error)
        return build_record(task, parsed, raw)


def build_record(task: Round2Task, parsed: dict[str, Any], raw_response: str) -> dict[str, Any]:
    """构造统一 DeepSeek 记录。"""
    payload = {
        "task_id": task.task_id,
        "recruitment_record_id": task.recruitment_record_id,
        "job_title": task.job_title,
        "deepseek_choice": parsed["best_candidate"],
        "deepseek_confidence": float(parsed["confidence"]),
        "deepseek_reasoning": str(parsed["reasoning"])[:300],
        "deepseek_raw_response": raw_response,
        "candidates": task.candidates,
        "human_choices": task.human_choices,
        "human_majority": task.human_majority,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def upsert_deepseek_record(record: dict[str, Any]) -> None:
    """把有效 DeepSeek 记录写入正式 PG 表。"""
    ensure_deepseek_table()
    sql = text(
        f"""
        insert into {DEEPSEEK_TABLE} (
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
    engine = create_pg_engine()
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "task_id": record["task_id"],
                "recruitment_record_id": record.get("recruitment_record_id", ""),
                "job_title": record["job_title"],
                "deepseek_choice": record["deepseek_choice"],
                "deepseek_confidence": record["deepseek_confidence"],
                "deepseek_reasoning": record["deepseek_reasoning"],
                "deepseek_raw_response": record["deepseek_raw_response"],
                "candidates": json.dumps(record["candidates"], ensure_ascii=False),
                "payload": json.dumps(record, ensure_ascii=False),
            },
        )


def append_diff(record: dict[str, Any], lock: threading.Lock) -> None:
    """在 DeepSeek 与人工多数票不一致时追加差异 CSV。"""
    human_majority = record.get("human_majority")
    deepseek_choice = record["deepseek_choice"]
    if not human_majority or human_majority == deepseek_choice:
        return

    DIFF_CSV.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        write_header = not DIFF_CSV.exists() or DIFF_CSV.stat().st_size == 0
        with DIFF_CSV.open("a", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            if write_header:
                writer.writerow(
                    [
                        "task_id",
                        "recruitment_record_id",
                        "job_title",
                        "human_majority",
                        "human_choices",
                        "deepseek_choice",
                        "deepseek_confidence",
                        "deepseek_reasoning",
                    ]
                )
            writer.writerow(
                [
                    record["task_id"],
                    record.get("recruitment_record_id", ""),
                    record["job_title"],
                    human_majority,
                    "|".join(record.get("human_choices", [])),
                    deepseek_choice,
                    f"{float(record['deepseek_confidence']):.4f}",
                    record["deepseek_reasoning"],
                ]
            )


def process_task(
    task: Round2Task,
    *,
    client: DeepSeekClient,
    retries: int,
    retry_sleep: float,
) -> tuple[bool, dict[str, Any]]:
    """处理单个任务，失败时返回错误记录。"""
    task_error = validate_task(task)
    if task_error:
        return False, {
            "task_id": task.task_id,
            "error_type": "invalid_task",
            "error": task_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    last_error = ""
    last_traceback = ""
    for attempt in range(retries + 1):
        try:
            return True, client.relabel(task)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_traceback = traceback.format_exc(limit=5)
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))

    return False, {
        "task_id": task.task_id,
        "error_type": "deepseek_call_failed",
        "error": last_error,
        "traceback": last_traceback,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_coverage_report(total_tasks: int) -> dict[str, int]:
    """写出 DeepSeek 覆盖报告。"""
    ensure_deepseek_table()
    engine = create_pg_engine()
    with engine.connect() as conn:
        covered = conn.execute(
            text(
                f"""
                select count(*)
                from {TASK_TABLE} t
                join {DEEPSEEK_TABLE} d on d.task_id = t.id
                """
            )
        ).scalar_one()
        missing_rows = conn.execute(
            text(
                f"""
                select t.id
                from {TASK_TABLE} t
                left join {DEEPSEEK_TABLE} d on d.task_id = t.id
                where d.task_id is null
                order by t.id
                """
            )
        ).fetchall()

    missing_ids = [int(row[0]) for row in missing_rows]
    MISSING_JSON.write_text(
        json.dumps({"missing_count": len(missing_ids), "missing_task_ids": missing_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    coverage = covered / total_tasks * 100 if total_tasks else 0.0
    COVERAGE_MD.write_text(
        "\n".join(
            [
                "# 第二轮 DeepSeek 检验性标注覆盖报告",
                "",
                f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- 第二轮任务总数: {total_tasks}",
                f"- 已有 DeepSeek 标注: {covered}",
                f"- 缺失 DeepSeek 标注: {len(missing_ids)}",
                f"- 覆盖率: {coverage:.2f}%",
                "",
                f"缺失任务 ID 已写入 `{MISSING_JSON.name}`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"total": total_tasks, "covered": int(covered), "missing": len(missing_ids)}


def count_total_tasks() -> int:
    """统计第二轮任务总数。"""
    engine = create_pg_engine()
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(f"select count(*) from {TASK_TABLE}")
            ).scalar_one()
        )


def run(args: argparse.Namespace) -> None:
    """执行第二轮 DeepSeek 检验性标注。"""
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)
    logger = configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_deepseek_table()

    tasks = load_round2_tasks(limit=args.limit)
    existing_ids = load_existing_deepseek_ids()
    progress = ProgressTracker(PROGRESS_JSON)
    progress_ids = progress.load()
    done_ids = existing_ids | progress_ids
    pending = [task for task in tasks if args.force or task.task_id not in done_ids]

    logger.info("第二轮任务数: %d", len(tasks))
    logger.info("PG 已有 DeepSeek 记录: %d", len(existing_ids))
    logger.info("进度文件已完成记录: %d", len(progress_ids))
    logger.info("待处理任务数: %d", len(pending))

    if args.dry_run:
        coverage = write_coverage_report(total_tasks=count_total_tasks())
        logger.info("dry-run 覆盖情况: %s", coverage)
        return

    client = DeepSeekClient(model=args.model, timeout=args.timeout)
    raw_writer = ThreadSafeJsonlWriter(RAW_JSONL)
    error_writer = ThreadSafeJsonlWriter(ERROR_JSONL)
    diff_lock = threading.Lock()

    success = 0
    failed = 0
    consecutive_failures = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    process_task,
                    task,
                    client=client,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                ): task
                for task in pending
            }
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                ok, payload = future.result()
                if ok:
                    upsert_deepseek_record(payload)
                    raw_writer.write(payload)
                    append_diff(payload, diff_lock)
                    success += 1
                    progress.mark_done(task.task_id)
                    consecutive_failures = 0
                else:
                    error_writer.write(payload)
                    failed += 1
                    consecutive_failures += 1
                    logger.warning("task_id=%s 失败: %s", task.task_id, payload.get("error"))
                    if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
                        logger.error("连续失败达到阈值，停止提交剩余结果。")
                        break

                if index % args.progress_interval == 0:
                    logger.info("进度: %d/%d, 成功=%d, 失败=%d", index, len(pending), success, failed)
    finally:
        raw_writer.close()
        error_writer.close()
        progress.flush()

    coverage = write_coverage_report(total_tasks=count_total_tasks())
    logger.info("完成: 本次成功=%d, 本次失败=%d, 覆盖=%s", success, failed, coverage)
    logger.info("过程文件目录: %s", OUTPUT_DIR)
    logger.info("日志文件: %s", RUN_LOG)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="第二轮任务 DeepSeek 检验性标注")
    parser.add_argument("--limit", type=int, default=0, help="限制读取任务数，0 表示全量")
    parser.add_argument("--workers", type=int, default=2, help="并发数，建议从 2-8 开始")
    parser.add_argument("--timeout", type=int, default=90, help="单次 API 超时秒数")
    parser.add_argument("--retries", type=int, default=1, help="失败重试次数")
    parser.add_argument("--retry-sleep", type=float, default=1.5, help="重试等待秒数")
    parser.add_argument("--model", default="deepseek-v4-pro", help="DeepSeek 模型名")
    parser.add_argument("--force", action="store_true", help="忽略 PG 断点，强制重跑并覆盖")
    parser.add_argument("--dry-run", action="store_true", help="只检查覆盖和待处理数量，不调用 API")
    parser.add_argument("--progress-interval", type=int, default=100, help="每 N 条打印一次进度")
    parser.add_argument("--max-consecutive-failures", type=int, default=0, help="连续失败熔断阈值，0 表示禁用")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
