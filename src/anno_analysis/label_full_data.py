#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量招聘数据 DeepSeek V4-Flash 标注流水线。

流程:
    1. V5 BGE 从职业大典叶子节点检索 Top-5 候选
    2. DeepSeek V4-Flash 判断最匹配职业（支持 NONE）
    3. 结果写入 PostgreSQL + 本地 JSONL
    4. 断点续跑：按 recruitment_record_id 追踪进度

用法:
    python -m src.anno_analysis.label_full_data --dry-run
    python -m src.anno_analysis.label_full_data --workers 50 --limit 10000
    python -m src.anno_analysis.label_full_data --workers 50

前置条件:
    - V5 BGE 模型已微调
    - DeepSeek API key 在 .env.local 中
    - WSL vLLM 不在用 GPU（BGE 需要 GPU 推理）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config.paths import get_project_paths
from src.db.postgres import create_pg_engine, table_exists

PROJECT = get_project_paths()
PROJECT_ROOT = PROJECT.project_root
OUTPUT_DIR = PROJECT.output_dir / "full_data_labeling"
LOG_DIR = PROJECT_ROOT / "logs"

# ── 表名 ──
SOURCE_TABLE = "public.job_description_parsed"
OCC_TABLE = "public.occ_dict_unified"
RESULT_TABLE = "public.deepseek_full_label"
QWEN3_RESULT_TABLE = "public.qwen3_full_label"

# ── 模型 ──
def _resolve_bge_model_path() -> str:
    """解析 V5 BGE 模型路径，优先通过训练输出目录查找。"""
    from src.occupation_retrieval.common import get_training_output_dir
    candidate = get_training_output_dir() / "bge-large-round2-finetuned-v5"
    if candidate.exists():
        return str(candidate)
    # 兜底：尝试 config 中的 bge 路径
    return str(PROJECT.bge_model_path)

BGE_MODEL_PATH = _resolve_bge_model_path()
DS_MODEL = "deepseek-v4-flash"
QWEN_MODEL = "Qwen2.5-3B-Instruct"
QWEN_BASE_URL = "http://127.0.0.1:8102/v1"

# ── 输出文件 ──
RAW_JSONL = OUTPUT_DIR / "full_label_raw.jsonl"
ERROR_JSONL = OUTPUT_DIR / "full_label_errors.jsonl"
PROGRESS_JSON = OUTPUT_DIR / "full_label_progress.json"
COVERAGE_MD = OUTPUT_DIR / "full_label_coverage.md"
RUN_LOG = LOG_DIR / "full_label.log"

# ── Prompt（强调 NONE）──
SYSTEM_PROMPT = """你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。
你的任务是根据招聘岗位的实际工作内容，从 5 个候选职业中选择最匹配的一个。

评判原则：
1. 以岗位描述和任职要求中的实际工作内容为主要判断依据，不要只看岗位名称。
2. 英文缩写、行业术语和技术词应保留原意并参与判断。
3. 【重要】如果 5 个候选职业都与岗位实际工作内容不匹配，请务必选择 "NONE"。
   不要强行匹配。不要为了选而选。宁可放弃也不要乱选。
4. 只输出严格 JSON，不要输出 Markdown 或额外解释。"""

USER_TEMPLATE = """请从以下 5 个候选职业中，选择与招聘岗位最匹配的一个。

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

VALID_CHOICES = {"A", "B", "C", "D", "E", "NONE"}
REQUIRED_KEYS = {"best_candidate", "confidence", "reasoning"}

# ── 结果表 DDL ──
RESULT_TABLE_DDL = """
create table if not exists {table} (
    recruitment_record_id text primary key,
    job_title text,
    requirements_text text,
    v5_bge_top5 jsonb,
    deepseek_choice text,
    deepseek_confidence double precision,
    deepseek_reasoning text,
    deepseek_raw_response text,
    created_at timestamptz default now()
)
"""

# ── 配置 ──
@dataclass
class LabelConfig:
    workers: int = 50
    limit: int = 0  # 0 = 全量
    batch_size: int = 500  # 每次从 PG 读取的批量大小
    bge_batch_size: int = 128  # BGE 编码批量大小
    ds_timeout: int = 90
    ds_retries: int = 1
    progress_interval: int = 500
    dry_run: bool = False
    save_interval: int = 100  # 每 N 条成功就刷一次进度文件


# ── 数据结构 ──
@dataclass
class LabelTask:
    recruitment_record_id: str
    job_title: str
    job_requirements: str
    rag_query: str


class ThreadSafeWriter:
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
    """断点续跑游标追踪（只存 last_rid 纯文本，PG 表为进度真相）。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0
        self._last_rid: str | None = None

    def load_count(self) -> int:
        """从 PG 加载已完成数量。"""
        return self._count

    def get_last_rid(self) -> str | None:
        """获取上次处理的最后一个 RID。"""
        if self.path.exists():
            try:
                saved = self.path.read_text(encoding="utf-8").strip()
                if saved:
                    self._last_rid = saved
            except Exception:
                pass
        return self._last_rid

    def mark(self, rid: str) -> None:
        """标记一条完成（仅内存计数，不写盘）。"""
        with self._lock:
            self._count += 1
            self._last_rid = rid

    def flush(self) -> None:
        """将游标位置写入纯文本文件（极轻量）。"""
        if self._last_rid:
            self.path.write_text(self._last_rid, encoding="utf-8")


# ── 日志 ──
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("full_label")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in [
        logging.StreamHandler(),
        logging.FileHandler(RUN_LOG, encoding="utf-8"),
    ]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# ── 数据库工具 ──
def ensure_result_table(table_name: str = RESULT_TABLE) -> None:
    """确保结果表存在。"""
    engine = create_pg_engine()
    with engine.begin() as conn:
        schema_name, tbl = table_name.split(".", 1)
        if table_exists(conn, schema_name, tbl):
            return
        conn.execute(text(RESULT_TABLE_DDL.format(table=table_name)))


def load_already_done_rids(table_name: str = RESULT_TABLE) -> set[str]:
    """从 PG 结果表加载已完成的 recruitment_record_id。"""
    engine = create_pg_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"select recruitment_record_id from {table_name}")).fetchall()
        return {str(row[0]) for row in rows}
    except Exception:
        return set()


# ── BGE 检索 ──
class BGERetriever:
    """V5 BGE 模型 + 职业大典叶子节点检索。"""

    def __init__(self):
        print("加载 V5 BGE 模型...")
        self.model = SentenceTransformer(str(PROJECT_ROOT / BGE_MODEL_PATH), device="cuda")
        self.model.max_seq_length = 256

        # 加载职业大典叶子节点
        engine = create_pg_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"select code, title, \"desc\", tasks from {OCC_TABLE} where node_type = 'occupation_leaf' order by code")
            ).mappings()
            self.codes: list[str] = []
            self.texts: list[str] = []
            for row in rows:
                code = str(row["code"]).strip()
                title = str(row["title"]).strip()
                desc = str(row.get("desc", "")).strip()
                tasks = str(row.get("tasks", "")).strip()
                parts = [title]
                if desc and desc != "nan":
                    parts.append(f"定义：{desc}")
                if tasks and tasks != "nan":
                    parts.append(f"任务：{tasks}")
                self.codes.append(code)
                self.texts.append("。".join(parts))

        # 预编码大典
        print(f"编码 {len(self.codes)} 个职业叶子节点...")
        with torch.no_grad():
            self.occ_emb = self.model.encode(
                self.texts, batch_size=128, normalize_embeddings=True,
                show_progress_bar=True, convert_to_tensor=True,
            )
        print(f"BGE 就绪")

    def retrieve_top5(self, queries: list[str]) -> list[list[tuple[str, str]]]:
        """返回每条 query 的 Top-5 候选 [(code, title), ...]."""
        with torch.no_grad():
            q_emb = self.model.encode(
                queries, batch_size=128, normalize_embeddings=True,
                show_progress_bar=False, convert_to_tensor=True,
            )
        sim = torch.mm(q_emb, self.occ_emb.T)
        _, top5_idx = torch.topk(sim, k=5, dim=1)
        top5_idx = top5_idx.cpu().tolist()

        results = []
        for idxs in top5_idx:
            candidates = [(self.codes[i], self.texts[i][:self.texts[i].index("。")] if "。" in self.texts[i] else self.texts[i][:40]) for i in idxs]
            results.append(candidates)
        return results


# ── 数据加载（游标分页，SQL 层过滤）──
def iter_tasks(
    batch_size: int,
    limit: int,
    skip_rids: set[str],
    last_processed_rid: str = "",
) -> Generator[tuple[list[LabelTask], str], None, None]:
    """从 PG 分页读取待标注数据，SQL 层过滤已完成 + 空文本。

    使用 keyset pagination (WHERE rid > last_rid) 替代 OFFSET，
    避免大偏移量时的性能退化。

    Yields:
        (batch, last_rid_in_batch): 每批数据和该批最后一个 RID。
    """
    engine = create_pg_engine()
    cursor = last_processed_rid or ""
    loaded = 0
    total_skipped = 0

    while True:
        if 0 < limit <= loaded:
            break

        with engine.connect() as conn:
            # 先查一批 RID，不包括已完成的
            rows = conn.execute(
                text(f"""
                    select recruitment_record_id, job_title,
                           coalesce(requirements_text, '') as requirements_text,
                           coalesce(rag_query_text, '') as rag_query_text
                    from {SOURCE_TABLE}
                    where recruitment_record_id > :cursor
                      and requirements_text is not null
                      and requirements_text != ''
                    order by recruitment_record_id
                    limit :limit
                """),
                {"cursor": cursor, "limit": batch_size * 2},  # 多取一些，因为要跳过已完成的
            ).mappings()

        batch: list[LabelTask] = []
        last_rid = cursor
        row_count = 0
        for row in rows:
            row_count += 1
            rid = str(row["recruitment_record_id"])
            last_rid = rid
            if rid in skip_rids:
                total_skipped += 1
                continue
            req = str(row["requirements_text"] or "").strip()
            rag = str(row["rag_query_text"] or "").strip()
            title = str(row["job_title"] or "").strip()
            batch.append(LabelTask(
                recruitment_record_id=rid,
                job_title=title,
                job_requirements=req,
                rag_query=rag,
            ))
            if len(batch) >= batch_size:
                break

        cursor = last_rid
        if row_count == 0:
            if not batch:
                break
        loaded += len(batch)
        yield batch, last_rid


# ── DeepSeek 客户端 ──
class DSFlashClient:
    """DeepSeek V4-Flash 客户端，封装共享客户端并做响应校验。"""

    def __init__(self):
        from src.llm.deepseek_client import DeepSeekClient, DeepSeekConfig
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self._shared = DeepSeekClient(DeepSeekConfig(
            api_key=api_key, model=DS_MODEL, timeout=90,
            disable_thinking=True,
        ))

    def call(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any] | None, str | None]:
        """返回 (parsed_json, error_message)。"""
        try:
            parsed, _ = self._shared.complete_json_with_raw(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=256,
            )
            missing = REQUIRED_KEYS - set(parsed.keys())
            if missing:
                return None, f"missing keys: {missing}"
            if parsed.get("best_candidate") not in VALID_CHOICES:
                return None, f"invalid choice: {parsed.get('best_candidate')}"
            conf = parsed.get("confidence")
            if not isinstance(conf, (int, float)) or not 0 <= float(conf) <= 1:
                return None, f"invalid confidence: {conf}"
            return parsed, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"


# ── Qwen3 客户端（本地 vLLM）──
class Qwen3VLLMClient:
    """Qwen3-8B 本地 vLLM 客户端（关闭 thinking）。"""

    def __init__(self):
        self.client = OpenAI(api_key="not-needed", base_url=QWEN_BASE_URL)

    def call(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            response = self.client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=256,
                timeout=120,
                extra_body={"enable_thinking": False},
            )
            raw = (response.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                import re
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    return None, f"JSON parse failed: {raw[:200]}"

            missing = REQUIRED_KEYS - set(parsed.keys())
            if missing:
                return None, f"missing keys: {missing}"
            if parsed.get("best_candidate") not in VALID_CHOICES:
                return None, f"invalid choice: {parsed.get('best_candidate')}"
            conf = parsed.get("confidence")
            if not isinstance(conf, (int, float)) or not 0 <= float(conf) <= 1:
                return None, f"invalid confidence: {conf}"
            return parsed, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"


# ── 单任务处理 ──
def process_one(
    task: LabelTask,
    top5: list[tuple[str, str]],
    client: DSFlashClient | Qwen3VLLMClient,
) -> dict[str, Any]:
    """处理单条标注任务。"""
    user_prompt = USER_TEMPLATE.format(
        job_title=task.job_title,
        job_requirements=task.job_requirements[:3000],
        code_a=top5[0][0], title_a=top5[0][1],
        code_b=top5[1][0], title_b=top5[1][1],
        code_c=top5[2][0], title_c=top5[2][1],
        code_d=top5[3][0], title_d=top5[3][1],
        code_e=top5[4][0], title_e=top5[4][1],
    )

    for attempt in range(2):  # 最多重试 1 次
        parsed, error = client.call(SYSTEM_PROMPT, user_prompt)
        if not error:
            return {
                "recruitment_record_id": task.recruitment_record_id,
                "job_title": task.job_title,
                "requirements_text": task.job_requirements,
                "v5_bge_top5": [
                    {"code": code, "title": title} for code, title in top5
                ],
                "deepseek_choice": parsed["best_candidate"],
                "deepseek_confidence": float(parsed["confidence"]),
                "deepseek_reasoning": str(parsed.get("reasoning", ""))[:200],
                "deepseek_raw_response": json.dumps(parsed, ensure_ascii=False),
                "error": None,
            }
        if attempt < 1:
            time.sleep(1.5)

    return {
        "recruitment_record_id": task.recruitment_record_id,
        "job_title": task.job_title,
        "requirements_text": task.job_requirements,
        "v5_bge_top5": [
            {"code": code, "title": title} for code, title in top5
        ],
        "deepseek_choice": None,
        "deepseek_confidence": None,
        "deepseek_reasoning": None,
        "deepseek_raw_response": None,
        "error": error,
    }


def upsert_result(record: dict[str, Any]) -> None:
    """将结果写入 PostgreSQL。"""
    sql = text(f"""
        insert into {RESULT_TABLE} (
            recruitment_record_id, job_title, requirements_text,
            v5_bge_top5, deepseek_choice, deepseek_confidence,
            deepseek_reasoning, deepseek_raw_response
        ) values (
            :rid, :job_title, :requirements_text,
            cast(:top5 as jsonb), :choice, :confidence,
            :reasoning, :raw_response
        )
        on conflict (recruitment_record_id) do update set
            job_title = excluded.job_title,
            requirements_text = excluded.requirements_text,
            v5_bge_top5 = excluded.v5_bge_top5,
            deepseek_choice = excluded.deepseek_choice,
            deepseek_confidence = excluded.deepseek_confidence,
            deepseek_reasoning = excluded.deepseek_reasoning,
            deepseek_raw_response = excluded.deepseek_raw_response
    """)
    engine = create_pg_engine()
    with engine.begin() as conn:
        conn.execute(sql, {
            "rid": record["recruitment_record_id"],
            "job_title": record["job_title"],
            "requirements_text": record["requirements_text"],
            "top5": json.dumps(record["v5_bge_top5"], ensure_ascii=False),
            "choice": record["deepseek_choice"],
            "confidence": record["deepseek_confidence"],
            "reasoning": record["deepseek_reasoning"],
            "raw_response": record["deepseek_raw_response"],
        })


# ── 主流程 ──
def run(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)
    logger = setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = LabelConfig(
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    # 1. 初始化（根据模型选择结果表）
    model_name = args.model
    result_table = QWEN3_RESULT_TABLE if model_name == "qwen3" else RESULT_TABLE
    ensure_result_table(result_table)
    pg_done = load_already_done_rids(result_table)
    progress = ProgressTracker(PROGRESS_JSON)
    # 恢复游标位置
    progress.get_last_rid()

    logger.info("PG 已完成: %d, 游标: %s", len(pg_done), progress._last_rid or "从头开始")

    if config.dry_run:
        # 估算总任务数
        engine = create_pg_engine()
        with engine.connect() as conn:
            total = conn.execute(text(f"select count(distinct recruitment_record_id) from {SOURCE_TABLE} where recruitment_record_id is not null")).scalar_one()
        remaining = total - len(pg_done)
        logger.info("全量: %d, 已完成: %d, 待处理: %d", total, len(pg_done), max(0, remaining))
        return

    # 2. 加载 BGE
    bge = BGERetriever()

    # 3. 初始化 LLM 客户端
    if model_name == "qwen3":
        llm_client = Qwen3VLLMClient()
        logger.info("使用本地 Qwen3-8B vLLM")
    else:
        llm_client = DSFlashClient()
        logger.info("使用 DeepSeek V4-Flash API")
    raw_writer = ThreadSafeWriter(RAW_JSONL)
    error_writer = ThreadSafeWriter(ERROR_JSONL)

    # 4. 流水线（持久 executor + 批量 PG 写入）
    total_processed = 0
    total_success = 0
    total_failed = 0
    start_time = time.perf_counter()
    pg_engine = create_pg_engine()  # 持久连接
    pending_records: list[dict[str, Any]] = []  # 批量写 PG 缓存

    def flush_pg(records: list[dict[str, Any]]) -> None:
        """批量写入 PostgreSQL。"""
        if not records:
            return
        sql = text(f"""
            insert into {result_table} (
                recruitment_record_id, job_title, requirements_text,
                v5_bge_top5, deepseek_choice, deepseek_confidence,
                deepseek_reasoning, deepseek_raw_response
            ) values (
                :rid, :job_title, :requirements_text,
                cast(:top5 as jsonb), :choice, :confidence,
                :reasoning, :raw_response
            )
            on conflict (recruitment_record_id) do update set
                job_title = excluded.job_title,
                requirements_text = excluded.requirements_text,
                v5_bge_top5 = excluded.v5_bge_top5,
                deepseek_choice = excluded.deepseek_choice,
                deepseek_confidence = excluded.deepseek_confidence,
                deepseek_reasoning = excluded.deepseek_reasoning,
                deepseek_raw_response = excluded.deepseek_raw_response
        """)
        with pg_engine.begin() as conn:
            for rec in records:
                conn.execute(sql, {
                    "rid": rec["recruitment_record_id"],
                    "job_title": rec["job_title"],
                    "requirements_text": rec["requirements_text"],
                    "top5": json.dumps(rec["v5_bge_top5"], ensure_ascii=False),
                    "choice": rec["deepseek_choice"],
                    "confidence": rec["deepseek_confidence"],
                    "reasoning": rec["deepseek_reasoning"],
                    "raw_response": rec["deepseek_raw_response"],
                })

    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            last_rid = progress.get_last_rid() or ""
            for batch, last_rid in iter_tasks(config.batch_size, config.limit, pg_done, last_rid):
                if not batch:
                    continue

                # 4a. BGE 检索 Top-5
                queries = [t.rag_query if t.rag_query else (f"{t.job_title} {t.job_requirements}")[:500] for t in batch]
                top5_results = bge.retrieve_top5(queries)

                # 4b. 并发调用 DS API
                futures = {
                    executor.submit(process_one, task, top5, llm_client): task
                    for task, top5 in zip(batch, top5_results)
                }
                for future in as_completed(futures):
                    record = future.result()
                    if record.get("error"):
                        error_writer.write(record)
                        total_failed += 1
                    else:
                        raw_writer.write(record)
                        pending_records.append(record)
                        progress.mark(record["recruitment_record_id"])
                        total_success += 1

                    total_processed += 1

                    # 定期批量写 PG
                    if len(pending_records) >= 200:
                        flush_pg(pending_records)
                        pending_records.clear()

                    if total_processed % config.progress_interval == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        remaining_est = (config.limit - total_processed) / rate if config.limit and rate > 0 else float("inf")
                        logger.info(
                            "进度: 处理=%d 成功=%d 失败=%d 速率=%.1f/s 预计剩余=%.0fs",
                            total_processed, total_success, total_failed, rate, remaining_est,
                        )

                # 每批结束 flush
                if pending_records:
                    flush_pg(pending_records)
                    pending_records.clear()

                # 定期刷游标到磁盘（轻量纯文本写入）
                progress.flush()

                if config.limit and total_processed >= config.limit:
                    break

    finally:
        flush_pg(pending_records)
        raw_writer.close()
        error_writer.close()
        progress.flush()

    elapsed = time.perf_counter() - start_time
    logger.info("完成: 处理=%d 成功=%d 失败=%d 耗时=%.0fs", total_processed, total_success, total_failed, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="全量招聘数据 DeepSeek Flash 标注")
    parser.add_argument("--model", type=str, default="flash", choices=["flash", "qwen3"], help="标注模型: flash (DeepSeek V4-Flash) / qwen3 (本地 Qwen3-8B vLLM)")
    parser.add_argument("--workers", type=int, default=50, help="并发数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数，0=全量")
    parser.add_argument("--dry-run", action="store_true", help="只检查状态，不调用 API")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
