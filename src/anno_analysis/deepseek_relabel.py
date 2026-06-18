"""DeepSeek 批量重标注脚本（v3：50 并发 + 数据校验 + 重试 + 错误隔离）。

对全部已标注任务，用 DeepSeek V4 Pro judge 模式对每个任务的 5 个候选独立评判，
保存原始输出 + 与人工标注的差异对比。

特性:
- 50 线程并发调用 (ThreadPoolExecutor)
- 数据完整性校验（JSON 字段齐全性、关键字段存在性）
- 失败自动重试 1 次
- 线程安全写入（threading.Lock 防竞争条件）
- 有效数据写入 raw JSONL，错误数据写入 error JSONL（隔离）
- 带时间戳和 task_id 的日志记录
- 断点续传（仅跳过已成功任务，失败任务下次重试）

用法:
    python scripts/deepseek_relabel.py                  # 全量 50 并发
    python scripts/deepseek_relabel.py --limit 500      # 500 条
    python scripts/deepseek_relabel.py --workers 100    # 100 并发
    python scripts/deepseek_relabel.py --resume         # 断点续传
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

# ---------------------------------------------------------------------------
# 日志：控制台 INFO，错误日志单独文件
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("relabel")
err_logger = logging.getLogger("relabel_err")
err_handler = logging.FileHandler(
    str(PROJECT_ROOT / "output" / "deepseek_relabel_errors.log"),
    encoding="utf-8",
)
err_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
err_logger.addHandler(err_handler)
err_logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 输出路径
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output" / "deepseek_relabel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_RAW = OUTPUT_DIR / "deepseek_relabel_raw.jsonl"
OUTPUT_DIFF = OUTPUT_DIR / "deepseek_relabel_diff.csv"
OUTPUT_PROGRESS = OUTPUT_DIR / "deepseek_relabel_progress.json"
OUTPUT_ERRORS = OUTPUT_DIR / "deepseek_relabel_errors.jsonl"

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。
你的任务是根据招聘岗位的描述，从 5 个候选职业中选择最匹配的一个。

评判原则：
1. 以岗位描述中的实际工作内容为主要判断依据，不要只看岗位名称。
2. 岗位名称中的英文缩写（如 LED、CNC、CAD、SQE、PLC、MES等）应作为技术关键词保留原意。
3. 如果你认为5个候选都不合适，请选择 "NONE"。
4. 输出必须是严格的 JSON，不要附带任何解释性文字。"""

JUDGE_USER_TEMPLATE = """请从以下 5 个候选职业中，选择与招聘岗位最匹配的一个。

【招聘岗位】
岗位名称：{job_title}
岗位要求：
{job_requirements}

【候选职业】
候选A: [{code_a}] {title_a}
候选B: [{code_b}] {title_b}
候选C: [{code_c}] {title_c}
候选D: [{code_d}] {title_d}
候选E: [{code_e}] {title_e}

请输出 JSON：
{{"best_candidate":"A"|"B"|"C"|"D"|"E"|"NONE","confidence":0.0,"reasoning":"30字内"}}"""  # noqa: E501

# 要求 JSON 中必须包含的键
REQUIRED_KEYS = {"best_candidate", "confidence", "reasoning"}
VALID_CANDIDATES = {"A", "B", "C", "D", "E", "NONE"}

# ---------------------------------------------------------------------------
# 线程安全写入器
# ---------------------------------------------------------------------------
class ThreadSafeWriter:
    """线程安全的文件追加写入器。

    使用 threading.Lock 确保多线程并发 write/flush 不会交叉，
    避免 JSONL 行损坏。
    """

    def __init__(self, path: Path, header: str = ""):
        self._lock = threading.Lock()
        self._path = path
        self._file = open(str(path), "a", encoding="utf-8")
        if header and self._path.stat().st_size == 0:
            self._file.write(header + "\n")
            self._file.flush()

    def write_line(self, text: str) -> None:
        with self._lock:
            self._file.write(text + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


# ---------------------------------------------------------------------------
# 数据加载（与旧版相同）
# ---------------------------------------------------------------------------
def load_unique_tasks(db_path: str, limit: int = 0) -> List[Dict[str, Any]]:
    conn = duckdb.connect(db_path, read_only=True)
    try:
        sql = """
            SELECT DISTINCT t.id, t.job_title, t.job_requirements,
                   t.cand_a_code, t.cand_a_title,
                   t.cand_b_code, t.cand_b_title,
                   t.cand_c_code, t.cand_c_title,
                   t.cand_d_code, t.cand_d_title,
                   t.cand_e_code, t.cand_e_title
            FROM recruit.main.label_studio_tasks_v2 t
            WHERE t.id IN (
                SELECT DISTINCT task_id FROM recruit.main.label_studio_annotations_v2
                WHERE best_candidate != ''
            )
            ORDER BY t.id
        """
        if limit > 0:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql).fetchall()
        tasks = []
        for row in rows:
            tasks.append({
                "task_id": row[0],
                "job_title": row[1] or "",
                "job_requirements": row[2] or "",
                "candidates": {
                    "A": {"code": row[3] or "", "title": row[4] or ""},
                    "B": {"code": row[5] or "", "title": row[6] or ""},
                    "C": {"code": row[7] or "", "title": row[8] or ""},
                    "D": {"code": row[9] or "", "title": row[10] or ""},
                    "E": {"code": row[11] or "", "title": row[12] or ""},
                },
            })
        return tasks
    finally:
        conn.close()


def load_annotator_choices(db_path: str) -> Dict[int, List[Dict]]:
    conn = duckdb.connect(db_path, read_only=True)
    try:
        rows = conn.execute("""
            SELECT task_id, annotator_id, best_candidate, reason
            FROM recruit.main.label_studio_annotations_v2
            WHERE best_candidate != ''
            ORDER BY task_id, annotator_id
        """).fetchall()
        result: Dict[int, List[Dict]] = {}
        for tid, aid, choice, reason in rows:
            choice_norm = choice.replace("候选", "").strip().replace("以上选项都不属于", "NONE")
            if tid not in result:
                result[tid] = []
            result[tid].append({
                "annotator_id": aid,
                "choice": choice_norm,
                "reason": (reason or "")[:100],
            })
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 进度管理（线程安全）
# ---------------------------------------------------------------------------
class ProgressTracker:
    """线程安全的进度追踪器。

    只在以下情况写入磁盘：
    - 每 N 个成功任务批量写一次（减少 IO）
    - 程序退出时最终写一次
    写入的 done_ids 仅包含已验证成功的任务 ID，失败任务不写入。
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._done: Set[int] = set()
        self._success_since_save = 0
        self._save_interval = 50

    def load(self) -> Set[int]:
        if not self._path.exists():
            return set()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            ids = set(data.get("done_ids", []))
            with self._lock:
                self._done = ids
            logger.info("已加载进度: %d 条已完成", len(ids))
            return ids
        except Exception:
            return set()

    def mark_done(self, task_id: int) -> None:
        with self._lock:
            self._done.add(task_id)
            self._success_since_save += 1
            if self._success_since_save >= self._save_interval:
                self._flush()

    def _flush(self) -> None:
        data = {"done_ids": sorted(list(self._done)), "count": len(self._done)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self._success_since_save = 0

    def finalize(self) -> None:
        with self._lock:
            self._flush()

    @property
    def count(self) -> int:
        return len(self._done)


# ---------------------------------------------------------------------------
# 数据校验
# ---------------------------------------------------------------------------
def validate_response(parsed: Dict[str, Any], task_id: int) -> Optional[str]:
    """校验 DeepSeek 返回的 JSON 数据完整性。

    检查项:
    1. 必须是 dict
    2. 必须包含 best_candidate, confidence, reasoning 三个键
    3. best_candidate 必须是 A/B/C/D/E/NONE 之一
    4. confidence 必须是 0-1 之间的数字
    5. reasoning 不能为空

    Args:
        parsed: 解析后的 JSON 字典。
        task_id: 任务 ID（仅用于错误信息）。

    Returns:
        Optional[str]: 校验失败时返回错误信息，通过时返回 None。
    """
    if not isinstance(parsed, dict) or not parsed:
        return f"Task #{task_id}: 响应不是有效的 JSON 对象"

    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        return f"Task #{task_id}: 缺少必要字段: {missing}"

    best = parsed.get("best_candidate", "")
    if best not in VALID_CANDIDATES:
        return f"Task #{task_id}: 无效的 best_candidate 值: '{best}'"

    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
        return f"Task #{task_id}: confidence 超出范围: {conf}"

    reasoning = parsed.get("reasoning", "")
    if not reasoning or not str(reasoning).strip():
        return f"Task #{task_id}: reasoning 为空"

    return None  # 校验通过


# ---------------------------------------------------------------------------
# DeepSeek API 调用器
# ---------------------------------------------------------------------------
class DeepSeekRelabeler:
    """DeepSeek 重标注器（线程安全——每个线程创建独立客户端）。"""

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self._api_key = api_key
        self._base_url = "https://api.deepseek.com"

    def _make_client(self) -> OpenAI:
        """每次调用创建新客户端（线程安全）。"""
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def relabel(self, task: Dict[str, Any], timeout_sec: int = 60) -> Dict[str, Any]:
        """对单条任务进行独立评判。

        包含: API 调用 → JSON 解析 → 数据校验 → 重试 1 次。

        Args:
            task: 含 job_title, job_requirements, candidates 的任务。
            timeout_sec: API 超时秒数。

        Returns:
            Dict: 含 best_candidate, confidence, reasoning, raw_response, is_valid。
        """
        tid = task["task_id"]
        c = task["candidates"]
        user_prompt = JUDGE_USER_TEMPLATE.format(
            job_title=task["job_title"],
            job_requirements=str(task["job_requirements"])[:3000],
            code_a=c["A"]["code"], title_a=c["A"]["title"],
            code_b=c["B"]["code"], title_b=c["B"]["title"],
            code_c=c["C"]["code"], title_c=c["C"]["title"],
            code_d=c["D"]["code"], title_d=c["D"]["title"],
            code_e=c["E"]["code"], title_e=c["E"]["title"],
        )

        # ---- 尝试 1 ----
        raw, error = self._call_api(user_prompt, timeout_sec)
        if error:
            err_logger.warning("Task #%d 首次调用失败: %s, 重试中...", tid, error)
            time.sleep(1.0)
            raw, error = self._call_api(user_prompt, timeout_sec)

        if error:
            err_logger.error("Task #%d 重试后仍失败: %s", tid, error)
            return {
                "best_candidate": "API_ERROR", "confidence": 0,
                "reasoning": error, "raw_response": raw or "",
                "is_valid": False, "error": error,
            }

        # ---- 解析 + 校验 ----
        parsed = self._parse(raw or "")
        validation_error = validate_response(parsed, tid)
        if validation_error:
            err_logger.warning("Task #%d 校验失败: %s", tid, validation_error)
            return {
                "best_candidate": "VALIDATION_ERROR", "confidence": 0,
                "reasoning": validation_error, "raw_response": raw or "",
                "is_valid": False, "error": validation_error,
            }

        return {
            "best_candidate": parsed["best_candidate"],
            "confidence": float(parsed.get("confidence", 0)),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
            "raw_response": raw or "",
            "is_valid": True, "error": None,
        }

    def _call_api(self, user_prompt: str, timeout_sec: int = 60) -> Tuple[Optional[str], Optional[str]]:
        """调用 DeepSeek API，返回 (raw_text, error)。

        DeepSeek V4 Pro 是推理模型，可能在 reasoning_content 中返回 JSON
        而 content 为空。本方法会优先使用 content，回退到 reasoning_content。

        Args:
            user_prompt: 用户 prompt。
            timeout_sec: 超时秒数。

        Returns:
            tuple: (原始响应文本或 None, 错误信息或 None)。
        """
        try:
            client = self._make_client()
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=5120,
                timeout=timeout_sec,
            )
            choice = resp.choices[0]
            message = choice.message
            content = (getattr(message, "content", None) or "").strip()
            reasoning = (getattr(message, "reasoning_content", None) or "").strip()

            # DeepSeek V4 Pro 推理模型可能将结果放在 reasoning_content 中
            raw = content or reasoning
            if not raw:
                return "", (
                    "API 返回空响应 (content 和 reasoning_content 均为空)"
                    f" finish_reason={getattr(choice, 'finish_reason', None)}"
                )
            if not content and reasoning:
                logger.debug("Task 使用 reasoning_content 回退 (len=%d)", len(reasoning))
            if getattr(choice, "finish_reason", None) == "length":
                return raw, "响应被截断 (finish_reason=length)"
            return raw, None
        except Exception as exc:
            tb = traceback.format_exc()
            err_logger.debug("API 调用异常详情:\n%s", tb)
            return None, f"{type(exc).__name__}: {str(exc)[:200]}"

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        """从 LLM 原始输出中提取 JSON。

        处理多种回退场景：
        1. 纯 JSON（理想情况）
        2. ```json ... ``` 代码块
        3. 嵌在 reasoning 文本中的 JSON 对象（含嵌套花括号）
        4. best_candidate 字段正则回退提取
        """
        text = raw.strip()
        # 去除各种 markdown 代码块标记
        for marker in ("```json", "```"):
            text = text.replace(marker, "")
        text = text.strip()

        # 尝试直接解析
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "best_candidate" in result:
                return result
        except json.JSONDecodeError:
            pass

        # 尝试提取最外层的 JSON 对象（支持嵌套花括号）
        # 找第一个 { 和对应的最后一个 }
        start = text.find("{")
        if start >= 0:
            # 从末尾反向找最后一个 }
            end = text.rfind("}")
            if end > start:
                try:
                    result = json.loads(text[start:end + 1])
                    if isinstance(result, dict) and "best_candidate" in result:
                        return result
                except json.JSONDecodeError:
                    pass

        # 正则尝试: 找含 best_candidate 的最小 JSON 对象
        for pattern in [r'\{"best_candidate"[^}]*\}', r'\{[^}]*"best_candidate"[^}]*\}']:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    continue

        # 最后回退: 正则提取 best_candidate 值
        m = re.search(r'"best_candidate"\s*:\s*"([^"]+)"', text)
        if m:
            conf_m = re.search(r'"confidence"\s*:\s*(0?\.\d+)', text)
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            return {
                "best_candidate": m.group(1),
                "confidence": float(conf_m.group(1)) if conf_m else 0.0,
                "reasoning": reason_m.group(1)[:200] if reason_m else text[:100],
            }
        return {}


# ---------------------------------------------------------------------------
# 并发批量处理
# ---------------------------------------------------------------------------
def process_single_task(
    task: Dict[str, Any],
    relabeler: DeepSeekRelabeler,
) -> Dict[str, Any]:
    """处理单个任务（供线程池 executor.submit 调用）。

    Args:
        task: 任务字典。
        relabeler: DeepSeek 重标注器。

    Returns:
        Dict: 含 task_id, ds_result, timestamp 的完整记录。
    """
    tid = task["task_id"]
    ds = relabeler.relabel(task)
    return {
        "task_id": tid,
        "job_title": task["job_title"],
        "deepseek_choice": ds["best_candidate"],
        "deepseek_confidence": ds["confidence"],
        "deepseek_reasoning": ds["reasoning"],
        "deepseek_raw_response": ds.get("raw_response", ""),
        "is_valid": ds.get("is_valid", False),
        "error": ds.get("error"),
        "candidates": task["candidates"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class ConsecutiveFailureError(Exception):
    """连续失败熔断异常。"""
    pass


def run_concurrent(
    tasks: List[Dict],
    annotator_choices: Dict[int, List[Dict]],
    relabeler: DeepSeekRelabeler,
    progress: ProgressTracker,
    workers: int = 50,
    timeout_per_task: int = 90,
    max_consecutive_failures: int = 3,
) -> Tuple[int, int]:
    """50 线程并发执行 DeepSeek 重标注。

    并发控制策略:
    - ThreadPoolExecutor(max_workers=50): I/O 密集型任务，线程数设高以最大化吞吐
    - as_completed 按完成顺序处理结果，不阻塞已完成的任务
    - 每个 future 有独立超时 (timeout_per_task)，超时任务不阻塞整体进度
    - 线程安全写入: raw_writer / err_writer / diff_writer 各持独立 Lock

    错误处理逻辑:
    - API_ERROR / VALIDATION_ERROR → 写入 error JSONL，不计入 progress
    - 空响应 → 同上
    - 网络超时 → 已被 relabel() 内部重试捕获，仍失败则记录
    - 写入异常 → try-catch + err_logger，不中断其他任务

    Args:
        tasks: 待处理任务列表。
        annotator_choices: 人工标注选择。
        relabeler: DeepSeek 重标注器。
        progress: 进度追踪器。
        workers: 并发线程数 (默认 50)。
        timeout_per_task: 单任务超时秒数。

    Returns:
        tuple: (成功数, 失败数)。
    """
    done_ids = progress.load()
    pending = [t for t in tasks if t["task_id"] not in done_ids]
    if not pending:
        logger.info("所有任务已完成，无需处理")
        return 0, 0

    logger.info("总任务: %d, 已完成: %d, 待处理: %d (并发: %d)",
                 len(tasks), len(done_ids), len(pending), workers)

    # 打开线程安全写入器
    raw_writer = ThreadSafeWriter(OUTPUT_RAW)
    err_writer = ThreadSafeWriter(OUTPUT_ERRORS,
        "task_id,timestamp,error_type,error_message,raw_response")
    # diff CSV 头
    diff_header = ("task_id,job_title,annotator_choices,deepseek_choice,"
                   "deepseek_confidence,deepseek_reasoning,agreement,"
                   "deepseek_selected_title,deepseek_selected_code")
    diff_writer = ThreadSafeWriter(OUTPUT_DIFF, diff_header)

    success_count = 0
    fail_count = 0
    submitted = 0
    completed = 0
    # 连续失败计数器 + 锁（防止并发更新竞态）
    _fail_lock = threading.Lock()
    consecutive_fails = 0

    def _on_success() -> None:
        nonlocal consecutive_fails
        with _fail_lock:
            consecutive_fails = 0

    def _on_failure() -> bool:
        """返回 True 表示应触发熔断。"""
        nonlocal consecutive_fails
        with _fail_lock:
            consecutive_fails += 1
            if consecutive_fails >= max_consecutive_failures:
                return True
            return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_map = {}
        for task in pending:
            future = executor.submit(process_single_task, task, relabeler)
            future_map[future] = task["task_id"]
            submitted += 1

        # 按完成顺序处理结果
        for future in as_completed(future_map):
            tid = future_map[future]
            completed += 1

            # 获取结果（带超时保护）
            try:
                record = future.result(timeout=timeout_per_task)
            except FutureTimeout:
                err_logger.error("Task #%d: future 超时 (>%ds)", tid, timeout_per_task)
                err_writer.write_line(json.dumps({
                    "task_id": tid, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_type": "timeout", "error": f"Future 超时 >{timeout_per_task}s",
                    "raw_response": "",
                }, ensure_ascii=False))
                fail_count += 1
                continue
            except Exception as exc:
                err_logger.error("Task #%d: future 异常: %s", tid, exc)
                err_writer.write_line(json.dumps({
                    "task_id": tid, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_type": "exception", "error": str(exc), "raw_response": "",
                }, ensure_ascii=False))
                fail_count += 1
                continue

            # ---- 有效性判断与分流写入 + 熔断 ----
            if not record["is_valid"]:
                fail_count += 1
                err_writer.write_line(json.dumps({
                    "task_id": tid,
                    "timestamp": record["timestamp"],
                    "error_type": record.get("deepseek_choice", "unknown"),
                    "error": record.get("error", ""),
                    "raw_response": record.get("deepseek_raw_response", ""),
                }, ensure_ascii=False))
                if _on_failure():
                    logger.critical(
                        "熔断触发: 连续 %d 次失败, task #%d 失败类型: %s",
                        max_consecutive_failures, tid,
                        record.get("deepseek_choice", "unknown"),
                    )
                    break  # 跳出 as_completed 循环
            else:
                _on_success()
                raw_writer.write_line(json.dumps(record, ensure_ascii=False))
                success_count += 1
                progress.mark_done(tid)
                _write_diff(record, annotator_choices, diff_writer)

            if completed % 100 == 0:
                pct = completed * 100 / submitted
                logger.info("进度: %d/%d (%.1f%%), 成功 %d, 连续失败 %d",
                             completed, submitted, pct, success_count, consecutive_fails)

    # 收尾（熔断时取消剩余未完成的 future）
    if consecutive_fails >= max_consecutive_failures:
        cancelled = 0
        for future in future_map:
            if not future.done():
                future.cancel()
                cancelled += 1
        logger.warning("熔断: 已取消 %d 个未完成任务", cancelled)

    raw_writer.close()
    err_writer.close()
    diff_writer.close()
    progress.finalize()
    logger.info("并发处理完成: 成功 %d, 失败 %d (总计 %d)",
                 success_count, fail_count, submitted)
    return success_count, fail_count


def _write_diff(
    record: Dict,
    annotator_choices: Dict[int, List[Dict]],
    writer: ThreadSafeWriter,
) -> None:
    """生成并写入一条差异记录（线程安全）。

    只在 DS 选择与标注员多数票不一致时写入。

    Args:
        record: DeepSeek 标注记录。
        annotator_choices: 人工标注选择。
        writer: 线程安全写入器。
    """
    tid = record["task_id"]
    human = annotator_choices.get(tid, [])
    human_choices = [h["choice"] for h in human]
    human_majority = Counter(human_choices).most_common(1)[0][0] if human_choices else "?"

    ds_choice = record["deepseek_choice"]
    if ds_choice == human_majority:
        return  # 一致，不写差异

    agreement = "disagree" if ds_choice != "NONE" else "ds_none"
    if ds_choice in ("API_ERROR", "VALIDATION_ERROR"):
        agreement = "error"

    ds_title = ""
    ds_code = ""
    if ds_choice in "ABCDE":
        ds_title = record["candidates"].get(ds_choice, {}).get("title", "")
        ds_code = record["candidates"].get(ds_choice, {}).get("code", "")

    ann_str = " | ".join(
        "#{}:{}".format(h["annotator_id"], h["choice"]) for h in human[:5]
    )
    line = (
        f'{tid},"{record["job_title"]}",'
        f'"{ann_str}",'
        f'{ds_choice},{record["deepseek_confidence"]:.2f},'
        f'"{record["deepseek_reasoning"]}",{agreement},'
        f'"{ds_title}","{ds_code}"'
    )
    writer.write_line(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DeepSeek 批量重标注 (v3: 50并发)")
    parser.add_argument("--limit", type=int, default=0, help="限制条数 (0=全部)")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--workers", type=int, default=1, help="并发线程数 (默认 1)")
    parser.add_argument("--timeout", type=int, default=90, help="单任务超时秒数")
    parser.add_argument("--max-fail", type=int, default=3,
                        help="连续失败熔断阈值 (默认 3, 设为 0 禁用)")
    args = parser.parse_args()

    db_path = str(PROJECT_ROOT / "output" / "recruit.duckdb")

    # 1. 加载数据
    logger.info("加载 unique 任务...")
    tasks = load_unique_tasks(db_path, args.limit)
    logger.info("加载标注员选择...")
    annotator_choices = load_annotator_choices(db_path)
    logger.info("任务数: %d, 有标注员的: %d", len(tasks), len(annotator_choices))

    # 2. 初始化
    relabeler = DeepSeekRelabeler()
    progress = ProgressTracker(OUTPUT_PROGRESS)

    # 3. 并发处理
    success, fail = run_concurrent(
        tasks=tasks,
        annotator_choices=annotator_choices,
        relabeler=relabeler,
        progress=progress,
        workers=args.workers,
        timeout_per_task=args.timeout,
        max_consecutive_failures=args.max_fail if args.max_fail > 0 else 999999,
    )

    # 4. 摘要
    print("\n" + "=" * 60)
    print("DeepSeek 重标注完成 (v3 并发)")
    if fail > 0 and fail >= args.max_fail:
        print("  *** 熔断触发: 连续失败 >= %d 次, 程序已中断 ***" % args.max_fail)
    print("=" * 60)
    print(f"  总任务数:    {len(tasks)}")
    print(f"  本次成功:    {success}")
    print(f"  本次失败:    {fail}")
    print(f"  累计完成:    {progress.count}")
    print(f"  并发数:      {args.workers}")
    print(f"  原始输出:    {OUTPUT_RAW}")
    print(f"  错误日志:    {OUTPUT_ERRORS}")
    print(f"  差异文件:    {OUTPUT_DIFF}")


if __name__ == "__main__":
    main()
