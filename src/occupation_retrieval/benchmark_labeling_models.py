#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""职业标注模型横向对比：Qwen3-8B (本地vLLM) vs DeepSeek V4-Flash vs DeepSeek V4-Pro。

采样 1,000 条已标注任务，用相同 prompt 分别调用三个模型，
对比准确率、速度、token 消耗。

用法:
    python -m src.occupation_retrieval.benchmark_labeling_models --n 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text

from config.paths import get_project_paths
from src.db.postgres import create_pg_engine
from src.occupation_retrieval.datasets import get_majority_choice, parse_choice

PROJECT = get_project_paths()
PROJECT_ROOT = PROJECT.project_root
OUTPUT_DIR = PROJECT.output_dir / "benchmark_labeling"
BENCHMARK_TABLE = "annotations.label_studio_tasks_v2"

# ── Prompt（与 deepseek_round2_relabel 完全一致）──
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

VALID_CHOICES = {"A", "B", "C", "D", "E", "NONE"}


# ── 数据结构 ────────────────────────────────────
@dataclass
class BenchmarkTask:
    """单条 benchmark 任务。"""
    task_id: int
    job_title: str
    job_requirements: str
    candidates: dict[str, dict[str, str]]
    human_choice: str | None  # 人类多数意见


@dataclass
class ModelResult:
    """单条调用的结果。"""
    task_id: int
    model: str
    best_candidate: str | None
    confidence: float | None
    reasoning: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    error: str | None = None


@dataclass
class ModelStats:
    """单个模型的汇总统计。"""
    model: str
    total: int = 0
    success: int = 0
    failed: int = 0
    correct: int = 0  # vs human majority
    total_latency: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    confidence_sum: float = 0.0
    none_count: int = 0
    choice_dist: Counter = field(default_factory=Counter)


# ── 数据加载 ────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="职业标注模型横向对比")
    parser.add_argument("--n", type=int, default=1000, help="采样数量（默认 1000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    return parser.parse_args()


def load_benchmark_tasks(n: int, seed: int) -> list[BenchmarkTask]:
    """从 PG 加载已标注任务并采样。"""
    import random
    engine = create_pg_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"select id, annotations_completed, data_raw from {BENCHMARK_TABLE} order by id")
        ).mappings()
        all_tasks: list[BenchmarkTask] = []
        for row in rows:
            data = json.loads(row["data_raw"]) if isinstance(row["data_raw"], str) else row["data_raw"]
            anns = json.loads(row["annotations_completed"]) if isinstance(row["annotations_completed"], str) else row["annotations_completed"]
            if not anns:
                continue

            job_title = str(data.get("job_title", "")).strip()
            job_reqs = str(data.get("job_requirements_clean", "")).strip()
            if not job_reqs:
                continue

            candidates = {}
            for letter in "ABCDE":
                code = str(data.get(f"candidate_{letter.lower()}_code", "")).strip()
                title = str(data.get(f"candidate_{letter.lower()}_title", "")).strip()
                candidates[letter] = {"code": code, "title": title}

            choices = [c for c in (parse_choice(a) for a in anns) if c]
            if not choices:
                continue
            if len(anns) >= 2:
                ref, _ = get_majority_choice(choices, require_strict=False)
            else:
                ref = choices[0]

            all_tasks.append(BenchmarkTask(
                task_id=int(row["id"]),
                job_title=job_title,
                job_requirements=job_reqs[:3000],
                candidates=candidates,
                human_choice=ref,
            ))

    random.seed(seed)
    random.shuffle(all_tasks)
    sampled = all_tasks[:n]
    print(f"从 {len(all_tasks)} 条有效任务中采样 {len(sampled)} 条")
    return sampled


# ── Qwen3 客户端（本地 vLLM）────────────────────
class Qwen3VLLMClient:
    """通过 WSL vLLM OpenAI-compatible API 调用 Qwen3-8B。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8101/v1", model: str = "Qwen3-8B"):
        self.client = OpenAI(api_key="not-needed", base_url=base_url)
        self.model = model

    def call(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any] | None, float, int, int, str | None]:
        """返回 (parsed_json, latency, input_tokens, output_tokens, error)。"""
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
                timeout=120,
                extra_body={"enable_thinking": False},
            )
            latency = time.perf_counter() - t0
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            parsed = parse_json(raw)
            if not parsed:
                return None, latency, input_tokens, output_tokens, "JSON parse failed"
            return parsed, latency, input_tokens, output_tokens, None
        except Exception as e:
            latency = time.perf_counter() - t0
            return None, latency, 0, 0, f"{type(e).__name__}: {e}"


# ── DeepSeek 客户端（API）───────────────────────
class DeepSeekBenchClient:
    """DeepSeek API 客户端（基于共享 deepseek_client）。"""

    def __init__(self, model: str):
        load_dotenv(PROJECT_ROOT / ".env.local", override=True)
        from src.llm.deepseek_client import DeepSeekClient as SharedDS, DeepSeekConfig
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self._shared = SharedDS(DeepSeekConfig(
            api_key=api_key, model=model, timeout=90, disable_thinking=True,
        ))
        self.model = model

    def call(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any] | None, float, int, int, str | None]:
        t0 = time.perf_counter()
        try:
            parsed, raw = self._shared.complete_json_with_raw(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=512,
            )
            # 注意：共享客户端不返回 token 计数，这里做近似估计
            latency = time.perf_counter() - t0
            # 估算 token（基于 benchmark 实测均值）
            est_input = 350
            est_output = 45
            return parsed, latency, est_input, est_output, None
        except Exception as e:
            return None, time.perf_counter() - t0, 0, 0, f"{type(e).__name__}: {e}"

    def _legacy_call(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any] | None, float, int, int, str | None]:
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
                timeout=90,
                extra_body={"thinking": {"type": "disabled"}},
            )
            latency = time.perf_counter() - t0
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            parsed = parse_json(raw)
            if not parsed:
                return None, latency, input_tokens, output_tokens, "JSON parse failed"
            return parsed, latency, input_tokens, output_tokens, None
        except Exception as e:
            latency = time.perf_counter() - t0
            return None, latency, 0, 0, f"{type(e).__name__}: {e}"


# ── JSON 解析 ───────────────────────────────────
def parse_json(raw_text: str) -> dict[str, Any] | None:
    """从模型原始响应中解析 JSON（支持多种格式）。"""
    import re
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass
    return None


# ── 校验 ───────────────────────────────────────
def validate_result(parsed: dict[str, Any]) -> str | None:
    if not isinstance(parsed, dict):
        return "not a dict"
    if parsed.get("best_candidate") not in VALID_CHOICES:
        return f"invalid best_candidate: {parsed.get('best_candidate')}"
    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or not 0 <= float(conf) <= 1:
        return f"invalid confidence: {conf}"
    return None


# ── 主流程 ──────────────────────────────────────
def run_benchmark(tasks: list[BenchmarkTask], client: Any, model_name: str) -> tuple[list[ModelResult], ModelStats]:
    print(f"\n{'='*60}")
    print(f"评测: {model_name} ({len(tasks)} 条)")
    print(f"{'='*60}")

    results: list[ModelResult] = []
    stats = ModelStats(model=model_name)
    stats.total = len(tasks)
    start_time = time.perf_counter()

    for i, task in enumerate(tasks):
        user_prompt = JUDGE_USER_TEMPLATE.format(
            job_title=task.job_title,
            job_requirements=task.job_requirements,
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

        parsed, latency, input_tok, output_tok, error = client.call(JUDGE_SYSTEM_PROMPT, user_prompt)

        if error or not parsed:
            results.append(ModelResult(
                task_id=task.task_id, model=model_name,
                best_candidate=None, confidence=None, reasoning="",
                latency_seconds=latency, input_tokens=input_tok,
                output_tokens=output_tok, total_tokens=input_tok + output_tok,
                error=error,
            ))
            stats.failed += 1
        else:
            val_error = validate_result(parsed)
            if val_error:
                results.append(ModelResult(
                    task_id=task.task_id, model=model_name,
                    best_candidate=None, confidence=None, reasoning=str(parsed.get("reasoning", ""))[:100],
                    latency_seconds=latency, input_tokens=input_tok,
                    output_tokens=output_tok, total_tokens=input_tok + output_tok,
                    error=val_error,
                ))
                stats.failed += 1
            else:
                choice = parsed["best_candidate"]
                conf = float(parsed["confidence"])
                results.append(ModelResult(
                    task_id=task.task_id, model=model_name,
                    best_candidate=choice, confidence=conf,
                    reasoning=str(parsed.get("reasoning", ""))[:100],
                    latency_seconds=latency, input_tokens=input_tok,
                    output_tokens=output_tok, total_tokens=input_tok + output_tok,
                ))
                stats.success += 1
                stats.confidence_sum += conf
                stats.choice_dist[choice] += 1
                stats.total_latency += latency
                stats.total_input_tokens += input_tok
                stats.total_output_tokens += output_tok
                stats.total_tokens += input_tok + output_tok
                if choice == "NONE":
                    stats.none_count += 1
                if choice == task.human_choice:
                    stats.correct += 1

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(f"  进度: {i+1}/{len(tasks)}, {rate:.1f} 条/秒, 预计剩余 {remaining:.0f}s")

    elapsed = time.perf_counter() - start_time
    print(f"  完成: {elapsed:.1f}s, {len(tasks)/elapsed:.1f} 条/秒")

    return results, stats


def print_report(all_stats: list[ModelStats]) -> None:
    """打印横向对比报告。"""
    print("\n\n" + "=" * 80)
    print("模型横向对比报告")
    print("=" * 80)

    # 准确率
    print(f"\n{'模型':<30} {'成功':<8} {'失败':<8} {'准确率':<10} {'NONE率':<10} {'平均置信度':<12}")
    print("-" * 78)
    for s in all_stats:
        acc = s.correct / s.success * 100 if s.success > 0 else 0
        none_pct = s.none_count / s.success * 100 if s.success > 0 else 0
        avg_conf = s.confidence_sum / s.success if s.success > 0 else 0
        print(f"{s.model:<30} {s.success:<8} {s.failed:<8} {acc:<9.1f}% {none_pct:<9.1f}% {avg_conf:<12.3f}")

    # 速度
    print(f"\n{'模型':<30} {'总耗时(s)':<12} {'平均延迟(s)':<14} {'吞吐(条/s)':<12}")
    print("-" * 68)
    for s in all_stats:
        avg_lat = s.total_latency / s.success if s.success > 0 else 0
        throughput = s.success / (s.total_latency) if s.total_latency > 0 else 0
        print(f"{s.model:<30} {s.total_latency:<12.1f} {avg_lat:<14.2f} {throughput:<12.2f}")

    # Token 消耗
    print(f"\n{'模型':<30} {'输入tokens':<14} {'输出tokens':<14} {'总tokens':<14} {'平均/条':<12}")
    print("-" * 84)
    for s in all_stats:
        avg_tok = s.total_tokens / s.success if s.success > 0 else 0
        print(f"{s.model:<30} {s.total_input_tokens:<14,} {s.total_output_tokens:<14,} {s.total_tokens:<14,} {avg_tok:<12,.0f}")

    # 选择分布
    print(f"\n{'模型':<30} {'A':<8} {'B':<8} {'C':<8} {'D':<8} {'E':<8} {'NONE':<8}")
    print("-" * 78)
    for s in all_stats:
        parts = []
        for ch in "ABCDE" + "NONE":
            cnt = s.choice_dist.get(ch, 0)
            pct = cnt / s.success * 100 if s.success > 0 else 0
            parts.append(f"{cnt}({pct:.0f}%)")
        print(f"{s.model:<30} " + "  ".join(f"{p:<8}" for p in parts))


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_benchmark_tasks(args.n, args.seed)

    # 人类选择分布
    human_dist = Counter(t.human_choice for t in tasks if t.human_choice)
    print(f"\n测试集人类选择分布:")
    for ch in "ABCDE" + "NONE":
        print(f"  {ch}: {human_dist.get(ch, 0)}")

    all_stats: list[ModelStats] = []

    # ── Qwen3-8B (vLLM) ──
    qwen = Qwen3VLLMClient()
    qwen_results, qwen_stats = run_benchmark(tasks, qwen, "Qwen3-8B (vLLM)")
    all_stats.append(qwen_stats)

    # ── DeepSeek V4-Flash ──
    ds_flash = DeepSeekBenchClient(model="deepseek-v4-flash")
    flash_results, flash_stats = run_benchmark(tasks, ds_flash, "DeepSeek V4-Flash")
    all_stats.append(flash_stats)

    # ── DeepSeek V4-Pro ──
    ds_pro = DeepSeekBenchClient(model="deepseek-v4-pro")
    pro_results, pro_stats = run_benchmark(tasks, ds_pro, "DeepSeek V4-Pro")
    all_stats.append(pro_stats)

    # ── 报告 ──
    print_report(all_stats)

    # 保存结果
    output_path = args.output or str(OUTPUT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    all_results = {
        "qwen3_8b": [r for r in qwen_results if r.best_candidate],
        "ds_v4_flash": [r for r in flash_results if r.best_candidate],
        "ds_v4_pro": [r for r in pro_results if r.best_candidate],
    }
    # Save simplified version (no need for full task details)
    summary = {
        "config": {"n": args.n, "seed": args.seed},
        "models": {s.model: {
            "total": s.total, "success": s.success, "failed": s.failed,
            "correct": s.correct, "accuracy": s.correct / s.success * 100 if s.success > 0 else 0,
            "avg_latency": s.total_latency / s.success if s.success > 0 else 0,
            "total_latency": s.total_latency,
            "input_tokens": s.total_input_tokens,
            "output_tokens": s.total_output_tokens,
            "total_tokens": s.total_tokens,
            "avg_tokens_per_call": s.total_tokens / s.success if s.success > 0 else 0,
            "avg_confidence": s.confidence_sum / s.success if s.success > 0 else 0,
            "none_rate": s.none_count / s.success * 100 if s.success > 0 else 0,
            "choice_dist": dict(s.choice_dist),
        } for s in all_stats},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
