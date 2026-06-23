"""Compare local Top10 second-stage selector accuracy across two local models."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import requests

from src.model_platform.config import load_model_runtime_config
from src.model_platform.llm import VLLMHTTPClient
from src.occupation_retrieval.offline_top10_qc import (
    DEFAULT_SELECTOR_TOP_K,
    DEFAULT_SELECTOR_WORKERS,
    QCSample,
    StepTimer,
    apply_llm_top10_selection,
    build_gold_samples,
    load_occupation_detail_matches,
    summarize_qc_samples,
    summarize_selector_metrics,
)
from src.utils.vllm_utils import load_vllm_config

from .common import load_annotations_from_pg, resolve_output_file

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_FILE = "local_top10_model_comparison.json"


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, str(level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


@dataclass(frozen=True)
class ModelRunConfig:
    label: str
    vllm_config_path: Path


def _resolve_output_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return resolve_output_file(raw)


def prepare_samples(limit: int, top_k_limit: int = DEFAULT_SELECTOR_TOP_K) -> list[QCSample]:
    tasks = load_annotations_from_pg(use_task_source_identity=True)
    samples = build_gold_samples(tasks)
    needed_recruitment_ids = [
        sample.recruitment_record_id
        for sample in samples
        if sample.recruitment_record_id and sample.gold_code
    ]
    matches = load_occupation_detail_matches(recruitment_record_ids=needed_recruitment_ids)
    summarize_qc_samples(samples, matches)
    eligible_samples = [
        sample
        for sample in samples
        if sample.gold_code and sample.top10_candidates
    ]
    top_k_limit = max(1, int(top_k_limit or DEFAULT_SELECTOR_TOP_K))
    for sample in eligible_samples:
        sample.top10_candidates = list(sample.top10_candidates[:top_k_limit])
    if limit > 0:
        eligible_samples = eligible_samples[:limit]
    return eligible_samples


def clone_sample(sample: QCSample) -> QCSample:
    return QCSample(
        task_id=sample.task_id,
        recruitment_record_id=sample.recruitment_record_id,
        job_title=sample.job_title,
        job_requirements=sample.job_requirements,
        gold_choice=sample.gold_choice,
        gold_code=sample.gold_code,
        gold_title=sample.gold_title,
        anchor=sample.anchor,
        top10_candidates=[dict(candidate) for candidate in sample.top10_candidates],
        hit_rank=sample.hit_rank,
        top1_code=sample.top1_code,
        top1_title=sample.top1_title,
        qc_label=sample.qc_label,
        task_source_identity_status=sample.task_source_identity_status,
    )


def build_model_runtime_with_config(config_path: Path):
    runtime = load_model_runtime_config()
    return runtime.__class__(
        default_llm_backend=runtime.default_llm_backend,
        fallback_llm_backend="",
        vllm_config_path=config_path,
        llm_env_file=runtime.llm_env_file,
        llm_request_timeout=runtime.llm_request_timeout,
        llm_retry=runtime.llm_retry,
        default_embedding_model=runtime.default_embedding_model,
        embedding_batch_size=runtime.embedding_batch_size,
        normalize_embeddings=runtime.normalize_embeddings,
        prefer_cuda=runtime.prefer_cuda,
        empty_cache_after_batch=runtime.empty_cache_after_batch,
    )


def start_local_server(config_path: Path) -> subprocess.Popen[str]:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_stem = config_path.stem
    stdout_path = log_dir / f"{log_stem}.out.log"
    stderr_path = log_dir / f"{log_stem}.err.log"
    stdout_file = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr_file = stderr_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "src.llm.vllm_server",
            "--config",
            str(config_path),
            "serve",
            "--skip-check",
        ],
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        cwd=str(Path.cwd()),
    )
    LOGGER.info("服务日志: stdout=%s stderr=%s", stdout_path, stderr_path)
    return process


def stop_local_server(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        LOGGER.warning("本地模型包装进程未在 10 秒内退出，开始清理进程树 pid=%s", process.pid)

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            LOGGER.warning("taskkill 后进程仍未退出 pid=%s", process.pid)
        return

    process.kill()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        LOGGER.warning("kill 后进程仍未退出 pid=%s", process.pid)


def wait_until_expected_model_ready(
    model_config: ModelRunConfig,
    process: subprocess.Popen[str],
    timeout_seconds: int = 1800,
) -> None:
    vllm_config = load_vllm_config(model_config.vllm_config_path)
    deadline = time.time() + timeout_seconds
    api_base = vllm_config.api_base
    models_url = f"{api_base}/models"
    expected_model = vllm_config.model_name
    http = requests.Session()
    last_error: str | None = None

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{model_config.label} 服务启动失败，exit_code={process.returncode}，"
                f"请查看 logs/{model_config.vllm_config_path.stem}.err.log"
            )
        try:
            response = http.get(models_url, timeout=5)
            if response.ok:
                payload = response.json()
                model_ids = [str(item.get("id", "")) for item in payload.get("data", [])]
                if expected_model in model_ids:
                    LOGGER.info("%s 服务就绪，模型列表=%s", model_config.label, model_ids)
                    return
                last_error = f"模型已响应但未见目标模型，当前 models={model_ids}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(
        f"{model_config.label} 服务在 {timeout_seconds} 秒内未就绪，"
        f"期望模型={expected_model}，最后状态={last_error or 'unknown'}"
    )


def _find_listening_pids(port: int) -> list[int]:
    if os.name != "nt":
        return []
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    pids: set[int] = set()
    needle = f":{port}"
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or needle not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address = parts[1]
        state = parts[3] if len(parts) >= 5 else ""
        pid_text = parts[-1]
        if needle not in local_address or state.upper() != "LISTENING":
            continue
        try:
            pids.add(int(pid_text))
        except ValueError:
            continue
    return sorted(pids)


def _kill_wsl_server_by_port(config_path: Path) -> None:
    vllm_config = load_vllm_config(config_path)
    port = int(vllm_config.port)
    distro = vllm_config.distro
    commands = [
        f"fuser -k {port}/tcp >/dev/null 2>&1 || true",
        f"pkill -f '--port {port}' >/dev/null 2>&1 || true",
    ]
    for shell_command in commands:
        subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-lc", shell_command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _force_cleanup_port_listeners(config_path: Path) -> None:
    if os.name != "nt":
        return
    vllm_config = load_vllm_config(config_path)
    pids = _find_listening_pids(vllm_config.port)
    if not pids:
        return
    LOGGER.warning("发现残留端口监听，开始按端口清理: port=%s pids=%s", vllm_config.port, pids)
    for pid in pids:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def wait_until_server_stops(config_path: Path, timeout_seconds: int = 60) -> None:
    vllm_config = load_vllm_config(config_path)
    deadline = time.time() + timeout_seconds
    models_url = f"{vllm_config.api_base}/models"
    http = requests.Session()

    while time.time() < deadline:
        try:
            response = http.get(models_url, timeout=3)
            if not response.ok:
                return
        except requests.RequestException:
            return
        time.sleep(1)

    _force_cleanup_port_listeners(config_path)
    _kill_wsl_server_by_port(config_path)

    second_deadline = time.time() + 20
    while time.time() < second_deadline:
        try:
            response = http.get(models_url, timeout=3)
            if not response.ok:
                return
        except requests.RequestException:
            return
        time.sleep(1)

    raise TimeoutError(f"服务在 {timeout_seconds} 秒内未停止: {vllm_config.api_base}")


def run_model_eval(
    model_config: ModelRunConfig,
    base_samples: Sequence[QCSample],
    *,
    workers: int = DEFAULT_SELECTOR_WORKERS,
    top_k_limit: int = DEFAULT_SELECTOR_TOP_K,
) -> dict[str, Any]:
    server_process: subprocess.Popen[str] | None = None
    started = time.perf_counter()
    try:
        LOGGER.info("启动本地模型服务: %s", model_config.label)
        server_process = start_local_server(model_config.vllm_config_path)
        wait_until_expected_model_ready(model_config, server_process, timeout_seconds=1800)

        runtime = build_model_runtime_with_config(model_config.vllm_config_path)
        samples = [clone_sample(sample) for sample in base_samples]
        outputs = apply_llm_top10_selection(
            samples,
            limit=0,
            backend=None,
            top_k_limit=top_k_limit,
            workers=workers,
        )

        metrics = summarize_selector_metrics(samples)
        return {
            "label": model_config.label,
            "vllm_config_path": str(model_config.vllm_config_path),
            "sample_count": len(samples),
            "metrics": metrics,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "outputs": outputs,
        }
    finally:
        stop_local_server(server_process)
        if server_process is not None:
            wait_until_server_stops(model_config.vllm_config_path, timeout_seconds=60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two local Top10 selector models.")
    parser.add_argument("--limit", type=int, default=200, help="Number of gold+Top10 samples to evaluate.")
    parser.add_argument(
        "--qwen3-config",
        "--qwen35-config",
        dest="qwen3_config",
        default="config/vllm_qwen3_8b.toml",
        help="vLLM config for Qwen3-8B.",
    )
    parser.add_argument(
        "--qwen36-config",
        default="config/vllm_qwen36_27b.toml",
        help="vLLM config for Qwen3.6-27B.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path or filename under output/occupation_retrieval.",
    )
    parser.add_argument(
        "--selector-top-k",
        type=int,
        default=DEFAULT_SELECTOR_TOP_K,
        help=f"Limit selector candidates to TopK. Default is {DEFAULT_SELECTOR_TOP_K}.",
    )
    parser.add_argument(
        "--selector-workers",
        type=int,
        default=DEFAULT_SELECTOR_WORKERS,
        help=f"Concurrent workers for selector requests. Default is {DEFAULT_SELECTOR_WORKERS}.",
    )
    parser.add_argument("--log-level", default="INFO", help="Console log level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    with StepTimer("准备评估样本"):
        samples = prepare_samples(
            limit=max(0, int(args.limit)),
            top_k_limit=max(1, int(args.selector_top_k)),
        )
        LOGGER.info("评估样本数: %s (top_k=%s)", len(samples), max(1, int(args.selector_top_k)))

    model_runs = [
        ModelRunConfig(label="Qwen3-8B", vllm_config_path=Path(args.qwen3_config)),
        ModelRunConfig(label="Qwen3.6-27B", vllm_config_path=Path(args.qwen36_config)),
    ]
    results = []
    for model_run in model_runs:
        with StepTimer(f"评估 {model_run.label}"):
            results.append(
                run_model_eval(
                    model_run,
                    samples,
                    workers=max(1, int(args.selector_workers)),
                    top_k_limit=max(1, int(args.selector_top_k)),
                )
            )

    output_path = _resolve_output_path(args.output_file)
    output_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Comparison saved to: %s", output_path)


if __name__ == "__main__":
    main()

