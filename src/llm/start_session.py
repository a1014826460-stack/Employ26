"""本地 vLLM 终端对话窗口。

先在一个 PowerShell 窗口启动服务：
    .\\.conda\\python.exe -m src.llm.vllm_server serve

再在另一个 PowerShell 窗口进入长期对话：
    .\\.conda\\python.exe src/llm/start_session.py --wait
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# 运行方式：从项目根目录执行 `python -m src.llm.start_session`，
# 确保 src.* 包可通过标准 Python 模块搜索路径正确导入。
from ..utils.utils import PROJECT_ROOT, safe_print  # noqa: E402
from ..utils.vllm_utils import (  # noqa: E402
    DEFAULT_VLLM_CONFIG_PATH,
    chat_completion,
    check_server,
    create_http_session,
    extract_message_parts,
    load_vllm_config,
    wait_until_ready,
)

DEFAULT_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "system_prompt.md"


def build_parser() -> argparse.ArgumentParser:
    """构造终端对话参数。"""
    parser = argparse.ArgumentParser(description="打开一个可长期对话的本地 vLLM 终端窗口。")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_VLLM_CONFIG_PATH),
        help="vLLM TOML 配置文件路径",
    )
    parser.add_argument("--wait", action="store_true", help="进入会话前等待 API 就绪")
    parser.add_argument("--timeout", type=int, default=None, help="等待 API 就绪的秒数")
    parser.add_argument("--system", default="", help="覆盖配置文件中的 system prompt")
    parser.add_argument(
        "--system-file",
        default="",
        help=f"从文件读取 system prompt，默认读取 {DEFAULT_SYSTEM_PROMPT_PATH}",
    )
    parser.add_argument("--thinking", action="store_true", help="启用 Qwen thinking 模式")
    parser.add_argument("--show-reasoning", action="store_true", help="打印 reasoning 字段")
    parser.add_argument("--max-tokens", type=int, default=None, help="每轮最大输出 token 数")
    parser.add_argument("--temperature", type=float, default=None, help="采样温度")
    return parser


def resolve_system_prompt(args: argparse.Namespace, default_prompt: str) -> str:
    """从命令行或配置中获取 system prompt。"""
    if args.system:
        return args.system.strip()
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8").strip()
    if DEFAULT_SYSTEM_PROMPT_PATH.exists():
        return DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return default_prompt


def print_session_help() -> None:
    """打印交互命令。"""
    print("可用命令:")
    print("  /help           查看命令")
    print("  /reset          清空上下文，保留 system prompt")
    print("  /system 文本    替换 system prompt，并清空上下文")
    print("  /exit 或 /quit  退出")


def append_assistant_message(
    messages: list[dict[str, Any]],
    content: str,
    reasoning: str,
) -> None:
    """将 assistant 回复写入上下文。"""
    if content:
        messages.append({"role": "assistant", "content": content})
    elif reasoning:
        messages.append({"role": "assistant", "content": reasoning})


def run_chat_loop(args: argparse.Namespace) -> None:
    """运行长期终端对话。"""
    config = load_vllm_config(args.config)
    session = create_http_session()
    system_prompt = resolve_system_prompt(args, config.system_prompt)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if args.wait:
        wait_until_ready(config, timeout_seconds=args.timeout, session=session)
    else:
        check_server(config, session=session)

    print(f"已连接: {config.api_base}")
    print(f"模型: {config.model_name}")
    print("输入 /help 查看命令。")

    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话结束。")
            return

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("会话结束。")
            return
        if user_text == "/help":
            print_session_help()
            continue
        if user_text == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("上下文已清空。")
            continue
        if user_text.startswith("/system "):
            system_prompt = user_text.removeprefix("/system ").strip()
            messages = [{"role": "system", "content": system_prompt}]
            print("system prompt 已更新，上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            response_data = chat_completion(
                config=config,
                messages=messages,
                session=session,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enable_thinking=args.thinking,
            )
        except Exception as exc:  # noqa: BLE001
            messages.pop()
            safe_print(f"请求失败: {exc}")
            continue

        content, reasoning, finish_reason = extract_message_parts(response_data)
        if args.show_reasoning and reasoning:
            safe_print("\n=== reasoning ===")
            safe_print(reasoning)
            safe_print("=== content ===")

        if content:
            safe_print(f"\n助手> {content}")
            append_assistant_message(messages, content, reasoning)
            continue

        if reasoning:
            safe_print(
                "\n助手> 本次响应没有最终 content，只返回了 reasoning。"
                f"finish_reason={finish_reason!r}。可以增大 --max-tokens，"
                "或不要启用 --thinking。"
            )
            append_assistant_message(messages, content, reasoning)
            continue

        safe_print(f"\n助手> 本次响应为空。finish_reason={finish_reason!r}")


def main() -> None:
    """命令行入口。"""
    run_chat_loop(build_parser().parse_args())


if __name__ == "__main__":
    main()
