"""统一 LLM 调用入口。

默认后端为 WSL vLLM OpenAI-compatible API。外部 API 仅作为显式 fallback。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Protocol, Sequence, Tuple

from src.utils.llm_router import LLMRouter, extract_json_from_response
from src.utils.vllm_utils import (
    chat_completion,
    extract_message_parts,
    load_vllm_config,
)

from .config import ModelRuntimeConfig, load_model_runtime_config

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """统一 LLM client 协议。"""

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """生成文本。"""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """生成并解析 JSON。"""

    def batch_complete_text(
        self,
        prompt_pairs: Sequence[Tuple[str, str]],
        *,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """批量生成文本。"""


@dataclass
class VLLMHTTPClient:
    """WSL vLLM HTTP client。"""

    runtime: ModelRuntimeConfig

    def __post_init__(self) -> None:
        self.config = load_vllm_config(self.runtime.vllm_config_path)

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = chat_completion(
            config=self.config,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
            enable_thinking=False,
        )
        content, reasoning, _ = extract_message_parts(response)
        return content or reasoning

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        text = self.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            strength=strength,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        parsed = extract_json_from_response(text)
        if parsed is None:
            raise ValueError("LLM 返回不是合法 JSON")
        return parsed

    def batch_complete_text(
        self,
        prompt_pairs: Sequence[Tuple[str, str]],
        *,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        return [
            self.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                strength=strength,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )
            for system_prompt, user_prompt in prompt_pairs
        ]


@dataclass
class ExternalAPIClient:
    """外部 OpenAI-compatible API client。"""

    runtime: ModelRuntimeConfig

    def __post_init__(self) -> None:
        self.router = LLMRouter.from_env(self.runtime.llm_env_file)

    @staticmethod
    def _normalize_router_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop protocol-level kwargs not accepted by LLMRouter."""
        normalized = dict(kwargs)
        normalized.pop("temperature", None)
        return normalized

    def complete_text(self, **kwargs: Any) -> str:
        return self.router.complete_text(**self._normalize_router_kwargs(kwargs))

    def complete_json(self, **kwargs: Any) -> dict[str, Any] | list[Any]:
        text = self.complete_text(**kwargs)
        parsed = extract_json_from_response(text)
        if parsed is None:
            raise ValueError("LLM 返回不是合法 JSON")
        return parsed

    def batch_complete_text(
        self,
        prompt_pairs: Sequence[Tuple[str, str]],
        **kwargs: Any,
    ) -> list[str]:
        return [
            self.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **kwargs,
            )
            for system_prompt, user_prompt in prompt_pairs
        ]


@dataclass
class FallbackLLMClient:
    """带 fallback 的 LLM client。"""

    primary: LLMClient
    fallback: LLMClient | None
    retry: int = 1

    def complete_text(self, **kwargs: Any) -> str:
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.retry) + 1):
            try:
                return self.primary.complete_text(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("默认 LLM 后端失败 attempt=%s error=%s", attempt, exc)
                if attempt < self.retry:
                    time.sleep(min(2 ** (attempt - 1), 8))
        if self.fallback is not None:
            return self.fallback.complete_text(**kwargs)
        raise RuntimeError(f"LLM 请求失败: {last_error}") from last_error

    def complete_json(self, **kwargs: Any) -> dict[str, Any] | list[Any]:
        text = self.complete_text(**kwargs)
        parsed = extract_json_from_response(text)
        if parsed is None:
            raise ValueError("LLM 返回不是合法 JSON")
        return parsed

    def batch_complete_text(
        self,
        prompt_pairs: Sequence[Tuple[str, str]],
        **kwargs: Any,
    ) -> list[str]:
        return [
            self.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **kwargs,
            )
            for system_prompt, user_prompt in prompt_pairs
        ]


def _build_backend(name: str, runtime: ModelRuntimeConfig) -> LLMClient:
    """按名称构建 LLM 后端。"""
    if name == "wsl_vllm":
        return VLLMHTTPClient(runtime)
    if name == "external_api":
        return ExternalAPIClient(runtime)
    raise ValueError(f"不支持的 LLM 后端: {name}")


def create_llm_client(
    backend: str | None = None,
    *,
    allow_fallback: bool = True,
) -> LLMClient:
    """创建统一 LLM client。"""
    runtime = load_model_runtime_config()
    primary_name = backend or runtime.default_llm_backend
    primary = _build_backend(primary_name, runtime)
    fallback = None
    if allow_fallback and runtime.fallback_llm_backend and runtime.fallback_llm_backend != primary_name:
        fallback = _build_backend(runtime.fallback_llm_backend, runtime)
    return FallbackLLMClient(primary=primary, fallback=fallback, retry=runtime.llm_retry)
