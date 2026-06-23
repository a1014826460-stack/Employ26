"""Unified DeepSeek API client with optional threaded batch execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, Mapping, Sequence

from openai import OpenAI


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


class DeepSeekResponseError(RuntimeError):
    """Raised when DeepSeek returns unusable content."""


@dataclass(frozen=True)
class DeepSeekConfig:
    """Runtime config for the shared DeepSeek client."""

    api_key: str
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout: int = 90
    retries: int = 1
    disable_thinking: bool = False  # 是否禁用推理模式


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Extract a JSON object from a raw model response."""
    text_value = str(raw_text or "").strip()
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
    """Unified DeepSeek JSON client."""

    def __init__(self, config: DeepSeekConfig) -> None:
        if not str(config.api_key or "").strip():
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self.config = config

    def _build_openai_client(self) -> OpenAI:
        return OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Mapping[str, Any] | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                kwargs = dict(
                    model=model or self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=response_format or {"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout or self.config.timeout,
                )
                if self.config.disable_thinking:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                response = self._build_openai_client().chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message
                raw = (getattr(message, "content", None) or "").strip()
                if not raw:
                    raw = (getattr(message, "reasoning_content", None) or "").strip()
                return raw
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.config.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"DeepSeek 请求失败: {last_error}") from last_error

    def complete_json_with_raw(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[dict[str, Any], str]:
        """Return both parsed JSON and the raw model text."""
        raw = self.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_response(raw)
        if not parsed:
            raise DeepSeekResponseError("DeepSeek 返回不是合法 JSON object")
        return parsed, raw

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        parsed, _ = self.complete_json_with_raw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return parsed

    def map_json(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        workers: int = 2,
    ) -> list[dict[str, Any]]:
        """Run multiple DeepSeek JSON calls concurrently."""
        if not requests:
            return []
        outputs: list[dict[str, Any] | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            future_to_index = {
                executor.submit(
                    self.complete_json,
                    system_prompt=str(item["system_prompt"]),
                    user_prompt=str(item["user_prompt"]),
                    model=item.get("model"),
                    timeout=item.get("timeout"),
                    temperature=float(item.get("temperature", 0.0)),
                    max_tokens=int(item.get("max_tokens", 1024)),
                ): index
                for index, item in enumerate(requests)
            }
            for future in as_completed(future_to_index):
                outputs[future_to_index[future]] = future.result()
        return [item for item in outputs if item is not None]


def build_deepseek_client(
    *,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    timeout: int = 90,
    retries: int = 1,
) -> DeepSeekClient:
    """Construct the shared DeepSeek client from environment."""
    return DeepSeekClient(
        DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model=model,
            timeout=timeout,
            retries=retries,
        )
    )
