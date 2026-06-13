"""统一的 LLM Router，优先兼容 GPT-5 Responses API。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_CHEAP_MODEL = "openai/gpt-5.4-mini"
DEFAULT_STRONG_MODEL = "openai/gpt-5.4"
DEFAULT_API_MODE = "responses"


@dataclass(frozen=True)
class LLMRouteConfig:
    """LLM 路由配置。"""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    cheap_model: str = DEFAULT_CHEAP_MODEL
    strong_model: str = DEFAULT_STRONG_MODEL
    api_mode: str = DEFAULT_API_MODE
    timeout: int = 120
    retry: int = 3
    max_output_tokens: int = 1024
    reasoning_effort: str = "low"


class LLMRouter:
    """统一的两级模型路由客户端。"""

    def __init__(self, config: LLMRouteConfig) -> None:
        self.config = config
        self._client = None

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "LLMRouter":
        values = load_llm_env(env_file)
        config = LLMRouteConfig(
            base_url=values.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=values.get("LLM_API_KEY", values.get("OPENAI_API_KEY", "")),
            cheap_model=values.get("LLM_CHEAP_MODEL", DEFAULT_CHEAP_MODEL),
            strong_model=values.get("LLM_STRONG_MODEL", DEFAULT_STRONG_MODEL),
            api_mode=values.get("LLM_API_MODE", DEFAULT_API_MODE),
            timeout=int(values.get("LLM_TIMEOUT", 120)),
            retry=int(values.get("LLM_RETRY", 3)),
            max_output_tokens=int(values.get("LLM_MAX_OUTPUT_TOKENS", 1024)),
            reasoning_effort=values.get("LLM_REASONING_EFFORT", "low"),
        )
        return cls(config)

    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.base_url)

    def get_model(self, strength: str) -> str:
        if strength == "strong":
            return self.config.strong_model
        return self.config.cheap_model

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Dict[str, Any]:
        text = self.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            strength=strength,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        parsed = extract_json_from_response(text)
        if parsed is None:
            raise ValueError("LLM 返回不是合法 JSON")
        return parsed

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        if not self.is_configured():
            raise RuntimeError("LLMRouter 未配置 api_key/base_url")

        model = self.get_model(strength)
        api_mode = self.config.api_mode

        last_error: Exception | None = None
        for attempt in range(1, self.config.retry + 1):
            try:
                if api_mode == "chat":
                    response = self._get_client().chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=int(max_output_tokens or self.config.max_output_tokens),
                        timeout=float(self.config.timeout),
                    )
                    return extract_chat_response_text(response)
                else:
                    request_kwargs = {
                        "model": model,
                        "input": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_output_tokens": int(max_output_tokens or self.config.max_output_tokens),
                        "timeout": float(self.config.timeout),
                    }
                    effort = reasoning_effort if reasoning_effort is not None else self.config.reasoning_effort
                    if effort:
                        request_kwargs["reasoning"] = {"effort": effort}
                    response = self._get_client().responses.create(**request_kwargs)
                    return extract_response_text(response)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM 请求失败 attempt=%d/%d model=%s error=%s", attempt, self.config.retry, model, exc)
                if attempt < self.config.retry:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"LLM 请求失败: {last_error}")

    def batch_complete_text(
        self,
        prompt_pairs: Sequence[Tuple[str, str]],
        *,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> List[str]:
        outputs: List[str] = []
        for system_prompt, user_prompt in prompt_pairs:
            outputs.append(
                self.complete_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    strength=strength,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )
            )
        return outputs

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("缺少 openai 依赖，无法初始化 LLMRouter") from exc
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=float(self.config.timeout),
            )
        return self._client


def load_llm_env(env_file: str | Path | None = None) -> Dict[str, str]:
    """读取环境变量与本地 .env。"""
    values = dict(os.environ)
    candidate_paths: List[Path] = []
    if env_file:
        candidate_paths.append(Path(env_file))
    else:
        project_root = Path(__file__).resolve().parents[2]
        candidate_paths.extend([
            project_root / ".env",
            project_root / ".env.local",
        ])

    for path in candidate_paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            values.setdefault(key, value)
    return values


def extract_response_text(response: Any) -> str:
    """从 Responses API 结果中提取纯文本。"""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", "")
            if content_type not in {"output_text", "text"}:
                continue
            text_value = getattr(content, "text", "")
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value.strip())
            elif hasattr(text_value, "value") and str(text_value.value).strip():
                parts.append(str(text_value.value).strip())
    return "\n".join(parts).strip()


def extract_chat_response_text(response: Any) -> str:
    """从 Chat Completions API 结果中提取纯文本。"""
    choices = getattr(response, "choices", []) or []
    if choices:
        message = getattr(choices[0], "message", None)
        if message:
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def extract_json_from_response(text: str) -> Dict[str, Any] | List[Any] | None:
    """尽量从模型输出中提取 JSON。"""
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_candidates = [idx for idx in [cleaned.find("{"), cleaned.find("[")] if idx >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(cleaned), start, -1):
        candidate = cleaned[start:end].strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def build_json_system_prompt(base_rule_text: str) -> str:
    """附加统一 JSON 输出约束。"""
    return f"{base_rule_text.strip()}\n\n只输出 JSON，不要输出 markdown，不要解释。"


def is_noisy_job_title(title: str) -> bool:
    text = str(title or "").strip()
    if not text:
        return True
    noise_patterns = [
        r"[A-Za-z]{2,}\d{2,}",
        r"[\/|_]{2,}",
        r"\([^)]{8,}\)",
        r"[（][^）]{8,}[）]",
    ]
    return any(__import__("re").search(pattern, text) for pattern in noise_patterns)


def score_candidate_margin(candidates: Sequence[Dict[str, Any]]) -> float:
    if len(candidates) < 2:
        return 1.0
    first = float(candidates[0].get("score", 0.0))
    second = float(candidates[1].get("score", 0.0))
    return first - second


def should_escalate(
    *,
    cheap_confidence: float | None,
    candidate_margin: float | None,
    is_new_title: bool,
    noisy_title: bool,
    context_conflict: bool,
    has_conflicting_candidates: bool,
    cheap_threshold: float = 0.82,
    margin_threshold: float = 0.08,
) -> bool:
    """明确的升级判别函数。"""
    if is_new_title:
        return True
    if noisy_title or context_conflict or has_conflicting_candidates:
        return True
    if cheap_confidence is not None and float(cheap_confidence) < cheap_threshold:
        return True
    if candidate_margin is not None and float(candidate_margin) < margin_threshold:
        return True
    return False
