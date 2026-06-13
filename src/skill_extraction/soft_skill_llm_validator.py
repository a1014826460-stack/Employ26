"""软技能 LLM 二次验证模块。

职责：
1. 接收 `SoftSkillMatcher.match_text()` 输出的候选软技能列表；
2. 通过 LLM 对每个候选做上下文验证，确认其在当前文本中确实是软技能（而非硬技能或职责描述），
   并检查分类（dimension）是否正确；
3. LLM 调用失败时降级为仅词典结果，标记 `confidence=0.5`。

用法：
    from src.skill_extraction.soft_skill_llm_validator import validate_soft_skills
    from src.model_platform.llm import create_llm_client

    llm_client = create_llm_client()
    validated = validate_soft_skills(candidates, context_text, llm_client)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── 提示词模板 ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一位人力资源 NLP 专家，擅长判断招聘文本中出现的词汇是否为"软技能"。

## 定义
- 软技能：人际交往、个性特质、通用能力方面的素质（如沟通能力、团队协作、创新思维）。
- 硬技能：具体的技术、工具、证书、编程语言等（如 Python、SQL、注册会计师）。
- 职责描述：描述工作内容或岗位要求的短语（如"负责项目管理"、"完成销售目标"），不是技能本身。

## 任务
给定一段招聘原文和若干候选软技能，请判断每个候选在该上下文中：
1. 是否确实是软技能（而非硬技能或职责描述）；
2. 如果是软技能，其维度分类是否合理。

## 维度说明
- extraversion（外向性）：沟通、协作、领导、表达等人际相关能力
- openness（开放性）：创新、学习、适应、探索等成长相关能力
- agreeableness（宜人性）：服务意识、同理心、合作意愿等
- conscientiousness（尽责性）：执行力、责任心、细致、自律等
- neuroticism（情绪稳定性）：抗压、情绪管理、冷静等
- other：无法归入以上维度的软技能

## 输出格式
严格返回一个 JSON 数组，每个元素格式：
{"name": "技能名", "is_soft_skill": true/false, "dimension": "维度名", "reason": "简要理由"}

- 如果候选不是软技能，`is_soft_skill` 设为 false，`dimension` 保持原值。
- 不要添加原文中没有的候选。
- 不要输出除 JSON 以外的任何文字。"""

_USER_PROMPT_TEMPLATE = """\
## 招聘原文
{context_text}

## 候选软技能列表
{candidates_json}

请逐个判断并返回 JSON 数组。"""


def _build_user_prompt(context_text: str, candidates: List[dict]) -> str:
    """构建用户提示词。"""
    candidates_json = json.dumps(
        [{"name": c["name"], "dimension": c["dimension"]} for c in candidates],
        ensure_ascii=False,
        indent=2,
    )
    return _USER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        candidates_json=candidates_json,
    )


def _parse_llm_response(response_text: str) -> Optional[List[dict]]:
    """从 LLM 响应中解析 JSON 数组。

    Args:
        response_text: LLM 返回的原始文本。

    Returns:
        解析后的字典列表，解析失败返回 None。
    """
    # 尝试直接解析
    text = response_text.strip()
    # 去除可能的 markdown 代码块包裹
    if text.startswith("```"):
        # 移除首尾的 ``` 行
        lines = text.split("\n")
        # 去掉第一行（```json 或 ```）和最后一行（```）
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 尝试从文本中提取 JSON 数组
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _merge_validation_results(
    candidates: List[dict],
    llm_results: Optional[List[dict]],
    fallback: bool = False,
) -> List[dict]:
    """将 LLM 验证结果与原始候选合并。

    Args:
        candidates: 原始候选列表（来自 match_text）。
        llm_results: LLM 返回的验证结果列表。
        fallback: 是否为降级模式（LLM 调用失败）。

    Returns:
        合并后的结果列表。
    """
    if fallback:
        # 降级模式：保留所有候选，降低置信度
        results = []
        for c in candidates:
            results.append(
                {
                    "name": c["name"],
                    "dimension": c["dimension"],
                    "confidence": 0.5,
                    "source": "dict_match+llm_fallback",
                }
            )
        return results

    if llm_results is None:
        # LLM 返回无法解析，降级处理
        return _merge_validation_results(candidates, llm_results=None, fallback=True)

    # 构建 LLM 结果的名称 → 结果映射
    llm_map: Dict[str, dict] = {}
    for item in llm_results:
        name = item.get("name", "")
        if name:
            llm_map[name] = item

    results: List[dict] = []
    for c in candidates:
        name = c["name"]
        llm_item = llm_map.get(name)

        if llm_item is None:
            # LLM 没有返回该候选的判断，视为未确认，保留但降低置信度
            results.append(
                {
                    "name": name,
                    "dimension": c["dimension"],
                    "confidence": 0.5,
                    "source": "dict_match+llm_unconfirmed",
                }
            )
            continue

        is_soft_skill = llm_item.get("is_soft_skill", False)
        if not is_soft_skill:
            # LLM 判定不是软技能，过滤掉
            logger.debug("LLM 过滤非软技能: %s", name)
            continue

        # LLM 确认是软技能，使用 LLM 建议的维度（如果有）
        validated_dimension = llm_item.get("dimension", c["dimension"])
        # LLM 确认后提升置信度
        new_confidence = max(c.get("confidence", 0.8), 0.9)

        results.append(
            {
                "name": name,
                "dimension": validated_dimension,
                "confidence": new_confidence,
                "source": "dict_match+llm_confirm",
            }
        )

    return results


def validate_soft_skills(
    candidates: List[dict],
    context_text: str,
    llm_client: Any,
) -> List[dict]:
    """对候选软技能做 LLM 二次验证。

    流程：
    1. 将候选列表和上下文文本发送给 LLM；
    2. LLM 逐个判断候选是否为软技能、分类是否正确；
    3. 根据 LLM 结果过滤和更新候选；
    4. LLM 调用失败时降级为仅词典结果，标记 confidence=0.5。

    Args:
        candidates: 候选软技能列表，每个元素为字典:
            {"name": str, "dimension": str, "confidence": float, "source": str}
        context_text: 原始招聘文本，用于 LLM 上下文判断。
        llm_client: LLM 客户端，需实现 complete_text(system_prompt, user_prompt) 方法。

    Returns:
        验证后的软技能列表，每个元素为字典:
        {"name": str, "dimension": str, "confidence": float, "source": str}
        source 字段值为 "dict_match+llm_confirm"（通过验证）或
        "dict_match+llm_fallback"（降级模式）。
    """
    if not candidates:
        return []

    if not context_text or not context_text.strip():
        # 无上下文文本，无法验证，降级处理
        logger.warning("上下文文本为空，无法进行 LLM 验证，降级为词典结果")
        return _merge_validation_results(candidates, llm_results=None, fallback=True)

    # 构建提示词
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _build_user_prompt(context_text, candidates)

    # 调用 LLM
    try:
        response_text = llm_client.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.warning("LLM 验证调用失败，降级为词典结果: %s", exc)
        return _merge_validation_results(candidates, llm_results=None, fallback=True)

    # 解析 LLM 响应
    llm_results = _parse_llm_response(response_text)
    if llm_results is None:
        logger.warning("LLM 响应解析失败，降级为词典结果。原始响应: %s", response_text[:200])
        return _merge_validation_results(candidates, llm_results=None, fallback=True)

    # 合并结果
    return _merge_validation_results(candidates, llm_results=llm_results, fallback=False)
