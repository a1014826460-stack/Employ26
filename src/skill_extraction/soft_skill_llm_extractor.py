"""LLM 直接抽取软技能（支持批量 + few-shot）。

批量模式：一次 LLM 调用处理多条文本，大幅提升速度。
Few-shot：从 enriched gold 中采样示例注入 prompt，提升覆盖率和维度准确率。

用法::

    from src.skill_extraction.soft_skill_llm_extractor import (
        extract_soft_skills,
        extract_soft_skills_batch,
        extract_hard_skills,
    )

    # 单条
    skills = extract_soft_skills(text="...", llm_client=client)

    # 批量 (5-10x faster)
    results = extract_soft_skills_batch(texts=["...", "..."], llm_client=client)
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """你是人力资源NLP专家。从岗位描述中提取软技能（人际特质、性格品质、通用能力），忽略硬技能（工具、软件、编程语言）。

## 大五人格维度
- conscientiousness: 尽责性（责任心、执行力、细心、认真、踏实、吃苦耐劳）
- extraversion: 外向性（沟通、表达、谈判、领导、组织、开朗）
- openness: 开放性（创新、学习、思维、上进、进取、适应）
- agreeableness: 宜人性（合作、团队、服务、同理心）
- neuroticism: 情绪稳定性（抗压、情绪管理、冷静）

## 规则
1. 提取文本中明确提到的软技能（不要凭空推断）
2. 每个技能输出规范名称（2-6个字）
3. 标注正确的大五维度
4. 每个文本最多10个软技能
5. 只输出 JSON 数组"""

FEWSHOT_EXAMPLES = """## 参考示例
- "具备良好的沟通表达能力" → "沟通能力" (extraversion)
- "责任心强" → "责任心" (conscientiousness)
- "能承受较大工作压力" → "抗压能力" (neuroticism)
- "性格开朗外向" → "性格开朗" (extraversion)
- "团队合作精神" → "团队合作" (agreeableness)
- "学习能力强" → "学习能力" (openness)
- "工作认真负责" → "认真负责" (conscientiousness)
- "吃苦耐劳" → "吃苦耐劳" (neuroticism)
- "细心严谨" → "细心" (conscientiousness)
- "创新思维" → "创新" (openness)"""

HARD_SKILL_SYSTEM_PROMPT = """你是技术招聘专家。从岗位描述中提取硬技能（技术工具、软件、编程语言、设备、证书等），忽略软技能和学历要求。

## 8 类硬技能
- programming_language: 编程语言（Python, Java, C++）
- framework: 框架（Spring, React, Django）
- database: 数据库（MySQL, Redis, MongoDB）
- tool: 工具软件（Photoshop, Excel, CAD）
- office: 办公软件（Word, PPT, WPS）
- equipment: 设备/仪器（示波器, 万用表, CNC）
- process: 工艺/方法（焊接, 注塑, 喷涂）
- certification: 证书/资质（PMP, CPA, 电工证）

## 规则
1. 提取文本中明确提到的硬技能（标准化名称）
2. 标注正确的 8 类分类
3. 不要提取学历、年限、软技能
4. 每个文本最多15个
5. 只输出 JSON 数组: [{"name": "Java", "category": "programming_language"}, ...]"""


def _load_fewshot_examples() -> List[Dict[str, str]]:
    """从 enriched gold 中加载 few-shot 示例（可选）。"""
    try:
        from pathlib import Path

        path = Path("output/skill_extraction/eval/soft_skill_gold_enriched.jsonl")
        if not path.exists():
            return []
        examples: List[Dict[str, str]] = []
        seen = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                for gs in sample.get("gold_soft_skills", []):
                    name = gs.get("name", "")
                    dim = gs.get("dimension", "")
                    if name and dim and name not in seen:
                        seen.add(name)
                        examples.append({"name": name, "dimension": dim})
        # 每维度最多取 5 个
        dim_counts: Dict[str, int] = {}
        selected: List[Dict[str, str]] = []
        random.shuffle(examples)
        for ex in examples:
            d = ex["dimension"]
            if dim_counts.get(d, 0) < 5:
                selected.append(ex)
                dim_counts[d] = dim_counts.get(d, 0) + 1
        return selected
    except Exception:
        return []


def extract_soft_skills(
    text: str,
    llm_client: Any,
    *,
    max_skills: int = 10,
    use_fewshot: bool = True,
) -> List[Dict[str, str]]:
    """使用 LLM 从文本中直接抽取软技能。

    参数:
        text: 岗位描述文本。
        llm_client: LLM 客户端。
        max_skills: 每段文本最多抽取的软技能数。
        use_fewshot: 是否注入 few-shot 示例。

    返回:
        list[dict]: 抽取结果，每项含 ``name`` 和 ``dimension``。
    """
    if not text or not text.strip():
        return []

    prompt = EXTRACT_SYSTEM_PROMPT
    if use_fewshot:
        prompt += "\n\n" + FEWSHOT_EXAMPLES
    prompt = prompt.replace("最多10个软技能", f"最多{max_skills}个软技能")

    try:
        resp = llm_client.complete_json(
            system_prompt=prompt,
            user_prompt=f"岗位描述:\n{text[:3000]}",
            strength="cheap",
            max_output_tokens=500,
        )
    except Exception as exc:
        logger.warning("LLM 抽取失败: %s", exc)
        return []

    if not isinstance(resp, list):
        return []

    skills: List[Dict[str, str]] = []
    for item in resp:
        if isinstance(item, dict):
            name = (item.get("name") or item.get("skill") or "").strip()
            dimension = (
                item.get("dimension") or item.get("category") or item.get("type") or ""
            ).strip()
            if name and dimension:
                skills.append({"name": name, "dimension": dimension})
    return skills


def extract_soft_skills_batch(
    texts: Sequence[str],
    llm_client: Any,
    *,
    batch_size: int = 5,
    max_skills: int = 10,
    use_fewshot: bool = True,
) -> List[List[Dict[str, str]]]:
    """批量从多段文本中抽取软技能（一次 LLM 调用处理多条）。

    参数:
        texts: 岗位描述文本列表。
        llm_client: LLM 客户端。
        batch_size: 每批处理的文本数。
        max_skills: 每段文本最多抽取的软技能数。
        use_fewshot: 是否注入 few-shot 示例。

    返回:
        list[list[dict]]: 每条文本的抽取结果。
    """
    results: List[List[Dict[str, str]]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # 构建批量 prompt
        batch_texts = "\n\n---\n\n".join(
            f"[样本{j+1}]\n{t[:2000]}" for j, t in enumerate(batch)
        )

        prompt = EXTRACT_SYSTEM_PROMPT
        if use_fewshot:
            prompt += "\n\n" + FEWSHOT_EXAMPLES
        prompt += f"\n\n## 批量处理\n共 {len(batch)} 段文本，用 --- 分隔。对每段输出一个 JSON 数组。\n输出格式: [[样本1结果], [样本2结果], ...]\n每个样本最多{max_skills}个技能。"

        try:
            resp = llm_client.complete_json(
                system_prompt=prompt,
                user_prompt=batch_texts,
                strength="cheap",
                max_output_tokens=500 * len(batch),
            )
        except Exception as exc:
            logger.warning("批量 LLM 抽取失败 (batch %d): %s", i // batch_size, exc)
            # 回退到单条
            for text in batch:
                results.append(
                    extract_soft_skills(
                        text, llm_client, max_skills=max_skills, use_fewshot=use_fewshot
                    )
                )
            continue

        if isinstance(resp, list):
            for j, item in enumerate(resp):
                skills: List[Dict[str, str]] = []
                if isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, dict):
                            name = (
                                entry.get("name") or entry.get("skill") or ""
                            ).strip()
                            dimension = (
                                entry.get("dimension") or entry.get("category") or ""
                            ).strip()
                            if name and dimension:
                                skills.append({"name": name, "dimension": dimension})
                results.append(skills)
            # 补齐缺失的
            while len(results) < i + len(batch):
                results.append([])
        else:
            # fallback
            for text in batch:
                results.append(
                    extract_soft_skills(
                        text, llm_client, max_skills=max_skills, use_fewshot=use_fewshot
                    )
                )

    return results


def extract_hard_skills(
    text: str,
    llm_client: Any,
    *,
    max_skills: int = 15,
) -> List[Dict[str, str]]:
    """使用 LLM 从文本中直接抽取硬技能。

    参数:
        text: 岗位描述文本。
        llm_client: LLM 客户端。
        max_skills: 最多抽取技能数。

    返回:
        list[dict]: [{"name": "Java", "category": "programming_language"}, ...]
    """
    if not text or not text.strip():
        return []

    prompt = HARD_SKILL_SYSTEM_PROMPT.replace("最多15个", f"最多{max_skills}个")

    try:
        resp = llm_client.complete_json(
            system_prompt=prompt,
            user_prompt=f"岗位描述:\n{text[:3000]}",
            strength="cheap",
            max_output_tokens=800,
        )
    except Exception as exc:
        logger.warning("硬技能 LLM 抽取失败: %s", exc)
        return []

    if not isinstance(resp, list):
        return []

    skills: List[Dict[str, str]] = []
    for item in resp:
        if isinstance(item, dict):
            name = (item.get("name") or "").strip()
            category = (item.get("category") or "").strip()
            if name and category:
                skills.append({"name": name, "category": category})
    return skills
