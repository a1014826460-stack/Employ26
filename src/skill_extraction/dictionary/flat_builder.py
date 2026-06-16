"""v2 平面化职业技能词典流水线。"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..config import SkillExtractionConfig


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _now_text() -> str:
    """返回当前时间文本。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ──────────────────────────────────────────────────────────────────────
#  v2 — FlatSkillPipeline  平面化技能词典流水线 (vLLM + Qwen3-8B)
# ──────────────────────────────────────────────────────────────────────

FLAT_TRAINING_SYSTEM_PROMPT = """\
你是一名招聘硬技能提取专家。你需要从岗位任职要求样本中提取可标准化、可复用的硬技能词。

## 提取规则

1. **只保留硬技能**：工具、软件、编程语言、框架、数据库、证书、行业方法、设备等。
2. **排除以下内容**：
   - 软素质（沟通能力、团队协作、责任心……）
   - 人格特质（积极主动、细心认真……）
   - 学历/年限/年龄要求
   - 福利待遇、薪资条件
   - 岗位名称、空泛职责动词（如"负责""处理"）
3. **同义词收敛**：同一技能的不同写法请收敛为一个标准名称，其余写法放入 aliases。
   - 例如 "PS" / "Photoshop" / "Adobe Photoshop" → name: "Photoshop", aliases: ["PS", "Adobe Photoshop"]
4. **高精度优先**：宁可漏掉边缘技能，也不要加入模糊、不确定的词。
5. **标注 skill_type**：请标注技能类别（编程语言、框架、数据库、工具软件、办公软件、\
证书/资质、专业知识、工艺/方法、设备/仪器 等）。
6. **不要输出技能容器词**：不要输出“测试”“仿真软件”“数据分析工具”“办公软件”“资格证书”“执业资格证书”“专业知识”\
这类过泛名称；必须尽量输出可直接落地的具体技能名、工具名、框架名、设备名或具体证书名。
7. **证书必须具体**：可以输出“CPA”“PMP”“医师资格证”“教师资格证”等具体证书，\
但不要输出“证书”“资格证”“执业资格证书”这种泛称。

## 输出格式

请 **只输出 JSON**，不要输出任何解释或 markdown 标记。

{"skills":[{"name":"标准技能名","aliases":["别名1","别名2"],"skill_type":"技能类别","notes":""}]}
"""

FLAT_TRAINING_USER_TEMPLATE = """\
职业中类: {category_name}

以下是该职业中类的 {count} 条岗位任职要求样本，请从中提取硬技能：

{samples_text}
"""

FLAT_EVALUATION_SYSTEM_PROMPT = """\
你是一名技能词典覆盖率评估专家。你需要判断现有技能词典是否能覆盖验证样本中提到的硬技能。

## 评估规则

1. 逐条检查验证样本中的硬技能是否已存在于词典中（匹配 name 或 aliases）。
2. 忽略软素质、学历年限、福利待遇等非硬技能项。
3. 只关注"精准匹配"——如果样本中出现的技能名在词典的 name 或 aliases 里，算覆盖。
4. 若发现未覆盖的硬技能，请提取出来作为缺失技能。
5. 不要把“测试”“仿真软件”“办公软件”“资格证书”“执业资格证书”“专业知识”\
这类泛称当作有效硬技能；如果文本里提到的是具体工具、框架、设备或具体证书，请优先给出具体名称。

## 输出格式

请 **只输出 JSON**，不要输出任何解释或 markdown 标记。

{"covered_count":N,"total_hard_skill_items":N,"accuracy":0.85,\
"missing_skills":[{"name":"缺失技能名","aliases":[],"skill_type":"类别","notes":""}]}
"""

FLAT_EVALUATION_USER_TEMPLATE = """\
## 当前技能词典（共 {skill_count} 项）

{skill_summary}

## 验证样本（共 {sample_count} 条任职要求）

{validation_text}

请评估上述词典对验证样本中硬技能的覆盖率，并提取未覆盖的硬技能。
"""

# 每条 prompt 最多包含的样本数（避免超出 vLLM 上下文窗口）
MAX_SAMPLES_PER_PROMPT: int = 15

# 每条样本文本的最大字符数（截断过长的任职要求，避免 prompt 超长）
MAX_SAMPLE_CHARS: int = 300

# 默认覆盖率阈值
DEFAULT_COVERAGE_THRESHOLD: float = 0.80

# 这些模式用于过滤“看起来像技能、实际上只是容器词/泛化概念”的条目。
# 目标是从词典源头减少会在匹配阶段制造系统性误报的技能项。
LOW_VALUE_SKILL_PATTERNS: List[str] = [
    r"(能力|素养|基础|知识|理论)$",
    r"(工具|软件|系统|平台)$",
    r"^(测试|检测|英语|普通话|电源)$",
    r"^(数据分析工具|仿真软件|办公软件|测试仪器|专业知识|理论基础|知识基础)$",
    r"^(资格证|资格证书|执业资格证书|上岗证|证书)$",
]

LOW_VALUE_ALIAS_PATTERNS: List[str] = [
    r"^(资格|资格证|资格证书|证书|执业证|上岗证|许可证|执照)$",
    r"^(工具|软件|系统|平台|知识|理论|能力|测试)$",
    r".*(资格证|资格证书|执业证|执业资格证书)$",
]

# 默认模型路径由 ``config/database.yaml`` 中的 ``LLM_model_path`` 提供。
DEFAULT_MODEL_PATH: str | None = None


class FlatSkillPipeline:
    """v2 平面化技能词典构建流水线。

    按「职业中类」采样岗位任职要求文本，使用本地 vLLM (Qwen3-8B)
    批量推理提取硬技能，输出全局平面化的技能列表。

    流水线步骤:
        1. 加载岗位数据并按职业中类分组采样（训练 100 + 验证 10）
        2. 对每个中类的训练样本，使用 vLLM 批量推理提取硬技能
        3. 合并所有中类的技能，按名称去重
        4. 对每个中类的验证样本，使用 vLLM 评估覆盖率
        5. 覆盖率不足时自动提取缺失技能并补充
        6. 最终全局去重（name + aliases 互不冲突）
        7. 保存为平面化 JSON 词典

    参数:
        config (SkillExtractionConfig): 全局配置（数据库路径、输出目录等）。
        model_path (str): Qwen3-8B 模型的本地路径。
        gpu_memory_utilization (float): GPU 显存利用率，范围 (0, 1)。
        max_model_len (int): vLLM 最大序列长度（输入 + 输出 token 总和）。
        max_num_seqs (int): vLLM 最大并发序列数。

    示例::

        config = load_skill_extraction_config()
        pipeline = FlatSkillPipeline(config, model_path="D:/model/Qwen3-8B")
        pipeline.run(
            train_size=100,
            validation_size=10,
            coverage_threshold=0.80,
        )
    """

    def __init__(
        self,
        config: SkillExtractionConfig,
        model_path: str | None = DEFAULT_MODEL_PATH,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 8192,
        max_num_seqs: int = 48,
    ) -> None:
        """初始化流水线。

        参数:
            config: 全局配置对象，包含数据库路径、输出目录等。
            model_path: Qwen3-8B 模型目录路径（HuggingFace 格式）。
                未显式传入时，从 ``config.llm_model_path`` 读取。
            gpu_memory_utilization: GPU 显存利用率。
                0.80 为推荐值（RTX 4090 24 GB），留约 4.8 GB 给系统。
            max_model_len: vLLM 最大上下文长度。
                8192 适用于本任务（prompt 约 500-2000 token，
                输出约 500-2048 token），留有余量。
            max_num_seqs: vLLM 最大并发序列数。
                48 为 RTX 4090 推荐值，平衡吞吐与显存。
        """
        self.config = config
        self.model_path = str(model_path or config.llm_model_path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self._llm = None  # 延迟初始化
        self._tokenizer = None

    # ── vLLM 引擎管理 ────────────────────────────────────────────

    def _ensure_llm(self) -> None:
        """延迟初始化 vLLM 推理引擎。

        首次调用时从 ``merge_similar_skills`` 模块导入
        ``init_vllm_engine`` 并初始化 ``vllm.LLM`` 实例。
        后续调用直接复用已有实例，避免重复加载模型权重（约 15.3 GB）。

        副作用:
            - 设置 ``self._llm`` 为 ``vllm.LLM`` 实例。
            - 设置 ``self._tokenizer`` 为模型的 tokenizer。

        异常:
            SystemExit: 如果 GPU 显存不足以初始化 KV cache。
        """
        if self._llm is not None:
            return

        from .merge import init_vllm_engine

        logger.info("正在初始化 vLLM 引擎（模型: %s）...", self.model_path)
        self._llm = init_vllm_engine(
            model_path=self.model_path,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
        )
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("vLLM 引擎初始化完成")

    def _vllm_batch_generate(
        self,
        prompt_pairs: List[Tuple[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> List[str]:
        """使用 vLLM 批量推理生成文本。

        接受 ``(system_prompt, user_prompt)`` 对列表，使用 tokenizer 的
        chat template 格式化为模型输入，然后一次性提交给 vLLM 引擎
        进行离线批量推理。

        参数:
            prompt_pairs: 每个元素为 ``(system_prompt, user_prompt)`` 的元组。
            max_tokens: 每条 prompt 的最大生成 token 数，默认 2048。
            temperature: 采样温度。0.1 为低温度，确保输出稳定和格式一致。

        返回:
            list[str]: 每条 prompt 对应的生成文本，长度与 ``prompt_pairs`` 相同。

        说明:
            vLLM 的离线批量推理模式（ ``LLM.generate()`` ）吞吐量最高，
            内部通过 continuous batching 和 PagedAttention 自动调度。
        """
        self._ensure_llm()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            repetition_penalty=1.05,
        )

        # 使用 tokenizer chat template 格式化 prompt
        formatted_prompts: List[str] = []
        for system_prompt, user_prompt in prompt_pairs:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                prompt_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                # 回退：简单拼接
                prompt_text = (
                    f"system: {system_prompt}\n"
                    f"user: {user_prompt}\n"
                    f"assistant:"
                )
            formatted_prompts.append(prompt_text)

        logger.info("vLLM 批量推理: 共 %d 条 prompt", len(formatted_prompts))
        outputs = self._llm.generate(formatted_prompts, sampling_params)

        results: List[str] = []
        for output in outputs:
            results.append(output.outputs[0].text)

        logger.info("vLLM 批量推理完成")
        return results

    # ── 数据加载与采样 ───────────────────────────────────────────

    @staticmethod
    def _get_requirement_text(row: pd.Series) -> str:
        """从匹配结果行中提取需求文本。

        优先使用 ``任职要求_items_text``（已切分的任职要求），
        若为空则回退到 ``职业匹配文本``（语义匹配的原始文本）。

        参数:
            row: 匹配结果 DataFrame 的一行。

        返回:
            str: 非空的需求文本。如果两者均为空，返回空字符串。
        """
        # 定义优先级字段列表
        priority_fields = [
            "任职要求_items_text",
            "岗位职责_items_text", 
            "岗位描述_清洗"
        ]

        # 循环遍历，返回第一个非空值
        for field in priority_fields:
            value = str(row.get(field, "") or "").strip()
            if value:
                return value

        # 所有字段都为空时，返回空字符串
        return ""

    def _load_and_sample(
        self,
        train_size: int = 100,
        validation_size: int = 10,
        seed: int = 42,
        limit_job_rows: int | None = None,
        limit_categories: int | None = None,
        parse_workers: int = 1,
    ) -> Dict[str, Dict[str, List[str]]]:
        """加载岗位数据并按职业中类分组采样。

        执行流程:
            1. 调用 ``prepare_requirement_matches_to_postgres`` 将岗位与职业分类匹配。
            2. 提取每行的需求文本（任职要求或匹配文本）。
            3. 按「中类」分组。
            4. 对每个中类随机采样 ``train_size`` 条训练文本和
               ``validation_size`` 条验证文本。

        参数:
            train_size: 每个中类的训练样本数，默认 100。
            validation_size: 每个中类的验证样本数，默认 10。
            seed: 随机种子，确保采样可复现。
            limit_job_rows: 调试用，限制读取的岗位行数。
            limit_categories: 调试用，限制处理的中类数量。
            parse_workers: 岗位描述解析的并发数。

        返回:
            dict[str, dict[str, list[str]]]: 以中类名称为键，值为::

                {
                    "train_texts": ["任职要求文本1", ...],   # 最多 train_size 条
                    "validation_texts": ["任职要求文本1", ...],  # 最多 validation_size 条
                }

        异常:
            FileNotFoundError: 数据库文件不存在时抛出。
        """
        from src.data_pipeline.requirement_match_prep import (
            prepare_requirement_matches_to_postgres,
        )

        logger.info("正在加载岗位数据并匹配职业分类...")
        matched_df = prepare_requirement_matches_to_postgres(
            config=self.config,
            limit_job_rows=limit_job_rows,
            parse_workers=max(1, parse_workers),
            parse_batch_size=2000,
            top_k=self.config.match_top_k,
        )

        # 提取需求文本
        matched_df["需求文本"] = matched_df.apply(self._get_requirement_text, axis=1)

        # 过滤空文本
        matched_df = matched_df[
            matched_df["需求文本"].str.strip().astype(bool)
        ].copy()

        logger.info("有效岗位行数: %d", len(matched_df))

        # 按中类分组采样
        import random

        category_samples: Dict[str, Dict[str, List[str]]] = {}
        grouped = matched_df.groupby("中类", sort=True)
        category_names = list(grouped.groups.keys())

        if limit_categories is not None:
            category_names = category_names[: int(limit_categories)]

        for category_name in category_names:
            group_df = grouped.get_group(category_name)
            texts = group_df["需求文本"].tolist()

            # 使用固定种子随机打乱
            rng = random.Random(seed)
            rng.shuffle(texts)

            total_needed = train_size + validation_size
            train_texts = texts[:train_size]
            validation_texts = texts[train_size:total_needed]

            if not train_texts:
                logger.warning("中类 '%s' 无可用训练样本，跳过", category_name)
                continue

            category_samples[category_name] = {
                "train_texts": train_texts,
                "validation_texts": validation_texts,
            }

        logger.info(
            "采样完成: %d 个中类, 训练样本 %d 条/中类, 验证样本 %d 条/中类",
            len(category_samples),
            train_size,
            validation_size,
        )
        return category_samples

    # ── 技能提取 ─────────────────────────────────────────────────

    def _extract_skills_from_all_categories(
        self,
        category_samples: Dict[str, Dict[str, List[str]]],
    ) -> List[Dict]:
        """对所有职业中类批量提取硬技能。

        遍历每个中类的训练样本，将样本按 ``MAX_SAMPLES_PER_PROMPT``
        分批构建 prompt，然后一次性提交给 vLLM 批量推理，最大化吞吐量。

        参数:
            category_samples: 来自 ``_load_and_sample`` 的采样结果，
                以中类名称为键，值包含 ``train_texts`` 和 ``validation_texts``。

        返回:
            list[dict]: 全局去重后的技能列表，每个元素包含::

                {
                    "name": "标准技能名",
                    "aliases": ["别名1", ...],
                    "skill_type": "技能类别",
                    "notes": ""
                }

        说明:
            为最大化 vLLM 吞吐量，本方法会先收集所有中类的 prompt，
            然后一次性提交批量推理，而非逐个中类推理。这充分利用了
            vLLM 的 continuous batching 和 PagedAttention 机制。
        """
        from .merge import extract_json_from_response

        logger.info("开始批量提取所有中类的硬技能...")

        # 收集所有 prompt（一次性提交，充分利用 vLLM continuous batching）
        all_prompt_pairs: List[Tuple[str, str]] = []
        # 记录每个中类对应的 prompt 索引范围: (category_name, start, end)
        prompt_category_map: List[Tuple[str, int, int]] = []

        for category_name, samples in category_samples.items():
            train_texts = samples["train_texts"]
            if not train_texts:
                continue

            start_idx = len(all_prompt_pairs)
            for chunk_start in range(0, len(train_texts), MAX_SAMPLES_PER_PROMPT):
                batch_texts = train_texts[
                    chunk_start : chunk_start + MAX_SAMPLES_PER_PROMPT
                ]
                samples_block = "\n".join(
                    f"{i + 1}. {text[:MAX_SAMPLE_CHARS]}"
                    for i, text in enumerate(batch_texts)
                )
                user_prompt = FLAT_TRAINING_USER_TEMPLATE.format(
                    category_name=category_name,
                    count=len(batch_texts),
                    samples_text=samples_block,
                )
                all_prompt_pairs.append(
                    (FLAT_TRAINING_SYSTEM_PROMPT, user_prompt)
                )
            end_idx = len(all_prompt_pairs)
            prompt_category_map.append((category_name, start_idx, end_idx))

        if not all_prompt_pairs:
            logger.warning("没有可用的训练 prompt")
            return []

        # 一次性批量推理
        raw_outputs = self._vllm_batch_generate(all_prompt_pairs)

        # 按中类解析结果
        all_skills: List[Dict] = []
        for category_name, start_idx, end_idx in prompt_category_map:
            category_skills: List[Dict] = []
            for idx in range(start_idx, end_idx):
                parsed = extract_json_from_response(raw_outputs[idx])
                if parsed is None:
                    logger.warning(
                        "中类 '%s': prompt #%d JSON 解析失败",
                        category_name,
                        idx - start_idx + 1,
                    )
                    continue
                # 兼容 LLM 直接输出数组 [{...}] 或包装对象 {"skills":[...]}
                if isinstance(parsed, list):
                    skills = parsed
                elif isinstance(parsed, dict):
                    skills = parsed.get("skills", [])
                else:
                    skills = []
                category_skills.extend(skills)

            deduped = self._merge_skills_by_name(category_skills)
            logger.info(
                "中类 '%s': 提取到 %d 个技能", category_name, len(deduped)
            )
            all_skills.extend(deduped)

        # 全局去重
        merged = self._merge_skills_by_name(all_skills)
        logger.info("所有中类合计提取技能: %d 个（去重后）", len(merged))
        return merged

    # ── 覆盖率评估与补词 ─────────────────────────────────────────

    def _evaluate_and_supplement_all(
        self,
        category_samples: Dict[str, Dict[str, List[str]]],
        current_skills: List[Dict],
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    ) -> List[Dict]:
        """对所有中类进行覆盖率评估并收集缺失技能。

        遍历每个中类的验证样本，使用 vLLM 评估当前词典的覆盖率。
        对于覆盖率不足的中类，提取缺失技能并汇总返回。

        参数:
            category_samples: 来自 ``_load_and_sample`` 的采样结果。
            current_skills: 当前词典中的全部技能列表。
            coverage_threshold: 覆盖率阈值（0~1），低于此值时提取缺失技能。

        返回:
            list[dict]: 所有中类汇总的缺失技能列表。
                如果所有中类均达标，返回空列表。

        说明:
            与提取阶段类似，本方法也会先收集所有中类的评估 prompt，
            然后一次性提交 vLLM 批量推理。
        """
        from .merge import extract_json_from_response

        logger.info("开始覆盖率评估（阈值 %.1f%%）...", coverage_threshold * 100)

        eval_prompt_pairs: List[Tuple[str, str]] = []
        eval_categories: List[str] = []

        # 构建技能摘要文本（所有中类共用同一份词典摘要）
        skill_lines = []
        for skill in current_skills[:200]:  # 限制摘要长度，避免超上下文
            aliases = skill.get("aliases", [])
            alias_text = (
                f" (别名: {', '.join(aliases[:5])})" if aliases else ""
            )
            skill_lines.append(f"- {skill['name']}{alias_text}")
        skill_summary = (
            "\n".join(skill_lines) if skill_lines else "(词典为空)"
        )

        for category_name, samples in category_samples.items():
            validation_texts = samples.get("validation_texts", [])
            if not validation_texts:
                continue

            validation_block = "\n".join(
                f"{i + 1}. {text[:MAX_SAMPLE_CHARS]}"
                for i, text in enumerate(validation_texts)
            )
            user_prompt = FLAT_EVALUATION_USER_TEMPLATE.format(
                skill_count=len(current_skills),
                skill_summary=skill_summary,
                sample_count=len(validation_texts),
                validation_text=validation_block,
            )
            eval_prompt_pairs.append(
                (FLAT_EVALUATION_SYSTEM_PROMPT, user_prompt)
            )
            eval_categories.append(category_name)

        if not eval_prompt_pairs:
            logger.info("没有可用的验证样本")
            return []

        # 一次性批量推理
        raw_outputs = self._vllm_batch_generate(eval_prompt_pairs)

        # 解析并收集缺失技能
        all_missing: List[Dict] = []
        passed_count = 0
        failed_count = 0

        for category_name, raw_text in zip(eval_categories, raw_outputs):
            parsed = extract_json_from_response(raw_text)
            if parsed is None or not isinstance(parsed, dict):
                logger.warning(
                    "中类 '%s': 评估结果 JSON 解析失败或格式异常", category_name
                )
                failed_count += 1
                continue

            accuracy = float(parsed.get("accuracy", 1.0))
            missing_skills = parsed.get("missing_skills", [])

            logger.info(
                "中类 '%s': 覆盖率 %.1f%%, 缺失 %d 个技能",
                category_name,
                accuracy * 100,
                len(missing_skills),
            )

            if accuracy >= coverage_threshold:
                passed_count += 1
            else:
                failed_count += 1
                all_missing.extend(missing_skills)

        logger.info(
            "覆盖率评估完成: %d 个中类达标, %d 个未达标, 共发现 %d 个缺失技能",
            passed_count,
            failed_count,
            len(all_missing),
        )
        return all_missing

    # ── 技能去重与标准化 ─────────────────────────────────────────

    @staticmethod
    def _normalize_skill_key(name: str) -> str:
        """将技能名称标准化为可比较的 key。

        标准化规则:
            1. 转小写
            2. 去除首尾空白
            3. 折叠内部连续空白为单个空格
            4. 保留 ``+``, ``#``, ``.`` 等对编程语言重要的字符

        参数:
            name: 原始技能名称。

        返回:
            str: 标准化后的名称（全小写、空白折叠）。

        示例:
            >>> FlatSkillPipeline._normalize_skill_key("  C++  编程  ")
            'c++ 编程'
        """
        import re as _re

        key = str(name).strip().lower()
        key = _re.sub(r"\s+", " ", key)
        # 对纯 ASCII 技能名进一步折叠空格，减少 "AUTO CAD" / "AutoCAD" 这类重复。
        if _re.fullmatch(r"[a-z0-9 .+#/\-]+", key):
            key = key.replace(" ", "")
        return key

    @staticmethod
    def _is_low_value_skill_name(name: str) -> bool:
        """判断技能名称是否过泛，不适合作为词典主名称。

        被过滤的典型条目包括：
            - 泛能力/泛知识：如“机械传动知识”“理论基础”
            - 泛工具容器：如“数据分析工具”“仿真软件”
            - 泛证书容器：如“资格证书”“执业资格证书”
        """
        import re as _re

        text = str(name).strip()
        if not text:
            return True
        return any(_re.search(pattern, text) for pattern in LOW_VALUE_SKILL_PATTERNS)

    @staticmethod
    def _is_low_value_alias(alias: str) -> bool:
        """判断 alias 是否过泛，过泛 alias 不应进入词典。"""
        import re as _re

        text = str(alias).strip()
        if not text:
            return True
        normalized = FlatSkillPipeline._normalize_skill_key(text)
        if len(normalized) <= 2 and not _re.fullmatch(r"[a-z0-9.+#/\-]+", normalized):
            return True
        return any(_re.fullmatch(pattern, text) for pattern in LOW_VALUE_ALIAS_PATTERNS)

    @staticmethod
    def _clean_skill_record(skill: Dict) -> Dict | None:
        """标准化并清洗单条技能记录。

        返回 ``None`` 表示该技能过泛或为空，应直接丢弃。
        """
        name = str(skill.get("name", "")).strip()
        if FlatSkillPipeline._is_low_value_skill_name(name):
            return None

        aliases = []
        for alias in skill.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if not alias_text:
                continue
            if FlatSkillPipeline._is_low_value_alias(alias_text):
                continue
            aliases.append(alias_text)

        return {
            "name": name,
            "aliases": aliases,
            "skill_type": str(skill.get("skill_type", "")).strip(),
            "notes": str(skill.get("notes", "")).strip(),
        }

    @staticmethod
    def _filter_skill_records(skills: List[Dict]) -> List[Dict]:
        """对技能列表执行统一清洗，移除低质量词条。"""
        filtered: List[Dict] = []
        for skill in skills:
            cleaned = FlatSkillPipeline._clean_skill_record(skill)
            if cleaned is not None:
                filtered.append(cleaned)
        return filtered

    @staticmethod
    def _merge_skills_by_name(skills: List[Dict]) -> List[Dict]:
        """按标准化名称去重合并技能列表。

        相同标准化名称的技能会被合并:
            - 保留第一次出现的 name 作为主名称
            - 后续出现的 name 和 aliases 并入主记录的 aliases
            - notes 和 skill_type 合并（非空优先）

        参数:
            skills: 待去重的技能列表。

        返回:
            list[dict]: 去重合并后的技能列表，保持首次出现的顺序。
        """
        merged_map: Dict[str, Dict] = {}  # normalized_key -> skill dict
        order: List[str] = []  # 保持插入顺序

        for raw_skill in skills:
            skill = FlatSkillPipeline._clean_skill_record(raw_skill)
            if skill is None:
                continue
            name = skill["name"]

            key = FlatSkillPipeline._normalize_skill_key(name)
            aliases = skill["aliases"]
            skill_type = skill["skill_type"]
            notes = skill["notes"]

            if key not in merged_map:
                merged_map[key] = {
                    "name": name,
                    "aliases": list(aliases),
                    "skill_type": skill_type,
                    "notes": notes,
                }
                order.append(key)
            else:
                existing = merged_map[key]

                # 将新出现的别名并入
                existing_name_key = FlatSkillPipeline._normalize_skill_key(
                    existing["name"]
                )
                existing_alias_keys = {
                    FlatSkillPipeline._normalize_skill_key(a)
                    for a in existing["aliases"]
                }

                # 新名称若与主名称不同，加入 aliases
                if key != existing_name_key:
                    if key not in existing_alias_keys:
                        existing["aliases"].append(name)
                        existing_alias_keys.add(key)

                for alias in aliases:
                    alias_key = FlatSkillPipeline._normalize_skill_key(alias)
                    if (
                        alias_key != existing_name_key
                        and alias_key not in existing_alias_keys
                    ):
                        existing["aliases"].append(alias)
                        existing_alias_keys.add(alias_key)

                # 补充 skill_type / notes（非空优先）
                if skill_type and not existing["skill_type"]:
                    existing["skill_type"] = skill_type
                if notes and notes not in existing.get("notes", ""):
                    if existing["notes"]:
                        existing["notes"] += "; " + notes
                    else:
                        existing["notes"] = notes

        # 去重 aliases
        result: List[Dict] = []
        for key in order:
            skill = merged_map[key]
            seen = {FlatSkillPipeline._normalize_skill_key(skill["name"])}
            unique_aliases = []
            for alias in skill["aliases"]:
                alias_key = FlatSkillPipeline._normalize_skill_key(alias)
                if alias_key not in seen:
                    seen.add(alias_key)
                    unique_aliases.append(alias)
            skill["aliases"] = sorted(unique_aliases)
            result.append(skill)

        return result

    @staticmethod
    def _deduplicate_final(skills: List[Dict]) -> List[Dict]:
        """全局去重：确保所有 name 和 aliases 互不冲突。

        检测并解决以下冲突:
            1. 技能 A 的 name 出现在技能 B 的 aliases 中
               → 从 B 的 aliases 中移除。
            2. 两个技能共享相同的 alias
               → 仅保留在第一个技能中。
            3. 技能 A 的 alias 等于技能 B 的 name
               → 从 A 的 aliases 中移除。

        参数:
            skills: 待去重的技能列表。

        返回:
            list[dict]: 全局去重后的技能列表，确保 name 和 aliases
            构成的全集无重复（用于正则匹配时不会产生歧义）。
        """
        import copy as _copy

        result = _copy.deepcopy(
            FlatSkillPipeline._filter_skill_records(skills)
        )

        # 建立 name 索引
        name_keys: set = set()
        for skill in result:
            name_keys.add(
                FlatSkillPipeline._normalize_skill_key(skill["name"])
            )

        # 全局 alias 去重: name 占用的 key 不能被 alias 使用
        used_alias_keys: set = set(name_keys)

        for skill in result:
            name_key = FlatSkillPipeline._normalize_skill_key(skill["name"])
            cleaned_aliases = []
            for alias in skill.get("aliases", []):
                alias_key = FlatSkillPipeline._normalize_skill_key(alias)
                if alias_key == name_key:
                    continue  # alias 与自身 name 相同
                if alias_key in used_alias_keys:
                    continue  # alias 已被其他技能的 name 或 alias 占用
                used_alias_keys.add(alias_key)
                cleaned_aliases.append(alias)
            skill["aliases"] = sorted(cleaned_aliases)

        return result

    # ── 输出 ─────────────────────────────────────────────────────

    def _save_dictionary(
        self, skills: List[Dict], output_path: Path | str
    ) -> None:
        """保存平面化技能词典为 JSON 文件。

        输出格式::

            {
                "metadata": {
                    "schema_version": 3,
                    "created_at": "2026-04-12T10:00:00",
                    "pipeline": "FlatSkillPipeline_v2",
                    "model": "Qwen3-8B",
                    "skill_count": 5000,
                    "alias_count": 12000
                },
                "skills": [
                    {
                        "name": "...",
                        "aliases": [...],
                        "skill_type": "...",
                        "notes": ""
                    },
                    ...
                ]
            }

        参数:
            skills: 最终的技能列表。
            output_path: 输出 JSON 文件路径。
        """
        skills = self._deduplicate_final(skills)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_aliases = sum(len(s.get("aliases", [])) for s in skills)

        data = {
            "metadata": {
                "schema_version": 3,
                "created_at": _now_text(),
                "pipeline": "FlatSkillPipeline_v2",
                "model": self.model_path,
                "skill_count": len(skills),
                "alias_count": total_aliases,
            },
            "skills": skills,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "词典已保存: %s (%d 技能, %d 别名)",
            output_path,
            len(skills),
            total_aliases,
        )

    # ── 主流程编排 ───────────────────────────────────────────────

    def run(
        self,
        train_size: int = 100,
        validation_size: int = 10,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
        seed: int = 42,
        limit_job_rows: int | None = None,
        limit_categories: int | None = None,
        parse_workers: int = 1,
        output_path: str | Path | None = None,
    ) -> None:
        """执行完整的平面化技能词典构建流程。

        编排以下步骤:
            1. 数据加载与采样（按职业中类）
            2. vLLM 批量推理提取硬技能
            3. 覆盖率评估与缺失技能补充
            4. 全局去重
            5. 保存词典

        参数:
            train_size: 每个中类的训练样本数。
            validation_size: 每个中类的验证样本数。
            coverage_threshold: 覆盖率阈值（0~1），低于此值时补充缺失技能。
            seed: 随机种子。
            limit_job_rows: 调试用，限制读取的岗位行数。
            limit_categories: 调试用，限制处理的中类数量。
            parse_workers: 岗位描述解析并发数。
            output_path: 输出词典路径。
                默认为 ``config.dict_dir / "flat_skill_dictionary.json"``。

        流程概览::

            数据加载 → 分组采样 → vLLM提取 → 合并去重
                                                  ↓
                                            覆盖率评估
                                                  ↓
                                    ┌─ 达标 → 保存词典
                                    └─ 未达标 → 补充缺失 → 再次去重 → 保存词典
        """
        if output_path is None:
            output_path = self.config.dict_dir / "flat_skill_dictionary.json"
        output_path = Path(output_path)

        logger.info("=" * 60)
        logger.info("  FlatSkillPipeline v2 — 平面化技能词典构建")
        logger.info("  模型: %s", self.model_path)
        logger.info(
            "  训练样本: %d/中类, 验证样本: %d/中类",
            train_size,
            validation_size,
        )
        logger.info("  覆盖率阈值: %.1f%%", coverage_threshold * 100)
        logger.info("=" * 60)

        # ── Step 1: 数据加载与采样 ──
        logger.info("[Step 1/5] 数据加载与采样...")
        category_samples = self._load_and_sample(
            train_size=train_size,
            validation_size=validation_size,
            seed=seed,
            limit_job_rows=limit_job_rows,
            limit_categories=limit_categories,
            parse_workers=parse_workers,
        )

        if not category_samples:
            logger.error("没有可用的采样数据，流程终止")
            return

        # ── Step 2: vLLM 批量提取硬技能 ──
        logger.info("[Step 2/5] vLLM 批量提取硬技能...")
        all_skills = self._extract_skills_from_all_categories(
            category_samples
        )

        if not all_skills:
            logger.error("未提取到任何技能，流程终止")
            return

        logger.info("初步提取技能数: %d", len(all_skills))

        # ── Step 3: 覆盖率评估与缺失技能补充 ──
        logger.info("[Step 3/5] 覆盖率评估与缺失技能补充...")
        missing_skills = self._evaluate_and_supplement_all(
            category_samples=category_samples,
            current_skills=all_skills,
            coverage_threshold=coverage_threshold,
        )

        if missing_skills:
            logger.info("补充缺失技能 %d 个", len(missing_skills))
            all_skills.extend(missing_skills)
            all_skills = self._merge_skills_by_name(all_skills)
            logger.info("补充后技能数: %d", len(all_skills))

        # ── Step 4: 全局去重 ──
        logger.info("[Step 4/5] 全局去重（确保 name + aliases 唯一）...")
        final_skills = self._deduplicate_final(all_skills)
        logger.info("最终技能数: %d", len(final_skills))

        # ── Step 5: 保存词典 ──
        logger.info("[Step 5/5] 保存词典...")
        self._save_dictionary(final_skills, output_path)

        total_aliases = sum(
            len(s.get("aliases", [])) for s in final_skills
        )
        logger.info("=" * 60)
        logger.info("  流程完成！")
        logger.info("  技能数: %d", len(final_skills))
        logger.info("  别名数: %d", total_aliases)
        logger.info("  输出文件: %s", output_path)
        logger.info("=" * 60)


