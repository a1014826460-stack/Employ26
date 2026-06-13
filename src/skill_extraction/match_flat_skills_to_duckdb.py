"""
平面化技能词典匹配 + LLM 验证脚本。

使用 ``dicts/flat_skill_dictionary.json``（平面化技能词典）对 DuckDB 表
``recruit.main.skill_extraction_requirement_matches`` 中的岗位文本做混合
正则 / 包含匹配，提取硬技能名称，结果写入
``recruit.main.hard_skill_match_results_dev``。

匹配文本优先级：
    1. ``任职要求_items_text`` — 优先使用（最贴近技能要求）
    2. ``岗位职责_items_text`` — 上述为空时回退
    3. ``岗位描述_清洗``       — 最后兜底

可选功能 ── LLM 验证（需 vLLM + Qwen3-8B）：
    对匹配结果随机抽样，使用本地 Qwen3 评估匹配质量：
    - 发现错误匹配 → 自动删除导致误匹配的 alias
    - 发现遗漏硬技能 → 自动补充到词典
    - 发现 alias 归属错误 → 自动修正

依赖已有模块（不重复实现）：
    - ``match_hard_skills_to_duckdb``: 归一化、切分、过滤等工具函数与常量
    - ``merge_similar_skills``: vLLM 引擎初始化、JSON 鲁棒提取

用法示例::

    # 仅匹配（不需要 GPU / vLLM）
    python -m src.skill_extraction.match_flat_skills_to_duckdb match

    # 匹配 + LLM 验证（需要 GPU）
    python -m src.skill_extraction.match_flat_skills_to_duckdb run \\
        --model D:/model/Qwen3-8B

    # 仅验证已有匹配结果（需要 GPU）
    python -m src.skill_extraction.match_flat_skills_to_duckdb validate \\
        --model D:/model/Qwen3-8B
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd

# ── 复用已有模块的工具函数（遵循「请勿重复生成函数」原则）──────────────
from .history.match_hard_skills_to_duckdb import (
    ALIAS_CANONICAL_MAP,
    GENERIC_SKILL_NAME_PATTERNS,
    HardSkillMatcher,
    TermEntry,
    _safe_text,
    create_result_table,
    normalize_match_text,
    safe_lower_text,
    split_items,
)
from .config import load_skill_extraction_config
from .iteration_rules import (
    get_canonical_output_overrides,
    get_contextual_term_rules,
    get_exact_generic_skill_blocklist,
    get_short_chinese_allowlist,
)
from .llm_labeling_utils import run_openai_prompt_pairs

# ============================================================================
#  日志配置
# ============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ============================================================================
#  常量
# ============================================================================

# 默认平面化词典路径
DEFAULT_FLAT_DICT_PATH = "dicts/flat_skill_dictionary.json"

# 默认输出表（dev 表，与正式表区分）
DEFAULT_OUTPUT_TABLE = "recruit.main.hard_skill_match_results_dev"

# 默认模型路径由 ``config/database.yaml`` 中的 ``LLM_model_path`` 提供。
DEFAULT_MODEL_PATH: str | None = None

# LLM 验证每条 prompt 包含的样本数
VALIDATION_SAMPLES_PER_PROMPT: int = 5

# LLM 验证每条样本文本的最大字符数
VALIDATION_MAX_TEXT_CHARS: int = 500

# LLM 验证默认抽样数量
DEFAULT_VALIDATION_SAMPLE_SIZE: int = 50
STRONG_REVALIDATION_PRECISION_THRESHOLD: float = 0.95
STRONG_REVALIDATION_PARSE_SUCCESS_THRESHOLD: float = 0.90
STRONG_REVALIDATION_MIN_SAMPLES: int = 20

# 词项最小长度阈值：过短的词项噪音极大，需跳过。
# 中文词项归一化后至少 3 字符（如"焊接"=2 太短，"焊接工艺"=4 可以）；
# ASCII 词项至少 2 字符（如 "C"=1 太短，"C++"=3 可以）。
MIN_CHINESE_TERM_LEN: int = 3
MIN_ASCII_TERM_LEN: int = 2

# 黑名单：这些词条即使出现在词典中也不应作为匹配结果输出。
# 它们要么是福利待遇、要么过于泛化，无法作为可统计的硬技能。
SKILL_BLACKLIST: set = {
    "五险一金", "社保", "双休", "带薪年假", "节日福利",
    "材料", "电脑", "测试", "检测", "英语", "普通话",
    "包装", "分拣", "升华", "扫码", "贴标", "质检",
    "电源", "光源", "镜头", "离心", "客户服务",
    "仿真软件", "数据分析工具", "机械传动知识",
    "专业知识", "理论基础", "知识基础",
}

# 宽泛 alias 黑名单：这些 alias 过于短或泛化，容易导致系统性误匹配。
# 典型案例："资格证" → "教师资格证"（任何含"资格证"的文本都会误匹配为教师资格证）。
BROAD_ALIAS_BLACKLIST: set = {
    "资格证", "品质", "函数", "电机", "模具", "印刷",
    "相机", "奶粉", "辅食", "消毒", "车辆", "排版",
    "打包", "催化", "氧化", "色彩", "节奏", "光照",
    "京东", "淘宝", "快团",
}

# 这些模式用于进一步过滤“看起来像技能、实际上过于泛化”的输出项。
# 目标是拦截“测试”“仿真软件”“资格证书”“数据分析工具”这一类容器词。
LOW_VALUE_SKILL_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(能力|素养|基础|知识|理论)$"),
    re.compile(r"(工具|软件|系统|平台)$"),
    re.compile(r"^(数据分析工具|仿真软件|测试仪器|办公软件)$"),
    re.compile(r"^(资格证|资格证书|执业资格证书|上岗证|证书)$"),
]

# 宽泛 alias 的问题通常比宽泛主词更严重，因为它们会把整类文本都吸附到某个具体技能。
# 这里补充正则规则，覆盖“资格证/证书/软件/工具/知识/能力”等高风险 alias。
LOW_VALUE_ALIAS_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^(资格|资格证|资格证书|证书|执业证|上岗证|许可证|执照)$"),
    re.compile(r"^(工具|软件|系统|平台|知识|理论|能力|测试)$"),
    re.compile(r".*(资格证|资格证书|执业证|执业资格证书)$"),
]

EXACT_GENERIC_SKILL_BLOCKLIST: set[str] = get_exact_generic_skill_blocklist()
CANONICAL_OUTPUT_OVERRIDES: Dict[str, str] = get_canonical_output_overrides()
SHORT_CHINESE_ALLOWLIST: set[str] = get_short_chinese_allowlist()


def _compile_contextual_term_rules() -> Dict[str, List[Dict]]:
    compiled: Dict[str, List[Dict]] = {}
    for item in get_contextual_term_rules():
        skill_name = _safe_text(item.get("skill_name", ""))
        if not skill_name:
            continue
        compiled.setdefault(skill_name.casefold(), []).append(
            {
                "match_terms": {
                    _safe_text(term).casefold()
                    for term in item.get("match_terms", [])
                    if _safe_text(term)
                },
                "require_any": [
                    re.compile(str(pattern), re.IGNORECASE)
                    for pattern in item.get("require_any", [])
                    if str(pattern).strip()
                ],
                "reject_if_any": [
                    re.compile(str(pattern), re.IGNORECASE)
                    for pattern in item.get("reject_if_any", [])
                    if str(pattern).strip()
                ],
            }
        )
    return compiled


CONTEXTUAL_TERM_RULES: Dict[str, List[Dict]] = _compile_contextual_term_rules()

# 结果表输出列
OUTPUT_COLUMNS = [
    "岗位名称",
    "岗位描述_清洗",
    "任职要求_items_text",
    "岗位职责_items_text",
    "sections_brief",
    "occupation_title",
    "occupation_code",
    "skill_name",
]

# ── LLM 验证 Prompt ────────────────────────────────────────────────────

VALIDATION_SYSTEM_PROMPT = """\
你是一名硬技能匹配结果审核专家。你的任务是验证正则匹配器从岗位文本中提取的硬技能是否正确。

## 审核规则

1. **正确匹配**：技能名确实在文本中被提到，且语义吻合。
2. **错误匹配**：
   - 技能名在文本中未出现（纯误匹配）。
   - 因 alias 导致错误匹配（例如 alias "PS" 匹配到了 "PS版" 中的 "PS"，
     但实际含义不是 Photoshop）。
   - 匹配到的是上下位关系不同的技能（例如匹配到 "Python" 但文本说的是 "Python Web"）。
3. **遗漏技能**：文本中明确提到了某个硬技能，但匹配结果中没有。
   - 只关注可标准化的硬技能（工具、软件、编程语言、框架、数据库、证书等）。
   - 忽略软素质、学历、年限等。
4. **alias 错误**：某个 alias 导致了系统性的错误匹配，需要从词典中移除。

## 输出格式

请 **只输出 JSON**，不要输出任何解释文字、markdown 标记或思考过程。

{"samples":[{"sample_index":0,"correct_skills":["Java","MySQL"],\
"wrong_skills":[{"skill":"错误技能名","reason":"误匹配原因"}],\
"missing_skills":[{"name":"遗漏技能名","skill_type":"类别","evidence":"文本中的证据片段"}],\
"alias_errors":[{"skill":"父技能名","alias":"问题alias","reason":"该alias不应映射到此技能"}]}]}
"""

VALIDATION_USER_TEMPLATE = """\
请审核以下 {count} 条岗位文本的硬技能匹配结果：

{samples_block}

请逐条审核，输出 JSON。
"""

STRONG_VALIDATION_SYSTEM_PROMPT = """\
你是一名高精度硬技能匹配审核专家。你将复核本地LLM给出的高准确率验证结果，避免本地模型自证正确。
要求：
1. 逐条判断匹配结果是否真的正确。
2. 重点检查泛词、alias误吸附、上下位误匹配、证书容器词误判。
3. 如果本地LLM判断过于乐观，应明确指出 wrong_skills、alias_errors、missing_skills。
4. 只输出 JSON，不要解释。
输出格式与普通验证阶段相同。
"""

VALIDATION_SAMPLE_TEMPLATE = """\
--- 样本 {index} ---
岗位文本：
{text}

匹配到的硬技能：{matched_skills}
"""


# ============================================================================
#  1. 词典加载
# ============================================================================

def should_trigger_strong_revalidation(summary: Dict) -> bool:
    """当本地验证结果过于乐观时，触发更强 API 模型复核。"""
    total_samples = int(summary.get("total_samples", 0) or 0)
    estimated_precision = float(summary.get("estimated_precision", 0.0) or 0.0)
    parse_success_rate = float(summary.get("parse_success_rate", 0.0) or 0.0)
    if total_samples < STRONG_REVALIDATION_MIN_SAMPLES:
        return False
    if estimated_precision >= STRONG_REVALIDATION_PRECISION_THRESHOLD and parse_success_rate >= STRONG_REVALIDATION_PARSE_SUCCESS_THRESHOLD:
        return True
    return False


def _build_strong_validation_prompts(samples: List[Dict]) -> List[Tuple[str, str]]:
    prompt_pairs: List[Tuple[str, str]] = []
    for start in range(0, len(samples), VALIDATION_SAMPLES_PER_PROMPT):
        batch = samples[start : start + VALIDATION_SAMPLES_PER_PROMPT]
        sample_blocks: List[str] = []
        for offset, sample in enumerate(batch):
            matched_skills = ", ".join(sample.get("matched_skills", [])) or "无"
            sample_blocks.append(
                VALIDATION_SAMPLE_TEMPLATE.format(
                    index=start + offset,
                    text=sample.get("text", "")[:VALIDATION_MAX_TEXT_CHARS],
                    matched_skills=matched_skills,
                )
            )
        prompt_pairs.append(
            (
                STRONG_VALIDATION_SYSTEM_PROMPT,
                VALIDATION_USER_TEMPLATE.format(count=len(batch), samples_block="\n\n".join(sample_blocks)),
            )
        )
    return prompt_pairs


def _summarize_validation_findings(raw_texts: List[str], total_samples: int, prompt_count: int) -> Dict:
    all_wrong: List[Dict] = []
    all_missing: List[Dict] = []
    all_alias_errors: List[Dict] = []
    parse_success = 0
    from .merge_similar_skills import extract_json_from_response

    for raw_text in raw_texts:
        parsed = extract_json_from_response(raw_text)
        if parsed is None or not isinstance(parsed, dict):
            continue
        parse_success += 1
        for sample_result in parsed.get("samples", []):
            for wrong in sample_result.get("wrong_skills", []):
                if isinstance(wrong, dict) and wrong.get("skill"):
                    all_wrong.append(wrong)
            for missing in sample_result.get("missing_skills", []):
                if isinstance(missing, dict) and missing.get("name"):
                    all_missing.append(missing)
            for alias_err in sample_result.get("alias_errors", []):
                if isinstance(alias_err, dict) and alias_err.get("alias"):
                    all_alias_errors.append(alias_err)

    wrong_summary = _deduplicate_findings(all_wrong, key_field="skill")
    missing_summary = _deduplicate_findings(all_missing, key_field="name")
    alias_error_summary = _deduplicate_alias_errors(all_alias_errors)
    return {
        "total_samples": total_samples,
        "prompt_count": prompt_count,
        "parse_success": parse_success,
        "parse_success_rate": parse_success / max(prompt_count, 1),
        "wrong_skills": wrong_summary,
        "missing_skills": missing_summary,
        "alias_errors": alias_error_summary,
        "estimated_precision": 1.0 - (len(wrong_summary) + len(alias_error_summary)) / max(total_samples, 1),
    }


def load_flat_dictionary(path: str | Path) -> Dict:
    """加载平面化技能词典 JSON。

    参数:
        path: 词典文件路径，应为 ``FlatSkillPipeline`` 产出的
              schema_version 3 格式文件。

    返回:
        dict: 完整词典数据，包含 ``metadata`` 和 ``skills`` 列表。

    异常:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: JSON 格式错误。

    示例::

        >>> data = load_flat_dictionary("dicts/flat_skill_dictionary.json")
        >>> len(data["skills"])
        5000
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"平面化技能词典文件不存在: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    skills = data.get("skills", [])
    logger.info("已加载平面化技能词典: %d 条技能, 来源: %s", len(skills), path)
    return data


def save_flat_dictionary(data: Dict, path: str | Path) -> None:
    """保存平面化技能词典为 JSON 文件。

    参数:
        data: 完整词典数据，包含 ``metadata`` 和 ``skills``。
        path: 输出路径。

    行为:
        - 自动创建父目录。
        - 使用 UTF-8 编码、2 空格缩进、不转义中文。
        - 更新 ``metadata.updated_at`` 时间戳。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data.setdefault("metadata", {})["updated_at"] = (
        datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("词典已保存: %s", out_path)


# ============================================================================
#  2. FlatHardSkillMatcher — 平面化词典匹配器
# ============================================================================

class FlatHardSkillMatcher:
    """基于平面化技能词典的硬技能匹配器。

    与 ``HardSkillMatcher``（按职业细类分层）不同，本类使用全局平面化
    技能列表进行匹配，不依赖 ``detail_path`` 进行职业细类范围限制。

    匹配策略（复用 ``match_hard_skills_to_duckdb`` 的混合匹配逻辑）:
        - 中文 / 混合技能词：归一化后做包含匹配。
        - 英文 / 缩写技能词：带单词边界约束的正则匹配。
        - 同时匹配 ``name`` 和全部 ``aliases``。

    参数:
        flat_dictionary: 平面化词典数据，至少包含 ``skills`` 列表。

    示例::

        >>> data = load_flat_dictionary("dicts/flat_skill_dictionary.json")
        >>> matcher = FlatHardSkillMatcher(data)
        >>> matcher.match_text("熟练掌握 Java 和 MySQL，了解 Redis")
        [{'skill_name': 'Java', 'category': 'programming_language'}, ...]
    """

    def __init__(self, flat_dictionary: Dict) -> None:
        """初始化匹配器，构建全局词项索引。

        参数:
            flat_dictionary: 包含 ``skills`` 列表的词典数据。
        """
        self.dictionary = flat_dictionary
        self.skills: List[Dict] = flat_dictionary.get("skills", [])
        self.term_index: List[TermEntry] = self._build_flat_term_index(
            self.skills
        )
        self._normalized_trie: Dict[str, Dict] = {}
        self._ascii_term_map: Dict[str, List[TermEntry]] = {}
        self._ascii_pattern: re.Pattern[str] | None = None
        self._build_recall_index(self.term_index)
        logger.info(
            "FlatHardSkillMatcher 初始化完成: %d 个技能, %d 个词项",
            len(self.skills),
            len(self.term_index),
        )

    def _build_flat_term_index(self, skills: List[Dict]) -> List[TermEntry]:
        """为全部技能构建平面化可匹配词项索引。

        遍历 ``skills`` 列表，将每个技能的 ``name`` 和 ``aliases``
        展开为 ``TermEntry`` 列表。按归一化后的长度降序排列，
        确保长词优先匹配，减少短词抢匹配导致的噪音。

        过滤规则（降低误匹配率）：
            - 跳过 ``SKILL_BLACKLIST`` 中的技能主名称。
            - 跳过 ``BROAD_ALIAS_BLACKLIST`` 中的宽泛 alias。
            - 中文词项归一化后长度 < ``MIN_CHINESE_TERM_LEN`` 时跳过。
            - ASCII 词项长度 < ``MIN_ASCII_TERM_LEN`` 时跳过。

        参数:
            skills: 技能字典列表，每个元素包含 ``name`` 和可选
                    ``aliases`` 字段。

        返回:
            list[TermEntry]: 按长度降序排列的全局词项索引。
        """
        entries: List[TermEntry] = []
        seen: set = set()
        skipped_blacklist = 0
        skipped_short = 0
        skipped_broad_alias = 0

        for skill in skills:
            skill_name = _safe_text(skill.get("name", ""))
            if not skill_name:
                continue

            # 跳过黑名单中的技能（整个 skill 被忽略）
            if skill_name in SKILL_BLACKLIST:
                skipped_blacklist += 1
                continue

            # 读取该技能的 category（可能为 None）
            skill_category = skill.get("category") or None

            # 收集 name + 全部 aliases
            terms: List[Tuple[str, str]] = [(skill_name, "name")]
            for alias in skill.get("aliases", []) or []:
                alias_text = _safe_text(alias)
                if alias_text:
                    terms.append((alias_text, "alias"))

            for term_text, term_role in terms:
                # 跳过宽泛 alias（如 "资格证" → "教师资格证" 的误匹配根源）
                if (
                    term_role == "alias"
                    and self._is_low_value_alias(term_text)
                ):
                    skipped_broad_alias += 1
                    continue

                is_ascii = HardSkillMatcher._is_ascii_like_term(term_text)
                normalized = normalize_match_text(term_text)

                # 跳过过短的词项（噪音极大）
                if is_ascii:
                    if len(term_text.strip()) < MIN_ASCII_TERM_LEN:
                        skipped_short += 1
                        continue
                else:
                    if len(normalized) < MIN_CHINESE_TERM_LEN and term_text not in SHORT_CHINESE_ALLOWLIST:
                        skipped_short += 1
                        continue

                key = (skill_name.casefold(), term_text.casefold())
                if key in seen:
                    continue
                seen.add(key)

                entries.append(
                    TermEntry(
                        skill_name=skill_name,
                        term_text=term_text,
                        term_role=term_role,
                        is_ascii_like=is_ascii,
                        normalized_term=normalized,
                        category=skill_category,
                    )
                )

        if skipped_blacklist or skipped_short or skipped_broad_alias:
            logger.info(
                "词项过滤: 跳过黑名单技能 %d, 过短词项 %d, 宽泛alias %d",
                skipped_blacklist,
                skipped_short,
                skipped_broad_alias,
            )

        # 长词优先，减少短词抢匹配
        entries.sort(
            key=lambda item: (
                len(item.normalized_term),
                len(item.term_text),
            ),
            reverse=True,
        )
        return entries

    def _build_recall_index(self, entries: List[TermEntry]) -> None:
        """构建高吞吐召回索引。

        实现策略：
            - ASCII 词项合并成一条带边界的大正则，降低逐词 ``re.search`` 开销。
            - 中文/混合词项构建归一化 Trie，一次线性扫描文本即可找出候选。

        这里仍然保留原有过滤规则和后处理逻辑，只替换召回层的数据结构。
        """
        ascii_term_map: Dict[str, List[TermEntry]] = {}
        normalized_trie: Dict[str, Dict] = {}

        for entry in entries:
            if entry.is_ascii_like:
                term_key = entry.term_text.lower()
                ascii_term_map.setdefault(term_key, []).append(entry)
                continue

            node = normalized_trie
            for char in entry.normalized_term:
                node = node.setdefault(char, {})
            node.setdefault("_entries", []).append(entry)

        ascii_terms = sorted(ascii_term_map.keys(), key=len, reverse=True)
        ascii_pattern: re.Pattern[str] | None = None
        if ascii_terms:
            ascii_pattern = re.compile(
                rf"(?<![a-z0-9])(?:{'|'.join(re.escape(term) for term in ascii_terms)})(?![a-z0-9])"
            )

        self._ascii_term_map = ascii_term_map
        self._ascii_pattern = ascii_pattern
        self._normalized_trie = normalized_trie

        logger.info(
            "已构建召回索引: ASCII词项 %d, 归一化Trie词项 %d",
            len(ascii_term_map),
            sum(1 for item in entries if not item.is_ascii_like),
        )

    @staticmethod
    def _is_low_value_alias(alias_text: str) -> bool:
        """判断 alias 是否过于宽泛，宽泛 alias 不应进入匹配索引。

        典型问题包括：
            - ``资格证`` 这类证书容器词，被错误映射成某个具体证书；
            - ``工具``、``软件``、``知识`` 这类泛称，导致整类文本误命中。
        """
        text = _safe_text(alias_text)
        if not text:
            return True
        if text in BROAD_ALIAS_BLACKLIST:
            return True
        if len(normalize_match_text(text)) <= 2 and not HardSkillMatcher._is_ascii_like_term(text):
            return True
        return any(pattern.fullmatch(text) for pattern in LOW_VALUE_ALIAS_PATTERNS)

    @staticmethod
    def _is_low_value_skill_name(skill_name: str) -> bool:
        """判断技能主名称是否过泛，不适合作为最终输出。

        这里过滤的是“不可直接统计或不可落地执行”的技能容器词，
        而不是所有抽象程度较高的术语。目的不是追求绝对召回，
        而是降低明显误报。
        """
        text = _safe_text(skill_name)
        if not text:
            return True
        if text in EXACT_GENERIC_SKILL_BLOCKLIST:
            return True
        if text in SKILL_BLACKLIST:
            return True
        return any(pattern.search(text) for pattern in LOW_VALUE_SKILL_PATTERNS)

    @staticmethod
    def _canonicalize_output_name(skill_name: str) -> str:
        text = _safe_text(skill_name)
        if not text:
            return ""
        return CANONICAL_OUTPUT_OVERRIDES.get(text.casefold(), text)

    def _resolve_output_skill_name(self, entry: TermEntry) -> str:
        """决定最终写入结果表的技能名。

        复用 ``HardSkillMatcher`` 的解析策略：
            1. 如果命中的是 skill 主名称，直接返回。
            2. 如果命中的是 alias，且父 skill 名过于泛化
               （如 "编程语言""数据库技术"），则优先返回 alias 的规范名。
            3. 否则返回父 skill 名，保持和词典主名称一致。

        参数:
            entry: 匹配到的词项。

        返回:
            str: 最终技能名。
        """
        if entry.term_role == "name":
            return self._canonicalize_output_name(entry.skill_name)

        if HardSkillMatcher._is_generic_skill_name(entry.skill_name):
            alias_name = HardSkillMatcher._canonicalize_alias(entry.term_text)
            if alias_name:
                return self._canonicalize_output_name(alias_name)

        return self._canonicalize_output_name(entry.skill_name)

    def _passes_contextual_term_rules(
        self,
        raw_text: str,
        entry: TermEntry,
        resolved_name: str,
    ) -> bool:
        rules = CONTEXTUAL_TERM_RULES.get(resolved_name.casefold(), [])
        if not rules:
            return True

        term_key = _safe_text(entry.term_text).casefold()
        applicable_rules = [
            rule
            for rule in rules
            if not rule["match_terms"] or term_key in rule["match_terms"]
        ]
        if not applicable_rules:
            return True

        for rule in applicable_rules:
            if any(pattern.search(raw_text) for pattern in rule["reject_if_any"]):
                continue
            if rule["require_any"] and not any(
                pattern.search(raw_text) for pattern in rule["require_any"]
            ):
                continue
            return True
        return False

    def _match_ascii_entries(self, raw_text: str) -> List[TermEntry]:
        """使用合并正则匹配 ASCII-like 候选词项。"""
        if not raw_text or self._ascii_pattern is None:
            return []

        matched_entries: List[TermEntry] = []
        for match in self._ascii_pattern.finditer(raw_text):
            term_key = match.group(0).lower()
            matched_entries.extend(self._ascii_term_map.get(term_key, []))
        return matched_entries

    def _match_normalized_entries(self, normalized_text: str) -> List[TermEntry]:
        """使用归一化 Trie 匹配中文/混合候选词项。"""
        if not normalized_text or not self._normalized_trie:
            return []

        matched_entries: List[TermEntry] = []
        text_length = len(normalized_text)

        for start_index in range(text_length):
            node = self._normalized_trie
            cursor = start_index
            longest_entries: List[TermEntry] | None = None

            while cursor < text_length:
                char = normalized_text[cursor]
                if char not in node:
                    break
                node = node[char]
                cursor += 1
                terminal_entries = node.get("_entries")
                if terminal_entries:
                    longest_entries = terminal_entries

            if longest_entries:
                matched_entries.extend(longest_entries)

        return matched_entries

    @staticmethod
    def _deduplicate_substring_matches(
        skill_names: List[str],
    ) -> List[str]:
        """去除子串重复匹配：若技能 A 的名称是技能 B 名称的子串，则只保留 B。

        典型场景：
            - "数据分析" 与 "数据分析能力" 同时命中 → 只保留 "数据分析能力"
            - "办公软件" 与 "Office办公软件" 同时命中 → 只保留 "Office办公软件"
            - "SQL" 与 "MySQL" → 保留两者（英文边界匹配已隔离，不算子串）
            - "CAD" 与 "AutoCAD" → 保留两者（同理）

        算法：
            对每对 (A, B)，如果 A 的归一化名称是 B 的归一化名称的
            真子串，则标记 A 为冗余。仅对中文/混合词项做子串检测，
            英文词项因已有边界匹配保护，不参与子串去重。

        参数:
            skill_names: 待去重的技能名列表。

        返回:
            list[str]: 去重后的技能名列表，保持原始顺序。
        """
        if len(skill_names) <= 1:
            return skill_names

        # 构建归一化映射
        normalized_map: List[Tuple[str, str, bool]] = []
        for name in skill_names:
            norm = normalize_match_text(name)
            is_ascii = HardSkillMatcher._is_ascii_like_term(name)
            normalized_map.append((name, norm, is_ascii))

        # 标记被更长匹配覆盖的短子串
        redundant: set = set()
        for i, (name_i, norm_i, ascii_i) in enumerate(normalized_map):
            if ascii_i:
                # 英文词项不参与子串去重（边界匹配已保护）
                continue
            if i in redundant:
                continue
            for j, (name_j, norm_j, ascii_j) in enumerate(normalized_map):
                if i == j or j in redundant or ascii_j:
                    continue
                # 如果 norm_i 是 norm_j 的真子串，标记 i 为冗余
                if (
                    len(norm_i) < len(norm_j)
                    and norm_i in norm_j
                ):
                    redundant.add(i)
                    break

        return [
            name
            for idx, (name, _, _) in enumerate(normalized_map)
            if idx not in redundant
        ]

    def match_text(self, text: str) -> List[Dict[str, str | None]]:
        """对单段文本做硬技能匹配，返回命中的技能列表。

        每个结果包含 ``skill_name`` 和 ``category`` 字段。
        ``category`` 来自词典中对应技能的 ``category`` 字段，
        未命中词典的候选值为 ``None``。

        匹配策略：
            - ASCII-like 词项（如 ``Java``, ``SQL``, ``C++``）：
              使用带单词边界的正则匹配，避免子串误匹配。
            - 中文 / 混合词项（如 ``质量管理体系``, ``财务分析``）：
              归一化后做字符串包含匹配。

        后处理：
            - 调用 ``_deduplicate_substring_matches`` 去除中文子串
              重复匹配（如 "数据分析" 被 "数据分析能力" 覆盖时移除前者）。

        参数:
            text: 待匹配的岗位文本。

        返回:
            list[dict]: 去重后的结果列表，每项包含
            ``skill_name``（str）和 ``category``（str | None）。
        """
        normalized_text = normalize_match_text(text)
        raw_text = safe_lower_text(text)
        if not normalized_text:
            return []

        candidates = self.match_candidates(text)
        matched_skill_names = [item["skill_name"] for item in candidates]
        deduped_names = self._deduplicate_substring_matches(matched_skill_names)
        keep_keys = {name.casefold() for name in deduped_names}
        return [
            {"skill_name": item["skill_name"], "category": item["category"]}
            for item in candidates
            if item["skill_name"].casefold() in keep_keys
        ]

    def match_candidates(self, text: str) -> List[Dict]:
        """返回候选技能及其命中词信息。

        该方法服务于二阶段链路：
            1. 先由词典召回候选技能；
            2. 再由上下文判别器判断候选是否保留。
        """
        normalized_text = normalize_match_text(text)
        raw_text = safe_lower_text(text)
        if not normalized_text:
            return []

        recall_entries = (
            self._match_normalized_entries(normalized_text)
            + self._match_ascii_entries(raw_text)
        )

        candidates: List[Dict] = []
        seen_skills: set = set()
        for entry in recall_entries:
            if not entry.normalized_term:
                continue

            resolved_name = self._resolve_output_skill_name(entry)
            if self._is_low_value_skill_name(resolved_name):
                continue
            if not self._passes_contextual_term_rules(text, entry, resolved_name):
                continue

            skill_key = resolved_name.casefold()
            if skill_key in seen_skills:
                continue
            seen_skills.add(skill_key)
            candidates.append(
                {
                    "skill_name": resolved_name,
                    "matched_term": entry.term_text,
                    "term_role": entry.term_role,
                    "category": entry.category,
                }
            )

        if not candidates:
            return []

        deduped_names = self._deduplicate_substring_matches(
            [item["skill_name"] for item in candidates]
        )
        keep_keys = {item.casefold() for item in deduped_names}
        return [
            item
            for item in candidates
            if item["skill_name"].casefold() in keep_keys
        ]

    def find_skill_by_name(self, name: str) -> Dict | None:
        """根据技能名或别名查找词典中的技能条目。

        参数:
            name: 技能名称（主名称或别名均可匹配）。

        返回:
            dict | None: 匹配到的技能字典，未找到返回 None。
        """
        key = name.strip().casefold()
        for skill in self.skills:
            if _safe_text(skill.get("name", "")).casefold() == key:
                return skill
            for alias in skill.get("aliases", []) or []:
                if _safe_text(alias).casefold() == key:
                    return skill
        return None


# ============================================================================
#  3. 数据加载与匹配
# ============================================================================

def _get_match_text(row: Dict) -> str:
    """按优先级从岗位行中提取用于匹配的文本。

    优先级：
        1. ``任职要求_items_text`` — 最精准的技能要求字段
        2. ``岗位职责_items_text`` — 回退字段
        3. ``岗位描述_清洗``       — 最后兜底

    参数:
        row: 岗位数据行（字典形式）。

    返回:
        str: 非空的匹配文本。如果所有字段均为空，返回空字符串。
    """
    text = _safe_text(row.get("任职要求_items_text", ""))
    if text:
        return text

    text = _safe_text(row.get("岗位职责_items_text", ""))
    if text:
        return text

    text = _safe_text(row.get("岗位描述_清洗", ""))
    return text


def fetch_flat_source_rows(
    conn: duckdb.DuckDBPyConnection,
    source_table: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """从 DuckDB 读取待匹配样本行。

    查询包含匹配所需的文本字段和岗位元信息字段。
    使用 ``TRY_CAST`` 和 ``COALESCE`` 保证在列不存在时优雅降级。

    参数:
        conn: 已连接的 DuckDB 连接。
        source_table: 源表全限定名。
        limit: 调试用，限制返回行数。

    返回:
        pd.DataFrame: 包含匹配所需字段的 DataFrame。
    """
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    query = f"""
        SELECT
            岗位名称,
            岗位描述_清洗,
            任职要求_items_text,
            岗位职责_items_text,
            sections_brief,
            occupation_title,
            occupation_code
        FROM {source_table}
        {limit_clause}
    """
    return conn.execute(query).df()


def match_flat_dataframe(
    source_df: pd.DataFrame,
    matcher: FlatHardSkillMatcher,
    context_classifier=None,
    context_batch_size: int = 64,
) -> pd.DataFrame:
    """对样本 DataFrame 做平面化词典硬技能匹配。

    逐行提取匹配文本（按优先级），使用 ``FlatHardSkillMatcher``
    进行匹配，将命中的技能名以 JSON 数组字符串存入 ``skill_name`` 列。

    对于切分后的结构化字段（如 ``任职要求_items_text``），
    按 ``|`` 分割为条目后 **逐条匹配** 并汇总去重。

    .. note::
        这里仍然保留“逐条匹配”的策略，但最终输出会再经过
        ``FlatHardSkillMatcher`` 的低价值技能过滤：
        这样可以兼顾两点：
            1. 不提前粗暴丢弃整条文本，避免漏掉混杂在学历/经验句中的有效技能；
            2. 又能在最终落表前去掉“证书容器词”“泛工具词”等明显噪声。

    参数:
        source_df: 从 ``fetch_flat_source_rows`` 获取的源数据。
        matcher: 已初始化的平面化词典匹配器。

    返回:
        pd.DataFrame: 包含原始字段 + ``skill_name`` JSON 列的结果表。
    """
    output_rows: List[Dict] = []
    row_candidates: List[Dict] = []

    for row_index, row in enumerate(source_df.to_dict(orient="records")):
        match_text = _get_match_text(row)
        items = split_items(match_text)
        local_candidates: List[Dict] = []

        recall_units = items if items else [match_text]
        for item_text in recall_units:
            for candidate in matcher.match_candidates(item_text):
                candidate_row = {
                    "row_index": row_index,
                    "text": item_text,
                    "job_title": _safe_text(row.get("岗位名称", "")),
                    "skill_name": candidate["skill_name"],
                    "matched_term": candidate["matched_term"],
                    "term_role": candidate["term_role"],
                }
                local_candidates.append(candidate_row)
                row_candidates.append(candidate_row)

        matched_skills: List[str] = []
        if context_classifier is None:
            seen_skills: set = set()
            for candidate in local_candidates:
                skill_name = candidate["skill_name"]
                skill_key = skill_name.casefold()
                if skill_key not in seen_skills:
                    seen_skills.add(skill_key)
                    matched_skills.append(skill_name)

        output_rows.append(
            {
                "岗位名称": _safe_text(row.get("岗位名称", "")),
                "岗位描述_清洗": _safe_text(row.get("岗位描述_清洗", "")),
                "任职要求_items_text": _safe_text(
                    row.get("任职要求_items_text", "")
                ),
                "岗位职责_items_text": _safe_text(
                    row.get("岗位职责_items_text", "")
                ),
                "sections_brief": _safe_text(row.get("sections_brief", "")),
                "occupation_title": _safe_text(
                    row.get("occupation_title", "")
                ),
                "occupation_code": _safe_text(
                    row.get("occupation_code", "")
                ),
                "skill_name": json.dumps(
                    matched_skills, ensure_ascii=False
                ),
            }
        )

    if context_classifier is not None and row_candidates:
        from .context_classifier import SkillContextCandidate

        classifier_inputs = [
            SkillContextCandidate(
                text=item["text"],
                skill_name=item["skill_name"],
                matched_term=item["matched_term"],
                term_role=item["term_role"],
                job_title=item["job_title"],
                sample_id=str(item["row_index"]),
            )
            for item in row_candidates
        ]
        prediction_rows = context_classifier.predict(
            classifier_inputs,
            batch_size=max(1, int(context_batch_size)),
        )

        matched_by_row: Dict[int, List[str]] = {}
        seen_per_row: Dict[int, set] = {}
        label_counter: Counter[str] = Counter()
        kept_count = 0
        for candidate, prediction in zip(row_candidates, prediction_rows):
            label_counter[str(prediction.get("label", ""))] += 1
            if not bool(prediction.get("keep", False)):
                continue
            row_index = int(candidate["row_index"])
            skill_name = candidate["skill_name"]
            skill_key = skill_name.casefold()
            seen_skills = seen_per_row.setdefault(row_index, set())
            if skill_key in seen_skills:
                continue
            seen_skills.add(skill_key)
            matched_by_row.setdefault(row_index, []).append(skill_name)
            kept_count += 1

        for row_index, output_row in enumerate(output_rows):
            output_row["skill_name"] = json.dumps(
                matched_by_row.get(row_index, []),
                ensure_ascii=False,
            )
        logger.info(
            "Context classifier kept %d/%d recalled candidates; labels=%s",
            kept_count,
            len(row_candidates),
            dict(label_counter),
        )

    result_df = pd.DataFrame(output_rows)
    if result_df.empty:
        result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    return result_df[OUTPUT_COLUMNS]


def write_flat_result_table(
    conn: duckdb.DuckDBPyConnection,
    result_df: pd.DataFrame,
    output_table: str,
) -> None:
    """将匹配结果写入 DuckDB 表。

    显式将 ``skill_name`` 列转为 DuckDB 的 ``JSON`` 类型，
    便于下游使用 ``json_extract`` 和数组展开等操作。

    参数:
        conn: DuckDB 连接。
        result_df: 匹配结果 DataFrame，必须包含 ``skill_name`` 列。
        output_table: 输出表的全限定名。
    """
    tmp_name = "tmp_flat_skill_match_results"
    conn.register(tmp_name, result_df)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT
            岗位名称,
            岗位描述_清洗,
            任职要求_items_text,
            岗位职责_items_text,
            sections_brief,
            occupation_title,
            occupation_code,
            CAST(skill_name AS JSON) AS skill_name
        FROM {tmp_name}
        """
    )
    conn.unregister(tmp_name)
    logger.info("结果已写入: %s", output_table)


# ============================================================================
#  4. LLM 验证
# ============================================================================

def _build_validation_prompts(
    samples: List[Dict],
) -> List[Tuple[str, str]]:
    """为验证样本构建 LLM prompt 对列表。

    将样本按 ``VALIDATION_SAMPLES_PER_PROMPT`` 分批打包，
    每批构建一个 ``(system_prompt, user_prompt)`` 元组。

    参数:
        samples: 验证样本列表，每个元素包含::

            {
                "text": "岗位文本",
                "matched_skills": ["Java", "MySQL", ...],
            }

    返回:
        list[tuple[str, str]]: prompt 对列表，用于提交给 vLLM 批量推理。
    """
    prompt_pairs: List[Tuple[str, str]] = []

    for chunk_start in range(0, len(samples), VALIDATION_SAMPLES_PER_PROMPT):
        batch = samples[
            chunk_start: chunk_start + VALIDATION_SAMPLES_PER_PROMPT
        ]
        sample_blocks = []
        for i, sample in enumerate(batch):
            text = sample["text"][:VALIDATION_MAX_TEXT_CHARS]
            skills_str = json.dumps(
                sample["matched_skills"], ensure_ascii=False
            )
            sample_blocks.append(
                VALIDATION_SAMPLE_TEMPLATE.format(
                    index=i, text=text, matched_skills=skills_str,
                )
            )

        user_prompt = VALIDATION_USER_TEMPLATE.format(
            count=len(batch),
            samples_block="\n".join(sample_blocks),
        )
        prompt_pairs.append((VALIDATION_SYSTEM_PROMPT, user_prompt))

    return prompt_pairs


def _collect_validation_samples(
    result_df: pd.DataFrame,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    seed: int = 42,
    min_skills: int = 1,
) -> List[Dict]:
    """从匹配结果中抽取验证样本。

    筛选条件：至少命中 ``min_skills`` 个技能的行。
    随机采样 ``sample_size`` 条，使用固定种子保证可复现。

    参数:
        result_df: 匹配结果 DataFrame，必须包含 ``skill_name`` 列。
        sample_size: 抽样数量。
        seed: 随机种子。
        min_skills: 最少命中技能数。

    返回:
        list[dict]: 验证样本列表，每个元素包含
        ``text`` 和 ``matched_skills``。
    """
    samples: List[Dict] = []

    for row in result_df.to_dict(orient="records"):
        skill_name_raw = row.get("skill_name", "[]")
        try:
            matched = json.loads(skill_name_raw)
        except (json.JSONDecodeError, TypeError):
            matched = []

        if len(matched) < min_skills:
            continue

        # 组装匹配文本
        text = _safe_text(row.get("任职要求_items_text", ""))
        if not text:
            text = _safe_text(row.get("岗位职责_items_text", ""))
        if not text:
            text = _safe_text(row.get("岗位描述_清洗", ""))
        if not text:
            continue

        samples.append(
            {"text": text, "matched_skills": matched, "row": row}
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    selected = samples[:sample_size]
    logger.info(
        "已从 %d 条候选中抽取 %d 条验证样本",
        len(samples),
        len(selected),
    )
    return selected


def validate_match_results(
    result_df: pd.DataFrame,
    model_path: str | None = DEFAULT_MODEL_PATH,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    seed: int = 42,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
) -> Dict:
    """使用本地 Qwen3 对匹配结果进行 LLM 验证。

    执行流程：
        1. 从匹配结果中随机抽取验证样本。
        2. 构建验证 prompt 并通过 vLLM 批量推理。
        3. 解析 LLM 输出，汇总错误匹配、遗漏技能和 alias 错误。

    参数:
        result_df: 匹配结果 DataFrame。
        model_path: Qwen3-8B 模型路径。
        sample_size: 抽样数量。
        seed: 随机种子。
        gpu_memory_utilization: GPU 显存利用率。
        max_model_len: vLLM 最大序列长度。
        max_num_seqs: vLLM 最大并发序列数。

    返回:
        dict: 验证结果汇总::

            {
                "total_samples": int,
                "prompt_count": int,
                "parse_success": int,
                "wrong_skills": [{"skill": str, "reason": str, "count": int}],
                "missing_skills": [{"name": str, "skill_type": str, "count": int}],
                "alias_errors": [{"skill": str, "alias": str, "reason": str}],
            }
    """
    resolved_model_path = str(model_path or load_skill_extraction_config().llm_model_path)

    # 延迟导入 vLLM 相关依赖（仅验证时需要）
    from .merge_similar_skills import init_vllm_engine, extract_json_from_response
    from vllm import SamplingParams

    # Step 1: 抽取验证样本
    samples = _collect_validation_samples(
        result_df, sample_size=sample_size, seed=seed,
    )
    if not samples:
        logger.warning("没有可用的验证样本")
        return {"total_samples": 0, "wrong_skills": [], "missing_skills": [], "alias_errors": []}

    # Step 2: 构建 prompt
    prompt_pairs = _build_validation_prompts(samples)
    logger.info("已构建 %d 个验证 prompt", len(prompt_pairs))

    # Step 3: vLLM 批量推理
    logger.info("正在初始化 vLLM 引擎进行验证...")
    llm = init_vllm_engine(
        model_path=resolved_model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    formatted_prompts: List[str] = []
    for system_prompt, user_prompt in prompt_pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt_text = (
                f"system: {system_prompt}\nuser: {user_prompt}\nassistant:"
            )
        formatted_prompts.append(prompt_text)

    logger.info("vLLM 验证推理: 共 %d 条 prompt", len(formatted_prompts))
    outputs = llm.generate(formatted_prompts, sampling_params)
    raw_texts = [output.outputs[0].text for output in outputs]
    logger.info("vLLM 验证推理完成")

    summary = _summarize_validation_findings(raw_texts, len(samples), len(prompt_pairs))

    logger.info(
        "验证汇总: %d 条样本, %d 个错误匹配, %d 个遗漏技能, %d 个 alias 错误",
        len(samples),
        len(summary["wrong_skills"]),
        len(summary["missing_skills"]),
        len(summary["alias_errors"]),
    )
    return summary


def run_strong_revalidation(
    result_df: pd.DataFrame,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    model: str = "openai/gpt-5.4",
    seed: int = 42,
) -> Dict:
    samples = _collect_validation_samples(result_df, sample_size=sample_size, seed=seed)
    if not samples:
        return {"total_samples": 0, "wrong_skills": [], "missing_skills": [], "alias_errors": []}
    prompt_pairs = _build_strong_validation_prompts(samples)
    outputs = run_openai_prompt_pairs(
        prompt_pairs=prompt_pairs,
        model=model,
        max_output_tokens=2048,
        reasoning_effort="medium",
    )
    summary = _summarize_validation_findings(outputs, len(samples), len(prompt_pairs))
    summary["review_model"] = model
    summary["review_stage"] = "strong_api_revalidation"
    return summary


def _deduplicate_findings(
    items: List[Dict], key_field: str,
) -> List[Dict]:
    """对验证发现去重并统计出现次数。

    参数:
        items: 原始发现列表。
        key_field: 用于去重的字段名。

    返回:
        list[dict]: 去重后的列表，每项附加 ``count`` 字段。
    """
    counter: Dict[str, Dict] = {}
    for item in items:
        key = _safe_text(item.get(key_field, "")).casefold()
        if not key:
            continue
        if key not in counter:
            counter[key] = {**item, "count": 1}
        else:
            counter[key]["count"] += 1
    return sorted(counter.values(), key=lambda x: x["count"], reverse=True)


def _deduplicate_alias_errors(items: List[Dict]) -> List[Dict]:
    """对 alias 错误去重。

    以 ``(skill, alias)`` 作为联合去重键。

    参数:
        items: 原始 alias 错误列表。

    返回:
        list[dict]: 去重后的 alias 错误列表。
    """
    seen: set = set()
    result: List[Dict] = []
    for item in items:
        skill = _safe_text(item.get("skill", "")).casefold()
        alias = _safe_text(item.get("alias", "")).casefold()
        key = (skill, alias)
        if key in seen or not alias:
            continue
        seen.add(key)
        result.append(item)
    return result


# ============================================================================
#  5. 词典自动修正
# ============================================================================

def apply_dictionary_corrections(
    flat_dictionary: Dict,
    validation_summary: Dict,
    min_wrong_count: int = 2,
    min_missing_count: int = 2,
) -> Dict:
    """根据 LLM 验证结果自动修正词典。

    修正策略：
        1. **错误 alias 删除**：对于 ``alias_errors`` 中的条目，
           从对应技能的 ``aliases`` 列表中移除问题 alias。
        2. **高频错误技能处理**：对于出现次数 ≥ ``min_wrong_count``
           的错误技能，如果有明确的 alias 导致了误匹配，
           也尝试移除该 alias。
        3. **遗漏技能补充**：对于出现次数 ≥ ``min_missing_count``
           的遗漏技能，添加到词典中。

    参数:
        flat_dictionary: 原始词典数据（会被深拷贝，不修改原始对象）。
        validation_summary: 来自 ``validate_match_results`` 的验证结果。
        min_wrong_count: 触发修正的最低错误出现次数。
        min_missing_count: 触发补充的最低遗漏出现次数。

    返回:
        dict: 修正后的词典数据（新副本）。
    """
    corrected = copy.deepcopy(flat_dictionary)
    skills: List[Dict] = corrected.get("skills", [])

    removed_aliases = 0
    added_skills = 0

    # ── 处理 alias 错误 ────────────────────────────────────────────
    alias_errors = validation_summary.get("alias_errors", [])
    for error in alias_errors:
        target_skill_name = _safe_text(error.get("skill", ""))
        problem_alias = _safe_text(error.get("alias", ""))
        if not target_skill_name or not problem_alias:
            continue

        for skill in skills:
            if _safe_text(skill.get("name", "")).casefold() != target_skill_name.casefold():
                continue

            original_aliases = skill.get("aliases", []) or []
            new_aliases = [
                a for a in original_aliases
                if _safe_text(a).casefold() != problem_alias.casefold()
            ]
            if len(new_aliases) < len(original_aliases):
                skill["aliases"] = new_aliases
                removed_aliases += 1
                logger.info(
                    "已移除 alias: '%s' (原属技能 '%s'), 原因: %s",
                    problem_alias,
                    target_skill_name,
                    error.get("reason", "未知"),
                )
            break

    # ── 处理高频错误匹配（尝试找到并移除导致误匹配的 alias）─────
    wrong_skills = validation_summary.get("wrong_skills", [])
    for wrong in wrong_skills:
        if wrong.get("count", 0) < min_wrong_count:
            continue

        wrong_name = _safe_text(wrong.get("skill", ""))
        if not wrong_name:
            continue

        # 如果 wrong_name 本身是某个技能的 alias，考虑移除
        for skill in skills:
            aliases = skill.get("aliases", []) or []
            for alias in aliases:
                if _safe_text(alias).casefold() == wrong_name.casefold():
                    skill["aliases"] = [
                        a for a in aliases
                        if _safe_text(a).casefold() != wrong_name.casefold()
                    ]
                    removed_aliases += 1
                    logger.info(
                        "已移除高频误匹配 alias: '%s' (原属技能 '%s')",
                        wrong_name,
                        skill.get("name", ""),
                    )
                    break

    # ── 补充遗漏技能 ──────────────────────────────────────────────
    missing_skills = validation_summary.get("missing_skills", [])
    # 构建已有技能名索引（含 aliases），用于去重
    existing_names: set = set()
    for skill in skills:
        existing_names.add(_safe_text(skill.get("name", "")).casefold())
        for alias in skill.get("aliases", []) or []:
            existing_names.add(_safe_text(alias).casefold())

    for missing in missing_skills:
        if missing.get("count", 0) < min_missing_count:
            continue

        name = _safe_text(missing.get("name", ""))
        if not name or name.casefold() in existing_names:
            continue

        new_skill = {
            "name": name,
            "aliases": [],
            "skill_type": _safe_text(missing.get("skill_type", "")),
            "notes": f"由 LLM 验证自动补充 ({datetime.now().strftime('%Y-%m-%d')})",
        }
        skills.append(new_skill)
        existing_names.add(name.casefold())
        added_skills += 1
        logger.info("已补充遗漏技能: '%s' (类型: %s)", name, new_skill["skill_type"])

    corrected["skills"] = skills

    # 更新 metadata
    metadata = corrected.setdefault("metadata", {})
    metadata["llm_validation_corrections"] = {
        "corrected_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "removed_aliases": removed_aliases,
        "added_skills": added_skills,
        "validation_sample_count": validation_summary.get("total_samples", 0),
    }

    logger.info(
        "词典修正完成: 移除 %d 个 alias, 补充 %d 个技能",
        removed_aliases,
        added_skills,
    )
    return corrected


# ============================================================================
#  6. 主流程编排
# ============================================================================

def run_match(
    dict_path: str | Path = DEFAULT_FLAT_DICT_PATH,
    source_table: str | None = None,
    output_table: str = DEFAULT_OUTPUT_TABLE,
    limit: int | None = None,
    context_classifier_model: str | None = None,
    context_threshold: float = 0.80,
    context_batch_size: int = 64,
) -> pd.DataFrame:
    """执行平面化词典匹配流程。

    步骤：
        1. 加载词典和源数据。
        2. 构建匹配器并逐行匹配。
        3. 将结果写入 DuckDB。

    参数:
        dict_path: 平面化词典路径。
        source_table: 源表名。None 时从配置读取。
        output_table: 输出表名。
        limit: 调试用行数限制。

    返回:
        pd.DataFrame: 匹配结果。
    """
    config = load_skill_extraction_config()
    source_table = source_table or config.requirement_match_table

    logger.info("=" * 60)
    logger.info("  平面化技能词典匹配")
    logger.info("  词典: %s", dict_path)
    logger.info("  源表: %s", source_table)
    logger.info("  输出表: %s", output_table)
    logger.info("=" * 60)

    flat_dict = load_flat_dictionary(dict_path)
    matcher = FlatHardSkillMatcher(flat_dict)
    context_classifier = None
    resolved_context_model = context_classifier_model
    if resolved_context_model:
        from .context_classifier import SkillContextClassifier

        context_classifier = SkillContextClassifier(
            model_path=resolved_context_model,
            threshold=context_threshold,
        )
        logger.info(
            "已启用上下文判别器: model=%s, threshold=%.2f",
            resolved_context_model,
            context_threshold,
        )

    with duckdb.connect(str(config.db_path)) as conn:
        conn.execute(f"PRAGMA threads={config.duckdb_threads}")
        source_df = fetch_flat_source_rows(
            conn=conn, source_table=source_table, limit=limit,
        )
        logger.info("已加载 %d 条源数据", len(source_df))

        result_df = match_flat_dataframe(
            source_df=source_df,
            matcher=matcher,
            context_classifier=context_classifier,
            context_batch_size=context_batch_size,
        )
        write_flat_result_table(
            conn=conn, result_df=result_df, output_table=output_table,
        )

    # 统计
    matched_count = 0
    total_skills = 0
    if not result_df.empty:
        for value in result_df["skill_name"].tolist():
            try:
                skills = json.loads(value) if isinstance(value, str) else []
            except json.JSONDecodeError:
                skills = []
            if skills:
                matched_count += 1
                total_skills += len(skills)

    logger.info("匹配完成:")
    logger.info("  处理样本数: %d", len(result_df))
    logger.info("  命中技能的样本数: %d (%.1f%%)",
                matched_count,
                matched_count / max(len(result_df), 1) * 100)
    logger.info("  技能命中总次数: %d", total_skills)
    logger.info("  平均每样本命中: %.1f 个",
                total_skills / max(matched_count, 1))

    return result_df


def run_validate(
    dict_path: str | Path = DEFAULT_FLAT_DICT_PATH,
    output_table: str = DEFAULT_OUTPUT_TABLE,
    model_path: str | None = DEFAULT_MODEL_PATH,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    seed: int = 42,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
    auto_correct: bool = True,
) -> Dict:
    """执行 LLM 验证并可选自动修正词典。

    步骤：
        1. 从已有匹配结果中抽样。
        3. 使用 vLLM 批量推理验证匹配质量。
        4. 当本地验证结果过于乐观时，调用更强 API 模型复核，防止“自证准确”。
        5. 根据验证结果修正词典（如果 ``auto_correct=True``）。
        6. 保存修正后的词典和验证报告。

    参数:
        dict_path: 词典路径。
        output_table: 匹配结果表名（用于读取已有结果）。
        model_path: Qwen3-8B 模型路径。
        sample_size: 验证抽样数量。
        seed: 随机种子。
        gpu_memory_utilization: GPU 显存利用率。
        max_model_len: vLLM 最大序列长度。
        max_num_seqs: vLLM 最大并发序列数。
        auto_correct: 是否自动修正词典。

    返回:
        dict: 验证结果汇总。
    """
    config = load_skill_extraction_config()

    resolved_model_path = str(model_path or config.llm_model_path)

    logger.info("=" * 60)
    logger.info("  LLM 验证匹配结果")
    logger.info("  词典: %s", dict_path)
    logger.info("  模型: %s", resolved_model_path)
    logger.info("  抽样数量: %d", sample_size)
    logger.info("=" * 60)

    # 读取已有匹配结果
    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        conn.execute(f"PRAGMA threads={config.duckdb_threads}")
        result_df = conn.execute(
            f"SELECT * FROM {output_table}"
        ).df()
    logger.info("已加载 %d 条匹配结果", len(result_df))

    # LLM 验证
    validation_summary = validate_match_results(
        result_df=result_df,
        model_path=resolved_model_path,
        sample_size=sample_size,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    # 保存验证报告
    report_dir = config.report_dir / "flat_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"validation_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, ensure_ascii=False, indent=2)
    logger.info("验证报告已保存: %s", report_path)

    if should_trigger_strong_revalidation(validation_summary):
        logger.info("本地验证准确率过高，触发更强 API 复核以防止自证准确")
        strong_summary = run_strong_revalidation(
            result_df=result_df,
            sample_size=sample_size,
            model=config.llm_strong_model,
            seed=seed,
        )
        strong_report_path = report_dir / f"validation_report_strong_{timestamp}.json"
        with open(strong_report_path, "w", encoding="utf-8") as f:
            json.dump(strong_summary, f, ensure_ascii=False, indent=2)
        validation_summary["strong_revalidation"] = strong_summary
        validation_summary["strong_revalidation_report_path"] = str(strong_report_path)
        logger.info("强复核报告已保存: %s", strong_report_path)

    # 自动修正词典
    if auto_correct and (
        validation_summary.get("wrong_skills")
        or validation_summary.get("missing_skills")
        or validation_summary.get("alias_errors")
    ):
        flat_dict = load_flat_dictionary(dict_path)
        corrected_dict = apply_dictionary_corrections(
            flat_dict, validation_summary,
        )

        # 保存为带时间戳的备份和覆盖原文件
        dict_path_obj = Path(dict_path)
        backup_path = (
            dict_path_obj.parent
            / f"{dict_path_obj.stem}_backup_{timestamp}{dict_path_obj.suffix}"
        )
        save_flat_dictionary(flat_dict, backup_path)
        logger.info("词典原版备份: %s", backup_path)

        save_flat_dictionary(corrected_dict, dict_path)
        logger.info("已覆盖更新词典: %s", dict_path)
    elif auto_correct:
        logger.info("验证未发现需要修正的问题，词典无变更")

    return validation_summary


def run_full(
    dict_path: str | Path = DEFAULT_FLAT_DICT_PATH,
    source_table: str | None = None,
    output_table: str = DEFAULT_OUTPUT_TABLE,
    model_path: str | None = DEFAULT_MODEL_PATH,
    limit: int | None = None,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    seed: int = 42,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
    auto_correct: bool = True,
    context_classifier_model: str | None = None,
    context_threshold: float = 0.80,
    context_batch_size: int = 64,
) -> None:
    """执行完整流程：匹配 → LLM 验证 → 词典修正。

    编排以下步骤：
        1. 加载词典并对全量岗位文本做硬技能匹配。
        2. 将匹配结果写入 DuckDB 开发表。
        3. 从结果中抽样，使用 vLLM + Qwen3 验证匹配质量。
        4. 根据验证结果自动修正词典（删除错误 alias、补充遗漏技能）。

    参数:
        dict_path: 平面化词典路径。
        source_table: DuckDB 源表名。
        output_table: 输出表名。
        model_path: Qwen3-8B 模型路径。
        limit: 调试用行数限制。
        sample_size: LLM 验证抽样数量。
        seed: 随机种子。
        gpu_memory_utilization: GPU 显存利用率。
        max_model_len: vLLM 最大序列长度。
        max_num_seqs: vLLM 最大并发序列数。
        auto_correct: 是否自动修正词典。
    """
    # Step 1: 匹配
    run_match(
        dict_path=dict_path,
        source_table=source_table,
        output_table=output_table,
        limit=limit,
        context_classifier_model=context_classifier_model,
        context_threshold=context_threshold,
        context_batch_size=context_batch_size,
    )

    # Step 2: LLM 验证 + 词典修正
    run_validate(
        dict_path=dict_path,
        output_table=output_table,
        model_path=model_path,
        sample_size=sample_size,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        auto_correct=auto_correct,
    )

    logger.info("=" * 60)
    logger.info("  全流程完成！")
    logger.info("=" * 60)


# ============================================================================
#  7. CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。

    支持的子命令：
        - ``match``:    仅执行正则匹配，不需要 GPU。
        - ``validate``: 仅执行 LLM 验证（需已有匹配结果），需要 GPU。
        - ``run``:      匹配 + 验证 + 词典修正的完整流程，需要 GPU。

    返回:
        argparse.ArgumentParser: 配置好的参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="平面化技能词典匹配 + LLM 验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
用法示例:
  # 仅匹配
  python -m src.skill_extraction.match_flat_skills_to_duckdb match

  # 完整流程（匹配 + LLM 验证 + 词典修正）
  python -m src.skill_extraction.match_flat_skills_to_duckdb run \\
      --model D:/model/Qwen3-8B

  # 仅验证
  python -m src.skill_extraction.match_flat_skills_to_duckdb validate \\
      --model D:/model/Qwen3-8B --sample-size 100
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── match ────────────────────────────────────────────────────────
    match_cmd = subparsers.add_parser(
        "match", help="仅执行正则匹配（不需要 GPU）",
    )
    match_cmd.add_argument(
        "--dictionary", default=DEFAULT_FLAT_DICT_PATH,
        help=f"平面化技能词典路径 (默认: {DEFAULT_FLAT_DICT_PATH})",
    )
    match_cmd.add_argument(
        "--source-table", default=None,
        help="DuckDB 源表名（默认从 config/database.yaml 读取）",
    )
    match_cmd.add_argument(
        "--output-table", default=DEFAULT_OUTPUT_TABLE,
        help=f"输出表名 (默认: {DEFAULT_OUTPUT_TABLE})",
    )
    match_cmd.add_argument(
        "--limit", type=int, default=None,
        help="调试用，限制处理行数",
    )
    match_cmd.add_argument(
        "--context-classifier-model", default=None,
        help="上下文判别器模型目录；提供后启用二阶段过滤",
    )
    match_cmd.add_argument(
        "--context-threshold", type=float, default=0.80,
        help="上下文判别器保留阈值 (默认: 0.80)",
    )
    match_cmd.add_argument(
        "--context-batch-size", type=int, default=64,
        help="上下文判别器推理 batch size (默认: 64)",
    )

    # ── validate ─────────────────────────────────────────────────────
    validate_cmd = subparsers.add_parser(
        "validate", help="仅执行 LLM 验证（需已有匹配结果和 GPU）",
    )
    validate_cmd.add_argument(
        "--dictionary", default=DEFAULT_FLAT_DICT_PATH,
        help=f"平面化技能词典路径 (默认: {DEFAULT_FLAT_DICT_PATH})",
    )
    validate_cmd.add_argument(
        "--output-table", default=DEFAULT_OUTPUT_TABLE,
        help=f"匹配结果表名 (默认: {DEFAULT_OUTPUT_TABLE})",
    )
    validate_cmd.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Qwen3-8B 模型路径（默认从 config/database.yaml 的 LLM_model_path 读取）",
    )
    validate_cmd.add_argument(
        "--sample-size", type=int, default=DEFAULT_VALIDATION_SAMPLE_SIZE,
        help=f"验证抽样数量 (默认: {DEFAULT_VALIDATION_SAMPLE_SIZE})",
    )
    validate_cmd.add_argument(
        "--seed", type=int, default=42, help="随机种子 (默认: 42)",
    )
    validate_cmd.add_argument(
        "--gpu-memory-utilization", type=float, default=0.80,
        help="GPU 显存利用率 (默认: 0.80)",
    )
    validate_cmd.add_argument(
        "--max-model-len", type=int, default=8192,
        help="vLLM 最大序列长度 (默认: 8192)",
    )
    validate_cmd.add_argument(
        "--max-num-seqs", type=int, default=48,
        help="vLLM 最大并发序列数 (默认: 48)",
    )
    validate_cmd.add_argument(
        "--no-auto-correct", action="store_true",
        help="禁用自动修正词典（仅输出报告）",
    )

    # ── run（完整流程）────────────────────────────────────────────
    run_cmd = subparsers.add_parser(
        "run", help="匹配 + LLM 验证 + 词典修正（完整流程，需要 GPU）",
    )
    run_cmd.add_argument(
        "--dictionary", default=DEFAULT_FLAT_DICT_PATH,
        help=f"平面化技能词典路径 (默认: {DEFAULT_FLAT_DICT_PATH})",
    )
    run_cmd.add_argument(
        "--source-table", default=None,
        help="DuckDB 源表名（默认从 config/database.yaml 读取）",
    )
    run_cmd.add_argument(
        "--output-table", default=DEFAULT_OUTPUT_TABLE,
        help=f"输出表名 (默认: {DEFAULT_OUTPUT_TABLE})",
    )
    run_cmd.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Qwen3-8B 模型路径（默认从 config/database.yaml 的 LLM_model_path 读取）",
    )
    run_cmd.add_argument(
        "--limit", type=int, default=None,
        help="调试用，限制处理行数",
    )
    run_cmd.add_argument(
        "--sample-size", type=int, default=DEFAULT_VALIDATION_SAMPLE_SIZE,
        help=f"验证抽样数量 (默认: {DEFAULT_VALIDATION_SAMPLE_SIZE})",
    )
    run_cmd.add_argument(
        "--seed", type=int, default=42, help="随机种子 (默认: 42)",
    )
    run_cmd.add_argument(
        "--gpu-memory-utilization", type=float, default=0.80,
        help="GPU 显存利用率 (默认: 0.80)",
    )
    run_cmd.add_argument(
        "--max-model-len", type=int, default=8192,
        help="vLLM 最大序列长度 (默认: 8192)",
    )
    run_cmd.add_argument(
        "--max-num-seqs", type=int, default=48,
        help="vLLM 最大并发序列数 (默认: 48)",
    )
    run_cmd.add_argument(
        "--no-auto-correct", action="store_true",
        help="禁用自动修正词典（仅输出报告）",
    )
    run_cmd.add_argument(
        "--context-classifier-model", default=None,
        help="上下文判别器模型目录；提供后启用二阶段过滤",
    )
    run_cmd.add_argument(
        "--context-threshold", type=float, default=0.80,
        help="上下文判别器保留阈值 (默认: 0.80)",
    )
    run_cmd.add_argument(
        "--context-batch-size", type=int, default=64,
        help="上下文判别器推理 batch size (默认: 64)",
    )

    return parser


def main() -> None:
    """CLI 入口函数。

    根据子命令分发到对应的流程：
        - ``match``:    调用 ``run_match()``
        - ``validate``: 调用 ``run_validate()``
        - ``run``:      调用 ``run_full()``
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "match":
        run_match(
            dict_path=args.dictionary,
            source_table=args.source_table,
            output_table=args.output_table,
            limit=args.limit,
            context_classifier_model=getattr(args, "context_classifier_model", None),
            context_threshold=getattr(args, "context_threshold", 0.80),
            context_batch_size=getattr(args, "context_batch_size", 64),
        )
        return

    if args.command == "validate":
        run_validate(
            dict_path=args.dictionary,
            output_table=args.output_table,
            model_path=args.model,
            sample_size=args.sample_size,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            auto_correct=not args.no_auto_correct,
        )
        return

    if args.command == "run":
        run_full(
            dict_path=args.dictionary,
            source_table=getattr(args, "source_table", None),
            output_table=args.output_table,
            model_path=args.model,
            limit=args.limit,
            sample_size=args.sample_size,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            auto_correct=not args.no_auto_correct,
            context_classifier_model=getattr(args, "context_classifier_model", None),
            context_threshold=getattr(args, "context_threshold", 0.80),
            context_batch_size=getattr(args, "context_batch_size", 64),
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
