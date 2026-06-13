"""平面化硬技能词典匹配器（纯匹配逻辑，不依赖 DuckDB）。

从 ``match_flat_skills_to_duckdb.py`` 中抽取的纯匹配核心：
    - ``load_flat_dictionary`` / ``save_flat_dictionary``：词典加载与保存
    - ``FlatHardSkillMatcher``：基于平面化技能词典的硬技能匹配器

本模块不导入 ``duckdb``，可被 PostgreSQL 链路安全引用。

用法::

    >>> from src.skill_extraction.hard_skill_matcher import load_flat_dictionary, FlatHardSkillMatcher
    >>> data = load_flat_dictionary("dicts/flat_skill_dictionary.json")
    >>> matcher = FlatHardSkillMatcher(data)
    >>> matcher.match_text("熟练掌握 Java 和 MySQL，了解 Redis")
    [{'skill_name': 'Java', 'category': 'programming_language'}, ...]
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ── 复用已有模块的工具函数 ──────────────────────────────────────────────
from ._matching_utils import (
    TermEntry,
    _safe_text,
    canonicalize_alias,
    is_ascii_like_term,
    is_generic_skill_name,
    normalize_match_text,
    safe_lower_text,
)
from .iteration_rules import (
    get_canonical_output_overrides,
    get_contextual_term_rules,
    get_exact_generic_skill_blocklist,
    get_short_chinese_allowlist,
)

logger = logging.getLogger(__name__)

# ============================================================================
#  常量
# ============================================================================

# 词项最小长度阈值：过短的词项噪音极大，需跳过。
# 中文词项归一化后至少 3 字符（如"焊接"=2 太短，"焊接工艺"=4 可以）；
# ASCII 词项至少 2 字符（如 "C"=1 太短，"C++"=3 可以）。
MIN_CHINESE_TERM_LEN: int = 3
MIN_ASCII_TERM_LEN: int = 2

# 黑名单：这些词条即使出现在词典中也不应作为匹配结果输出。
# 它们要么是福利待遇、要么过于泛化，无法作为可统计的硬技能。
SKILL_BLACKLIST: set = {
    "五险一金",
    "社保",
    "双休",
    "带薪年假",
    "节日福利",
    "材料",
    "电脑",
    "测试",
    "检测",
    "英语",
    "普通话",
    "包装",
    "分拣",
    "升华",
    "扫码",
    "贴标",
    "质检",
    "电源",
    "光源",
    "镜头",
    "离心",
    "客户服务",
    "仿真软件",
    "数据分析工具",
    "机械传动知识",
    "专业知识",
    "理论基础",
    "知识基础",
}

# 宽泛 alias 黑名单：这些 alias 过于短或泛化，容易导致系统性误匹配。
# 典型案例："资格证" → "教师资格证"（任何含"资格证"的文本都会误匹配为教师资格证）。
BROAD_ALIAS_BLACKLIST: set = {
    "资格证",
    "品质",
    "函数",
    "电机",
    "模具",
    "印刷",
    "相机",
    "奶粉",
    "辅食",
    "消毒",
    "车辆",
    "排版",
    "打包",
    "催化",
    "氧化",
    "色彩",
    "节奏",
    "光照",
    "京东",
    "淘宝",
    "快团",
}

# 这些模式用于进一步过滤"看起来像技能、实际上过于泛化"的输出项。
# 目标是拦截"测试""仿真软件""资格证书""数据分析工具"这一类容器词。
LOW_VALUE_SKILL_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(能力|素养|基础|知识|理论)$"),
    re.compile(r"(工具|软件|系统|平台)$"),
    re.compile(r"^(数据分析工具|仿真软件|测试仪器|办公软件)$"),
    re.compile(r"^(资格证|资格证书|执业资格证书|上岗证|证书)$"),
]

# 宽泛 alias 的问题通常比宽泛主词更严重，因为它们会把整类文本都吸附到某个具体技能。
# 这里补充正则规则，覆盖"资格证/证书/软件/工具/知识/能力"等高风险 alias。
LOW_VALUE_ALIAS_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^(资格|资格证|资格证书|证书|执业证|上岗证|许可证|执照)$"),
    re.compile(r"^(工具|软件|系统|平台|知识|理论|能力|测试)$"),
    re.compile(r".*(资格证|资格证书|执业证|执业资格证书)$"),
]

EXACT_GENERIC_SKILL_BLACKLIST: set[str] = get_exact_generic_skill_blocklist()
CANONICAL_OUTPUT_OVERRIDES: Dict[str, str] = get_canonical_output_overrides()
SHORT_CHINESE_ALLOWLIST: set[str] = get_short_chinese_allowlist()


def _compile_contextual_term_rules() -> Dict[str, List[Dict]]:
    """编译上下文规则为可快速查询的结构。"""
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


# ============================================================================
#  词典加载与保存
# ============================================================================


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

    data.setdefault("metadata", {})["updated_at"] = datetime.now().strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("词典已保存: %s", out_path)


# ============================================================================
#  FlatHardSkillMatcher — 平面化词典匹配器
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
        self.term_index: List[TermEntry] = self._build_flat_term_index(self.skills)
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
                if term_role == "alias" and self._is_low_value_alias(term_text):
                    skipped_broad_alias += 1
                    continue

                is_ascii = is_ascii_like_term(term_text)
                normalized = normalize_match_text(term_text)

                # 跳过过短的词项（噪音极大）
                if is_ascii:
                    if len(term_text.strip()) < MIN_ASCII_TERM_LEN:
                        skipped_short += 1
                        continue
                else:
                    if (
                        len(normalized) < MIN_CHINESE_TERM_LEN
                        and term_text not in SHORT_CHINESE_ALLOWLIST
                    ):
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
        if len(normalize_match_text(text)) <= 2 and not is_ascii_like_term(text):
            return True
        return any(pattern.fullmatch(text) for pattern in LOW_VALUE_ALIAS_PATTERNS)

    @staticmethod
    def _is_low_value_skill_name(skill_name: str) -> bool:
        """判断技能主名称是否过泛，不适合作为最终输出。

        这里过滤的是"不可直接统计或不可落地执行"的技能容器词，
        而不是所有抽象程度较高的术语。目的不是追求绝对召回，
        而是降低明显误报。
        """
        text = _safe_text(skill_name)
        if not text:
            return True
        if text in EXACT_GENERIC_SKILL_BLACKLIST:
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

        if is_generic_skill_name(entry.skill_name):
            alias_name = canonicalize_alias(entry.term_text)
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
            is_ascii = is_ascii_like_term(name)
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
                if len(norm_i) < len(norm_j) and norm_i in norm_j:
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

        recall_entries = self._match_normalized_entries(
            normalized_text
        ) + self._match_ascii_entries(raw_text)

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
            item for item in candidates if item["skill_name"].casefold() in keep_keys
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
