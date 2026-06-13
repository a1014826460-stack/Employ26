"""软技能关键词匹配模块。

职责：
1. 加载 `dicts/soft_skill_dictionary.json`，构建名称 → 技能条目的映射（含别名）；
2. 对输入文本做关键词匹配，返回匹配到的软技能列表；
3. 与 `dicts/flat_skill_dictionary.json` 硬技能词典做冲突检测，排除同名硬技能；
4. 复用 `dicts/blacklist_soft_skills.txt` 过滤非技能词。

用法：
    from src.skill_extraction.soft_skill_matcher import SoftSkillMatcher
    matcher = SoftSkillMatcher()
    results = matcher.match_text("具备良好的沟通能力和团队协作精神")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from config.paths import get_project_paths
from ._dict_paths import get_current_soft_skill_dict_path

logger = logging.getLogger(__name__)


@dataclass
class SoftSkillMatch:
    """单条软技能匹配结果。"""

    name: str
    dimension: str
    confidence: float
    source: str = "dict_match"

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "confidence": self.confidence,
            "source": self.source,
        }


class SoftSkillMatcher:
    """基于词典的软技能关键词匹配器。

    通过加载软技能词典，在输入文本中查找技能名称和别名的出现，
    并排除与硬技能词典冲突的条目以及黑名单中的非技能词。

    Attributes:
        _skill_map: 从匹配关键词（技能名或别名）到 (标准技能名, 维度) 的映射。
        _hard_skill_names: 硬技能词典中所有名称和别名的集合，用于冲突检测。
        _blacklist: 黑名单词汇集合，匹配到这些词时跳过。
    """

    def __init__(
        self,
        soft_skill_dict_path: Optional[Path] = None,
        hard_skill_dict_path: Optional[Path] = None,
        blacklist_path: Optional[Path] = None,
    ) -> None:
        """初始化软技能匹配器。

        Args:
            soft_skill_dict_path: 软技能词典 JSON 路径，默认读取 dicts/soft_skill_dictionary.json。
            hard_skill_dict_path: 硬技能词典 JSON 路径，默认读取 dicts/flat_skill_dictionary.json。
            blacklist_path: 黑名单文件路径，默认读取 dicts/blacklist_soft_skills.txt。
        """
        paths = get_project_paths()
        project_root = paths.project_root

        if soft_skill_dict_path is None:
            self._soft_skill_dict_path = get_current_soft_skill_dict_path()
        else:
            self._soft_skill_dict_path = soft_skill_dict_path
        self._hard_skill_dict_path = (
            hard_skill_dict_path or project_root / "dicts" / "flat_skill_dictionary.json"
        )
        self._blacklist_path = (
            blacklist_path or project_root / "dicts" / "blacklist_soft_skills.txt"
        )

        self._skill_map: Dict[str, tuple[str, str]] = {}
        self._hard_skill_names: Set[str] = set()
        self._blacklist: Set[str] = set()

        self._load_soft_skill_dictionary()
        self._load_hard_skill_names()
        self._load_blacklist()

    def _load_soft_skill_dictionary(self) -> None:
        """加载软技能词典，构建关键词 → (标准名, 维度) 映射。"""
        if not self._soft_skill_dict_path.exists():
            logger.warning("软技能词典不存在: %s", self._soft_skill_dict_path)
            return

        with open(self._soft_skill_dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dimensions = data.get("dimensions", {})
        for dim_key, dim_info in dimensions.items():
            skills = dim_info.get("skills", [])
            for skill in skills:
                canonical_name = skill["name"]
                dimension = skill.get("dimension", dim_key)
                # 标准名称映射
                self._skill_map[canonical_name] = (canonical_name, dimension)
                # 别名映射
                aliases = skill.get("aliases", [])
                for alias in aliases:
                    if alias not in self._skill_map:
                        self._skill_map[alias] = (canonical_name, dimension)

        logger.info(
            "已加载软技能词典: %d 个匹配关键词（含别名）",
            len(self._skill_map),
        )

    def _load_hard_skill_names(self) -> None:
        """加载硬技能词典名称和别名，用于冲突检测。"""
        if not self._hard_skill_dict_path.exists():
            logger.warning("硬技能词典不存在: %s，跳过冲突检测", self._hard_skill_dict_path)
            return

        with open(self._hard_skill_dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for skill in data.get("skills", []):
            self._hard_skill_names.add(skill["name"])
            for alias in skill.get("aliases", []):
                self._hard_skill_names.add(alias)

        logger.info(
            "已加载硬技能词典: %d 个名称/别名用于冲突检测",
            len(self._hard_skill_names),
        )

    def _load_blacklist(self) -> None:
        """加载黑名单词汇。若文件不存在则跳过。"""
        if not self._blacklist_path.exists():
            logger.info("黑名单文件不存在: %s，跳过黑名单过滤", self._blacklist_path)
            return

        with open(self._blacklist_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self._blacklist.add(word)

        logger.info("已加载黑名单: %d 个词汇", len(self._blacklist))

    def _is_hard_skill_conflict(self, skill_name: str) -> bool:
        """检查软技能名称是否与硬技能词典冲突。

        Args:
            skill_name: 待检查的软技能标准名称。

        Returns:
            True 表示存在冲突，该软技能应被排除。
        """
        return skill_name in self._hard_skill_names

    def match_text(self, text: str) -> List[dict]:
        """对输入文本做关键词匹配，返回匹配到的软技能列表。

        匹配策略：对每个已知的关键词（技能名或别名），检查是否出现在输入文本中。
        若出现且不在黑名单中、标准名称不与硬技能冲突，则作为匹配结果返回。

        Args:
            text: 待匹配的输入文本。

        Returns:
            匹配结果列表，每个元素为字典:
            {"name": str, "dimension": str, "confidence": float, "source": str}
        """
        if not text or not text.strip():
            return []

        text_stripped = text.strip()
        seen_canonical: Set[str] = set()
        results: List[dict] = []

        # 按关键词长度降序排序，优先匹配长关键词（避免短词覆盖长词）
        sorted_keywords = sorted(self._skill_map.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            if keyword not in text_stripped:
                continue

            canonical_name, dimension = self._skill_map[keyword]

            # 已经通过其他关键词匹配到了同一个标准技能
            if canonical_name in seen_canonical:
                continue

            # 黑名单过滤
            if keyword in self._blacklist or canonical_name in self._blacklist:
                continue

            # 硬技能冲突检测
            if self._is_hard_skill_conflict(canonical_name):
                continue

            # 根据匹配类型确定置信度
            if keyword == canonical_name:
                confidence = 0.9
            else:
                # 别名匹配
                confidence = 0.85

            seen_canonical.add(canonical_name)
            results.append(
                SoftSkillMatch(
                    name=canonical_name,
                    dimension=dimension,
                    confidence=confidence,
                    source="dict_match",
                ).to_dict()
            )

        return results
