"""soft_skill_seed_extractor 与 soft_skill_dictionary_builder 模块的单元测试。

覆盖范围：
- _map_to_dimension: 关键词匹配、空值处理、未知词
- _group_by_dimension: 频次过滤、分组逻辑
- extract_soft_skill_seeds: 数据库失败兜底、空结果兜底
- _build_fallback_seeds: 兜底种子词结构
- DIMENSION_KEYWORDS: 大五维度定义完整性
- build_soft_skill_dictionary: 词典构建、LLM 扩展、静态别名
- load_dictionary / save_dictionary: 文件读写
- 词典格式验证：schema_version、维度覆盖、技能条目结构
"""

from __future__ import annotations

import pytest
from collections import Counter
from unittest.mock import patch, MagicMock

import json
import tempfile
from pathlib import Path

from src.skill_extraction.soft_skill_seed_extractor import (
    DIMENSION_KEYWORDS,
    MIN_FREQUENCY,
    _build_fallback_seeds,
    _group_by_dimension,
    _map_to_dimension,
    extract_soft_skill_seeds,
)
from src.skill_extraction.soft_skill_dictionary_builder import (
    DIMENSION_DISPLAY_NAMES,
    _STATIC_ALIASES,
    _build_skill_entry,
    _get_static_aliases,
    _expand_with_llm,
    build_soft_skill_dictionary,
    load_dictionary,
    save_dictionary,
)


# ─── DIMENSION_KEYWORDS 完整性测试 ────────────────────────────────


class TestDimensionKeywords:
    """大五维度关键词定义测试。"""

    def test_exactly_five_dimensions(self):
        """恰好定义了 5 个大五维度。"""
        assert len(DIMENSION_KEYWORDS) == 5

    def test_expected_dimension_names(self):
        """维度名称正确。"""
        expected = {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }
        assert set(DIMENSION_KEYWORDS.keys()) == expected

    def test_each_dimension_has_keywords(self):
        """每个维度至少定义了 3 个关键词。"""
        for dim, keywords in DIMENSION_KEYWORDS.items():
            assert len(keywords) >= 3, f"维度 '{dim}' 关键词不足 3 个"

    def test_keywords_are_non_empty_strings(self):
        """所有关键词为非空字符串。"""
        for dim, keywords in DIMENSION_KEYWORDS.items():
            for kw in keywords:
                assert isinstance(kw, str) and kw.strip(), (
                    f"维度 '{dim}' 包含无效关键词: {kw!r}"
                )


# ─── _map_to_dimension 测试 ──────────────────────────────────────


class TestMapToDimension:
    """_map_to_dimension 函数测试。"""

    def test_openness_keywords(self):
        """开放性关键词正确映射。"""
        assert _map_to_dimension("创新思维") == "openness"
        assert _map_to_dimension("好奇心强") == "openness"
        assert _map_to_dimension("有想象力") == "openness"
        assert _map_to_dimension("学习能力") == "openness"
        assert _map_to_dimension("灵活应变") == "openness"

    def test_conscientiousness_keywords(self):
        """尽责性关键词正确映射。"""
        assert _map_to_dimension("工作细心") == "conscientiousness"
        assert _map_to_dimension("责任心强") == "conscientiousness"
        assert _map_to_dimension("自律") == "conscientiousness"
        assert _map_to_dimension("做事严谨") == "conscientiousness"
        assert _map_to_dimension("高效执行") == "conscientiousness"

    def test_extraversion_keywords(self):
        """外向性关键词正确映射。"""
        assert _map_to_dimension("沟通能力强") == "extraversion"
        assert _map_to_dimension("团队协作") == "extraversion"
        assert _map_to_dimension("有领导力") == "extraversion"
        assert _map_to_dimension("表达能力好") == "extraversion"

    def test_agreeableness_keywords(self):
        """宜人性关键词正确映射。"""
        assert _map_to_dimension("善于合作") == "agreeableness"
        assert _map_to_dimension("有同理心") == "agreeableness"
        assert _map_to_dimension("友善待人") == "agreeableness"
        assert _map_to_dimension("服务意识强") == "agreeableness"

    def test_neuroticism_keywords(self):
        """情绪稳定性关键词正确映射。"""
        assert _map_to_dimension("抗压能力强") == "neuroticism"
        assert _map_to_dimension("情绪管理") == "neuroticism"
        assert _map_to_dimension("冷静处理") == "neuroticism"
        assert _map_to_dimension("心理承受力好") == "neuroticism"

    def test_empty_string_returns_none(self):
        """空字符串返回 None。"""
        assert _map_to_dimension("") is None

    def test_none_returns_none(self):
        """None 输入返回 None。"""
        assert _map_to_dimension(None) is None

    def test_whitespace_only_returns_none(self):
        """纯空白字符串返回 None。"""
        assert _map_to_dimension("   ") is None

    def test_unknown_skill_returns_none(self):
        """无法匹配任何维度的技能返回 None。"""
        assert _map_to_dimension("Python编程") is None
        assert _map_to_dimension("三年工作经验") is None
        assert _map_to_dimension("985院校") is None

    def test_leading_trailing_whitespace_stripped(self):
        """前后空白不影响匹配。"""
        assert _map_to_dimension("  创新  ") == "openness"
        assert _map_to_dimension("\t沟通能力\n") == "extraversion"


# ─── _group_by_dimension 测试 ────────────────────────────────────


class TestGroupByDimension:
    """_group_by_dimension 函数测试。"""

    def test_filters_low_frequency(self):
        """频次低于阈值的词被过滤。"""
        counts = Counter({
            "创新": 5,
            "沟通能力": 3,
            "自律": 2,   # 低于阈值
            "Python": 10,  # 无法映射到维度
        })
        result = _group_by_dimension(counts, min_frequency=3)
        assert "创新" in result["openness"]
        assert "沟通能力" in result["extraversion"]
        assert "自律" not in result["conscientiousness"]

    def test_unmapped_skills_excluded(self):
        """无法映射到维度的高频词不出现在结果中。"""
        counts = Counter({
            "Python": 10,
            "Java": 8,
            "MySQL": 6,
        })
        result = _group_by_dimension(counts, min_frequency=3)
        for words in result.values():
            assert len(words) == 0

    def test_sorted_by_frequency_descending(self):
        """每个维度内种子词按频次降序排列。"""
        counts = Counter({
            "创新": 10,
            "想象力": 5,
            "学习能力": 8,
        })
        result = _group_by_dimension(counts, min_frequency=3)
        words = result["openness"]
        assert words == ["创新", "学习能力", "想象力"]

    def test_all_dimensions_present(self):
        """结果中包含所有 5 个维度的键。"""
        counts = Counter({"创新": 5})
        result = _group_by_dimension(counts, min_frequency=3)
        assert set(result.keys()) == set(DIMENSION_KEYWORDS.keys())

    def test_empty_counter(self):
        """空 Counter 返回所有维度为空列表。"""
        result = _group_by_dimension(Counter(), min_frequency=3)
        for words in result.values():
            assert words == []

    def test_min_frequency_boundary(self):
        """频次恰好等于阈值的词被纳入。"""
        counts = Counter({"创新": 3})
        result = _group_by_dimension(counts, min_frequency=3)
        assert "创新" in result["openness"]

    def test_min_frequency_one_below(self):
        """频次恰好低于阈值的词被排除。"""
        counts = Counter({"创新": 2})
        result = _group_by_dimension(counts, min_frequency=3)
        assert "创新" not in result["openness"]


# ─── _build_fallback_seeds 测试 ──────────────────────────────────


class TestBuildFallbackSeeds:
    """_build_fallback_seeds 函数测试。"""

    def test_returns_all_dimensions(self):
        """兜底结果包含所有 5 个维度。"""
        result = _build_fallback_seeds()
        assert set(result.keys()) == set(DIMENSION_KEYWORDS.keys())

    def test_each_dimension_has_seeds(self):
        """每个维度至少有 3 个兜底种子词。"""
        result = _build_fallback_seeds()
        for dim, words in result.items():
            assert len(words) >= 3, f"维度 '{dim}' 兜底种子词不足 3 个"

    def test_seeds_are_from_dimension_keywords(self):
        """兜底种子词来自维度定义关键词。"""
        result = _build_fallback_seeds()
        for dim, words in result.items():
            for w in words:
                assert w in DIMENSION_KEYWORDS[dim], (
                    f"种子词 '{w}' 不在维度 '{dim}' 的关键词列表中"
                )

    def test_seeds_are_subset(self):
        """兜底种子词是维度关键词的子集（取前 5 个）。"""
        result = _build_fallback_seeds()
        for dim, words in result.items():
            assert words == DIMENSION_KEYWORDS[dim][:5]


# ─── extract_soft_skill_seeds 测试 ───────────────────────────────


class TestExtractSoftSkillSeeds:
    """extract_soft_skill_seeds 函数测试。"""

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_db_failure_uses_fallback(self, mock_fetch):
        """数据库查询失败时使用兜底种子词。"""
        mock_fetch.side_effect = RuntimeError("数据库连接失败")
        result = extract_soft_skill_seeds(fallback_on_db_error=True)
        assert set(result.keys()) == set(DIMENSION_KEYWORDS.keys())
        for words in result.values():
            assert len(words) > 0

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_db_failure_raises_when_fallback_disabled(self, mock_fetch):
        """禁用兜底时，数据库错误直接抛出。"""
        mock_fetch.side_effect = RuntimeError("数据库连接失败")
        with pytest.raises(RuntimeError, match="数据库连接失败"):
            extract_soft_skill_seeds(fallback_on_db_error=False)

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_empty_db_result_uses_fallback(self, mock_fetch):
        """数据库返回空列表时使用兜底种子词。"""
        mock_fetch.return_value = []
        result = extract_soft_skill_seeds()
        for words in result.values():
            assert len(words) > 0

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_normal_extraction(self, mock_fetch):
        """正常提取流程：从数据库标注词中提取并分组。"""
        mock_fetch.return_value = [
            "创新", "创新", "创新", "创新",  # openness, count=4
            "沟通能力", "沟通能力", "沟通能力",  # extraversion, count=3
            "责任心", "责任心",  # conscientiousness, count=2 (below threshold)
            "Python", "Python", "Python", "Python",  # unmapped, count=4
            "抗压能力", "抗压能力", "抗压能力", "抗压能力", "抗压能力",  # neuroticism, count=5
        ]
        result = extract_soft_skill_seeds(min_frequency=3)

        assert "创新" in result["openness"]
        assert "沟通能力" in result["extraversion"]
        assert "抗压能力" in result["neuroticism"]
        # 低于阈值
        assert "责任心" not in result.get("conscientiousness", [])
        # 无法映射
        for words in result.values():
            assert "Python" not in words

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_comma_separated_skills_parsed(self, mock_fetch):
        """逗号分隔的多值字段被正确拆分。"""
        mock_fetch.return_value = [
            "创新,想象力",
            "创新,想象力",
            "创新,想象力",
        ]
        result = extract_soft_skill_seeds(min_frequency=3)
        assert "创新" in result["openness"]
        assert "想象力" in result["openness"]

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_chinese_comma_separator(self, mock_fetch):
        """中文逗号分隔符也能正确处理。"""
        mock_fetch.return_value = [
            "创新，想象力",
            "创新，想象力",
            "创新，想象力",
        ]
        result = extract_soft_skill_seeds(min_frequency=3)
        assert "创新" in result["openness"]
        assert "想象力" in result["openness"]

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_custom_min_frequency(self, mock_fetch):
        """自定义最低频次阈值生效。"""
        mock_fetch.return_value = [
            "创新", "创新",  # count=2
        ]
        # min_frequency=2 时应纳入
        result = extract_soft_skill_seeds(min_frequency=2)
        assert "创新" in result["openness"]

        # min_frequency=3 时应排除（回到默认 mock）
        result = extract_soft_skill_seeds(min_frequency=3)
        assert "创新" not in result["openness"]

    @patch(
        "src.skill_extraction.soft_skill_seed_extractor._fetch_soft_skills_from_db"
    )
    def test_result_structure(self, mock_fetch):
        """返回结构为 dict[str, list[str]]。"""
        mock_fetch.return_value = ["创新", "创新", "创新"]
        result = extract_soft_skill_seeds()
        assert isinstance(result, dict)
        assert len(result) == 5
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, list)
            for item in value:
                assert isinstance(item, str)


# ─── MIN_FREQUENCY 常量测试 ──────────────────────────────────────


class TestMinFrequency:
    """MIN_FREQUENCY 常量测试。"""

    def test_default_value_is_3(self):
        """默认最低频次为 3。"""
        assert MIN_FREQUENCY == 3

    def test_is_positive_integer(self):
        """最低频次为正整数。"""
        assert isinstance(MIN_FREQUENCY, int)
        assert MIN_FREQUENCY > 0


# ═══════════════════════════════════════════════════════════════════
# soft_skill_dictionary_builder 测试
# ═══════════════════════════════════════════════════════════════════


# ─── DIMENSION_DISPLAY_NAMES 测试 ─────────────────────────────────


class TestDimensionDisplayNames:
    """维度中文名称映射测试。"""

    def test_covers_all_five_dimensions(self):
        """中文名称映射覆盖全部 5 个大五维度。"""
        expected = set(DIMENSION_KEYWORDS.keys())
        assert set(DIMENSION_DISPLAY_NAMES.keys()) == expected

    def test_values_are_chinese_strings(self):
        """所有中文名称为非空字符串。"""
        for key, name in DIMENSION_DISPLAY_NAMES.items():
            assert isinstance(name, str) and name.strip(), (
                f"维度 '{key}' 中文名称无效: {name!r}"
            )


# ─── _get_static_aliases 测试 ─────────────────────────────────────


class TestGetStaticAliases:
    """_get_static_aliases 函数测试。"""

    def test_known_seed_returns_aliases(self):
        """已知种子词返回非空别名列表。"""
        aliases = _get_static_aliases("创新")
        assert isinstance(aliases, list)
        assert len(aliases) > 0
        assert "创新能力" in aliases

    def test_unknown_seed_returns_empty(self):
        """未知种子词返回空列表。"""
        aliases = _get_static_aliases("完全不存在的词xyz")
        assert aliases == []

    def test_aliases_are_strings(self):
        """所有别名为非空字符串。"""
        for seed, aliases in _STATIC_ALIASES.items():
            for alias in aliases:
                assert isinstance(alias, str) and alias.strip(), (
                    f"种子词 '{seed}' 的别名无效: {alias!r}"
                )


# ─── _build_skill_entry 测试 ──────────────────────────────────────


class TestBuildSkillEntry:
    """_build_skill_entry 函数测试。"""

    def test_basic_structure(self):
        """返回结构包含 name、aliases、dimension。"""
        entry = _build_skill_entry("创新", "openness", ["创新能力"])
        assert entry["name"] == "创新"
        assert entry["aliases"] == ["创新能力"]
        assert entry["dimension"] == "openness"

    def test_empty_aliases(self):
        """未提供 aliases 时默认为空列表。"""
        entry = _build_skill_entry("测试", "openness")
        assert entry["aliases"] == []

    def test_none_aliases_defaults_to_empty(self):
        """显式传 None 时 aliases 为空列表。"""
        entry = _build_skill_entry("测试", "openness", aliases=None)
        assert entry["aliases"] == []


# ─── _expand_with_llm 测试 ────────────────────────────────────────


class TestExpandWithLLM:
    """_expand_with_llm 函数测试。"""

    def test_parses_json_array_response(self):
        """LLM 返回合法 JSON 数组时正确解析。"""
        mock_client = MagicMock()
        mock_client.complete_text.return_value = '["变体1", "变体2", "变体3"]'
        result = _expand_with_llm("创新", "openness", "开放性", mock_client)
        assert result == ["变体1", "变体2", "变体3"]
        mock_client.complete_text.assert_called_once()

    def test_parses_json_embedded_in_text(self):
        """LLM 返回包含 JSON 数组的文本时正确提取。"""
        mock_client = MagicMock()
        mock_client.complete_text.return_value = '以下是变体词：\n["变体A", "变体B"]\n请参考。'
        result = _expand_with_llm("创新", "openness", "开放性", mock_client)
        assert result == ["变体A", "变体B"]

    def test_llm_failure_returns_empty(self):
        """LLM 调用异常时返回空列表。"""
        mock_client = MagicMock()
        mock_client.complete_text.side_effect = RuntimeError("LLM 不可用")
        result = _expand_with_llm("创新", "openness", "开放性", mock_client)
        assert result == []

    def test_invalid_json_returns_empty(self):
        """LLM 返回非法 JSON 时返回空列表。"""
        mock_client = MagicMock()
        mock_client.complete_text.return_value = "这不是 JSON"
        result = _expand_with_llm("创新", "openness", "开放性", mock_client)
        assert result == []

    def test_non_array_json_returns_empty(self):
        """LLM 返回非数组 JSON 时返回空列表。"""
        mock_client = MagicMock()
        mock_client.complete_text.return_value = '{"key": "value"}'
        result = _expand_with_llm("创新", "openness", "开放性", mock_client)
        assert result == []

    def test_passes_correct_prompts(self):
        """传给 LLM 的提示词包含维度和种子词信息。"""
        mock_client = MagicMock()
        mock_client.complete_text.return_value = '["变体1"]'
        _expand_with_llm("创新", "openness", "开放性", mock_client)

        call_kwargs = mock_client.complete_text.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs[1].get("system_prompt", "")
        user_prompt = call_kwargs.kwargs.get("user_prompt") or call_kwargs[1].get("user_prompt", "")
        assert "开放性" in user_prompt
        assert "创新" in user_prompt
        assert "openness" in user_prompt


# ─── build_soft_skill_dictionary 测试 ─────────────────────────────


class TestBuildSoftSkillDictionary:
    """build_soft_skill_dictionary 函数测试。"""

    def _make_seeds(self) -> Dict[str, List[str]]:
        """构造测试用种子词。"""
        return {
            "openness": ["创新", "学习能力"],
            "conscientiousness": ["细心", "责任心"],
            "extraversion": ["沟通能力"],
            "agreeableness": ["合作"],
            "neuroticism": ["抗压能力"],
        }

    def test_schema_version(self):
        """输出包含 schema_version=1。"""
        dictionary = build_soft_skill_dictionary(self._make_seeds())
        assert dictionary["schema_version"] == 1

    def test_dimensions_key_exists(self):
        """输出包含 dimensions 键。"""
        dictionary = build_soft_skill_dictionary(self._make_seeds())
        assert "dimensions" in dictionary

    def test_all_five_dimensions_present(self):
        """输出包含全部 5 个大五维度。"""
        dictionary = build_soft_skill_dictionary(self._make_seeds())
        expected = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
        assert set(dictionary["dimensions"].keys()) == expected

    def test_dimension_has_name_and_skills(self):
        """每个维度包含 name 和 skills 字段。"""
        dictionary = build_soft_skill_dictionary(self._make_seeds())
        for key, dim in dictionary["dimensions"].items():
            assert "name" in dim, f"维度 '{key}' 缺少 name"
            assert "skills" in dim, f"维度 '{key}' 缺少 skills"

    def test_skill_entry_structure(self):
        """每个技能条目包含 name、aliases、dimension。"""
        dictionary = build_soft_skill_dictionary(self._make_seeds())
        for key, dim in dictionary["dimensions"].items():
            for skill in dim["skills"]:
                assert "name" in skill, f"维度 '{key}' 技能缺少 name"
                assert "aliases" in skill, f"维度 '{key}' 技能缺少 aliases"
                assert "dimension" in skill, f"维度 '{key}' 技能缺少 dimension"
                assert isinstance(skill["aliases"], list)
                assert skill["dimension"] == key

    def test_seeds_become_skill_names(self):
        """种子词成为技能条目的 name。"""
        seeds = self._make_seeds()
        dictionary = build_soft_skill_dictionary(seeds)
        for dim_key, seed_words in seeds.items():
            skill_names = {s["name"] for s in dictionary["dimensions"][dim_key]["skills"]}
            for seed in seed_words:
                assert seed in skill_names, (
                    f"种子词 '{seed}' 未出现在维度 '{dim_key}' 的技能列表中"
                )

    def test_static_aliases_populated(self):
        """不提供 LLM 时，已知种子词有静态别名。"""
        seeds = {"openness": ["创新"]}
        dictionary = build_soft_skill_dictionary(seeds)
        skill = dictionary["dimensions"]["openness"]["skills"][0]
        assert skill["name"] == "创新"
        assert len(skill["aliases"]) > 0
        assert "创新能力" in skill["aliases"]

    def test_aliases_exclude_name(self):
        """别名列表不包含 name 本身。"""
        dictionary = build_soft_skill_dictionary(self._make_seeds())
        for dim in dictionary["dimensions"].values():
            for skill in dim["skills"]:
                assert skill["name"] not in skill["aliases"]

    def test_with_llm_client(self):
        """提供 LLM 客户端时调用 LLM 扩展。"""
        mock_client = MagicMock()
        mock_client.complete_text.return_value = '["LLM变体1", "LLM变体2"]'
        seeds = {"openness": ["创新"]}
        dictionary = build_soft_skill_dictionary(seeds, llm_client=mock_client)
        skill = dictionary["dimensions"]["openness"]["skills"][0]
        # LLM 变体应出现在别名中
        assert "LLM变体1" in skill["aliases"]
        assert "LLM变体2" in skill["aliases"]
        # 静态别名也应保留
        assert "创新能力" in skill["aliases"]

    def test_deduplication_across_aliases(self):
        """LLM 返回的别名与静态别名不会重复。"""
        mock_client = MagicMock()
        mock_client.complete_text.return_value = '["创新能力", "新变体"]'
        seeds = {"openness": ["创新"]}
        dictionary = build_soft_skill_dictionary(seeds, llm_client=mock_client)
        skill = dictionary["dimensions"]["openness"]["skills"][0]
        # "创新能力" 不应重复出现
        assert skill["aliases"].count("创新能力") == 1
        assert "新变体" in skill["aliases"]

    def test_empty_seeds(self):
        """空种子词字典返回空技能列表。"""
        dictionary = build_soft_skill_dictionary({})
        assert len(dictionary["dimensions"]) == 0

    def test_dimension_display_names_correct(self):
        """维度中文名与 DIMENSION_DISPLAY_NAMES 一致。"""
        dictionary = build_soft_skill_dictionary(self._make_seeds())
        for key, dim in dictionary["dimensions"].items():
            assert dim["name"] == DIMENSION_DISPLAY_NAMES[key]


# ─── save_dictionary / load_dictionary 测试 ────────────────────────


class TestSaveLoadDictionary:
    """词典文件读写测试。"""

    def _make_minimal_dictionary(self) -> Dict[str, Any]:
        """构造最小词典用于测试。"""
        return {
            "schema_version": 1,
            "dimensions": {
                "openness": {
                    "name": "开放性",
                    "skills": [
                        {"name": "创新", "aliases": ["创新能力"], "dimension": "openness"},
                    ],
                },
            },
        }

    def test_save_creates_file(self):
        """save_dictionary 创建文件。"""
        dictionary = self._make_minimal_dictionary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dict.json"
            result = save_dictionary(dictionary, path)
            assert result == path
            assert path.exists()

    def test_save_content_is_valid_json(self):
        """保存的文件是合法 JSON。"""
        dictionary = self._make_minimal_dictionary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dict.json"
            save_dictionary(dictionary, path)
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["schema_version"] == 1

    def test_save_preserves_chinese(self):
        """保存的文件正确保留中文。"""
        dictionary = self._make_minimal_dictionary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dict.json"
            save_dictionary(dictionary, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "开放性" in content
            assert "创新" in content

    def test_load_roundtrip(self):
        """保存后加载，内容一致。"""
        dictionary = self._make_minimal_dictionary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dict.json"
            save_dictionary(dictionary, path)
            loaded = load_dictionary(path)
            assert loaded == dictionary

    def test_load_nonexistent_raises(self):
        """加载不存在的文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_dictionary("/tmp/nonexistent_soft_skill_dict_xyz.json")

    def test_load_invalid_format_raises(self):
        """加载格式不合法的文件抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            with open(path, "w") as f:
                json.dump({"not_schema": True}, f)
            with pytest.raises(ValueError, match="格式不合法"):
                load_dictionary(path)

    def test_save_creates_parent_dirs(self):
        """save_dictionary 自动创建父目录。"""
        dictionary = self._make_minimal_dictionary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "test_dict.json"
            save_dictionary(dictionary, path)
            assert path.exists()


# ─── dicts/soft_skill_dictionary.json 文件验证 ─────────────────────


class TestSoftSkillDictionaryFile:
    """验证 dicts/soft_skill_dictionary.json 文件格式和内容。"""

    @pytest.fixture(autouse=True)
    def _load_file(self):
        """加载词典文件，支持 shim 重定向。"""
        dict_path = Path(__file__).resolve().parent.parent.parent / "dicts" / "soft_skill_dictionary.json"
        if not dict_path.exists():
            pytest.skip("词典文件不存在，跳过文件验证测试")
        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 如果是 shim，跟随 _redirect 加载实际词典
        if "_redirect" in data:
            resolve_path = dict_path.parent / data["_redirect"]
            if not resolve_path.exists():
                pytest.skip(f"重定向目标不存在: {resolve_path}")
            with open(resolve_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        self.dictionary = data

    def test_schema_version(self):
        """词典 schema_version 为 1。"""
        assert self.dictionary["schema_version"] == 1

    def test_has_dimensions_key(self):
        """词典包含 dimensions 键。"""
        assert "dimensions" in self.dictionary

    def test_covers_all_five_dimensions(self):
        """词典覆盖全部 5 个大五维度。"""
        expected = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
        assert set(self.dictionary["dimensions"].keys()) == expected

    def test_each_dimension_has_name(self):
        """每个维度有中文名称。"""
        for key, dim in self.dictionary["dimensions"].items():
            assert "name" in dim
            assert isinstance(dim["name"], str) and dim["name"].strip()

    def test_each_dimension_at_least_10_skills(self):
        """每个维度至少有 10 个技能条目。"""
        for key, dim in self.dictionary["dimensions"].items():
            assert len(dim["skills"]) >= 10, (
                f"维度 '{key}' 只有 {len(dim['skills'])} 个技能，要求至少 10 个"
            )

    def test_skill_entry_fields(self):
        """每个技能条目包含 name、aliases、dimension。"""
        for key, dim in self.dictionary["dimensions"].items():
            for skill in dim["skills"]:
                assert "name" in skill, f"维度 '{key}' 技能缺少 name"
                assert "aliases" in skill, f"维度 '{key}' 技能缺少 aliases"
                assert "dimension" in skill, f"维度 '{key}' 技能缺少 dimension"

    def test_skill_dimension_matches_parent(self):
        """技能条目的 dimension 值与所在维度键一致。"""
        for key, dim in self.dictionary["dimensions"].items():
            for skill in dim["skills"]:
                assert skill["dimension"] == key, (
                    f"技能 '{skill['name']}' 的 dimension='{skill['dimension']}'"
                    f"与所在维度 '{key}' 不一致"
                )

    def test_aliases_are_lists(self):
        """所有 aliases 为列表类型。"""
        for dim in self.dictionary["dimensions"].values():
            for skill in dim["skills"]:
                assert isinstance(skill["aliases"], list)

    def test_skill_names_are_non_empty(self):
        """所有技能名称为非空字符串。"""
        for dim in self.dictionary["dimensions"].values():
            for skill in dim["skills"]:
                assert isinstance(skill["name"], str) and skill["name"].strip()

    def test_no_duplicate_skill_names_in_dimension(self):
        """同一维度内无重复的技能名称。"""
        for key, dim in self.dictionary["dimensions"].items():
            names = [s["name"] for s in dim["skills"]]
            assert len(names) == len(set(names)), (
                f"维度 '{key}' 存在重复技能名称"
            )

    def test_dimension_display_names_match(self):
        """词典中的维度中文名与模块定义一致。"""
        for key, dim in self.dictionary["dimensions"].items():
            expected_name = DIMENSION_DISPLAY_NAMES.get(key)
            if expected_name:
                assert dim["name"] == expected_name, (
                    f"维度 '{key}' 中文名 '{dim['name']}' "
                    f"与模块定义 '{expected_name}' 不一致"
                )
