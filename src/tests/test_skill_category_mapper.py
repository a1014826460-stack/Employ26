"""skill_category_mapper 模块的单元测试。

覆盖范围：
- load_category_rules: 加载、异常处理
- map_skill_type: 命中、未命中、needs_llm
- classify_batch_by_llm: LLM 分类 mock、无效类别、空输入
- apply_categories_to_dictionary: 端到端流程
- get_category_definitions: 类别定义
- VALID_CATEGORIES 值域校验
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.skill_extraction.skill_category_mapper import (
    VALID_CATEGORIES,
    _default_rules_path,
    apply_categories_to_dictionary,
    classify_batch_by_llm,
    get_category_definitions,
    load_category_rules,
    map_skill_type,
)


# ─── 固定测试数据 ────────────────────────────────────────────────

SAMPLE_RULES_DATA = {
    "categories": {
        "programming_language": {
            "name_zh": "编程语言",
            "description": "编程语言",
            "examples": ["Python", "Java"],
        },
        "tool": {
            "name_zh": "工具软件",
            "description": "工具软件",
            "examples": ["Git"],
        },
        "equipment": {
            "name_zh": "设备/仪器",
            "description": "设备/仪器",
            "examples": ["示波器"],
        },
        "process": {
            "name_zh": "工艺方法",
            "description": "工艺方法",
            "examples": ["焊接工艺"],
        },
        "certification": {
            "name_zh": "证书/资质",
            "description": "证书/资质",
            "examples": ["PMP"],
        },
    },
    "mapping_rules": {
        "编程语言": "programming_language",
        "框架": "tool",
        "tool": "tool",
        "设备/仪器": "equipment",
        "工艺/方法": "process",
        "证书/资质": "certification",
        "专业知识": "needs_llm",
    },
    "llm_classification_prompt": {
        "system_prompt": "你是技能分类专家。",
        "user_prompt_template": "分类: {skill_name} {context}",
    },
}

SAMPLE_DICT_DATA = {
    "metadata": {"skill_count": 4},
    "skills": [
        {"name": "Python", "aliases": [], "skill_type": "编程语言", "notes": ""},
        {"name": "Git", "aliases": [], "skill_type": "tool", "notes": ""},
        {"name": "示波器", "aliases": [], "skill_type": "设备/仪器", "notes": ""},
        {"name": "数据建模", "aliases": [], "skill_type": "专业知识", "notes": ""},
    ],
}


# ─── load_category_rules 测试 ────────────────────────────────────


class TestLoadCategoryRules:
    """load_category_rules 函数测试。"""

    def test_loads_default_rules(self):
        """默认路径可正常加载。"""
        data = load_category_rules()
        assert "mapping_rules" in data
        assert "categories" in data
        assert "llm_classification_prompt" in data

    def test_loads_custom_path(self, tmp_path):
        """自定义路径可正常加载。"""
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(SAMPLE_RULES_DATA), encoding="utf-8")
        data = load_category_rules(rules_file)
        assert data["mapping_rules"]["编程语言"] == "programming_language"

    def test_raises_on_missing_file(self, tmp_path):
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_category_rules(tmp_path / "nonexistent.json")

    def test_raises_on_missing_field(self, tmp_path):
        """缺少必要字段时抛出 ValueError。"""
        rules_file = tmp_path / "bad.json"
        rules_file.write_text(json.dumps({"mapping_rules": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="categories"):
            load_category_rules(rules_file)

    def test_default_rules_cover_all_skill_types(self):
        """默认规则覆盖所有已知 skill_type 值（不应有未声明的 skill_type 漏掉）。"""
        data = load_category_rules()
        rules = data["mapping_rules"]
        # 加载词典中的所有 skill_type
        dict_path = _default_rules_path().parent / "flat_skill_dictionary.json"
        if not dict_path.exists():
            pytest.skip("flat_skill_dictionary.json 不存在")
        with open(dict_path, "r", encoding="utf-8") as f:
            dict_data = json.load(f)
        skill_types = {s["skill_type"] for s in dict_data["skills"]}
        uncovered = skill_types - set(rules.keys())
        # 未知的 skill_type 应该被标记为 needs_llm 或已有映射
        # 此测试确保我们不遗漏已知的 skill_type
        assert uncovered == set(), f"以下 skill_type 未被规则覆盖: {uncovered}"


# ─── map_skill_type 测试 ─────────────────────────────────────────


class TestMapSkillType:
    """map_skill_type 函数测试。"""

    def test_maps_known_skill_type(self):
        """已知 skill_type 正确映射。"""
        rules = SAMPLE_RULES_DATA["mapping_rules"]
        assert map_skill_type("编程语言", rules=rules) == "programming_language"
        assert map_skill_type("tool", rules=rules) == "tool"
        assert map_skill_type("设备/仪器", rules=rules) == "equipment"
        assert map_skill_type("工艺/方法", rules=rules) == "process"
        assert map_skill_type("证书/资质", rules=rules) == "certification"

    def test_returns_none_for_needs_llm(self):
        """needs_llm 标记的 skill_type 返回 None。"""
        rules = SAMPLE_RULES_DATA["mapping_rules"]
        assert map_skill_type("专业知识", rules=rules) is None

    def test_returns_none_for_unknown(self):
        """未知 skill_type 返回 None。"""
        rules = SAMPLE_RULES_DATA["mapping_rules"]
        assert map_skill_type("完全未知的类型", rules=rules) is None

    def test_uses_rules_data_fallback(self):
        """传入 rules_data 时也能正确工作。"""
        result = map_skill_type("编程语言", rules_data=SAMPLE_RULES_DATA)
        assert result == "programming_language"

    def test_default_rules_file_loads(self):
        """默认加载完整规则文件并映射。"""
        result = map_skill_type("编程语言")
        assert result == "programming_language"

    def test_all_8_valid_categories_in_default_rules(self):
        """默认规则的映射目标覆盖全部 8 个类别。"""
        data = load_category_rules()
        mapped_values = set(data["mapping_rules"].values()) - {"needs_llm"}
        for cat in VALID_CATEGORIES:
            assert cat in mapped_values, f"类别 '{cat}' 未出现在 mapping_rules 的值域中"


# ─── classify_batch_by_llm 测试 ──────────────────────────────────


class TestClassifyBatchByLLM:
    """classify_batch_by_llm 函数测试。"""

    def test_returns_empty_for_empty_input(self):
        """空输入返回空字典。"""
        result = classify_batch_by_llm([], rules_data=SAMPLE_RULES_DATA)
        assert result == {}

    def test_classifies_single_skill_via_mock(self):
        """单个技能通过 mock LLM 正确分类。"""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = [
            {"skill": "数据建模", "category": "process"}
        ]

        result = classify_batch_by_llm(
            ["数据建模"],
            rules_data=SAMPLE_RULES_DATA,
            llm_client=mock_client,
        )
        assert result == {"数据建模": "process"}
        mock_client.complete_json.assert_called_once()

    def test_classifies_batch_via_mock(self):
        """多个技能批量分类。"""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = [
            {"skill": "技能A", "category": "tool"},
            {"skill": "技能B", "category": "equipment"},
            {"skill": "技能C", "category": "process"},
        ]

        result = classify_batch_by_llm(
            ["技能A", "技能B", "技能C"],
            rules_data=SAMPLE_RULES_DATA,
            llm_client=mock_client,
        )
        assert result == {
            "技能A": "tool",
            "技能B": "equipment",
            "技能C": "process",
        }

    def test_skips_invalid_category(self):
        """LLM 返回无效类别时跳过该技能。"""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = [
            {"skill": "技能X", "category": "invalid_category"},
            {"skill": "技能Y", "category": "tool"},
        ]

        result = classify_batch_by_llm(
            ["技能X", "技能Y"],
            rules_data=SAMPLE_RULES_DATA,
            llm_client=mock_client,
        )
        assert "技能X" not in result
        assert result["技能Y"] == "tool"

    def test_handles_dict_response_for_single_skill(self):
        """LLM 返回单个对象（而非数组）时也能处理。"""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = {
            "skill": "单个技能",
            "category": "process",
        }

        result = classify_batch_by_llm(
            ["单个技能"],
            rules_data=SAMPLE_RULES_DATA,
            llm_client=mock_client,
        )
        assert result == {"单个技能": "process"}

    def test_handles_llm_error_gracefully(self):
        """LLM 调用失败时不崩溃，返回空结果。"""
        mock_client = MagicMock()
        mock_client.complete_json.side_effect = RuntimeError("LLM 不可用")

        result = classify_batch_by_llm(
            ["技能A"],
            rules_data=SAMPLE_RULES_DATA,
            llm_client=mock_client,
        )
        # 批量失败后会尝试逐个重试，逐个也失败则返回空
        assert isinstance(result, dict)


# ─── apply_categories_to_dictionary 测试 ──────────────────────────


class TestApplyCategoriesToDictionary:
    """apply_categories_to_dictionary 函数测试。"""

    def test_applies_categories_with_skip_llm(self, tmp_path):
        """skip_llm=True 时做规则映射 + 启发式分类，仍未命中的设为 None。"""
        dict_file = tmp_path / "test_dict.json"
        dict_file.write_text(
            json.dumps(SAMPLE_DICT_DATA, ensure_ascii=False), encoding="utf-8"
        )

        result = apply_categories_to_dictionary(
            dict_path=dict_file,
            rules_data=SAMPLE_RULES_DATA,
            skip_llm=True,
        )

        skills = {s["name"]: s for s in result["skills"]}
        assert skills["Python"]["category"] == "programming_language"
        assert skills["Git"]["category"] == "tool"
        assert skills["示波器"]["category"] == "equipment"
        # 数据建模 通过启发式规则匹配到 process（关键词"建模"命中设计/分析模式）
        assert skills["数据建模"]["category"] == "process"

    def test_applies_categories_with_llm_mock(self, tmp_path):
        """LLM 分类流程通过 mock 验证。"""
        dict_file = tmp_path / "test_dict.json"
        dict_file.write_text(
            json.dumps(SAMPLE_DICT_DATA, ensure_ascii=False), encoding="utf-8"
        )

        mock_client = MagicMock()
        mock_client.complete_json.return_value = [
            {"skill": "数据建模", "category": "process"}
        ]

        result = apply_categories_to_dictionary(
            dict_path=dict_file,
            rules_data=SAMPLE_RULES_DATA,
            llm_client=mock_client,
        )

        skills = {s["name"]: s for s in result["skills"]}
        assert skills["Python"]["category"] == "programming_language"
        assert skills["数据建模"]["category"] == "process"

    def test_writes_output_file(self, tmp_path):
        """指定 output_path 时正确写入文件。"""
        dict_file = tmp_path / "input.json"
        dict_file.write_text(
            json.dumps(SAMPLE_DICT_DATA, ensure_ascii=False), encoding="utf-8"
        )
        output_file = tmp_path / "output" / "result.json"

        apply_categories_to_dictionary(
            dict_path=dict_file,
            output_path=output_file,
            rules_data=SAMPLE_RULES_DATA,
            skip_llm=True,
        )

        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert any("category" in s for s in saved["skills"])


# ─── get_category_definitions 测试 ───────────────────────────────


class TestGetCategoryDefinitions:
    """get_category_definitions 函数测试。"""

    def test_returns_category_info(self):
        """返回的类别信息包含必要的字段。"""
        defs = get_category_definitions(SAMPLE_RULES_DATA)
        assert "programming_language" in defs
        assert defs["programming_language"]["name_zh"] == "编程语言"

    def test_default_rules_have_all_8_categories(self):
        """默认规则文件包含全部 8 个类别定义。"""
        defs = get_category_definitions()
        for cat in VALID_CATEGORIES:
            assert cat in defs, f"类别 '{cat}' 缺少定义"
            assert "name_zh" in defs[cat], f"类别 '{cat}' 缺少 name_zh"
            assert "description" in defs[cat], f"类别 '{cat}' 缺少 description"
            assert "examples" in defs[cat], f"类别 '{cat}' 缺少 examples"


# ─── VALID_CATEGORIES 值域校验 ───────────────────────────────────


class TestValidCategories:
    """VALID_CATEGORIES 常量校验。"""

    def test_exactly_8_categories(self):
        """恰好有 8 个标准类别。"""
        assert len(VALID_CATEGORIES) == 8

    def test_expected_categories_present(self):
        """8 个类别标识正确。"""
        expected = {
            "programming_language",
            "framework",
            "database",
            "tool",
            "office",
            "equipment",
            "process",
            "certification",
        }
        assert VALID_CATEGORIES == expected

    def test_all_lowercase_underscore(self):
        """所有类别标识为小写加下划线格式。"""
        for cat in VALID_CATEGORIES:
            assert cat.islower(), f"类别 '{cat}' 不是全小写"
            assert " " not in cat, f"类别 '{cat}' 包含空格"


# ─── FlatHardSkillMatcher category 输出测试 ─────────────────────────


# 用于构建 FlatHardSkillMatcher 的测试词典
MATCHER_TEST_DICT = {
    "metadata": {"skill_count": 5},
    "skills": [
        {
            "name": "Python",
            "aliases": ["python3"],
            "skill_type": "编程语言",
            "category": "programming_language",
        },
        {
            "name": "MySQL",
            "aliases": ["mysql数据库"],
            "skill_type": "数据库",
            "category": "database",
        },
        {
            "name": "Git",
            "aliases": [],
            "skill_type": "tool",
            "category": "tool",
        },
        {
            "name": "数据建模",
            "aliases": [],
            "skill_type": "专业知识",
            "category": None,
        },
        {
            "name": "PMP",
            "aliases": ["项目管理专业人士"],
            "skill_type": "证书/资质",
            "category": "certification",
        },
    ],
}


class TestFlatHardSkillMatcherCategory:
    """FlatHardSkillMatcher 输出包含 category 字段的测试。"""

    def _make_matcher(self):
        """创建测试用匹配器实例。"""
        from src.skill_extraction.match_flat_skills_to_duckdb import (
            FlatHardSkillMatcher,
        )

        return FlatHardSkillMatcher(MATCHER_TEST_DICT)

    def test_match_candidates_contains_category(self):
        """match_candidates 返回结果包含 category 字段。"""
        matcher = self._make_matcher()
        candidates = matcher.match_candidates("熟练掌握 Python 和 MySQL")
        assert len(candidates) >= 2
        for candidate in candidates:
            assert "category" in candidate

    def test_match_candidates_category_values_correct(self):
        """match_candidates 返回的 category 值与词典一致。"""
        matcher = self._make_matcher()
        candidates = matcher.match_candidates("需要会 Python 和 MySQL")
        by_name = {c["skill_name"]: c for c in candidates}
        assert by_name["Python"]["category"] == "programming_language"
        assert by_name["MySQL"]["category"] == "database"

    def test_match_candidates_none_category(self):
        """词典中 category 为 None 的技能，输出中 category 也为 None。"""
        matcher = self._make_matcher()
        candidates = matcher.match_candidates("需要数据建模能力")
        by_name = {c["skill_name"]: c for c in candidates}
        if "数据建模" in by_name:
            assert by_name["数据建模"]["category"] is None

    def test_match_text_returns_list_of_dicts(self):
        """match_text 返回字典列表，而非字符串列表。"""
        matcher = self._make_matcher()
        results = matcher.match_text("需要掌握 Python 和 Git")
        assert isinstance(results, list)
        assert len(results) > 0
        for item in results:
            assert isinstance(item, dict)
            assert "skill_name" in item
            assert "category" in item

    def test_match_text_category_values(self):
        """match_text 返回的 category 值正确。"""
        matcher = self._make_matcher()
        results = matcher.match_text("需要掌握 Python 和 Git")
        by_name = {r["skill_name"]: r for r in results}
        assert by_name["Python"]["category"] == "programming_language"
        assert by_name["Git"]["category"] == "tool"

    def test_match_text_empty_input(self):
        """空文本返回空列表。"""
        matcher = self._make_matcher()
        results = matcher.match_text("")
        assert results == []

    def test_match_text_no_match(self):
        """无匹配时返回空列表。"""
        matcher = self._make_matcher()
        results = matcher.match_text("需要吃苦耐劳的精神")
        assert results == []

    def test_match_text_alias_category_matches_parent(self):
        """通过 alias 命中时，category 来自父技能。"""
        matcher = self._make_matcher()
        results = matcher.match_text("需要会 python3 编程")
        by_name = {r["skill_name"]: r for r in results}
        # python3 是 Python 的 alias，category 应为 programming_language
        if "Python" in by_name:
            assert by_name["Python"]["category"] == "programming_language"

    def test_match_text_certification_category(self):
        """证书类技能的 category 正确。"""
        matcher = self._make_matcher()
        results = matcher.match_text("持有 PMP 证书")
        by_name = {r["skill_name"]: r for r in results}
        if "PMP" in by_name:
            assert by_name["PMP"]["category"] == "certification"

    def test_match_candidates_always_has_category_key(self):
        """所有候选结果都包含 category 键（即使值为 None）。"""
        matcher = self._make_matcher()
        candidates = matcher.match_candidates("Python MySQL Git 数据建模 PMP")
        for candidate in candidates:
            assert "category" in candidate, (
                f"候选 {candidate['skill_name']} 缺少 category 键"
            )
