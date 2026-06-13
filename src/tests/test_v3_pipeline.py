"""V3Pipeline 统一技能抽取管线的单元测试。

覆盖范围：
- 双管线集成：硬技能和软技能同时匹配
- 硬技能优先规则：同名技能归类为硬技能
- 空值处理：空文本、缺失字段、空记录列表
- RecordResult 数据结构
- 辅助函数 _get_match_text、_safe_text、_merge_deduplicate
- V3 评估脚本：指标计算逻辑
- V3 结果写入器：表创建、批量 upsert、管线集成
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, call, patch

import pytest

from src.skill_extraction.v3_pipeline import (
    RecordResult,
    V3Pipeline,
    _get_match_text,
    _merge_deduplicate,
    _safe_text,
)
from src.skill_extraction.v3_result_writer import (
    _prepare_row,
    create_v3_results_table,
    write_v3_results,
)

from src.skill_extraction.eval_v3 import (
    HardSkillMetrics,
    HardSkillSample,
    SoftSkillMetrics,
    SoftSkillSample,
    V3EvalReport,
    evaluate_hard_skills,
    evaluate_soft_skills,
    _compute_precision_recall_f1,
    _normalize_skill_name,
    _parse_skill_list,
)


# ─── 辅助：模拟匹配器 ───────────────────────────────────────────────────────


class _MockHardSkillMatcher:
    """可配置返回值的模拟硬技能匹配器。"""

    def __init__(self, results: List[dict] | None = None) -> None:
        self._results = results or []
        self.call_count = 0
        self.last_text: str | None = None

    def match_text(self, text: str) -> List[dict]:
        self.call_count += 1
        self.last_text = text
        return list(self._results)


class _MockSoftSkillMatcher:
    """可配置返回值的模拟软技能匹配器。"""

    def __init__(self, results: List[dict] | None = None) -> None:
        self._results = results or []
        self.call_count = 0
        self.last_text: str | None = None

    def match_text(self, text: str) -> List[dict]:
        self.call_count += 1
        self.last_text = text
        return list(self._results)


class _MockLLMClient:
    """模拟 LLM 客户端，直接透传候选（不做过滤）。"""

    def __init__(self, response: str | None = None) -> None:
        self._response = response
        self.call_count = 0

    def complete_text(self, *, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        if self._response is not None:
            return self._response
        # 默认：确认所有候选
        return "[]"


# ─── RecordResult 测试 ───────────────────────────────────────────────────────


class TestRecordResult:
    """测试 RecordResult 数据结构。"""

    def test_empty_result(self):
        """空结果应返回正确的计数和字典格式。"""
        result = RecordResult(recruitment_record_id="r1", job_title="工程师")
        assert result.hard_skill_count == 0
        assert result.soft_skill_count == 0
        d = result.to_dict()
        assert d["recruitment_record_id"] == "r1"
        assert d["job_title"] == "工程师"
        assert d["hard_skills"] == []
        assert d["soft_skills"] == []
        assert d["hard_skill_count"] == 0
        assert d["soft_skill_count"] == 0

    def test_with_skills(self):
        """有技能数据时应正确计算计数。"""
        result = RecordResult(
            recruitment_record_id="r2",
            job_title="开发",
            hard_skills=[
                {"name": "Python", "category": "programming_language"},
                {"name": "SQL", "category": "database"},
            ],
            soft_skills=[
                {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
            ],
        )
        assert result.hard_skill_count == 2
        assert result.soft_skill_count == 1
        d = result.to_dict()
        assert len(d["hard_skills"]) == 2
        assert len(d["soft_skills"]) == 1

    def test_to_dict_keys(self):
        """to_dict 输出应包含所有必要字段。"""
        result = RecordResult(recruitment_record_id="r3", job_title="测试")
        keys = set(result.to_dict().keys())
        expected = {
            "recruitment_record_id", "job_title",
            "hard_skills", "hard_skill_count",
            "soft_skills", "soft_skill_count",
        }
        assert keys == expected


# ─── _safe_text 测试 ────────────────────────────────────────────────────────


class TestSafeText:
    """测试 _safe_text 辅助函数。"""

    def test_none_returns_empty(self):
        assert _safe_text(None) == ""

    def test_normal_string(self):
        assert _safe_text("hello") == "hello"

    def test_nan_string(self):
        assert _safe_text("nan") == ""
        assert _safe_text("NaN") == ""

    def test_none_string(self):
        assert _safe_text("None") == ""

    def test_whitespace(self):
        assert _safe_text("  hello  ") == "hello"

    def test_integer(self):
        assert _safe_text(42) == "42"


# ─── _get_match_text 测试 ───────────────────────────────────────────────────


class TestGetMatchText:
    """测试 _get_match_text 文本提取优先级。"""

    def test_requirements_text_priority(self):
        """requirements_text 优先级最高。"""
        record = {
            "requirements_text": "需要 Java 经验",
            "duties_text": "负责开发",
            "job_description_clean": "完整描述",
        }
        assert _get_match_text(record) == "需要 Java 经验"

    def test_duties_text_fallback(self):
        """requirements_text 为空时回退到 duties_text。"""
        record = {
            "requirements_text": "",
            "duties_text": "负责开发",
            "job_description_clean": "完整描述",
        }
        assert _get_match_text(record) == "负责开发"

    def test_job_description_clean_fallback(self):
        """前两个都为空时回退到 job_description_clean。"""
        record = {
            "requirements_text": "",
            "duties_text": "",
            "job_description_clean": "完整描述",
        }
        assert _get_match_text(record) == "完整描述"

    def test_all_empty(self):
        """所有字段为空时返回空字符串。"""
        record = {
            "requirements_text": None,
            "duties_text": None,
            "job_description_clean": None,
        }
        assert _get_match_text(record) == ""

    def test_missing_keys(self):
        """字段缺失时返回空字符串。"""
        assert _get_match_text({}) == ""


# ─── _merge_deduplicate 测试 ────────────────────────────────────────────────


class TestMergeDeduplicate:
    """测试硬技能优先去重逻辑。"""

    def test_no_overlap(self):
        """硬技能和软技能无重叠时，两者都保留。"""
        hard = [{"skill_name": "Python", "category": "programming_language"}]
        soft = [{"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"}]
        h, s = _merge_deduplicate(hard, soft)
        assert len(h) == 1
        assert len(s) == 1

    def test_overlap_prefers_hard(self):
        """同名技能同时出现时，归类为硬技能，从软技能中移除。"""
        hard = [{"skill_name": "Python", "category": "programming_language"}]
        soft = [{"name": "Python", "dimension": "other", "confidence": 0.8, "source": "dict_match"}]
        h, s = _merge_deduplicate(hard, soft)
        assert len(h) == 1
        assert len(s) == 0

    def test_case_insensitive_overlap(self):
        """重叠检测应不区分大小写。"""
        hard = [{"skill_name": "Python", "category": "programming_language"}]
        soft = [{"name": "python", "dimension": "other", "confidence": 0.8, "source": "dict_match"}]
        h, s = _merge_deduplicate(hard, soft)
        assert len(h) == 1
        assert len(s) == 0

    def test_partial_overlap(self):
        """部分重叠时，不重叠的软技能保留。"""
        hard = [
            {"skill_name": "Python", "category": "programming_language"},
            {"skill_name": "SQL", "category": "database"},
        ]
        soft = [
            {"name": "SQL", "dimension": "other", "confidence": 0.8, "source": "dict_match"},
            {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
        ]
        h, s = _merge_deduplicate(hard, soft)
        assert len(h) == 2
        assert len(s) == 1
        assert s[0]["name"] == "沟通能力"

    def test_empty_inputs(self):
        """空输入应返回空输出。"""
        assert _merge_deduplicate([], []) == ([], [])


# ─── V3Pipeline.process_record 测试 ─────────────────────────────────────────


class TestV3PipelineProcessRecord:
    """测试 V3Pipeline 单条记录处理。"""

    def test_basic_dual_pipeline(self):
        """硬技能和软技能应同时被匹配。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Python", "category": "programming_language"},
            {"skill_name": "SQL", "category": "database"},
        ])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
            {"name": "团队协作", "dimension": "extraversion", "confidence": 0.85, "source": "dict_match"},
        ])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "001",
            "job_title": "Python 开发工程师",
            "requirements_text": "需要 Python 和 SQL，具备沟通能力和团队协作精神",
        }
        result = pipeline.process_record(record)

        assert result.recruitment_record_id == "001"
        assert result.job_title == "Python 开发工程师"
        assert result.hard_skill_count == 2
        assert result.soft_skill_count == 2
        assert result.hard_skills[0]["name"] == "Python"
        assert result.hard_skills[0]["category"] == "programming_language"
        assert result.soft_skills[0]["name"] == "沟通能力"

    def test_hard_skill_priority_rule(self):
        """同一技能名同时命中硬技能和软技能时，归类为硬技能。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Python", "category": "programming_language"},
        ])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "Python", "dimension": "other", "confidence": 0.8, "source": "dict_match"},
            {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
        ])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "002",
            "job_title": "开发",
            "requirements_text": "精通 Python，善于沟通",
        }
        result = pipeline.process_record(record)

        # Python 应归类为硬技能
        hard_names = {s["name"] for s in result.hard_skills}
        assert "Python" in hard_names
        # Python 不应出现在软技能中
        soft_names = {s["name"] for s in result.soft_skills}
        assert "Python" not in soft_names
        assert "沟通能力" in soft_names

    def test_hard_skill_priority_case_insensitive(self):
        """硬技能优先规则应不区分大小写。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Python", "category": "programming_language"},
        ])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "python", "dimension": "other", "confidence": 0.8, "source": "dict_match"},
        ])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "003",
            "job_title": "开发",
            "requirements_text": "Python",
        }
        result = pipeline.process_record(record)

        assert result.hard_skill_count == 1
        assert result.soft_skill_count == 0

    def test_empty_text_no_skills(self):
        """空文本不应匹配到任何技能。"""
        hard_matcher = _MockHardSkillMatcher([{"skill_name": "Java", "category": "programming_language"}])
        soft_matcher = _MockSoftSkillMatcher([{"name": "沟通", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"}])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "004",
            "job_title": "空岗位",
            "requirements_text": "",
            "duties_text": "",
            "job_description_clean": "",
        }
        result = pipeline.process_record(record)

        assert result.hard_skill_count == 0
        assert result.soft_skill_count == 0
        # 匹配器不应被调用（因为空文本）
        assert hard_matcher.call_count == 0
        assert soft_matcher.call_count == 0

    def test_missing_fields_handled(self):
        """字段缺失时应优雅处理。"""
        hard_matcher = _MockHardSkillMatcher([])
        soft_matcher = _MockSoftSkillMatcher([])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "005",
        }
        result = pipeline.process_record(record)

        assert result.recruitment_record_id == "005"
        assert result.job_title == ""
        assert result.hard_skill_count == 0
        assert result.soft_skill_count == 0

    def test_no_hard_skills_only_soft(self):
        """仅有软技能命中时应正确输出。"""
        hard_matcher = _MockHardSkillMatcher([])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "学习能力", "dimension": "openness", "confidence": 0.9, "source": "dict_match"},
        ])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "006",
            "job_title": "实习生",
            "requirements_text": "具备良好的学习能力",
        }
        result = pipeline.process_record(record)

        assert result.hard_skill_count == 0
        assert result.soft_skill_count == 1
        assert result.soft_skills[0]["name"] == "学习能力"

    def test_no_soft_skills_only_hard(self):
        """仅有硬技能命中时应正确输出。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Java", "category": "programming_language"},
        ])
        soft_matcher = _MockSoftSkillMatcher([])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "007",
            "job_title": "Java 开发",
            "requirements_text": "精通 Java",
        }
        result = pipeline.process_record(record)

        assert result.hard_skill_count == 1
        assert result.soft_skill_count == 0


# ─── V3Pipeline.process_records 测试 ────────────────────────────────────────


class TestV3PipelineProcessRecords:
    """测试 V3Pipeline 批量处理。"""

    def test_multiple_records(self):
        """应正确处理多条记录。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Python", "category": "programming_language"},
        ])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
        ])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        records = [
            {"recruitment_record_id": "r1", "job_title": "A", "requirements_text": "Python 和沟通"},
            {"recruitment_record_id": "r2", "job_title": "B", "requirements_text": "Python 和沟通"},
        ]
        results = pipeline.process_records(records)

        assert len(results) == 2
        assert results[0].recruitment_record_id == "r1"
        assert results[1].recruitment_record_id == "r2"
        assert hard_matcher.call_count == 2
        assert soft_matcher.call_count == 2

    def test_empty_records_list(self):
        """空记录列表应返回空结果。"""
        pipeline = V3Pipeline(_MockHardSkillMatcher(), _MockSoftSkillMatcher())
        results = pipeline.process_records([])
        assert results == []


# ─── V3Pipeline + LLM 验证测试 ─────────────────────────────────────────────


class TestV3PipelineWithLLM:
    """测试 V3Pipeline 集成 LLM 验证。"""

    def test_llm_validation_called(self):
        """启用 LLM 时，软技能应经过 LLM 验证。"""
        hard_matcher = _MockHardSkillMatcher([])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
            {"name": "创新", "dimension": "openness", "confidence": 0.85, "source": "dict_match"},
        ])

        # LLM 确认沟通能力，过滤创新
        llm_response = json.dumps([
            {"name": "沟通能力", "is_soft_skill": True, "dimension": "extraversion"},
            {"name": "创新", "is_soft_skill": False, "dimension": "openness"},
        ], ensure_ascii=False)
        llm_client = _MockLLMClient(response=llm_response)

        pipeline = V3Pipeline(hard_matcher, soft_matcher, llm_client=llm_client)

        record = {
            "recruitment_record_id": "r1",
            "job_title": "测试",
            "requirements_text": "善于沟通和创新",
        }
        result = pipeline.process_record(record)

        # LLM 过滤了创新，只保留沟通能力
        assert result.soft_skill_count == 1
        assert result.soft_skills[0]["name"] == "沟通能力"
        assert llm_client.call_count == 1

    def test_no_llm_skips_validation(self):
        """未启用 LLM 时，软技能不做二次验证。"""
        hard_matcher = _MockHardSkillMatcher([])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
        ])

        pipeline = V3Pipeline(hard_matcher, soft_matcher, llm_client=None)

        record = {
            "recruitment_record_id": "r1",
            "job_title": "测试",
            "requirements_text": "善于沟通",
        }
        result = pipeline.process_record(record)

        # 不经过 LLM 验证，原始结果保留
        assert result.soft_skill_count == 1
        assert result.soft_skills[0]["source"] == "dict_match"

    def test_llm_not_called_when_no_soft_candidates(self):
        """无软技能候选时，不应调用 LLM。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Java", "category": "programming_language"},
        ])
        soft_matcher = _MockSoftSkillMatcher([])
        llm_client = _MockLLMClient()

        pipeline = V3Pipeline(hard_matcher, soft_matcher, llm_client=llm_client)

        record = {
            "recruitment_record_id": "r1",
            "job_title": "开发",
            "requirements_text": "精通 Java",
        }
        pipeline.process_record(record)

        assert llm_client.call_count == 0


# ─── 端到端集成测试（使用真实匹配器 mock）────────────────────────────────────


class TestV3PipelineEndToEnd:
    """端到端集成测试，模拟完整流程。"""

    def test_full_pipeline_output_format(self):
        """验证完整的输出格式符合规范。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Python", "category": "programming_language"},
            {"skill_name": "MySQL", "category": "database"},
        ])
        soft_matcher = _MockSoftSkillMatcher([
            {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
            {"name": "团队协作", "dimension": "extraversion", "confidence": 0.85, "source": "dict_match"},
            {"name": "Python", "dimension": "other", "confidence": 0.8, "source": "dict_match"},  # 与硬技能重叠
        ])
        pipeline = V3Pipeline(hard_matcher, soft_matcher)

        record = {
            "recruitment_record_id": "rec_001",
            "job_title": "Python 全栈开发工程师",
            "requirements_text": "需要 Python 和 MySQL 经验，具备沟通能力和团队协作精神",
            "duties_text": "",
            "job_description_clean": "",
        }
        result = pipeline.process_record(record)
        d = result.to_dict()

        # 验证结构
        assert d["recruitment_record_id"] == "rec_001"
        assert d["job_title"] == "Python 全栈开发工程师"
        assert d["hard_skill_count"] == 2
        assert d["soft_skill_count"] == 2  # Python 被硬技能优先规则过滤

        # 验证硬技能内容
        hard_names = {s["name"] for s in d["hard_skills"]}
        assert "Python" in hard_names
        assert "MySQL" in hard_names
        for skill in d["hard_skills"]:
            assert "name" in skill
            assert "category" in skill

        # 验证软技能内容
        soft_names = {s["name"] for s in d["soft_skills"]}
        assert "Python" not in soft_names  # 硬技能优先
        assert "沟通能力" in soft_names
        assert "团队协作" in soft_names
        for skill in d["soft_skills"]:
            assert "name" in skill
            assert "dimension" in skill
            assert "confidence" in skill
            assert "source" in skill

    def test_all_empty_fields(self):
        """所有字段为空时应返回空结果，不报错。"""
        pipeline = V3Pipeline(_MockHardSkillMatcher(), _MockSoftSkillMatcher())

        record = {"recruitment_record_id": "empty", "job_title": ""}
        result = pipeline.process_record(record)

        assert result.hard_skill_count == 0
        assert result.soft_skill_count == 0
        assert result.to_dict()["hard_skills"] == []
        assert result.to_dict()["soft_skills"] == []


# ─── 评估脚本辅助函数测试 ────────────────────────────────────────────────────


class TestComputePrecisionRecallF1:
    """测试精确率/召回率/F1 计算。"""

    def test_perfect_match(self):
        """完美匹配时 P=R=F1=1.0。"""
        result = _compute_precision_recall_f1(tp=10, fp=0, fn=0)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_predictions(self):
        """无预测时 P=0, R=0, F1=0。"""
        result = _compute_precision_recall_f1(tp=0, fp=0, fn=10)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_match(self):
        """部分匹配时应正确计算。"""
        result = _compute_precision_recall_f1(tp=5, fp=2, fn=3)
        # precision = 5/7, recall = 5/8
        assert abs(result["precision"] - 5 / 7) < 1e-6
        assert abs(result["recall"] - 5 / 8) < 1e-6
        expected_f1 = 2 * (5 / 7) * (5 / 8) / ((5 / 7) + (5 / 8))
        assert abs(result["f1"] - expected_f1) < 1e-6

    def test_all_false_positives(self):
        """全部误报时 R=0。"""
        result = _compute_precision_recall_f1(tp=0, fp=5, fn=0)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


class TestNormalizeSkillName:
    """测试技能名称归一化。"""

    def test_case_insensitive(self):
        """应不区分大小写。"""
        assert _normalize_skill_name("Python") == _normalize_skill_name("python")
        assert _normalize_skill_name("SQL") == _normalize_skill_name("sql")

    def test_whitespace_trimmed(self):
        """应去除首尾空白。"""
        assert _normalize_skill_name("  Java  ") == "java"

    def test_empty(self):
        """空字符串应返回空字符串。"""
        assert _normalize_skill_name("") == ""
        assert _normalize_skill_name(None) == ""


class TestParseSkillList:
    """测试技能列表解析。"""

    def test_list_input(self):
        """列表输入应直接返回。"""
        assert _parse_skill_list(["Python", "Java"]) == ["Python", "Java"]

    def test_json_string(self):
        """JSON 字符串应正确解析。"""
        assert _parse_skill_list('["Python", "Java"]') == ["Python", "Java"]

    def test_pipe_separated(self):
        """管道分隔应正确解析。"""
        assert _parse_skill_list("Python|Java|SQL") == ["Python", "Java", "SQL"]

    def test_comma_separated(self):
        """逗号分隔应正确解析。"""
        assert _parse_skill_list("Python, Java, SQL") == ["Python", "Java", "SQL"]

    def test_single_skill(self):
        """单个技能应返回单元素列表。"""
        assert _parse_skill_list("Python") == ["Python"]

    def test_none_input(self):
        """None 输入应返回空列表。"""
        assert _parse_skill_list(None) == []

    def test_empty_string(self):
        """空字符串应返回空列表。"""
        assert _parse_skill_list("") == []


# ─── 硬技能评估测试 ──────────────────────────────────────────────────────────


class _EvalMockHardSkillMatcher:
    """评估测试用的模拟硬技能匹配器。"""

    def __init__(self, results_map: dict | None = None) -> None:
        self._results_map = results_map or {}
        self._default = []

    def match_text(self, text: str) -> List[dict]:
        for key, value in self._results_map.items():
            if key in text:
                return value
        return self._default


class _EvalMockSoftSkillMatcher:
    """评估测试用的模拟软技能匹配器。"""

    def __init__(self, results_map: dict | None = None) -> None:
        self._results_map = results_map or {}
        self._default = []

    def match_text(self, text: str) -> List[dict]:
        for key, value in self._results_map.items():
            if key in text:
                return value
        return self._default


class TestEvaluateHardSkills:
    """测试硬技能评估逻辑。"""

    def test_perfect_match(self):
        """完美匹配时 P=R=F1=1.0。"""
        samples = [
            HardSkillSample(
                sample_id="s1",
                text="需要 Python 和 SQL",
                gold_skills=["Python", "SQL"],
            ),
        ]
        matcher = _EvalMockHardSkillMatcher({
            "Python": [
                {"skill_name": "Python", "category": "programming_language"},
                {"skill_name": "SQL", "category": "database"},
            ],
        })
        # matcher 在匹配 "需要 Python 和 SQL" 时返回 Python+SQL
        metrics = evaluate_hard_skills(samples, matcher)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.sample_count == 1

    def test_false_positive(self):
        """误报应降低精确率。"""
        samples = [
            HardSkillSample(
                sample_id="s1",
                text="需要 Python",
                gold_skills=["Python"],
            ),
        ]
        # 匹配器返回了 Python 和 Java，但 Java 不在 gold 中
        matcher = _EvalMockHardSkillMatcher({
            "Python": [
                {"skill_name": "Python", "category": "programming_language"},
                {"skill_name": "Java", "category": "programming_language"},
            ],
        })
        metrics = evaluate_hard_skills(samples, matcher)
        # precision = 1/2, recall = 1/1
        assert abs(metrics.precision - 0.5) < 1e-6
        assert metrics.recall == 1.0

    def test_false_negative(self):
        """漏报应降低召回率。"""
        samples = [
            HardSkillSample(
                sample_id="s1",
                text="需要 Python 和 Java",
                gold_skills=["Python", "Java"],
            ),
        ]
        # 匹配器只返回了 Python
        matcher = _EvalMockHardSkillMatcher({
            "Python": [
                {"skill_name": "Python", "category": "programming_language"},
            ],
        })
        metrics = evaluate_hard_skills(samples, matcher)
        assert metrics.precision == 1.0
        # recall = 1/2
        assert abs(metrics.recall - 0.5) < 1e-6

    def test_category_accuracy(self):
        """分类准确率应正确计算。"""
        samples = [
            HardSkillSample(
                sample_id="s1",
                text="Python",
                gold_skills=["Python"],
                gold_categories={"python": "programming_language"},
            ),
        ]
        # 分类正确
        matcher_correct = _EvalMockHardSkillMatcher({
            "Python": [{"skill_name": "Python", "category": "programming_language"}],
        })
        metrics_correct = evaluate_hard_skills(samples, matcher_correct)
        assert metrics_correct.category_accuracy == 1.0

        # 分类错误
        matcher_wrong = _EvalMockHardSkillMatcher({
            "Python": [{"skill_name": "Python", "category": "database"}],
        })
        metrics_wrong = evaluate_hard_skills(samples, matcher_wrong)
        assert metrics_wrong.category_accuracy == 0.0

    def test_empty_samples(self):
        """空样本应返回零指标。"""
        metrics = evaluate_hard_skills([], _EvalMockHardSkillMatcher())
        assert metrics.sample_count == 0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_exact_match_rate(self):
        """完全匹配率应正确计算。"""
        samples = [
            HardSkillSample(sample_id="s1", text="A", gold_skills=["Python"]),
            HardSkillSample(sample_id="s2", text="B", gold_skills=["Java"]),
        ]
        # 第一个完美匹配，第二个有误报
        matcher = _EvalMockHardSkillMatcher({
            "A": [{"skill_name": "Python", "category": "programming_language"}],
            "B": [
                {"skill_name": "Java", "category": "programming_language"},
                {"skill_name": "SQL", "category": "database"},
            ],
        })
        metrics = evaluate_hard_skills(samples, matcher)
        # 只有第一个完全匹配
        assert abs(metrics.exact_match_rate - 0.5) < 1e-6


# ─── 软技能评估测试 ──────────────────────────────────────────────────────────


class TestEvaluateSoftSkills:
    """测试软技能评估逻辑。"""

    def test_perfect_coverage(self):
        """完美覆盖时覆盖率=1.0。"""
        samples = [
            SoftSkillSample(
                sample_id="s1",
                text="具备沟通能力和团队协作精神",
                gold_skills=[
                    {"name": "沟通能力", "dimension": "extraversion"},
                    {"name": "团队协作", "dimension": "extraversion"},
                ],
            ),
        ]
        matcher = _EvalMockSoftSkillMatcher({
            "沟通": [
                {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
                {"name": "团队协作", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
            ],
        })
        metrics = evaluate_soft_skills(samples, matcher)
        assert metrics.coverage == 1.0
        assert metrics.precision == 1.0
        assert metrics.dimension_accuracy == 1.0
        assert metrics.sample_count == 1

    def test_partial_coverage(self):
        """部分覆盖时覆盖率应正确计算。"""
        samples = [
            SoftSkillSample(
                sample_id="s1",
                text="需要沟通能力和创新思维",
                gold_skills=[
                    {"name": "沟通能力", "dimension": "extraversion"},
                    {"name": "创新思维", "dimension": "openness"},
                ],
            ),
        ]
        # 只匹配到了沟通能力
        matcher = _EvalMockSoftSkillMatcher({
            "沟通": [
                {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
            ],
        })
        metrics = evaluate_soft_skills(samples, matcher)
        # coverage = 1/2
        assert abs(metrics.coverage - 0.5) < 1e-6
        # precision = 1/1 (只预测了1个且命中)
        assert metrics.precision == 1.0

    def test_dimension_accuracy(self):
        """维度准确率应正确计算。"""
        samples = [
            SoftSkillSample(
                sample_id="s1",
                text="需要沟通能力",
                gold_skills=[
                    {"name": "沟通能力", "dimension": "extraversion"},
                ],
            ),
        ]
        # 维度错误
        matcher = _EvalMockSoftSkillMatcher({
            "沟通": [
                {"name": "沟通能力", "dimension": "openness", "confidence": 0.9, "source": "dict_match"},
            ],
        })
        metrics = evaluate_soft_skills(samples, matcher)
        assert metrics.coverage == 1.0  # 命中了
        assert metrics.dimension_accuracy == 0.0  # 但维度错误

    def test_empty_samples(self):
        """空样本应返回零指标。"""
        metrics = evaluate_soft_skills([], _EvalMockSoftSkillMatcher())
        assert metrics.sample_count == 0
        assert metrics.coverage == 0.0
        assert metrics.precision == 0.0

    def test_extra_predictions(self):
        """多余预测应降低精确率。"""
        samples = [
            SoftSkillSample(
                sample_id="s1",
                text="需要沟通能力",
                gold_skills=[
                    {"name": "沟通能力", "dimension": "extraversion"},
                ],
            ),
        ]
        # 匹配器返回了2个，但 gold 只有1个
        matcher = _EvalMockSoftSkillMatcher({
            "沟通": [
                {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
                {"name": "团队协作", "dimension": "extraversion", "confidence": 0.8, "source": "dict_match"},
            ],
        })
        metrics = evaluate_soft_skills(samples, matcher)
        # coverage = 1/1
        assert metrics.coverage == 1.0
        # precision = 1/2 (预测2个，命中1个)
        assert abs(metrics.precision - 0.5) < 1e-6

    def test_error_rows_recorded(self):
        """误差行应被正确记录。"""
        samples = [
            SoftSkillSample(
                sample_id="s1",
                text="需要沟通能力和创新思维",
                gold_skills=[
                    {"name": "沟通能力", "dimension": "extraversion"},
                    {"name": "创新思维", "dimension": "openness"},
                ],
            ),
        ]
        matcher = _EvalMockSoftSkillMatcher({
            "沟通": [
                {"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"},
            ],
        })
        metrics = evaluate_soft_skills(samples, matcher)
        # 创新思维缺失，应记录误差
        assert len(metrics.error_rows) == 1
        assert metrics.error_rows[0]["sample_id"] == "s1"


# ─── 数据结构测试 ────────────────────────────────────────────────────────────


class TestEvalDataStructures:
    """测试评估数据结构。"""

    def test_hard_skill_metrics_to_dict(self):
        """HardSkillMetrics 应正确转换为字典。"""
        metrics = HardSkillMetrics(
            precision=0.9, recall=0.8, f1=0.85,
            category_accuracy=0.95, exact_match_rate=0.7,
            tp=10, fp=2, fn=3, sample_count=5,
        )
        d = metrics.to_dict()
        assert d["precision"] == 0.9
        assert d["recall"] == 0.8
        assert d["f1"] == 0.85
        assert d["category_accuracy"] == 0.95
        assert d["exact_match_rate"] == 0.7
        assert d["tp"] == 10
        assert d["fp"] == 2
        assert d["fn"] == 3
        assert d["sample_count"] == 5

    def test_soft_skill_metrics_to_dict(self):
        """SoftSkillMetrics 应正确转换为字典。"""
        metrics = SoftSkillMetrics(
            coverage=0.8, precision=0.9, dimension_accuracy=0.95,
            predicted_count=10, gold_count=8, matched_count=7,
            sample_count=5,
        )
        d = metrics.to_dict()
        assert d["coverage"] == 0.8
        assert d["precision"] == 0.9
        assert d["dimension_accuracy"] == 0.95
        assert d["predicted_count"] == 10
        assert d["gold_count"] == 8
        assert d["matched_count"] == 7
        assert d["sample_count"] == 5

    def test_v3_eval_report_to_dict(self):
        """V3EvalReport 应正确转换为字典。"""
        report = V3EvalReport(
            evaluated_at="2026-01-01T00:00:00",
            hard_skill_metrics=HardSkillMetrics(precision=0.9),
            soft_skill_metrics=SoftSkillMetrics(coverage=0.8),
            dataset_summary={"hard_skill_sample_count": 10},
        )
        d = report.to_dict()
        assert d["evaluated_at"] == "2026-01-01T00:00:00"
        assert d["hard_skill_metrics"]["precision"] == 0.9
        assert d["soft_skill_metrics"]["coverage"] == 0.8
        assert d["dataset_summary"]["hard_skill_sample_count"] == 10


# ─── _prepare_row 测试 ──────────────────────────────────────────────────────


class TestPrepareRow:
    """测试 _prepare_row 将结果字典转换为 SQL 行元组。"""

    def test_basic_conversion(self):
        """应正确转换标准结果字典。"""
        result = {
            "recruitment_record_id": "r1",
            "source_table": "public.job_description_parsed",
            "source_row_number": 42,
            "job_title": "Python 开发",
            "hard_skills": [{"name": "Python", "category": "programming_language"}],
            "hard_skill_count": 1,
            "soft_skills": [{"name": "沟通能力", "dimension": "extraversion"}],
            "soft_skill_count": 1,
            "pipeline_version": "v3",
            "extracted_at": datetime(2026, 6, 10, 12, 0, 0),
        }
        row = _prepare_row(result)

        assert row[0] == "r1"  # recruitment_record_id
        assert row[1] == "public.job_description_parsed"  # source_table
        assert row[2] == 42  # source_row_number
        assert row[3] == "Python 开发"  # job_title
        # hard_skills 应被序列化为 JSON 字符串
        assert json.loads(row[4]) == [{"name": "Python", "category": "programming_language"}]
        assert row[5] == 1  # hard_skill_count
        # soft_skills 应被序列化为 JSON 字符串
        assert json.loads(row[6]) == [{"name": "沟通能力", "dimension": "extraversion"}]
        assert row[7] == 1  # soft_skill_count
        assert row[8] == "v3"  # pipeline_version
        assert row[9] == datetime(2026, 6, 10, 12, 0, 0)  # extracted_at

    def test_empty_skills(self):
        """空技能列表应序列化为空 JSON 数组。"""
        result = {
            "recruitment_record_id": "r2",
            "job_title": "测试",
        }
        row = _prepare_row(result)
        assert json.loads(row[4]) == []
        assert json.loads(row[6]) == []

    def test_missing_optional_fields(self):
        """可选字段缺失时应使用默认值。"""
        result = {"recruitment_record_id": "r3"}
        row = _prepare_row(result)
        assert row[0] == "r3"
        assert row[1] is None  # source_table
        assert row[2] is None  # source_row_number
        assert row[3] == ""  # job_title
        assert row[8] == "v3"  # pipeline_version 默认值
        assert isinstance(row[9], datetime)  # extracted_at 默认当前时间

    def test_extracted_at_none_uses_now(self):
        """extracted_at 为 None 时应使用当前时间。"""
        result = {
            "recruitment_record_id": "r4",
            "job_title": "测试",
            "extracted_at": None,
        }
        row = _prepare_row(result)
        assert isinstance(row[9], datetime)


# ─── create_v3_results_table 测试 ──────────────────────────────────────────


class TestCreateV3ResultsTable:
    """测试 create_v3_results_table 表创建逻辑（mock PostgreSQL）。"""

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_creates_table_and_indexes(self, mock_get_conn):
        """应执行建表和建索引的 SQL 语句。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        create_v3_results_table()

        # 验证执行了 4 条 SQL（表 + 3 个索引）
        assert mock_cursor.execute.call_count == 4
        # 验证提交和关闭
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

        # 验证 SQL 包含预期的关键字
        executed_sqls = [args[0][0] for args in mock_cursor.execute.call_args_list]
        assert any("CREATE TABLE" in sql and "skill_extraction_v3_results" in sql for sql in executed_sqls)
        assert any("idx_v3_results_rid" in sql for sql in executed_sqls)
        assert any("idx_v3_results_hard_skills" in sql for sql in executed_sqls)
        assert any("idx_v3_results_soft_skills" in sql for sql in executed_sqls)

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_rollback_on_error(self, mock_get_conn):
        """出错时应 rollback 并抛出异常。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("模拟错误")
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        with pytest.raises(Exception, match="模拟错误"):
            create_v3_results_table()

        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_accepts_custom_pg_params(self, mock_get_conn):
        """应将自定义 pg_params 传递给 _get_connection。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        custom_params = {"host": "10.0.0.1", "port": 5433, "dbname": "TestDB"}
        create_v3_results_table(pg_params=custom_params)

        mock_get_conn.assert_called_once_with(custom_params)


# ─── write_v3_results 测试 ──────────────────────────────────────────────────


class TestWriteV3Results:
    """测试 write_v3_results 批量写入逻辑（mock PostgreSQL）。"""

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_writes_single_result(self, mock_get_conn):
        """应正确写入单条结果。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        results = [
            {
                "recruitment_record_id": "r1",
                "job_title": "Python 开发",
                "hard_skills": [{"name": "Python", "category": "programming_language"}],
                "hard_skill_count": 1,
                "soft_skills": [],
                "soft_skill_count": 0,
            }
        ]

        count = write_v3_results(results)

        assert count == 1
        assert mock_cursor.execute.call_count == 1
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_writes_multiple_results(self, mock_get_conn):
        """应正确写入多条结果。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        results = [
            {"recruitment_record_id": f"r{i}", "job_title": f"岗位{i}"}
            for i in range(10)
        ]

        count = write_v3_results(results)

        assert count == 10
        assert mock_cursor.execute.call_count == 10

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_batch_commit(self, mock_get_conn):
        """应按 batch_size 分批提交。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        results = [
            {"recruitment_record_id": f"r{i}", "job_title": f"岗位{i}"}
            for i in range(10)
        ]

        # batch_size=3，10 条记录应提交 4 次（第3、6、9行各一次 + 最终一次）
        count = write_v3_results(results, batch_size=3)

        assert count == 10
        assert mock_conn.commit.call_count == 4

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_empty_results_returns_zero(self, mock_get_conn):
        """空结果列表应返回 0 且不连接数据库。"""
        count = write_v3_results([])

        assert count == 0
        mock_get_conn.assert_not_called()

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_rollback_on_error(self, mock_get_conn):
        """出错时应 rollback。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("写入失败")
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        with pytest.raises(Exception, match="写入失败"):
            write_v3_results([{"recruitment_record_id": "r1"}])

        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.skill_extraction.v3_result_writer._get_connection")
    def test_upsert_sql_contains_on_conflict(self, mock_get_conn):
        """SQL 应包含 ON CONFLICT 子句实现 upsert。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        write_v3_results([{"recruitment_record_id": "r1", "job_title": "测试"}])

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "ON CONFLICT" in executed_sql
        assert "recruitment_record_id" in executed_sql


# ─── V3Pipeline + writer 集成测试 ──────────────────────────────────────────


class TestV3PipelineWithWriter:
    """测试 V3Pipeline 与写入器的集成。"""

    def test_writer_called_in_run(self):
        """run() 完成后应调用 writer 写入结果。"""
        hard_matcher = _MockHardSkillMatcher([
            {"skill_name": "Python", "category": "programming_language"},
        ])
        soft_matcher = _MockSoftSkillMatcher([])

        mock_writer = MagicMock(return_value=2)

        pipeline = V3Pipeline(
            hard_matcher, soft_matcher, writer=mock_writer,
        )

        # mock _fetch_records 返回数据
        fake_records = [
            {"recruitment_record_id": "r1", "job_title": "A", "requirements_text": "Python"},
            {"recruitment_record_id": "r2", "job_title": "B", "requirements_text": "Python"},
        ]

        with patch.object(V3Pipeline, "_fetch_records", return_value=fake_records):
            with patch("config.paths.get_project_paths") as mock_paths:
                mock_paths.return_value.pg_connection_params = {"host": "localhost"}
                results = pipeline.run()

        assert len(results) == 2
        # writer 应被调用一次，传入 2 条结果
        mock_writer.assert_called_once()
        written_data = mock_writer.call_args[0][0]
        assert len(written_data) == 2
        assert written_data[0]["recruitment_record_id"] == "r1"

    def test_no_writer_skips_writing(self):
        """未配置 writer 时不应报错。"""
        hard_matcher = _MockHardSkillMatcher([])
        soft_matcher = _MockSoftSkillMatcher([])

        pipeline = V3Pipeline(hard_matcher, soft_matcher, writer=None)

        fake_records = [
            {"recruitment_record_id": "r1", "job_title": "A", "requirements_text": "test"},
        ]

        with patch.object(V3Pipeline, "_fetch_records", return_value=fake_records):
            with patch("config.paths.get_project_paths") as mock_paths:
                mock_paths.return_value.pg_connection_params = {"host": "localhost"}
                results = pipeline.run()

        assert len(results) == 1

    def test_writer_not_called_when_no_results(self):
        """无结果时不应调用 writer。"""
        hard_matcher = _MockHardSkillMatcher([])
        soft_matcher = _MockSoftSkillMatcher([])
        mock_writer = MagicMock(return_value=0)

        pipeline = V3Pipeline(hard_matcher, soft_matcher, writer=mock_writer)

        with patch.object(V3Pipeline, "_fetch_records", return_value=[]):
            with patch("config.paths.get_project_paths") as mock_paths:
                mock_paths.return_value.pg_connection_params = {"host": "localhost"}
                results = pipeline.run()

        assert len(results) == 0
        mock_writer.assert_not_called()
