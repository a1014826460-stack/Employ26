"""Tests for PostgreSQL skill dictionary audit rules."""

from __future__ import annotations

from dataclasses import dataclass

from src.skill_extraction.evaluation.pg_dictionary_audit import (
    PgDictionarySnapshot,
    SkillDictionaryRow,
    audit_alias_conflicts,
    audit_risky_aliases,
    build_audit_report,
)


@dataclass
class _HardSample:
    gold_skills: list[str]


@dataclass
class _SoftSample:
    gold_skills: list[dict[str, str]]


def test_audit_alias_conflicts_detects_cross_dictionary_and_alias_name_overlap():
    rows = [
        SkillDictionaryRow(
            dict_type="hard",
            skill_code="H1",
            skill_name="活动策划",
            category="process",
            aliases=["活动组织"],
            is_active=True,
            is_blacklisted=False,
        ),
        SkillDictionaryRow(
            dict_type="soft",
            skill_code="S1",
            skill_name="组织能力",
            category="extraversion",
            aliases=["活动组织"],
            is_active=False,
            is_blacklisted=True,
        ),
        SkillDictionaryRow(
            dict_type="soft",
            skill_code="S2",
            skill_name="谦逊",
            category="agreeableness",
            aliases=[],
            is_active=True,
            is_blacklisted=False,
        ),
        SkillDictionaryRow(
            dict_type="soft",
            skill_code="S3",
            skill_name="谦虚",
            category="agreeableness",
            aliases=["谦逊"],
            is_active=True,
            is_blacklisted=False,
        ),
    ]

    conflicts = audit_alias_conflicts(rows)
    conflict_types = {row["conflict_type"] for row in conflicts}

    assert "cross_hard_soft" in conflict_types
    assert "alias_equals_other_skill_name" in conflict_types


def test_audit_risky_aliases_flags_short_and_generic_terms():
    rows = [
        SkillDictionaryRow(
            dict_type="hard",
            skill_code="H1",
            skill_name="律师证",
            category="certification",
            aliases=["法律职业资格证书"],
            is_active=True,
            is_blacklisted=False,
        ),
        SkillDictionaryRow(
            dict_type="soft",
            skill_code="S1",
            skill_name="谦虚",
            category="agreeableness",
            aliases=["谦逊"],
            is_active=True,
            is_blacklisted=False,
        ),
    ]

    risky = audit_risky_aliases(rows)
    reasons = {row["alias"]: row["reason"] for row in risky}

    assert reasons["法律职业资格证书"] == "generic"
    assert reasons["谦逊"] == "too_short"


def test_build_audit_report_computes_gold_dictionary_coverage():
    snapshot = PgDictionarySnapshot(
        columns=[],
        rows=[
            SkillDictionaryRow(
                dict_type="hard",
                skill_code="H1",
                skill_name="Java",
                category="programming_language",
                aliases=["JDK"],
                is_active=True,
                is_blacklisted=False,
            ),
            SkillDictionaryRow(
                dict_type="soft",
                skill_code="S1",
                skill_name="沟通能力",
                category="extraversion",
                aliases=["沟通技巧"],
                is_active=True,
                is_blacklisted=False,
            ),
            SkillDictionaryRow(
                dict_type="soft",
                skill_code="S2",
                skill_name="外向",
                category="extraversion",
                aliases=["开朗"],
                is_active=False,
                is_blacklisted=True,
            ),
        ],
        counts={
            "hard_skills_rows": 1,
            "hard_blank_skill_names": 0,
            "hard_rows_with_aliases": 1,
            "soft_skills_rows": 2,
            "soft_active_rows": 1,
            "soft_blacklisted_rows": 1,
            "soft_blank_skill_names": 0,
            "soft_rows_with_aliases": 2,
        },
    )

    summary, conflicts, risky_aliases, probes = build_audit_report(
        snapshot=snapshot,
        hard_samples=[_HardSample(gold_skills=["Java", "Python"])],
        soft_samples=[
            _SoftSample(
                gold_skills=[
                    {"name": "沟通能力", "dimension": "extraversion"},
                    {"name": "开朗", "dimension": "extraversion"},
                ]
            )
        ],
        probe_limit=0,
    )

    coverage = summary["gold_dictionary_coverage"]
    assert coverage["hard_gold_coverage"] == 0.5
    assert coverage["hard_gold_missing"] == ["Python"]
    assert coverage["soft_gold_coverage"] == 0.5
    assert coverage["soft_gold_missing"] == ["开朗"]
    assert conflicts == []
    assert risky_aliases
    assert probes == []


def test_build_audit_report_distinguishes_canonical_conflict_from_wrong_mapping():
    class _SoftMatcher:
        def match_text(self, text: str):
            return [
                {
                    "name": "谦逊",
                    "dimension": "agreeableness",
                    "confidence": 0.9,
                    "source": "dict_match",
                }
            ]

    snapshot = PgDictionarySnapshot(
        columns=[],
        rows=[
            SkillDictionaryRow(
                dict_type="soft",
                skill_code="S1",
                skill_name="谦虚",
                category="agreeableness",
                aliases=["谦逊"],
                is_active=True,
                is_blacklisted=False,
            ),
            SkillDictionaryRow(
                dict_type="soft",
                skill_code="S2",
                skill_name="谦逊",
                category="agreeableness",
                aliases=[],
                is_active=True,
                is_blacklisted=False,
            ),
        ],
        counts={
            "hard_skills_rows": 0,
            "hard_blank_skill_names": 0,
            "hard_rows_with_aliases": 0,
            "soft_skills_rows": 2,
            "soft_active_rows": 2,
            "soft_blacklisted_rows": 0,
            "soft_blank_skill_names": 0,
            "soft_rows_with_aliases": 1,
        },
    )

    summary, _conflicts, _risky_aliases, probes = build_audit_report(
        snapshot=snapshot,
        hard_samples=[],
        soft_samples=[],
        soft_matcher=_SoftMatcher(),
    )

    assert probes[0]["mapping_status"] == "canonical_conflict"
    assert probes[0]["wrong_mapping_detected"] is False
    assert summary["alias_audit"]["wrong_mapping_probe_rows"] == 0
    assert summary["alias_audit"]["canonical_conflict_probe_rows"] == 1
