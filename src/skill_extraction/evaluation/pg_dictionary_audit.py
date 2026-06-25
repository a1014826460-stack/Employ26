"""PostgreSQL skill dictionary coverage and alias audit.

This module verifies that ``dict.hard_skills`` and ``dict.soft_skills`` can
cover the current gold datasets, and surfaces alias mappings that may create
false positives or unstable canonical names.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from config.paths import get_project_paths


@dataclass(frozen=True)
class SkillDictionaryRow:
    """One skill row loaded from a PostgreSQL dictionary table."""

    dict_type: str
    skill_code: str
    skill_name: str
    category: str
    aliases: list[str]
    is_active: bool
    is_blacklisted: bool


@dataclass(frozen=True)
class PgDictionarySnapshot:
    """Dictionary rows and table metadata loaded from PostgreSQL."""

    columns: list[dict[str, str]]
    rows: list[SkillDictionaryRow]
    counts: dict[str, int]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "none", "nan", "null"} else text


def _normalize(value: Any) -> str:
    return _safe_text(value).casefold()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)]
    return []


def _term_entries(row: SkillDictionaryRow) -> list[tuple[str, str]]:
    terms = [(row.skill_name, "name")]
    terms.extend((alias, "alias") for alias in row.aliases)
    return [(term, role) for term, role in terms if _safe_text(term)]


def load_pg_dictionary_snapshot(
    *,
    schema: str = "dict",
    pg_params: dict[str, Any] | None = None,
) -> PgDictionarySnapshot:
    """Load hard/soft dictionary rows and basic table metadata from PostgreSQL."""
    import psycopg2
    from psycopg2 import sql

    params = pg_params or get_project_paths().pg_connection_params
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name IN ('hard_skills', 'soft_skills')
                ORDER BY table_name, ordinal_position
                """,
                (schema,),
            )
            columns = [
                {"table": table, "column": column, "data_type": data_type}
                for table, column, data_type in cur.fetchall()
            ]

            cur.execute(
                sql.SQL(
                    """
                    SELECT COUNT(*),
                           COUNT(*) FILTER (
                               WHERE skill_name IS NULL OR btrim(skill_name) = ''
                           ),
                           COUNT(*) FILTER (
                               WHERE aliases IS NOT NULL
                                 AND jsonb_typeof(aliases) = 'array'
                                 AND jsonb_array_length(aliases) > 0
                           )
                    FROM {}.hard_skills
                    """
                ).format(sql.Identifier(schema))
            )
            hard_count, hard_blank_names, hard_with_aliases = cur.fetchone()

            cur.execute(
                sql.SQL(
                    """
                    SELECT COUNT(*),
                           COUNT(*) FILTER (WHERE is_active = true),
                           COUNT(*) FILTER (WHERE is_blacklisted = true),
                           COUNT(*) FILTER (
                               WHERE skill_name IS NULL OR btrim(skill_name) = ''
                           ),
                           COUNT(*) FILTER (
                               WHERE aliases IS NOT NULL
                                 AND jsonb_typeof(aliases) = 'array'
                                 AND jsonb_array_length(aliases) > 0
                           )
                    FROM {}.soft_skills
                    """
                ).format(sql.Identifier(schema))
            )
            (
                soft_count,
                soft_active_count,
                soft_blacklisted_count,
                soft_blank_names,
                soft_with_aliases,
            ) = cur.fetchone()

            cur.execute(
                sql.SQL(
                    """
                    SELECT 'hard' AS dict_type, skill_code, skill_name, category,
                           aliases, true AS is_active, false AS is_blacklisted
                    FROM {}.hard_skills
                    UNION ALL
                    SELECT 'soft' AS dict_type, skill_code, skill_name, category,
                           aliases, is_active, is_blacklisted
                    FROM {}.soft_skills
                    """
                ).format(sql.Identifier(schema), sql.Identifier(schema))
            )
            raw_rows = cur.fetchall()
    finally:
        conn.close()

    rows = [
        SkillDictionaryRow(
            dict_type=dict_type,
            skill_code=_safe_text(skill_code),
            skill_name=_safe_text(skill_name),
            category=_safe_text(category),
            aliases=_as_list(aliases),
            is_active=bool(is_active),
            is_blacklisted=bool(is_blacklisted),
        )
        for (
            dict_type,
            skill_code,
            skill_name,
            category,
            aliases,
            is_active,
            is_blacklisted,
        ) in raw_rows
    ]

    counts = {
        "hard_skills_rows": int(hard_count),
        "hard_blank_skill_names": int(hard_blank_names),
        "hard_rows_with_aliases": int(hard_with_aliases),
        "soft_skills_rows": int(soft_count),
        "soft_active_rows": int(soft_active_count),
        "soft_blacklisted_rows": int(soft_blacklisted_count),
        "soft_blank_skill_names": int(soft_blank_names),
        "soft_rows_with_aliases": int(soft_with_aliases),
    }
    return PgDictionarySnapshot(columns=columns, rows=rows, counts=counts)


def _build_term_index(rows: Sequence[SkillDictionaryRow]) -> dict[str, list[dict[str, Any]]]:
    term_to_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.skill_name:
            continue
        for term, role in _term_entries(row):
            normalized = _normalize(term)
            if not normalized:
                continue
            term_to_entries[normalized].append(
                {
                    "dict_type": row.dict_type,
                    "skill_code": row.skill_code,
                    "skill_name": row.skill_name,
                    "category": row.category,
                    "term": term,
                    "term_role": role,
                    "is_active": row.is_active,
                    "is_blacklisted": row.is_blacklisted,
                }
            )
    return term_to_entries


def _dictionary_terms(
    rows: Sequence[SkillDictionaryRow],
) -> dict[str, set[str]]:
    terms = {"hard": set(), "soft_active": set(), "soft_all": set()}
    for row in rows:
        for term, _role in _term_entries(row):
            normalized = _normalize(term)
            if not normalized:
                continue
            if row.dict_type == "hard":
                terms["hard"].add(normalized)
            elif row.dict_type == "soft":
                terms["soft_all"].add(normalized)
                if row.is_active and not row.is_blacklisted:
                    terms["soft_active"].add(normalized)
    return terms


def audit_alias_conflicts(rows: Sequence[SkillDictionaryRow]) -> list[dict[str, str]]:
    """Find duplicate and cross-dictionary alias/name mappings."""
    term_to_entries = _build_term_index(rows)
    conflicts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for normalized_term, entries in term_to_entries.items():
        canonical_keys = {
            (entry["dict_type"], _normalize(entry["skill_name"])) for entry in entries
        }
        if len(canonical_keys) <= 1:
            continue
        dict_types = {entry["dict_type"] for entry in entries}
        conflict_type = (
            "cross_hard_soft"
            if len(dict_types) > 1
            else f"within_{next(iter(dict_types))}"
        )
        key = (normalized_term, conflict_type)
        if key in seen:
            continue
        seen.add(key)
        conflicts.append(
            {
                "normalized_term": normalized_term,
                "conflict_type": conflict_type,
                "mappings": json.dumps(entries, ensure_ascii=False),
            }
        )

    for normalized_term, entries in term_to_entries.items():
        if len(entries) <= 1:
            continue
        skill_names = {_normalize(entry["skill_name"]) for entry in entries}
        if normalized_term not in skill_names:
            continue
        has_alias_to_other_name = any(
            entry["term_role"] == "alias"
            and normalized_term != _normalize(entry["skill_name"])
            for entry in entries
        )
        if not has_alias_to_other_name:
            continue
        key = (normalized_term, "alias_equals_other_skill_name")
        if key in seen:
            continue
        seen.add(key)
        conflicts.append(
            {
                "normalized_term": normalized_term,
                "conflict_type": "alias_equals_other_skill_name",
                "mappings": json.dumps(entries, ensure_ascii=False),
            }
        )

    return conflicts


_GENERIC_ALIAS_PATTERNS = [
    re.compile(r"^(能力|知识|理论|工具|软件|系统|平台|证书|资格证|经验)$"),
    re.compile(r".*(资格证|资格证书|执业证|上岗证)$"),
]
_ASCII_ALIAS_PATTERN = re.compile(r"[A-Za-z0-9_+#.\-/]+")


def audit_risky_aliases(rows: Sequence[SkillDictionaryRow]) -> list[dict[str, Any]]:
    """Find short or generic aliases that are likely to over-match."""
    risky: list[dict[str, Any]] = []
    for row in rows:
        for alias in row.aliases:
            compact = re.sub(r"\s+", "", alias)
            is_ascii_like = bool(_ASCII_ALIAS_PATTERN.fullmatch(compact))
            too_short = (is_ascii_like and len(compact) <= 1) or (
                not is_ascii_like and len(compact) <= 2
            )
            generic = any(pattern.fullmatch(compact) for pattern in _GENERIC_ALIAS_PATTERNS)
            if not too_short and not generic:
                continue
            risky.append(
                {
                    "dict_type": row.dict_type,
                    "skill_code": row.skill_code,
                    "skill_name": row.skill_name,
                    "category": row.category,
                    "alias": alias,
                    "reason": ",".join(
                        item
                        for item in [
                            "too_short" if too_short else "",
                            "generic" if generic else "",
                        ]
                        if item
                    ),
                    "is_active": row.is_active,
                    "is_blacklisted": row.is_blacklisted,
                }
            )
    return risky


def _hard_gold_names(hard_samples: Sequence[Any]) -> list[str]:
    return sorted(
        {
            _safe_text(skill)
            for sample in hard_samples
            for skill in getattr(sample, "gold_skills", [])
            if _safe_text(skill)
        },
        key=str.casefold,
    )


def _soft_gold_names(soft_samples: Sequence[Any]) -> list[str]:
    names: set[str] = set()
    for sample in soft_samples:
        for item in getattr(sample, "gold_skills", []):
            if isinstance(item, dict):
                name = _safe_text(item.get("name"))
                if name:
                    names.add(name)
    return sorted(names, key=str.casefold)


def _probe_alias_mappings(
    risky_aliases: Sequence[dict[str, Any]],
    *,
    rows: Sequence[SkillDictionaryRow],
    hard_matcher: Any | None = None,
    soft_matcher: Any | None = None,
    probe_limit: int = 200,
) -> list[dict[str, Any]]:
    term_to_entries = _build_term_index(rows)
    probe_rows: list[dict[str, Any]] = []
    for item in risky_aliases[:probe_limit]:
        alias = str(item["alias"])
        expected_name = str(item["skill_name"])
        text = f"岗位要求包含{alias}相关经验。"
        alias_entries = term_to_entries.get(_normalize(alias), [])
        same_dict_canonical_names = {
            _normalize(entry["skill_name"])
            for entry in alias_entries
            if entry["dict_type"] == item["dict_type"]
        }
        if item["dict_type"] == "hard":
            matched = hard_matcher.match_text(text) if hard_matcher else []
            matched_names = {_normalize(match.get("skill_name")) for match in matched}
        else:
            matched = soft_matcher.match_text(text) if soft_matcher else []
            matched_names = {_normalize(match.get("name")) for match in matched}
        if not matched:
            mapping_status = "no_match"
        elif _normalize(expected_name) in matched_names:
            mapping_status = "expected"
        elif matched_names & same_dict_canonical_names:
            mapping_status = "canonical_conflict"
        else:
            mapping_status = "wrong_mapping"
        probe_rows.append(
            {
                "dict_type": item["dict_type"],
                "alias": alias,
                "expected_skill_name": expected_name,
                "matched": json.dumps(matched, ensure_ascii=False),
                "mapping_status": mapping_status,
                "wrong_mapping_detected": mapping_status == "wrong_mapping",
                "probe_text": text,
            }
        )
    return probe_rows


def build_audit_report(
    *,
    snapshot: PgDictionarySnapshot,
    hard_samples: Sequence[Any],
    soft_samples: Sequence[Any],
    hard_matcher: Any | None = None,
    soft_matcher: Any | None = None,
    probe_limit: int = 200,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build summary, conflict rows, risky aliases, and optional probe rows."""
    dictionary_terms = _dictionary_terms(snapshot.rows)
    hard_gold = _hard_gold_names(hard_samples)
    soft_gold = _soft_gold_names(soft_samples)
    missing_hard = [
        skill for skill in hard_gold if _normalize(skill) not in dictionary_terms["hard"]
    ]
    missing_soft = [
        skill
        for skill in soft_gold
        if _normalize(skill) not in dictionary_terms["soft_active"]
    ]

    conflicts = audit_alias_conflicts(snapshot.rows)
    risky_aliases = audit_risky_aliases(snapshot.rows)
    probes = _probe_alias_mappings(
        risky_aliases,
        rows=snapshot.rows,
        hard_matcher=hard_matcher,
        soft_matcher=soft_matcher,
        probe_limit=probe_limit,
    )

    counts = dict(snapshot.counts)
    counts.update(
        {
            "hard_terms_name_plus_alias": len(dictionary_terms["hard"]),
            "soft_active_terms_name_plus_alias": len(dictionary_terms["soft_active"]),
            "soft_all_terms_name_plus_alias": len(dictionary_terms["soft_all"]),
        }
    )

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "database": get_project_paths().pg_dbname,
        "schemas_checked": ["dict"],
        "table_columns": snapshot.columns,
        "dictionary_counts": counts,
        "gold_dictionary_coverage": {
            "hard_gold_unique": len(hard_gold),
            "hard_gold_covered_by_pg_name_or_alias": len(hard_gold) - len(missing_hard),
            "hard_gold_coverage": (len(hard_gold) - len(missing_hard))
            / max(len(hard_gold), 1),
            "hard_gold_missing": missing_hard,
            "soft_gold_unique": len(soft_gold),
            "soft_gold_covered_by_active_pg_name_or_alias": len(soft_gold)
            - len(missing_soft),
            "soft_gold_coverage": (len(soft_gold) - len(missing_soft))
            / max(len(soft_gold), 1),
            "soft_gold_missing": missing_soft,
        },
        "alias_audit": {
            "conflict_rows": len(conflicts),
            "conflict_type_counts": dict(
                Counter(row["conflict_type"] for row in conflicts)
            ),
            "risky_alias_rows": len(risky_aliases),
            "probe_rows": len(probes),
            "wrong_mapping_probe_rows": sum(
                1 for row in probes if row["wrong_mapping_detected"]
            ),
            "canonical_conflict_probe_rows": sum(
                1 for row in probes if row.get("mapping_status") == "canonical_conflict"
            ),
        },
    }
    return summary, conflicts, risky_aliases, probes


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_outputs(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    conflicts: Sequence[dict[str, Any]],
    risky_aliases: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Write JSON/CSV audit artifacts and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"pg_dictionary_audit_{stamp}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    conflict_path = output_dir / f"pg_alias_conflicts_{stamp}.csv"
    _write_csv(
        conflict_path,
        conflicts,
        ["normalized_term", "conflict_type", "mappings"],
    )

    risky_path = output_dir / f"pg_risky_aliases_{stamp}.csv"
    _write_csv(
        risky_path,
        risky_aliases,
        [
            "dict_type",
            "skill_code",
            "skill_name",
            "category",
            "alias",
            "reason",
            "is_active",
            "is_blacklisted",
        ],
    )

    probe_path = output_dir / f"pg_alias_probe_results_{stamp}.csv"
    _write_csv(
        probe_path,
        probes,
        [
            "dict_type",
            "alias",
            "expected_skill_name",
            "matched",
            "mapping_status",
            "wrong_mapping_detected",
            "probe_text",
        ],
    )

    return {
        "summary": summary_path,
        "conflicts": conflict_path,
        "risky_aliases": risky_path,
        "alias_probes": probe_path,
    }


def run_pg_dictionary_audit(
    *,
    output_dir: Path,
    hard_dataset: Path,
    soft_dataset: Path,
    schema: str = "dict",
    probe_limit: int = 200,
) -> dict[str, Any]:
    """Run PostgreSQL dictionary coverage and alias audit end to end."""
    from .v3 import _load_hard_skill_dataset, _load_soft_skill_dataset
    from ..hard.matcher import FlatHardSkillMatcher, load_flat_dictionary_from_pg
    from ..soft.matcher import SoftSkillMatcher

    snapshot = load_pg_dictionary_snapshot(schema=schema)
    hard_samples = _load_hard_skill_dataset(hard_dataset)
    soft_samples = _load_soft_skill_dataset(soft_dataset)

    hard_matcher = FlatHardSkillMatcher(load_flat_dictionary_from_pg())
    soft_matcher = SoftSkillMatcher()
    summary, conflicts, risky_aliases, probes = build_audit_report(
        snapshot=snapshot,
        hard_samples=hard_samples,
        soft_samples=soft_samples,
        hard_matcher=hard_matcher,
        soft_matcher=soft_matcher,
        probe_limit=probe_limit,
    )
    paths = write_audit_outputs(
        output_dir=output_dir,
        summary=summary,
        conflicts=conflicts,
        risky_aliases=risky_aliases,
        probes=probes,
    )
    return {
        "summary": summary,
        "paths": {key: str(value) for key, value in paths.items()},
    }
