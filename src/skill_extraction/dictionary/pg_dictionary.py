"""PostgreSQL skill dictionary loaders.

This module converts validated rows in the ``dict`` schema back into the
in-memory structures used by the existing V3 matchers.  Keeping the adapter
small lets the matcher logic stay file-source agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from config.paths import get_project_paths

logger = logging.getLogger(__name__)


def _connect(pg_params: dict[str, Any] | None = None):
    import psycopg2

    params = pg_params or get_project_paths().pg_connection_params
    return psycopg2.connect(**params)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def load_flat_dictionary_from_pg(
    *,
    pg_params: dict[str, Any] | None = None,
    schema: str = "dict",
) -> Dict[str, Any]:
    """Load hard skills from PostgreSQL in flat-dictionary shape."""
    query = f"""
        SELECT skill_code, skill_name, category, aliases, skill_type, notes,
               proficiency_level, last_updated, source_version, source_file
        FROM {schema}.hard_skills
        ORDER BY skill_name
    """

    conn = _connect(pg_params)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    skills: List[Dict[str, Any]] = []
    for (
        skill_code,
        skill_name,
        category,
        aliases,
        skill_type,
        notes,
        proficiency_level,
        last_updated,
        source_version,
        source_file,
    ) in rows:
        item: Dict[str, Any] = {
            "skill_code": skill_code,
            "name": skill_name,
            "aliases": _as_list(aliases),
            "category": category,
            "skill_type": skill_type,
            "notes": notes or "",
        }
        if proficiency_level:
            item["proficiency_level"] = proficiency_level
        skills.append(item)

    logger.info("已从 PostgreSQL 加载硬技能词典: %d 条", len(skills))
    return {
        "metadata": {
            "schema_version": 3,
            "source": f"{schema}.hard_skills",
            "source_version": rows[0][8] if rows else None,
            "source_file": rows[0][9] if rows else None,
            "last_updated": max((row[7] for row in rows), default=None).isoformat()
            if rows
            else None,
        },
        "skills": skills,
    }


def load_soft_dictionary_from_pg(
    *,
    pg_params: dict[str, Any] | None = None,
    schema: str = "dict",
    active_only: bool = True,
) -> Dict[str, Any]:
    """Load soft skills from PostgreSQL in soft dictionary JSON shape."""
    active_clause = "WHERE is_active = true" if active_only else ""
    query = f"""
        SELECT skill_code, skill_name, category, dimension_name, aliases,
               is_blacklisted, is_active, last_updated, source_version, source_file
        FROM {schema}.soft_skills
        {active_clause}
        ORDER BY category, skill_name
    """

    conn = _connect(pg_params)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    dimensions: Dict[str, Dict[str, Any]] = {}
    for (
        skill_code,
        skill_name,
        category,
        dimension_name,
        aliases,
        is_blacklisted,
        is_active,
        last_updated,
        source_version,
        source_file,
    ) in rows:
        bucket = dimensions.setdefault(
            category,
            {"name": dimension_name or category, "skills": []},
        )
        bucket["skills"].append(
            {
                "skill_code": skill_code,
                "name": skill_name,
                "aliases": _as_list(aliases),
                "dimension": category,
                "is_blacklisted": bool(is_blacklisted),
                "is_active": bool(is_active),
            }
        )

    logger.info(
        "已从 PostgreSQL 加载软技能词典: %d 条%s",
        len(rows),
        "（仅 active）" if active_only else "",
    )
    return {
        "schema_version": 1,
        "metadata": {
            "source": f"{schema}.soft_skills",
            "source_version": rows[0][8] if rows else None,
            "source_file": rows[0][9] if rows else None,
            "last_updated": max((row[7] for row in rows), default=None).isoformat()
            if rows
            else None,
        },
        "dimensions": dimensions,
    }


def load_hard_skill_names_from_pg(
    *,
    pg_params: dict[str, Any] | None = None,
    schema: str = "dict",
) -> set[str]:
    """Load hard skill names and aliases for soft-skill conflict filtering."""
    data = load_flat_dictionary_from_pg(pg_params=pg_params, schema=schema)
    names: set[str] = set()
    for skill in data.get("skills", []):
        name = str(skill.get("name") or "").strip()
        if name:
            names.add(name)
        for alias in skill.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if alias_text:
                names.add(alias_text)
    return names


def load_soft_blacklist_from_pg(
    *,
    pg_params: dict[str, Any] | None = None,
    schema: str = "dict",
) -> set[str]:
    """Load blacklisted soft skill names and aliases from PostgreSQL."""
    query = f"""
        SELECT skill_name, aliases
        FROM {schema}.soft_skills
        WHERE is_blacklisted = true OR is_active = false
    """
    conn = _connect(pg_params)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    terms: set[str] = set()
    for skill_name, aliases in rows:
        name = str(skill_name or "").strip()
        if name:
            terms.add(name)
        terms.update(_as_list(aliases))
    logger.info("已从 PostgreSQL 加载软技能黑名单: %d 个词汇", len(terms))
    return terms
