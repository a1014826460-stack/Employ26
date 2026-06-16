from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "skill_dictionary_iteration.json"


@lru_cache(maxsize=4)
def load_iteration_rules(path: str | Path | None = None) -> Dict[str, Any]:
    rules_path = Path(path) if path else DEFAULT_RULES_PATH
    if not rules_path.exists():
        return {}
    with open(rules_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def get_exact_generic_skill_blocklist(path: str | Path | None = None) -> set[str]:
    rules = load_iteration_rules(path)
    names = rules.get("generic_skill_blocklist", [])
    return {str(name).strip() for name in names if str(name).strip()}


def get_canonical_output_overrides(path: str | Path | None = None) -> Dict[str, str]:
    rules = load_iteration_rules(path)
    mapping = rules.get("canonical_output_overrides", {})
    normalized: Dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = str(raw_key).strip().casefold()
        value = str(raw_value).strip()
        if key and value:
            normalized[key] = value
    return normalized


def get_short_chinese_allowlist(path: str | Path | None = None) -> set[str]:
    rules = load_iteration_rules(path)
    names = rules.get("short_chinese_allowlist", [])
    return {str(name).strip() for name in names if str(name).strip()}


def get_contextual_term_rules(path: str | Path | None = None) -> list[Dict[str, Any]]:
    rules = load_iteration_rules(path)
    raw_rules = rules.get("contextual_term_rules", [])
    normalized: list[Dict[str, Any]] = []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        skill_name = str(item.get("skill_name", "")).strip()
        if not skill_name:
            continue
        normalized.append(
            {
                "skill_name": skill_name,
                "match_terms": [
                    str(term).strip()
                    for term in item.get("match_terms", [])
                    if str(term).strip()
                ],
                "require_any": [
                    str(pattern).strip()
                    for pattern in item.get("require_any", [])
                    if str(pattern).strip()
                ],
                "reject_if_any": [
                    str(pattern).strip()
                    for pattern in item.get("reject_if_any", [])
                    if str(pattern).strip()
                ],
            }
        )
    return normalized
