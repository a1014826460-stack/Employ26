"""Shared label definitions for hard-skill context classification."""

from __future__ import annotations

from typing import Dict


VALID_HARD_SKILL_LABEL = "valid_hard_skill"
TOO_GENERIC_LABEL = "too_generic"
WRONG_ALIAS_MAPPING_LABEL = "wrong_alias_mapping"
NOT_SKILL_LABEL = "not_skill"

LABEL_TO_ID: Dict[str, int] = {
    VALID_HARD_SKILL_LABEL: 0,
    TOO_GENERIC_LABEL: 1,
    WRONG_ALIAS_MAPPING_LABEL: 2,
    NOT_SKILL_LABEL: 3,
}
ID_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_ID.items()}

# Prefer conservative rejection when vote counts are tied.
LABEL_TIE_BREAKER: Dict[str, int] = {
    WRONG_ALIAS_MAPPING_LABEL: 0,
    TOO_GENERIC_LABEL: 1,
    NOT_SKILL_LABEL: 2,
    VALID_HARD_SKILL_LABEL: 3,
}

DEFAULT_CONTEXT_THRESHOLD: float = 0.80

