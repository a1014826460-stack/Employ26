"""硬技能匹配、分类与 PostgreSQL 写入。"""

from .matcher import FlatHardSkillMatcher, load_flat_dictionary, save_flat_dictionary
from .pg_matcher import run_match_pg, should_trigger_strong_revalidation

__all__ = [
    "FlatHardSkillMatcher",
    "load_flat_dictionary",
    "save_flat_dictionary",
    "run_match_pg",
    "should_trigger_strong_revalidation",
]
