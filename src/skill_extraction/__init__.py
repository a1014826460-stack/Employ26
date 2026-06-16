"""技能抽取包。

当前正式主线是 V3 PostgreSQL 双管线：硬技能词典匹配 + 软技能词典/LLM
验证，结果写入 ``public.skill_extraction_v3_results``。
"""

from .hard import FlatHardSkillMatcher
from .pipeline import V3Pipeline as SkillExtractionPipeline
from .soft import SoftSkillMatcher

__all__ = [
    "FlatHardSkillMatcher",
    "SoftSkillMatcher",
    "SkillExtractionPipeline",
]
