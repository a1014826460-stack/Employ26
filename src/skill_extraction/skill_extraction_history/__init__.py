"""历史版技能抽取模块。"""

__all__ = [
    "OccupationSampleBuilder",
    "OccupationSkillDictionaryStore",
    "OccupationSkillPipeline",
    "RequirementCoverageEvaluator",
]


def __getattr__(name: str):
    if name == "OccupationSampleBuilder":
        from .data_source import OccupationSampleBuilder

        return OccupationSampleBuilder
    if name == "OccupationSkillDictionaryStore":
        from .dictionary_store import OccupationSkillDictionaryStore

        return OccupationSkillDictionaryStore
    if name == "OccupationSkillPipeline":
        from .occupation_skill_pipeline import OccupationSkillPipeline

        return OccupationSkillPipeline
    if name == "RequirementCoverageEvaluator":
        from .coverage import RequirementCoverageEvaluator

        return RequirementCoverageEvaluator
    raise AttributeError(name)
