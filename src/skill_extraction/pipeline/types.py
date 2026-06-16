"""技能抽取管线共享数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordResult:
    """单条岗位的技能抽取结果。"""

    recruitment_record_id: str
    job_title: str
    source_table: str | None = None
    source_row_number: int | None = None
    hard_skills: list[dict[str, Any]] = field(default_factory=list)
    soft_skills: list[dict[str, Any]] = field(default_factory=list)
    pipeline_version: str = "v3"

    @property
    def hard_skill_count(self) -> int:
        """硬技能数量。"""
        return len(self.hard_skills)

    @property
    def soft_skill_count(self) -> int:
        """软技能数量。"""
        return len(self.soft_skills)

    def to_dict(self) -> dict[str, Any]:
        """转换为 PostgreSQL 写入和 JSON 输出使用的字典。"""
        return {
            "recruitment_record_id": self.recruitment_record_id,
            "source_table": self.source_table,
            "source_row_number": self.source_row_number,
            "job_title": self.job_title,
            "hard_skills": self.hard_skills,
            "hard_skill_count": self.hard_skill_count,
            "soft_skills": self.soft_skills,
            "soft_skill_count": self.soft_skill_count,
            "pipeline_version": self.pipeline_version,
        }
