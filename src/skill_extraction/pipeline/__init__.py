"""技能抽取 V3 PostgreSQL 管线。"""

from .pipeline import V3Pipeline, create_v3_pipeline
from .types import RecordResult
from .writer import create_v3_results_table, write_v3_results

__all__ = [
    "RecordResult",
    "V3Pipeline",
    "create_v3_pipeline",
    "create_v3_results_table",
    "write_v3_results",
]
