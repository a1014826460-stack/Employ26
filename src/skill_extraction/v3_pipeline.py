"""V3 统一技能抽取管线。

整合硬技能匹配（FlatHardSkillMatcher）和软技能匹配（SoftSkillMatcher + LLM 验证），
对岗位描述文本同时执行两条管线，合并输出结构化结果。

硬技能优先规则：当同一技能名同时命中硬技能和软技能时，归类为硬技能。

用法::

    # 从 PostgreSQL 读取数据运行
    python -m src.skill_extraction.v3_pipeline run

    # 从内存数据运行（测试或自定义数据源）
    python -m src.skill_extraction.v3_pipeline process --help
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ─── 输出数据结构 ───────────────────────────────────────────────────────────


@dataclass
class RecordResult:
    """单条岗位的技能抽取结果。"""

    recruitment_record_id: str
    job_title: str
    hard_skills: List[Dict[str, Any]] = field(default_factory=list)
    soft_skills: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def hard_skill_count(self) -> int:
        return len(self.hard_skills)

    @property
    def soft_skill_count(self) -> int:
        return len(self.soft_skills)

    def to_dict(self) -> Dict[str, Any]:
        """转换为输出字典格式。"""
        return {
            "recruitment_record_id": self.recruitment_record_id,
            "job_title": self.job_title,
            "hard_skills": self.hard_skills,
            "hard_skill_count": self.hard_skill_count,
            "soft_skills": self.soft_skills,
            "soft_skill_count": self.soft_skill_count,
        }


# ─── 辅助函数 ───────────────────────────────────────────────────────────────


def _safe_text(value: Any) -> str:
    """安全地将值转为字符串，None 和 NaN 返回空字符串。"""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", ""}:
        return ""
    return text


def _get_match_text(record: Dict[str, Any]) -> str:
    """从岗位记录中提取用于匹配的文本。

    优先级：
        1. ``requirements_text`` — 任职要求
        2. ``duties_text`` — 岗位职责
        3. ``job_description_clean`` — 清洗后的完整描述

    参数:
        record: 岗位数据记录。

    返回:
        str: 匹配文本，可能为空字符串。
    """
    for field_name in ("requirements_text", "duties_text", "job_description_clean"):
        text = _safe_text(record.get(field_name, ""))
        if text:
            return text
    return ""


def _merge_deduplicate(
    hard_skills: List[Dict[str, Any]],
    soft_skills: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """合并硬技能和软技能，应用硬技能优先规则。

    当同一技能名同时出现在硬技能和软技能中时，归类为硬技能，从软技能中移除。

    参数:
        hard_skills: 硬技能列表，每项含 ``skill_name`` 和 ``category``。
        soft_skills: 软技能列表，每项含 ``name``、``dimension``、``confidence``、``source``。

    返回:
        tuple: (去重后的硬技能列表, 去重后的软技能列表)
    """
    hard_skill_names = {s["skill_name"].casefold() for s in hard_skills}
    filtered_soft = [
        s for s in soft_skills if s["name"].casefold() not in hard_skill_names
    ]
    return hard_skills, filtered_soft


# ─── 核心管线类 ─────────────────────────────────────────────────────────────


class V3Pipeline:
    """V3 统一技能抽取管线。

    整合硬技能匹配（FlatHardSkillMatcher）和软技能匹配
    （SoftSkillMatcher + SoftSkillLLMValidator），对每条岗位描述
    同时运行两条管线并合并输出。

    参数:
        hard_skill_matcher: 平面化硬技能匹配器实例。
        soft_skill_matcher: 软技能词典匹配器实例。
        llm_client: LLM 客户端，用于软技能二次验证。为 None 时跳过 LLM 验证。
        writer: 结果写入函数，签名为 ``(list[dict]) -> int``。为 None 时不写入数据库。

    示例::

        >>> pipeline = V3Pipeline(hard_matcher, soft_matcher, llm_client)
        >>> results = pipeline.process_records(records)
        >>> for r in results:
        ...     print(r.to_dict())
    """

    def __init__(
        self,
        hard_skill_matcher: Any,
        soft_skill_matcher: Any,
        llm_client: Any = None,
        writer: Any = None,
    ) -> None:
        self._hard_matcher = hard_skill_matcher
        self._soft_matcher = soft_skill_matcher
        self._llm_client = llm_client
        self._writer = writer

    def process_record(self, record: Dict[str, Any]) -> RecordResult:
        """对单条岗位记录执行双管线技能抽取。

        参数:
            record: 岗位数据字典，需包含 ``recruitment_record_id``、
                    ``job_title`` 以及文本字段。

        返回:
            RecordResult: 结构化的技能抽取结果。
        """
        record_id = _safe_text(record.get("recruitment_record_id", ""))
        job_title = _safe_text(record.get("job_title", ""))
        match_text = _get_match_text(record)

        # ── 硬技能匹配 ──
        raw_hard = self._hard_matcher.match_text(match_text) if match_text else []
        # 标准化输出格式：FlatHardSkillMatcher 返回 [{"skill_name": ..., "category": ...}]
        hard_skills = [
            {"name": item["skill_name"], "category": item.get("category")}
            for item in raw_hard
        ]

        # ── 软技能匹配 ──
        raw_soft = self._soft_matcher.match_text(match_text) if match_text else []

        # ── 软技能 LLM 验证（可选）──
        if self._llm_client is not None and raw_soft:
            from .soft_skill_llm_validator import validate_soft_skills

            raw_soft = validate_soft_skills(
                candidates=raw_soft,
                context_text=match_text,
                llm_client=self._llm_client,
            )

        # ── 硬技能优先去重 ──
        hard_names = {s["name"].casefold() for s in hard_skills}
        soft_skills = [s for s in raw_soft if s["name"].casefold() not in hard_names]

        return RecordResult(
            recruitment_record_id=record_id,
            job_title=job_title,
            hard_skills=hard_skills,
            soft_skills=soft_skills,
        )

    def process_records(
        self,
        records: Sequence[Dict[str, Any]],
    ) -> List[RecordResult]:
        """对多条岗位记录执行双管线技能抽取。

        此方法接受内存中的数据，不依赖数据库连接，便于测试。

        参数:
            records: 岗位数据记录序列，每条为字典。

        返回:
            list[RecordResult]: 每条记录的技能抽取结果。
        """
        results: List[RecordResult] = []
        total = len(records)
        for idx, record in enumerate(records, 1):
            result = self.process_record(record)
            results.append(result)
            if idx % 100 == 0 or idx == total:
                logger.info("处理进度: %d / %d", idx, total)
        return results

    def run(
        self,
        source_table: str = "public.job_description_parsed",
    ) -> List[RecordResult]:
        """从 PostgreSQL 读取数据并运行双管线。

        如果构造时传入了 ``writer`` 参数，处理完成后自动将结果写入 PostgreSQL。

        参数:
            source_table: 源数据表全限定名，默认 ``public.job_description_parsed``。

        返回:
            list[RecordResult]: 所有记录的技能抽取结果。

        异常:
            RuntimeError: 数据库连接失败时抛出。
        """
        from config.paths import get_project_paths

        paths = get_project_paths()
        pg_params = paths.pg_connection_params

        logger.info("从 PostgreSQL 读取数据: %s", source_table)
        logger.info(
            "数据库: %s@%s:%s/%s",
            pg_params.get("user", ""),
            pg_params.get("host", ""),
            pg_params.get("port", ""),
            pg_params.get("dbname", ""),
        )

        records = self._fetch_records(pg_params, source_table)
        logger.info("已读取 %d 条记录", len(records))

        results = self.process_records(records)

        # 统计
        hard_count = sum(1 for r in results if r.hard_skill_count > 0)
        soft_count = sum(1 for r in results if r.soft_skill_count > 0)
        logger.info(
            "管线完成: %d 条记录, %d 条命中硬技能 (%.1f%%), %d 条命中软技能 (%.1f%%)",
            len(results),
            hard_count,
            hard_count / max(len(results), 1) * 100,
            soft_count,
            soft_count / max(len(results), 1) * 100,
        )

        # 写入 PostgreSQL（如果配置了 writer）
        if self._writer is not None and results:
            result_dicts = [r.to_dict() for r in results]
            written = self._writer(result_dicts)
            logger.info("结果已写入 PostgreSQL: %d 条", written)

        return results

    @staticmethod
    def _fetch_records(
        pg_params: Dict[str, Any],
        source_table: str,
    ) -> List[Dict[str, Any]]:
        """从 PostgreSQL 读取岗位记录。

        参数:
            pg_params: PostgreSQL 连接参数字典。
            source_table: 源表全限定名。

        返回:
            list[dict]: 岗位记录列表。
        """
        import psycopg2

        conn = psycopg2.connect(**pg_params)
        try:
            with conn.cursor() as cur:
                # 优先尝试包含来源追溯字段的查询；若源表无这些列则回退到基础查询。
                try:
                    cur.execute(
                        f"""
                        SELECT
                            recruitment_record_id,
                            job_title,
                            requirements_text,
                            duties_text,
                            job_description_clean,
                            source_table,
                            source_row_number
                        FROM {source_table}
                    """
                    )
                except Exception:
                    conn.rollback()
                    cur.execute(
                        f"""
                        SELECT
                            recruitment_record_id,
                            job_title,
                            requirements_text,
                            duties_text,
                            job_description_clean
                        FROM {source_table}
                    """
                    )
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        finally:
            conn.close()

        return [dict(zip(columns, row)) for row in rows]


# ─── 便捷构造函数 ───────────────────────────────────────────────────────────


def create_v3_pipeline(
    dict_path: Optional[Path] = None,
    use_llm: bool = False,
) -> V3Pipeline:
    """创建 V3Pipeline 实例的便捷工厂函数。

    自动加载平面化硬技能词典和软技能词典，初始化匹配器。
    可选初始化 LLM 客户端用于软技能二次验证。

    参数:
        dict_path: 平面化技能词典路径，为 None 时使用默认路径。
        use_llm: 是否初始化 LLM 客户端。设为 True 时需确保 vLLM 服务可用。

    返回:
        V3Pipeline: 初始化好的管线实例。
    """
    from config.paths import get_project_paths

    paths = get_project_paths()
    project_root = paths.project_root

    # 硬技能匹配器
    from .hard_skill_matcher import FlatHardSkillMatcher, load_flat_dictionary

    hard_dict_path = dict_path or (
        project_root / "dicts" / "flat_skill_dictionary.json"
    )
    hard_dict = load_flat_dictionary(str(hard_dict_path))
    hard_matcher = FlatHardSkillMatcher(hard_dict)

    # 软技能匹配器
    from .soft_skill_matcher import SoftSkillMatcher

    soft_matcher = SoftSkillMatcher()

    # LLM 客户端（可选）
    llm_client = None
    if use_llm:
        from src.model_platform.llm import create_llm_client

        llm_client = create_llm_client()
        logger.info("已初始化 LLM 客户端用于软技能验证")
    else:
        logger.info("未启用 LLM 验证，软技能仅使用词典匹配")

    return V3Pipeline(
        hard_skill_matcher=hard_matcher,
        soft_skill_matcher=soft_matcher,
        llm_client=llm_client,
    )


# ─── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="V3 统一技能抽取管线：硬技能 + 软技能双管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
用法示例:
  # 从 PostgreSQL 读取数据运行
  python -m src.skill_extraction.v3_pipeline run

  # 指定源表
  python -m src.skill_extraction.v3_pipeline run --source-table public.job_description_parsed

  # 启用 LLM 软技能验证
  python -m src.skill_extraction.v3_pipeline run --use-llm

  # 输出结果到 JSON 文件
  python -m src.skill_extraction.v3_pipeline run --output results.json
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run 子命令 ──
    run_cmd = subparsers.add_parser(
        "run",
        help="从 PostgreSQL 读取数据并运行双管线",
    )
    run_cmd.add_argument(
        "--source-table",
        default="public.job_description_parsed",
        help="源数据表全限定名 (默认: public.job_description_parsed)",
    )
    run_cmd.add_argument(
        "--dict-path",
        default=None,
        help="平面化技能词典路径 (默认: dicts/flat_skill_dictionary.json)",
    )
    run_cmd.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="启用 LLM 软技能二次验证 (需要 vLLM 服务可用)",
    )
    run_cmd.add_argument(
        "--output",
        default=None,
        help="输出 JSON 文件路径 (默认: 输出到日志)",
    )

    # ── process 子命令（从文件读取）──
    process_cmd = subparsers.add_parser(
        "process",
        help="从 JSON 文件读取数据并运行双管线",
    )
    process_cmd.add_argument(
        "input_file",
        help="输入 JSON 文件路径，每行一条记录",
    )
    process_cmd.add_argument(
        "--dict-path",
        default=None,
        help="平面化技能词典路径",
    )
    process_cmd.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="启用 LLM 软技能二次验证",
    )
    process_cmd.add_argument(
        "--output",
        default=None,
        help="输出 JSON 文件路径",
    )

    return parser


def _save_results(results: List[RecordResult], output_path: str) -> None:
    """将结果保存为 JSON 文件。"""
    output_data = [r.to_dict() for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存: %s (%d 条)", output_path, len(output_data))


def main() -> None:
    """CLI 入口函数。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dict_path = Path(args.dict_path) if args.dict_path else None
    pipeline = create_v3_pipeline(dict_path=dict_path, use_llm=args.use_llm)

    if args.command == "run":
        results = pipeline.run(source_table=args.source_table)
    elif args.command == "process":
        with open(args.input_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        logger.info("从文件读取 %d 条记录: %s", len(records), args.input_file)
        results = pipeline.process_records(records)
    else:
        parser.print_help()
        return

    if args.output:
        _save_results(results, args.output)
    else:
        # 输出摘要到日志
        total_hard = sum(r.hard_skill_count for r in results)
        total_soft = sum(r.soft_skill_count for r in results)
        logger.info(
            "结果摘要: %d 条记录, 硬技能命中 %d 次, 软技能命中 %d 次",
            len(results),
            total_hard,
            total_soft,
        )


if __name__ == "__main__":
    main()
