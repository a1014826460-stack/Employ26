"""活跃分析链路的统一 CLI。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import Callable

from src.analysis.education_distribution_analysis import EducationDistributionAnalyzer
from src.analysis.generate_excel_summary import ExcelReportGenerator
from src.analysis.generate_standardized_tables import StandardizedTableGenerator
from src.analysis.industry_trend_analysis import IndustryTrendAnalyzer
from src.analysis.occupation_salary_analysis import OccupationSalaryAnalyzer
from src.analysis.analysis_common import collect_output_files
from src.analysis.requirement_text_analysis import (
    DEFAULT_EXTRACTOR_VERSION,
    DEFAULT_GROUP_SIZE,
    DEFAULT_MONTHLY_GROUP_SIZE,
    DEFAULT_TOP_N,
    AnalysisParams,
    analyze_requirement_texts,
    build_current_output_dir,
)
from src.analysis.structured_common import (
    build_structured_output_dir,
    set_write_legacy_csv_copies,
    write_run_manifest,
)
from src.analysis.structured_dimension_analysis import StructuredDimensionAnalyzer
from src.analysis.structured_pg_source import (
    build_structured_source_coverage,
    write_structured_source_coverage,
)


logger = logging.getLogger(__name__)


STRUCTURED_PARALLEL_STEPS: tuple[tuple[str, Callable[[Path | None, Path], None]], ...] = (
    (
        "occupation_salary_analysis",
        lambda base_dir, output_dir: OccupationSalaryAnalyzer(base_dir=base_dir, output_dir=output_dir).run(),
    ),
    (
        "education_distribution_analysis",
        lambda base_dir, output_dir: EducationDistributionAnalyzer(base_dir=base_dir, output_dir=output_dir).run(),
    ),
    (
        "industry_trend_analysis",
        lambda base_dir, output_dir: IndustryTrendAnalyzer(base_dir=base_dir, output_dir=output_dir).run(),
    ),
    (
        "structured_dimension_analysis",
        lambda base_dir, output_dir: StructuredDimensionAnalyzer(base_dir=base_dir, output_dir=output_dir).run(),
    ),
)


def _run_step(step_name: str, action: Callable[[], None]) -> None:
    """记录并运行一个分析步骤，保留原始异常方便排查。"""
    logger.info("运行分析步骤: %s", step_name)
    action()


def _run_structured_parallel_steps(
    *,
    base_dir: Path | None,
    output_dir: Path,
    max_workers: int,
) -> list[str]:
    """并发运行互不依赖的结构化统计步骤。"""
    resolved_workers = max(1, int(max_workers))
    step_items = list(STRUCTURED_PARALLEL_STEPS)
    if resolved_workers == 1:
        completed_steps: list[str] = []
        for step_name, step_action in step_items:
            _run_step(step_name, lambda action=step_action: action(base_dir, output_dir))
            completed_steps.append(step_name)
        return completed_steps

    completed_by_index: list[tuple[int, str]] = []
    logger.info("并发运行结构化统计步骤: workers=%s, steps=%s", resolved_workers, len(step_items))
    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        futures = {
            executor.submit(_run_step, step_name, lambda action=step_action: action(base_dir, output_dir)): (
                index,
                step_name,
            )
            for index, (step_name, step_action) in enumerate(step_items)
        }
        for future in as_completed(futures):
            step_index, step_name = futures[future]
            future.result()
            completed_by_index.append((step_index, step_name))
    return [step_name for _, step_name in sorted(completed_by_index)]


def run_structured(args: argparse.Namespace) -> None:
    """运行结构化统计链路。"""
    base_dir = Path(args.base_dir) if args.base_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else build_structured_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_steps: list[str] = []
    structured_workers = max(1, int(args.structured_workers))
    set_write_legacy_csv_copies(bool(args.with_legacy_copies))
    coverage = build_structured_source_coverage()
    write_structured_source_coverage(output_dir)
    logger.info(
        "结构化主输入覆盖率: normalized_rows=%s, matched_rows=%s, matched_share=%.4f",
        coverage["normalized_rows"],
        coverage["matched_rows"],
        coverage["matched_share"],
    )
    if float(coverage["matched_share"]) < 0.8:
        logger.warning(
            "职业匹配覆盖率偏低（matched_share=%.4f），职业维度相关报表可能不能代表总体数据。",
            coverage["matched_share"],
        )

    completed_steps.extend(
        _run_structured_parallel_steps(
            base_dir=base_dir,
            output_dir=output_dir,
            max_workers=structured_workers,
        )
    )

    if not bool(args.skip_standardized):
        _run_step(
            "generate_standardized_tables",
            lambda: StandardizedTableGenerator(base_dir=base_dir, output_dir=output_dir).generate_all(),
        )
        completed_steps.append("generate_standardized_tables")

    if bool(args.with_excel):
        _run_step(
            "generate_excel_summary",
            lambda: ExcelReportGenerator(base_dir=base_dir, output_dir=output_dir).create_summary_report(),
        )
        completed_steps.append("generate_excel_summary")

    write_run_manifest(
        output_dir,
        workflow="structured_analysis",
        steps=completed_steps,
        params={
            "with_excel": bool(args.with_excel),
            "skip_standardized": bool(args.skip_standardized),
            "structured_workers": structured_workers,
            "with_legacy_copies": bool(args.with_legacy_copies),
            "source": "postgres",
            "normalized_table": "public.recruitment_jobs_normalized",
            "occupation_match_table": "public.skill_extraction_requirement_matches",
        },
        input_files=[
            "postgres:public.recruitment_jobs_normalized",
            "postgres:public.skill_extraction_requirement_matches",
        ],
        output_files=collect_output_files(output_dir, extra_outputs=["run_manifest.json"]),
    )


def run_requirements(args: argparse.Namespace) -> None:
    """运行 requirement text 约束抽取统计链路。"""
    extractor_version = str(args.extractor_version).strip() or DEFAULT_EXTRACTOR_VERSION
    output_dir = Path(args.output_dir) if args.output_dir else build_current_output_dir()
    analyze_requirement_texts(
        output_dir=output_dir,
        params=AnalysisParams(
            top_n=int(args.top_n),
            min_group_size=int(args.min_group_size),
            min_monthly_group_size=int(args.min_monthly_group_size),
            extractor_version=extractor_version,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """构建统一分析 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="Employ26 活跃分析链路")
    subparsers = parser.add_subparsers(dest="command", required=True)

    structured = subparsers.add_parser("structured", help="结构化统计链路（兼容: 直接运行）")
    structured_subparsers = structured.add_subparsers(dest="structured_command")
    structured_run = structured_subparsers.add_parser("run", help="运行结构化统计链路")
    _add_structured_args(structured)
    _add_structured_args(structured_run)
    structured.set_defaults(func=run_structured)
    structured_run.set_defaults(func=run_structured)

    requirements = subparsers.add_parser("requirements", help="requirement text 统计链路（兼容: 直接运行）")
    requirements_subparsers = requirements.add_subparsers(dest="requirements_command")
    requirements_run = requirements_subparsers.add_parser("run", help="运行 requirement text 约束统计")
    _add_requirements_args(requirements)
    _add_requirements_args(requirements_run)
    requirements.set_defaults(func=run_requirements)
    requirements_run.set_defaults(func=run_requirements)

    return parser


def _add_structured_args(parser: argparse.ArgumentParser) -> None:
    """为结构化统计命令添加公共参数。"""
    parser.add_argument(
        "--with-excel",
        action="store_true",
        help="生成最终 Excel 汇总",
    )
    parser.add_argument(
        "--skip-standardized",
        action="store_true",
        help="跳过规范化 CSV 汇总表",
    )
    parser.add_argument(
        "--structured-workers",
        type=int,
        default=4,
        help="并发运行互不依赖结构化统计步骤的 worker 数；设为 1 可恢复串行执行",
    )
    parser.add_argument(
        "--with-legacy-copies",
        action="store_true",
        help="同时导出历史中文文件名 CSV 副本；默认只保留英文规范文件名",
    )
    parser.add_argument("--output-dir", default="", help="显式指定结构化统计输出目录")
    parser.add_argument("--base-dir", default="", help="显式指定项目根目录（兼容旧脚本）")


def _add_requirements_args(parser: argparse.ArgumentParser) -> None:
    """为 requirement text 命令添加公共参数。"""
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--min-monthly-group-size", type=int, default=DEFAULT_MONTHLY_GROUP_SIZE)
    parser.add_argument("--extractor-version", default=DEFAULT_EXTRACTOR_VERSION)
    parser.add_argument("--output-dir", default="", help="显式指定 requirement text 输出目录")


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
