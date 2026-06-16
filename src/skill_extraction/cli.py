"""技能抽取统一命令入口。

用法::

    python -m src.skill_extraction.cli pipeline run
    python -m src.skill_extraction.cli eval list
    python -m src.skill_extraction.cli hard match-pg
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def _add_pipeline_commands(subparsers: argparse._SubParsersAction) -> None:
    pipeline = subparsers.add_parser("pipeline", help="运行 V3 双管线")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command")

    run_cmd = pipeline_sub.add_parser("run", help="从 PostgreSQL 读取并运行")
    run_cmd.add_argument("--source-table", default=None)
    run_cmd.add_argument("--dict-path", default=None)
    run_cmd.add_argument("--use-llm", action="store_true", default=False)
    run_cmd.add_argument("--output", default=None)

    process_cmd = pipeline_sub.add_parser("process", help="从 JSON 文件处理")
    process_cmd.add_argument("input_file")
    process_cmd.add_argument("--dict-path", default=None)
    process_cmd.add_argument("--use-llm", action="store_true", default=False)
    process_cmd.add_argument("--output", default=None)


def _add_eval_commands(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser("eval", help="评估与版本对比")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")
    eval_sub.add_parser("run", help="运行评估并写入注册表")
    compare = eval_sub.add_parser("compare", help="对比两个评估版本")
    compare.add_argument("version_a")
    compare.add_argument("version_b")
    eval_sub.add_parser("list", help="列出评估记录")


def _add_hard_commands(subparsers: argparse._SubParsersAction) -> None:
    hard = subparsers.add_parser("hard", help="硬技能工具")
    hard_sub = hard.add_subparsers(dest="hard_command")
    match_pg = hard_sub.add_parser("match-pg", help="运行 PostgreSQL 硬技能匹配")
    match_pg.add_argument("--dict-path", default=None)
    match_pg.add_argument("--source-table", default=None)
    match_pg.add_argument("--output-table", default=None)
    match_pg.add_argument("--limit", type=int, default=None)
    match_pg.add_argument("--no-write", action="store_true", default=False)


def _add_soft_commands(subparsers: argparse._SubParsersAction) -> None:
    soft = subparsers.add_parser("soft", help="软技能工具")
    soft_sub = soft.add_subparsers(dest="soft_command")
    build = soft_sub.add_parser("build-dictionary", help="构建软技能词典")
    build.add_argument("--output", default=None)
    build.add_argument("--use-llm", action="store_true", default=False)


def _add_dict_commands(subparsers: argparse._SubParsersAction) -> None:
    dictionary = subparsers.add_parser("dict", help="硬技能词典迭代")
    dict_sub = dictionary.add_subparsers(dest="dict_command")
    iterate = dict_sub.add_parser("iterate", help="运行保守词典迭代")
    iterate.add_argument("--dictionary", default="dicts/flat_skill_dictionary.json")
    iterate.add_argument(
        "--dataset",
        default="output/skill_extraction/regression/flat_skill_regression_dataset.jsonl",
    )
    iterate.add_argument("--rules", default="config/skill_dictionary_iteration.json")


def _add_label_commands(subparsers: argparse._SubParsersAction) -> None:
    label = subparsers.add_parser("label", help="LLM 标注数据集生成")
    label_sub = label.add_subparsers(dest="label_command")
    label_sub.add_parser("regression", help="生成硬技能回归数据集")
    label_sub.add_parser("context", help="生成上下文分类数据集")


def build_parser() -> argparse.ArgumentParser:
    """构建统一 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="Employ26 技能抽取统一 CLI")
    subparsers = parser.add_subparsers(dest="command")
    _add_pipeline_commands(subparsers)
    _add_eval_commands(subparsers)
    _add_hard_commands(subparsers)
    _add_soft_commands(subparsers)
    _add_dict_commands(subparsers)
    _add_label_commands(subparsers)
    return parser


def _save_pipeline_results(results: Sequence, output_path: str) -> None:
    """保存管线结果 JSON。"""
    data = [item.to_dict() for item in results]
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("结果已保存: %s", output_path)


def _handle_pipeline(args: argparse.Namespace) -> None:
    from .pipeline import create_v3_pipeline

    dict_path = Path(args.dict_path) if getattr(args, "dict_path", None) else None
    pipeline = create_v3_pipeline(dict_path=dict_path, use_llm=args.use_llm)
    if args.pipeline_command == "run":
        results = pipeline.run(source_table=args.source_table)
    elif args.pipeline_command == "process":
        records = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
        results = pipeline.process_records(records)
    else:
        raise SystemExit("请指定 pipeline 子命令")
    if args.output:
        _save_pipeline_results(results, args.output)


def _handle_eval(args: argparse.Namespace) -> None:
    from .evaluation.cli import cmd_compare, cmd_list, cmd_run

    if args.eval_command == "run":
        cmd_run()
    elif args.eval_command == "compare":
        cmd_compare(args.version_a, args.version_b)
    elif args.eval_command == "list":
        cmd_list()
    else:
        raise SystemExit("请指定 eval 子命令")


def _handle_hard(args: argparse.Namespace) -> None:
    from .hard.pg_matcher import run_match_pg

    if args.hard_command != "match-pg":
        raise SystemExit("请指定 hard 子命令")
    run_match_pg(
        dict_path=args.dict_path,
        source_table=args.source_table,
        output_table=args.output_table,
        limit=args.limit,
        write=not args.no_write,
    )


def _handle_soft(args: argparse.Namespace) -> None:
    if args.soft_command != "build-dictionary":
        raise SystemExit("请指定 soft 子命令")
    from .soft.dictionary_builder import main as builder_main

    # 保留原 builder 的 argparse 行为，避免重复实现其 LLM 参数细节。
    builder_main()


def _handle_dict(args: argparse.Namespace) -> None:
    if args.dict_command != "iterate":
        raise SystemExit("请指定 dict 子命令")
    from .dictionary.iterate_flat import run_iteration

    run_iteration(
        dict_path=args.dictionary,
        dataset_path=args.dataset,
        rules_path=args.rules,
    )


def _handle_label(args: argparse.Namespace) -> None:
    if args.label_command == "regression":
        from .labeling.regression_dataset import main
    elif args.label_command == "context":
        from .labeling.context_dataset import main
    else:
        raise SystemExit("请指定 label 子命令")
    main()


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "pipeline":
        _handle_pipeline(args)
    elif args.command == "eval":
        _handle_eval(args)
    elif args.command == "hard":
        _handle_hard(args)
    elif args.command == "soft":
        _handle_soft(args)
    elif args.command == "dict":
        _handle_dict(args)
    elif args.command == "label":
        _handle_label(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
