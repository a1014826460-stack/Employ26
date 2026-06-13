"""软技能评估 CLI — 统一入口。

用法::

    python -m src.skill_extraction.eval_cli run
    python -m src.skill_extraction.eval_cli compare v1 v2
    python -m src.skill_extraction.eval_cli list
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_eval_dir() -> Path:
    """获取评估输出目录。"""
    from config.paths import get_project_paths

    return get_project_paths().project_root / "output" / "skill_extraction" / "eval"


def cmd_list(eval_dir: Optional[Path] = None) -> None:
    """列出所有评估记录。

    参数:
        eval_dir: 评估输出目录，为 None 时使用默认路径。
    """
    from ._eval_registry import load_registry

    registry_dir = eval_dir or _get_eval_dir()
    registry = load_registry(registry_dir)
    evaluations = registry.get("evaluations", [])

    if not evaluations:
        print("no eval records")
        return

    print(
        f"{'version':<8} {'evaluated_at':<22} {'soft_cov':<10} {'soft_prec':<10} {'hard_f1':<10}"
    )
    print("-" * 60)
    for r in evaluations:
        soft = r.get("soft_skill_metrics", {})
        hard = r.get("hard_skill_metrics", {})
        print(
            f"{r.get('dict_version', '?'):<8} "
            f"{r.get('evaluated_at', '?')[:19]:<22} "
            f"{soft.get('coverage', 0):.4f}   "
            f"{soft.get('precision', 0):.4f}   "
            f"{hard.get('f1', 0):.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="soft skill eval CLI",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="run eval and write registry")
    run_parser.add_argument(
        "--use-llm", action="store_true", help="enable LLM soft skill validation"
    )
    run_parser.add_argument(
        "--soft-dataset",
        default=None,
        help="soft skill gold dataset path (default: auto)",
    )
    run_parser.add_argument(
        "--hard-dataset",
        default=None,
        help="hard skill gold dataset path (default: auto)",
    )
    cmp_parser = sub.add_parser("compare", help="compare two versions")
    cmp_parser.add_argument("version_a")
    cmp_parser.add_argument("version_b")
    sub.add_parser("list", help="list all eval records")

    return parser


def cmd_run(
    eval_dir: Optional[Path] = None,
    hard_dataset: Optional[Path] = None,
    soft_dataset: Optional[Path] = None,
    use_llm: bool = False,
) -> None:
    """运行评估并将结果写入注册表。

    参数:
        eval_dir: 评估输出目录。
        hard_dataset: 硬技能 gold dataset 路径。
        soft_dataset: 软技能 gold dataset 路径。
        use_llm: 是否启用 LLM 软技能验证。
    """
    from ._dict_paths import get_current_soft_skill_dict_path
    from ._eval_registry import append_eval_record
    from .eval_v3 import (
        _load_hard_skill_dataset,
        _load_soft_skill_dataset,
        evaluate,
    )
    from .hard_skill_matcher import FlatHardSkillMatcher, load_flat_dictionary
    from .soft_skill_matcher import SoftSkillMatcher

    project_root = Path(__file__).resolve().parents[2]
    registry_dir = eval_dir or _get_eval_dir()

    dict_path = get_current_soft_skill_dict_path()
    version = dict_path.stem
    logger.info("current soft skill dict version: %s", version)

    hard_path = hard_dataset or (registry_dir / "hard_skill_eval_dataset.jsonl")
    soft_path = soft_dataset or (registry_dir / "soft_skill_gold_dataset.jsonl")
    hard_samples = _load_hard_skill_dataset(hard_path)
    soft_samples = _load_soft_skill_dataset(soft_path)
    logger.info("loaded data: hard=%d, soft=%d", len(hard_samples), len(soft_samples))

    hard_dict_path = project_root / "dicts" / "flat_skill_dictionary.json"
    hard_dict = load_flat_dictionary(str(hard_dict_path))
    hard_matcher = FlatHardSkillMatcher(hard_dict)
    soft_matcher = SoftSkillMatcher()

    llm_client = None
    if use_llm:
        from src.model_platform.llm import create_llm_client

        llm_client = create_llm_client(backend="external_api")
        logger.info("已启用 LLM 软技能验证")

    version_report_dir = registry_dir / version
    report = evaluate(
        hard_samples=hard_samples,
        soft_samples=soft_samples,
        hard_matcher=hard_matcher,
        soft_matcher=soft_matcher,
        llm_client=llm_client,
        llm_extract=use_llm,
        output_dir=version_report_dir,
    )

    record = {
        "dict_version": version,
        "evaluated_at": report.evaluated_at,
        "soft_skill_metrics": report.soft_skill_metrics.to_dict(),
        "hard_skill_metrics": report.hard_skill_metrics.to_dict(),
        "gold_source": "annotations.label_studio_tasks_v2",
        "sample_count": max(
            report.dataset_summary.get("hard_skill_sample_count", 0),
            report.dataset_summary.get("soft_skill_sample_count", 0),
        ),
    }
    import json

    (version_report_dir / "summary.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    append_eval_record(registry_dir, record)

    _update_latest_link(registry_dir, version)

    soft = report.soft_skill_metrics
    hard = report.hard_skill_metrics
    print(f"\n=== eval done (dict version: {version}) ===")
    print(
        f"soft — coverage: {soft.coverage:.4f}  precision: {soft.precision:.4f}  dim_acc: {soft.dimension_accuracy:.4f}"
    )
    print(
        f"hard — precision: {hard.precision:.4f}  recall: {hard.recall:.4f}  f1: {hard.f1:.4f}"
    )
    print(f"report dir: {version_report_dir}")


def cmd_compare(
    version_a: str,
    version_b: str,
    eval_dir: Optional[Path] = None,
) -> None:
    """Compare eval metrics between two dict versions.

    参数:
        version_a: Baseline version.
        version_b: Comparison version.
        eval_dir: Eval output directory.
    """
    from ._eval_registry import get_record_by_version

    registry_dir = eval_dir or _get_eval_dir()

    record_a = get_record_by_version(registry_dir, version_a)
    record_b = get_record_by_version(registry_dir, version_b)

    if not record_a:
        print(f"version {version_a} not found in registry")
        return
    if not record_b:
        print(f"version {version_b} not found in registry")
        return

    soft_a = record_a.get("soft_skill_metrics", {})
    soft_b = record_b.get("soft_skill_metrics", {})
    hard_a = record_a.get("hard_skill_metrics", {})
    hard_b = record_b.get("hard_skill_metrics", {})

    def _delta(a: float, b: float) -> str:
        diff = b - a
        arrow = "+" if diff > 0 else ("-" if diff < 0 else "=")
        return f"{arrow}{abs(diff):.4f}"

    def _pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    metrics = [
        ("soft-coverage", "coverage", True),
        ("soft-precision", "precision", True),
        ("soft-dim_acc", "dimension_accuracy", True),
        ("hard-precision", "precision", False),
        ("hard-recall", "recall", False),
        ("hard-f1", "f1", False),
        ("hard-cat_acc", "category_accuracy", False),
    ]

    print(f"\n{'metric':<18} {version_a:<10} {version_b:<10} delta")
    print("-" * 55)
    for label, key, is_soft in metrics:
        a_val = (soft_a if is_soft else hard_a).get(key, 0)
        b_val = (soft_b if is_soft else hard_b).get(key, 0)
        d = _delta(a_val, b_val)
        print(f"{label:<18} {_pct(a_val):<10} {_pct(b_val):<10} {d}")


def _update_latest_link(eval_dir: Path, version: str) -> None:
    """更新 latest 链接指向最新版本。

    参数:
        eval_dir: 评估输出目录。
        version: 版本号字符串。
    """
    latest_link = eval_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version, target_is_directory=True)
    except (OSError, AttributeError):
        (eval_dir / "latest.txt").write_text(version, encoding="utf-8")


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "run":
        cmd_run(
            use_llm=getattr(args, "use_llm", False),
            soft_dataset=(
                Path(getattr(args, "soft_dataset", None))
                if getattr(args, "soft_dataset", None)
                else None
            ),
            hard_dataset=(
                Path(getattr(args, "hard_dataset", None))
                if getattr(args, "hard_dataset", None)
                else None
            ),
        )
    elif args.command == "compare":
        cmd_compare(args.version_a, args.version_b)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
