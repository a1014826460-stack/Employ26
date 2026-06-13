"""V3 技能抽取管线评估脚本。

同时评估硬技能和软技能的抽取质量：
- 硬技能：复用 regression_eval.py 的 precision/recall/F1 逻辑，扩展分类准确率指标。
- 软技能：从标注数据中抽取有 soft_skill 标注的样本，计算覆盖率、分类准确率、精确率。

设计原则：
- PostgreSQL 不可用时，通过内存数据传入进行测试（evaluate 方法）。
- PostgreSQL 可用时，通过 run 方法从数据库读取数据。
- 报告输出到 output/skill_extraction/reports/v3_eval/。

用法::

    python -m src.skill_extraction.eval_v3 --fail-under-precision 0.85 --fail-under-f1 0.80
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config.paths import get_project_paths

logger = logging.getLogger(__name__)


# ─── 数据结构 ───────────────────────────────────────────────────────────────


@dataclass
class HardSkillSample:
    """硬技能评估样本。"""

    sample_id: str
    text: str
    gold_skills: List[str]
    gold_categories: Optional[Dict[str, str]] = None  # skill_name → category


@dataclass
class SoftSkillSample:
    """软技能评估样本。"""

    sample_id: str
    text: str
    gold_skills: List[Dict[str, str]]  # [{"name": ..., "dimension": ...}]


@dataclass
class HardSkillMetrics:
    """硬技能评估指标。"""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    category_accuracy: float = 0.0
    exact_match_rate: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sample_count: int = 0
    error_rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "category_accuracy": self.category_accuracy,
            "exact_match_rate": self.exact_match_rate,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "sample_count": self.sample_count,
        }


@dataclass
class SoftSkillMetrics:
    """软技能评估指标。"""

    coverage: float = 0.0
    precision: float = 0.0
    dimension_accuracy: float = 0.0
    predicted_count: int = 0
    gold_count: int = 0
    matched_count: int = 0
    sample_count: int = 0
    error_rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "coverage": self.coverage,
            "precision": self.precision,
            "dimension_accuracy": self.dimension_accuracy,
            "predicted_count": self.predicted_count,
            "gold_count": self.gold_count,
            "matched_count": self.matched_count,
            "sample_count": self.sample_count,
        }


@dataclass
class V3EvalReport:
    """V3 评估总报告。"""

    evaluated_at: str
    hard_skill_metrics: HardSkillMetrics
    soft_skill_metrics: SoftSkillMetrics
    dataset_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "evaluated_at": self.evaluated_at,
            "hard_skill_metrics": self.hard_skill_metrics.to_dict(),
            "soft_skill_metrics": self.soft_skill_metrics.to_dict(),
            "dataset_summary": self.dataset_summary,
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


def _normalize_skill_name(name: str) -> str:
    """归一化技能名称，用于匹配比较。"""
    return _safe_text(name).casefold()


def _parse_skill_list(value: Any) -> List[str]:
    """解析技能列表字段，兼容 JSON 数组、逗号分隔和管道分隔格式。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)]

    text = _safe_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [_safe_text(item) for item in parsed if _safe_text(item)]

    if "|" in text:
        return [_safe_text(item) for item in text.split("|") if _safe_text(item)]
    if "," in text:
        return [_safe_text(item) for item in text.split(",") if _safe_text(item)]
    return [text]


def _compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """计算精确率、召回率和 F1 值。"""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _load_jsonl(path: Path) -> List[Dict]:
    """加载 JSONL 文件。"""
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_hard_skill_dataset(path: Path) -> List[HardSkillSample]:
    """加载硬技能评估数据集。

    支持 JSONL 和 CSV 格式。数据集需包含以下字段：
    - sample_id: 样本 ID
    - text: 岗位描述文本
    - gold_skills: 标注的技能列表
    - gold_categories（可选）: 标注的分类信息

    参数:
        path: 数据集文件路径。

    返回:
        list[HardSkillSample]: 评估样本列表。
    """
    if not path.exists():
        raise FileNotFoundError(f"硬技能评估数据集不存在: {path}")

    if path.suffix.lower() == ".jsonl":
        raw_rows = _load_jsonl(path)
    elif path.suffix.lower() == ".csv":
        raw_rows = pd.read_csv(path, encoding="utf-8").to_dict(orient="records")
    else:
        raise ValueError("仅支持 .jsonl 或 .csv 格式的数据集")

    samples: List[HardSkillSample] = []
    for index, row in enumerate(raw_rows):
        sample_id = _safe_text(row.get("sample_id", "")) or f"sample_{index:06d}"
        text = _safe_text(row.get("text", ""))
        if not text:
            continue

        gold_skills = _parse_skill_list(row.get("gold_skills", []))

        # 解析分类信息（可选）
        gold_categories = None
        if "gold_categories" in row:
            raw_cats = row["gold_categories"]
            if isinstance(raw_cats, dict):
                gold_categories = {
                    _normalize_skill_name(k): _safe_text(v) for k, v in raw_cats.items()
                }
            elif isinstance(raw_cats, str):
                try:
                    parsed = json.loads(raw_cats)
                    if isinstance(parsed, dict):
                        gold_categories = {
                            _normalize_skill_name(k): _safe_text(v)
                            for k, v in parsed.items()
                        }
                except json.JSONDecodeError:
                    pass

        samples.append(
            HardSkillSample(
                sample_id=sample_id,
                text=text,
                gold_skills=gold_skills,
                gold_categories=gold_categories,
            )
        )

    return samples


def _load_soft_skill_dataset(path: Path) -> List[SoftSkillSample]:
    """加载软技能评估数据集。

    支持 JSONL 和 CSV 格式。数据集需包含以下字段：
    - sample_id: 样本 ID
    - text: 岗位描述文本
    - gold_soft_skills: 标注的软技能列表 [{"name": ..., "dimension": ...}]

    参数:
        path: 数据集文件路径。

    返回:
        list[SoftSkillSample]: 评估样本列表。
    """
    if not path.exists():
        raise FileNotFoundError(f"软技能评估数据集不存在: {path}")

    if path.suffix.lower() == ".jsonl":
        raw_rows = _load_jsonl(path)
    elif path.suffix.lower() == ".csv":
        raw_rows = pd.read_csv(path, encoding="utf-8").to_dict(orient="records")
    else:
        raise ValueError("仅支持 .jsonl 或 .csv 格式的数据集")

    samples: List[SoftSkillSample] = []
    for index, row in enumerate(raw_rows):
        sample_id = _safe_text(row.get("sample_id", "")) or f"sample_{index:06d}"
        text = _safe_text(row.get("text", ""))
        if not text:
            continue

        # 解析 gold_soft_skills
        raw_gold = row.get("gold_soft_skills", [])
        if isinstance(raw_gold, str):
            try:
                raw_gold = json.loads(raw_gold)
            except json.JSONDecodeError:
                raw_gold = []

        gold_skills = []
        if isinstance(raw_gold, list):
            for item in raw_gold:
                if isinstance(item, dict):
                    name = _safe_text(item.get("name", ""))
                    dimension = _safe_text(item.get("dimension", ""))
                    if name:
                        gold_skills.append({"name": name, "dimension": dimension})

        samples.append(
            SoftSkillSample(
                sample_id=sample_id,
                text=text,
                gold_skills=gold_skills,
            )
        )

    return samples


# ─── 评估逻辑 ───────────────────────────────────────────────────────────────


def evaluate_hard_skills(
    samples: Sequence[HardSkillSample],
    hard_matcher: Any,
) -> HardSkillMetrics:
    """评估硬技能匹配效果。

    参数:
        samples: 硬技能评估样本列表。
        hard_matcher: 硬技能匹配器实例，需实现 match_text(text) 方法。

    返回:
        HardSkillMetrics: 评估指标。
    """
    if not samples:
        return HardSkillMetrics(sample_count=0)

    tp = 0
    fp = 0
    fn = 0
    exact_match_count = 0
    category_correct = 0
    category_total = 0
    error_rows: List[Dict[str, Any]] = []

    for sample in samples:
        predicted = hard_matcher.match_text(sample.text)
        predicted_names = {
            _normalize_skill_name(item["skill_name"]) for item in predicted
        }
        gold_names = {_normalize_skill_name(item) for item in sample.gold_skills}

        true_positive = predicted_names & gold_names
        false_positive = predicted_names - gold_names
        false_negative = gold_names - predicted_names

        tp += len(true_positive)
        fp += len(false_positive)
        fn += len(false_negative)

        if not false_positive and not false_negative:
            exact_match_count += 1

        # 分类准确率
        if sample.gold_categories:
            for item in predicted:
                norm_name = _normalize_skill_name(item["skill_name"])
                if norm_name in true_positive and norm_name in sample.gold_categories:
                    category_total += 1
                    predicted_cat = _safe_text(item.get("category", ""))
                    gold_cat = sample.gold_categories[norm_name]
                    if _normalize_skill_name(predicted_cat) == _normalize_skill_name(
                        gold_cat
                    ):
                        category_correct += 1

        # 记录误差行
        if false_positive or false_negative:
            error_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "text": sample.text,
                    "predicted_skills": json.dumps(
                        sorted([item["skill_name"] for item in predicted]),
                        ensure_ascii=False,
                    ),
                    "gold_skills": json.dumps(
                        sorted(sample.gold_skills), ensure_ascii=False
                    ),
                    "false_positives": json.dumps(
                        sorted(
                            [
                                item["skill_name"]
                                for item in predicted
                                if _normalize_skill_name(item["skill_name"])
                                in false_positive
                            ]
                        ),
                        ensure_ascii=False,
                    ),
                    "false_negatives": json.dumps(
                        sorted(
                            [
                                s
                                for s in sample.gold_skills
                                if _normalize_skill_name(s) in false_negative
                            ]
                        ),
                        ensure_ascii=False,
                    ),
                }
            )

    metrics_dict = _compute_precision_recall_f1(tp, fp, fn)

    return HardSkillMetrics(
        precision=metrics_dict["precision"],
        recall=metrics_dict["recall"],
        f1=metrics_dict["f1"],
        category_accuracy=category_correct / max(category_total, 1),
        exact_match_rate=exact_match_count / max(len(samples), 1),
        tp=tp,
        fp=fp,
        fn=fn,
        sample_count=len(samples),
        error_rows=error_rows,
    )


def evaluate_soft_skills(
    samples: Sequence[SoftSkillSample],
    soft_matcher: Any,
    llm_client: Any = None,
    llm_extract: bool = False,
) -> SoftSkillMetrics:
    """评估软技能匹配效果。

    参数:
        samples: 软技能评估样本列表。
        soft_matcher: 软技能匹配器实例，需实现 match_text(text) 方法。
        llm_client: LLM 客户端，用于验证或直接抽取。
        llm_extract: True 时使用 LLM 直接抽取代替词典匹配。
        llm_client: LLM 客户端，用于软技能二次验证。为 None 时跳过 LLM 验证。

    返回:
        SoftSkillMetrics: 评估指标。
    """
    if not samples:
        return SoftSkillMetrics(sample_count=0)

    total_predicted = 0
    total_gold = 0
    total_matched = 0
    dimension_correct = 0
    dimension_total = 0
    error_rows: List[Dict[str, Any]] = []

    for sample in samples:
        # 运行软技能匹配
        if llm_extract and llm_client is not None:
            from .soft_skill_llm_extractor import extract_soft_skills

            predicted = extract_soft_skills(
                text=sample.text,
                llm_client=llm_client,
            )
        else:
            predicted = soft_matcher.match_text(sample.text)

            # 可选的 LLM 验证
            if llm_client is not None and predicted:
                from .soft_skill_llm_validator import validate_soft_skills

                predicted = validate_soft_skills(
                    candidates=predicted,
                    context_text=sample.text,
                    llm_client=llm_client,
                )

        gold_skills = sample.gold_skills
        gold_names = {_normalize_skill_name(g["name"]) for g in gold_skills}
        predicted_names = {_normalize_skill_name(p["name"]) for p in predicted}

        # 覆盖率：gold 中有多少被 predicted 命中
        # 模糊匹配模式：若 pred 名称是 gold 名称的子串（或反之），视为命中
        matched_names = set()
        pred_to_gold = {}  # pred_name -> matched_gold_name
        for gn in gold_names:
            for pn in predicted_names:
                if gn == pn or gn in pn or pn in gn:
                    matched_names.add(gn)
                    pred_to_gold[pn] = gn
                    break
        total_matched += len(matched_names)
        total_gold += len(gold_names)
        total_predicted += len(predicted_names)

        # 维度准确率（使用 pred→gold 模糊映射）
        for pred_item in predicted:
            norm_pred_name = _normalize_skill_name(pred_item["name"])
            matched_gold = pred_to_gold.get(norm_pred_name)
            if matched_gold:
                for gold_item in gold_skills:
                    if _normalize_skill_name(gold_item["name"]) == matched_gold:
                        dimension_total += 1
                        pred_dim = _safe_text(pred_item.get("dimension", ""))
                        gold_dim = _safe_text(gold_item.get("dimension", ""))
                        if _normalize_skill_name(pred_dim) == _normalize_skill_name(
                            gold_dim
                        ):
                            dimension_correct += 1
                        break

        # 误差行：gold中未模糊匹配到的 = missing，pred中未命中任何gold的 = extra
        missing = gold_names - matched_names
        extra = set()
        for pn in predicted_names:
            hit = False
            for gn in gold_names:
                if gn == pn or gn in pn or pn in gn:
                    hit = True
                    break
            if not hit:
                extra.add(pn)
        if missing or extra:
            error_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "text": sample.text,
                    "predicted_skills": json.dumps(
                        sorted([p["name"] for p in predicted]),
                        ensure_ascii=False,
                    ),
                    "gold_skills": json.dumps(
                        sorted([g["name"] for g in gold_skills]),
                        ensure_ascii=False,
                    ),
                    "missing_skills": json.dumps(
                        sorted(
                            [
                                s
                                for s in gold_skills
                                if _normalize_skill_name(s["name"]) in missing
                            ],
                            key=lambda x: x["name"],
                        ),
                        ensure_ascii=False,
                    ),
                    "extra_skills": json.dumps(
                        sorted(
                            [
                                p["name"]
                                for p in predicted
                                if _normalize_skill_name(p["name"]) in extra
                            ]
                        ),
                        ensure_ascii=False,
                    ),
                }
            )

    return SoftSkillMetrics(
        coverage=total_matched / max(total_gold, 1),
        precision=total_matched / max(total_predicted, 1),
        dimension_accuracy=dimension_correct / max(dimension_total, 1),
        predicted_count=total_predicted,
        gold_count=total_gold,
        matched_count=total_matched,
        sample_count=len(samples),
        error_rows=error_rows,
    )


# ─── 主评估入口 ─────────────────────────────────────────────────────────────


def evaluate(
    hard_samples: Sequence[HardSkillSample],
    soft_samples: Sequence[SoftSkillSample],
    hard_matcher: Any,
    soft_matcher: Any,
    llm_client: Any = None,
    llm_extract: bool = False,
    output_dir: Optional[Path] = None,
) -> V3EvalReport:
    """执行 V3 管线的完整评估。

    参数:
        hard_samples: 硬技能评估样本。
        soft_samples: 软技能评估样本。
        hard_matcher: 硬技能匹配器实例。
        soft_matcher: 软技能匹配器实例。
        llm_client: LLM 客户端，用于软技能二次验证。为 None 时跳过。
        output_dir: 报告输出目录，为 None 时使用默认路径。

    返回:
        V3EvalReport: 评估报告。
    """
    logger.info(
        "开始 V3 评估: 硬技能 %d 样本, 软技能 %d 样本",
        len(hard_samples),
        len(soft_samples),
    )

    # 硬技能评估
    hard_metrics = evaluate_hard_skills(hard_samples, hard_matcher)
    logger.info(
        "硬技能评估: P=%.4f R=%.4f F1=%.4f 分类准确率=%.4f",
        hard_metrics.precision,
        hard_metrics.recall,
        hard_metrics.f1,
        hard_metrics.category_accuracy,
    )

    # 软技能评估
    soft_metrics = evaluate_soft_skills(
        soft_samples, soft_matcher, llm_client, llm_extract=llm_extract
    )
    logger.info(
        "软技能评估: 覆盖率=%.4f 精确率=%.4f 维度准确率=%.4f",
        soft_metrics.coverage,
        soft_metrics.precision,
        soft_metrics.dimension_accuracy,
    )

    report = V3EvalReport(
        evaluated_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        hard_skill_metrics=hard_metrics,
        soft_skill_metrics=soft_metrics,
        dataset_summary={
            "hard_skill_sample_count": len(hard_samples),
            "soft_skill_sample_count": len(soft_samples),
        },
    )

    # 输出报告
    if output_dir is None:
        paths = get_project_paths()
        output_dir = paths.report_dir / "v3_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 汇总报告
    summary_path = output_dir / f"v3_eval_summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("评估汇总: %s", summary_path)

    # 硬技能误差明细
    if hard_metrics.error_rows:
        hard_error_path = output_dir / f"hard_skill_errors_{timestamp}.csv"
        pd.DataFrame(hard_metrics.error_rows).to_csv(
            hard_error_path,
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("硬技能误差明细: %s", hard_error_path)

    # 软技能误差明细
    if soft_metrics.error_rows:
        soft_error_path = output_dir / f"soft_skill_errors_{timestamp}.csv"
        pd.DataFrame(soft_metrics.error_rows).to_csv(
            soft_error_path,
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("软技能误差明细: %s", soft_error_path)

    return report


def run(
    hard_dataset_path: Optional[str] = None,
    soft_dataset_path: Optional[str] = None,
    dict_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_llm: bool = False,
) -> V3EvalReport:
    """从数据文件加载评估数据并运行评估。

    参数:
        hard_dataset_path: 硬技能数据集路径，为 None 时使用默认路径。
        soft_dataset_path: 软技能数据集路径，为 None 时使用默认路径。
        dict_path: 硬技能词典路径，为 None 时使用默认路径。
        output_dir: 报告输出目录。
        use_llm: 是否使用 LLM 进行软技能验证。

    返回:
        V3EvalReport: 评估报告。
    """
    from config.paths import get_project_paths

    paths = get_project_paths()
    project_root = paths.project_root

    # 默认路径
    if hard_dataset_path is None:
        hard_dataset_path = str(
            project_root
            / "output"
            / "skill_extraction"
            / "regression"
            / "flat_skill_regression_dataset.jsonl"
        )
    if soft_dataset_path is None:
        soft_dataset_path = str(
            project_root
            / "output"
            / "skill_extraction"
            / "soft_skill_eval_dataset.jsonl"
        )
    if dict_path is None:
        dict_path = str(project_root / "dicts" / "flat_skill_dictionary.json")

    # 加载硬技能数据集
    hard_samples = _load_hard_skill_dataset(Path(hard_dataset_path))
    logger.info("加载硬技能评估数据: %d 样本", len(hard_samples))

    # 加载软技能数据集
    soft_samples = _load_soft_skill_dataset(Path(soft_dataset_path))
    logger.info("加载软技能评估数据: %d 样本", len(soft_samples))

    # 初始化匹配器
    from .hard_skill_matcher import FlatHardSkillMatcher, load_flat_dictionary

    hard_dict = load_flat_dictionary(dict_path)
    hard_matcher = FlatHardSkillMatcher(hard_dict)

    from .soft_skill_matcher import SoftSkillMatcher

    soft_matcher = SoftSkillMatcher()

    # LLM 客户端（可选）
    llm_client = None
    if use_llm:
        from src.model_platform.llm import create_llm_client

        llm_client = create_llm_client()
        logger.info("已初始化 LLM 客户端用于软技能验证")

    # 输出目录
    out_dir = Path(output_dir) if output_dir else None

    return evaluate(
        hard_samples=hard_samples,
        soft_samples=soft_samples,
        hard_matcher=hard_matcher,
        soft_matcher=soft_matcher,
        llm_client=llm_client,
        output_dir=out_dir,
    )


# ─── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="V3 技能抽取管线评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hard-dataset",
        default=None,
        help="硬技能评估数据集路径 (JSONL/CSV)",
    )
    parser.add_argument(
        "--soft-dataset",
        default=None,
        help="软技能评估数据集路径 (JSONL/CSV)",
    )
    parser.add_argument(
        "--dictionary",
        default=None,
        help="硬技能词典路径 (默认: dicts/flat_skill_dictionary.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="报告输出目录 (默认: output/skill_extraction/reports/v3_eval)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="启用 LLM 软技能二次验证",
    )
    parser.add_argument(
        "--fail-under-precision",
        type=float,
        default=None,
        help="硬技能精确率低于该阈值时返回非零退出码",
    )
    parser.add_argument(
        "--fail-under-f1",
        type=float,
        default=None,
        help="硬技能 F1 低于该阈值时返回非零退出码",
    )
    return parser


def main() -> None:
    """CLI 入口函数。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    report = run(
        hard_dataset_path=args.hard_dataset,
        soft_dataset_path=args.soft_dataset,
        dict_path=args.dictionary,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
    )

    # 检查阈值
    hard_metrics = report.hard_skill_metrics

    if args.fail_under_precision is not None:
        if hard_metrics.precision < args.fail_under_precision:
            logger.error(
                "硬技能精确率 %.4f 低于阈值 %.4f",
                hard_metrics.precision,
                args.fail_under_precision,
            )
            raise SystemExit(1)

    if args.fail_under_f1 is not None:
        if hard_metrics.f1 < args.fail_under_f1:
            logger.error(
                "硬技能 F1 %.4f 低于阈值 %.4f",
                hard_metrics.f1,
                args.fail_under_f1,
            )
            raise SystemExit(1)

    logger.info("V3 评估完成")


if __name__ == "__main__":
    main()
