"""Train and run a multiclass hard-skill context classifier."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import load_skill_extraction_config
from .context_labels import (
    DEFAULT_CONTEXT_THRESHOLD,
    ID_TO_LABEL,
    LABEL_TO_ID,
    VALID_HARD_SKILL_LABEL,
)
from ..utils.llm_labeling_utils import safe_text


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_DATASET_OUTPUT = "output/skill_extraction/context_classifier/context_dataset.jsonl"
DEFAULT_MODEL_OUTPUT = "output/skill_extraction/context_classifier/model"


def truncate_text(text: object, max_chars: int = 320) -> str:
    """截断文本，控制判别器输入长度。

    参数:
        text: 原始文本。
        max_chars: 保留的最大字符数。

    返回:
        str: 归一化并截断后的文本。
    """
    normalized = safe_text(text)
    return normalized[:max_chars]


@dataclass(frozen=True)
class SkillContextCandidate:
    """上下文判别阶段使用的候选技能对象。

    字段说明:
        text: 岗位原文或切分后的局部文本。
        skill_name: 词典中的标准技能名。
        matched_term: 文本中实际命中的词或 alias。
        term_role: 命中词来源，例如 `name` / `alias`。
        job_title: 岗位名称。
        sample_id: 样本编号。
    """

    text: str
    skill_name: str
    matched_term: str
    term_role: str = ""
    job_title: str = ""
    sample_id: str = ""


def build_classifier_text_pair(candidate: SkillContextCandidate) -> tuple[str, str]:
    """将候选技能编码为 BERT 句对输入。

    参数:
        candidate: 单个待判别的技能候选。

    返回:
        tuple[str, str]:
            - `text_a`: 结构化元信息，如技能名、命中词、岗位名；
            - `text_b`: 原始岗位上下文文本。
    """
    meta_parts = [
        f"skill:{safe_text(candidate.skill_name)}",
        f"matched:{safe_text(candidate.matched_term)}",
    ]
    if safe_text(candidate.job_title):
        meta_parts.append(f"job:{safe_text(candidate.job_title)}")
    if safe_text(candidate.term_role):
        meta_parts.append(f"source:{safe_text(candidate.term_role)}")
    text_a = " [SEP] ".join(meta_parts)
    text_b = truncate_text(safe_text(candidate.text), max_chars=320)
    return text_a, text_b


class SkillContextInferenceDataset(Dataset):
    """推理阶段使用的 Dataset 封装。

    该类只负责按索引暴露 tokenizer 编码后的张量。
    """

    def __init__(self, encoded_inputs: Dict[str, torch.Tensor]) -> None:
        self.encoded_inputs = encoded_inputs

    def __len__(self) -> int:
        return int(self.encoded_inputs["input_ids"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.encoded_inputs.items()}


class SkillContextTrainingDataset(Dataset):
    """训练阶段使用的 Dataset 封装。

    与推理数据集相比，该类额外携带监督标签 `labels`。
    """

    def __init__(self, encoded_inputs: Dict[str, torch.Tensor], labels: Sequence[int]) -> None:
        self.encoded_inputs = encoded_inputs
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encoded_inputs.items()}
        item["labels"] = self.labels[index]
        return item


class SkillContextClassifier:
    """基于 BERT 的多分类上下文判别器。

    该判别器接在词典召回之后，用于过滤：
    - 泛词误报
    - alias 错映射
    - 非技能误命中
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = DEFAULT_CONTEXT_THRESHOLD,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        candidates: Sequence[SkillContextCandidate],
        batch_size: int = 32,
    ) -> List[Dict]:
        """对候选技能批量做多分类预测。

        参数:
            candidates: 待判别的候选技能列表。
            batch_size: 推理批大小。

        返回:
            List[Dict]: 与输入候选一一对应的预测结果，包含标签、分数和保留标记。
        """
        if not candidates:
            return []

        text_a_list: List[str] = []
        text_b_list: List[str] = []
        for candidate in candidates:
            text_a, text_b = build_classifier_text_pair(candidate)
            text_a_list.append(text_a)
            text_b_list.append(text_b)

        encoded = self.tokenizer(
            text_a_list,
            text_b_list,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = SkillContextInferenceDataset(encoded)

        outputs: List[Dict] = []
        valid_label_index = LABEL_TO_ID[VALID_HARD_SKILL_LABEL]
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                batch = {
                    key: value[start : start + batch_size].to(self.device)
                    for key, value in encoded.items()
                }
                logits = self.model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                top_probs, top_indices = torch.max(probs, dim=-1)

                for row_index in range(probs.shape[0]):
                    top_label = ID_TO_LABEL[int(top_indices[row_index])]
                    valid_score = float(probs[row_index, valid_label_index])
                    label_scores = {
                        ID_TO_LABEL[class_index]: float(probs[row_index, class_index])
                        for class_index in range(probs.shape[1])
                    }
                    outputs.append(
                        {
                            "label": top_label,
                            "score": float(top_probs[row_index]),
                            "valid_score": valid_score,
                            "keep": top_label == VALID_HARD_SKILL_LABEL and valid_score >= self.threshold,
                            "scores": label_scores,
                        }
                    )
        return outputs


def _read_jsonl(path: Path) -> List[Dict]:
    """读取 UTF-8 JSONL 文件。

    参数:
        path: JSONL 文件路径。

    返回:
        List[Dict]: 逐行解析后的字典列表。
    """
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_context_dataset(path: str | Path) -> List[Dict]:
    """加载上下文判别训练集并校验标签合法性。

    参数:
        path: 训练集 JSONL 路径。

    返回:
        List[Dict]: 原始记录列表。
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Context dataset does not exist: {dataset_path}")
    if dataset_path.suffix.lower() != ".jsonl":
        raise ValueError("Only JSONL context datasets are currently supported.")

    records = _read_jsonl(dataset_path)
    for index, record in enumerate(records):
        label = safe_text(record.get("label", ""))
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unsupported label at row {index}: {label}")
    return records


def build_context_dataset_from_regression(
    regression_dataset_path: str | Path,
    dictionary_path: str | Path,
    output_path: str | Path = DEFAULT_DATASET_OUTPUT,
    negative_keep_ratio: float = 1.0,
    seed: int = 42,
) -> Dict:
    """从回归集构造弱监督版上下文训练集。

    参数:
        regression_dataset_path: 回归集路径。
        dictionary_path: 平面技能词典路径。
        output_path: 输出 JSONL 路径。
        negative_keep_ratio: 负样本保留比例。
        seed: 随机种子。

    返回:
        Dict: 构建摘要，包括记录数和标签分布。

    说明:
        该函数仅用于兼容性兜底，只会生成 `valid_hard_skill` 和 `not_skill`
        两类标签。若需要完整多分类训练集，应优先使用 `llm_label_context_dataset.py`。
    """
    from .hard_skill_matcher import FlatHardSkillMatcher, load_flat_dictionary
    from .regression_eval import load_regression_dataset

    dataset = load_regression_dataset(regression_dataset_path)
    matcher = FlatHardSkillMatcher(load_flat_dictionary(dictionary_path))
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    written = 0
    label_counter: Counter[str] = Counter()

    with open(output_file, "w", encoding="utf-8") as file_obj:
        for row in dataset:
            gold_keys = {safe_text(skill).casefold() for skill in row.gold_skills}
            candidates = matcher.match_candidates(row.text)
            for candidate in candidates:
                label = (
                    VALID_HARD_SKILL_LABEL
                    if candidate["skill_name"].casefold() in gold_keys
                    else "not_skill"
                )
                if label != VALID_HARD_SKILL_LABEL and rng.random() > float(negative_keep_ratio):
                    continue
                record = {
                    "sample_id": row.sample_id,
                    "text": row.text,
                    "job_title": "",
                    "skill_name": candidate["skill_name"],
                    "matched_term": candidate["matched_term"],
                    "term_role": candidate["term_role"],
                    "label": label,
                }
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                label_counter[label] += 1

    summary = {
        "dataset_path": str(output_file),
        "record_count": written,
        "label_distribution": dict(label_counter),
    }
    logger.info("Weak context dataset written to %s (%d rows)", output_file, written)
    return summary


def _encode_dataset(
    tokenizer,
    records: Sequence[Dict],
    max_length: int,
) -> tuple[Dict[str, torch.Tensor], List[int]]:
    """将原始训练记录编码为 Transformer 输入张量。

    参数:
        tokenizer: HuggingFace tokenizer。
        records: 原始样本记录。
        max_length: 最大 token 长度。

    返回:
        tuple[Dict[str, torch.Tensor], List[int]]:
            - 编码后的输入张量；
            - 与之对齐的标签 id 列表。
    """
    candidates = [
        SkillContextCandidate(
            text=safe_text(record.get("text", "")),
            skill_name=safe_text(record.get("skill_name", "")),
            matched_term=safe_text(record.get("matched_term", "")),
            term_role=safe_text(record.get("term_role", "")),
            job_title=safe_text(record.get("job_title", "")),
            sample_id=safe_text(record.get("sample_id", "")),
        )
        for record in records
    ]
    text_a_list: List[str] = []
    text_b_list: List[str] = []
    for candidate in candidates:
        text_a, text_b = build_classifier_text_pair(candidate)
        text_a_list.append(text_a)
        text_b_list.append(text_b)

    encoded = tokenizer(
        text_a_list,
        text_b_list,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = [LABEL_TO_ID[safe_text(record.get("label", "not_skill"))] for record in records]
    return encoded, labels


def train_context_classifier(
    dataset_path: str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT,
    base_model_path: str | Path | None = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> Dict:
    """训练多分类上下文判别器。

    参数:
        dataset_path: 训练集 JSONL 路径。
        output_dir: 模型输出目录。
        base_model_path: 基础 BERT 模型目录；为空时从配置读取。
        num_train_epochs: 训练轮数。
        per_device_train_batch_size: 训练批大小。
        per_device_eval_batch_size: 验证批大小。
        learning_rate: 学习率。
        max_length: 最大 token 长度。
        validation_ratio: 验证集占比。
        seed: 随机种子。

    返回:
        Dict: 训练摘要，包括训练集规模、验证集规模和标签分布。
    """
    config = load_skill_extraction_config()
    base_model_path = str(base_model_path or config.bert_model_path)
    records = load_context_dataset(dataset_path)
    if not records:
        raise ValueError("Context dataset is empty.")

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * (1 - float(validation_ratio))))
    train_records = shuffled[:split_index]
    eval_records = shuffled[split_index:] or shuffled[: max(1, min(128, len(shuffled)))]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_encoded, train_labels = _encode_dataset(tokenizer, train_records, max_length=max_length)
    eval_encoded, eval_labels = _encode_dataset(tokenizer, eval_records, max_length=max_length)
    train_dataset = SkillContextTrainingDataset(train_encoded, train_labels)
    eval_dataset = SkillContextTrainingDataset(eval_encoded, eval_labels)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,
        num_train_epochs=float(num_train_epochs),
        per_device_train_batch_size=int(per_device_train_batch_size),
        per_device_eval_batch_size=int(per_device_eval_batch_size),
        learning_rate=float(learning_rate),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        seed=int(seed),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    label_distribution = Counter(safe_text(record.get("label", "")) for record in records)
    summary = {
        "base_model_path": base_model_path,
        "output_dir": str(output_path),
        "train_size": len(train_records),
        "eval_size": len(eval_records),
        "label_distribution": dict(label_distribution),
    }
    logger.info(
        "Context classifier training finished: %s (train=%d, eval=%d)",
        output_path,
        len(train_records),
        len(eval_records),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    返回:
        argparse.ArgumentParser: 已注册子命令和参数的解析器。
    """
    parser = argparse.ArgumentParser(description="Hard-skill context classifier")
    subparsers = parser.add_subparsers(dest="command")

    build_ds = subparsers.add_parser("build-dataset", help="Build a weakly labeled dataset from a regression set.")
    build_ds.add_argument("--regression-dataset", required=True, help="Regression dataset path.")
    build_ds.add_argument("--dictionary", default="dicts/flat_skill_dictionary.json", help="Flat dictionary path.")
    build_ds.add_argument("--output", default=DEFAULT_DATASET_OUTPUT, help="Output JSONL path.")
    build_ds.add_argument("--negative-keep-ratio", type=float, default=1.0, help="Keep ratio for fallback negatives.")
    build_ds.add_argument("--seed", type=int, default=42, help="Random seed.")

    train_cmd = subparsers.add_parser("train", help="Train the multiclass context classifier.")
    train_cmd.add_argument("--dataset", required=True, help="Context JSONL dataset path.")
    train_cmd.add_argument("--output-dir", default=DEFAULT_MODEL_OUTPUT, help="Model output directory.")
    train_cmd.add_argument("--base-model", default=None, help="Base BERT model path. Defaults to BERT_path in config.")
    train_cmd.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    train_cmd.add_argument("--train-batch-size", type=int, default=16, help="Training batch size.")
    train_cmd.add_argument("--eval-batch-size", type=int, default=32, help="Eval batch size.")
    train_cmd.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    train_cmd.add_argument("--max-length", type=int, default=256, help="Maximum token length.")
    train_cmd.add_argument("--validation-ratio", type=float, default=0.1, help="Validation ratio.")
    train_cmd.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    """命令行入口函数。

    根据子命令分发到数据集构建或模型训练流程。
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-dataset":
        build_context_dataset_from_regression(
            regression_dataset_path=args.regression_dataset,
            dictionary_path=args.dictionary,
            output_path=args.output,
            negative_keep_ratio=args.negative_keep_ratio,
            seed=args.seed,
        )
        return

    if args.command == "train":
        train_context_classifier(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            base_model_path=args.base_model,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
