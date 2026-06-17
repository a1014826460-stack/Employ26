"""occupation_retrieval 的标注解析与样本构造工具。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


def parse_choice(annotation: dict[str, Any]) -> str | None:
    """从 Label Studio 单条标注中提取 A-E 或 NONE 选择。

    Args:
        annotation: Label Studio annotation 记录。

    Returns:
        str | None: 返回 `A` 到 `E`、`NONE`，无法解析时返回 `None`。
    """
    for result in annotation.get("result", []):
        if result.get("from_name") != "best_candidate_choice":
            continue
        choices = result.get("value", {}).get("choices", [])
        if not choices:
            return None
        raw = str(choices[0])
        if len(raw) >= 2 and raw[-1] in "ABCDE":
            return raw[-1]
        if "不" in raw:
            return "NONE"
    return None


def get_task_choices(
    task: dict[str, Any],
    *,
    include_none: bool = True,
) -> list[str]:
    """提取任务中所有可解析的标注选择。

    Args:
        task: `load_annotations_from_pg()` 返回的一条任务记录。
        include_none: 是否保留 `NONE` 选择。

    Returns:
        list[str]: 规范化后的选择列表。
    """
    choices: list[str] = []
    for annotation in task.get("annotations", []):
        choice = parse_choice(annotation)
        if choice is None:
            continue
        if choice == "NONE" and not include_none:
            continue
        choices.append(choice)
    return choices


def get_majority_choice(
    choices: Sequence[str],
    *,
    require_strict: bool = True,
) -> tuple[str | None, int]:
    """计算选择列表中的多数意见。

    Args:
        choices: 已规范化的选择列表。
        require_strict: 为 `True` 时要求最高票超过总票数一半。

    Returns:
        tuple[str | None, int]: 多数选择和票数；无多数时选择为 `None`。
    """
    if not choices:
        return None, 0
    counter = Counter(choices)
    choice, count = counter.most_common(1)[0]
    if require_strict and count <= len(choices) / 2:
        return None, count
    return choice, count


def build_candidate_records(data: dict[str, Any]) -> list[dict[str, str]]:
    """从任务 data 字段构造 A-E 候选记录。

    Args:
        data: Label Studio 任务 data 字段。

    Returns:
        list[dict[str, str]]: 每项包含 letter/code/title/source。
    """
    records: list[dict[str, str]] = []
    for letter in "abcde":
        upper = letter.upper()
        records.append(
            {
                "letter": upper,
                "code": str(data.get(f"candidate_{letter}_code", "") or "").strip(),
                "title": str(data.get(f"candidate_{letter}_title", "") or "").strip(),
                "source": str(data.get(f"candidate_{letter}_source", "") or "").strip(),
            }
        )
    return records


def build_anchor(job_title: str, job_requirements: str) -> str:
    """构造检索 anchor 文本。

    Args:
        job_title: 岗位名称。
        job_requirements: 岗位要求文本。

    Returns:
        str: 与历史脚本一致的标题加要求文本。
    """
    title = str(job_title or "").strip()
    requirements = str(job_requirements or "").strip()
    if title and requirements:
        return f"{title} {requirements}"
    return title or requirements
