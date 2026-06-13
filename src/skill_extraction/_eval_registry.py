"""评估注册表读写工具。

注册表文件 ``output/skill_extraction/eval/registry.json`` 以 JSON 格式
存储所有评估记录，每条记录包含词典版本、指标、评估时间等信息。

用法::

    from src.skill_extraction._eval_registry import load_registry, append_eval_record

    registry_dir = Path("output/skill_extraction/eval")
    registry = load_registry(registry_dir)
    append_eval_record(registry_dir, {
        "dict_version": "v1",
        "evaluated_at": "2026-06-12T14:00:00",
        "soft_skill_metrics": {...},
        "hard_skill_metrics": {...},
        ...
    })
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_registry_path(registry_dir: Path) -> Path:
    """获取 registry.json 的完整路径。"""
    return registry_dir / "registry.json"


def load_registry(registry_dir: Path) -> Dict[str, Any]:
    """加载评估注册表，不存在时返回空注册表。

    参数:
        registry_dir: 评估输出目录。

    返回:
        dict: 注册表数据，格式为 ``{"evaluations": [...]}``。
    """
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = _get_registry_path(registry_dir)
    if not path.exists():
        default: Dict[str, Any] = {"evaluations": []}
        path.write_text(
            json.dumps(default, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_eval_record(registry_dir: Path, record: Dict[str, Any]) -> None:
    """向注册表追加一条评估记录。

    参数:
        registry_dir: 评估输出目录。
        record: 评估记录字典，至少包含 ``dict_version`` 和 ``evaluated_at``。
    """
    registry = load_registry(registry_dir)
    registry.setdefault("evaluations", []).append(record)
    path = _get_registry_path(registry_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def get_record_by_version(registry_dir: Path, version: str) -> Optional[Dict[str, Any]]:
    """获取指定词典版本的最新评估记录。

    参数:
        registry_dir: 评估输出目录。
        version: 词典版本号，如 "v1"。

    返回:
        dict | None: 最新记录，未找到时返回 None。
    """
    registry = load_registry(registry_dir)
    candidates = [
        r for r in registry.get("evaluations", []) if r.get("dict_version") == version
    ]
    if not candidates:
        return None
    return candidates[-1]


def list_records(registry_dir: Path) -> List[Dict[str, Any]]:
    """列出所有版本的最新评估记录（每个版本取最后一条）。

    参数:
        registry_dir: 评估输出目录。

    返回:
        list[dict]: 每个版本的最新记录列表。
    """
    registry = load_registry(registry_dir)
    latest: Dict[str, Dict[str, Any]] = {}
    for r in registry.get("evaluations", []):
        version = r.get("dict_version", "unknown")
        latest[version] = r
    return list(latest.values())
