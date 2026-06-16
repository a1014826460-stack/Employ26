"""软技能词典版本路径解析工具。

本模块不依赖 DuckDB、PostgreSQL 或任何外部服务。
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def _get_dict_dir() -> Path:
    """获取软技能词典目录的绝对路径。"""
    from config.paths import get_project_paths

    return get_project_paths().project_root / "dicts" / "soft_skill"


# 模块级变量（仅用于测试时通过 monkeypatch 覆盖）
_SOFT_SKILL_DICT_DIR: Path | None = None


def _resolve_dict_dir() -> Path:
    """解析词典目录路径（支持测试注入）。"""
    if _SOFT_SKILL_DICT_DIR is not None:
        return _SOFT_SKILL_DICT_DIR
    return _get_dict_dir()


def get_current_soft_skill_dict_path() -> Path:
    """读取 current.txt 获取当前版本，返回对应词典文件的绝对路径。

    返回:
        Path: 当前活跃版本词典文件的路径。

    异常:
        FileNotFoundError: current.txt 不存在。
    """
    dict_dir = _resolve_dict_dir()
    current_file = dict_dir / "current.txt"
    if not current_file.exists():
        raise FileNotFoundError(
            f"版本标记文件不存在: {current_file}\n"
            f"请在 {dict_dir} 下创建 current.txt，内容为版本号（如 v1）"
        )
    version = current_file.read_text(encoding="utf-8").strip()
    dict_path = dict_dir / f"{version}.json"
    if not dict_path.exists():
        raise FileNotFoundError(
            f"词典文件不存在: {dict_path}\n"
            f"当前版本标记为 {version}，但对应文件未找到"
        )
    return dict_path


def get_soft_skill_dict_path_for_version(version: str) -> Path:
    """根据版本号返回词典文件路径（不检查文件是否存在）。

    参数:
        version: 版本标识，如 "v1"、"v2"。

    返回:
        Path: 对应版本的词典文件路径。
    """
    return _resolve_dict_dir() / f"{version}.json"


def list_soft_skill_dict_versions() -> List[str]:
    """列出所有可用版本号。

    返回:
        list[str]: 版本号列表，按字母序排列。
    """
    dict_dir = _resolve_dict_dir()
    if not dict_dir.exists():
        return []
    versions: List[str] = []
    for f in sorted(dict_dir.iterdir()):
        if f.suffix == ".json" and f.stem.startswith("v"):
            versions.append(f.stem)
    return versions
