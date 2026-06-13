"""测试词典路径解析工具。"""

import pytest

from src.skill_extraction._dict_paths import (
    get_current_soft_skill_dict_path,
    get_soft_skill_dict_path_for_version,
    list_soft_skill_dict_versions,
)


def test_get_current_soft_skill_dict_path(tmp_path, monkeypatch):
    """从 current.txt 读取当前版本并返回对应词典路径。"""
    dict_dir = tmp_path / "soft_skill"
    dict_dir.mkdir()
    (dict_dir / "current.txt").write_text("v1")
    (dict_dir / "v1.json").write_text('{"version": "v1"}')

    monkeypatch.setattr(
        "src.skill_extraction._dict_paths._SOFT_SKILL_DICT_DIR",
        dict_dir,
    )
    result = get_current_soft_skill_dict_path()
    assert result == dict_dir / "v1.json"


def test_get_soft_skill_dict_path_for_version(tmp_path, monkeypatch):
    """根据版本号返回对应路径。"""
    dict_dir = tmp_path / "soft_skill"
    dict_dir.mkdir()

    monkeypatch.setattr(
        "src.skill_extraction._dict_paths._SOFT_SKILL_DICT_DIR",
        dict_dir,
    )
    result = get_soft_skill_dict_path_for_version("v3")
    assert result == dict_dir / "v3.json"


def test_list_soft_skill_dict_versions(tmp_path, monkeypatch):
    """列出目录下所有版本。"""
    dict_dir = tmp_path / "soft_skill"
    dict_dir.mkdir()
    (dict_dir / "v1.json").touch()
    (dict_dir / "v2.json").touch()
    (dict_dir / "current.txt").write_text("v1")
    (dict_dir / "README.md").touch()

    monkeypatch.setattr(
        "src.skill_extraction._dict_paths._SOFT_SKILL_DICT_DIR",
        dict_dir,
    )
    versions = list_soft_skill_dict_versions()
    assert set(versions) == {"v1", "v2"}


def test_get_current_soft_skill_dict_path_missing_current_txt(tmp_path, monkeypatch):
    """current.txt 不存在时抛出 FileNotFoundError。"""
    dict_dir = tmp_path / "soft_skill"
    dict_dir.mkdir()

    monkeypatch.setattr(
        "src.skill_extraction._dict_paths._SOFT_SKILL_DICT_DIR",
        dict_dir,
    )
    with pytest.raises(FileNotFoundError, match="current.txt"):
        get_current_soft_skill_dict_path()


def test_get_current_soft_skill_dict_path_missing_version_file(tmp_path, monkeypatch):
    """current.txt 指向的版本文件不存在时抛出 FileNotFoundError。"""
    dict_dir = tmp_path / "soft_skill"
    dict_dir.mkdir()
    (dict_dir / "current.txt").write_text("v99")

    monkeypatch.setattr(
        "src.skill_extraction._dict_paths._SOFT_SKILL_DICT_DIR",
        dict_dir,
    )
    with pytest.raises(FileNotFoundError, match="v99"):
        get_current_soft_skill_dict_path()
