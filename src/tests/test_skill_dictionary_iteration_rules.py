"""Tests for skill dictionary iteration rule loading."""

from pathlib import Path

from src.skill_extraction.dictionary.iteration_rules import (
    DEFAULT_RULES_PATH,
    get_exact_generic_skill_blocklist,
)


def test_default_iteration_rules_path_points_to_project_config() -> None:
    """默认迭代规则路径应指向项目根目录下的 config。"""
    assert DEFAULT_RULES_PATH == Path("config/skill_dictionary_iteration.json").resolve()
    assert DEFAULT_RULES_PATH.exists()


def test_default_generic_blocklist_is_loaded() -> None:
    """默认黑名单应从真实配置文件加载，而不是空集合。"""
    blocklist = get_exact_generic_skill_blocklist()

    assert "Microsoft Office" in blocklist
    assert "Microsoft Excel" in blocklist
