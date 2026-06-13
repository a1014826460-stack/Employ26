# Soft Skill Iteration Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立软技能匹配的可迭代改进框架：词典版本化 + 评估注册表 + 统一评估 CLI。

**Architecture:** 词典迁移到 `dicts/soft_skill/v1.json`，版本由 `current.txt` 控制。`eval_cli.py` 包装 `eval_v3.py` 的核心评估逻辑，增加注册表读写和版本对比。所有下游模块通过辅助函数解析当前版本路径。

**Tech Stack:** Python 3.10+, pytest, json, pathlib, argparse

---

### Task 1: 词典版本化基础设施

**Files:**
- Create: `dicts/soft_skill/current.txt`
- Create: `dicts/soft_skill/v1.json`（从 `dicts/soft_skill_dictionary.json` 复制）
- Create: `src/skill_extraction/_dict_paths.py`
- Test: `src/tests/test_dict_paths.py`

- [ ] **Step 1: 创建版本目录和 v1.json**

```bash
mkdir -p dicts/soft_skill
cp dicts/soft_skill_dictionary.json dicts/soft_skill/v1.json
```

- [ ] **Step 2: 创建 current.txt**

写入文件 `dicts/soft_skill/current.txt`，内容为单行字符串 `v1`（无换行符导致的空白）：

```
v1
```

- [ ] **Step 3: 编写 _dict_paths.py 测试**

创建 `src/tests/test_dict_paths.py`：

```python
"""测试词典路径解析工具。"""
from pathlib import Path
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


def test_get_current_soft_skill_dict_path_missing_file(tmp_path, monkeypatch):
    """current.txt 不存在时抛出 FileNotFoundError。"""
    dict_dir = tmp_path / "soft_skill"
    dict_dir.mkdir()

    monkeypatch.setattr(
        "src.skill_extraction._dict_paths._SOFT_SKILL_DICT_DIR",
        dict_dir,
    )
    import pytest
    with pytest.raises(FileNotFoundError, match="current.txt"):
        get_current_soft_skill_dict_path()
```

- [ ] **Step 4: 运行测试验证失败**

```bash
python -m pytest src/tests/test_dict_paths.py -v
```
预期：4 个 FAIL（模块不存在）

- [ ] **Step 5: 实现 _dict_paths.py**

创建 `src/skill_extraction/_dict_paths.py`：

```python
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


# 模块级缓存（仅用于测试时通过 monkeypatch 覆盖）
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
```

- [ ] **Step 6: 运行测试验证通过**

```bash
python -m pytest src/tests/test_dict_paths.py -v
```
预期：4 个 PASS

- [ ] **Step 7: 提交**

```bash
git add dicts/soft_skill/ src/skill_extraction/_dict_paths.py src/tests/test_dict_paths.py
git commit -m "feat: add soft skill dictionary versioning infrastructure"
```

---

### Task 2: 更新所有词典路径引用

**Files:**
- Modify: `src/skill_extraction/soft_skill_matcher.py`
- Modify: `src/skill_extraction/soft_skill_dictionary_builder.py`
- Modify: `src/skill_extraction/v3_pipeline.py`
- Modify: `src/skill_extraction/eval_v3.py`

- [ ] **Step 1: 更新 soft_skill_matcher.py**

读取文件找到 `DEFAULT_DICT_PATH` 或硬编码的词典路径，替换为使用 `_dict_paths`：

将文件顶部或 `__init__` 中的路径加载逻辑替换。当前 `soft_skill_matcher.py` 的 `__init__` 大约在 line 70-92，其中 `dict_path` 参数允许传入，默认值为 `dicts/soft_skill_dictionary.json`。改为默认使用 `_dict_paths`。

在 `soft_skill_matcher.py` 中添加导入：

```python
from ._dict_paths import get_current_soft_skill_dict_path
```

找到 `__init__` 方法中的默认路径参数，将默认值改为 `None`，在方法内部解析：

```python
def __init__(self, dict_path: str | Path | None = None) -> None:
    if dict_path is None:
        dict_path = str(get_current_soft_skill_dict_path())
    # ... 其余逻辑不变
```

- [ ] **Step 2: 更新 soft_skill_dictionary_builder.py**

查找 `DEFAULT_OUTPUT_PATH` 或类似常量，将其从 `dicts/soft_skill_dictionary.json` 改为 `dicts/soft_skill/v{N}.json` 模式。但该模块的默认输出路径是全局变量，改为通过参数控制更灵活。保持默认输出到 `dicts/soft_skill/` 目录。

- [ ] **Step 3: 更新 v3_pipeline.py 的 create_v3_pipeline**

在 `create_v3_pipeline()` 函数中（约 line 340），当前硬技能词典路径回退到 `dicts/flat_skill_dictionary.json`。软技能词典通过 `SoftSkillMatcher()` 无参构造加载。这已经通过 Task 2 Step 1 的修改自动适配，无需额外改动。

但确认 `create_v3_pipeline` 中初始化 `SoftSkillMatcher` 时不再传硬编码路径。

- [ ] **Step 4: 更新 eval_v3.py 的 run() 函数**

在 `run()` 函数中（约 line 686），当前硬技能词典默认路径为 `dicts/flat_skill_dictionary.json`，保持不动。软技能词典在 `run()` 内通过 `SoftSkillMatcher()` 加载，已通过 Task 2 Step 1 适配。

- [ ] **Step 5: 运行现有测试确认无回归**

```bash
python -m pytest src/tests/test_soft_skill_matcher.py src/tests/test_v3_pipeline.py -v
```
预期：全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/skill_extraction/soft_skill_matcher.py src/skill_extraction/soft_skill_dictionary_builder.py src/skill_extraction/v3_pipeline.py src/skill_extraction/eval_v3.py
git commit -m "refactor: route soft skill dict loading through versioned paths"
```

---

### Task 3: 评估注册表读写

**Files:**
- Create: `src/skill_extraction/_eval_registry.py`
- Test: `src/tests/test_eval_registry.py`

- [ ] **Step 1: 编写 _eval_registry.py 测试**

创建 `src/tests/test_eval_registry.py`：

```python
"""测试评估注册表读写。"""
import json
from pathlib import Path
from src.skill_extraction._eval_registry import (
    load_registry,
    append_eval_record,
    get_record_by_version,
    list_records,
)


SAMPLE_RECORD = {
    "dict_version": "v1",
    "evaluated_at": "2026-06-12T14:00:00",
    "soft_skill_metrics": {
        "coverage": 0.1141,
        "precision": 0.0876,
        "dimension_accuracy": 0.8495,
    },
    "hard_skill_metrics": {
        "precision": 0.7018,
        "recall": 0.9053,
        "f1": 0.7907,
        "category_accuracy": 1.0,
    },
    "gold_source": "annotations.label_studio_tasks_v2",
    "sample_count": 300,
}


def test_load_registry_creates_if_missing(tmp_path):
    """registry.json 不存在时自动创建空注册表。"""
    result = load_registry(tmp_path)
    assert result == {"evaluations": []}
    assert (tmp_path / "registry.json").exists()


def test_load_registry_reads_existing(tmp_path):
    """读取已有注册表。"""
    existing = {"evaluations": [SAMPLE_RECORD]}
    (tmp_path / "registry.json").write_text(
        json.dumps(existing, ensure_ascii=False)
    )
    result = load_registry(tmp_path)
    assert len(result["evaluations"]) == 1


def test_append_eval_record(tmp_path):
    """追加一条评估记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    with open(tmp_path / "registry.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["evaluations"]) == 1
    assert data["evaluations"][0]["dict_version"] == "v1"


def test_append_multiple_records(tmp_path):
    """追加多条记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    record2 = dict(SAMPLE_RECORD, dict_version="v2")
    append_eval_record(tmp_path, record2)
    with open(tmp_path / "registry.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["evaluations"]) == 2


def test_get_record_by_version(tmp_path):
    """按版本号获取最新记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    record2 = dict(
        SAMPLE_RECORD,
        dict_version="v1",
        evaluated_at="2026-06-12T15:00:00",
    )
    append_eval_record(tmp_path, record2)
    result = get_record_by_version(tmp_path, "v1")
    # 返回最新一条
    assert result["evaluated_at"] == "2026-06-12T15:00:00"


def test_get_record_by_version_not_found(tmp_path):
    """版本不存在时返回 None。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    result = get_record_by_version(tmp_path, "v99")
    assert result is None


def test_list_records(tmp_path):
    """列出所有版本的最新评估记录。"""
    append_eval_record(tmp_path, SAMPLE_RECORD)
    record2 = dict(SAMPLE_RECORD, dict_version="v2")
    append_eval_record(tmp_path, record2)
    records = list_records(tmp_path)
    assert len(records) == 2
    versions = {r["dict_version"] for r in records}
    assert versions == {"v1", "v2"}
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest src/tests/test_eval_registry.py -v
```
预期：7 个 FAIL

- [ ] **Step 3: 实现 _eval_registry.py**

创建 `src/skill_extraction/_eval_registry.py`：

```python
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
from datetime import datetime
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


def get_record_by_version(
    registry_dir: Path, version: str
) -> Optional[Dict[str, Any]]:
    """获取指定词典版本的最新评估记录。

    参数:
        registry_dir: 评估输出目录。
        version: 词典版本号，如 "v1"。

    返回:
        dict | None: 最新记录，未找到时返回 None。
    """
    registry = load_registry(registry_dir)
    candidates = [
        r
        for r in registry.get("evaluations", [])
        if r.get("dict_version") == version
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest src/tests/test_eval_registry.py -v
```
预期：7 个 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skill_extraction/_eval_registry.py src/tests/test_eval_registry.py
git commit -m "feat: add eval registry read/write with version tracking"
```

---

### Task 4: eval_cli.py — list 命令

**Files:**
- Create: `src/skill_extraction/eval_cli.py`
- Test: `src/tests/test_eval_cli.py`

- [ ] **Step 1: 编写 list 命令测试**

创建 `src/tests/test_eval_cli.py`：

```python
"""测试 eval_cli 命令。"""
import json
import pytest
from pathlib import Path
from src.skill_extraction.eval_cli import build_parser, cmd_list


SAMPLE_RECORD = {
    "dict_version": "v1",
    "evaluated_at": "2026-06-12T14:00:00",
    "soft_skill_metrics": {
        "coverage": 0.1141,
        "precision": 0.0876,
        "dimension_accuracy": 0.8495,
    },
    "hard_skill_metrics": {
        "precision": 0.7018,
        "recall": 0.9053,
        "f1": 0.7907,
        "category_accuracy": 1.0,
    },
    "gold_source": "annotations.label_studio_tasks_v2",
    "sample_count": 300,
}


class TestBuildParser:
    def test_list_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_run_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"

    def test_compare_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["compare", "v1", "v2"])
        assert args.command == "compare"
        assert args.version_a == "v1"
        assert args.version_b == "v2"

    def test_no_command_shows_help(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])


class TestCmdList:
    def test_empty_registry(self, tmp_path, capsys):
        from src.skill_extraction._eval_registry import load_registry
        load_registry(tmp_path)  # 创建空注册表
        cmd_list(tmp_path)
        captured = capsys.readouterr()
        assert "暂无评估记录" in captured.out
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest src/tests/test_eval_cli.py -v
```
预期：FAIL（模块不存在）

- [ ] **Step 3: 实现 eval_cli.py 框架和 list 命令**

创建 `src/skill_extraction/eval_cli.py`：

```python
"""软技能评估 CLI — 统一入口。

用法::

    python -m src.skill_extraction.eval_cli run
    python -m src.skill_extraction.eval_cli compare v1 v2
    python -m src.skill_extraction.eval_cli list
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        print("暂无评估记录。")
        return

    print(f"{'版本':<8} {'评估时间':<22} {'软覆盖率':<10} {'软精确率':<10} {'硬F1':<10}")
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
        description="软技能评估 CLI — 运行评估、对比版本、查看记录",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    sub.add_parser("run", help="运行评估并写入注册表")

    # compare
    cmp_parser = sub.add_parser("compare", help="对比两个版本的指标")
    cmp_parser.add_argument("version_a", help="基准版本（如 v1）")
    cmp_parser.add_argument("version_b", help="对比版本（如 v2）")

    # list
    sub.add_parser("list", help="列出所有评估记录")

    return parser


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
        logger.info("run 命令将在 Task 5 中实现")
    elif args.command == "compare":
        logger.info("compare 命令将在 Task 6 中实现")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest src/tests/test_eval_cli.py -v
```
预期：5 个 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skill_extraction/eval_cli.py src/tests/test_eval_cli.py
git commit -m "feat: add eval_cli with list command"
```

---

### Task 5: eval_cli.py — run 命令

**Files:**
- Modify: `src/skill_extraction/eval_cli.py`
- Modify: `src/tests/test_eval_cli.py`（扩展）

- [ ] **Step 1: 编写 run 命令测试**

在 `src/tests/test_eval_cli.py` 中添加：

```python
class TestCmdRun:
    def test_run_creates_registry_record(self, tmp_path, monkeypatch):
        """run 命令应写入注册表记录并保存报告文件。"""
        import json
        from src.skill_extraction.eval_cli import cmd_run

        # 需要 gold 数据集 — 创建最小测试集
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        hard_data = [
            {
                "sample_id": "test_1",
                "text": "熟练使用 Java 和 MySQL",
                "gold_skills": ["Java", "MySQL"],
                "gold_categories": {"Java": "programming_language"},
            }
        ]
        soft_data = [
            {
                "sample_id": "test_1",
                "text": "具备沟通能力和责任心",
                "gold_soft_skills": [
                    {"name": "沟通能力", "dimension": "extraversion"},
                    {"name": "责任心", "dimension": "conscientiousness"},
                ],
            }
        ]
        (gold_dir / "hard.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in hard_data)
        )
        (gold_dir / "soft.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in soft_data)
        )

        eval_dir = tmp_path / "eval"
        cmd_run(
            eval_dir=eval_dir,
            hard_dataset=gold_dir / "hard.jsonl",
            soft_dataset=gold_dir / "soft.jsonl",
        )

        # 验证注册表
        with open(eval_dir / "registry.json", "r", encoding="utf-8") as f:
            registry = json.load(f)
        assert len(registry["evaluations"]) == 1
        record = registry["evaluations"][0]
        assert record["dict_version"] == "v1"
        assert "soft_skill_metrics" in record
        assert "hard_skill_metrics" in record

        # 验证版本报告目录
        version_dir = eval_dir / "v1"
        assert version_dir.exists()
        assert (version_dir / "summary.json").exists()
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest src/tests/test_eval_cli.py::TestCmdRun -v
```
预期：FAIL（`cmd_run` 未实现或功能不全）

- [ ] **Step 3: 实现 cmd_run**

在 `eval_cli.py` 中添加：

```python
def cmd_run(
    eval_dir: Optional[Path] = None,
    hard_dataset: Optional[Path] = None,
    soft_dataset: Optional[Path] = None,
) -> None:
    """运行评估并将结果写入注册表。

    参数:
        eval_dir: 评估输出目录，为 None 时使用默认路径。
        hard_dataset: 硬技能 gold 数据集路径，为 None 时使用默认路径。
        soft_dataset: 软技能 gold 数据集路径，为 None 时使用默认路径。
    """
    from ._dict_paths import get_current_soft_skill_dict_path
    from ._eval_registry import append_eval_record
    from .eval_v3 import (
        V3EvalReport,
        _load_hard_skill_dataset,
        _load_soft_skill_dataset,
        evaluate,
    )
    from .hard_skill_matcher import FlatHardSkillMatcher, load_flat_dictionary
    from .soft_skill_matcher import SoftSkillMatcher

    project_root = Path(__file__).resolve().parents[2]
    registry_dir = eval_dir or _get_eval_dir()

    # 确定词典版本
    dict_path = get_current_soft_skill_dict_path()
    version = dict_path.stem  # "v1", "v2", etc.
    logger.info("当前软技能词典版本: %s", version)

    # 加载数据集
    hard_path = hard_dataset or (
        registry_dir / "hard_skill_eval_dataset.jsonl"
    )
    soft_path = soft_dataset or (
        registry_dir / "soft_skill_gold_dataset.jsonl"
    )
    hard_samples = _load_hard_skill_dataset(hard_path)
    soft_samples = _load_soft_skill_dataset(soft_path)
    logger.info(
        "加载数据: 硬技能 %d 条, 软技能 %d 条",
        len(hard_samples),
        len(soft_samples),
    )

    # 初始化匹配器
    hard_dict_path = project_root / "dicts" / "flat_skill_dictionary.json"
    hard_dict = load_flat_dictionary(str(hard_dict_path))
    hard_matcher = FlatHardSkillMatcher(hard_dict)
    soft_matcher = SoftSkillMatcher()

    # 运行评估
    version_report_dir = registry_dir / version
    report = evaluate(
        hard_samples=hard_samples,
        soft_samples=soft_samples,
        hard_matcher=hard_matcher,
        soft_matcher=soft_matcher,
        llm_client=None,
        output_dir=version_report_dir,
    )

    # 写入注册表
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
    append_eval_record(registry_dir, record)

    # 更新 latest 符号链接（如平台支持）
    _update_latest_link(registry_dir, version)

    # 输出摘要
    soft = report.soft_skill_metrics
    hard = report.hard_skill_metrics
    print(f"\n=== 评估完成 (词典版本: {version}) ===")
    print(f"软技能 — 覆盖率: {soft.coverage:.4f}  精确率: {soft.precision:.4f}  维度准确率: {soft.dimension_accuracy:.4f}")
    print(f"硬技能 — 精确率: {hard.precision:.4f}  召回率: {hard.recall:.4f}  F1: {hard.f1:.4f}")
    print(f"报告目录: {version_report_dir}")


def _update_latest_link(eval_dir: Path, version: str) -> None:
    """更新 latest 指向最新评估版本。

    在支持的平台上创建符号链接；Windows 上写一个 latest.txt 文件作为替代。
    """
    latest_link = eval_dir / "latest"
    # 尝试符号链接
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version, target_is_directory=True)
    except (OSError, AttributeError):
        # Windows fallback：写 latest.txt
        (eval_dir / "latest.txt").write_text(version, encoding="utf-8")
```

在 `main()` 中更新 `run` 分支：

```python
    if args.command == "list":
        cmd_list()
    elif args.command == "run":
        cmd_run()
    elif args.command == "compare":
        logger.info("compare 命令将在 Task 6 中实现")
    else:
        parser.print_help()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest src/tests/test_eval_cli.py -v
```
预期：全部 PASS

- [ ] **Step 5: 运行实际评估验证端到端**

```bash
python -m src.skill_extraction.eval_cli run
```
预期：输出评估摘要，写入 `output/skill_extraction/eval/registry.json` 和 `output/skill_extraction/eval/v1/`。

- [ ] **Step 6: 提交**

```bash
git add src/skill_extraction/eval_cli.py src/tests/test_eval_cli.py
git commit -m "feat: add eval_cli run command with registry integration"
```

---

### Task 6: eval_cli.py — compare 命令

**Files:**
- Modify: `src/skill_extraction/eval_cli.py`
- Modify: `src/tests/test_eval_cli.py`（扩展）

- [ ] **Step 1: 编写 compare 命令测试**

在 `src/tests/test_eval_cli.py` 中添加：

```python
class TestCmdCompare:
    def test_compare_shows_delta(self, tmp_path, capsys):
        """对比应显示两个版本的指标变化。"""
        from src.skill_extraction._eval_registry import append_eval_record
        from src.skill_extraction.eval_cli import cmd_compare

        record_v1 = {
            "dict_version": "v1",
            "evaluated_at": "2026-06-12T14:00:00",
            "soft_skill_metrics": {
                "coverage": 0.1141,
                "precision": 0.0876,
                "dimension_accuracy": 0.8495,
            },
            "hard_skill_metrics": {
                "precision": 0.7018,
                "recall": 0.9053,
                "f1": 0.7907,
                "category_accuracy": 1.0,
            },
            "gold_source": "annotations",
            "sample_count": 300,
        }
        record_v2 = dict(record_v1, dict_version="v2")
        record_v2["soft_skill_metrics"] = {
            "coverage": 0.25,
            "precision": 0.15,
            "dimension_accuracy": 0.86,
        }

        append_eval_record(tmp_path, record_v1)
        append_eval_record(tmp_path, record_v2)

        cmd_compare("v1", "v2", eval_dir=tmp_path)
        captured = capsys.readouterr()
        assert "coverage" in captured.out
        assert "0.1141" in captured.out
        assert "0.2500" in captured.out

    def test_compare_missing_version(self, tmp_path, capsys):
        """版本不存在时给出提示。"""
        from src.skill_extraction.eval_cli import cmd_compare

        cmd_compare("v1", "v99", eval_dir=tmp_path)
        captured = capsys.readouterr()
        assert "未找到" in captured.out
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest src/tests/test_eval_cli.py::TestCmdCompare -v
```
预期：FAIL

- [ ] **Step 3: 实现 cmd_compare**

在 `eval_cli.py` 中添加：

```python
def cmd_compare(
    version_a: str,
    version_b: str,
    eval_dir: Optional[Path] = None,
) -> None:
    """对比两个词典版本的评估指标。

    参数:
        version_a: 基准版本。
        version_b: 对比版本。
        eval_dir: 评估输出目录。
    """
    from ._eval_registry import get_record_by_version

    registry_dir = eval_dir or _get_eval_dir()

    record_a = get_record_by_version(registry_dir, version_a)
    record_b = get_record_by_version(registry_dir, version_b)

    if not record_a:
        print(f"未找到版本 {version_a} 的评估记录。")
        return
    if not record_b:
        print(f"未找到版本 {version_b} 的评估记录。")
        return

    soft_a = record_a.get("soft_skill_metrics", {})
    soft_b = record_b.get("soft_skill_metrics", {})
    hard_a = record_a.get("hard_skill_metrics", {})
    hard_b = record_b.get("hard_skill_metrics", {})

    def _delta(key: str, metrics_a: dict, metrics_b: dict) -> str:
        a = metrics_a.get(key, 0)
        b = metrics_b.get(key, 0)
        diff = b - a
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "─")
        return f"{arrow} {diff:+.4f}"

    def _fmt_pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    print(f"\n{'指标':<18} {version_a:<10} {version_b:<10} Δ")
    print("-" * 55)
    for label, key in [
        ("软技能-覆盖率", "coverage"),
        ("软技能-精确率", "precision"),
        ("软技能-维度准确率", "dimension_accuracy"),
        ("硬技能-精确率", "precision"),
        ("硬技能-召回率", "recall"),
        ("硬技能-F1", "f1"),
        ("硬技能-分类准确率", "category_accuracy"),
    ]:
        a_val = soft_a.get(key, 0) if "软" in label else hard_a.get(key, 0)
        b_val = soft_b.get(key, 0) if "软" in label else hard_b.get(key, 0)
        d = _delta(key, soft_a if "软" in label else hard_a, soft_b if "软" in label else hard_b)
        print(f"{label:<18} {_fmt_pct(a_val):<10} {_fmt_pct(b_val):<10} {d}")
```

在 `main()` 中更新 `compare` 分支：

```python
    elif args.command == "compare":
        cmd_compare(args.version_a, args.version_b)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest src/tests/test_eval_cli.py -v
```
预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skill_extraction/eval_cli.py src/tests/test_eval_cli.py
git commit -m "feat: add eval_cli compare command with delta reporting"
```

---

### Task 7: 向后兼容 shim

**Files:**
- Modify: `dicts/soft_skill_dictionary.json`（改为引用文件）

- [ ] **Step 1: 保留旧路径的兼容引用**

将 `dicts/soft_skill_dictionary.json` 的内容替换为一个 JSON 引用文件：

```json
{
  "_note": "此文件为向后兼容保留。请使用 dicts/soft_skill/current.txt 获取当前版本。",
  "_redirect": "dicts/soft_skill/v1.json"
}
```

同时确保任何仍读取旧路径的代码能得到合理提示而非静默失败。

- [ ] **Step 2: 运行全量测试确认无回归**

```bash
python -m pytest src/tests/ -q
```
预期：280 个 PASS

- [ ] **Step 3: 提交**

```bash
git add dicts/soft_skill_dictionary.json
git commit -m "refactor: add backward-compat shim for old soft skill dict path"
```

---

### Task 8: 首次迭代 — LLM 词典扩展 + 评估

**Files:**
- Create: `dicts/soft_skill/v2.json`
- Modify: `dicts/soft_skill/current.txt`

- [ ] **Step 1: 复制 v1 → v2**

```bash
cp dicts/soft_skill/v1.json dicts/soft_skill/v2.json
```

- [ ] **Step 2: 运行 LLM 词典扩展**

检查 vLLM 服务是否可用。如果不可用，手动添加高频缺失词的别名：

在当前 `v2.json` 中为每个维度补充别名。重点补充在 gold 数据中高频出现但在词典中缺失的词：
- `吃苦耐劳` → conscientiousness
- `上进心` / `进取心` → openness
- `沟通表达能力` / `沟通协调能力` → 作为 "沟通能力" 的别名
- `责任心强` / `有责任心` → 作为 "责任心" 的别名

如果 vLLM 可用，运行：
```bash
python -m src.skill_extraction.soft_skill_dictionary_builder --use-llm --output dicts/soft_skill/v2.json
```

- [ ] **Step 3: 更新 current.txt**

```bash
echo "v2" > dicts/soft_skill/current.txt
```

- [ ] **Step 4: 运行评估**

```bash
python -m src.skill_extraction.eval_cli run
```

- [ ] **Step 5: 对比结果**

```bash
python -m src.skill_extraction.eval_cli compare v1 v2
```

- [ ] **Step 6: 提交**

```bash
git add dicts/soft_skill/ src/skill_extraction/eval_cli.py
git commit -m "feat: first soft skill dictionary iteration (v2) with expanded aliases"
```

---

### Task 9: 文档更新

**Files:**
- Modify: `src/skill_extraction/README.md`

- [ ] **Step 1: 更新 README 添加迭代框架说明**

在 README 中添加软技能迭代框架的使用说明：

```markdown
### 软技能迭代框架

词典版本化 + 评估注册表 + CLI。

```bash
# 查看当前版本
cat dicts/soft_skill/current.txt

# 创建新版本
cp dicts/soft_skill/v1.json dicts/soft_skill/v2.json
# 编辑 v2.json ...
echo "v2" > dicts/soft_skill/current.txt

# 运行评估
python -m src.skill_extraction.eval_cli run

# 对比版本
python -m src.skill_extraction.eval_cli compare v1 v2

# 查看历史
python -m src.skill_extraction.eval_cli list
```
```

- [ ] **Step 2: 提交**

```bash
git add src/skill_extraction/README.md
git commit -m "docs: add soft skill iteration framework usage guide"
```
