# Soft Skill Iteration Framework 设计文档

**Goal**

建立一个可迭代的软技能匹配改进框架。不追求一步达标，但确保每次改进可量化验证。

**截止时间**：无硬性截止。框架本身为长期维护工具。

## 范围

### 覆盖

- 词典版本化管理（`dicts/soft_skill/`）
- 评估注册表（`output/skill_extraction/eval/registry.json`）
- 统一评估 CLI（`src/skill_extraction/eval_cli.py`，`run` / `compare` / `list`）
- `soft_skill_matcher.py`、`soft_skill_dictionary_builder.py` 等加载路径适配新目录结构

### 不覆盖

- 不改动硬技能词典管理（`flat_skill_dictionary.json` 保持现状）
- 不改动 `eval_v3.py` 的核心评估逻辑（仅新增 CLI wrapper）
- 不在此阶段实现自动迭代（每次迭代由人工触发）

## 已确认事实

1. 软技能 gold 数据：`output/skill_extraction/eval/soft_skill_gold_dataset.jsonl`（300 条，来源于 `annotations.label_studio_tasks_v2.annotations_completed_jsonb`）
2. 硬技能评估数据：`output/skill_extraction/eval/hard_skill_eval_dataset.jsonl`（83 条，自动生成）
3. 当前软技能词典：`dicts/soft_skill_dictionary.json`（73 个技能，222 个别名，schema_version 1）
4. 当前评估结果：覆盖率 11.41%、精确率 8.76%、维度准确率 84.95%
5. LLM 词典扩展（`soft_skill_dictionary_builder.py --use-llm`）已实现但从未运行
6. PostgreSQL 中有 12,567 条任务包含 35,901 个人工标注的软技能文本跨度

## 设计

### 1. 词典版本化

```
dicts/soft_skill/
├── v1.json              # 当前词典（从 soft_skill_dictionary.json 迁移）
├── v2.json              # 第一次改进后
├── v3.json              # 第二次改进后
└── current.txt          # 当前活跃版本标识（内容: "v1"）
```

- 词典内部格式不变（`schema_version` + `dimensions` + `skills` + `aliases`）
- `current.txt` 使下游代码无需显式传版本号：加载时读 `current.txt` → 加载对应文件
- 每次改进流程：复制当前版本 → 修改 → 递增版本号 → 更新 `current.txt` → 运行评估
- `soft_skill_matcher.py` 和 `soft_skill_dictionary_builder.py` 的 `DEFAULT_DICT_PATH` 更新为新路径
- 旧 `dicts/soft_skill_dictionary.json` 作为符号链接或别名指向 `current` 对应的版本，保持向后兼容

### 2. 评估注册表

```
output/skill_extraction/eval/
├── registry.json        # 总索引（所有评估记录）
├── v1/
│   ├── summary.json     # 指标摘要
│   ├── soft_skill_errors.csv
│   └── hard_skill_errors.csv
├── v2/
│   └── ...
└── latest -> v1/        # 指向最新版本的符号链接
```

`registry.json` 结构：

```json
{
  "evaluations": [
    {
      "dict_version": "v1",
      "evaluated_at": "2026-06-12T14:00:00",
      "soft_skill_metrics": {
        "coverage": 0.1141,
        "precision": 0.0876,
        "dimension_accuracy": 0.8495
      },
      "hard_skill_metrics": {
        "precision": 0.7018,
        "recall": 0.9053,
        "f1": 0.7907,
        "category_accuracy": 1.0
      },
      "gold_source": "annotations.label_studio_tasks_v2",
      "sample_count": 300
    }
  ]
}
```

- 每次 `eval_cli run` 自动追加新记录
- 对比命令 `eval_cli compare` 读取任意两条记录并计算 Δ
- 指标名称与 `eval_v3.py` 中的 `HardSkillMetrics` / `SoftSkillMetrics` 严格一致

### 3. 评估 CLI

```
python -m src.skill_extraction.eval_cli run       # 运行评估
python -m src.skill_extraction.eval_cli compare A B  # 对比 vA vs vB
python -m src.skill_extraction.eval_cli list        # 列出所有评估
```

**`run`** 流程：
1. 读取 `current.txt` 获取当前词典版本
2. 加载 gold 数据集（hard + soft）
3. 初始化匹配器（加载当前版本词典）
4. 调用 `eval_v3.evaluate()` 运行评估
5. 将指标写入 `registry.json`（追加新记录）
6. 将报告文件写入 `output/skill_extraction/eval/{version}/`
7. 更新 `latest` 符号链接
8. 输出指标摘要到终端

**`compare <version_a> <version_b>`** 流程：
1. 从 `registry.json` 读取两个版本的记录
2. 逐指标对比，计算绝对差值（percentage points）
3. 格式化输出，↑ 标记提升，↓ 标记退化
4. 如果退化，高亮标注

**`list`** 流程：
1. 读取 `registry.json`
2. 表格形式输出所有评估记录：版本、时间、覆盖率、精确率、F1

### 4. 加载路径适配

| 模块 | 当前路径 | 新路径 |
|---|---|---|
| `soft_skill_matcher.py` | `dicts/soft_skill_dictionary.json` | `dicts/soft_skill/{current}.json` |
| `soft_skill_dictionary_builder.py` | `dicts/soft_skill_dictionary.json` | `dicts/soft_skill/{current}.json` |
| `soft_skill_seed_extractor.py` | N/A（只读 DB） | 不变 |
| `soft_skill_llm_validator.py` | N/A（不直接读词典） | 不变 |
| `v3_pipeline.py` | N/A（通过 matcher 间接使用） | 不变 |

加载辅助函数（新增在 `soft_skill_matcher.py` 或独立工具模块）：

```python
def get_current_soft_skill_dict_path() -> Path:
    """读取 current.txt，返回当前版本词典的绝对路径。"""
    current_file = PROJECT_ROOT / "dicts" / "soft_skill" / "current.txt"
    version = current_file.read_text().strip()
    return PROJECT_ROOT / "dicts" / "soft_skill" / f"{version}.json"
```

### 5. 向后兼容

- `dicts/soft_skill_dictionary.json` → 保留为符号链接，指向 `soft_skill/v1.json`
- 如果无法创建符号链接（Windows），写入一个只有路径引用的 JSON 文件
- `DEFAULT_SOFT_SKILL_DICT_PATH` 常量更新为新路径，旧路径 fallback 检测

## 错误处理

- `current.txt` 不存在 → 报错并提示创建
- 词典文件不存在 → 报错并列出可用版本
- `registry.json` 不存在 → 首次运行时自动创建
- Gold 数据集不存在 → 报错并提示生成方法

## 测试

- 测试 `get_current_soft_skill_dict_path()` 在不同 `current.txt` 内容下的行为
- 测试 `registry.json` 读写和追加
- 测试 `eval_cli` 三个子命令的基础参数解析
- 不测试 `eval_v3` 内部的评估逻辑（已有覆盖，无需重复）

## 架构约束

- 遵循 `CLAUDE.md`：无硬编码路径、优先复用 `config.paths`
- 新模块不引入 DuckDB 依赖
- 所有新增代码带中文 docstring 和类型提示
