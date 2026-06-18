# DeepSeek 第二轮检验性标注实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 DeepSeek 对第二轮 Label Studio 任务做可断点续联的检验性标注，并把结果稳定落到 PostgreSQL 与规范化输出目录。

**Architecture:** 以 `annotations.label_studio_tasks_v2` 为输入，按任务逐条调用 DeepSeek judge 模型，成功记录写入 `annotations.deepseek_relabel_raw`，同时输出 JSONL、diff、覆盖率和缺失任务清单。断点续联同时依赖 PostgreSQL 已写入记录和本地进度文件，运行日志统一写到 `logs/`，过程产物统一写到 `output/deepseek_relabel/round2/`。

**Tech Stack:** Python 3.10+, PostgreSQL, SQLAlchemy, OpenAI-compatible DeepSeek API, `pathlib`, `threading`, `csv`, `json`.

---

### Task 1: 稳定任务读取与解析

**Files:**
- Modify: `src/anno_analysis/deepseek_round2_relabel.py`

- [ ] **Step 1: 调整任务查询**

```python
sql = f"""
    select
        id as task_id,
        coalesce(recruitment_record_id, '') as recruitment_record_id,
        coalesce(annotations_completed_jsonb::text, annotations_completed) as annotations_completed,
        coalesce(data_raw_jsonb::text, data_raw) as data_raw
    from {TASK_TABLE}
    order by id
"""
```

- [ ] **Step 2: 保持候选与人工多数票解析**

```python
data = safe_json_loads(row["data_raw"], {})
annotations = safe_json_loads(row["annotations_completed"], [])
job_title, job_requirements = extract_task_text(data)
candidates = extract_candidates(data)
human_choices, human_majority = parse_human_choices(annotations)
```

- [ ] **Step 3: 运行局部编译检查**

Run: `python -m compileall -q src/anno_analysis/deepseek_round2_relabel.py`
Expected: no syntax error.

### Task 2: 强化输出与断点续联

**Files:**
- Modify: `src/anno_analysis/deepseek_round2_relabel.py`

- [ ] **Step 1: 将差异输出改为标准 CSV writer**

```python
with DIFF_CSV.open("a", encoding="utf-8", newline="") as file_obj:
    writer = csv.writer(file_obj)
    if write_header:
        writer.writerow([...])
    writer.writerow([...])
```

- [ ] **Step 2: 默认关闭熔断中断**

```python
parser.add_argument("--max-consecutive-failures", type=int, default=0, help="连续失败熔断阈值，0 表示禁用")
```

- [ ] **Step 3: 用 PG 记录 + 进度文件合并去重**

```python
done_ids = existing_ids | progress_ids
pending = [task for task in tasks if args.force or task.task_id not in done_ids]
```

- [ ] **Step 4: dry-run 只做覆盖率检查，不调用 API**

```python
if args.dry_run:
    coverage = write_coverage_report(total_tasks=len(tasks))
    logger.info("dry-run 覆盖情况: %s", coverage)
    return
```

### Task 3: 同步回填路径与文档

**Files:**
- Modify: `src/occupation_retrieval/common.py`
- Modify: `src/occupation_retrieval/README.md`

- [ ] **Step 1: 优先读取 round2 新 JSONL 回填路径**

```python
candidates = [
    PROJECT_ROOT / "output" / "deepseek_relabel" / "round2" / "round2_deepseek_relabel_raw.jsonl",
    PROJECT_ROOT / "output" / "deepseek_relabel" / "deepseek_relabel_raw.jsonl",
]
```

- [ ] **Step 2: 更新 README 的 DeepSeek 结果说明**

```markdown
- DeepSeek 重标结果：优先 PostgreSQL `annotations.deepseek_relabel_raw`，首次缺表或空表时会从 `output/deepseek_relabel/round2/round2_deepseek_relabel_raw.jsonl` 回填；旧路径仅作历史兼容
```

- [ ] **Step 3: 复查文本无英文旧路径遗漏**

Run: `rg -n "output/deepseek_relabel/deepseek_relabel_raw.jsonl|deepseek_round2_relabel" src/occupation_retrieval README.md CLAUDE.md`
Expected: only intentional compatibility references remain.

### Task 4: 验证

**Files:**
- Test: `python -m compileall -q src`
- Test: `python -m src.anno_analysis.deepseek_round2_relabel --dry-run --limit 5`

- [ ] **Step 1: 编译整仓库**

Run: `python -m compileall -q src`
Expected: exit code 0.

- [ ] **Step 2: 运行 dry-run**

Run: `python -m src.anno_analysis.deepseek_round2_relabel --dry-run --limit 5`
Expected: 输出覆盖率报告，生成 `output/deepseek_relabel/round2/round2_deepseek_relabel_coverage.md`。

- [ ] **Step 3: 复核产物位置**

Run: `Get-ChildItem logs, output\\deepseek_relabel\\round2`
Expected: 日志在 `logs/`，过程文件在 `output/deepseek_relabel/round2/`。
