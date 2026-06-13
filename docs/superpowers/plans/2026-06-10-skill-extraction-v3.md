# Skill Extraction V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有 V2 硬技能词典增加 8 分类标签，新建软技能抽取管线，达到覆盖率 >90%、准确率 >85%。截止 2026-06-12 24:00。

**Architecture:** Phase 1 基于 `skill_type` 现有值做映射 + LLM 补全，为 1907 个硬技能增加 `category` 字段。Phase 2 新建大五人格软技能词典。Phase 3 实现软技能匹配管线并集成到统一入口。Phase 4 创建评估脚本和 PostgreSQL 输出表。

**Tech Stack:** Python 3.10+, PostgreSQL, psycopg2, pytest, `src.model_platform.llm`

---

### Task 1: Build Skill Category Mapping Rules

**Files:**
- Create: `dicts/skill_category_rules.json`
- Create: `src/skill_extraction/skill_category_mapper.py`
- Test: `src/tests/test_skill_category_mapper.py`

- [ ] 创建 `dicts/skill_category_rules.json`，包含 `skill_type` → `category` 映射规则和 LLM 分类用的类别定义。
- [ ] 映射规则覆盖全部已有 `skill_type` 值（约 100 种），每条规则格式为 `{"skill_type_value": "category"}`。
- [ ] 类别定义包含 8 类的英文标识、中文名称、典型示例，供 LLM 分类 prompt 使用。
- [ ] 实现 `src/skill_extraction/skill_category_mapper.py`：
  - `load_category_rules() -> dict`：加载映射规则
  - `map_skill_type(skill_type: str) -> str | None`：单条映射，返回 `category` 或 `None`
  - `classify_batch_by_llm(skill_names: list[str]) -> dict[str, str]`：通过 `src.model_platform.llm.create_llm_client()` 对未映射技能批量分类
  - `apply_categories_to_dictionary(dict_path: str) -> dict`：为词典中每个技能增加 `category` 字段
- [ ] 新增 `src/tests/test_skill_category_mapper.py`，覆盖：映射命中、映射未命中、LLM 分类 mock、category 值域校验。

### Task 2: Apply Categories to Flat Skill Dictionary

**Files:**
- Modify: `dicts/flat_skill_dictionary.json`
- Create: `output/skill_extraction/reports/category_mapping_report.json`

- [ ] 运行 `apply_categories_to_dictionary()`，为 1907 个技能增加 `category` 字段。
- [ ] 对 `map_skill_type()` 返回 `None` 的技能（约 200 个），调用 `classify_batch_by_llm()` 补全。
- [ ] 生成映射报告 `output/skill_extraction/reports/category_mapping_report.json`，包含：映射覆盖率、各 category 数量分布、LLM 分类的技能列表。
- [ ] 人工抽检 LLM 结果（抽样 5%），确认准确率 >95%。
- [ ] 确认所有 1907 个技能的 `category` 值均在 8 类枚举范围内。

### Task 3: Update FlatHardSkillMatcher Output with Category

**Files:**
- Modify: `src/skill_extraction/match_flat_skills_to_duckdb.py`
- Test: `src/tests/test_skill_category_mapper.py`（扩展）

- [ ] 在 `FlatHardSkillMatcher` 的 `match_text()` 返回结果中增加 `category` 字段。
- [ ] 匹配时从词典中读取每个技能的 `category`，附带到输出结果中。
- [ ] 未命中词典的候选不附带 `category`（值为 `None`）。
- [ ] 扩展测试，验证 `match_text()` 输出包含 `category` 且值正确。

### Task 4: Extract Soft Skill Seeds from Annotations

**Files:**
- Create: `src/skill_extraction/soft_skill_seed_extractor.py`
- Test: `src/tests/test_soft_skill_extractor.py`

- [ ] 实现 `src/skill_extraction/soft_skill_seed_extractor.py`：
  - `extract_soft_skill_seeds() -> dict[str, list[str]]`：从 `annotations.label_studio_annotations_v2` 的 `soft_skill` 字段提取高频标注词，按大五维度分组。
  - 通过 `config.paths.get_project_paths().pg_connection_params` 获取数据库连接。
  - 统计每个 `soft_skill` 值的出现频次，取频次 >=3 的作为种子词。
  - 将种子词映射到大五维度（基于关键词匹配：如含"创新"→ `openness`，含"沟通"→ `extraversion`）。
- [ ] 新增 `src/tests/test_soft_skill_extractor.py`，覆盖：维度映射、频次过滤、空值处理。

### Task 5: Build Soft Skill Dictionary

**Files:**
- Create: `dicts/soft_skill_dictionary.json`
- Create: `src/skill_extraction/soft_skill_dictionary_builder.py`
- Test: `src/tests/test_soft_skill_extractor.py`（扩展）

- [ ] 实现 `src/skill_extraction/soft_skill_dictionary_builder.py`：
  - `build_soft_skill_dictionary(seeds: dict, llm_client) -> dict`：基于种子词，用 LLM 扩展每个维度的变体词。
  - 输出格式：`{"schema_version": 1, "dimensions": {"openness": {"name": "开放性", "skills": [...]}, ...}}`。
  - 每个 skill 条目格式：`{"name": "...", "aliases": [...], "dimension": "openness"}`。
  - 通过 `src.model_platform.llm.create_llm_client()` 创建 LLM 客户端。
- [ ] 生成 `dicts/soft_skill_dictionary.json`，确认每个维度至少 10 个种子词。
- [ ] 扩展测试，验证词典格式和维度覆盖。

### Task 6: Implement Soft Skill Matcher

**Files:**
- Create: `src/skill_extraction/soft_skill_matcher.py`
- Test: `src/tests/test_soft_skill_matcher.py`

- [ ] 实现 `src/skill_extraction/soft_skill_matcher.py`：
  - `SoftSkillMatcher` 类，加载 `dicts/soft_skill_dictionary.json`。
  - `match_text(text: str) -> list[dict]`：对输入文本做关键词匹配，返回 `[{"name": "沟通能力", "dimension": "extraversion", "confidence": 0.9, "source": "dict_match"}]`。
  - 支持同义词映射（从词典的 `aliases` 字段读取）。
  - 匹配结果不与硬技能词典冲突（检查 `dicts/flat_skill_dictionary.json` 中是否存在同名技能）。
  - 复用 `dicts/blacklist_soft_skills.txt` 过滤非技能词。
- [ ] 新增 `src/tests/test_soft_skill_matcher.py`，覆盖：精确匹配、同义词匹配、硬技能冲突过滤、黑名单过滤。

### Task 7: Implement Soft Skill LLM Validator

**Files:**
- Create: `src/skill_extraction/soft_skill_llm_validator.py`
- Test: `src/tests/test_soft_skill_matcher.py`（扩展）

- [ ] 实现 `src/skill_extraction/soft_skill_llm_validator.py`：
  - `validate_soft_skills(candidates: list[dict], context_text: str, llm_client) -> list[dict]`：对候选软技能做 LLM 二次验证。
  - 验证内容：该词在当前上下文中确实是软技能（而非硬技能或职责描述）、分类是否正确。
  - 通过 `src.model_platform.llm.create_llm_client()` 创建客户端。
  - LLM 调用失败时降级为仅词典结果，标记 `confidence=0.5`。
  - 输出格式与 `match_text()` 一致，`source` 字段更新为 `"dict_match+llm_confirm"`。
- [ ] 扩展测试，验证 LLM 验证逻辑（mock LLM 响应）。

### Task 8: Create Unified V3 Pipeline Entry Point

**Files:**
- Create: `src/skill_extraction/v3_pipeline.py`
- Test: `src/tests/test_v3_pipeline.py`

- [ ] 实现 `src/skill_extraction/v3_pipeline.py`：
  - `V3Pipeline` 类，整合硬技能匹配（V2 扩展）和软技能匹配（新建）。
  - `run(source_table: str = "public.job_description_parsed") -> None`：从源数据表读取岗位描述，运行双管线，合并输出。
  - 硬技能：调用 `FlatHardSkillMatcher.match_text()`，附带 `category`。
  - 软技能：调用 `SoftSkillMatcher.match_text()` + `SoftSkillLLMValidator.validate_soft_skills()`。
  - 同一技能同时命中硬技能和软技能时，归类为硬技能。
  - 结果写入内存中的结构化数据（后续 Task 10 写入 PostgreSQL）。
  - 通过 `config.paths.get_project_paths()` 获取所有路径，不硬编码。
- [ ] CLI 入口：`python -m src.skill_extraction.v3_pipeline --help`。
- [ ] 新增 `src/tests/test_v3_pipeline.py`，覆盖：双管线集成、硬技能优先规则、空值处理。

### Task 9: Create Evaluation Script

**Files:**
- Create: `src/skill_extraction/eval_v3.py`
- Test: `src/tests/test_v3_pipeline.py`（扩展）

- [ ] 实现 `src/skill_extraction/eval_v3.py`：
  - 加载回归测试数据集（硬技能）和标注数据（软技能）。
  - 硬技能评估：复用 `regression_eval.py` 的 precision/recall/F1 逻辑，扩展分类准确率指标。
  - 软技能评估：从 `annotations.label_studio_annotations_v2` 中抽取有 `soft_skill` 标注的样本，计算覆盖率、分类准确率、精确率。
  - CLI 参数：`--fail-under-precision 0.85`、`--fail-under-f1 0.80`。
  - 输出报告到 `output/skill_extraction/reports/v3_eval/`。
  - 通过 `config.paths.get_project_paths()` 获取路径。
- [ ] 扩展测试，验证评估指标计算逻辑。

### Task 10: Create PostgreSQL Output Table and Writer

**Files:**
- Create: `src/skill_extraction/v3_result_writer.py`
- Modify: `config/database.yaml`
- Test: `src/tests/test_v3_pipeline.py`（扩展）

- [ ] 查阅 `Employ26-database.md` 确认无 `skill_extraction_v3_results` 同名表。
- [ ] 在 `config/database.yaml` 中添加 `skill_extraction_v3_results` 表配置。
- [ ] 实现 `src/skill_extraction/v3_result_writer.py`：
  - `create_v3_results_table() -> None`：创建 `public.skill_extraction_v3_results` 表（含索引）。
  - `write_v3_results(results: list[dict]) -> None`：批量写入结果，使用 `recruitment_record_id` 唯一键 upsert。
  - 通过 `config.paths.get_project_paths().pg_connection_params` 获取数据库连接。
- [ ] 在 `V3Pipeline.run()` 中集成写入逻辑。
- [ ] 扩展测试，验证表创建和写入（mock PostgreSQL）。

### Task 11: Verify and Document

**Files:**
- Modify: `src/skill_extraction/README.md`
- Modify: `Employ26-database.md`

- [ ] 运行 `python -m compileall -q src`，确认编译通过。
- [ ] 运行 `pytest src/tests/ -v`，确认所有测试通过（含新增测试和 V2 回归测试）。
- [ ] 运行 `python -m src.skill_extraction.eval_v3`，确认评估达标：
  - 硬技能覆盖率 >90%、准确率 >85%、分类准确率 >80%
  - 软技能覆盖率 >80%、分类准确率 >75%
- [ ] 更新 `src/skill_extraction/README.md`，补充 V3 管线说明和运行命令。
- [ ] 更新 `Employ26-database.md`，补充 `skill_extraction_v3_results` 表文档。
- [ ] 汇总剩余问题和后续改进方向。
