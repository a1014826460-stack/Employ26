# Skill Extraction

`src/skill_extraction` 是技能抽取的活跃实现目录。当前正式主线是 V3
PostgreSQL 双管线：硬技能词典匹配 + 软技能词典/LLM 验证，结果写入
`public.skill_extraction_v3_results`。

历史 DuckDB 脚本和 v1 实验流程不再作为活跃入口；如需查看旧实现，请到
`src/archive/skill_extraction_history/`。

## 目录结构

- `core/`：共享工具、词典路径、匹配归一化函数。
- `hard/`：硬技能词典加载、匹配、8 类分类、上下文分类器、PostgreSQL 调试匹配。
- `soft/`：软技能种子提取、词典构建、词典匹配、LLM 验证。
- `pipeline/`：V3 数据源、双管线处理、结果类型、PostgreSQL 写入。
- `evaluation/`：V3 评估、回归评估、评估注册表、版本对比。
- `dictionary/`：平面硬技能词典构建和保守迭代工具。
- `labeling/`：LLM 自动生成回归集和上下文训练集。
- `cli.py`：统一命令入口。

## 推荐入口

```bash
python -m src.skill_extraction.cli --help
```

常用命令：

```bash
python -m src.skill_extraction.cli pipeline run
python -m src.skill_extraction.cli pipeline process input.json --output results.json
python -m src.skill_extraction.cli eval run
python -m src.skill_extraction.cli eval compare v1 v2
python -m src.skill_extraction.cli hard match-pg
python -m src.skill_extraction.cli soft build-dictionary
python -m src.skill_extraction.cli dict iterate
```

## 数据流

1. 默认读取 `public.job_description_parsed`。
2. 对同一 `source_table + source_row_number` 取 `parsed_at DESC` 的最新解析结果。
3. 硬技能管线使用 `hard.matcher.FlatHardSkillMatcher`。
4. 软技能管线使用 `soft.matcher.SoftSkillMatcher`，可选接入 LLM 验证。
5. 同一技能同时命中硬技能和软技能时，硬技能优先。
6. 结果写入 `public.skill_extraction_v3_results`，以 `recruitment_record_id` upsert。

## 配置与词典

- PostgreSQL 连接、表名和模型路径统一通过 `config.paths.get_project_paths()` 和
  `config/database.yaml` 读取。
- 硬技能词典：`dicts/flat_skill_dictionary.json`
- 硬技能分类规则：`dicts/skill_category_rules.json`
- 软技能当前版本：`dicts/soft_skill/current.txt`
- 软技能版本词典：`dicts/soft_skill/v*.json`

不要在新代码中引入 DuckDB 作为正式存储，也不要手写 PostgreSQL 连接字符串。

## 维护约定

- 新公开函数、类和 CLI 行为必须有类型提示和中文 Google 风格 docstring。
- 新流程默认使用 PostgreSQL；临时测试数据写入数据库时使用 `test` schema。
- 旧设计文档 `DESIGN_v2.md` 和 `SKILL_DICTIONARY_WORKFLOW_SOP.md` 仅作历史参考，
  不再作为新开发约束。
