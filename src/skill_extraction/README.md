# Skill Extraction

`src/skill_extraction` 当前分成三套入口：

- `v1`
  历史版，按职业细类分层维护技能词典，相关模块已收拢到 `history/`
- `v2`
  前一版主流程，直接构造平面化职业硬技能词典，并支持自动评测、自动标注和二阶段过滤
- `v3`
  当前主流程，在 v2 基础上新增硬技能 8 类分类、软技能（大五人格）词典匹配与 LLM 验证，双管线并行输出结构化结果

## 目录说明

注意：`src/skill_extraction` 的”技能词典”与 `src/job_title_parsing/occupation_dictionary_pipeline.py` 的”职业词典”是两条不同链路。
- 技能词典：提取 JD 中的技能项
- 职业词典：岗位名称标准化 / canonical occupation / alias 映射

### v2 模块

- `pipeline_v1.py` — `v1` 入口
- `pipeline_v2.py` — `v2` 平面词典构造入口
- `occupation_skill_pipeline.py` — `v2` 主实现
- `match_flat_skills_to_duckdb.py` — 平面词典匹配、LLM 抽检、上下文判别接入；v3 中复用 `FlatHardSkillMatcher` 做硬技能匹配
- `regression_eval.py` — v2 回归评测
- `llm_label_regression_dataset.py` — 自动生成回归集
- `llm_label_context_dataset.py` — 自动生成上下文判别训练集
- `context_classifier.py` — 多分类上下文判别器
- `DESIGN_v2.md` — v2 详细设计文档

### v3 新增模块

- `v3_pipeline.py` — V3 统一管线入口，整合硬技能和软技能双管线
- `eval_v3.py` — V3 评估脚本，同时评估硬技能精度/召回/分类和软技能覆盖率/分类
- `v3_result_writer.py` — PostgreSQL 结果写入模块，目标表 `public.skill_extraction_v3_results`
- `skill_category_mapper.py` — 硬技能 8 类分类映射器（规则 + 启发式 + LLM 兜底）
- `soft_skill_seed_extractor.py` — 从标注数据提取软技能种子词并按大五人格维度分组
- `soft_skill_dictionary_builder.py` — 基于种子词用 LLM 扩展构建软技能词典
- `soft_skill_matcher.py` — 软技能关键词匹配器
- `soft_skill_llm_validator.py` — 软技能 LLM 上下文二次验证模块

## v1 运行方式

```bash
python -m src.skill_extraction.pipeline_v1 prepare
python -m src.skill_extraction.pipeline_v1 iterate
python -m src.skill_extraction.pipeline_v1 status
```

`v1` 适合需要保留“职业细类 -> 技能词典”分层结构的场景。

## v2 运行流程

### 1. 构造平面词典

```bash
python -m src.skill_extraction.pipeline_v2
```

### 2. 自动生成回归集（本地 LLM 主抽取 + GPT 高精度终审）

```bash
python -m src.skill_extraction.llm_label_regression_dataset ^
  --sample-size 400 ^
  --num-votes 3 ^
  --use-openai-final-review ^
  --openai-review-model openai/gpt-5.4-mini
```

### 3. 自动生成上下文训练集

```bash
python -m src.skill_extraction.llm_label_context_dataset ^
  --regression-dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --num-votes 3
```

### 4. 训练上下文判别器

```bash
python -m src.skill_extraction.context_classifier train ^
  --dataset output/skill_extraction/context_classifier/context_dataset_llm.jsonl ^
  --output-dir output/skill_extraction/context_classifier/model
```

### 5. 运行回归评测

```bash
python -m src.skill_extraction.regression_eval ^
  --dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --fail-under-precision 0.90 ^
  --fail-under-f1 0.80
```

### 6. 启用二阶段过滤做匹配

```bash
python -m src.skill_extraction.match_flat_skills_to_duckdb match ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --context-classifier-model output/skill_extraction/context_classifier/model ^
  --context-threshold 0.80
```

## v3 运行流程

### 1. 构建软技能词典（可选，已有现成词典可跳过）

```bash
# 从标注数据提取种子词
python -m src.skill_extraction.soft_skill_seed_extractor

# 用 LLM 扩展构建完整词典
python -m src.skill_extraction.soft_skill_dictionary_builder
```

### 2. 运行 V3 管线

```bash
# 从 PostgreSQL 读取岗位描述并运行双管线
python -m src.skill_extraction.v3_pipeline run

# 启用 LLM 软技能验证（需要 vLLM 服务可用）
python -m src.skill_extraction.v3_pipeline run --use-llm

# 从 JSON 文件读取数据
python -m src.skill_extraction.v3_pipeline process input.json --output results.json
```

### 3. 运行评估

```bash
python -m src.skill_extraction.eval_v3
python -m src.skill_extraction.eval_v3 --fail-under-precision 0.85 --fail-under-f1 0.80
```

### 4. 为技能词典补充 category 字段（一次性）

```bash
python -m src.skill_extraction.skill_category_mapper
```

## v3 架构说明

### 双管线设计

V3 在 v2 硬技能匹配的基础上新增软技能管线，两条管线并行执行：

1. **硬技能管线**：复用 `FlatHardSkillMatcher` 进行平面词典匹配，结果附加 8 类分类标签
2. **软技能管线**：`SoftSkillMatcher` 关键词匹配 -> `SoftSkillLLMValidator` 上下文验证
3. **合并去重**：同一技能名命中硬技能和软技能时，归类为硬技能（硬技能优先规则）

### 硬技能 8 类分类

所有硬技能词典项通过 `skill_category_mapper` 映射到以下 8 个标准类别：

| 类别 | 说明 | 示例 |
| --- | --- | --- |
| `programming_language` | 编程语言 | Python, Java, C++ |
| `framework` | 框架 | Spring, React, PyTorch |
| `database` | 数据库 | MySQL, Redis, MongoDB |
| `tool` | 工具软件 | Git, Docker, Jenkins |
| `office` | 办公软件 | Excel, PPT, Word |
| `equipment` | 设备/仪器 | 示波器, PLC, 万用表 |
| `process` | 工艺方法 | 焊接, 注塑, SMT |
| `certification` | 证书/资质 | CPA, PMP, 注册会计师 |

### 软技能大五人格分类

软技能按大五人格模型分为 5 个维度：

| 维度 | 中文名 | 典型技能 |
| --- | --- | --- |
| `openness` | 开放性 | 创新、学习能力、适应力 |
| `conscientiousness` | 尽责性 | 责任心、执行力、细心 |
| `extraversion` | 外向性 | 沟通能力、团队协作、领导力 |
| `agreeableness` | 宜人性 | 同理心、服务意识、合作 |
| `neuroticism` | 情绪稳定性 | 抗压能力、情绪管理、冷静 |

### 数据词典

- `dicts/flat_skill_dictionary.json` — 平面化硬技能词典（1907 条，含 `category` 字段）
- `dicts/soft_skill_dictionary.json` — 软技能词典（大五人格维度结构）
- `dicts/skill_category_rules.json` — 硬技能类别映射规则

### 结果输出

- **PostgreSQL**：`public.skill_extraction_v3_results` 表
- **JSON 文件**：通过 `--output` 参数指定路径
- **评估报告**：`output/skill_extraction/reports/v3_eval/`

## 当前 v2 的原理

### 词典构造

1. 读取 DuckDB 中已完成职业匹配的岗位文本
2. 按职业中类采样
3. 用本地 LLM 抽取硬技能并构造平面词典

### 词典匹配

1. 第一阶段做高速词典召回
2. 第二阶段用多分类上下文判别器过滤误报
3. 最终把保留下来的技能写回 DuckDB

### 自动训练数据

1. 用 LLM 从 JD 自动生成 `gold_skills`
2. 用 LLM 对词典召回候选打多分类标签
3. 用这些自动标注数据训练上下文判别器

## 标签定义

- `valid_hard_skill`
  候选在当前文本中是有效硬技能
- `too_generic`
  候选过于泛化，不适合作为技能词典项
- `wrong_alias_mapping`
  alias 命中了文本，但语义不属于当前技能
- `not_skill`
  命中内容不是硬技能

## 配置来源

统一读取：

- `config/database.yaml`
- 本地 `.env.local`（LLM API 配置，不提交 git）

当前 LLM 使用策略：

1. 技能提取主流程：优先使用本地 GPU + vLLM + 本地模型目录，目的是降低 token/API 成本
2. 高精度终审/难例判别：使用远端 GPT 模型（默认 `openai/gpt-5.4-mini`，必要时升级）
3. 默认本地模型选择顺序：
   - `models/hf/Qwen2.5-14B-Instruct`
   - `models/hf/DeepSeek-R1-Distill-Qwen-14B`
   - `models/hf/Qwen2.5-7B-Instruct`
   - `config/database.yaml` 中的 `LLM_model_path`

可查看当前自动选中的本地模型：

```bash
python3 -m src.skill_extraction.pipeline_v2 --print-model-choice
```

## 职业词典试运行

小批次 pilot：

```bash
python3 -m src.job_title_parsing.occupation_dictionary_pipeline --pilot-size 10
```

仅生成评估与 review，不写回主词典：

```bash
python3 -m src.job_title_parsing.occupation_dictionary_pipeline --pilot-size 10 --dry-run
```

输出位置：
- review: `output/occupation_dictionary/review/`
- report: `output/occupation_dictionary/reports/`
- state: `output/occupation_dictionary/iteration_state.json`
- cache: `output/occupation_dictionary/cache/classification_cache.json`

- `v1` 的相关功能模块已经整体下沉到 `history/`
- `v2` 的词典构造主逻辑保持不变，这次新增的是自动标注、回归评测和多分类上下文过滤链路
- 更完整的设计说明见 `src/skill_extraction/DESIGN_v2.md`

## Standardized Dictionary Workflow

- SOP: `src/skill_extraction/SKILL_DICTIONARY_WORKFLOW_SOP.md`
- Controller: `src/skill_extraction/skill_dictionary_workflow.py`

Recommended commands:

```bash
python -m src.skill_extraction.skill_dictionary_workflow baseline
python -m src.skill_extraction.skill_dictionary_workflow run
python -m src.skill_extraction.skill_dictionary_workflow status
```

## 软技能迭代框架

词典版本化 + 评估注册表 + CLI。每次改进创建新版本词典，运行评估，对比指标。

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

# 列出所有评估记录
python -m src.skill_extraction.eval_cli list
```
