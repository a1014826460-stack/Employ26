# CLAUDE.md

> 最后更新: 2026-06-13
> 适用对象: Codex、Claude Code、Cursor Agent 等代码代理

本文件是本仓库的代理协作规范。它不负责介绍项目背景，也不承担新成员 onboarding；这些内容放在 `README.md`。  
代理在开始实现、重构、修复、评审前，应优先遵守本文件中的硬约束。

## 1. 文档分工

- `README.md` 面向人类读者，负责说明项目目标、目录结构、快速开始和功能入口
- `CLAUDE.md` 面向代码代理，负责说明修改边界、架构约束、验证要求和执行规范
- 两者出现冲突时，以用户当次指令为最高优先级；若用户未特别说明，代理应优先遵守本文件

## 2. 代理工作目标

代理在本仓库中的默认目标是：

- 在不破坏现有数据流程的前提下完成用户指定任务
- 优先修改活跃代码，避免无关重构
- 保持 PostgreSQL、路径管理、词典管理等既有架构约束稳定
- 输出尽量可验证、可回滚、可维护的改动

## 3. 项目快照

- 项目主题: 广东省招聘数据 NLP 分析
- 数据规模: 原始数据约 5GB
- Python 版本: 3.10+
- 当前活跃方向:
  - `src/job_title_parsing/` 岗位名称匹配
  - `src/data_pipeline/` 数据准备、描述切分与结果增强
  - `src/skill_extraction/` 技能抽取
  - `src/rag/` 本地职业知识库检索
  - `src/llm/` 与 `src/utils/llm_router.py` LLM 基础设施
- 历史流程与实验代码放在 `archive/`，默认不作为新功能入口

## 4. 修改优先级与边界

### 优先修改的位置

- 当前任务直接涉及的活跃模块
- 与任务相关的测试文件
- 与任务强相关的配置文件

### 默认不要动的位置

- `archive/` 下的历史代码，除非用户明确要求
- 与当前需求无关的大范围格式化、重命名、迁移
- 数据库表结构和全局路径约定，除非任务明确要求

### 遇到下列情况要先停一下再决定

- 需要改动多个子系统才能完成任务
- 发现用户已有未提交改动且与当前修改冲突
- 需要引入新依赖、改表结构、改公共接口

## 5. 必须遵守的架构约束

### 数据存储

- PostgreSQL 是唯一正式数据库
- 除报告类文本外，原始数据、中间结果、最终结构化产出均写入 PostgreSQL
- 禁止为新流程引入 DuckDB 作为正式存储
- 如果代码中存在历史命名遗留，例如带 `duckdb` 的旧文件名，不代表新逻辑可以继续依赖 DuckDB

### 路径与配置

- 禁止硬编码绝对路径，如 `D:\...`
- 所有项目路径、模型路径、PostgreSQL 连接参数统一通过 `config/paths.py` 获取
- 模型运行后端、WSL vLLM、PyTorch device 策略统一通过 `config/model_runtime.yaml` 与 `src/model_platform/` 获取
- 配置读取优先级为：显式参数 > `EMPLOYDATA_*` 环境变量 > `config/database.yaml` > 代码内兜底默认值
- 不要在上层脚本中重复拼接路径、手写连接字符串或猜测表名
- 需要 SQLAlchemy 连接 URL 时，优先使用 `paths.pg_sqlalchemy_url()`，不要自行拼接密码、端口和数据库名

推荐用法：

```python
from config.paths import get_project_paths

paths = get_project_paths()
pg_params = paths.pg_connection_params
pg_url = paths.pg_sqlalchemy_url()
bge_model_path = paths.bge_model_path
tasks_table = paths.get_table_name("annotations", "tasks_v2")
```

### 模型平台

- 新增或修改生成式 LLM 调用时，默认使用 `src.model_platform.llm.create_llm_client()`，不要在业务脚本里直接加载 Qwen/DeepSeek 权重
- 生成式 LLM 默认后端是 WSL vLLM OpenAI-compatible API，配置见 `config/model_runtime.yaml` 与 `config/vllm.toml`
- BGE embedding、BGE 微调、BERT 训练/预测仍使用 PyTorch / Transformers / SentenceTransformers，不强制走 vLLM
- BGE embedding 统一通过 `src.model_platform.embeddings.get_embedding_model()` 或 `config.paths.get_project_paths().bge_model_path`
- PyTorch device 与模型路径解析优先使用 `src.model_platform.torch_runtime`
- 不要把 WSL vLLM 当成所有 PyTorch 工作负载的统一运行环境；它只负责生成式 LLM 推理

### 数据组织

- 当前正式 PostgreSQL 数据库名是 `Employ26`，不要写成小写 `employ26`
- 当前业务 schema 为 `51job`、`Liepin`、`Zhilian`、`annotations`、`public`
- 招聘源数据位于 `51job` / `Liepin` / `Zhilian` 三组 schema 中，每组包含 `raw_data`、`cleaned_data`、`sample`
- Label Studio 与 DeepSeek 标注数据位于 `annotations` schema，优先使用 `annotations.v_label_studio_task_annotations_v2` 做日常联查
- 职业词典、匹配结果、训练特征等公共处理表主要位于 `public`
- benchmark、压测、临时验证或其他测试数据需要写入 PostgreSQL 时，必须使用 `test` schema，禁止写入 `public` schema
- 根目录 `Employ26-database.md` 是数据库结构、字段、索引和引用规范的权威文档；改数据库或写 SQL 前必须先查它
- `config/database.yaml` 只允许写真实存在的 PostgreSQL 表；如果 PG 没有对应表，宁可不配置，也不要写封存 DuckDB 表名
- 报告类文本输出到 `output/reports/`
- 词典、黑名单、别名表等词汇资源放在 `dicts/`，不要硬编码进 Python

### 导入与运行方式

- 禁止使用 `sys.path.insert()` 或 `sys.path.append()` 修补导入
- 从项目根目录使用 `python -m src.xxx` 运行
- 包内优先使用相对导入
- 禁止 `from module import *`

## 6. 代码风格规范

### 基本规则

- 变量、函数、模块名使用英文
- 注释使用中文
- docstring 使用中文
- 新增公开接口必须带类型提示
- 新增 `.py` 文件应包含模块级 docstring

### docstring 要求

公开类、函数、方法使用 Google 风格 docstring，至少写清：

- 功能做什么
- 参数含义
- 返回值
- 必要时说明异常或副作用

### 风格工具

- `black`
- `isort`
- `flake8`
- `pytest`
- `python -m compileall -q src`

如果任务只改动局部文件，优先验证局部；如果改动涉及公共逻辑，再扩大验证范围。

## 7. 活跃模块速记

### 岗位名称匹配

- 目录: `src/job_title_parsing/`
- CLI 入口: `python -m src.job_title_parsing.cli`
- 典型任务: 职业大典预处理、层级词典构建、岗位匹配、结果评估

### 技能抽取

- 目录: `src/skill_extraction/`
- 主入口: `python -m src.skill_extraction.occupation_skill_pipeline`
- 统一配置入口: `load_skill_extraction_config()`
- 目标倾向: 高精度、低噪声，避免把软素质、职责套话、福利词误识别为技能

### 数据准备流水线

- 目录: `src/data_pipeline/`
- 典型任务: 岗位描述切分、职业整合、技能词典前置匹配准备
- 说明: 旧版 `output/nlp_processed` / DuckDB 辅助链路已归档，不再视为活跃实现

### RAG

- 目录: `src/rag/`
- CLI 入口: `python -m src.rag.cli`

### BGE 流水线

- 目录: `src/bge/`
- 采用 `step_01` 到 `step_07` 顺序命名

### 模型平台

- 目录: `src/model_platform/`
- 职责: 统一 LLM client、embedding 模型加载、PyTorch device 和模型路径解析

## 8. 代理执行流程

收到任务后，代理默认按下面顺序工作：

1. 先读与任务直接相关的代码和配置，不凭文件名猜实现
2. 优先复用已有模块、配置和数据流，不重复造入口
3. 仅修改完成任务所必需的文件
4. 改动后至少做编译、测试或最小可运行验证
5. 向用户说明改了什么、验证了什么、还有什么风险

## 9. 常用命令

### 活跃入口

```bash
python -m src.job_title_parsing.cli match --progress
python -m src.skill_extraction.occupation_skill_pipeline --help
python -m src.rag.cli query --title "Java开发工程师" --requirements "Spring..."
python -m src.bge.step_01_deduplicate
```

### 验证命令

```bash
python -m compileall -q src
pytest src/tests/ -v
black --check src tests
flake8 src tests
```

## 10. 提交与归档约定

- 提交信息优先使用 Conventional Commits
- 历史代码进入 `archive/`
- 新功能不要建立在 `archive/` 代码之上，除非先明确恢复为活跃代码

## 11. 给代理的简短提醒

- 不要把 `README.md` 当成强约束规范文件，它主要服务人类读者
- 不要因为仓库里有历史脚本就默认沿用旧实现
- 不要为了“顺手”做无关清理
- 先保证正确性和边界清晰，再考虑额外重构
