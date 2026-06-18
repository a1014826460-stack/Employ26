# 广东省招聘数据 NLP 分析项目

> 最后更新: 2026-06-17

基于广东省三大招聘网站 2022-2025 年招聘数据的 NLP 分析项目，聚焦岗位名称匹配、技能抽取、职业知识检索与相关分析流程。

## 文档分工

- `README.md`: 给人看，负责项目介绍、启动方式、目录导航和常用入口
- `CLAUDE.md`: 给代码代理看，负责修改规范、架构约束、验证要求和协作边界

如果你希望 Codex 严格遵守仓库规范，建议在任务里明确写一句：  
`先阅读根目录 CLAUDE.md，并按其中规范执行。`

## 项目概览

- 数据来源: 智联招聘、猎聘网、前程无忧
- 时间范围: 2022-2025
- 数据规模: 原始数据约 5GB
- Python 版本: 3.10+
- 正式数据库: PostgreSQL

当前活跃工作主要集中在：

- `src/job_title_parsing/` 岗位名称匹配
- `src/skill_extraction/` 技能抽取
- `src/rag/` 本地职业知识库检索
- `src/llm/` LLM 基础设施
- `src/bge/` BGE 微调与迭代流水线

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 检查数据库配置

默认配置位于 `config/database.yaml`，当前正式 PostgreSQL 数据库为 `Employ26`。通常不需要额外设置环境变量；只有本机连接参数不同的时候再覆盖：

```bash
# PostgreSQL
export EMPLOYDATA_PG_HOST=localhost
export EMPLOYDATA_PG_PORT=5432
export EMPLOYDATA_PG_DBNAME=Employ26
export EMPLOYDATA_PG_USER=postgres
export EMPLOYDATA_PG_PASSWORD=your_password

# 模型路径（可选）
export EMPLOYDATA_BGE_MODEL_PATH=models/bge-base-zh-v1.5
export EMPLOYDATA_QWEN_MODEL_PATH=models/Qwen3-8B
```

Windows PowerShell 可使用：

```powershell
$env:EMPLOYDATA_PG_HOST="localhost"
$env:EMPLOYDATA_PG_PORT="5432"
$env:EMPLOYDATA_PG_DBNAME="Employ26"
$env:EMPLOYDATA_PG_USER="postgres"
$env:EMPLOYDATA_PG_PASSWORD="your_password"
```

### 3. 运行常用入口

```bash
# 岗位匹配
python -m src.job_title_parsing.cli match --progress

# 技能抽取
python -m src.skill_extraction.cli --help

# RAG 检索
python -m src.rag.cli query --title "Java开发工程师" --requirements "Spring..."

# BGE 流水线
python -m src.bge.step_01_deduplicate
```

## 项目结构

```text
Employ26/
├── src/                        # 活跃源代码
│   ├── job_title_parsing/      # 岗位名称匹配
│   ├── skill_extraction/       # 技能抽取
│   ├── rag/                    # RAG 检索
│   ├── llm/                    # LLM 基础设施
│   ├── bge/                    # BGE 微调与迭代
│   ├── bert/                   # BERT NER
│   ├── data_pipeline/          # 活跃数据准备流水线
│   ├── analysis/               # 报表、汇总与交付
│   ├── db/                     # PostgreSQL 访问层
│   ├── visualization/          # 可视化
│   └── tests/                  # 测试
├── config/                     # 配置与路径管理
├── dicts/                      # 词典、黑名单、别名表
├── logs/                       # 运行日志与排错日志
├── output/                     # 输出目录
├── archive/                    # 历史代码与实验归档
├── README.md
└── CLAUDE.md
```

## 数据与输出约定

- PostgreSQL 是唯一正式数据库
- 当前正式数据库名为 `Employ26`
- 当前业务 schema 为 `51job`、`Liepin`、`Zhilian`、`annotations`、`public`
- 招聘源数据位于 `51job` / `Liepin` / `Zhilian` schema，各自包含 `raw_data`、`cleaned_data`、`sample`
- 标注数据位于 `annotations` schema，常用联查视图为 `annotations.v_label_studio_task_annotations_v2`
- 职业词典、匹配结果、训练特征等公共处理表主要位于 `public`
- 数据库结构、字段、索引与引用规范详见 `Employ26-database.md`
- 日志文件统一存放到 `logs/`
- `output/` 统一存放中间结果、最终产物、报告、图表和归档副本
- 报告类文本和图表默认输出到 `output/reports/`
- 数据文件约 5GB，不提交到 git
- 过期输出优先归档到 `output/archive/YYYY-MM/`，并在文件名中追加 `_archived`

配置、模型路径和数据库连接参数统一通过 `config/paths.py` 管理，并支持 `EMPLOYDATA_*` 环境变量覆盖。

## 常见入口

### 岗位名称匹配

```bash
python -m src.job_title_parsing.cli preprocess-catalog
python -m src.job_title_parsing.cli build-hierarchy-dict
python -m src.job_title_parsing.cli match --progress
python -m src.job_title_parsing.cli evaluate
```

### 技能抽取

```bash
python -m src.skill_extraction.cli --help
```

### RAG

```bash
python -m src.rag.cli build
python -m src.rag.cli query --title "Java开发工程师" --requirements "..."
python -m src.rag.cli judge --title "..." --requirements "..." --candidates-json "[...]"
```

### 第二轮 DeepSeek 检验性标注

```bash
python -m src.anno_analysis.deepseek_round2_relabel --dry-run
python -m src.anno_analysis.deepseek_round2_relabel --workers 2
```

输出会进入 `output/deepseek_relabel/round2/`，运行日志写入 `logs/`。

### 历史完整流程

```bash
python archive/process/run_all_analysis.py
```

说明：该入口已归档，当前新增功能默认不应建立在 `archive/` 代码上。

旧版 `output/nlp_processed` / DuckDB 辅助链路已归档，当前新增功能默认不应继续建立在
已归档的历史脚本之上。

## 开发协作

人类协作者重点看这里：

- 想了解项目和入口，看本文件
- 想约束 Codex、Claude Code、Cursor Agent 的修改行为，看 `CLAUDE.md`
- 想让代理严格按规范工作，在任务描述里明确引用 `CLAUDE.md`

## 验证命令

```bash
python -m compileall -q src
pytest src/tests/ -v
black --check src tests
flake8 src tests
```

## 补充说明

- 旧流程、旧实验、历史版本统一归档到 `archive/`
- 代码中如出现历史遗留命名，不代表仍是当前推荐架构
- 更细的代理执行规范、代码风格和修改边界，请查看 `CLAUDE.md`
