# `src/data_pipeline` 数据管道脚本说明

本目录负责把原始招聘数据整理成分析链路可以稳定复用的 PostgreSQL 中间层。当前主线是：

```text
招聘平台原始表
  -> public.recruitment_jobs_normalized
  -> public.job_description_parsed
  -> public.skill_extraction_requirement_matches
  -> src.analysis 统计报表
```

常用数据库配置来自 `config/database.yaml`，PostgreSQL 连接和表名解析由 `src.db` 下的模块负责。

## 推荐运行顺序

样本数据链路：

```bash
python -m src.data_pipeline.backfill_recruitment_jobs_normalized
python -m src.data_pipeline.description_parsing --input-source postgres --source-table '"51job".sample' '"Liepin".sample' '"Zhilian".sample' --write-postgres
python -m src.data_pipeline.requirement_match_prep
```

全量 cleaned_data 链路：

```bash
python -m src.data_pipeline.backfill_recruitment_jobs_normalized --full --workers 2 --chunk-size 25000
python -m src.data_pipeline.description_parsing --input-source postgres --source-table '"51job".cleaned_data' '"Liepin".cleaned_data' '"Zhilian".cleaned_data' --write-postgres
python -m src.data_pipeline.requirement_match_prep --parse-workers 4 --parse-batch-size 20000
```

试跑建议：

```bash
python -m src.data_pipeline.backfill_recruitment_jobs_normalized --dry-run
python -m src.data_pipeline.backfill_recruitment_jobs_normalized --limit-rows 1000 --max-chunks 1 --benchmark
python -m src.data_pipeline.description_parsing --input-source postgres --source-table '"51job".sample' --limit-rows 1000 --output-csv output/description_parsing_sample.csv
python -m src.data_pipeline.requirement_match_prep --limit-job-rows 1000
```

## 脚本索引

### `backfill_recruitment_jobs_normalized.py`

统一招聘规范层回填脚本。把三家招聘平台 source 表写入 `public.recruitment_jobs_normalized`，为后续描述解析、职业匹配和分析报表提供统一字段。

默认 source 表：

- `'"51job".sample'`
- `'"Liepin".sample'`
- `'"Zhilian".sample'`

传入 `--full` 后改用：

- `'"51job".cleaned_data'`
- `'"Liepin".cleaned_data'`
- `'"Zhilian".cleaned_data'`

使用方法：

```bash
python -m src.data_pipeline.backfill_recruitment_jobs_normalized
python -m src.data_pipeline.backfill_recruitment_jobs_normalized --full
python -m src.data_pipeline.backfill_recruitment_jobs_normalized --source-table '"51job".sample'
python -m src.data_pipeline.backfill_recruitment_jobs_normalized --dry-run
```

常用参数：

- `--normalized-table`：目标规范层表名，默认 `public.recruitment_jobs_normalized`。
- `--dry-run`：只输出缺口统计，不写数据库。
- `--all-source-rows`：默认只补缺口；传入后重扫 source 全表并 upsert。
- `--limit-rows`：每张 source 表最多读取多少行，适合试跑。
- `--workers`：SQL chunk 并发 worker 数，默认 `2`。
- `--chunk-size`：每个 SQL chunk 覆盖的 source 行数，默认 `25000`。
- `--max-chunks`：只执行前 N 个 chunk，适合压测。
- `--benchmark` / `--benchmark-json`：输出 chunk 压测日志或 JSON。
- `--resume-run-id`：复用指定 run_id 继续执行。
- `--retry-failed-chunks`：恢复运行时重试失败 chunk。
- `--python-batches`：使用旧的 Python DataFrame 分批写入路径。
- `--repair-source-table-values`：先修复历史错误的 `source_table` 值再回填。

主要写入：

- `public.recruitment_jobs_normalized`
- 回填运行状态表和 locator 表，默认表名由 `src.db.recruitment_jobs_normalized` 定义。

### `description_parsing.py`

岗位描述结构化切分脚本。将岗位描述清洗并切分为岗位职责、任职要求、福利待遇、其他信息等结构，并生成 RAG/职业匹配文本。

使用 CSV 输入：

```bash
python -m src.data_pipeline.description_parsing --input-source csv --input-csv data/jobs.csv --output-csv output/parsed_jobs.csv
```

使用 PostgreSQL 输入并写回解析结果表：

```bash
python -m src.data_pipeline.description_parsing --input-source postgres --source-table '"51job".sample' --write-postgres
python -m src.data_pipeline.description_parsing --input-source postgres --source-table '"51job".sample' '"Liepin".sample' '"Zhilian".sample' --write-postgres
```

常用参数：

- `--input-source csv|postgres`：输入来源，默认 `csv`。
- `--input-csv`：CSV 输入文件路径。
- `--output-csv`：可选，保存解析结果 CSV。
- `--desc-col`：岗位描述列名，默认 `岗位描述`。
- `--title-col`：岗位名称列名，默认 `岗位名称`。
- `--source-table`：PostgreSQL 输入源表名，可传多个。
- `--source-platform`：来源平台名，可与 `--source-table` 一一对应；不传则自动推断。
- `--target-table`：解析结果目标表，默认 `public.job_description_parsed`。
- `--write-postgres`：将解析结果写入 PostgreSQL。
- `--limit-rows`：调试时限制 PostgreSQL 读取行数。
- `--only-risk-rows`：仅重跑上一版本解析结果中的高风险行。
- `--risk-parser-version`：高风险行筛选使用的旧版本，默认 `latest`。
- `--parse-workers`：岗位描述切分并发数，默认 `32`。
- `--parse-batch-size`：岗位描述切分批大小，默认 `20000`。

主要输出列：

- `岗位描述_清洗`
- `岗位描述_切分JSON`
- `任职要求_items_text`
- `岗位职责_items_text`
- `unclassified_text`
- `sections_brief`
- `RAG匹配文本`
- `RAG匹配来源`

主要写入：

- `public.job_description_parsed`

### `requirement_match_prep.py`

技能词典流程预处理脚本。它会从 `public.recruitment_jobs_normalized` 读取招聘记录，调用 `description_parsing.parse_desc_df` 切分岗位描述，再用本地 BGE 职业匹配模型匹配职业细类，最后写入职业匹配结果表。

使用方法：

```bash
python -m src.data_pipeline.requirement_match_prep
python -m src.data_pipeline.requirement_match_prep --limit-job-rows 1000
python -m src.data_pipeline.requirement_match_prep --parse-workers 4 --parse-batch-size 20000 --top-k 5
```

常用参数：

- `--database-config`：数据库配置文件路径，默认 `config/database.yaml`。
- `--limit-job-rows`：仅用于调试，限制招聘记录读取行数。
- `--parse-workers`：岗位描述切分并发数，默认 `1`。
- `--parse-batch-size`：岗位描述切分批大小，默认 `2000`。
- `--top-k`：职业匹配保留候选数量，默认使用 skill extraction 配置。

前置要求：

- `public.recruitment_jobs_normalized` 已回填。
- 职业词典表可用。
- 本地 BGE 模型路径可用，当前代码说明中使用 `D:\model\bge-base-zh-finetuned`。

主要写入：

- `skill_extraction.requirement_match_table` 配置对应表，当前分析链路通常读取 `public.skill_extraction_requirement_matches`。

注意：该脚本写表时使用 `if_exists='replace'`，会替换目标匹配结果表。

### `occupation_detail_match_full.py`

全量职业细类识别脚本。它从 `public.recruitment_jobs_normalized` 分批读取招聘记录，使用当前最佳 `v1 + bge-large` finetuned 模型检索职业细类，底层保留 Top10 候选，默认结果字段仍取 Top1，并写入 `public.occupation_detail_matches`。

试跑 1000 行：

```bash
python -m src.data_pipeline.occupation_detail_match_full --limit-rows 1000 --batch-size 1000
```

正式断点续跑：

```bash
python -m src.data_pipeline.occupation_detail_match_full --batch-size 20000
```

强制重算覆盖已有结果：

```bash
python -m src.data_pipeline.occupation_detail_match_full --batch-size 20000 --no-resume
```

默认配置来自 `config/database.yaml`：

- `skill_extraction.occupation_detail_match_table`
- `skill_extraction.occupation_detail_model_path`
- `skill_extraction.occupation_detail_top_k`

说明：

- 该脚本不会覆盖 `public.skill_extraction_requirement_matches`。
- `public.occupation_detail_matches.occupation_code` / `occupation_title` 固定表示 Top1 最终输出。
- `public.occupation_detail_matches.top10_candidates` 保存 Top10 候选，用于人工复核、误差分析或后续 rerank。

### `occupation_integration.py`

历史 CSV 整合适配器。读取旧的 NLP 处理 CSV 和职业解析 CSV，生成带职业字段、月份、城市、行业标准化字段的 `output/integrated/*_整合_*.csv`。

使用方法：

```bash
python -m src.data_pipeline.occupation_integration
python -m src.data_pipeline.occupation_integration --sample
```

默认读取：

- 全量 NLP 目录：`output/nlp_processed_full`
- 样本 NLP 目录：`output/nlp_processed`
- 职业解析目录：`output/job_title_parsing`

主要输出：

- `output/integrated/*_整合_*.csv`

当前状态：保留为历史兼容。新版分析主链路已经改为直接读取 PostgreSQL 规范层和匹配结果层，不再依赖 `output/integrated`。

### `description_schema.py`

岗位描述解析配置模块，不建议单独运行。集中定义解析版本、默认表名、CSV 旧列名、PostgreSQL 列名映射、标题别名和正则片段。

核心常量：

- `PARSER_VERSION`
- `DEFAULT_PARSED_TABLE`
- `LEGACY_OUTPUT_COLUMNS`
- `LEGACY_TO_PG_COLUMN_MAP`
- `PG_COLUMN_ORDER`
- `TITLE_ALIASES`

典型调用方：

- `description_parsing.py`
- `text_cleaning.py`
- `src.db.job_description_parsed`

### `text_cleaning.py`

招聘文本清洗工具模块，不建议单独运行。负责 HTML 标签清理、乱码/零宽字符清理、条目边界清洗和岗位描述全文规范化。

主要函数：

- `strip_tags(text)`：移除 HTML 标签，并把段落类标签转换为换行。
- `remove_noise(text)`：移除零宽字符和明显噪声标点。
- `sanitize_item(text)`：清理单个切分条目。
- `normalize_text(text)`：清洗岗位描述全文，并为内联标题、编号列表补换行。

典型调用方：

- `description_parsing.py`

### `__init__.py`

包标记文件，无需直接运行。

## 与 `src/analysis` 的关系

分析目录默认依赖本目录生成的 PostgreSQL 中间层：

- `src.analysis.cli structured run` 依赖 `public.recruitment_jobs_normalized` 和职业匹配结果表。
- `src.analysis.cli requirements run` 依赖 `public.recruitment_jobs_normalized`、`public.job_description_parsed` 和 `analysis_lexicon` 当前 release。

如果结构化统计里职业匹配覆盖率偏低，优先检查：

- `backfill_recruitment_jobs_normalized.py` 是否已覆盖目标 source 表。
- `requirement_match_prep.py` 是否已成功写入职业匹配结果表。
- 配置中的 `skill_extraction.requirement_match_table` 是否与分析脚本读取的表一致。

如果 requirement text 报表样本数偏低，优先检查：

- `description_parsing.py` 是否已写入 `public.job_description_parsed`。
- `requirements_text` 是否非空。
- `analysis_lexicon` 是否存在唯一 `is_current = true` 的正式 release。
