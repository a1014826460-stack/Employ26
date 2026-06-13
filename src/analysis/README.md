# `src/analysis` 分析脚本说明

本目录负责从 PostgreSQL 规范层和中间事实层生成分析报表。当前建议不要直接回到原始招聘表做统计，而是优先使用统一 CLI 读取以下表：

- `public.recruitment_jobs_normalized`：招聘记录统一规范层。
- `public.skill_extraction_requirement_matches`：职业细类匹配结果层。
- `public.job_description_parsed`：岗位描述结构化切分结果。
- `public.requirement_constraint_facts`：requirement text 约束事实层，由分析链路写入。
- `analysis_lexicon` schema：requirement text 规则和词汇资源。

默认输出目录：

- 结构化统计：`output/reports/structured_analysis_{mm-dd}/`
- requirement text：`output/reports/req_analysis_{mm-dd}/`

## 推荐入口

优先使用统一 CLI：

```bash
python -m src.analysis.cli structured run --with-excel
python -m src.analysis.cli requirements run
```

如需指定输出目录：

```bash
python -m src.analysis.cli structured run --output-dir output/reports/structured_analysis_test
python -m src.analysis.cli requirements run --output-dir output/reports/req_analysis_test
```

结构化统计常用参数：

- `--with-excel`：在 CSV 和 Markdown 之后生成 Excel 汇总。
- `--skip-standardized`：跳过规范化汇总表生成。
- `--structured-workers 1`：改为串行运行结构化统计步骤，便于排错。
- `--with-legacy-copies`：额外生成历史中文文件名 CSV 副本。
- `--base-dir`：兼容旧脚本的项目根目录参数。

requirement text 常用参数：

- `--top-n`：Markdown 报告中展示的 Top N 数量，默认 `20`。
- `--min-group-size`：城市、行业、公司规模分组最小样本数，默认 `50`。
- `--min-monthly-group-size`：月度分组最小样本数，默认 `20`。
- `--extractor-version`：写入 `public.requirement_constraint_facts` 时使用的抽取器版本。

## 推荐运行顺序

结构化统计主链路：

```bash
python -m src.data_pipeline.backfill_recruitment_jobs_normalized
python -m src.data_pipeline.requirement_match_prep
python -m src.analysis.cli structured run --with-excel
```

requirement text 约束分析链路：

```bash
python -m src.db.analysis_lexicon --ensure-schema
python -m src.db.analysis_lexicon --bootstrap-v1 --version v2_curated_requirement_analysis
python -m src.db.requirement_constraint_facts --ensure-schema
python -m src.data_pipeline.backfill_recruitment_jobs_normalized
python -m src.data_pipeline.description_parsing --input-source postgres --source-table '"51job".sample' '"Liepin".sample' '"Zhilian".sample' --write-postgres
python -m src.analysis.cli requirements run
```

## 脚本索引

### `cli.py`

统一分析入口。负责调度结构化统计链路和 requirement text 链路。

使用方法：

```bash
python -m src.analysis.cli structured run
python -m src.analysis.cli structured run --with-excel --structured-workers 4
python -m src.analysis.cli requirements run
```

结构化统计会并发执行 `occupation_salary_analysis.py`、`education_distribution_analysis.py`、`industry_trend_analysis.py`、`structured_dimension_analysis.py`，再串行执行 `generate_standardized_tables.py` 和可选的 `generate_excel_summary.py`。

### `occupation_salary_analysis.py`

职业与职业类别薪资分析。读取 `public.recruitment_jobs_normalized` 和职业匹配结果，解析 `salary_raw`，输出薪资分布、月度趋势、学历交叉薪资和 HTML 图表。

使用方法：

```bash
python -m src.analysis.occupation_salary_analysis
```

主要输出：

- `职业类别薪资分析报告.md`
- `salary_by_occupation_category_month.csv`
- `salary_by_occupation_month.csv`
- `salary_by_education_occupation_category.csv`
- `salary_by_education_occupation.csv`
- `职业类别薪资分析图.html`

### `education_distribution_analysis.py`

学历需求分布分析。读取结构化统计主输入，标准化学历要求，按职业类别、职业、年度、月度统计学历分布。

使用方法：

```bash
python -m src.analysis.education_distribution_analysis
```

主要输出：

- `学历需求分布分析报告.md`
- `education_by_occupation_category_year.csv`
- `education_by_occupation_year.csv`
- `education_by_occupation_category_month.csv`
- `education_by_occupation_month.csv`

### `industry_trend_analysis.py`

行业景气度分析。读取结构化统计主输入，按城市、行业、月度统计招聘量，并生成行业热度报告和图表。

使用方法：

```bash
python -m src.analysis.industry_trend_analysis
```

主要输出：

- `行业景气度分析报告.md`
- `city_industry_monthly_jobs.csv`
- `industry_monthly_jobs.csv`
- `行业景气度分析图.html`

### `structured_dimension_analysis.py`

结构化维度补充分析。统计经验要求、公司规模、城市职业需求等补充维度。

使用方法：

```bash
python -m src.analysis.structured_dimension_analysis
```

主要输出：

- `结构化维度补充分析报告.md`
- `experience_by_occupation.csv`
- `company_size_by_city_industry.csv`
- `city_occupation_demand.csv`

### `generate_standardized_tables.py`

规范化汇总表生成器。读取结构化统计批次目录中已有 CSV，并补充更稳定的交付表；其中学历月度薪资趋势会重新读取 PostgreSQL 结构化主输入。

使用方法：

```bash
python -m src.analysis.generate_standardized_tables
```

前置要求：建议先运行 `occupation_salary_analysis.py`。

主要输出：

- `standardized_salary_by_education_occupation.csv`
- `standardized_salary_by_occupation_month.csv`
- `standardized_salary_by_education_month.csv`

### `generate_excel_summary.py`

Excel 汇总报告生成器。把结构化统计批次目录中的 CSV 和 Markdown 摘要写入一个 Excel 文件。

使用方法：

```bash
python -m src.analysis.generate_excel_summary
```

前置要求：建议先运行结构化统计主链路，至少生成薪资、学历、行业和补充维度 CSV。

主要输出：

- `广东省招聘数据分析汇总报告.xlsx`

### `requirement_text_analysis.py`

requirement text 第二阶段正式分析入口。读取 `public.recruitment_jobs_normalized` 和 `public.job_description_parsed`，使用 `analysis_lexicon.requirement_rules` 抽取招聘约束，写入 `public.requirement_constraint_facts`，再输出聚合报表。

使用方法：

```bash
python -m src.analysis.requirement_text_analysis
python -m src.analysis.requirement_text_analysis --top-n 30 --min-group-size 100
python -m src.analysis.requirement_text_analysis --output-dir output/reports/req_analysis_test
```

主要输出：

- `run_manifest.json`
- `coverage_diagnostics.csv`
- `lexicon_summary.csv`
- `constraint_dimension_frequency.csv`
- `constraint_value_distribution.csv`
- `constraint_by_city_industry.csv`
- `template_noise_report.csv`
- `requirement_stringency_index.csv`
- `report.md`
- PostgreSQL 表 `public.requirement_constraint_facts`

当前正式口径聚焦 requirement 约束、模板噪声和招聘门槛强度；hard skill / soft skill 暂不作为正式结论。

### `requirement_constraint_extraction.py`

requirement 约束抽取工具模块，不建议单独运行。提供条目切分、文本规范化、模板噪声识别、经验/学历/年龄/性别/证书/语言等约束抽取，以及事实行转换函数。

典型调用方：

- `requirement_text_analysis.py`
- 单元测试或交互式调试

### `structured_pg_source.py`

结构化统计 PostgreSQL 数据源模块，不建议单独运行。负责从统一规范层和职业匹配结果层构造结构化统计 DataFrame，并补齐历史兼容字段。

核心能力：

- 读取 `public.recruitment_jobs_normalized`。
- 读取 `public.skill_extraction_requirement_matches`。
- 优先用 `recruitment_record_id` 回连，旧数据可兼容 `__source_table + __source_row_number`。
- 生成 `input_coverage_summary.json` 覆盖率摘要。

### `structured_common.py`

结构化统计公共工具模块，不建议单独运行。负责解析输出路径、控制历史中文 CSV 副本、写入规范 CSV。

### `analysis_common.py`

分析链路通用工具模块，不建议单独运行。负责构建批次输出目录、写入 `run_manifest.json`、标准化城市/行业/公司规模、解析发布时间月份，以及兼容历史来源表名。

### `__init__.py`

包标记文件，无需直接运行。

## 输出与清单

两条主链路都会写入 `run_manifest.json`，记录本次运行的步骤、参数、输入表和输出文件。结构化统计还会写入 `input_coverage_summary.json`，用于检查规范层记录数、职业匹配覆盖率、薪资/学历/发布时间字段覆盖率。

结构化统计默认输出英文规范 CSV；如旧脚本仍依赖中文文件名，可在统一 CLI 中传入 `--with-legacy-copies`。

## 职业词典表说明

当前职业匹配默认使用统一职业词典：

- `public.occ_dict_unified`：统一职业词典主表。
- `public.occ_dict`：兼容 view。
- `public.occ_dict_detailed`：兼容 view。
- `public.occ_dict_pro`：兼容 view。
- `public.occ_dict_class`：兼容 view。

新增统计字段时，优先补到 PostgreSQL 规范层、职业匹配结果层或 requirement 事实层，再由 `structured_pg_source.py` 暴露给分析脚本。

## 历史 CSV 适配器说明

`src.data_pipeline.occupation_integration` 只保留为历史兼容适配器，用于读取旧职业解析 CSV 并生成 `output/integrated`。当前 `src/analysis` 主链路不再依赖它。
