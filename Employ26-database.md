# Employ26 数据库文档

## 1. 文档目的

本文档面向本仓库的开发、分析和数据维护工作，目标是：

- 说明 `Employ26` PostgreSQL 数据库的真实结构
- 提供可直接引用的 schema / 表 / 字段导航
- 标出当前可稳定使用的关联键
- 给出后续“索引建设”和“表引用规范”的建议

本文档基于 2026-06-08 对本机 PostgreSQL 的实时探查结果生成。

## 2. 连接信息

当前探查结果：

- PostgreSQL 主机：`localhost`
- 端口：`5432`
- 可连接数据库：`Employ26`
- 当前用户：`postgres`

注意：

- 仓库当前配置默认数据库名已修正为 `Employ26`
- 实际存在的数据库名是大写开头的 `Employ26`
- 当前代码中如果直接使用 `employ26`，会连接失败

建议：

- 通过 `config.paths.get_project_paths()` 读取连接参数，或使用 `paths.pg_sqlalchemy_url()` 生成 SQLAlchemy URL
- 不要在脚本里手写 `employ26`、`recruit.main.*` 或自造连接字符串

## 3. 整体概览

数据库中共有 5 个业务 schema：

- `51job`
- `Liepin`
- `Zhilian`
- `annotations`
- `public`

共 28 张业务表。

按 `pg_stat_user_tables.n_live_tup` 估算的规模如下：

| schema | 表数 | 近似总行数 |
| --- | ---: | ---: |
| `51job` | 3 | 9,161,792 |
| `Liepin` | 3 | 4,031,911 |
| `Zhilian` | 3 | 7,752,907 |
| `annotations` | 4 | 48,331 |
| `public` | 15 | 256,200 |

## 4. Schema 分层理解

可以把这 5 个 schema 理解成下面几层：

### 4.1 招聘原始与清洗数据层

- `51job.raw_data`
- `51job.cleaned_data`
- `51job.sample`
- `Liepin.raw_data`
- `Liepin.cleaned_data`
- `Liepin.sample`
- `Zhilian.raw_data`
- `Zhilian.cleaned_data`
- `Zhilian.sample`

这三组表结构基本一致，是三个招聘平台的数据镜像。

治理原则：

- `raw_data` 继续作为平台原始镜像和溯源依据保留，不建议物理合并三家平台原始表
- `cleaned_data` 暂作为历史兼容表保留，但不再建议作为新流程入口
- 如果只是清洗“岗位描述”文本，后续应沉淀为派生字段或解析特征表，而不是复制整张招聘大表
- 新流程需要跨平台分析时，应读取统一规范层或解析特征层，而不是直接面向三套平台 schema 写重复逻辑

### 4.2 招聘统一规范层

统一规范层用于把三家平台的同义字段映射成英文公共字段，并用 `source_platform` 区分平台来源。

当前建议入口：

- 当前正式入口：`public.recruitment_jobs_normalized`
- 中期：如规模继续扩大，可独立为 `recruitment.jobs`

列名策略：

- 新表、新 view、新公共接口统一使用英文列名
- 历史中文字段继续保留在原始镜像层，例如 `岗位名称`、`岗位描述`、`岗位描述_清洗`
- 暂不直接 rename 历史大表字段，避免破坏仍引用中文列名的既有脚本

### 4.3 岗位描述解析特征层

岗位描述解析结果沉淀到：

- `public.job_description_parsed`

该表由 `src.data_pipeline.description_parsing` 写入，定位是可复用的结构化文本特征层，用于技能抽取、RAG、BGE 等流程复用解析结果，避免每个流程重复切分“岗位职责 / 任职要求”。

### 4.4 标注数据层

- `annotations.label_studio_annotations`
- `annotations.label_studio_tasks_v2`
- `annotations.label_studio_annotations_v2`
- `annotations.deepseek_relabel_raw`

这是 Label Studio 导入 PostgreSQL 后的人类标注结果，以及 DeepSeek 重标结果。它是第二轮标注实验、训练集构造和人机分歧分析的核心来源。

### 4.5 职业词典、词汇资源与匹配结果层

- `public.occ_dict_unified`
- `public.occ_dict`
- `public.occ_dict_detailed`
- `public.occ_dict_pro`
- `public.occ_dict_class`
- `analysis_lexicon.lexicon_release`
- `analysis_lexicon.user_dictionary`
- `analysis_lexicon.stopwords`
- `analysis_lexicon.phrase_rules`
- `analysis_lexicon.requirement_rules`
- `public.requirement_constraint_facts`
- `public.match_training_features`
- `public.rag_match_results_v2`
- `public.job_match_results`

这部分是职业大典、requirement text 统计词汇资源、Phase 2 requirement 约束事实层、RAG 匹配特征、职业匹配结果的核心表。

### 4.6 技能抽取实验层

- `public.skill_extraction_requirement_matches`
- `public.hard_skill_match_results_dev`
- `public.hard_skill_match_details_dev`
- `public.hard_skill_match_summary_dev`
- `public.skill_extraction_v3_results`

这部分主要服务技能抽取、技能匹配与调试。其中 `skill_extraction_v3_results` 是 V3 管线的正式输出表，存储硬技能（含 8 类分类）和软技能（含大五人格维度）的结构化结果。

### 4.7 采样与测试层

- `public.jd_raw`
- `public.medium_sample`
- `public.test_sample`
- `public.e2e_test`

## 5. 当前结构现状

实时探查结果显示：

- 当前 28 张业务表中，`annotations.deepseek_relabel_raw` 已有 `task_id` 主键
- `annotations.label_studio_tasks_v2` 已有 `id` 主键
- `annotations.label_studio_annotations_v2` 已有 `(task_id, annotation_id)` 复合主键
- `annotations.label_studio_annotations_v2.task_id` 已外键关联到 `annotations.label_studio_tasks_v2.id`
- 除 `annotations.label_studio_annotations_v2.task_id` 外，其他业务关系尚未补充显式外键
- `annotations` 标注链路已补充常用查询索引和 JSONB GIN 索引
- 其他多数业务表仍缺少主键、外键、索引和注释

这意味着：

- `annotations` schema 已具备较稳定的任务-标注明细关系约束
- 跨 schema 的关系仍主要依赖“约定字段”，例如 `task_id`、`row_id`、`code`
- 三大招聘平台大表和多数 `public` 结果表后续仍可能需要补索引

## 6. 核心逻辑键

虽然没有显式主外键，但当前库里存在一组事实上的“逻辑关联键”：

### 6.1 `task_id` / `id`

主要用于标注任务链路：

- `annotations.label_studio_tasks_v2.id`
- `annotations.label_studio_annotations_v2.task_id`
- `public.match_training_features.task_id`
- `public.rag_match_results_v2.task_id`

实测情况：

- `annotations.label_studio_tasks_v2.id` 共 `18611` 条，且全部唯一
- `annotations.label_studio_annotations_v2` 的 `(task_id, annotation_id)` 共 `19380` 条，且全部唯一
- `public.match_training_features.task_id` 共 `18606` 条，且全部唯一
- `public.rag_match_results_v2.task_id` 共 `10` 条，且全部唯一

### 6.2 `row_id`（历史快照链路）

主要用于把历史标注任务回放到当年的 Label Studio 导出快照：

- `annotations.label_studio_tasks_v2.row_id`
- `output/data5/Tier2_Matched_Data.csv` 前 30 行
- `output/data5/Tier3_Pending_Data.csv` 全量

实测情况：

- `annotations.label_studio_tasks_v2` 共 `18611` 条任务
- `row_id` 范围为 `0` 到 `59999`，与历史导出快照行号空间一致
- 它不应再作为招聘源记录主键使用；正式招聘身份应以回填后的 `recruitment_record_id` 为准

### 6.3 `code`

主要用于职业词典和职业候选结果对齐：

- `public.occ_dict_unified.code`
- `public.occ_dict.code`
- `public.occ_dict_detailed.code`
- `public.occ_dict_pro.code`
- `public.match_training_features.cand_[A-E]_code`
- `public.rag_match_results_v2.best_code`
- `public.skill_extraction_requirement_matches.top1_code`
- `public.skill_extraction_requirement_matches.occupation_code`

### 6.4 `sample_row_id + __source_table`（历史技能链路）

主要用于技能抽取类结果回溯样本来源：

- `public.skill_extraction_requirement_matches`
- `public.hard_skill_match_details_dev`
- `public.hard_skill_match_summary_dev`

实测情况：

- `public.skill_extraction_requirement_matches` 的历史版本使用 `sample_row_id`
- `public.hard_skill_match_summary_dev` 的历史版本使用 `sample_row_id`
- `public.hard_skill_match_details_dev` 共有 `23` 条明细，但只对应 `7` 个源样本，属于一对多明细表

## 7. 推荐 Join 路径

以下是当前最稳的几条关联路径。

### 7.1 标注任务 -> 标注明细

```sql
select *
from annotations.label_studio_tasks_v2 t
join annotations.label_studio_annotations_v2 a
  on a.task_id = t.id;
```

实测命中：`19380`

### 7.2 标注任务 -> 训练特征

```sql
select *
from annotations.label_studio_tasks_v2 t
join public.match_training_features f
  on f.task_id = t.id;
```

实测命中：`18606`

这条路径适合：

- 回看训练样本来自哪条标注任务
- 按标注任务分析候选特征

### 7.3 标注任务 -> RAG 结果

```sql
select *
from annotations.label_studio_tasks_v2 t
join public.rag_match_results_v2 r
  on r.task_id = t.id;
```

实测命中：`10`

说明：

- `rag_match_results_v2` 当前只是很小的结果表，不是完整产物表

### 7.4 标注任务 -> 原始 JD

```sql
select *
from annotations.label_studio_tasks_v2 t
join public.jd_raw j
  on t.row_id = j.row_id;
```

实测命中：`18260`

这条路径非常重要，适合：

- 从标注任务回溯原始 JD 文本
- 将标注结果与原始岗位描述合并分析

### 7.5 技能匹配结果 -> 样本来源

`public.skill_extraction_requirement_matches` 当前不能通过 `job_id` 稳定关联到 `public.job_match_results`，因为：

- `skill_extraction_requirement_matches.job_id` 全部为空
- `job_match_results.job_id` 也全部为空
- 两表直接 join 实测命中为 `0`

因此技能类结果当前更适合通过：

- `sample_row_id`
- `__source_table`
- `岗位名称`

来做回溯，而不是依赖 `job_id`。

## 8. 表清单与用途

### 8.1 `51job` schema

#### `51job.raw_data`

- 近似行数：`4,553,075`
- 字段数：`10`
- 用途：51job 原始招聘数据

字段：

- `发布时间`
- `岗位名称`
- `工作城市`
- `薪资水平`
- `经验要求`
- `学历要求`
- `岗位描述`
- `公司名称`
- `公司规模`
- `公司行业`

#### `51job.cleaned_data`

- 近似行数：`4,563,205`
- 字段数：`10`
- 用途：51job 清洗后的招聘数据
- 字段与 `raw_data` 一致

#### `51job.sample`

- 近似行数：`45,512`
- 字段数：`10`
- 用途：51job 样本集
- 字段与 `raw_data` 一致

### 8.2 `Liepin` schema

#### `Liepin.raw_data`

- 近似行数：`2,004,265`
- 字段数：`10`
- 用途：猎聘原始招聘数据

#### `Liepin.cleaned_data`

- 近似行数：`2,007,599`
- 字段数：`10`
- 用途：猎聘清洗后数据

#### `Liepin.sample`

- 近似行数：`20,047`
- 字段数：`10`
- 用途：猎聘样本集

### 8.3 `Zhilian` schema

#### `Zhilian.raw_data`

- 近似行数：`3,854,391`
- 字段数：`10`
- 用途：智联原始招聘数据

#### `Zhilian.cleaned_data`

- 近似行数：`3,860,137`
- 字段数：`10`
- 用途：智联清洗后数据

#### `Zhilian.sample`

- 近似行数：`38,379`
- 字段数：`10`
- 用途：智联样本集

### 8.4 `annotations` schema

`annotations` schema 当前实际包含 4 张表：

- `annotations.label_studio_annotations`
- `annotations.label_studio_tasks_v2`
- `annotations.label_studio_annotations_v2`
- `annotations.deepseek_relabel_raw`

整体判断：

- schema 单独存放标注相关数据是合理的
- `label_studio_tasks_v2` 与 `label_studio_annotations_v2` 拆成任务表和标注明细表是合理的
- `deepseek_relabel_raw` 放在同一 schema 中也合理，便于做人类标注和模型重标对照
- 当前已补充核心约束、常用索引、JSONB 镜像字段、标准化 `choice_code`、时间戳镜像字段和便捷视图

#### 为什么不建议合并 `tasks_v2` 和 `annotations_v2`

不建议把 `annotations.label_studio_tasks_v2` 和 `annotations.label_studio_annotations_v2` 物理合并成一张表。

原因：

- 二者是典型的一对多关系：一个任务可能有 0、1、2 条，甚至十几条标注
- 如果合并为一张宽表，任务级字段会被重复存储多次，例如岗位文本、候选 A-E、`data_raw`
- 重复存储会带来更新异常：同一个任务的候选信息如果需要修正，必须修改多行
- 标注员一致性、标注耗时、分歧分析天然是“按 annotation 行”统计，拆表更适合
- 如果反过来把 annotations 全部塞回任务表 JSON 中，日常 SQL 统计会更难写

推荐做法：

- 底层继续保留 `tasks_v2` 和 `annotations_v2` 两张规范表
- 上层新增查询视图，给日常使用提供“合并表”的便利

#### 为什么建议将 text JSON 迁移到 jsonb

`data_raw`、`annotations_completed`、legacy 表中的 `annotations`、`data` 等字段当前是 `text`，本质上只是“把 JSON 当字符串存起来”。迁移到 `jsonb` 不是简单换字段名，而是让 PostgreSQL 能理解这些字段的 JSON 结构。

实际作用：

- 可以直接查询 JSON 内部字段，例如 `data_raw_jsonb ->> 'job_title'`
- 可以对 JSON 路径或包含关系建立 GIN 索引
- 可以校验字段是否存在、类型是否合理，减少坏数据进入后续流程
- 可以减少 Python 脚本反复读取 text 后 `json.loads()` 的成本和错误点
- 可以用生成列或表达式索引沉淀高频字段，例如从 `data_raw_jsonb` 生成候选 code

推荐迁移方式：

- 优先新增 `*_jsonb` 镜像字段，保留原 `text` 字段兼容旧脚本
- 确认所有读取脚本迁移完成后，再考虑是否删除旧 `text` 字段

#### `annotations.label_studio_annotations`

- 近似行数：`5,170`
- 字段数：`28`
- 用途：早期 legacy 标注任务级导入表
- 特征：保留大块 JSON 文本，如 `annotations`、`data`、`meta`
- 已新增 `jsonb` 镜像字段和时间戳镜像字段
- 规范性评价：适合作为历史归档表，不建议作为新分析入口

关键字段：

- `id`
- `project`
- `total_annotations`
- `created_at`
- `updated_at`
- `annotations`
- `data`

新增字段：

- `annotations_jsonb`
- `data_jsonb`
- `meta_jsonb`
- `drafts_jsonb`
- `predictions_jsonb`
- `comment_authors_jsonb`
- `created_at_ts`
- `updated_at_ts`

#### `annotations.label_studio_tasks_v2`

- 近似行数：`18,611`
- 字段数：`32`
- 用途：第二轮标注任务主表
- `id` 当前为 task 主键
- `row_id` 是历史 Label Studio 导出快照中的行号，不应再视为招聘源表主键
- `recruitment_record_id` 用于承接历史任务回填后的正式招聘记录身份
- `annotations_completed` 和 `data_raw` 仍保留 `text` 原字段，并已新增 `annotations_completed_jsonb` 和 `data_raw_jsonb`
- `created_at`、`updated_at` 仍保留原始文本，并已新增 `created_at_ts`、`updated_at_ts`

关键字段：

- `id`
- `row_id`
- `recruitment_record_id`
- `sample_source`
- `job_title`
- `job_requirements`
- `is_validation`
- `cand_a_code` ~ `cand_e_code`
- `cand_a_title` ~ `cand_e_title`
- `cand_a_source` ~ `cand_e_source`
- `annotations_completed`
- `data_raw`
- `annotations_completed_jsonb`
- `data_raw_jsonb`
- `created_at_ts`
- `updated_at_ts`

样本来源分布：

- `tier3_main`: `18581`
- `tier2_validation`: `30`

#### `annotations.label_studio_annotations_v2`

- 近似行数：`19,380`
- 字段数：`11`
- 用途：第二轮标注明细表，一条 annotation 一行
- `(task_id, annotation_id)` 当前为复合主键
- `task_id` 已外键关联到 `label_studio_tasks_v2.id`
- 当前无孤儿 annotation
- `best_candidate` 保留中文展示文本，已新增标准化字段 `choice_code`

关键字段：

- `task_id`
- `annotation_id`
- `annotator_id`
- `lead_time_sec`
- `best_candidate`
- `soft_skill`
- `reason`
- `created_at`
- `choice_code`
- `created_at_ts`

标准化选择分布：

- `A`: `3198`
- `B`: `3082`
- `C`: `2712`
- `D`: `2781`
- `E`: `2718`
- `NONE`: `4889`

#### `annotations.deepseek_relabel_raw`

- 近似行数：`5,170`
- 字段数：`8`
- 用途：DeepSeek 对 Label Studio 任务的重标结果
- `task_id` 已有主键
- `candidates` 和 `payload` 已是 `jsonb`，可作为后续 JSON 字段规范化参考

关键字段：

- `task_id`
- `job_title`
- `deepseek_choice`
- `deepseek_confidence`
- `deepseek_reasoning`
- `deepseek_raw_response`
- `candidates`
- `payload`

#### `annotations.label_studio_task_rrid_backfill_audit`

- 用途：历史标注任务 `recruitment_record_id` 回填审计表
- 定位：一次性治理后的长期审计留痕
- 主键：`task_id`

关键字段：

- `task_id`
- `historical_row_id`
- `mapping_status`
- `mapping_rule`
- `confidence_tier`
- `source_table`
- `source_row_number`
- `recruitment_record_id`
- `candidate_count`
- `best_similarity_score`
- `second_similarity_score`
- `evidence_summary`
- `backfill_version`
- `backfilled_at`

### 8.5 `public` schema

#### 词典类表

##### `public.occ_dict_unified`

- 用途：统一职业词典主表
- 定位：当前正式职业词典入口；同时承载职业叶子节点与分类骨架节点
- 兼容策略：`public.occ_dict`、`public.occ_dict_detailed`、`public.occ_dict_pro`、`public.occ_dict_class` 已退役为兼容 view

关键字段：

- `node_key`
- `node_type`
- `is_terminal`
- `code`
- `title`
- `desc`
- `tasks`
- `级别`
- `分类代码`
- `职业代码`
- `大类`
- `中类`
- `小类`
- `细类`
- `task_list`
- `task_text_joined`
- `title_clean`
- `desc_clean`
- `hierarchy_text`
- `aliases`
- `retrieval_title_text`
- `retrieval_desc_text`
- `retrieval_task_text`

##### `public.occ_dict`

- 近似行数：`1,698`
- 字段：`code`, `title`, `desc`, `tasks`
- 用途：基础职业词典兼容 view
- 备注：从 `public.occ_dict_unified` 投影职业叶子节点，短期保留用于历史兼容

##### `public.occ_dict_detailed`

- 近似行数：`1,698`
- 字段数：`11`
- 在 `occ_dict` 基础上加入：
  - `级别`
  - `分类代码`
  - `职业代码`
  - `大类`
  - `中类`
  - `小类`
  - `细类`
- 备注：兼容 view；与 `occ_dict` 基本 1:1，对下游来说更像增强过渡层

##### `public.occ_dict_pro`

- 近似行数：`1,698`
- 字段数：`20`
- 用途：增强版职业词典兼容 view，适合兼容旧检索与 RAG 脚本
- 备注：字段最全，但正式主入口已切换到 `public.occ_dict_unified`

特征字段：

- `task_list`
- `task_text_joined`
- `title_clean`
- `desc_clean`
- `hierarchy_text`
- `aliases`
- `retrieval_title_text`
- `retrieval_desc_text`
- `retrieval_task_text`

##### `public.occ_dict_class`

- 近似行数：`2,324`
- 字段数：`7`
- 用途：职业分层分类字典兼容 view
- 备注：从 `public.occ_dict_unified` 投影分类骨架节点，不是前三张表的简单重复

当前状态与推荐：

- 四张旧物理表已退役出 `public`，并封存到 `archive_occ`
- `public.occ_dict`、`public.occ_dict_detailed`、`public.occ_dict_pro`、`public.occ_dict_class` 当前均为兼容 view
- 下游检索、职业匹配、预处理默认使用 `public.occ_dict_unified`
- 层级骨架回查优先使用 `public.occ_dict_class`
- 旧脚本如未及时改造，可临时继续读取兼容 view

#### RAG / 职业匹配类表

##### `public.match_training_features`

- 近似行数：`18,606`
- 字段数：`48`
- 用途：标注任务训练特征表

这是一个很关键的训练表，包含：

- 任务基础信息：`task_id`, `job_title`, `job_requirements`, `label`
- 候选聚合特征：`agg_*`
- 候选 A-E 的 code/title/desc/source/sim/rank/category

##### `public.rag_match_results_v2`

- 近似行数：`10`
- 字段数：`13`
- 用途：RAG 匹配结果表

关键字段：

- `task_id`
- `job_title`
- `job_requirements`
- `best_code`
- `best_title`
- `confidence`
- `reasoning`
- `evidence`
- `top1_retrieval_code`
- `top1_retrieval_title`
- `top1_retrieval_score`
- `candidates_json`
- `from_cache`

##### `public.job_match_results`

- 近似行数：`50`
- 字段数：`17`
- 用途：岗位匹配结果

当前注意事项：

- `job_id` 当前全为空，暂时不能作为稳定 join 键

#### JD / 样本类表

##### `public.jd_raw`

- 近似行数：`21,547`
- 字段数：`13`
- 用途：原始 JD 主表

关键字段：

- `row_id`
- `job_title`
- `clean_title`
- `jd_snippet`
- `text`
- `occ_category`
- `occ_core`
- `hard_skills`
- `label`

##### `public.medium_sample`

- 近似行数：`500`
- 字段数：`10`
- 用途：中等规模样本表

##### `public.test_sample`

- 近似行数：`100`
- 字段数：`10`
- 用途：测试样本表

##### `public.e2e_test`

- 近似行数：`50`
- 字段数：`10`
- 用途：端到端测试表

#### 岗位描述解析特征表

##### `public.recruitment_jobs_normalized`

- 用途：三家招聘平台 sample / 后续公共招聘主表的统一规范层
- 定位：跨平台分析正式入口，保留统一英文列名与来源定位信息
- 当前标准身份字段：`recruitment_record_id`

关键字段：

- `recruitment_record_id`
- `source_platform`
- `source_table`
- `source_row_number`
- `source_native_job_id`
- `dedupe_fingerprint`
- `job_title`
- `job_description_raw`
- `work_city`
- `company_name`
- `publish_date`
- `salary_raw`
- `education_requirement_raw`
- `experience_requirement_raw`
- `company_size_raw`
- `company_industry_raw`

说明：

- `*_raw` 列先承接原始结构化维度，后续如需标准化列，再以增量方式追加
- `source_table + source_row_number` 是当前最稳的来源定位键
- `dedupe_fingerprint` 用于跨批次稳定识别同源招聘记录
- requirement text 第二阶段直接以这张表作为招聘主表入口

##### `public.job_description_parsed`

- 用途：岗位描述结构化解析结果表，由 `src.data_pipeline.description_parsing` 写入
- 定位：解析特征层，不替代三平台原始表，也不继续复制整张 `cleaned_data`
- 列名：统一使用英文列名，作为后续新流程公共接口
- 当前线上表仍保留历史契约，主定位键实质上是 `source_table + source_row_number + parser_version`

关键字段：

- `source_platform`
- `source_table`
- `source_row_number`
- `source_record_id`
- `job_title`
- `job_description_raw`
- `job_description_clean`
- `description_sections` (`jsonb`)
- `requirements_text`
- `duties_text`
- `unclassified_text`
- `sections_brief`
- `rag_query_text`
- `rag_query_source`
- `parser_version`
- `parsed_at`

推荐索引：

- `source_platform`
- `(source_table, source_row_number)`
- `description_sections` GIN

说明：

- 当前线上库里同一 `source_table + source_row_number` 往往存在多版 `parser_version`
- requirement text 第二阶段运行时应按 `parsed_at DESC` 取每条来源记录的最新权威解析结果
- `source_table + source_row_number` 是当前最稳的回溯键，用于回溯原始平台表或统一规范层来源位置
- `description_sections` 保存完整切分 JSON，`requirements_text` / `duties_text` 保存高频检索文本
- `parser_version` 保留为结果元数据，不再参与公共身份定义

##### `public.requirement_constraint_facts`

- 用途：requirement text 第二阶段正式中间层
- 定位：一条招聘记录的一条 requirement item 的一个约束事实
- 来源：`src.analysis.requirement_text_analysis` 抽取后写入

关键字段：

- `fact_id`
- `recruitment_record_id`
- `source_table`
- `source_row_number`
- `item_index`
- `item_text_raw`
- `item_text_normalized`
- `dimension_name`
- `constraint_type`
- `raw_value`
- `normalized_value`
- `operator`
- `value_min`
- `value_max`
- `unit`
- `evidence_text`
- `rule_id`
- `extractor_version`

说明：

- 当前只承载非技能 requirement 约束，不承载 hard skill / soft skill 正式分类结论
- 当前主要维度包括经验、学历、年龄、性别、证书、语言、出差、班次、身体条件、工作方式
- 这张表是后续 requirement 统计、门槛强度分析和分层报表的唯一正式复用入口

#### 技能抽取类表

##### `public.skill_extraction_requirement_matches`

- 近似行数：`103,938`
- 字段数：`54`
- 用途：技能抽取与职业候选对齐结果，是当前 `public` schema 中最大的业务结果表

关键字段分组：

- 标准引用与溯源：
  - `recruitment_record_id`
  - `source_table`
  - `source_row_number`
- 文本字段：
  - `岗位名称`
  - `岗位描述`
  - `岗位描述_清洗`
  - `岗位描述_切分JSON`
  - `任职要求_items_text`
  - `岗位职责_items_text`
  - `unclassified_text`
- 查询与匹配：
  - `query_text`
  - `query_source`
  - `selected_candidate_rank`
  - `is_matched`
- 职业结果：
  - `occupation_code`
  - `occupation_title`
  - `大类`
  - `中类`
  - `小类`
  - `细类`
- Top1-Top5 候选：
  - `top1_code` ~ `top5_code`
  - `top1_title` ~ `top5_title`
  - `top1_score` ~ `top5_score`
  - `top1_detail_path` ~ `top5_detail_path`
  - `top1_detail_name` ~ `top5_detail_name`

说明：

- 当前活跃公共链路应以 `recruitment_record_id` 作为标准引用字段
- `source_table` / `source_row_number` 仅用于来源回溯

##### `public.hard_skill_match_results_dev`

- 近似行数：`103,938`
- 字段数：`8`
- 用途：硬技能匹配主结果表

关键字段：

- `岗位名称`
- `岗位描述_清洗`
- `任职要求_items_text`
- `岗位职责_items_text`
- `sections_brief`
- `occupation_title`
- `occupation_code`
- `skill_name` (`jsonb`)

##### `public.hard_skill_match_details_dev`

- 近似行数：`23`
- 字段数：`14`
- 用途：硬技能匹配明细调试表

##### `public.hard_skill_match_summary_dev`

- 近似行数：`20`
- 字段数：`11`
- 用途：硬技能匹配汇总调试表

##### `public.skill_extraction_v3_results`

- 用途：V3 统一技能抽取管线输出表，同时存储硬技能和软技能结果
- 定位：V3 管线的正式结果表，替代 `skill_extraction_requirement_matches` 作为新流程入口
- 写入模块：`src.skill_extraction.v3_result_writer`
- 唯一键：`recruitment_record_id`
- 配置键：`config/database.yaml` 中 `tables.skill_extraction_v3_results`

字段：

- `id` — 自增主键
- `recruitment_record_id` — 招聘记录唯一标识（唯一键）
- `source_table` — 来源表名
- `source_row_number` — 来源行号
- `job_title` — 岗位名称
- `hard_skills` (`jsonb`) — 硬技能列表，每项含 `name` 和 `category`（8 类之一）
- `hard_skill_count` — 硬技能数量
- `soft_skills` (`jsonb`) — 软技能列表，每项含 `name`、`dimension`、`confidence`、`source`
- `soft_skill_count` — 软技能数量
- `pipeline_version` — 管线版本，默认 `v3`
- `extracted_at` — 抽取时间戳

硬技能 category 取值：`programming_language`、`framework`、`database`、`tool`、`office`、`equipment`、`process`、`certification`

软技能 dimension 取值：`openness`、`conscientiousness`、`extraversion`、`agreeableness`、`neuroticism`

索引：

- `idx_v3_results_rid` — `recruitment_record_id`
- `idx_v3_results_hard_skills` — `hard_skills` GIN
- `idx_v3_results_soft_skills` — `soft_skills` GIN

推荐查询：

```sql
-- 查看某条记录的技能抽取结果
SELECT
    recruitment_record_id,
    job_title,
    hard_skills,
    hard_skill_count,
    soft_skills,
    soft_skill_count,
    extracted_at
FROM public.skill_extraction_v3_results
WHERE recruitment_record_id = :rid;

-- 按硬技能类别统计
SELECT
    (hs ->> 'category') AS category,
    COUNT(*) AS skill_count
FROM public.skill_extraction_v3_results,
     LATERAL jsonb_array_elements(hard_skills) AS hs
GROUP BY category
ORDER BY skill_count DESC;

-- 按软技能维度统计
SELECT
    (ss ->> 'dimension') AS dimension,
    COUNT(*) AS skill_count
FROM public.skill_extraction_v3_results,
     LATERAL jsonb_array_elements(soft_skills) AS ss
GROUP BY dimension
ORDER BY skill_count DESC;
```

## 9. 常用查询模板

### 9.1 查看第二轮标注任务及其标注明细

```sql
select
    t.id as task_id,
    t.row_id,
    t.sample_source,
    t.job_title,
    t.job_requirements,
    t.is_validation,
    a.annotation_id,
    a.annotator_id,
    a.best_candidate,
    a.soft_skill,
    a.reason
from annotations.label_studio_tasks_v2 t
left join annotations.label_studio_annotations_v2 a
  on a.task_id = t.id
order by t.id, a.annotation_id;
```

### 9.2 推荐的标注合并视图

该视图用于日常查询时把 task 与 annotation 放到同一结果集中，但不改变底层两张表结构。

```sql
create or replace view annotations.v_label_studio_task_annotations_v2 as
select
    t.id as task_id,
    t.row_id,
    t.sample_source,
    t.job_title,
    t.job_requirements,
    t.is_validation,
    t.cand_a_code,
    t.cand_a_title,
    t.cand_a_source,
    t.cand_b_code,
    t.cand_b_title,
    t.cand_b_source,
    t.cand_c_code,
    t.cand_c_title,
    t.cand_c_source,
    t.cand_d_code,
    t.cand_d_title,
    t.cand_d_source,
    t.cand_e_code,
    t.cand_e_title,
    t.cand_e_source,
    a.annotation_id,
    a.annotator_id,
    a.lead_time_sec,
    a.best_candidate,
    case
        when a.best_candidate = '候选A' then 'A'
        when a.best_candidate = '候选B' then 'B'
        when a.best_candidate = '候选C' then 'C'
        when a.best_candidate = '候选D' then 'D'
        when a.best_candidate = '候选E' then 'E'
        when a.best_candidate = '以上选项都不属于' then 'NONE'
        else null
    end as choice_code,
    a.soft_skill,
    a.reason,
    a.created_at as annotation_created_at
from annotations.label_studio_tasks_v2 t
left join annotations.label_studio_annotations_v2 a
  on a.task_id = t.id;
```

说明：

- 该视图已在 `Employ26` 数据库中创建
- 如果后续给 `annotations_v2` 新增实体字段 `choice_code`，视图中的 `case` 可以改为直接读取字段

### 9.3 回溯某条标注任务对应的原始 JD

```sql
select
    t.id as task_id,
    t.row_id,
    t.job_title,
    t.job_requirements,
    j.clean_title,
    j.jd_snippet,
    j.text
from annotations.label_studio_tasks_v2 t
join public.jd_raw j
  on j.row_id = t.row_id
where t.id = :task_id;
```

### 9.4 查看训练特征与标注任务

```sql
select
    f.task_id,
    t.job_title,
    f.label,
    f.agg_mean_sim,
    f.cand_A_code,
    f.cand_B_code,
    f.cand_C_code,
    f.cand_D_code,
    f.cand_E_code
from public.match_training_features f
join annotations.label_studio_tasks_v2 t
  on t.id = f.task_id;
```

### 9.5 用职业 code 关联职业词典

```sql
select
    r.best_code,
    d.title,
    d.大类,
    d.中类,
    d.小类,
    d.细类
from public.rag_match_results_v2 r
left join public.occ_dict_unified d
  on d.code = r.best_code
 and d.node_type = 'occupation_leaf';
```

### 9.6 查看技能匹配结果来源

```sql
select
    recruitment_record_id,
    source_table,
    source_row_number,
    岗位名称,
    occupation_code,
    occupation_title,
    is_matched,
    top1_code,
    top1_title,
    top1_score
from public.skill_extraction_requirement_matches
where source_table = '"51job".sample';
```

### 9.7 查看岗位描述解析结果并回溯来源

```sql
select
    source_platform,
    recruitment_record_id,
    source_table,
    source_row_number,
    job_title,
    requirements_text,
    duties_text,
    parser_version,
    parsed_at
from public.job_description_parsed
where source_table = '"51job".sample'
order by source_row_number
limit 100;
```

如果需要检查 JSONB 切分结果：

```sql
select
    recruitment_record_id,
    description_sections -> 'sections' as sections
from public.job_description_parsed
where description_sections ? 'sections'
limit 20;
```

## 10. 已执行的 `annotations` 优化

本节记录已经在 `Employ26` 数据库中执行过的 `annotations` schema 优化。对应 SQL 已沉淀到 `docs/optimize_annotations_schema.sql`。

### 10.1 标注链路约束和索引

```sql
alter table annotations.label_studio_tasks_v2
add constraint label_studio_tasks_v2_pkey primary key (id);

alter table annotations.label_studio_annotations_v2
add constraint label_studio_annotations_v2_pkey primary key (task_id, annotation_id);

alter table annotations.label_studio_annotations_v2
add constraint label_studio_annotations_v2_task_id_fkey
foreign key (task_id)
references annotations.label_studio_tasks_v2 (id);

create index if not exists idx_label_studio_annotations_v2_task_id
on annotations.label_studio_annotations_v2 (task_id);

create index if not exists idx_label_studio_tasks_v2_row_id
on annotations.label_studio_tasks_v2 (row_id);

create index if not exists idx_match_training_features_task_id
on public.match_training_features (task_id);
```

已验证：

- `label_studio_tasks_v2.id` 为主键
- `label_studio_annotations_v2(task_id, annotation_id)` 为复合主键
- `label_studio_annotations_v2.task_id` 已外键关联 `label_studio_tasks_v2.id`
- `label_studio_annotations_v2` 无孤儿记录

### 10.2 已新增的标注 JSONB 字段

```sql
alter table annotations.label_studio_tasks_v2
add column if not exists data_raw_jsonb jsonb;

alter table annotations.label_studio_tasks_v2
add column if not exists annotations_completed_jsonb jsonb;

create index if not exists idx_label_studio_tasks_v2_data_raw_jsonb_gin
on annotations.label_studio_tasks_v2 using gin (data_raw_jsonb);

create index if not exists idx_label_studio_tasks_v2_annotations_completed_jsonb_gin
on annotations.label_studio_tasks_v2 using gin (annotations_completed_jsonb);
```

已验证：

- `label_studio_tasks_v2.data_raw_jsonb` 无空值
- `label_studio_tasks_v2.annotations_completed_jsonb` 无空值
- legacy 表的 `annotations_jsonb`、`data_jsonb`、`meta_jsonb`、`drafts_jsonb`、`predictions_jsonb`、`comment_authors_jsonb` 均已回填

### 10.3 已新增的标准化字段和视图

```sql
alter table annotations.label_studio_annotations_v2
add column if not exists choice_code text;

create or replace view annotations.v_label_studio_task_annotations_v2 as
select ...
```

已验证：

- `choice_code` 无空值，取值为 `A/B/C/D/E/NONE`
- `created_at_ts`、`updated_at_ts` 已回填
- `annotations.v_label_studio_task_annotations_v2` 可查询
- 视图行数为 `19385`，包含 `19380` 条标注明细和 `5` 个无标注明细任务

### 10.4 仍建议后续补充的索引

以下索引尚未在本次执行中处理，可按实际查询压力择机补充：

#### 职业词典与 RAG 结果

```sql
create unique index if not exists idx_occ_dict_unified_leaf_code
on public.occ_dict_unified (code)
where node_type = 'occupation_leaf' and code is not null;

create index if not exists idx_occ_dict_unified_occ_code
on public.occ_dict_unified ("职业代码");

create index if not exists idx_occ_dict_unified_class_code
on public.occ_dict_unified ("分类代码");

create index if not exists idx_rag_match_results_v2_task_id
on public.rag_match_results_v2 (task_id);
```

#### 技能抽取结果

```sql
create index if not exists idx_skill_req_matches_source_sample
on public.skill_extraction_requirement_matches (__source_table, sample_row_id);

create index if not exists idx_skill_req_matches_occupation_code
on public.skill_extraction_requirement_matches (occupation_code);

create index if not exists idx_skill_req_matches_top1_code
on public.skill_extraction_requirement_matches (top1_code);
```

#### 原始 JD 回溯

```sql
create index if not exists idx_jd_raw_row_id
on public.jd_raw (row_id);
```

## 11. 后续规范建议

为了让数据库更容易“索引和引用”，建议后续按下面方式收敛：

1. 继续补齐非 annotations 表的主键或唯一索引
   - `public.match_training_features(task_id)`
   - `public.jd_raw(row_id)`

2. 给跨 schema 逻辑关联补外键或至少补注释
   - `public.match_training_features.task_id -> annotations.label_studio_tasks_v2.id`
   - `annotations.label_studio_tasks_v2.row_id -> public.jd_raw.row_id`

3. 统一命名风格
   - 当前库中中英文列名混用
   - 不同表存在 `job_title` / `岗位名称` 并行
   - 建议后续新增表统一采用英文列名，保留中文列名只用于历史表

4. 建立招聘统一规范入口
   - 保留 `51job`、`Liepin`、`Zhilian` 三个原始 schema 作为平台镜像
   - 新增 `public.recruitment_jobs_normalized` 或 `recruitment.jobs` 作为跨平台分析入口
   - 使用 `recruitment_record_id` 作为标准身份字段
   - 使用 `source_platform`、`source_table`、`source_row_number` 保留溯源能力

5. 将岗位描述解析结果作为公共特征层
   - 新流程优先读取 `public.job_description_parsed`
   - 技能抽取、RAG、BGE 流程不应重复解析同一份岗位描述
   - `cleaned_data` 仅作为历史兼容表保留，不再作为新流程默认入口

6. 将其他高频 JSON 文本字段逐步结构化
   - `candidates_json -> candidates_jsonb`

7. 继续补表注释和列注释
   - `annotations` 核心表已补充基础注释
   - 其他业务表仍应补充注释，方便自动生成数据字典

## 12. 文档结论

`Employ26` 当前已经具备可用的数据资产，但数据库本身仍偏“结果仓库”而不是“约束完善的业务库”：

- 数据很多，尤其是三大招聘平台原始表
- 标注任务链路和训练特征链路已经基本成型
- 职业词典、RAG、技能抽取结果也已进入库中
- `annotations` schema 已具备主键、外键、常用索引、JSONB 镜像字段、标准化选择字段和便捷查询视图
- 多数非 annotations 业务表的主键、外键、索引、注释仍明显不足

因此，后续如果要提升“引用稳定性”和“查询性能”，优先级最高的工作是：

1. 将 `annotations` 的规范化做法推广到训练特征、职业词典和技能抽取结果表
2. 补齐高频 join 字段的索引
3. 统一任务链路和职业 code 链路的引用方式
4. 逐步把其他高频 JSON 文本字段迁移到 `jsonb`
5. 将招聘主数据收敛到英文列名的统一规范层，并把岗位描述解析结果沉淀到 `public.job_description_parsed`
