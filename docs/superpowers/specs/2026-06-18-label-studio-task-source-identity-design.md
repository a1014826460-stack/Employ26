# Label Studio Task Source Identity Design

## 背景

当前 `annotations.label_studio_tasks_v2` 的 `task_id` 表示标注任务身份，`public.recruitment_jobs_normalized.recruitment_record_id` 表示正式招聘记录身份。历史 `row_id` 只是 Label Studio 导出快照行号，不是招聘源表主键，不能直接作为 `task_id -> source record` 的可靠关联键。

此前 `recruitment_record_id` 修复已经证明：

- 旧 `row_id` 回放链路可能与当前 `data_raw` 任务内容漂移
- 使用任务当前 `job_title + job_requirements_clean` 可修复大部分映射
- 仍有 `244` 条任务不能自动唯一确认，需要保留“候选证据”而不是强行写死单一结论

因此需要建立一套独立的“任务源身份映射层”，将：

- 标注任务的当前文本载荷
- 候选源招聘记录
- 自动判定状态
- 人工复核结果

统一沉淀到 PostgreSQL，可供 `offline_top10_qc.py` 等下游脚本直接使用。

## 目标

建立两张新表：

1. `annotations.label_studio_task_source_identity`
2. `annotations.label_studio_task_source_identity_candidates`

并提供一个全量构建脚本：

- 覆盖 `annotations.label_studio_tasks_v2` 全量任务
- 为每个 `task_id` 写入当前主身份结论
- 为每个 `task_id` 写入 TopN 候选证据
- 保留 `244` 条重点复核对象为 `REVIEW_REQUIRED`

同时让 `src/occupation_retrieval/offline_top10_qc.py` 优先读取该映射层，而不是直接信任任务表中的 `recruitment_record_id`。

## 非目标

- 本轮不直接覆盖 `annotations.label_studio_tasks_v2.recruitment_record_id`
- 本轮不设计新的前端复核系统
- 本轮不引入 MCP / API 层
- 本轮不处理超出 `annotations.label_studio_tasks_v2` 的其他标注表

## 数据模型

### 主表：`annotations.label_studio_task_source_identity`

一行一个 `task_id`，表示“当前主结论”。

建议字段：

- `task_id` 主键，外键到 `annotations.label_studio_tasks_v2.id`
- `identity_status`
  - `AUTO_CONFIRMED`
  - `REVIEW_REQUIRED`
  - `UNMATCHED`
  - `MANUAL_CONFIRMED`
  - `MANUAL_REJECTED`
- `identity_rule`
- `selected_rank`
- `selected_score`
- `selected_recruitment_record_id`
- `selected_source_platform`
- `selected_source_table`
- `selected_source_row_number`
- `selected_source_native_job_id`
- `selected_dedupe_fingerprint`
- `selected_company_name`
- `selected_work_city`
- `selected_salary_raw`
- `selected_publish_date`
- `selected_job_title`
- `selected_job_description_raw`
- `task_job_title`
- `task_job_requirements`
- `task_sample_source`
- `task_snapshot_row_id`
- `task_payload_fingerprint`
- `resolver_version`
- `review_decision`
- `review_notes`
- `reviewed_by`
- `reviewed_at`
- `created_at`
- `updated_at`

### 候选表：`annotations.label_studio_task_source_identity_candidates`

一行一个 `task_id + candidate_rank`，表示候选证据层。

建议字段：

- `task_id`
- `candidate_rank`
- `candidate_score`
- `is_selected`
- `selection_reason`
- `recruitment_record_id`
- `source_platform`
- `source_table`
- `source_row_number`
- `source_native_job_id`
- `dedupe_fingerprint`
- `job_title`
- `company_name`
- `work_city`
- `salary_raw`
- `publish_date`
- `job_description_raw`
- `education_requirement_raw`
- `experience_requirement_raw`
- `company_industry_raw`
- `created_at`

主键建议为 `(task_id, candidate_rank)`。

## 状态机

### 自动构建阶段

- `AUTO_CONFIRMED`
  - 唯一高置信候选
- `REVIEW_REQUIRED`
  - 有候选，但不能唯一确认
- `UNMATCHED`
  - 找不到可信候选

### 人工复核阶段

- `REVIEW_REQUIRED -> MANUAL_CONFIRMED`
  - 人工指定最终候选
- `REVIEW_REQUIRED -> MANUAL_REJECTED`
  - 人工确认当前不应落到任何源记录
- `REVIEW_REQUIRED -> REVIEW_REQUIRED`
  - 暂不判定

### 重跑规则

自动重跑可以刷新：

- `AUTO_CONFIRMED`
- `REVIEW_REQUIRED`
- `UNMATCHED`

自动重跑不能覆盖：

- `MANUAL_CONFIRMED`
- `MANUAL_REJECTED`

## 构建逻辑

### 输入

- `annotations.label_studio_tasks_v2`
  - `id`
  - `row_id`
  - `sample_source`
  - `job_title`
  - `job_requirements`
  - `data_raw` / `data_raw_jsonb`
- `public.recruitment_jobs_normalized`
- 三家 sample 表：
  - `"51job".sample`
  - `"Liepin".sample`
  - `"Zhilian".sample`

### 任务侧 source of truth

优先从任务当前 `data_raw` 提取：

- `job_title`
- `job_requirements_clean`
- `sample_source`
- `row_id`

不再信任旧 `row_id -> source_row_number` 的直接映射。

### 候选生成

沿用当前修复脚本思路：

1. 用标准化标题在 sample 表中取同标题候选
2. 用 `job_requirements_clean` 与 sample `岗位描述` 计算匹配分数
3. 取得分排序后的 TopN 候选
4. 写入候选表

### 主结论生成

- 若唯一高分命中，主表写 `AUTO_CONFIRMED`
- 若多候选同分或差距不足，主表写 `REVIEW_REQUIRED`
- 若无可信候选，主表写 `UNMATCHED`

### 任务载荷指纹

新增 `task_payload_fingerprint`，基于以下字段构建稳定 hash：

- `task_job_title`
- `task_job_requirements`
- `task_sample_source`
- `task_snapshot_row_id`

用途：

- 检测任务载荷后续是否发生变化
- 为后续增量刷新提供 cheap-change detection

## 与 `offline_top10_qc.py` 的对接

### 新读取口径

`offline_top10_qc.py` 不再直接信任 `annotations.label_studio_tasks_v2.recruitment_record_id`。

改为：

1. 读取标注任务
2. 关联 `annotations.label_studio_task_source_identity`
3. 优先使用以下状态的 `selected_recruitment_record_id`
   - `AUTO_CONFIRMED`
   - `MANUAL_CONFIRMED`
4. 对 `REVIEW_REQUIRED` / `UNMATCHED` / `MANUAL_REJECTED` 单独统计

### 报告增强

报告中新增：

- 身份映射状态分布
- 因身份未确认而排除的样本数
- `REVIEW_REQUIRED` 样本数
- `UNMATCHED` 样本数

这样可以把“检索性能问题”和“任务源身份不确定问题”分开看。

## 实现边界

### 新增模块

- `src/db/label_studio_task_source_identity.py`
  - 建表 / upsert / 查询 helper
- `src/utils/build_label_studio_task_source_identity.py`
  - 全量构建 CLI

### 修改模块

- `src/occupation_retrieval/common.py`
  - 为任务加载增加可选身份映射接入
- `src/occupation_retrieval/offline_top10_qc.py`
  - 切到新映射层并增加状态统计

### 测试

- 新增主表/候选表 helper 单元测试
- 新增构建逻辑测试
- 扩展 `test_occupation_retrieval_offline_top10_qc.py`
  - 验证 `REVIEW_REQUIRED` / `UNMATCHED` 的统计与排除逻辑

## 风险与规避

### 风险 1：244 条被错误自动压成单值

规避：

- 主表与候选表分离
- `REVIEW_REQUIRED` 不自动写死单值

### 风险 2：下游脚本混用旧 `recruitment_record_id`

规避：

- `offline_top10_qc.py` 显式优先读取身份表
- README 更新身份口径

### 风险 3：重跑覆盖人工结果

规避：

- `MANUAL_CONFIRMED` / `MANUAL_REJECTED` 自动保护

## 验收标准

完成后应满足：

1. PostgreSQL 中存在两张新表
2. 全量 `18611` 条任务在主表中都有记录
3. 当前 `244` 条重点样本在主表中标记为 `REVIEW_REQUIRED`
4. 候选表已写入每条任务的 TopN 候选证据
5. `offline_top10_qc.py` 能读取身份表并输出新口径报告
6. 测试通过，且至少完成一次全量构建 dry-run / real-run 验证
