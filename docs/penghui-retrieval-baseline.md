# Penghui Retrieval Baseline

当前冻结的 Penghui 检索基线不是单个模型目录，而是一套可重复执行的实验配方。后续任何 `bge-m3` 或其他方案都应先在这套固定配方上做同口径挑战，再讨论是否替换基线。

## 1. 基线目标

- 优先优化 `Retrieval Effectiveness`
- 以统一离线评估结果判定优劣
- 当前默认挑战对象是新的底座模型，而不是同时改动多项实验规则

## 2. 冻结的数据契约

### 标注任务

- 表：`annotations.label_studio_tasks_v2`
- 字段口径：
  - `id`
  - `annotations_completed`
  - `data_raw`
- 当前基线通过 Python 解析 `annotations_completed` 与 `data_raw`，重建任务级 `annotations` / `data` 结构

说明：

- 这是当前 `src.penghui.common.load_annotations_from_pg()` 和 `src.penghui.train_rag_round2` 的实际输入契约
- 虽然数据库里已经存在 `annotations_completed_jsonb`、`data_raw_jsonb` 和视图 `annotations.v_label_studio_task_annotations_v2`，但它们不属于当前已冻结基线的一部分
- 如果未来切换到 `jsonb` 字段或标注明细视图，应作为新的挑战方案处理，而不是宣称“基线没变”

### DeepSeek 重标结果

- 表：`annotations.deepseek_relabel_raw`
- 关键字段：
  - `task_id`
  - `deepseek_choice`
  - `deepseek_confidence`
  - `deepseek_reasoning`
  - `payload`

说明：

- 该表在当前基线中主要用于分歧分析和部分训练变体
- 如果表为空，可由 `output/deepseek_relabel/deepseek_relabel_raw.jsonl` 回填，但回填动作不改变基线表契约

### 职业词典

- 表：`public.occ_dict_unified`
- 关键字段：
  - `code`
  - `title`
  - `desc`
  - `tasks`

## 3. 冻结的 v1 训练配方

### 基础模型

- 底座：`bge-large`
- 解析方式：`config.paths.get_project_paths().bge_model_path` 或环境变量 `EMPLOYDATA_BGE_MODEL_PATH`

### 正样本构造

- 仅保留满足以下条件的任务：
  - `job_requirements_clean` 非空
  - 存在至少一个非 `NONE` 的人工选择
  - 若是多标注任务，则要求存在明显多数意见
  - 多数意见对应的候选 code 非空
  - 对应 code 能在 `public.occ_dict_detailed` 中找到职业文本
- anchor 文本：
  - `"{job_title} {job_requirements_clean}"`
- positive 文本：
  - `title`
  - 可拼接 `desc`
  - 可拼接 `tasks`

### train/test split

- 多标注任务全部进入测试集
- 单标注任务按 `test_ratio = 0.15` 抽样进入测试集
- 其余单标注任务进入训练集
- 随机种子：`42`

### 训练超参数

- `batch_size = 32`
- `epochs = 2`
- `learning_rate = 2e-5`
- `max_seq_length = 256`
- `warmup_ratio = 0.1`
- loss：`MultipleNegativesRankingLoss`

### 当前基线模型产物

- 模型目录：`output/penghui/rag_round2_training/bge-large-round2-finetuned`
- 局部评估结果：`output/penghui/rag_round2_training/evaluation_results.json`

## 4. 冻结的统一评估标准

正式优劣判断不看各训练脚本自带的 `evaluation_*.json`，而看统一评估脚本：

- 入口：`python -m src.penghui.eval_models_multimetric`
- 报告：`output/penghui/model_comparison.txt`

统一评估样本口径：

- 使用 `annotations.label_studio_tasks_v2` 中可构建有效候选比较的任务
- 当前 `Eval samples = 13867`
- MRR 和层级准确率在这批样本中固定抽样 `3000` 条，随机种子为 `42`

当前冻结结论：

- `v1 (全量)` 是当前综合最强基线
- `bge-m3` 尚未成为已验证挑战者，必须先在本配方上完成同口径训练与评估

## 5. 挑战者规则

未来挑战基线时，优先一次只变更一个主变量：

- 可先只改底座模型，例如 `bge-large -> bge-m3`
- 不应同时改变：
  - 数据契约
  - 样本构造规则
  - split 规则
  - 训练超参数
  - 统一评估口径

如果需要同时改变多个变量，应明确标注为新实验配方，而不是基线延伸。
