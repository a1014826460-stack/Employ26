# `src/occupation_retrieval` 目录说明
# @ occupation_retrieval 2026-06-08
## 1. 目录定位

`src/occupation_retrieval/` 目前是一组围绕“第二轮人工标注数据”和“RAG 检索模型微调”的实验脚本集合，主要用于：

- 复现第二轮数据有效性分析
- 分析人工标注与 DeepSeek 重标结果的分歧
- 基于不同样本筛选策略微调 BGE 检索模型
- 对多个微调版本做离线对比评估

这批脚本整体更接近“实验工作台”，而不是已经模块化、可长期维护的正式流水线。它们之间存在较多重复逻辑，适合先保留实验结论，再逐步收敛成公共模块。

## 2. 共用输入与产出

大多数脚本依赖以下数据：

- 标注数据：PostgreSQL `annotations.label_studio_tasks_v2`
- 其中历史任务的正式招聘身份应优先读取 `recruitment_record_id`；旧 `row_id` 只表示导出快照行号，不再代表招聘源主键
- 在 `src/occupation_retrieval/common.py` 的公共加载口径里，`task_id` 只表示标注任务身份，不能替代 `recruitment_record_id`
- DeepSeek 重标结果：优先 PostgreSQL `annotations.deepseek_relabel_raw`，首次缺表或空表时会优先从 `output/deepseek_relabel/round2/round2_deepseek_relabel_raw.jsonl` 回填；旧路径 `output/deepseek_relabel/deepseek_relabel_raw.jsonl` 仅作兼容兜底
- 职业词典：PostgreSQL `public.occ_dict_unified`
- 基础向量模型：`config.paths.get_project_paths().bge_model_path` 或环境变量 `EMPLOYDATA_BGE_MODEL_PATH`

常见输出位置：

- Markdown 报告：`output/occupation_retrieval/*.md`
- JSON 结果：`output/occupation_retrieval/*.json`
- 微调模型与评估文件：`output/occupation_retrieval/rag_round2_training/`
- 离线 Top10 质检报告：`output/occupation_retrieval/offline_top10_qc_report.md`
- 任务源身份主表：`annotations.label_studio_task_source_identity`
- 任务源身份候选表：`annotations.label_studio_task_source_identity_candidates`
- 数据库归档清单：PostgreSQL `archive_occ.archive_manifest`
- 人工与 DeepSeek 分歧复核 CSV：`output/occupation_retrieval/human_deepseek_disagreements_20260622.csv`
- 归档与导出 SQL：`output/occupation_retrieval/sql/archive_deprecated_annotations_20260622.sql`、`output/occupation_retrieval/sql/export_human_deepseek_disagreements_20260622.sql`

## 2.1 四项任务的端到端执行方案

本节把当前目录里的实验脚本整理成四个可复现任务。除非特别说明，所有命令都在项目根目录执行，并固定以下运行前提：

- Python 入口：优先使用项目环境 `.\.conda\python.exe -m ...`；如果当前 shell 已激活同一环境，也可以用 `python -m ...`。
- 字符编码：Windows PowerShell 下建议先执行 `$env:PYTHONIOENCODING='utf-8'`，避免中文报告输出乱码。
- 基础 BGE 模型：优先读取环境变量 `EMPLOYDATA_BGE_MODEL_PATH`；未设置时读取 `config.paths.get_project_paths().bge_model_path`；两者都不可用时使用 HuggingFace 模型名 `BAAI/bge-large-zh-v1.5`。
- 随机种子：训练脚本默认 `random_seed=42`，需要复现实验时不要修改该值。
- 权威任务身份：`task_id` 只表示 Label Studio 标注任务；跨表关联、日志追踪和样本去重优先使用 `recruitment_record_id`。
- 人工 gold 口径：多标注任务优先使用超过 50% 的多数意见；单标注任务使用唯一人工选择；`NONE` 表示 A-E 候选都不合适，通常不进入检索模型正样本训练。

### 任务 1：复现第二轮数据有效性分析

目标：复现第二轮人工标注数据的整体质量判断，回答“这批标注是否足够用于分析和训练”。

数据来源：

- 人工标注任务：PostgreSQL `annotations.label_studio_tasks_v2`，由 `load_annotations_from_pg()` 读取 `annotations_completed` 和 `data_raw`。
- DeepSeek 重标结果：PostgreSQL `annotations.deepseek_relabel_raw`，为空或缺表时由 `ensure_deepseek_table()` 从 `output/deepseek_relabel/round2/round2_deepseek_relabel_raw.jsonl` 回填。
- 候选来源字段：每条任务 `data_raw` 中的 `candidate_a_source` 到 `candidate_e_source`，用于判断人工选择是否来自 raw RAG Top1-Top5。
- 验证样本标记：`data_raw.is_validation_sample == "1"`。

预处理步骤：

1. 读取全部任务，并解析每条任务的 `annotations_completed` 为标注列表。
2. 对每条 annotation 调用 `extract_choice()` 或公共 `parse_choice()`，把原始选择规范化为 `A/B/C/D/E/NONE/None`。
3. 按任务聚合标注，得到 `single_tasks` 和 `multi_tasks`。
4. 对多标注任务计算多数意见：若最高票数 `top_count > 标注数 / 2`，则认为存在多数意见；否则记为无明确多数。
5. 对 `NONE` 单独计数；在 DeepSeek 对齐、训练正样本、TopK 命中等需要职业代码的指标里排除 `NONE`。
6. 读取 DeepSeek 重标结果，并按 `task_id` 与人工任务对齐。
7. 识别验证样本子集，单独计算分布、一致性和 DeepSeek 对齐指标。

执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.reproduce_round2_validity
```

输出文件：

- 主报告：`output/occupation_retrieval/round2_validity_report.md`

有效性指标和计算方法：

- 有效选择率：人工标注中选择 `A-E` 的比例。公式：`valid_count / total_annotations`。用于判断候选集是否经常覆盖真实职业。
- `NONE` 比例：人工标注中选择 `NONE` 的比例。公式：`none_count / total_annotations`。比例过高说明 A-E 候选召回不足或任务文本质量差。
- 完全一致率：多标注任务中所有标注员选择完全相同的任务比例。公式：`full_agree_tasks / multi_annotator_tasks`。
- pairwise agreement：同一任务内任意两个标注员选择相同的 pair 占全部标注员 pair 的比例，再对任务求平均。单任务公式：`agree_pairs / C(n, 2)`。
- 多数意见存在率：多标注任务中存在超过半数相同选择的比例。公式：`majority_tasks / multi_annotator_tasks`。
- RAG Top1 命中率：人工有效选择等于原始候选中来源含 `top1` 的候选字母时记为命中。公式：`human_chose_top1 / human_valid_choices`。
- RAG Top1-5 累计命中率：人工有效选择在来源含 `top1` 到 `top5` 的候选内时记为命中。公式：`human_chose_top1_to_top5 / human_valid_choices`。
- DeepSeek-人工一致率：DeepSeek 选择与人工参考选择相同的比例。单标注参考为唯一人工选择，多标注参考为多数意见。公式：`deepseek_agree / compared_tasks`。
- DeepSeek 置信度分桶准确率：按 `deepseek_confidence` 四舍五入或分桶后，分别计算 DeepSeek 与人工参考一致的比例。
- 标注员跟随多数率：某标注员在多标注任务中与该任务多数意见一致的比例。公式：`agree_majority_count / multi_annotation_count`。
- 验证样本一致性：仅在 `is_validation_sample=1` 子集上重复计算完全一致率、pairwise agreement 和 DeepSeek-人工一致率。

判定标准：

- 可训练：多数意见存在率高，人工有效选择率高，且 `NONE` 比例没有异常上升；这说明 A-E 候选和人工选择能形成稳定正样本。
- 可用于统一评估：验证样本和多标注样本的 pairwise agreement、完全一致率不能明显低于全量样本，否则评估集自身不稳定。
- 需要复核：DeepSeek 高置信分桶仍大量不一致、`NONE` 比例异常、某些标注员跟随多数率显著偏低、RAG Top1-5 命中率偏低。
- 不把单一指标当最终结论：例如 RAG Top1 低不等于数据无效，它可能只说明未微调检索器排序较弱；应同时看 Top1-5、人工一致性和 DeepSeek 对齐。

### 任务 2：分析人工标注与 DeepSeek 重标结果的分歧

目标：量化人类与 DeepSeek 的不一致，并把分歧拆成可解释的错误类型，服务后续样本筛选、人工复核和模型训练。

分歧定义：

- 任务级分歧：在同一 `task_id` 上，人工参考选择 `human_choice` 与 `deepseek_choice` 都存在且都不是 `NONE`，但二者字母不同。
- 代码级分歧：人工参考选择映射到的 `human_code` 与 DeepSeek 选择映射到的 `deepseek_code` 不同。
- 层级分歧：代码不同但可能属于同一小类、中类或大类，需要用职业代码前缀判断分歧严重程度。
- 语义排序分歧：人工选择和 DeepSeek 选择在 BGE 全量职业检索排序中的名次不同，用于判断模型语义空间更支持哪一方。

量化指标：

- 不一致率：`disagree_tasks / compared_tasks`。其中 `compared_tasks` 只包含人工参考和 DeepSeek 选择都有效的任务。
- 一致率：`agree_tasks / compared_tasks`，等于 `1 - 不一致率`。
- Cohen's Kappa：衡量两方标注一致性扣除随机一致后的程度，适合“人工参考 vs DeepSeek”两个标注源。公式：`kappa = (p_o - p_e) / (1 - p_e)`，其中 `p_o` 是观察一致率，`p_e = sum_c(p_human(c) * p_deepseek(c))`，`c` 遍历 `A/B/C/D/E/NONE` 或只遍历有效 `A-E`，但报告中必须注明口径。
- Fleiss' Kappa：衡量多个标注员之间的一致性，适合多人工标注员，不适合直接替代人工-DeepSeek两方比较。
- 分歧中偏向人类比例：统一横评中，当人类与 DeepSeek 不一致时，如果模型给人类选择的 rank 小于 DeepSeek 选择的 rank，记为偏向人类。公式：`human_side_count / ds_disagreement_total`。
- 层级距离分布：`hierarchy_distance()` 输出 `0/1/2/3/4`，分别表示同细类、同小类不同细类、同中类不同小类、同大类不同中类、完全不同大类。
- 语义排名优势：在分歧样本中比较 `human_semantic_rank` 和 `deepseek_semantic_rank`，rank 更小的一方被 BGE 语义空间更支持。

执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.disagreement_deep_analysis
```

输出文件：

- 分歧报告：`output/occupation_retrieval/disagreement_analysis.md`

对比流程：

1. 读取人工标注、DeepSeek 重标和职业词典。
2. 解析人工选择；多标注任务取多数意见，单标注任务取唯一选择。
3. 将 `human_choice` 和 `deepseek_choice` 映射为职业代码和职业标题。
4. 用基础 BGE 模型编码全部职业文本，得到职业向量库。
5. 对每条岗位 anchor 编码，计算它与全部职业文本的相似度排序。
6. 找到人工职业代码的 `sem_rank` 和 DeepSeek 职业代码的 `ds_sem_rank`。
7. 计算人类与 DeepSeek 的层级距离。
8. 按分歧类型、语义排名、DeepSeek 置信度、标注人数、是否选择 RAG Top1 等维度汇总。
9. 输出代表性案例，供人工复核。

差异分类体系：

- DeepSeek 可能错：人工选择有多标注多数支持，人工代码语义排名靠前，例如 `sem_rank <= 5`，而 DeepSeek 排名更靠后。
- 人工可能错：人工选择语义排名很低，例如 `sem_rank >= 100`，DeepSeek 选择语义排名靠前，例如 `ds_sem_rank <= 20`，且 DeepSeek 置信度较高。
- 双方都可能错：人工和 DeepSeek 的语义排名都很低，例如二者都 `>= 50`，说明 A-E 内部可能没有理想答案或岗位文本不足。
- 边界模糊：人工和 DeepSeek 都在语义 Top10 内，但选择不同，通常是相近职业细类的边界问题。
- 跨领域分歧：层级距离 `>= 3`，表示至少跨中类或跨大类，优先进入人工复核。
- 候选排序诱导分歧：人工是否选了 raw RAG Top1 与分歧率明显相关，说明候选展示顺序或召回候选质量可能影响人工判断。

分歧原因分析方法：

- 看层级距离：距离小通常是职业细类边界模糊，距离大通常是岗位理解、候选召回或标注错误。
- 看语义排名：如果一方 rank 明显更小，说明当前 BGE 语义空间更支持该方，但不能直接等同于真值。
- 看 DeepSeek 置信度：高置信分歧优先人工复核，因为它可能暴露人工错误，也可能暴露 DeepSeek 系统性偏差。
- 看标注人数和 pairwise agreement：多标注且 pairwise 高的人工选择可信度更高；单标注样本需要更谨慎。
- 看岗位标题关键词和职业大类：岗位标题暗示的大类与所选职业大类冲突时，优先检查候选代码。
- 看候选来源：如果人工经常偏离 Top1 但仍在 Top5，说明模型召回可能够用但排序不足；如果不在 Top5，说明候选召回不足。

### 任务 3：基于不同样本筛选策略微调 BGE 检索模型

目标：比较不同训练样本构造方式对 BGE 检索模型的影响，找出兼顾标签质量、样本多样性和业务指标的训练策略。

共同训练设置：

- 输入 anchor：`job_title + job_requirements_clean`。
- positive 文本：职业词典中对应 code 的 `title`、`desc`、`tasks` 拼接文本。
- 基础模型：默认 `BAAI/bge-large-zh-v1.5` 或本地 `EMPLOYDATA_BGE_MODEL_PATH`。
- 训练框架：`SentenceTransformer`。
- 损失函数：`MultipleNegativesRankingLoss`。
- 默认超参数：`batch_size=32`、`learning_rate=2e-5`、`max_seq_length=256`、`warmup_ratio=0.1`、`random_seed=42`。
- 训练设备：`get_runtime_device()` 自动选择 `cuda` 或 `cpu`。
- 模型输出目录：`output/occupation_retrieval/rag_round2_training/`。

策略 A：全量人工正样本基线（v1）

- 脚本：`src.occupation_retrieval.train_rag_round2`
- 数据构成：单标注有效 A-E 样本直接入候选池；多标注任务要求存在超过 50% 的多数意见；排除 `NONE`、缺职业代码、职业代码不在词典中的样本。
- 切分方式：多标注任务全部进入测试集，单标注任务按 `test_ratio=0.15` 抽入测试集，其余训练。
- 超参数：`epochs=2`，其余使用共同训练设置。
- 执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.train_rag_round2 `
  --output-model-name v1-bge-large `
  --run-label v1-bge-large
```

- 输出：模型目录 `output/occupation_retrieval/rag_round2_training/v1-bge-large`，评估文件 `evaluation_v1_bge_large.json`。
- 适用判断：作为冻结基线或新底座挑战的第一条线，优点是保留真实分布和长尾样本，风险是包含人工噪声。

策略 B：人工审核/多数意见 + DeepSeek 一致筛选（Gold/Silver，v3）

- 脚本：`src.occupation_retrieval.train_rag_round2_v3`
- Gold 数据：多标注任务中人工多数意见与 DeepSeek 选择一致。
- Silver 数据：单标注任务中人工选择与 DeepSeek 选择一致。
- Excluded 数据：DeepSeek 与人工不一致、人工为 `NONE`、无明确多数或无法映射职业代码的样本。
- 训练数据构成：只使用 Gold + Silver。
- 测试数据构成：未进入 Gold/Silver 的其他有效人工样本。
- 超参数：`epochs=3`，其余使用共同训练设置。
- 执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.train_rag_round2_v3
```

- 输出：模型目录 `output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned-v3`，评估文件 `evaluation_v3.json`。
- 适用判断：用于测试“减少标签噪声是否提升性能”。风险是训练集变小，并引入 DeepSeek 偏好。

策略 C：分歧 + 语义排名筛选（v4）

- 脚本：`src.occupation_retrieval.train_rag_round2_v4`
- 正样本：DeepSeek 与人工一致，且人工职业代码的 `semantic_rank <= 10`。
- hard negative 诊断集：DeepSeek 与人工分歧，且人工职业代码的 `semantic_rank >= 30`。
- 中间样本：不满足正样本或 hard negative 条件的样本，主要进入测试池。
- 训练数据构成：正样本按 85%/15% 划分训练和正样本测试；hard negative 当前不进入 loss，只用于诊断。
- 超参数：`epochs=3`、`pos_semantic_rank_max=10`、`neg_semantic_rank_min=30`，其余使用共同训练设置。
- 执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.train_rag_round2_v4
```

- 输出：模型目录 `output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned-v4`，评估文件 `evaluation_v4.json`。
- 适用判断：用于测试“DeepSeek 一致性 + 底座语义支持”是否能得到更干净样本。风险是偏向容易样本，减少困难边界学习。

策略 D：多维质量分层加权（weighted）

- 脚本：`src.occupation_retrieval.train_rag_weighted`
- 筛选信号：标注人数、多数意见、pairwise agreement、DeepSeek 一致性、DeepSeek 置信度、BGE 语义排名、岗位标题关键词与职业大类匹配、标注员历史质量、`NONE` 比例、是否选择 raw RAG Top1。
- 样本层级：`S/A/B/C/D`，分数越高标签越可信。
- 训练数据构成：`D` 级丢弃；`S/A/B/C` 按过采样倍数加入训练。
- 默认过采样：`S=10`、`A=7`、`B=3`、`C=1`、`D=0`。
- 超参数：`epochs=3`，其余使用共同训练设置。
- 执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.train_rag_weighted
```

- 输出：模型目录 `output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned-weighted`，评估文件 `evaluation_weighted.json`。
- 适用判断：用于测试“保留更多样本但提高高质量样本出现概率”是否优于硬过滤。风险是过采样会放大某些高频职业或规则偏差。

策略 E：DeepSeek 高置信加法采样（v5）

- 脚本：`src.occupation_retrieval.train_rag_round2_v5`
- 数据构成：保留 v1 全量人工正样本；DeepSeek 与人工一致且 `deepseek_confidence >= 0.80` 的样本额外过采样；只有 DeepSeek 但无有效人标的任务在 `deepseek_confidence >= 0.90` 时作为训练对；DeepSeek 与人工不一致但极高置信的样本默认不加入。
- 默认超参数：`epochs=3`、`ds_agree_oversample=2`、`ds_disagree_oversample=0`。
- 执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.train_rag_round2_v5 `
  --output-model-name v5-bge-large `
  --run-label v5-bge-large `
  --ds-agree-oversample 2 `
  --ds-disagree-oversample 0 `
  --epochs 3
```

- 输出：模型目录 `output/occupation_retrieval/rag_round2_training/v5-bge-large`，评估文件 `evaluation_v5_bge_large.json`。
- 适用判断：用于测试“DeepSeek 做加法而不是做减法”是否优于 v3/v4。风险是高置信 DeepSeek 错误会被重复放大。

策略 F：主动学习采样 + 人工复核闭环

- 当前状态：本目录还没有单独的主动学习训练脚本；可用 `disagreement_deep_analysis.py` 和 `multidim_validation.py` 产出待复核样本，再把人工复核后的标签并入 `annotations.label_studio_tasks_v2` 或单独沉淀为下一轮训练表。
- 样本筛选规则：优先抽取 `human_choice != deepseek_choice`、`semantic_rank >= 50`、`ds_sem_rank <= 20`、层级距离 `>= 3`、多模型 Top1 不一致、或人工 gold 未进 Top10 的样本。
- 人工复核规则：每条样本至少由 2 名熟悉职业分类的复核员判断；若两人不一致，追加第 3 人；最终 gold 采用超过 50% 的复核多数意见。
- 训练数据构成：保留 v1 全量人工正样本；追加复核通过样本；对“高不确定但复核确认”的困难样本可过采样 2x；复核后仍无一致意见的样本进入诊断集，不进训练。
- 推荐超参数：沿用 v1 的 `batch_size=32`、`learning_rate=2e-5`、`max_seq_length=256`、`warmup_ratio=0.1`；首轮 `epochs=2`，如果新增复核样本超过原训练集 20%，再尝试 `epochs=3`。
- 评估指标：统一横评的 `candidate_acc`、`mrr`、`ds_side_human_pct`、`subclass_acc`；另外单独统计主动学习复核子集上的 Top1、Top5、MRR，观察困难样本是否改善。
- 迭代机制：每轮从当前最强模型的错误样本中抽取 200-500 条复核；训练新版本；统一横评；若总体指标不降且困难子集提升，再进入下一轮。

评估指标：

- 训练脚本自测指标：Top1、Top3、Top5、Top10，表示人工 gold 职业代码是否出现在全量职业检索前 K 名。
- 多标注子集 Top1：只在多标注测试样本上计算 Top1，作为相对高可信评估切片。
- negative set Top10 hit rate：在疑似负样本上，模型仍把人工选择放进 Top10 的比例；对 v4 的 hard negative 诊断来说越低越符合“该人工选择可疑”的假设。
- 统一横评指标：以 `eval_models_multimetric.py` 为最终准绳，包括候选选择准确率、平均人工 rank、分歧中偏向人类比例、MRR、层级准确率。

模型迭代机制：

1. 固定一个变量：每轮只改变底座模型、样本筛选规则、超参数中的一类，避免无法归因。
2. 每个版本必须保存模型目录、训练配置、训练集/测试集构成和评估 JSON。
3. 自测通过后，把模型加入统一横评，而不是直接比较不同训练脚本各自的 `evaluation_*.json`。
4. 若新版本在 `candidate_acc` 和 `mrr` 同时不低于 v1，再看 `ds_side_human_pct` 和层级准确率。
5. 若新版本只提升大类准确率但降低细类准确率，通常不能替换生产默认模型，只能作为召回或复核辅助。
6. 对失败版本做归因：检查训练样本数是否过少、是否偏向容易样本、是否丢失长尾职业、是否引入 DeepSeek 偏差、测试集是否与基线不同口径。

### 任务 4：多个微调版本离线对比评估结果

目标：在同一评估集、同一指标、同一职业词典下横向比较 baseline、v1、v3、v4、weighted、v5 或新挑战模型，避免被各训练脚本不同 test split 误导。

对比实验设计：

- 基线模型：默认 `baseline (bge-large)`，即未微调的基础 BGE 模型。
- 冻结生产候选：默认 `v1 (全量)`，即 `bge-large-round2-finetuned`。
- 对比模型：`v3 (Silver/Gold)`、`v4 (Medium分歧)`、weighted、v5、bge-m3 或其他新版本。
- 评估数据集：从 `annotations.label_studio_tasks_v2` 读取全部有岗位要求、有人工有效选择、且至少能构造 3 个候选的样本。
- 人工参考：多标注任务使用出现次数最多的人工选择，单标注任务使用唯一选择；当前统一横评脚本不强制要求多标注必须超过 50%，因此解释报告时要注明这一点。
- 职业词典：`public.occ_dict_unified`。
- DeepSeek 对照：`annotations.deepseek_relabel_raw`，只用于分歧仲裁指标，不作为最终 gold。
- 抽样：MRR 和层级准确率默认最多随机抽 `3000` 条样本，随机种子固定为 `np.random.seed(42)`。

执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.eval_models_multimetric `
  --model weighted=output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned-weighted `
  --model v5-bge-large=output/occupation_retrieval/rag_round2_training/v5-bge-large
```

如果不传 `--model`，脚本默认评估：

- `v1 (全量)`
- `v3 (Silver/Gold)`
- `v4 (Medium分歧)`
- `baseline (bge-large)`

输出文件：

- 横评报告：`output/occupation_retrieval/model_comparison.md`

离线评估指标：

- Candidate Accuracy：在 Label Studio A-E 候选中，模型排序第一的候选字母是否等于人工参考字母。公式：`candidate_hit / candidate_total`。
- Mean Human Rank：人工参考候选在 A-E 候选相似度排序中的平均名次。越接近 1 越好。
- Disagreement Human-side Ratio：在人类与 DeepSeek 分歧样本中，模型把人类选择排在 DeepSeek 选择前面的比例。公式：`ds_side_human / ds_total`。
- MRR：Mean Reciprocal Rank，平均倒数排名。对每条样本取人工 gold 职业在全量职业排序中的名次 `rank`，计算 `1 / rank`，再求平均；未进 Top50 时脚本按 `rank=51` 处理。
- Recall@k：也称 TopK Accuracy，正确职业出现在全量检索前 K 名即命中。公式：`hit_at_k / total`。训练脚本常输出 Top1/Top3/Top5/Top10。
- Subclass Accuracy：模型 Top1 职业与人工 gold 的职业代码前三段一致的比例。
- Midclass Accuracy：模型 Top1 职业与人工 gold 的职业代码前两段一致的比例。
- Major Accuracy：模型 Top1 职业与人工 gold 的职业代码第一段一致的比例。
- Rank#1-Rank#5 分布：人工参考候选在 A-E 候选排序中分别位于第 1 到第 5 的比例，用于观察模型排序质量。

结果对比框架：

| 模型 | 训练策略 | 训练样本口径 | Candidate Acc ↑ | Mean Rank ↓ | MRR ↑ | Subclass ↑ | Midclass ↑ | Major ↑ | Disagreement Human-side ↑ | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 未微调 BGE | 无训练 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 判断是否需要微调 |
| v1 | 全量人工正样本 | 保留真实分布 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 当前冻结基线 |
| v3 | Gold/Silver | 强过滤高一致样本 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 判断降噪是否有效 |
| v4 | 分歧 + 语义排名 | 容易正样本 + 负样本诊断 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 判断语义筛选是否过强 |
| weighted | 质量分层加权 | 高质量样本过采样 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 判断加权是否有效 |
| v5 | DS 高置信加法 | v1 + DS 一致过采样 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 运行横评后记录 | 判断 DeepSeek 加法是否有效 |

性能差异归因：

- v1 优于过滤方案：通常说明样本多样性和长尾覆盖比强过滤更重要。
- v3/v4 大类准确率提升但细类/MRR 下降：通常说明模型学到了粗粒度方向，但失去细类区分能力。
- v4 不如 v1：可能因为 `semantic_rank <= 10` 偏向容易样本，训练集过小，且 hard negative 未真正进入 loss。
- weighted 不如 v1：可能因为过采样放大了高频职业、规则偏差或训练分布与评估分布不一致。
- v5 不如 v1：可能因为 DeepSeek 高置信错误被重复采样，或 DS 一致样本本身集中在容易职业。
- bge-m3 不如 bge-large：可能是底座模型领域适配、中文职业词典表达、向量维度或训练配方没有重新调参导致。
- Candidate Acc 与 MRR 方向不一致：说明模型在 A-E 局部候选排序和全量职业检索排序上的能力不同，应按业务入口选择主指标。

## 2.2 统一术语定义

- BGE：BAAI General Embedding 的简称，是北京智源/BAAI 发布的一组通用文本向量模型；本项目用它把岗位文本和职业词典文本编码到同一向量空间。
- BGE-large：本项目当前冻结基线使用的 BGE 大模型版本，默认路径为 `bge-large-zh-v1.5` 或本地等价模型目录。
- BGE-M3 / bge-m3：BGE 系列的多语言、多粒度、多功能 embedding 模型，在本项目中作为挑战底座而不是当前生产默认模型。
- DeepSeek：用于第二轮重标和分歧复核的大语言模型或模型服务；本项目把它作为“第二意见”，不直接等同于绝对真值。
- DeepSeek 重标：用 DeepSeek 对同一批 Label Studio 任务重新选择 A-E 或 `NONE`，并保存选择、置信度和理由。
- DeepSeek 置信度：DeepSeek 输出中表示它对自己选择把握程度的数值，字段通常为 `deepseek_confidence`。
- PostgreSQL：关系型数据库，本项目用它保存人工标注任务、DeepSeek 重标结果和职业词典。
- JSONL：每行一个 JSON 对象的文本格式，适合保存流式或批量模型输出；DeepSeek 重标结果有 JSONL 兜底文件。
- HuggingFace：常用模型托管和加载生态；当本地模型路径不存在且名称形如 `BAAI/bge-large-zh-v1.5` 时，SentenceTransformer 可按 HuggingFace 模型名加载。
- SentenceTransformer：用于加载 embedding 模型、编码文本向量和微调检索模型的 Python 框架。
- SQLAlchemy：Python 数据库访问库，本项目通过它连接 PostgreSQL。
- pandas：Python 表格数据处理库，本项目用它读取职业词典表并转换为 DataFrame。
- Label Studio：人工标注平台；本项目的第二轮人工标注任务来自 Label Studio 导出的 PostgreSQL 表。
- 第二轮数据：`annotations.label_studio_tasks_v2` 中的人工标注任务集合，是当前 occupation retrieval 实验的主要训练和评估来源。
- RAG：Retrieval-Augmented Generation，检索增强生成；在本目录中更多指“先检索职业候选，再供人工或 LLM 判断”的检索流程。
- 检索模型：输入岗位文本，输出相似职业文本排序的 embedding 模型。
- BGE 检索模型：以 BGE 为底座并针对岗位-职业匹配任务微调后的检索模型。
- embedding：文本向量表示，把文本编码成数值向量，便于用相似度计算语义接近程度。
- 向量空间：embedding 所在的数学空间，语义相近的文本通常距离更近或方向更接近。
- anchor：训练和检索中的查询文本，本项目通常是 `job_title + job_requirements_clean`。
- positive：与 anchor 匹配的正确职业文本，通常由职业词典的 `title`、`desc`、`tasks` 拼接而成。
- negative：与 anchor 不匹配的职业文本，用于训练或诊断模型区分能力。
- hard negative：困难负样本，表面语义相近但不是正确答案，常用于提升模型细粒度区分能力。
- in-batch negatives：同一个训练 batch 中其他样本的 positive 被自动当作当前 anchor 的负样本。
- MultipleNegativesRankingLoss：SentenceTransformer 中的对比学习损失函数，目标是让每个 anchor 更接近自己的 positive，而远离 batch 内其他 positive。
- 微调：在已有预训练模型基础上，用本项目标注数据继续训练，使模型更适合岗位-职业匹配任务。
- 样本筛选策略：决定哪些样本进入训练、测试、诊断或被丢弃的一组规则，例如置信度筛选、人工审核筛选、主动学习采样。
- 置信度筛选：按 DeepSeek 置信度、标注一致性或质量分数阈值选择样本的策略。
- 人工审核筛选：优先保留多标注多数意见、人工复核通过或标注员一致性高的样本。
- 主动学习采样：优先挑选模型不确定、分歧大、可能带来最大信息增益的样本交给人工复核，再把复核结果加入训练。
- Gold：高可信训练样本；本项目 v3 中指多标注任务里人工多数意见与 DeepSeek 一致的样本。
- Silver：中高可信训练样本；本项目 v3 中指单标注任务里人工选择与 DeepSeek 一致的样本。
- Excluded：被排除的样本，通常因为人工与 DeepSeek 不一致、没有有效人工选择、缺职业代码或无法形成多数意见。
- tier：样本质量层级，例如 `S/A/B/C/D`。
- oversampling：过采样，把某些样本重复加入训练集，提高它们在训练中出现的概率。
- weighted training：加权训练；本项目当前用 oversampling 近似实现样本权重，而不是直接修改 loss 权重。
- 训练集：用于更新模型参数的数据。
- 测试集：训练过程中不用于更新模型参数、只用于评估模型效果的数据。
- 验证样本：`is_validation_sample=1` 的样本子集，用于单独观察标注质量或评估稳定性。
- 离线评估：不调用线上服务、不改变生产系统，只在固定数据集上批量计算指标的评估方式。
- baseline：基线模型或基线配方，用来和新模型比较；本项目同时保留未微调 BGE baseline 和 v1 冻结训练基线。
- v1：全量人工正样本训练方案，是当前冻结的生产候选基线。
- v3：Gold/Silver 噪声过滤方案。
- v4：DeepSeek 分歧与语义排名结合的中等过滤方案。
- v5：保留 v1 全量人工样本，并对 DeepSeek 高置信一致样本做加法过采样的方案。
- weighted：基于多维质量分层并过采样的训练方案。
- A-E 候选：Label Studio 任务中展示给人工标注员的五个职业候选。
- NONE：人工或 DeepSeek 判断 A-E 候选都不合适时的选择。
- gold：评估时作为正确答案的参考标签；本项目多数场景以人工选择或人工多数意见作为 gold。
- majority vote / 多数意见：同一任务多个标注选择中票数最多且超过半数的选择。
- pairwise agreement：同一任务内任意两名标注员选择相同的比例。
- full agreement / 完全一致：同一任务所有标注员选择完全相同。
- Inter-Annotator Agreement：标注员间一致性，衡量多名标注员对同一批任务判断是否稳定。
- Cohen's Kappa：两个标注源的一致性指标，会扣除随机一致的影响。
- Fleiss' Kappa：多个标注员的一致性指标，会扣除随机一致的影响。
- accuracy / 准确率：预测正确数量除以总数量。
- Candidate Accuracy：模型在 A-E 候选中排第一的候选与人工 gold 相同的比例。
- TopK retrieval：取相似度最高的前 K 个职业候选。
- TopK Accuracy：正确职业出现在全量检索前 K 名的比例。
- Recall@k：召回率@k，与 TopK Accuracy 在本项目的单正确答案检索场景中等价。
- Hit@k：命中率@k，含义同正确答案是否出现在前 K 名。
- MRR：Mean Reciprocal Rank，平均倒数排名，正确答案越靠前分数越高。
- NDCG@k：Normalized Discounted Cumulative Gain at k，归一化折损累计增益，衡量前 k 个排序结果的相关性和位置质量；当前统一横评脚本注释中提到该指标，但报告主表暂未正式输出。
- reciprocal rank：单条样本的倒数排名，公式为 `1 / rank`。
- Mean Human Rank：人工选择在模型排序中的平均名次，越低越好。
- semantic rank：某个职业代码在 BGE 全量职业相似度排序中的名次。
- cosine similarity：余弦相似度，衡量两个向量方向接近程度。
- dot product：点积；当向量已归一化时，点积可等价用作余弦相似度。
- normalize_embeddings：向量归一化，把向量缩放到单位长度，方便用点积计算相似度。
- CUDA：NVIDIA GPU 并行计算平台；可用于加速模型训练和向量编码。
- CPU：中央处理器；无 GPU 时脚本会回退到 CPU，但训练速度会更慢。
- max_seq_length：模型最多读取的 token 序列长度，超出部分会被截断。
- batch_size：每次送入模型训练或编码的一批样本数量。
- epoch：训练轮数，表示模型完整遍历训练集一次。
- learning_rate：学习率，控制每次参数更新的步长。
- warmup_ratio：预热比例，训练初期逐步增大学习率以稳定优化。
- random_seed：随机种子，用于固定随机切分、打乱顺序和抽样结果。
- test_ratio：从可训练样本中划入测试集的比例。
- 数据契约：脚本对输入表、字段、含义和解析方式的固定约定。
- recruitment_record_id：正式招聘记录身份，用于跨表追踪同一招聘来源。
- task_id：Label Studio 标注任务身份，不能替代招聘源主键。
- occupation dict / 职业词典：`public.occ_dict_unified`，包含职业代码、标题、定义和任务描述。
- 职业代码层级：职业代码按 `-` 分段，前 1 段表示大类，前 2 段表示中类，前 3 段表示细类或小类映射口径。
- Subclass Accuracy：职业代码前三段一致的比例。
- Midclass Accuracy：职业代码前两段一致的比例。
- Major Accuracy：职业代码第一段一致的比例。
- 层级距离：人类与 DeepSeek 所选职业代码在职业层级上的距离，数值越大表示分歧越严重。
- 离线 Top10 QC：固定 Top10 检索结果上的离线质检流程，用于检查 Top10 是否覆盖人工 gold，以及二阶段 LLM 选择是否有帮助。
- LLM-over-Top10：先由 BGE 给出 Top10，再让 LLM 在 Top10 内选择最终 winner 的二阶段流程。
- model comparison：多模型统一横评报告，输出到 `output/occupation_retrieval/model_comparison.md`。

## 3. 脚本总览

| 脚本 | 主要作用 | 典型输出 | 当前定位 |
| --- | --- | --- | --- |
| `reproduce_round2_validity.py` | 复现第二轮数据有效性分析 | `output/occupation_retrieval/round2_validity_report.md` | 数据集整体体检 |
| `deep_analysis_round2.py` | 统计任务级/标注级 TopK 命中与多数意见情况 | `output/occupation_retrieval/deep_analysis_round2.md` | 轻量分析脚本 |
| `disagreement_deep_analysis.py` | 深挖人类与 DeepSeek 分歧模式 | `output/occupation_retrieval/disagreement_analysis.md` | 分歧诊断 |
| `multidim_validation.py` | 用多信号给样本打质量分层 | `output/occupation_retrieval/multidim_validation_report.md`、`output/occupation_retrieval/multidim_validation_results.json` | 标注质检 |
| `offline_top10_qc.py` | 离线比较 raw Top1、LLM-over-Top10 与人工 gold，并输出分布与分歧参考 | `output/occupation_retrieval/offline_top10_qc_report.md` | Top10 离线质检 |
| `train_rag_round2.py` | v1：直接用第二轮标注训练基础检索模型 | 模型目录、`output/occupation_retrieval/rag_round2_training/evaluation_results.json` | 基线微调方案 |
| `train_rag_round2_v3.py` | v3：用 Gold/Silver 样本训练 | 模型目录、`output/occupation_retrieval/rag_round2_training/evaluation_v3.json` | 噪声过滤方案 |
| `train_rag_round2_v4.py` | v4：按分歧与语义排名筛正负样本 | 模型目录、`output/occupation_retrieval/rag_round2_training/evaluation_v4.json` | 中等强度过滤方案 |
| `train_rag_weighted.py` | 置信分层加权训练 | 模型目录、`output/occupation_retrieval/rag_round2_training/evaluation_weighted.json` | 质量加权方案 |
| `eval_models_multimetric.py` | 比较 baseline、v1、v3、v4 多项指标 | `output/occupation_retrieval/model_comparison.md` | 模型横向评估 |

## 3.1 当前冻结基线

当前路线已经明确把“基线”定义成一套可复现配方，而不是单个模型目录。正式说明见：

- 基线配方文档：[docs/occupation-retrieval-baseline.md](/d:/PythonProjects/Employ26/docs/occupation-retrieval-baseline.md)
- ADR: [docs/adr/0002-freeze-occupation-retrieval-baseline-on-v1-task-table-recipe.md](/d:/PythonProjects/Employ26/docs/adr/0002-freeze-occupation-retrieval-baseline-on-v1-task-table-recipe.md)

当前冻结结论：

- 基线训练配方以 `train_rag_round2.py` 的 `v1` 为准
- 基线数据契约暂时冻结在 `annotations.label_studio_tasks_v2`
- 当前输入字段口径仍是 `annotations_completed` + `data_raw` 的任务主表解析结果
- 职业词典冻结为 `public.occ_dict_unified`，字段以 `code`、`title`、`desc`、`tasks` 为训练和评估文本来源
- `annotations_completed_jsonb`、`data_raw_jsonb` 与 `annotations.v_label_studio_task_annotations_v2` 属于后续可挑战的“数据契约升级方案”，不与基线冻结动作同时进行
- 正式定胜负以 `eval_models_multimetric.py` 及 `output/occupation_retrieval/model_comparison.md` 为准
- 当前综合最强模型是 `output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned`

当前最小 CLI 契约：

- `train_rag_round2.py`
  - `--base-model-path`
  - `--output-model-name`
  - `--run-label`
- `eval_models_multimetric.py`
  - `--model NAME=PATH`，可重复传入

命名约定从下一次新跑开始生效：

- 推荐目录名：`配方名 + 底座名`
- 例如：
  - `v1-bge-large`
  - `v1-bge-m3`
- 历史目录如 `bge-large-round2-finetuned` 先保留，不主动重命名

## 4. 各脚本说明

### `reproduce_round2_validity.py`

作用：

- 复现第二轮数据集有效性报告
- 统计单标注/多标注分布、pairwise agreement、majority 存在率
- 对 DeepSeek 与人类多数意见的一致性做对照
- 单独分析 `is_validation_sample=1` 的验证样本

优点：

- 覆盖面最全，适合先了解第二轮数据质量
- 输出结构完整，适合当作目录中的总览入口

不足：

- 报告逻辑集中在一个超长 `main()` 中，可复用性弱
- 统计口径写死在脚本里，不支持参数化选择数据文件或输出位置
- 没有把关键统计过程拆成公共函数，后续别的脚本重复实现了相似逻辑

建议修复：

- 拆分为“数据加载 / 标注聚合 / 指标统计 / 报告写出”四层函数
- 增加命令行参数，允许指定输入文件和输出文件
- 把 majority、pairwise agreement、DeepSeek 对齐统计抽到公共工具模块

### `deep_analysis_round2.py`

作用：

- 基于任务多数意见统计 RAG 候选 Top1-Top5 命中率
- 区分单标注任务、多标注任务、逐标注样本三种口径
- 给出 `NONE` 选择比例和多标注平均跟随率

优点：

- 逻辑相对独立，适合快速回答“候选召回质量是否够用”
- 已使用 `pathlib`，结构比其他脚本更简洁

不足：

- 结果只打印到控制台，没有沉淀报告文件
- 数据文件名仍然硬编码到脚本常量
- 与其他脚本重复解析 Label Studio 标注格式

建议修复：

- 增加文本或 JSON 输出
- 改成从统一配置读取数据源
- 复用公共 `parse_choice` / `load_annotations` 工具函数

### `disagreement_deep_analysis.py`

作用：

- 识别“人类与 DeepSeek 不一致”的任务
- 从语义排名、层级距离、职业大类冲突、候选排名等角度分析错误模式
- 输出分歧样本特征统计和代表案例

优点：

- 对“分歧来自哪里”解释性较强
- 已经开始利用职业层级信息，而不是只看 Top1 准确率

不足：

- 采用逐条 `model.encode([anchor])` 的方式做语义分析，运行成本较高
- 使用追加写文件并在开始时删除旧文件，流程可读性一般
- 与 `multidim_validation.py`、`eval_models_multimetric.py` 共用的大量词典与分歧逻辑没有抽象

建议修复：

- 批量编码 anchor，减少重复推理
- 改成先收集 `output_lines`，最后一次性写文件
- 抽出职业层级距离、DeepSeek 对齐分析等公共函数

### `multidim_validation.py`

作用：

- 为每条样本计算多维质量信号
- 综合语义排名、大类关键词、标注员一致性、DeepSeek 一致性、标注员历史质量、`NONE` 比例等信号
- 输出质检报告和结构化 JSON 结果

优点：

- 是本目录里最接近“样本质量评分器”的脚本
- 输出了 JSON，便于后续继续筛样本或可视化

不足：

- 单脚本承担了特征工程、评分、案例展示、文件输出多个职责
- 大类关键词硬编码在脚本中，不方便调参与版本管理
- 每条任务单独编码 anchor，性能较差
- 缺少对评分规则的显式配置说明，复现实验时不够透明

建议修复：

- 把关键词字典迁移到 `dicts/` 或配置文件
- 把“信号计算”和“tier 判定”拆成独立模块
- 增加评分配置对象，避免阈值散落在代码中
- 批量向量化计算 semantic rank

### `train_rag_round2.py`

作用：

- 作为 v1 基线方案，直接从第二轮标注中抽取正样本对
- 默认把多标注任务整体放入测试集，单标注任务按比例切分
- 用 `MultipleNegativesRankingLoss` 微调基础 BGE 模型
- 输出微调模型与评估 JSON

优点：

- 流程清晰，是后续各版本训练脚本的起点
- 有相对完整的训练/测试切分与 baseline 对比

不足：

- 只使用正样本，没有显式 hard negative
- 训练、评估、数据抽取都写在同一个脚本里
- 测试集可能包含训练集中未覆盖的职业，结果会受分布影响
- 代码里仍有部分“BGE-M3”字样，但实际模型目录命名是 `bge-large`，命名不够统一

建议修复：

- 明确模型命名，避免 `BGE-M3` 与 `bge-large` 混用
- 把数据构造、训练、评估拆分
- 为测试切分策略补充固定配置和说明

### `train_rag_round2_v3.py`

作用：

- 只保留 DeepSeek 与人类一致的数据
- 多标注一致样本记为 Gold，单标注一致样本记为 Silver
- 用 Gold + Silver 训练，并把其余有效样本作为测试集

优点：

- 核心思路简单，噪声控制直观
- 比 v1 更强调标签可信度

不足：

- 测试集定义为“所有未入选 Gold/Silver 的样本”，天然更难，和 v1 结果并不完全同口径
- 仍与其他训练脚本重复了多数数据加载与解析逻辑
- 依赖 `output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned` 作为对比模型时，没有显式检查该模型是否存在

建议修复：

- 明确记录训练集/测试集口径，避免横向比较误读
- 在运行前检查对比模型目录和输入文件是否存在
- 与 v1/v4/weighted 共用一套数据准备模块

### `train_rag_round2_v4.py`

作用：

- 用 “DeepSeek 一致 + 语义排名靠前” 构造正样本
- 用 “DeepSeek 分歧 + 语义排名靠后” 识别 hard negative 候选
- 正样本参与训练，负样本和中间样本参与评估

优点：

- 比 v3 多引入了语义排名这一层过滤
- 负样本集合有助于分析模型是否在可疑样本上过拟合

不足：

- 在预编码和训练前初始化模型时把设备直接写成 `cuda`，无 GPU 环境会直接失败
- `compute_semantic_rank()` 对每条样本单独编码，数据量大时很慢
- “负样本不参与训练，只用于评估”的策略写在代码里，但没有抽成可配置实验参数

建议修复：

- 改为自动选择 `cuda` / `cpu`
- 批量计算 anchor embedding 和 semantic rank
- 把正负样本阈值与测试集采样规模改成可配置参数

### `train_rag_weighted.py`

作用：

- 基于多维信号给样本打 `S/A/B/C/D` 质量等级
- 通过 oversampling 近似实现不同样本权重
- 输出加权训练模型及评估结果

优点：

- 是本目录里最完整的“样本打分 -> 加权训练”方案
- 将样本质量问题直接映射到训练权重，实验方向明确

不足：

- 多处把设备直接写成 `cuda`，CPU 环境不可运行
- 语义排名仍然按样本逐条编码，开销大
- oversample 倍数写死在代码中，不利于复现实验和做网格搜索
- 训练集/测试集划分混合了规则筛选和随机补样，统计口径需要更清楚地记录

建议修复：

- 先修正硬编码 GPU 问题
- 将 oversample 系数、测试集规模、tier 规则参数化
- 输出训练集和测试集的样本构成摘要，便于后续复盘

### `eval_models_multimetric.py`

作用：

- 对 baseline、v1、v3、v4 多个模型做统一评估
- 评估候选命中、候选排序、人类与 DeepSeek 分歧仲裁、层级准确率、MRR 等指标
- 输出横向比较报告

优点：

- 是目录中最适合做“最终模型对比”的脚本
- 指标维度比单纯看 Top1 更完整

不足：

- `MODEL_PATHS` 写死在脚本里，新增模型需要手改代码
- 指标定义没有单独文档化，不同训练版本之间的比较口径容易混淆
- 没有检查模型目录是否存在，缺模型时会在运行阶段才失败

建议修复：

- 改为从参数或配置文件读取待评估模型列表
- 为每个指标补一段简短说明并沉淀为固定报告模板
- 运行前做输入文件、模型目录、依赖完整性检查

## 5. 脚本之间的关系

可以把这批脚本理解为三层：

1. 数据质量分析层  
   `reproduce_round2_validity.py`、`deep_analysis_round2.py`

2. 分歧与样本筛选层  
   `disagreement_deep_analysis.py`、`multidim_validation.py`、`offline_top10_qc.py`

3. 训练与评估层  
   `train_rag_round2.py`、`train_rag_round2_v3.py`、`train_rag_round2_v4.py`、`train_rag_weighted.py`、`eval_models_multimetric.py`

推荐阅读顺序：

1. 先看 `reproduce_round2_validity.py`
2. 再看 `multidim_validation.py` 和 `disagreement_deep_analysis.py`
3. 然后看 `train_rag_round2.py`、`train_rag_round2_v3.py`、`train_rag_round2_v4.py`、`train_rag_weighted.py`
4. 最后用 `eval_models_multimetric.py` 做横向对比

补充说明：

- 各训练脚本输出的 `evaluation_*.json` 主要用于各自实验内部诊断
- 这些文件的测试集口径不同，不适合作为 `v1 / v3 / v4 / weighted / bge-m3` 的正式横向比较依据
- 若要比较不同底座或不同训练方案，应统一回到 `eval_models_multimetric.py`
- `offline_top10_qc.py` 会同时保留 raw Top1 和 LLM-over-Top10 结果：先比较 Top10 是否命中 gold，再比较 `Top1 vs LLM winner vs human gold`；当 `LLM winner != human gold` 时，再实时调用 `deepseek-v4-pro` 看完整 Top10，明确输出 `support_llm` / `support_gold` / `support_neither`，并附带可选 LLM miss 复核。

## 5.2 V1 到 V4 的原理、技术栈与实验含义

### 共同技术栈

这几版模型本质上都在做同一件事：把“招聘岗位文本”和“职业细类词典文本”编码到同一个向量空间，然后用相似度排序找出最可能的职业细类。

涉及的核心组件如下：

- PostgreSQL：当前权威数据源。标注任务来自 `annotations.label_studio_tasks_v2`，DeepSeek 重标来自 `annotations.deepseek_relabel_raw`，职业细类词典来自 `public.occ_dict_unified`。
- SQLAlchemy / pandas：负责从 PostgreSQL 读取任务、重标结果和职业词典，再转成 Python 结构或 DataFrame。
- SentenceTransformer：训练和推理框架，用来加载 BGE 底座模型、生成文本 embedding，并执行微调。
- BGE：`BAAI` 系列通用 embedding 模型。当前冻结基线使用 `bge-large-zh-v1.5`，第一轮挑战模型是 `bge-m3`。
- embedding：文本向量表示。招聘文本和职业文本都被编码成向量，语义越接近，向量方向通常越接近。
- normalize_embeddings：向量归一化。归一化后，向量点积可以等价近似为 cosine similarity，方便用矩阵乘法快速排序。
- cosine similarity / dot product：相似度计算。代码中常见 `torch.mm(anchor_embeddings, occ_embeddings.T)`，表示一次性计算一批招聘文本与全部职业细类文本的相似度矩阵。
- TopK retrieval：取相似度最高的前 K 个职业细类，例如 Top1、Top3、Top5、Top10。
- MultipleNegativesRankingLoss：对比学习损失函数。一个 batch 里每条 `anchor-positive` 是正样本，其它样本的 positive 自动成为 in-batch negatives，训练目标是让正确职业文本比其它职业文本更靠前。
- hard negative：困难负样本。语义上容易混淆、但不应作为正确答案的候选，用来诊断或增强模型区分能力。
- CUDA / CPU：向量编码和训练可以在 GPU 或 CPU 上运行；当前公共工具会自动选择可用设备。

### V1：全量正样本基线

`train_rag_round2.py` 是当前冻结基线。

它的核心假设是：第二轮人工标注虽然有噪声，但整体信号足够强，保留更多样本多样性比过早过滤更重要。

V1 的数据构造方式：

- 对每条任务读取 `job_title` 和 `job_requirements_clean`，拼成 anchor：`岗位名称 + 岗位要求`。
- 从人工标注中读取 `A-E` 选择，忽略 `NONE`。
- 单标注任务直接采用该标注选择。
- 多标注任务要求存在明显多数意见，再采用多数意见。
- 根据选择字母找到对应 `candidate_x_code`。
- 在 `public.occ_dict_unified` 中找到该 code 的 `title`、`desc`、`tasks`，拼成 positive 文本。

V1 的训练方式：

- 训练样本是 `(anchor, positive)` 正样本对。
- 使用 `MultipleNegativesRankingLoss`。
- 默认 `batch_size=32`、`epochs=2`、`learning_rate=2e-5`、`max_seq_length=256`。
- 多标注任务全部进入测试集，单标注任务按 `test_ratio=0.15` 进入测试集，其余进入训练集。

V1 的意义：

- 它不是“最复杂”的方案，但它保留了最多真实标注分布。
- 它给后续 V3、V4、weighted、bge-m3 挑战提供同一条可复现基准线。

### V3：Gold / Silver 噪声过滤方案

`train_rag_round2_v3.py` 的目标是降低人工标注噪声。

它引入 DeepSeek 重标作为第二意见：

- Gold：多标注任务中，人类多数意见与 DeepSeek 选择一致。
- Silver：单标注任务中，人类选择与 DeepSeek 选择一致。
- Excluded：人类与 DeepSeek 不一致，或无法形成有效选择的样本。

V3 的核心假设是：人类和 DeepSeek 一致时，标签可信度更高。

V3 的优点：

- 训练集标签更“干净”。
- 适合验证“过滤噪声是否能提升模型”。

V3 的风险：

- 过滤会损失大量边界样本和长尾样本。
- DeepSeek 的偏好会被引入训练集，模型可能更像 DeepSeek，而不一定更像人工标注共识。
- V3 自带 `evaluation_v3.json` 的测试集是“未进入 Gold/Silver 的有效样本”，与 V1 的测试集不完全同口径，因此不能只看这个 JSON 判断 V3 是否超过 V1。

### V4：DeepSeek 分歧 + 语义排名的中等过滤方案

`train_rag_round2_v4.py` 进一步加入了语义排名。

它先用底座模型计算人工答案在全量职业词典中的 semantic rank：

- semantic rank = 正确职业 code 按相似度排序后的名次。
- rank 越小，说明底座模型本来就越容易把它排到前面。

V4 的样本规则：

- 正样本：DeepSeek 与人类一致，并且人工答案 `semantic_rank <= 10`。
- hard negative 诊断集：DeepSeek 与人类分歧，并且人工答案 `semantic_rank >= 30`。
- 中间样本：不满足上述条件的样本，主要进入测试池。

V4 的核心假设是：如果 DeepSeek 同意，而且底座模型语义上也认为它靠前，这个样本更可信；如果 DeepSeek 不同意，而且底座模型也认为人工答案很靠后，这个样本可能是疑似噪声或困难样本。

V4 的优点：

- 比 V3 多了一层 embedding 语义证据。
- 可以产出 hard negative 诊断集，帮助定位模型容易混淆的样本。
- 对后续构造更强训练集有启发价值。

V4 的风险：

- 训练正样本明显变少，当前 `evaluation_v4.json` 中 `train_pos=1437`。
- 语义排名来自底座模型，过滤规则会放大底座模型已有偏好。
- `semantic_rank <= 10` 会偏向“容易样本”，削弱模型学习困难边界的机会。
- hard negative 当前主要用于评估诊断，并没有真正作为训练负样本进入 loss。

### Weighted：质量分层加权方案

`train_rag_weighted.py` 不是严格的 V2/V3/V4 顺序版本，而是另一条思路：不简单丢弃样本，而是给样本分层加权。

它会根据人工一致性、DeepSeek 一致性、语义排名、样本质量信号把样本分成 `S/A/B/C/D` 等级，然后用 oversampling 近似实现权重。

涉及概念：

- tier：样本质量层级。
- oversampling：把高质量样本重复加入训练集，提高它们在训练中的出现概率。
- weighted training：广义上的加权训练；当前实现不是直接给 loss 加权，而是通过重复采样近似实现。

Weighted 的意义是保留更多样本，同时强调高可信样本。但当前统一横评中它没有超过 V1，因此暂不作为生产基线。

## 5.3 如何检测模型效果

本目录有两类评估结果，必须区分。

第一类是训练脚本自带评估：

- `evaluation_results.json`：V1 自测。
- `evaluation_v3.json`：V3 自测。
- `evaluation_v4.json`：V4 自测。
- `evaluation_weighted.json`：weighted 自测。
- `evaluation_v1_bge_m3.json`：V1 配方 + bge-m3 底座自测。

这些文件适合看“该脚本内部是否训练成功”和“相对未微调底座提升多少”，但因为 test split 不完全一致，不适合直接定最终胜负。

第二类是统一横评：

- 入口：`python -m src.occupation_retrieval.eval_models_multimetric`
- 报告：`output/occupation_retrieval/model_comparison.md`

统一横评会把多个模型放到同一批评估样本和同一套指标下比较，因此更适合作为最终选择依据。

主要指标解释如下：

- Candidate Accuracy：在 Label Studio 原始 `A-E` 候选中，模型把哪一个候选排第一。如果模型 Top1 字母等于人工选择字母，就算命中。公式是 `candidate_hit / candidate_total`。
- Mean Human Rank：人工选择在模型排序中的平均名次。1 表示模型总是把人工选择排第一，越低越好。
- Disagreement Human-side Ratio：当人类和 DeepSeek 不一致时，比较模型排序更靠近人类还是更靠近 DeepSeek。若人类选择的 rank 小于 DeepSeek 选择的 rank，记为偏向人类。
- MRR：Mean Reciprocal Rank，平均倒数排名。若正确职业排第 1，得分 1；排第 2，得分 1/2；排第 10，得分 1/10；没进 Top50 时按脚本记为较低分。公式是 `mean(1 / rank)`。
- Subclass Accuracy：职业代码前 3 段一致即视为细类层级命中，例如 `4-04-05` 层级一致。
- Midclass Accuracy：职业代码前 2 段一致即视为中类命中。
- Major Accuracy：职业代码第 1 段一致即视为大类命中。
- TopK Accuracy：全量职业检索时，正确 code 是否出现在前 K 个候选中。Top10 通常高于 Top5，但给 Agent 的最终默认输出仍建议保留 Top1，同时保存 Top10 候选供复核和后续推理使用。

统一横评当前结论：

| 指标 | 当前最优 | 数值 |
| --- | --- | --- |
| 候选选择准确率 | V1 | 57.4% |
| 人类选择平均排位 | V1 | 1.706 |
| 分歧中偏向人类比例 | V1 | 60.7% |
| 全量检索 MRR | V1 | 0.677 |
| 细类准确率 | V1 | 58.6% |
| 中类准确率 | V1 | 67.2% |
| 大类准确率 | V3 | 76.2% |

## 5.4 为什么最终选择 V1，而不是 V4

最终选择 V1，不是因为 V1 理论上最复杂，而是因为它在统一横评里赢了最关键的业务指标。

当前 `output/occupation_retrieval/model_comparison.md` 显示：

- V1 候选选择准确率 57.4%，V4 为 52.8%。
- V1 MRR 为 0.677，V4 为 0.606。
- V1 细类准确率 58.6%，V4 为 53.3%。
- V1 中类准确率 67.2%，V4 为 62.4%。
- V1 分歧中偏向人类比例 60.7%，V4 为 42.8%。

这些指标更贴近当前目标：让 Agent 给出更准确的职业细类 Top10，并且默认 Top1 更可靠。

V4 理论上“看起来应该更优”的原因：

- 它同时使用人工标注、DeepSeek 重标和底座语义排名。
- 它试图过滤低质量样本。
- 它引入 hard negative 思路，符合检索模型常见优化方向。

但 V4 实际没有最优，主要原因是：

- 过滤过强，正样本只剩 `1437`，相比 V1 丢失了大量真实分布和长尾职业信号。
- `semantic_rank <= 10` 偏向容易样本，模型对困难边界的学习不足。
- DeepSeek 一致性不是绝对真值，会把 DeepSeek 的判断偏差带入训练集。
- hard negative 当前没有真正进入 `MultipleNegativesRankingLoss` 的训练目标，只是作为 held-out 诊断集。
- V4 的测试集和 V1 自测口径不同，必须回到统一横评判断，而统一横评显示 V1 更稳。

因此，V4 不是废弃方案，而是诊断与下一轮实验的素材。若要让 V4 真正挑战 V1，建议下一步把 hard negative 显式纳入训练，或降低过滤强度，让它既保留样本多样性，又能利用分歧样本。

## 5.5 当前该使用哪个模型

当前建议：

- 生产默认使用：`output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned`
- 配方名称：`v1-bge-large`
- 数据源：PostgreSQL `annotations.label_studio_tasks_v2` + `public.occ_dict_unified`
- 评估依据：`eval_models_multimetric.py` 的统一横评结果

`bge-m3` 第一轮挑战结果接近 V1 但没有超过 V1：

- V1-bge-m3 候选选择准确率 56.8%，低于 V1-bge-large 的 57.4%。
- V1-bge-m3 MRR 0.664，低于 V1-bge-large 的 0.677。
- V1-bge-m3 细类准确率 57.7%，低于 V1-bge-large 的 58.6%。

所以当前不建议用 bge-m3 替换冻结基线。bge-m3 可以继续作为挑战者，但应在同一数据契约、同一训练配方、同一统一评估下继续比较。

## 5.6 Agent API：Top10 检索 + llama 报告

当前已经新增一个轻量 HTTP API，作为职业细类识别 Agent 的补充能力。

入口：

- 服务文件：`src/llm/occupation_agent_api.py`
- 共享业务逻辑：`src/llm/occupation_agent_service.py`
- 默认 fine-tuned 模型：`output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned`
- 默认 TopK：`10`
- LLM 后端：复用 `config/vllm.toml` 中配置的本地 llama.cpp / vLLM OpenAI 兼容服务

启动本地 LLM 服务：

```powershell
.\.conda\python.exe -m src.llm.vllm_server serve
```

另开一个终端启动职业细类 Agent API：

```powershell
.\.conda\python.exe -m src.llm.occupation_agent_api --host 127.0.0.1 --port 8120
```

请求示例：

```powershell
$body = @{
  job_title = "算法工程师"
  job_description = "负责推荐算法、用户画像、模型训练和效果评估"
  top_k = 10
  include_llm_report = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8120/occupation/analyze" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

返回内容包括：

- `top1`：默认建议采用的职业细类
- `top10_candidates`：BGE 检索得到的 Top10 候选、分数、职业层级路径
- `llm_report`：llama 对 Top10 的解释、候选比较、风险提示和 Agent 使用建议

如果只想要 BGE Top10，不调用 llama：

```json
{
  "job_title": "算法工程师",
  "job_description": "负责推荐算法、用户画像、模型训练和效果评估",
  "top_k": 10,
  "include_llm_report": false
}
```

## 5.7 第一轮 `bge-m3` 挑战运行方式

推荐按“单变量挑战”执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.train_rag_round2 `
  --base-model-path D:\model\bge-m3 `
  --output-model-name v1-bge-m3 `
  --run-label v1-bge-m3
```

训练完成后，用统一评估把挑战者加入比较：

```powershell
$env:EMPLOYDATA_BGE_MODEL_PATH='D:\model\bge-large-zh-v1.5'
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.occupation_retrieval.eval_models_multimetric `
  --model v1-bge-m3=output/occupation_retrieval/rag_round2_training/v1-bge-m3
```

说明：

- 第一轮挑战只替换底座模型，不同时修改样本构造、split、超参数和统一评估口径
- 若 `bge-m3` 要取代当前基线，应至少在 `candidate_acc` 与 `mrr` 上同时不弱于当前 `v1`

## 5.8 离线 Top10 QC 报告

如果只想做离线分析和质检，不改训练和评估口径，建议直接跑：

```powershell
python -m src.occupation_retrieval.offline_top10_qc
```

默认会输出：

- `Hit@1 / Hit@3 / Hit@5 / Hit@10`
- `HIT@10 / MISS@10 / GOLD_NONE / MISSING_GOLD / MISSING_MATCH`
- `Top1 accuracy / LLM-over-Top10 accuracy / LLM better than Top1`
- gold 职业细类分布、占比、均匀性判断
- Top10 miss 样本摘录

如需加入本地 LLM 二阶段选择和复核：

```powershell
python -m src.occupation_retrieval.offline_top10_qc --use-llm-review
```

当前 Top10 二阶段选择仍走项目统一入口 `src.model_platform.llm.create_llm_client()`，实际后端由 `config/model_runtime.yaml` 决定；默认是本地 `wsl_vllm`，有 fallback 才会尝试外部 API。DeepSeek 已收口到共享入口 `src.llm.deepseek_client`，供 `offline_top10_qc.py` 的实时二裁和 `src.anno_analysis.deepseek_round2_relabel` 复用；其中离线质检只在 `LLM winner != human gold` 时触发 DeepSeek，再看完整 Top10 做支持 LLM / 支持 gold / 两边都不理想的判定。

## 6. 当前收敛状态

本目录已经完成一轮 P0/P1 级别的低风险工程收敛，目标是减少口径漂移和运行时隐性失败，同时不改变 frozen baseline 的训练配方、评估口径和默认模型路径。

已完成：

- `common.py` 统一负责 PostgreSQL 数据读取、DeepSeek 回填读取、职业词典读取、输出目录、训练输出目录、运行设备和 CUDA 清理。
- `datasets.py` 统一负责 Label Studio 选择解析、任务选择列表、多数意见、候选 A-E 记录构造和 anchor 文本构造。
- `metrics.py` 统一负责候选命中、倒数排名、层级命中和模型指标汇总。
- `train_rag_round2.py`、`eval_models_multimetric.py`、`deep_analysis_round2.py` 已接入公共解析/指标工具。
- `disagreement_deep_analysis.py`、`multidim_validation.py`、`train_rag_round2_v3.py`、`train_rag_round2_v4.py`、`train_rag_weighted.py` 已复用公共 `parse_choice()`，减少标注解析口径漂移。
- `eval_models_multimetric.py` 已对候选排序阶段的 anchor 做批量编码，并在加载本地模型前提供更清晰的路径缺失错误。
- 已补充最小单元测试，覆盖候选选择解析、多标注 majority、候选记录构造、anchor 构造和核心指标汇总。

仍保留为实验脚本的部分：

- `v1 / v3 / v4 / weighted` 的训练样本构造和筛选规则仍分别保留在各自脚本中，以保证历史实验可复现。
- tier 规则、大类关键词、oversample 倍数等实验规则仍在对应脚本中维护，暂不抽成统一配置。
- 部分诊断脚本仍有逐条语义排名计算；后续只有在频繁重跑或运行时间成为瓶颈时再批量优化。
- CLI 入口保持兼容，暂不增加新的数据源切换参数。

## 7. 后续改进建议

若后续继续把本目录提升为长期维护模块，建议按下面顺序推进：

1. 将 tier 规则、语义排名阈值、测试集抽样规模、oversample 系数抽成显式配置对象。
2. 给分析脚本补充 `--output-file` 和 `--limit` 等轻量 CLI 参数，方便小样本复核。
3. 将仍然频繁运行的逐条 `model.encode([anchor])` 改成批量编码。
4. 只保留 1 个总览分析入口、1 个样本筛选入口、1 个基础训练入口和 1 个多模型评估入口；其他历史版本转入 `archive/` 或标注为 `experimental/`。

## 8. 建议的后续归档方式

如果这批实验已经完成阶段性结论，后续建议做一次轻量整理：

- 保留 1 个总览分析入口
- 保留 1 个样本筛选入口
- 保留 1 个基础训练入口
- 保留 1 个多模型评估入口
- 其他历史版本转入 `archive/` 或标注为 `experimental/`

这样可以减少“v1 / v3 / v4 / weighted”继续并列扩散，避免后续维护成本持续上升。

### 8.1 已执行的数据库归档与复核样本导出

2026-06-22 已对 PostgreSQL `Employ26` 中 `annotations` schema 做了一次保守归档：

- `annotations.label_studio_annotations` 已完整复制到 `archive_occ.label_studio_annotations__20260622_001500`，原表重命名为 `annotations.label_studio_annotations_archived_20260622_001500`，行数为 `5170`。
- `annotations.label_studio_tasks_v2_rrid_backup_20260618_022807` 已完整复制到 `archive_occ.label_studio_tasks_v2_rrid_backup_20260618_022807__20260622_001500`，原表重命名为 `annotations.label_studio_tasks_v2_rrid_backup_20260618_022807_archived_20260622_001500`，行数为 `18611`。
- 归档 manifest 记录在 `archive_occ.archive_manifest`，包含归档原因、检测规则、源表行数、归档表行数和回滚 SQL。
- 可复用脚本：`output/occupation_retrieval/sql/archive_deprecated_annotations_20260622.sql`。

同日已导出人工标注与 DeepSeek 重标结果的分歧复核 CSV：

- 文件：`output/occupation_retrieval/human_deepseek_disagreements_20260622.csv`
- 编码：UTF-8 with BOM
- 记录数：`11893` 条非“相同”记录
- 列：`职位名称`、`任职要求`、`人工标注原始结果`、`DeepSeek标注结果`、`差异情况`、`Top5候选`
- 数据口径：当前数据库不存在 `annotation_results` 表，因此导出查询由 `annotations.label_studio_tasks_v2` 与 `annotations.deepseek_relabel_raw` 联结生成。
- 可复用查询：`output/occupation_retrieval/sql/export_human_deepseek_disagreements_20260622.sql`。


