# `src/penghui` 目录说明
# @ PengHui 2026-06-08
## 1. 目录定位

`src/penghui/` 目前是一组围绕“第二轮人工标注数据”和“RAG 检索模型微调”的实验脚本集合，主要用于：

- 复现第二轮数据有效性分析
- 分析人工标注与 DeepSeek 重标结果的分歧
- 基于不同样本筛选策略微调 BGE 检索模型
- 对多个微调版本做离线对比评估

这批脚本整体更接近“实验工作台”，而不是已经模块化、可长期维护的正式流水线。它们之间存在较多重复逻辑，适合先保留实验结论，再逐步收敛成公共模块。

## 2. 共用输入与产出

大多数脚本依赖以下数据：

- 标注数据：PostgreSQL `annotations.label_studio_tasks_v2`
- 其中历史任务的正式招聘身份应优先读取 `recruitment_record_id`；旧 `row_id` 只表示导出快照行号，不再代表招聘源主键
- 在 `src/penghui/common.py` 的公共加载口径里，`task_id` 只表示标注任务身份，不能替代 `recruitment_record_id`
- DeepSeek 重标结果：优先 PostgreSQL `annotations.deepseek_relabel_raw`，首次缺表或空表时会从 `output/deepseek_relabel/deepseek_relabel_raw.jsonl` 回填
- 职业词典：PostgreSQL `public.occ_dict_unified`
- 基础向量模型：`config.paths.get_project_paths().bge_model_path` 或环境变量 `EMPLOYDATA_BGE_MODEL_PATH`

常见输出位置：

- 文本报告：`output/penghui/*.txt`
- JSON 结果：`output/penghui/*.json`
- 微调模型与评估文件：`output/penghui/rag_round2_training/`

## 3. 脚本总览

| 脚本 | 主要作用 | 典型输出 | 当前定位 |
| --- | --- | --- | --- |
| `reproduce_round2_validity.py` | 复现第二轮数据有效性分析 | `output/penghui/round2_validity_report.txt` | 数据集整体体检 |
| `deep_analysis_round2.py` | 统计任务级/标注级 TopK 命中与多数意见情况 | `output/penghui/deep_analysis_round2.txt` | 轻量分析脚本 |
| `disagreement_deep_analysis.py` | 深挖人类与 DeepSeek 分歧模式 | `output/penghui/disagreement_analysis.txt` | 分歧诊断 |
| `multidim_validation.py` | 用多信号给样本打质量分层 | `output/penghui/multidim_validation_report.txt`、`output/penghui/multidim_validation_results.json` | 标注质检 |
| `train_rag_round2.py` | v1：直接用第二轮标注训练基础检索模型 | 模型目录、`output/penghui/rag_round2_training/evaluation_results.json` | 基线微调方案 |
| `train_rag_round2_v3.py` | v3：用 Gold/Silver 样本训练 | 模型目录、`output/penghui/rag_round2_training/evaluation_v3.json` | 噪声过滤方案 |
| `train_rag_round2_v4.py` | v4：按分歧与语义排名筛正负样本 | 模型目录、`output/penghui/rag_round2_training/evaluation_v4.json` | 中等强度过滤方案 |
| `train_rag_weighted.py` | 置信分层加权训练 | 模型目录、`output/penghui/rag_round2_training/evaluation_weighted.json` | 质量加权方案 |
| `eval_models_multimetric.py` | 比较 baseline、v1、v3、v4 多项指标 | `output/penghui/model_comparison.txt` | 模型横向评估 |

## 3.1 当前冻结基线

当前路线已经明确把“基线”定义成一套可复现配方，而不是单个模型目录。正式说明见：

- 基线配方文档：[docs/penghui-retrieval-baseline.md](/d:/PythonProjects/Employ26/docs/penghui-retrieval-baseline.md)
- ADR: [docs/adr/0002-freeze-penghui-baseline-on-v1-task-table-recipe.md](/d:/PythonProjects/Employ26/docs/adr/0002-freeze-penghui-baseline-on-v1-task-table-recipe.md)

当前冻结结论：

- 基线训练配方以 `train_rag_round2.py` 的 `v1` 为准
- 基线数据契约暂时冻结在 `annotations.label_studio_tasks_v2`
- 当前输入字段口径仍是 `annotations_completed` + `data_raw` 的任务主表解析结果
- 职业词典冻结为 `public.occ_dict_unified`，字段以 `code`、`title`、`desc`、`tasks` 为训练和评估文本来源
- `annotations_completed_jsonb`、`data_raw_jsonb` 与 `annotations.v_label_studio_task_annotations_v2` 属于后续可挑战的“数据契约升级方案”，不与基线冻结动作同时进行
- 正式定胜负以 `eval_models_multimetric.py` 及 `output/penghui/model_comparison.txt` 为准
- 当前综合最强模型是 `output/penghui/rag_round2_training/bge-large-round2-finetuned`

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
- 依赖 `output/penghui/rag_round2_training/bge-large-round2-finetuned` 作为对比模型时，没有显式检查该模型是否存在

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
   `disagreement_deep_analysis.py`、`multidim_validation.py`

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

- 入口：`python -m src.penghui.eval_models_multimetric`
- 报告：`output/penghui/model_comparison.txt`

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

当前 `output/penghui/model_comparison.txt` 显示：

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

- 生产默认使用：`output/penghui/rag_round2_training/bge-large-round2-finetuned`
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
- 默认 fine-tuned 模型：`output/penghui/rag_round2_training/bge-large-round2-finetuned`
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
.\.conda\python.exe -m src.penghui.train_rag_round2 `
  --base-model-path D:\model\bge-m3 `
  --output-model-name v1-bge-m3 `
  --run-label v1-bge-m3
```

训练完成后，用统一评估把挑战者加入比较：

```powershell
$env:EMPLOYDATA_BGE_MODEL_PATH='D:\model\bge-large-zh-v1.5'
$env:PYTHONIOENCODING='utf-8'
.\.conda\python.exe -m src.penghui.eval_models_multimetric `
  --model v1-bge-m3=output/penghui/rag_round2_training/v1-bge-m3
```

说明：

- 第一轮挑战只替换底座模型，不同时修改样本构造、split、超参数和统一评估口径
- 若 `bge-m3` 要取代当前基线，应至少在 `candidate_acc` 与 `mrr` 上同时不弱于当前 `v1`

## 6. 当前共性问题

这批脚本的共性问题比较明显，优先级大致如下：

### P0：先修复，否则稳定性较差

- 多个脚本把输入文件名写死为单个导出文件，不支持切换批次数据
- `train_rag_weighted.py` 仍需检查并消除潜在 GPU 设备硬编码，避免无 GPU 环境不可运行
- 不同脚本都各自实现了一遍标注解析、majority 计算、词典加载，容易出现口径漂移

### P1：影响复现与维护

- 大量阈值、关键词、采样规模、oversample 倍数直接写在脚本中
- 模型路径、输出文件名、对比模型列表没有做统一配置
- 训练集/测试集构造规则在不同版本间差异较大，但缺少统一说明

### P2：影响性能与工程质量

- 多个脚本对每条任务逐条 `model.encode([anchor])`，运行效率偏低
- 缺少基础测试，至少应覆盖标注解析、majority、tier 判定等核心逻辑
- 报告写出方式不统一，有的只打印控制台，有的边跑边追加写文本

## 7. 推荐修复顺序

建议按下面顺序收敛：

1. 抽公共模块  
   新建如 `src/penghui/common.py`，统一实现：
   - 标注文件加载
   - `parse_choice()`
   - majority / pairwise agreement 统计
   - 职业词典加载
   - DeepSeek 结果加载

2. 给脚本加 CLI 参数  
   至少支持：
   - `--annotation-file`
   - `--deepseek-file`
   - `--dict-file`
   - `--output-dir` / `--output-file`
   - `--device`

3. 消除硬编码 GPU  
   统一改成自动选择 `cuda` 或 `cpu`。

4. 向量化语义排名计算  
   预先批量编码 anchor，避免每条任务单独推理。

5. 统一实验配置  
   将 tier 规则、语义排名阈值、测试集抽样规模、oversample 系数抽成配置对象。

6. 补最小测试  
   至少覆盖：
   - 候选选择解析
   - 多标注 majority 判定
   - tier 计算结果
   - 数据切分基本约束

## 8. 建议的后续归档方式

如果这批实验已经完成阶段性结论，后续建议做一次轻量整理：

- 保留 1 个总览分析入口
- 保留 1 个样本筛选入口
- 保留 1 个基础训练入口
- 保留 1 个多模型评估入口
- 其他历史版本转入 `archive/` 或标注为 `experimental/`

这样可以减少“v1 / v3 / v4 / weighted”继续并列扩散，避免后续维护成本持续上升。
