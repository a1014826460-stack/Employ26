# Skill Extraction V2 设计文档

> 历史说明：本文档仅记录 V2 / DuckDB 时期的设计背景，不再作为当前实现入口。
> 当前主线请以 `src/skill_extraction/README.md` 和
> `python -m src.skill_extraction.cli --help` 为准。

## 1. 目标

`src/skill_extraction` 的 v2 流程面向两个核心目标：

1. 以较高精度、较高吞吐从招聘文本中抽取硬技能。
2. 建立一条可长期维护、可自动回归评测、可持续迭代的工程化链路。

当前版本不再把“词典构造”“技能匹配”“训练集构造”“上下文过滤”混在一个脚本里，而是拆成明确的阶段，并允许每一阶段单独运行、单独评估。

## 2. 配置约定

统一配置来源：

- [config/database.yaml](/d:/PythonProjects/Employ26/config/database.yaml:1)

当前关键配置项：

```yaml
LLM_model_path: D:/model/Qwen3-8B
BERT_path: D:\model\chinese-roberta-wwm-ext
```

含义：

- `LLM_model_path`
  用于平面词典构造、LLM 自动标注、匹配结果抽检等环节。
- `BERT_path`
  用于上下文判别器训练与推理。

代码入口统一通过 [config.py](/d:/PythonProjects/Employ26/src/skill_extraction/config.py:1) 读取，不再依赖硬编码模型路径。

## 3. 重构后的模块结构

### 3.1 词典构造

- [pipeline_v2.py](/d:/PythonProjects/Employ26/src/skill_extraction/pipeline_v2.py:1)
  v2 平面词典构造入口。
- [occupation_skill_pipeline.py](/d:/PythonProjects/Employ26/src/skill_extraction/occupation_skill_pipeline.py:1)
  平面词典构造主实现，负责采样、prompt 构造、vLLM 调用、技能去重与覆盖率补充。

### 3.2 词典匹配与结果校验

- [match_flat_skills_to_duckdb.py](/d:/PythonProjects/Employ26/src/skill_extraction/match_flat_skills_to_duckdb.py:1)
  使用平面词典对 DuckDB 岗位文本做硬技能匹配，并支持：
  - 高速召回
  - 上下文判别器二阶段过滤
  - LLM 抽检与词典自动修正

### 3.3 自动评测

- [regression_eval.py](/d:/PythonProjects/Employ26/src/skill_extraction/regression_eval.py:1)
  固定回归集评测脚本，输出 `precision / recall / F1 / exact match rate` 以及误报漏报明细。

### 3.4 全自动 LLM 标注

- [llm_label_regression_dataset.py](/d:/PythonProjects/Employ26/src/skill_extraction/llm_label_regression_dataset.py:1)
  从 DuckDB 样本自动生成回归集。
- [llm_label_context_dataset.py](/d:/PythonProjects/Employ26/src/skill_extraction/llm_label_context_dataset.py:1)
  基于回归集和词典召回结果，自动生成上下文判别训练集。
- [llm_labeling_utils.py](/d:/PythonProjects/Employ26/src/skill_extraction/llm_labeling_utils.py:1)
  复用 DuckDB 取数、样本分层采样、vLLM 批量推理、JSONL 落盘等公共能力。

### 3.5 上下文判别器

- [context_labels.py](/d:/PythonProjects/Employ26/src/skill_extraction/context_labels.py:1)
  统一定义多分类标签。
- [context_classifier.py](/d:/PythonProjects/Employ26/src/skill_extraction/context_classifier.py:1)
  多分类上下文判别器训练与推理实现。

## 4. 整体运行流程

### 4.1 第一步：构造平面词典

入口：

```bash
python -m src.skill_extraction.pipeline_v2
```

流程：

1. 从 `recruit.main.skill_extraction_requirement_matches` 读取岗位文本。
2. 优先使用 `任职要求_items_text`，其次使用 `岗位职责_items_text`，最后回退到 `岗位描述_清洗`。
3. 按职业中类聚合样本。
4. 用本地 Qwen3 批量抽取硬技能。
5. 统一去重并生成 `flat_skill_dictionary.json`。

### 4.2 第二步：自动生成回归集

入口：

```bash
python -m src.skill_extraction.labeling.regression_dataset ^
  --sample-size 400 ^
  --num-votes 3
```

流程：

1. 从 DuckDB 抽取岗位样本。
2. 先做文本去重，减少模板化 JD 的重复干扰。
3. 按职业中类做 round-robin 分层采样，避免样本偏斜。
4. 用本地 LLM 输出 `gold_skills` 和 `gold_skill_items`。
5. 要求每个技能必须带 `evidence`，且 `evidence` 必须是原文连续子串。
6. 如果开启 `--num-votes > 1`，会做多轮投票聚合，只保留支持票数达标的技能。

输出格式示例：

```json
{
  "sample_id": "sample_0000001",
  "text": "熟悉 Python、Spark、Hive",
  "gold_skills": ["Python", "Spark", "Hive"],
  "gold_skill_items": [
    {"name": "Python", "evidence": "Python", "skill_type": "编程语言", "vote_count": 3}
  ]
}
```

### 4.3 第三步：自动生成上下文判别训练集

入口：

```bash
python -m src.skill_extraction.labeling.context_dataset ^
  --regression-dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --num-votes 3
```

流程：

1. 使用平面词典召回器从每条岗位文本中找出候选技能。
2. 把候选技能连同岗位上下文一起发给本地 LLM。
3. 让 LLM 对每个候选打多分类标签。
4. 多轮投票后，只保留支持票数达标的候选标签。

当前标签体系：

- `valid_hard_skill`
- `too_generic`
- `wrong_alias_mapping`
- `not_skill`

输出格式示例：

```json
{
  "sample_id": "sample_0000001",
  "text": "熟悉 Python、Spark、Hive",
  "skill_name": "Python",
  "matched_term": "Python",
  "term_role": "name",
  "label": "valid_hard_skill",
  "label_reason": "原文明确要求 Python 开发能力",
  "label_vote_count": 3
}
```

### 4.4 第四步：训练上下文判别器

入口：

```bash
python -m src.skill_extraction.hard.context_classifier train ^
  --dataset output/skill_extraction/context_classifier/context_dataset_llm.jsonl ^
  --output-dir output/skill_extraction/context_classifier/model
```

流程：

1. 从 `BERT_path` 加载基础模型。
2. 构造句对输入：
   - `text_a`: `skill_name + matched_term + term_role + job_title`
   - `text_b`: 岗位原文局部上下文
3. 训练多分类判别器。
4. 导出模型目录供匹配阶段直接加载。

### 4.5 第五步：生产匹配

入口：

```bash
python -m src.skill_extraction.match_flat_skills_to_duckdb match ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --context-classifier-model output/skill_extraction/context_classifier/model ^
  --context-threshold 0.80
```

流程：

1. 第一阶段：词典召回
   - ASCII 技能走合并正则。
   - 中文/混合技能走 Trie 风格的归一化召回。
2. 第二阶段：上下文判别
   - 只保留 `label == valid_hard_skill`
   - 且 `valid_score >= threshold` 的候选
3. 落表到 DuckDB。

## 5. 关键原理

### 5.1 为什么要保留“高速召回层”

高速召回的意思不是“更聪明的模型”，而是“先快速找出所有可能是技能的候选项”。

当前实现把词典召回拆成两条通道：

1. `ASCII 合并正则`
   适合 `Python / Java / SQL / C++ / ERP / CAD`
2. `中文/混合词归一化扫描`
   适合中文技能名、证书名、设备名、工艺术语

好处：

- 吞吐高
- 词典越大越稳定
- 不需要对每个 alias 单独扫全文

### 5.2 为什么还需要上下文判别器

词典召回追求高召回，不追求最终精度，所以天然会产生三类误差：

1. 泛词误报
   例如“测试”“资格证书”“办公软件”
2. alias 错吸附
   例如文本里出现短 alias，但语义并不属于该技能
3. 非技能误命中
   例如职责动词、普通名词、软技能

上下文判别器的职责不是“重新抽技能”，而是对召回候选做二阶段裁剪。这样比“全量 vLLM 逐条判别”更适合长期生产：

- 更快
- 更稳定
- 更容易做回归控制
- 更容易累积训练数据

### 5.3 为什么训练集也要让 LLM 生成

当前你明确要求不引入人工标注，所以这版设计采取“全自动 LLM 标注”策略：

1. 回归集由 LLM 从原始 JD 自动抽取 `gold_skills`
2. 上下文判别训练集由 LLM 自动给候选打多分类标签
3. 多轮投票聚合后再用于训练

这样做的重点不是“完全相信单次 LLM 输出”，而是通过：

- 证据约束
- 样本去重
- 分层采样
- 多轮投票

来压低自动标注噪声。

## 6. 当前实现落地内容

已经完成：

1. `LLM_model_path` / `BERT_path` 接入配置层。
2. 平面词典匹配器的高速召回层。
3. 固定回归集评测脚本。
4. 多分类上下文判别器实现。
5. 完全自动的 LLM 回归集生成脚本。
6. 完全自动的 LLM 上下文训练集生成脚本。
7. 匹配阶段接入多分类判别器。

同时还修复了一个已有问题：

- `match_flat_skills_to_duckdb.py` 中 `main()` 对 `validate` 子命令传错参数的问题已经理顺。

## 7. 推荐运行顺序

### 7.1 初次建链

```bash
python -m src.skill_extraction.pipeline_v2
python -m src.skill_extraction.labeling.regression_dataset --sample-size 400 --num-votes 3
python -m src.skill_extraction.labeling.context_dataset --regression-dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl --dictionary dicts/flat_skill_dictionary.json --num-votes 3
python -m src.skill_extraction.hard.context_classifier train --dataset output/skill_extraction/context_classifier/context_dataset_llm.jsonl --output-dir output/skill_extraction/context_classifier/model
python -m src.skill_extraction.evaluation.regression --dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl --dictionary dicts/flat_skill_dictionary.json --fail-under-precision 0.90 --fail-under-f1 0.80
python -m src.skill_extraction.match_flat_skills_to_duckdb match --dictionary dicts/flat_skill_dictionary.json --context-classifier-model output/skill_extraction/context_classifier/model
```

### 7.2 周期性增量更新

```bash
python -m src.skill_extraction.labeling.regression_dataset --sample-size 200 --num-votes 3
python -m src.skill_extraction.labeling.context_dataset --regression-dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl --dictionary dicts/flat_skill_dictionary.json --num-votes 3
python -m src.skill_extraction.hard.context_classifier train --dataset output/skill_extraction/context_classifier/context_dataset_llm.jsonl --output-dir output/skill_extraction/context_classifier/model_next
python -m src.skill_extraction.evaluation.regression --dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl --dictionary dicts/flat_skill_dictionary.json --fail-under-precision 0.90 --fail-under-f1 0.80
```

## 8. 已知局限

虽然这版已经进入“可长期维护”的结构，但还有三个现实边界：

1. 回归集本身仍然是 LLM 自动标注，不等于真实人工金标。
2. 词典构造阶段仍以 prompt 抽取为主，对极长尾技能依然受模型知识边界影响。
3. 目前判别器还是纯分类头，没有接入 span extraction 或 mention grounding。

## 9. 下一阶段建议

如果继续追求更高精度，优先级建议如下：

1. 在回归集中记录 `gold_skill_items.evidence_span`，让训练与评测更可解释。
2. 为 `match_flat_skills_to_duckdb.py` 增加“错误标签落盘”，便于持续分析 `too_generic / wrong_alias_mapping / not_skill` 的来源。
3. 给词典增加版本化元数据与自动晋升门禁。
4. 在长期阶段引入 mention extraction 模型，再与词典规范化结合。

## 10. 结论

当前 v2 已经从“单纯的 LLM 词典脚本”升级成一条完整的自动化链路：

1. LLM 自动构造训练数据
2. BERT 多分类判别器压误报
3. 高速词典召回保证吞吐
4. 回归评测负责发布门禁

这套结构比“纯词典”更准，也比“全量 vLLM 逐条判别”更适合长期维护。
