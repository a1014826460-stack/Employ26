# Skill Extraction V3 设计文档

**Goal**

在现有 V2 硬技能词典匹配管线基础上，完成硬技能 8 分类标签扩展和软技能抽取管线新建，达到覆盖率 >90%、准确率 >85% 的质量目标。截止时间：2026-06-12 24:00。

本设计**不改动** V2 的核心匹配引擎（`FlatHardSkillMatcher`）、词典格式（schema_version 3）、或回归评估框架。

## 范围

### 覆盖

- 为 `dicts/flat_skill_dictionary.json` 增加 `category` 字段，值域为 8 个硬技能类别
- 在 `FlatHardSkillMatcher` 输出层增加分类标签
- 新建软技能抽取模块，输出大五人格 5 类软技能
- 统一评估脚本：覆盖率、准确率、分类准确率
- 结果写入 PostgreSQL 新表

### 不覆盖

- 不重做 V2 的词典构建流程（`FlatSkillPipeline`）
- 不改动 `context_classifier.py` 的训练逻辑
- 不改动 `regression_eval.py` 的评估框架（仅扩展指标）
- 不涉及 BGE 微调或 RAG 检索管线

## 已确认事实

1. V2 硬技能词典 `dicts/flat_skill_dictionary.json` 包含 1907 个技能、1135 个别名，schema_version 3
2. 词典中 `skill_type` 字段已有值，且大部分与 V3 的 8 类高度重合（如"编程语言"37、"框架"39、"工具软件"290、"设备/仪器"360、"工艺/方法"320、"证书/资质"260、"办公软件"11、"数据库"9），但存在异名（如"framework"、"tool"、"device"）和需要细分的兜底类（"专业知识"417）
3. 词典中 `category` 字段不存在（0/1907 有值），需新增英文标识版本用于统一值域
4. V2 匹配引擎 `FlatHardSkillMatcher` 支持 ASCII 正则 + 中文 Trie 两阶段召回
5. V2 上下文分类器 `SkillContextClassifier` 使用 BERT 4 类分类（valid_hard_skill / too_generic / wrong_alias_mapping / not_skill）
6. `annotations.label_studio_annotations_v2` 包含 `soft_skill` 字段，用于标注软技能
7. 软技能分类依据为大五人格理论：Neuroticism、Extraversion、Openness to Experience、Agreeableness、Conscientiousness
8. 硬技能 8 类：编程语言、框架、数据库、工具软件、办公软件、设备/仪器、工艺方法、证书/资质
9. 源数据表：`public.job_description_parsed`（唯一键 `recruitment_record_id`）
10. V2 结果目前写入 DuckDB `recruit.main.hard_skill_match_results_dev`，V3 应写入 PostgreSQL

## Alternatives Considered

### Option A: 纯 LLM 抽取（不用词典）

直接用 LLM 从岗位描述中抽取硬技能和软技能，不依赖预构建词典。

优点：

- 无需维护词典，开箱即用
- 理论上覆盖面最广

缺点：

- 每条记录都要调用 LLM，处理 10 万条记录成本高、耗时长
- LLM 输出不可复现，同一条记录多次抽取结果不同（项目历史已验证此问题）
- 无法保证 8 分类的准确性和一致性

结论：不采用。项目历史中 v1/v1.1 已验证纯 LLM 方案不可靠。

### Option B: 词典匹配 + LLM 验证（本设计方案）

硬技能基于现有 V2 词典扩展 8 分类标签，软技能新建词典 + LLM 二次验证。

优点：

- 复用 V2 已验证的 1907 个硬技能词典，无需从零构建
- 词典匹配保证可复现性和高精度
- LLM 仅用于验证和补全，调用量可控
- 8 分类基于 `skill_type` 现有值映射，工作量小

缺点：

- 软技能词典需要新建，种子词质量依赖标注数据
- 边界词（如"管理"）的分类仍需上下文判断

结论：采用。

### Option C: 纯 ML 分类器（V3 原方案）

用 Random Forest 分类器（125 维特征）自动判断候选词是否为技能并分类。

优点：

- 能发现词典中没有的新技能（高召回）
- 推理速度快，无需调用 LLM

缺点：

- 需要大量高质量标注数据训练，当前标注数据不足（`soft_skill` 字段有效数据待确认）
- 原 V3 分类器代码已从活跃目录移除，说明团队已放弃此路径
- 精确率（85-88%）低于词典匹配方案（90%+）

结论：不采用。训练数据不足，且原实现已被弃用。

## 硬技能 8 分类方案

### 类别定义

| 英文标识 | 中文名称 | 典型示例 |
|----------|----------|----------|
| `programming_language` | 编程语言 | Python, Java, C++, SQL |
| `framework` | 框架 | Spring Boot, Vue, React, Django |
| `database` | 数据库 | MySQL, PostgreSQL, Redis, MongoDB |
| `tool` | 工具软件 | Git, Docker, Jenkins, JIRA |
| `office` | 办公软件 | Microsoft Excel, PowerPoint, WPS |
| `equipment` | 设备/仪器 | 示波器, 频谱分析仪, 万用表 |
| `process` | 工艺方法 | SMT贴片, CNC加工, 六西格玛 |
| `certification` | 证书/资质 | PMP, CPA, CFA, 律师证 |

### 分类方法

采用 `skill_type` 映射 + LLM 补全的两阶段方法：

**阶段 1：`skill_type` 直接映射**

`skill_type` 现有值中，约 1700 个（89%）可直接映射到 V3 的 8 类：

| `skill_type` 现有值 | 映射目标 | 数量 |
|---------------------|----------|------|
| `编程语言` | `programming_language` | 37 |
| `框架`、`framework` | `framework` | 39 |
| `数据库` | `database` | 9 |
| `工具软件`、`tool`、`财务软件`、`设计软件` 等 | `tool` | ~290 |
| `办公软件` | `office` | 11 |
| `设备/仪器`、`设备操作`、`检测设备`、`device` 等 | `equipment` | ~360 |
| `工艺/方法`、`工艺方法`、`制造工艺`、`process` 等 | `process` | ~320 |
| `证书/资质`、`certification`、`证书`、`资格证书` 等 | `certification` | ~260 |

映射规则存储在 `dicts/skill_category_rules.json`，不硬编码进 Python。

**阶段 2：LLM 辅助分类**

对无法直接映射的技能（约 200 个，主要是"专业知识"417 个中的长尾项和其他杂类），使用 LLM 批量分类：

- 通过 `src.model_platform.llm.create_llm_client()` 创建 LLM 客户端，不直接加载模型权重
- 每批 50 个技能名称，附带类别定义和示例
- 输出 JSON：`{"skill_name": "category"}`
- 对 LLM 结果做人工抽检（抽样 5%），确保准确率 >95%

### 词典扩展

在 `dicts/flat_skill_dictionary.json` 的每个 skill 条目中增加 `category` 字段：

```json
{
  "name": "Python",
  "aliases": ["python", "Python3"],
  "skill_type": "编程语言",
  "category": "programming_language",
  "notes": ""
}
```

`category` 字段为必填，值域为上述 8 个英文标识之一。

## 软技能抽取方案

### 设计思路

软技能抽取采用**词典召回 + LLM 验证**的两阶段管线，与硬技能的"词典匹配 + 上下文分类"架构对齐。

### 阶段 1：大五人格软技能词典构建

基于大五人格理论，构建 5 类软技能种子词典：

| 大五维度 | 英文标识 | 种子词示例 |
|----------|----------|------------|
| 开放性 | `openness` | 创新、好奇心、审美、想象力、灵活、学习能力 |
| 尽责性 | `conscientiousness` | 细心、责任心、自律、计划性、严谨、高效 |
| 外向性 | `extraversion` | 沟通能力、团队协作、领导力、表达能力、活跃 |
| 宜人性 | `agreeableness` | 合作、同理心、友善、包容、服务意识 |
| 情绪稳定性 | `neuroticism` | 抗压能力、情绪管理、冷静、心理承受力 |

种子词来源：
- 大五人格理论标准描述词
- `annotations.label_studio_annotations_v2.soft_skill` 中的高频标注词
- LLM 扩展（给定维度定义，让 LLM 生成变体词）

词典存储位置：`dicts/soft_skill_dictionary.json`

```json
{
  "schema_version": 1,
  "dimensions": {
    "openness": {"name": "开放性", "skills": [...]},
    "conscientiousness": {"name": "尽责性", "skills": [...]},
    "extraversion": {"name": "外向性", "skills": [...]},
    "agreeableness": {"name": "宜人性", "skills": [...]},
    "neuroticism": {"name": "情绪稳定性", "skills": [...]}
  }
}
```

### 阶段 2：软技能匹配与验证

1. **词典召回**：对岗位描述文本做关键词匹配，支持同义词映射
2. **LLM 验证**：通过 `src.model_platform.llm.create_llm_client()` 创建客户端，对召回结果做二次验证，确认：
   - 该词在当前上下文中确实是软技能（而非硬技能或职责描述）
   - 分类是否正确
3. **置信度打分**：基于匹配方式（精确匹配 / 同义词 / LLM 确认）给每条结果打分

### 软技能与硬技能的边界处理

- 软技能词典与硬技能词典互斥：如果一个词同时出现在两个词典中，优先归类为硬技能
- 边界词（如"管理"、"开发"）通过上下文判断：出现在"任职要求"中偏向软技能，出现在"技能要求"中偏向硬技能
- V2 的 `context_classifier` 的 `not_skill` 类别可用于过滤误召回的软技能候选
- 复用 V2 的黑名单系统（10 类 723+ 词，位于 `dicts/blacklist_*.txt`）过滤非技能词
- 软技能候选必须通过 `blacklist_soft_skills.txt` 的反向检查（该文件中的词不应被误判为硬技能）

## 数据流

```
输入：
  public.job_description_parsed (岗位描述)
    ↓
硬技能管线（V2 扩展）：
  FlatHardSkillMatcher.match_text() → 候选硬技能
    → context_classifier 过滤 → 8 分类标签 → 硬技能结果
    ↓
软技能管线（新建）：
  软技能词典召回 → LLM 验证 → 大五分类 → 软技能结果
    ↓
合并输出：
  public.skill_extraction_v3_results
```

## 输出表设计

在 `public` schema 下新建统一结果表：

```sql
CREATE TABLE public.skill_extraction_v3_results (
    id SERIAL PRIMARY KEY,
    recruitment_record_id TEXT NOT NULL,
    source_table TEXT,
    source_row_number INTEGER,
    job_title TEXT,
    
    -- 硬技能结果
    hard_skills JSONB,           -- [{"name": "Python", "category": "programming_language", "confidence": 0.95, "source": "dict_match"}]
    hard_skill_count INTEGER,
    
    -- 软技能结果
    soft_skills JSONB,           -- [{"name": "沟通能力", "dimension": "extraversion", "confidence": 0.85, "source": "dict_match+llm_confirm"}]
    soft_skill_count INTEGER,
    
    -- 元数据
    pipeline_version TEXT DEFAULT 'v3',
    extracted_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(recruitment_record_id)
);

CREATE INDEX idx_v3_results_rid ON public.skill_extraction_v3_results(recruitment_record_id);
CREATE INDEX idx_v3_results_hard_skills ON public.skill_extraction_v3_results USING GIN(hard_skills);
CREATE INDEX idx_v3_results_soft_skills ON public.skill_extraction_v3_results USING GIN(soft_skills);
```

## 评估标准

### 硬技能评估

| 指标 | 定义 | 目标 |
|------|------|------|
| 覆盖率 | 算法提取技能数 / 实际存在技能数 | >90% |
| 准确率 | 正确分类为 Hard/Soft/None 的比例 | >85% |
| 分类准确率 | 硬技能 8 分类正确的比例 | >80% |
| 精确率 | 提取结果中真正是技能的比例 | >85% |
| F1 | 精确率和召回率的调和平均 | >0.80 |

### 软技能评估

| 指标 | 定义 | 目标 |
|------|------|------|
| 覆盖率 | 算法提取软技能数 / 实际存在软技能数 | >80% |
| 分类准确率 | 大五维度分类正确的比例 | >75% |
| 精确率 | 提取结果中真正是软技能的比例 | >80% |

### 评估方法

- 硬技能：复用现有 `regression_eval.py` 框架，扩展分类准确率指标
- 软技能：从 `annotations.label_studio_annotations_v2` 中抽取有 `soft_skill` 标注的样本作为测试集
- 统一评估入口：`python -m src.skill_extraction.eval_v3`
- 评估门禁：`--fail-under-precision 0.85`、`--fail-under-f1 0.80`（与 V2 回归评估阈值对齐）
- 评估报告输出到 `output/skill_extraction/reports/v3_eval/`

## 错误处理

| 场景 | 处理策略 |
|------|----------|
| 硬技能分类规则与 LLM 结果冲突 | 以规则结果为准（规则优先级高于 LLM） |
| 软技能词典召回为空 | 跳过该记录，不输出空结果 |
| LLM 调用失败 | 降级为仅词典召回结果，标记 `confidence=0.5` |
| 同一技能同时命中硬技能和软技能 | 归类为硬技能，从软技能结果中移除 |
| `recruitment_record_id` 为空 | 跳过该记录，记录警告日志 |
| 分类结果不在 8 类/5 类值域内 | 归入最近似类别，记录警告 |

## Testing Strategy

第一批测试至少覆盖：

**硬技能分类测试：**

- `skill_type` 映射规则覆盖所有 8 个类别，无遗漏
- 映射后的 `category` 值均在 8 类枚举范围内
- LLM 补全结果通过人工抽检（准确率 >95%）
- `FlatHardSkillMatcher` 输出包含 `category` 字段

**软技能抽取测试：**

- 软技能词典每个维度至少有 10 个种子词
- 词典召回结果不与硬技能词典冲突
- LLM 验证模块能正确区分软技能与非软技能
- 置信度打分在 [0, 1] 范围内

**集成测试：**

- 统一评估脚本 `eval_v3` 可独立运行，输出完整报告
- 结果写入 `public.skill_extraction_v3_results`，`recruitment_record_id` 唯一
- 硬技能和软技能结果不交叉（同一技能不同时出现在两个字段中）

**回归测试：**

- 现有 V2 测试全部通过（`pytest src/tests/ -v`）
- V2 匹配结果不受影响（`FlatHardSkillMatcher` 核心逻辑不变）

验证方式：

- 新增 `src/tests/test_skill_extraction_v3.py`
- 运行 `python -m compileall -q src`
- 运行 `pytest src/tests/ -v`

## 实现分阶段

### Phase 1：硬技能 8 分类（优先级最高）

1. 构建 `skill_type` → `category` 映射规则（`dicts/skill_category_rules.json`）
2. 运行映射，覆盖约 89% 的技能（~1700 个）
3. 对无法映射的约 200 个技能，用 LLM 批量分类
4. 人工抽检 LLM 结果，修正错误
5. 更新 `dicts/flat_skill_dictionary.json`，为每个技能增加 `category` 字段
6. 修改 `FlatHardSkillMatcher` 输出层，附带分类标签

### Phase 2：软技能词典构建

1. 从 `annotations.label_studio_annotations_v2.soft_skill` 提取高频标注词
2. 基于大五人格理论扩展种子词（LLM 辅助）
3. 构建 `dicts/soft_skill_dictionary.json`
4. 验证词典覆盖率

### Phase 3：软技能匹配管线

1. 实现软技能词典召回模块
2. 实现 LLM 验证模块
3. 实现置信度打分
4. 集成到统一管线

### Phase 4：统一评估与入库

1. 实现 `eval_v3` 统一评估脚本
2. 创建 PostgreSQL 输出表
3. 实现结果写入
4. 运行全量评估，确认达标

## 成功标准

- [ ] 1907 个硬技能全部有 `category` 字段，值在 8 类范围内
- [ ] `skill_type` → `category` 映射覆盖率 >89%，LLM 补全准确率 >95%
- [ ] 硬技能覆盖率 >90%、准确率 >85%、分类准确率 >80%
- [ ] 软技能词典覆盖大五人格 5 个维度，每维度至少 10 个种子词
- [ ] 软技能覆盖率 >80%、分类准确率 >75%
- [ ] 统一评估脚本可独立运行，输出完整报告
- [ ] 结果写入 `public.skill_extraction_v3_results`，无数据丢失
- [ ] `python -m compileall -q src` 编译通过
- [ ] 现有 V2 测试不受影响（`pytest src/tests/ -v`）

## 合规说明

本设计遵守 `CLAUDE.md` 中的全部架构约束。以下仅列出本次任务中**需要特别注意**的几条：

- 结果写入 PostgreSQL（`public.skill_extraction_v3_results`），不引入 DuckDB
- LLM 调用通过 `src.model_platform.llm.create_llm_client()`，不直接加载模型权重
- 分类规则库存储在 `dicts/skill_category_rules.json`，不硬编码进 Python
- 统一评估入口：`python -m src.skill_extraction.eval_v3`

## 开放边界

以下问题与 V3 相关但**不在本次设计范围内**：

1. V2 词典的迭代改进（已有 `skill_dictionary_workflow.py` 独立管理）
2. 软技能的标注数据扩充（需要新的人工标注轮次）
3. 硬技能与职业类别的关联分析（属于下游分析任务）
4. 模型微调（BGE、BERT 分类器的再训练）
5. 历史结果的回溯更新（V3 结果只覆盖新抽取的记录）
