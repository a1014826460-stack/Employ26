# Employ26

Employ26 是一个围绕广东省招聘数据进行岗位匹配、技能抽取与职业检索分析的项目。它把多平台招聘文本、标注结果和派生特征收敛成可复用的数据资产。

## Language

**Experimental Workspace**:
用于复现实验、比较方案和沉淀阶段性结论的代码区域。它服务于探索和验证，不直接等同于正式长期维护的生产入口。
_Avoid_: Production pipeline, formal module, official entrypoint

**Retrieval Effectiveness**:
在检索实验语境中，优先优化候选选择准确率、排序质量、层级命中率和 MRR 等离线检索质量指标，而不是优先优化训练耗时、推理时延或模型体积。
_Avoid_: Training efficiency as the primary goal, deployment cost as the default objective

**Baseline-First Model Selection**:
在底座模型存在多个候选时，先固化当前表现最好且可复现的基线训练与评估流程，再让新底座作为挑战者在同口径下比较，而不是在基线尚未冻结时同时漂移多个变量。
_Avoid_: Changing base model and training recipe at the same time, comparing models under different evaluation setups

**Unified Retrieval Evaluation**:
检索基线与挑战者模型的优劣，必须以统一数据集、统一指标定义和统一评估脚本得出的结果判定；训练脚本各自附带的局部评估只可用于单次实验内诊断，不作为跨方案定胜负的正式标准。
_Avoid_: Comparing models by per-script test sets, mixing local experiment metrics with canonical benchmark results

**Baseline Recipe**:
检索基线不是单个已训练模型目录，而是一套可重复执行并可重新产出同类结果的固定实验配方。它至少同时冻结训练数据构造规则、train/test split 规则、训练超参数和统一评估流程。
_Avoid_: Treating one saved model artifact as the whole baseline, changing data rules while claiming the baseline is unchanged

**Frozen Baseline Data Contract**:
当实验基线被声明为可复现配方时，其输入数据源也必须冻结到具体 PostgreSQL 表名与字段口径，而不是只说“来自标注数据”或“来自职业词典”。后续挑战者模型必须在同一数据契约上比较。
_Avoid_: Vague data provenance, switching between tables/views/field semantics without renaming the recipe

**Single-Variable Challenger Run**:
基线的第一轮挑战应只改变一个主变量，并把其他配方元素全部冻结不动。对于底座模型挑战，这意味着只替换 embedding base model，本轮不同时调整样本构造、split 规则、训练超参数或正式评估口径。
_Avoid_: Multi-variable “challenge” runs, crediting a new base model for gains caused by recipe changes

**Challenger Win Criteria**:
底座模型挑战只有在统一评估中同时守住主检索指标并未明显牺牲层级准确率时，才可视为取代基线。对 occupation_retrieval 检索实验而言，第一轮主胜负指标是 `candidate_acc` 与 `mrr`，而 `subclass_acc` 与 `midclass_acc` 作为约束指标不应明显退步。
_Avoid_: Declaring victory from a single flattering metric, trading away core retrieval quality for local gains

**Parameterized Baseline Runner**:
当检索基线被定义为可复现配方时，应优先通过参数化同一训练入口来切换底座模型和产物命名，而不是复制出新的专用脚本。这样可以把挑战者运行约束在同一配方框架中，减少实现漂移。
_Avoid_: Forking a new training script per base model, encoding recipe changes as script proliferation

**Recipe-Backbone Artifact Naming**:
检索实验产物应同时显式表达“配方名”和“底座名”，使同一配方下的不同底座挑战者可以直接对照。目录名和结果文件名不应再只暴露历史版本号或只暴露某一侧信息。
_Avoid_: Ambiguous artifact names, model directories that hide whether the recipe or the backbone changed

**Forward-Only Artifact Renaming**:
新的产物命名规范从下一次新跑开始生效，不主动重命名已经验证过的历史产物目录。历史目录可继续作为已确认实例存在，直到新的参数化入口稳定后再决定是否做兼容迁移。
_Avoid_: Renaming validated historical artifacts while introducing a new recipe contract, mixing compatibility migration with experiment definition

**Parameterized Canonical Evaluation**:
正式统一评估入口也应接受显式模型列表或模型映射参数，而不是把待评估模型写死在源码中。这样挑战者模型可以在不修改评估代码的前提下进入同口径比较。
_Avoid_: Editing benchmark source code for each challenger run, hidden model lists baked into the evaluator

**Minimal Challenger CLI**:
在基线固化和第一轮挑战阶段，训练入口与统一评估入口只暴露支撑单变量挑战所需的最小参数集合，而不是提前扩张成通用实验平台。当前最小集合是训练入口支持 `--base-model-path`、`--output-model-name`、`--run-label`，统一评估入口支持重复传入 `--model NAME=PATH`。
_Avoid_: Premature general-purpose experiment CLIs, adding knobs unrelated to the first challenger comparison

**Public Feature Layer**:
供多个下游流程复用的正式结构化数据层。它的职责是沉淀稳定、可重复使用的派生特征，而不是充当某个单独脚本的私有中间结果。
_Avoid_: Private temp table, one-off cache, script-local intermediate

**Job Description Parsing Result**:
对单条招聘岗位描述做结构化切分后形成的可复用文本特征。它属于 Public Feature Layer，面向后续匹配、抽取和检索流程共享使用。
_Avoid_: Cleaned-data copy, disposable parsing output

**Recruitment Record**:
来自某个招聘平台的一条完整职位发布记录。它是项目中的基础数据单元，其中包含岗位名称、岗位描述等原始招聘信息；同岗跨平台默认视为不同 Recruitment Record。
_Avoid_: Cleaned-data row, parsing result

**Job Description Text**:
Recruitment Record 中承载职责、要求、福利等招聘叙述的自由文本。描述解析关注的是这段文本本身，而不是整张招聘表的复制或清洗。
_Avoid_: Whole cleaned table, copied job table

**Source Locator**:
用于唯一定位某条来源招聘记录的标准标识符，格式为 `"{source_table}:{source_row_number}"`。新流程中的来源回溯、跨步骤传递和结果引用都应以它为准。
_Avoid_: sample_row_id, source_record_id, ad-hoc source key

**Recruitment Record ID**:
用于跨时间稳定标识一条招聘记录的正式业务主键。它优先承接平台原生岗位 ID；若源平台没有提供，则在统一规范层首次生成并冻结。
_Avoid_: Source Locator, row_number, temp hash

**Recruitment Normalized Layer**:
面向跨平台复用的招聘统一规范层。它负责把不同平台的招聘记录收敛为一致的公共字段，并承载稳定的 Recruitment Record ID。
_Avoid_: Raw mirror, feature layer, experiment output

**Recruitment Normalized Table**:
Recruitment Normalized Layer 的实体表实现。在当前阶段，它以 `public.recruitment_jobs_normalized` 的形式承载统一字段、来源定位信息和冻结后的 Recruitment Record ID。
_Avoid_: View-only projection, disposable export

**Frozen Recruitment Record ID**:
在 Recruitment Normalized Table 中首次确定后不再重算的 Recruitment Record ID。后续同步可以更新记录内容，但不应改变它所代表的业务身份。
_Avoid_: Recomputed ID, per-run temporary key

**Dedupe Fingerprint**:
在缺少平台原生岗位 ID 时，用于识别“这次来的记录是否与历史记录属于同一招聘记录”的内部认同字段。它服务于 Frozen Recruitment Record ID 的稳定分配，但不应作为下游公共引用契约。
_Avoid_: Public identifier, Source Locator, temporary checksum

**Normalized Recruitment Input**:
供公共特征层优先消费的统一招聘输入。长期来看，岗位描述解析等下游复用流程应以 Recruitment Normalized Table 为主上游，而不是直接面向各平台源表。
_Avoid_: Direct raw-table input as the long-term default

**Current Authoritative Parsing Result**:
面向下游消费的当前唯一权威岗位描述解析结果。它按 `recruitment_record_id` 唯一存储，重跑时覆盖为最新结果，而不是并存保留多个 parser 版本。
_Avoid_: Versioned parsing history as the default public contract

**Recruitment Record Reference**:
新公共表之间引用招聘记录时使用的标准关联字段名。该字段统一命名为 `recruitment_record_id`，用于表达正式业务身份，而不是流程内定位符；现有活跃公共表在重构时也应收敛到该字段。
_Avoid_: sample_row_id, source_record_id, generic job_id

**Requirement Match Result**:
面向职业细类匹配与技能词典前置准备复用的岗位要求匹配结果。它把岗位要求文本与职业候选对齐，并以 `recruitment_record_id` 作为标准引用字段；它不是最终技能识别结果。
_Avoid_: Skill match result, final skill inventory, sample_row_id as the public reference contract

**Requirement Text Statistical Analysis**:
对 `requirements_text` 语料做描述性统计的分析活动，关注高频词项、搭配、分层分布和时间变化；第一阶段产出是可复核的文本信号，不默认把这些词项直接解释为技能。
_Avoid_: Skill analysis by default, generic NLP analysis, topic modeling by default

**N-gram Statistical Unit**:
第一阶段文本统计的最小分析单位是词项和短语 n-gram，例如 unigram、bigram、trigram，而不是句子主题、文档聚类或隐含语义簇。
_Avoid_: Topic as the default unit, document cluster as the first unit, sentence-level semantics by default

**Requirement Structure Category**:
对招聘要求文本中的词项或短语做结构化归类的分析层，用来区分硬技能、软素质、学历、经验、证书和其他约束。它服务于解释招聘要求构成，不等同于最终技能识别结果。
_Avoid_: Skill classification as the whole requirement analysis, raw frequency list as the only structure

**Authoritative Structured Requirement Field**:
当某类招聘要求已经在源数据中以稳定结构化字段存在时，应优先直接使用该字段做正式统计，而不是从文本中重复提取同一信息。文本侧的同类信号只用于补充、校验或解释，不作为第一口径。
_Avoid_: Re-extracting structured fields as the default path, treating text-derived values as the primary source when stable columns already exist

**Structured Requirement Dimension in Recruitment Normalized Layer**:
像 `薪资水平`、`学历要求`、`经验要求` 这类跨流程高频复用且源表已提供的结构化维度，应直接进入 Recruitment Normalized Layer，避免下游分析长期回源关联或在文本侧重复提取。
_Avoid_: Leaving reusable dimensions only in raw tables, rebuilding the same dimension per downstream script

**Raw-First Structured Dimension Adoption**:
统一规范层新增结构化维度时，先稳定承接各平台已有的原始字段值，再增量补标准化列。不要在第一次纳入时同时混做字段落库、跨平台标准化和业务解释。
_Avoid_: Forcing full normalization in the first migration, mixing storage adoption with semantic standardization

**Raw-Suffixed Public Column Naming**:
统一规范层中承接原始结构化值的公共列应显式使用 `*_raw` 后缀，例如 `salary_raw`、`education_requirement_raw`、`experience_requirement_raw`、`company_size_raw`、`company_industry_raw`。这样可以和后续标准化列长期并存而不混淆语义。
_Avoid_: Bare names like `salary` for raw text values, ambiguous public column names

**First-Stage Requirement Analysis Source**:
第一阶段 `Requirement Text Statistical Analysis` 以 `public.recruitment_jobs_normalized` 和 `public.job_description_parsed` 作为正式主输入，而不是继续依赖 `output/integrated` 或历史 CSV 链路。
_Avoid_: Legacy CSV pipeline as the primary source, mixed source-of-truth between PG and exports

**Current Parsing Snapshot Only**:
第一阶段要求文本统计对每个 `recruitment_record_id` 只消费一条当前有效的解析结果，不把旧 parser 版本历史并入同一次统计。
_Avoid_: Mixing historical parser versions into one analysis snapshot, counting one record multiple times because of re-parsing

**Record-Weighted Primary Statistics**:
第一阶段要求文本统计的主口径按职位发布记录计数，同一 `requirements_text` 出现多次就按多条 Recruitment Record 累积进入结果。同时应提供去重文本口径的对照结果，用于识别模板化 JD 对统计的放大效应。
_Avoid_: Text-deduplicated counts as the only primary market signal, ignoring template amplification diagnostics

**Record Presence Frequency**:
第一阶段单个词项或短语的频次按“命中该词项的招聘记录数”统计，而不是按同一条 `requirements_text` 中的重复出现次数累加。它回答的是“多少条招聘记录提出了这个要求”。
_Avoid_: Term frequency within one document as the default headline metric, repetition-weighted counts

**Minimum Phrase Support Threshold**:
第一阶段 bigram / trigram 主结果只保留达到最低记录支持度阈值的短语；低于阈值的短语仅作为诊断输出保留，不进入正式高频表。
_Avoid_: Publishing one-off phrase combinations as headline findings, threshold-free phrase tables

**Item-Bounded Phrase Extraction**:
第一阶段 bigram / trigram 优先在切分后的单条 requirement item 内部生成，而不是跨整段 `requirements_text` 任意拼接。这样可以避免把相邻但无关的要求条目误拼成假短语。
_Avoid_: Cross-item n-gram concatenation, whole-paragraph phrase generation by default

**Publish-Month Trend Basis**:
第一阶段时间趋势统计统一按可解析的 `publish_date` 归到 `YYYY-MM` 月度粒度，并且只把发布时间可解析的记录纳入时间序列主口径。
_Avoid_: Mixing unparsable dates into the headline trend series, ambiguous time granularity

**Lightweight Dimension Normalization**:
对城市、行业、公司规模这类分层维度，第一阶段先基于统一规范层中的原始字段做轻量标准化映射，再进入正式分组统计。目标是压缩碎片化类别、提升跨平台可比性，而不是一次性完成重语义标准化。
_Avoid_: Grouping headline tables directly on raw fragmented strings, heavyweight taxonomy migration in the first pass

**Top-N Headline Grouping**:
城市、行业、公司规模等分层结果在主报告中只展示 `Top N` 类别，尾部低频类别合并为“其他”；完整明细则单独导出，避免主表被长尾类别淹没。
_Avoid_: Expanding all low-frequency groups in the headline report, hiding full detail by not exporting raw breakdowns

**First-Stage Requirement Analysis Deliverables**:
第一阶段正式产物至少包括四类：总体高频 unigram / bigram / trigram 表、按职业/城市/行业/公司规模/月度分层的高频词项表、`requirements_text` 覆盖率与 `duties_text fallback` 诊断表，以及去重文本口径对发布记录口径的对照表。
_Avoid_: Single flat frequency table as the whole deliverable, mixing diagnostics into the main headline tables without separation

**Descriptive-Only First Stage**:
第一阶段只做描述性统计、对照表和导出产物，不在同一轮里混入自动分类器、LLM 标注、主题模型或新的正式回写表。
_Avoid_: Mixing modeling and persistence into the first descriptive pass, redefining the analysis target while implementing new ML layers

**Lightweight Tokenization First**:
第一阶段要求文本统计优先采用通用中文分词器，并通过项目内词典、停用词表和短语合并规则补强，而不是先构建复杂的术语抽取器。目标是先获得稳定、可复核、可迭代的统计底盘。
_Avoid_: Building a bespoke extractor before the first descriptive baseline, treating complex term extraction as the default starting point

**PostgreSQL-Backed Lexicon Storage**:
第一阶段要求文本统计所依赖的停用词表、短语合并规则和人工词典，正式存放在 PostgreSQL 的专用 schema 中，并作为唯一正式 source of truth。运行时直接从 PostgreSQL 读取，`dicts/` 不再承担正式口径，只保留为迁移材料、导出副本或备份。
_Avoid_: Treating repo files as the long-term primary lexicon store, hidden dual maintenance without an explicit source of truth

**Stage-Scoped Lexicon Schema**:
用于承载词汇资源的新 PostgreSQL schema 在第一步只服务于第一阶段 `Requirement Text Statistical Analysis`，不预先扩张成全项目通用词汇资源层。它的边界应由当前统计分析所需资源定义，而不是被未来潜在场景提前撑大。
_Avoid_: Designing a project-wide lexicon platform in the first pass, loading unrelated downstream resources into the initial schema

**Resource-Specific Lexicon Tables**:
第一阶段词汇资源 schema 优先按资源职责拆成多张表，例如 `stopwords`、`phrase_rules`、`user_dictionary`，而不是先做一张靠 `type` 字段区分的大一统词汇表。不同资源类型应保留各自清晰的字段结构、约束和维护边界。
_Avoid_: One giant generic lexicon table, overloading one schema object with incompatible resource shapes

**Governed Lexicon Resource Rows**:
第一阶段词汇资源表从一开始就应自带最基础的治理字段，例如 `is_active`、`version`、`source`、`notes`、`updated_at`。词条不是一次性静态数据，而是会持续迭代的分析规则资产。
_Avoid_: Bare lexicon rows without lifecycle metadata, unmanaged rule tables with no provenance or activation state

**Dedicated Analysis Lexicon Schema**:
第一阶段要求文本统计所需的词汇资源表应放在独立的新 PostgreSQL schema `analysis_lexicon` 中，而不是继续堆进 `public`。这个 schema 用来表达“分析阶段专用资源”的边界，和公共业务表分开治理。
_Avoid_: Mixing stage-scoped lexicon resources into `public`, hiding analysis-only tables among shared business tables

**Normalized Lexicon Term Pair**:
`analysis_lexicon.user_dictionary` 中的词条应同时保存展示用的 `term` 和匹配归并用的 `normalized_term`。原始词面与归一化词面不是一回事，后续展示、匹配和聚合都不应强绑在同一个字段上。
_Avoid_: Single-field lexicon entries that mix display text and normalized matching keys

**Typed Phrase Rules**:
`analysis_lexicon.phrase_rules` 应从一开始就区分不同规则语义，至少包括“短语合并规则”和“短语排除规则”。它们不应被压成一张只有通用文本字段的无类型规则表。
_Avoid_: Untyped phrase rule rows, mixing merge and exclusion semantics in one generic rule field

**Scoped Stopwords**:
`analysis_lexicon.stopwords` 应从一开始就区分作用域，至少分开“全局停用词”和“仅 requirement analysis 使用的停用词”。停用词不是天然全项目通用资源，不同分析目标可能需要不同停用策略。
_Avoid_: One undifferentiated global stopword list, silently applying requirement-specific stopwords to unrelated pipelines

**Normalized Stopword Pair**:
`analysis_lexicon.stopwords` 也应显式保存 `term` 与 `normalized_term`，使停用词在大小写、全半角和符号变体上有稳定归并键。
_Avoid_: Raw-only stopword entries, unstable matching on superficial text variants

**Scoped Stopword Uniqueness**:
`analysis_lexicon.stopwords` 应对 `scope + normalized_term` 建立唯一约束，同一作用域下不应重复定义相同停用词。
_Avoid_: Duplicate stopword rows within one scope, ambiguous stopword precedence

**Lexicon Term Type**:
`analysis_lexicon.user_dictionary` 应保存 `term_type`，至少包含 `hard_skill_hint`、`soft_skill_hint`、`certificate_hint`、`tool_hint` 和 `noise_term`，用于解释和后续结构化分类。
_Avoid_: Type-less vocabulary rows, forcing all lexicon entries into one undifferentiated bucket

**Enabled User Dictionary Terms**:
`analysis_lexicon.user_dictionary` 应显式保存 `enabled` 字段，词条不是默认永久生效的静态资源，而是可启停的规则资产。
_Avoid_: Assuming all dictionary terms are always active, removing term-level activation control

**Preferred Lexicon Term**:
`analysis_lexicon.user_dictionary` 应显式保存唯一的 `preferred_term`，用于聚合、展示和导出时的首选词面，而不是靠排序规则隐式决定。
_Avoid_: Implicit primary term selection, display ordering as a hidden source of truth

**Normalized Term Canonicalization**:
`analysis_lexicon.user_dictionary` 允许多个 `term` 变体映射到同一个 `normalized_term`，但应明确保留一个首选展示词面，用于聚合和输出的一致性。
_Avoid_: One-to-one-only lexicon entries, letting display variants fragment one canonical key

**Typed Normalized Term Uniqueness**:
`analysis_lexicon.user_dictionary` 应对 `normalized_term + term_type` 建立唯一约束，同一归一化词在同一细类型下不应重复定义。
_Avoid_: Duplicate typed lexicon rows for the same normalized term, ambiguous aggregation keys

**Multi-Type Normalized Terms**:
同一个 `normalized_term` 可以同时存在多个 `term_type`，例如一个词既可作为 `tool_hint`，也可作为 `hard_skill_hint`。项目不强制一个归一化词只能对应单一细类型。
_Avoid_: Forcing every normalized term into a single term_type when the domain semantics overlap

**Deterministic Phrase Merge Rules**:
`analysis_lexicon.phrase_rules` 的短语合并规则应先限定为确定性的显式映射，而不是正则替换或模糊模式。第一阶段先追求可复核和可控，避免引入过宽的合并逻辑。
_Avoid_: Fuzzy merge patterns, regex-heavy phrase normalization in the first pass

**Ngram-Scoped Stopwords**:
`analysis_lexicon.stopwords` 应支持按 n-gram 层级或作用域生效，允许某些词只停用 unigram，而不阻止它参与 bigram / trigram。
_Avoid_: Flat stopword application across all n-gram levels, overblocking useful multiword phrases

**Noise Terms as Exclusion Only**:
`noise_term` 只用于识别和排除，不进入任何正式高频主表或主结果排行。
_Avoid_: Treating noise terms as part of the headline findings

**Requirement Analysis Output Folder**:
第一阶段正式结果统一落到 `output/reports/req_analysis_{mm-dd}/`，其中 `mm-dd` 按运行日期填充。这样可以保持每次分析结果按批次隔离并便于版本化。
_Avoid_: Dumping all outputs into a shared reports root, making run-specific artifacts hard to separate

**English CSV Columns, Chinese Report Headings**:
第一阶段导出的 CSV 保留英文列名，而面向人阅读的主报告标题使用中文。这样可以兼顾脚本稳定性与阅读友好性。
_Avoid_: Chinese column names in machine-facing CSVs, English-only report titles for human output

**CSV-Plus-Markdown Deliverables**:
第一阶段正式产物至少固定输出 CSV 和 Markdown 两类文件；HTML 可作为可选增强，但不作为每次都必须生成的硬约束。
_Avoid_: TXT-only human reports, requiring HTML for every run, omitting simple machine-readable and human-readable exports

**Term-Type Counts in Main Output**:
第一阶段主结果应对 `hard_skill_hint`、`soft_skill_hint`、`certificate_hint`、`tool_hint` 等 `term_type` 分别计数并输出，以便同时观察“热度”和“结构”。
_Avoid_: Only publishing aggregate counts without type-level structure

**Lexicon Release Registry**:
`analysis_lexicon` 应有独立的 `lexicon_release` / `lexicon_version` 记录层，用来指向某次正式生效的词汇资源快照。
_Avoid_: Unversioned lexicon assets, forcing downstream runs to guess which rule snapshot was used

**Minimal Lexicon Release Fields**:
`analysis_lexicon.lexicon_release` 至少应包含 `version`、`is_current`、`released_at`、`released_by`、`notes` 这些基础字段，以支撑第一阶段的正式版本管理。
_Avoid_: Under-specified release records, release metadata spread across ad-hoc fields

**Current-Release-Only Runtime Read**:
第一阶段运行时只读取 `analysis_lexicon.lexicon_release.is_current = true` 对应版本下的启用词条，不做多版本混读。
_Avoid_: Mixing multiple lexicon releases at runtime, hidden cross-version rule blending

**Release-Bound Lexicon Resources**:
`analysis_lexicon.user_dictionary`、`analysis_lexicon.stopwords` 和 `analysis_lexicon.phrase_rules` 都应显式挂接到 `lexicon_release`，而不是只靠全局 current 状态间接判断所属版本。
_Avoid_: Resource rows with implicit release membership, mixed-version assets under one runtime read

**Run-Level Lexicon Versioning**:
第一阶段每次分析运行都应在输出中记录所用的 `analysis_lexicon` 版本号。
_Avoid_: Anonymous runs without lexicon provenance, making output comparisons non-reproducible

**Required Lexicon Term Type**:
`analysis_lexicon.user_dictionary.term_type` 应为必填字段，不应允许空值。
_Avoid_: Nullable term types, unstable output grouping and filtering semantics

**Rule Source Provenance**:
`analysis_lexicon.phrase_rules` 应记录规则来源，例如 `manual`、`derived`、`imported`，以便判断规则可信度与回溯来源。
_Avoid_: Source-less rules, losing provenance for merge/exclusion decisions

**Coverage and Residual Output**:
第一阶段主报告应同时输出 `requirements_text` 覆盖率、`duties_text fallback` 回退率，以及 `analysis_lexicon` 过滤后保留下来的样本量。
_Avoid_: Reporting only final counts without showing coverage and filtering impact

**Graceful Empty Resource Handling**:
第一阶段运行时，如果某个当前 release 下的资源类别为空，流程仍可继续执行，但必须在输出中明确记录该资源类别为空。
_Avoid_: Hard-failing the whole analysis because one resource table is empty, silently ignoring missing resource classes

**Lexicon Category Layer**:
`analysis_lexicon.user_dictionary` 除 `term_type` 外还应有更粗粒度的 `category`，例如 `skill`、`soft_trait`、`certificate`、`tool`、`noise`，用于主表展示和分组汇总。
_Avoid_: Flat-only term typing, forcing every report to group on overly granular labels

**Phrase Rule Activation Fields**:
`analysis_lexicon.phrase_rules` 应显式保存 `enabled` 和 `priority` 字段，用于处理规则冲突和生效顺序。
_Avoid_: Implicit rule ordering, untracked active/inactive rule states

**Explicit Phrase Replacement Fields**:
`analysis_lexicon.phrase_rules` 应显式包含 `source_term`、`normalized_source_term`、`replacement_term` 等确定性替换字段，而不是把规则压成单段文本。
_Avoid_: Opaque phrase rule text blobs, implicit source/replacement parsing

**Typed Phrase Rule Uniqueness**:
`analysis_lexicon.phrase_rules` 应对 `normalized_source_term + rule_type` 建立唯一约束，同一来源短语在同一规则类型下不应出现多条互相竞争的规则。
_Avoid_: Competing rules of the same type for one normalized source term, duplicate deterministic mappings

**Stopword Strength Levels**:
`analysis_lexicon.stopwords` 应区分 `hard_stop` 和 `soft_stop`，让绝对排除与降权/局部停用分开管理。
_Avoid_: One-size-fits-all stopword treatment, overblocking or underblocking terms

**Retained Disabled Stopwords**:
`analysis_lexicon.stopwords` 中 `enabled = false` 的词条应继续保留，而不是物理删除。停用词的启停状态本身就是资源治理的一部分。
_Avoid_: Hard-deleting disabled stopwords, losing rule history context unnecessarily

**Top Terms and Phrases by Type**:
第一阶段主报告应按 `term_type` 额外输出 `Top terms` 和 `Top phrases` 两类表，确保既能看结构，也能看具体词项与短语形态。
_Avoid_: Only aggregate counts by term type, hiding the actual lexicon surface forms

**No Length Gate on Main Sample**:
第一阶段主分析样本继续保留所有 `requirements_text` 非空记录，不再额外设置长度门槛。
_Avoid_: Implicitly filtering short but valid requirement texts out of the main sample

**No Auto Snapshot Export**:
第一阶段暂不强制每次正式发布词汇资源后自动导出快照文件；若需要快照，可按需手动或在后续单独补自动化。
_Avoid_: Mandatory snapshot export in the first pass, coupling every release to extra filesystem side effects

**No Lexicon Audit Log Requirement**:
第一阶段词汇资源 schema 不强制保留逐条 `who/when/what changed` 审计日志；只要版本、来源和启用状态可追溯即可。
_Avoid_: Mandatory change-event audit tables for every lexicon edit, overengineering the first schema

**Lexicon Variants Field**:
`analysis_lexicon.user_dictionary` 可显式保存 `example_terms` 或 `variants`，并优先采用 JSON 数组字段承载常见词形变体，而不是只依赖 `term` 和 `normalized_term` 的隐式对应关系。
_Avoid_: Hiding common variants only in implicit mappings, making the lexicon harder to inspect

**Deterministic Phrase Replacement**:
`analysis_lexicon.phrase_rules` 的规则应采用“单条规则对应一个明确替换结果”的确定性映射，不引入复杂 pattern-based 模糊改写。
_Avoid_: Ambiguous pattern rewriting, fuzzy phrase normalization as the default rule style

**Scoped Ngram Stopwords**:
`analysis_lexicon.stopwords` 的 `scope` 应细分到 `unigram`、`bigram`、`trigram`，而不是只靠一段泛化文本描述适用范围。
_Avoid_: Free-form scope descriptions, unclear n-gram applicability

**Overall and Category Headline Output**:
第一阶段主报告中的 `Top terms` / `Top phrases` 应分别输出 `overall` 与各 `category` 视图，先看全局，再看分层。
_Avoid_: Single-level headline outputs without overall/category separation

**Exportable Lexicon Snapshot**:
`analysis_lexicon` 应预留导出当前正式快照的能力，便于把某次 release 固化为离线文件。
_Avoid_: Database-only lexicon with no snapshot/export path for reproducible runs

**Default Phrase Support Threshold**:
第一阶段 bigram / trigram 进入主结果表的默认最低记录支持度阈值为 `20`。
_Avoid_: Leaving phrase support thresholds implicit, changing the headline threshold ad hoc between runs

**Default Top-N Window**:
第一阶段主报告中类别、词项和短语的默认 `Top N` 展示窗口为 `20`。
_Avoid_: Run-to-run drift in headline list sizes, undocumented top-k choices

**Minimum Group Size for Headline Breakdown**:
第一阶段分层统计表中，一个分组至少应有 `50` 条招聘记录才进入主报告。
_Avoid_: Publishing tiny long-tail groups as headline comparisons, unstable subgroup tables

**Minimum Monthly Group Size**:
第一阶段月度趋势表中，一个月-一个分组至少应有 `20` 条招聘记录才进入正式趋势结果。
_Avoid_: Drawing trend lines from tiny monthly cells, unstable month-by-group comparisons

**Requirement Analysis Run Manifest**:
每次第一阶段运行都应在 `req_analysis_{mm-dd}` 目录下生成 `run_manifest.json`，记录运行时间、`analysis_lexicon` 版本号、样本量和关键参数阈值。
_Avoid_: Anonymous run folders, making later result comparison dependent on memory or log scraping

**Textual Lexicon Versioning**:
`analysis_lexicon.lexicon_release.version` 采用手工控制的文本版本号，例如 `v0.1.0`，而不是自动递增整数。
_Avoid_: Opaque numeric-only release identifiers, release numbering with no semantic intent

**Bigserial Resource IDs**:
`analysis_lexicon` 下三张资源表与 release 表的主键统一采用 `bigserial id`，不使用自然键充当主键。
_Avoid_: Natural-key primary keys for mutable lexicon resources, inconsistent identifier strategy across tables

**Text Plus Check Domain Fields**:
第一阶段 `term_type`、`category`、`scope`、`rule_type`、`stop_strength` 等字段优先使用 `text + check 约束`，而不是 PostgreSQL enum type。
_Avoid_: Premature enum-type lock-in, making early domain iteration harder than necessary

**Coverage Diagnostic Columns**:
第一阶段覆盖率/回退率诊断表至少包含 `total_records`、`requirements_nonempty_records`、`duties_fallback_records`、`publish_date_parseable_records`、`retained_after_lexicon_records`。
_Avoid_: Free-form diagnostics without a fixed minimal metrics set

**Variants JSON Column Naming**:
`analysis_lexicon.user_dictionary` 中承载词形变体的字段名统一为 `variants_json`。
_Avoid_: Ambiguous or overloaded variant field names, ORM-unfriendly generic array column names

**Typed Headline Tables by Overall and Category**:
第一阶段主报告中的 `Top terms` / `Top phrases` 既要有 `overall` 视图，也要按 `category` 输出分层视图。
_Avoid_: Single-level headline tables, hiding structured breakdowns behind one global list

**Noise Filter Export**:
第一阶段应单独导出 `noise_terms_filtered.csv`，用于查看被排除掉的高频噪声词项。
_Avoid_: Hiding filtered noise terms completely, making lexicon noise governance hard to inspect

**Unigram-Only Fallback for Weak Itemization**:
当某条 `requirements_text` 没有可靠 item 切分时，该记录仍保留在 unigram 统计中，但不参与 bigram / trigram 统计。
_Avoid_: Forcing unreliable phrase extraction from weakly segmented requirement text

**Standard Overall Frequency Columns**:
第一阶段总体高频表的正式 CSV 至少包含 `ngram_level`、`term_text`、`normalized_term`、`record_count`、`record_share`、`term_type`、`category`、`lexicon_hit`、`source_scope`。
_Avoid_: Unstable output schemas for overall frequency tables, ad-hoc column selection by run

**Standard Breakdown Frequency Columns**:
第一阶段分层高频表的正式 CSV 至少包含 `dimension_name`、`dimension_value`、`ngram_level`、`term_text`、`normalized_term`、`record_count`、`record_share`、`term_type`、`category`。
_Avoid_: Inconsistent subgroup table schemas, mixing dimensions into free-form output columns

**Fixed Markdown Report Order**:
第一阶段 Markdown 主报告固定按以下顺序组织章节：`一、运行摘要`、`二、样本覆盖率与回退诊断`、`三、总体高频 unigram`、`四、总体高频 bigram/trigram`、`五、按 term_type/category 的结构分析`、`六、按职业/城市/行业/公司规模分层`、`七、月度趋势`、`八、去重文本 vs 发布记录口径对照`。
_Avoid_: Ad-hoc report section ordering, moving core diagnostics around between runs

**Hard Skill**:
在本项目中，Hard Skill 仅指可操作、可训练、可验证的专业能力项，当前限定为八类：编程语言、框架、数据库、工具软件、办公软件、设备/仪器、工艺方法、证书/资质。
_Avoid_: Generic requirement item, soft trait, education requirement, experience requirement

**Soft Skill**:
在本项目中，Soft Skill 指 JD 中表达的软素质要求信号，例如沟通、责任心、协作、主动性和抗压性。大五人格仅作为归类框架使用，不把这些文本信号直接解释为人格测量结果。
_Avoid_: Personality diagnosis, generic requirement item, hard skill

**Requirement-First Skill Demand Text**:
技能需求统计默认使用 `requirements_text` 作为主分析文本；仅当其为空时才回退到 `duties_text`，并排除福利、联系方式、公司介绍等其他信息段落。
_Avoid_: Raw job description as the default corpus, duties-first corpus, mixed misc text

**Requirement Match Preparation Script**:
用于生成 Requirement Match Result 的实现脚本名，例如 `src.data_pipeline.requirement_match_prep`。它是技术实现入口，不是领域中的正式数据概念。
_Avoid_: Treating a script name as the canonical data model term

**Opaque Recruitment Record ID**:
无业务语义、仅用于稳定标识 Recruitment Record 的正式 ID。它不编码平台、标题、日期等业务信息，可读信息应由其他字段承载。
_Avoid_: Readable composite key, semantic identifier

**Recruitment Record ID Assignment**:
`recruitment_record_id` 只在记录首次进入 Recruitment Normalized Table 且未命中历史记录时生成一次。后续同步可以更新记录内容，但不应重新分配该 ID。
_Avoid_: Reassignment on content update, regenerated ID on re-run

**Source Trace Fields**:
用于回溯招聘记录原始来源位置的辅助字段，例如 `source_table` 和 `source_row_number`。它们服务于溯源和排查，但不参与 Recruitment Record 的业务身份定义，也不替代 `recruitment_record_id`。
_Avoid_: Business primary key, standard cross-table reference

**Deprecated Reference Fields**:
旧链路中用于引用招聘记录的历史字段名，例如 `sample_row_id`、`source_record_id`、`row_id` 和旧式 `job_id`。在新公共链路中它们应视为废弃字段，不再作为标准引用键。
_Avoid_: Standard identifier, canonical reference

**Historical Annotation RRID Backfill**:
为历史 `annotations.label_studio_tasks_v2` 任务补齐 `recruitment_record_id` 的一次性治理动作。它只回填正式业务身份字段，不改写任务文本、候选或标注语义，并要求同时保留可复演的审计证据。
_Avoid_: Silent remap, semantic overwrite

**Historical Annotation Snapshot Row ID**:
历史 `label_studio_tasks_v2.row_id` 在第二轮导出链路中表示 Label Studio 导出快照里的行号，而不是招聘源表主键。它只能用于回放历史任务对应的快照行，不能直接当作招聘记录身份。
_Avoid_: public.jd_raw.row_id, canonical source primary key


