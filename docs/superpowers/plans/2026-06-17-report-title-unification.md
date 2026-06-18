# Report Title Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify analysis report titles to English while keeping report body text mainly in Simplified Chinese.

**Architecture:** Update the report-writing sections in the active analysis scripts so Markdown and HTML headers use English titles and section labels, without changing aggregation logic, CSV schemas, or database outputs. Keep the narrative content and inline explanations in Simplified Chinese where they already describe methodology or results.

**Tech Stack:** Python, pandas, Markdown, HTML, pytest.

---

### Task 1: Update structured analysis report headers to English

**Files:**
- Modify: `src/analysis/occupation_salary_analysis.py`
- Modify: `src/analysis/education_distribution_analysis.py`
- Modify: `src/analysis/industry_trend_analysis.py`
- Modify: `src/analysis/structured_dimension_analysis.py`

- [ ] **Step 1: Replace the top-level report titles and section headings with English strings**

```python
# occupation_salary_analysis.py
# Use English report title and section headings; keep the narrative body in Chinese.
# Example replacements:
#   "# 广东省招聘数据 - 职业类别薪资分析报告" -> "# Guangdong Recruitment Data - Occupation Salary Analysis Report"
#   "## 一、职业类别薪资统计" -> "## 1. Occupation Category Salary Statistics"
#   "## 二、职业核心词薪资排行 (Top 50)" -> "## 2. Occupation Core Salary Ranking (Top 50)"
#   "## 三、学历×职业类别薪资分析" -> "## 3. Education x Occupation Category Salary Analysis"
#   "## 四、学历×职业薪资分析（主口径）" -> "## 4. Education x Occupation Salary Analysis (Primary View)"

# education_distribution_analysis.py
#   "# 广东省招聘数据 - 学历需求分布分析报告" -> "# Guangdong Recruitment Data - Education Requirement Distribution Report"
#   "## 一、职业类别年度学历分布" -> "## 1. Annual Education Distribution by Occupation Category"
#   "## 二、职业年度学历分布（Top 20职业示例）" -> "## 2. Annual Education Distribution by Occupation (Top 20 Examples)"
#   "## 三、学历需求趋势分析（按职业类别）" -> "## 3. Education Requirement Trends by Occupation Category"
#   "## 四、数据说明" -> "## 4. Notes"

# industry_trend_analysis.py
#   "# 广东省招聘数据 - 行业景气度分析报告" -> "# Guangdong Recruitment Data - Industry Trend Analysis Report"
#   "## 一、行业整体招聘量排行 (Top 30)" -> "## 1. Overall Industry Hiring Volume Ranking (Top 30)"
#   "## 二、各城市主要行业分布" -> "## 2. Major Industry Distribution by City"

# structured_dimension_analysis.py
#   "# 结构化维度补充分析报告" -> "# Structured Dimension Supplementary Analysis Report"
#   "## 一、经验要求 × 职业" -> "## 1. Experience Requirements x Occupation"
#   "## 二、公司规模 × 城市 × 行业" -> "## 2. Company Size x City x Industry"
#   "## 三、城市 × 职业需求" -> "## 3. City x Occupation Demand"
```

- [ ] **Step 2: Run a focused syntax and smoke check for the four scripts**

Run: `python -m compileall -q src/analysis/occupation_salary_analysis.py src/analysis/education_distribution_analysis.py src/analysis/industry_trend_analysis.py src/analysis/structured_dimension_analysis.py`
Expected: exits successfully with no syntax errors.

- [ ] **Step 3: Verify the generated report files now begin with English headings**

Run the relevant analysis entry points or inspect the existing report templates to confirm the first heading line is English and the body remains Chinese.
Expected: Markdown reports start with English titles; Chinese explanatory text remains in the body.

### Task 2: Update requirement text report headers to English

**Files:**
- Modify: `src/analysis/requirement_text_analysis.py`

- [ ] **Step 1: Replace the report title and section headings with English strings**

```python
# In _build_report_text():
#   "# Requirement Text 约束抽取与分析报告" -> "# Requirement Text Constraint Extraction and Analysis Report"
#   "## 一、运行摘要" -> "## 1. Run Summary"
#   "## 结论注记" -> "## Key Notes"
#   "## 二、样本覆盖率与抽取诊断" -> "## 2. Sample Coverage and Extraction Diagnostics"
#   "## 三、约束维度频率" -> "## 3. Constraint Dimension Frequency"
#   "## 四、约束值分布" -> "## 4. Constraint Value Distribution"
#   "## 五、城市 / 行业 / 公司规模差异" -> "## 5. City / Industry / Company Size Differences"
#   "## 六、模板噪声报告" -> "## 6. Template Noise Report"
#   "## 七、招聘门槛强度" -> "## 7. Hiring Stringency Index"
#   "## 八、说明" -> "## 8. Notes"
```

- [ ] **Step 2: Keep the explanatory bullets in Simplified Chinese**

```python
# Example body lines stay Chinese:
#   "- 本期正式结论聚焦 requirement 约束、模板噪声与招聘门槛强度，不包含 hard skill / soft skill 分类研究。"
#   "- hard skill / soft skill 继续列为 TODO，后续单独做更细的词典治理与标注验证。"
```

- [ ] **Step 3: Run a focused syntax and smoke check**

Run: `python -m compileall -q src/analysis/requirement_text_analysis.py`
Expected: exits successfully with no syntax errors.

### Task 3: Align Excel summary labels with the new English report titles

**Files:**
- Modify: `src/analysis/generate_excel_summary.py`

- [ ] **Step 1: Update the summary sheet labels to English report names**

```python
report_files = [
    ('职业类别薪资分析报告.md', 'Occupation Salary Analysis'),
    ('学历需求分布分析报告.md', 'Education Requirement Distribution'),
    ('行业景气度分析报告.md', 'Industry Trend Analysis'),
    ('结构化维度补充分析报告.md', 'Structured Dimension Supplementary Analysis'),
]
```

- [ ] **Step 2: Run a syntax check for the summary script**

Run: `python -m compileall -q src/analysis/generate_excel_summary.py`
Expected: exits successfully.

- [ ] **Step 3: Commit the documentation and code changes separately if desired**

Run:
`git add src/analysis/*.py docs/superpowers/plans/2026-06-17-report-title-unification.md`
`git commit -m "feat: unify analysis report titles"`
Expected: clean commit with the title-unification changes only.
