# Occupation Retrieval Offline Top10 QC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline Markdown QC report for occupation retrieval that checks whether Top10 candidates contain the gold occupation detail, summarizes occupation-detail distribution, and optionally adds local LLM review notes for miss cases.

**Architecture:** Add one focused offline QC module under `src/occupation_retrieval` that reads annotation tasks and `public.occupation_detail_matches`, computes deterministic hit/miss outcomes, aggregates distribution statistics, and renders a Markdown report. Keep the LLM layer optional and advisory only: it reviews miss samples via `src.model_platform.llm.create_llm_client()` but never changes the deterministic metrics.

**Tech Stack:** Python, PostgreSQL, pandas, `src.occupation_retrieval.common`, `src.occupation_retrieval.datasets`, `src.model_platform.llm`, pytest.

---

### Task 1: Lock down the QC helpers with tests

**Files:**
- Create: `src/tests/test_occupation_retrieval_offline_top10_qc.py`

- [ ] **Step 1: Write the failing test**

```python
from src.occupation_retrieval.offline_top10_qc import (
    compute_top10_hit,
    evaluate_distribution,
    parse_top10_candidates,
)


def test_compute_top10_hit_detects_exact_code_match():
    row = {"top10_candidates": [{"rank": 1, "code": "A"}, {"rank": 2, "code": "B"}]}
    assert compute_top10_hit("B", row) is True
    assert compute_top10_hit("C", row) is False


def test_parse_top10_candidates_handles_missing_and_json_text():
    assert parse_top10_candidates(None) == []
    assert parse_top10_candidates('[{"rank": 1, "code": "A"}]')[0]["code"] == "A"


def test_evaluate_distribution_returns_stable_summary():
    summary = evaluate_distribution({"A": 3, "B": 1})
    assert summary["total"] == 4
    assert summary["top_code"] == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: FAIL because the QC module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement the QC module with pure helper functions first, then the report builder and optional LLM review wrapper.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tests/test_occupation_retrieval_offline_top10_qc.py src/occupation_retrieval/offline_top10_qc.py
git commit -m "feat: add offline occupation retrieval top10 qc helpers"
```

### Task 2: Implement the offline Markdown QC report

**Files:**
- Create: `src/occupation_retrieval/offline_top10_qc.py`
- Modify: `src/occupation_retrieval/README.md`

- [ ] **Step 1: Write the failing test**

Add tests for report assembly, deterministic hit labeling, and optional LLM review shaping with a fake client.

- [ ] **Step 2: Run the focused test subset and confirm the missing module path**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: one or more failures until the report builder and review flow are implemented.

- [ ] **Step 3: Implement the offline QC script**

Include:
```python
python -m src.occupation_retrieval.offline_top10_qc --output-file output/occupation_retrieval/offline_top10_qc.md
```

The script should:
1. Load tasks from `annotations.label_studio_tasks_v2`
2. Load Top10 rows from `public.occupation_detail_matches`
3. Map each task’s majority choice to the gold occupation code
4. Mark rows as `HIT@10`, `MISS@10`, `GOLD_NONE`, or `MISSING_MATCH`
5. Aggregate occupation distribution counts and shares
6. Optionally invoke `create_llm_client()` for miss-review notes only
7. Render a single Markdown report

- [ ] **Step 4: Update the README**

Document the new offline QC entry point, the deterministic hit logic, the distribution summary, and the fact that LLM review is advisory only.

- [ ] **Step 5: Run verification commands**

Run:
```powershell
python -m compileall -q src\occupation_retrieval
pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v
python -m src.occupation_retrieval.offline_top10_qc --help
```

- [ ] **Step 6: Commit**

```bash
git add src/occupation_retrieval/offline_top10_qc.py src/occupation_retrieval/README.md src/tests/test_occupation_retrieval_offline_top10_qc.py
git commit -m "feat: add offline occupation retrieval top10 qc report"
```

### Task 3: Final review

**Files:**
- Modify: `src/occupation_retrieval/README.md`

- [ ] **Step 1: Review the report output contract**

Confirm the report includes:
1. Overall hit/miss counts
2. Per-row QC labels
3. Occupation distribution table
4. Uniformity assessment
5. Optional LLM miss-review notes

- [ ] **Step 2: Run the new CLI once on the local environment**

Run the module with the default output path and inspect the Markdown output file.

- [ ] **Step 3: Commit any final wording fixes**

Keep the public CLI stable and leave the LLM choice governed by `src.model_platform.llm.create_llm_client()`.
