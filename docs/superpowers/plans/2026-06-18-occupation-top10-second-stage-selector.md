# Occupation Top10 Second-Stage Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared second-stage Top10 selector so both the offline QC report and the online occupation agent can let an LLM choose the final occupation detail from BGE Top10 candidates, then evaluate `Top1 accuracy`, `LLM-over-Top10 accuracy`, and whether LLM improves over raw Top1. Add DeepSeek arbitration for offline disagreement cases where the LLM winner differs from human gold.

**Architecture:** Introduce a focused selector module under `src/occupation_retrieval` that accepts job text plus Top10 candidates and returns a structured selected winner. Refactor `src/llm/occupation_agent_service.py` to reuse that selector. Extend `offline_top10_qc.py` so it compares raw Top1, LLM-selected winner, and human gold, and only invokes DeepSeek arbitration for disagreement rows in offline evaluation.

**Tech Stack:** Python, PostgreSQL, project `create_llm_client()` runtime, existing vLLM client utilities, pytest.

---

### Task 1: Lock down the shared selector contract with tests

**Files:**
- Create: `src/tests/test_occupation_top10_selector.py`

- [ ] **Step 1: Write the failing test**

```python
from src.occupation_retrieval.top10_selector import (
    Top10SelectionResult,
    normalize_selector_payload,
)


def test_normalize_selector_payload_extracts_ranked_winner():
    payload = {
        "selected_rank": 2,
        "selected_code": "2-02-10-09",
        "selected_title": "人工智能工程技术人员",
        "reason": "更贴近岗位职责",
        "confidence": 0.82,
        "needs_review": False,
    }
    result = normalize_selector_payload(payload, top10_candidates=[
        {"rank": 1, "code": "2-02-10-03", "title": "计算机软件工程技术人员"},
        {"rank": 2, "code": "2-02-10-09", "title": "人工智能工程技术人员"},
    ])
    assert isinstance(result, Top10SelectionResult)
    assert result.selected_rank == 2
    assert result.selected_code == "2-02-10-09"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_occupation_top10_selector.py -v`
Expected: FAIL because the module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `src/occupation_retrieval/top10_selector.py` with:
- `Top10SelectionResult`
- `normalize_selector_payload(...)`
- selector prompt builders
- an LLM-backed `select_from_top10(...)`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/tests/test_occupation_top10_selector.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/occupation_retrieval/top10_selector.py src/tests/test_occupation_top10_selector.py
git commit -m "feat: add shared occupation top10 selector"
```

### Task 2: Refactor the online occupation agent to reuse the selector

**Files:**
- Modify: `src/llm/occupation_agent_service.py`
- Modify: `src/tests/test_occupation_agent_service.py`

- [ ] **Step 1: Add a failing test for structured selector reuse**

Write a test that monkeypatches the shared selector and asserts the agent response exposes:
- raw `top1`
- `top10_candidates`
- structured `llm_selected`
- `final_selection`

- [ ] **Step 2: Run the focused agent tests**

Run: `pytest src/tests/test_occupation_agent_service.py -v`
Expected: FAIL until the service adopts the selector.

- [ ] **Step 3: Implement the refactor**

Update `OccupationAgentService.analyze()` so:
- BGE still retrieves Top10
- raw `top1` remains available
- the shared selector is called when `include_llm_report` or final selection is requested
- the service returns both raw retrieval and second-stage decision data

- [ ] **Step 4: Re-run the agent tests**

Run: `pytest src/tests/test_occupation_agent_service.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/occupation_agent_service.py src/tests/test_occupation_agent_service.py
git commit -m "feat: reuse shared top10 selector in occupation agent"
```

### Task 3: Extend offline QC into a true Top1-vs-LLM-vs-gold evaluator

**Files:**
- Modify: `src/occupation_retrieval/offline_top10_qc.py`
- Modify: `src/tests/test_occupation_retrieval_offline_top10_qc.py`

- [ ] **Step 1: Add failing tests for the new metrics**

Add tests that cover:
- raw `top1_code`
- `llm_selected_code`
- `gold_code`
- `llm_beats_top1`
- `llm_changes_top1_and_worsens`

- [ ] **Step 2: Run the focused QC tests**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: FAIL until the report computes the new metrics.

- [ ] **Step 3: Implement the new offline evaluation flow**

Update the QC script to:
- keep deterministic raw Top1 stats
- call the shared selector over Top10
- compute:
  - `Top1 accuracy`
  - `LLM-over-Top10 accuracy`
  - `LLM better than Top1`
  - `LLM same as Top1`
  - `LLM changed Top1 but became worse`

- [ ] **Step 4: Keep report output readable**

Render Markdown sections for:
- raw Top1 vs gold
- LLM winner vs gold
- delta analysis
- sample-level comparisons

- [ ] **Step 5: Re-run the QC tests**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/occupation_retrieval/offline_top10_qc.py src/tests/test_occupation_retrieval_offline_top10_qc.py
git commit -m "feat: evaluate llm-over-top10 against raw top1"
```

### Task 4: Add DeepSeek arbitration for offline disagreement rows

**Files:**
- Modify: `src/occupation_retrieval/offline_top10_qc.py`
- Modify: `src/tests/test_occupation_retrieval_offline_top10_qc.py`

- [ ] **Step 1: Add a failing test for arbitration-only-on-disagreement**

Write a test that ensures DeepSeek arbitration is only called when:
- `llm_selected_code` exists
- `gold_code` exists
- `llm_selected_code != gold_code`

- [ ] **Step 2: Run the focused QC tests**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: FAIL until disagreement arbitration exists.

- [ ] **Step 3: Implement offline-only DeepSeek arbitration**

Add a helper that returns structured arbitration fields:
- `arbitration_status`
- `arbitration_support`
- `arbitration_reason`
- `arbitration_review_needed`

It should reuse the project LLM runtime with a DeepSeek-specific backend override or prompt route only inside offline evaluation.

- [ ] **Step 4: Re-run tests**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/occupation_retrieval/offline_top10_qc.py src/tests/test_occupation_retrieval_offline_top10_qc.py
git commit -m "feat: add deepseek arbitration for llm-human disagreements"
```

### Task 5: Documentation and verification

**Files:**
- Modify: `src/occupation_retrieval/README.md`

- [ ] **Step 1: Update docs**

Document:
- shared Top10 second-stage selector
- raw Top1 vs LLM-over-Top10 metrics
- offline-only DeepSeek arbitration
- how this maps to future Agent behavior

- [ ] **Step 2: Run verification**

Run:
```powershell
python -m compileall -q src\occupation_retrieval src\llm
pytest src/tests/test_occupation_top10_selector.py src/tests/test_occupation_retrieval_offline_top10_qc.py src/tests/test_occupation_agent_service.py -v
python -m src.occupation_retrieval.offline_top10_qc --help
```

- [ ] **Step 3: Commit**

```bash
git add src/occupation_retrieval/README.md
git commit -m "docs: describe shared top10 selector and offline arbitration"
```
