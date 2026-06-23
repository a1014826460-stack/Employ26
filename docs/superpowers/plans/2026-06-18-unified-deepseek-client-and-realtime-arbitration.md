# Unified DeepSeek Client And Realtime Arbitration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified DeepSeek calling entry under `src/llm`, support threaded batch usage, migrate `deepseek_round2_relabel.py` to it, and enable real-time DeepSeek arbitration in `offline_top10_qc.py` only when `LLM winner != human gold`.

**Architecture:** Create a focused `src/llm/deepseek_client.py` module that owns API config, JSON response parsing, retries, and threaded batch execution. Refactor the round2 relabel script to reuse this client, then replace the current offline QC DeepSeek reference with a real-time arbitration step that reads the full Top10 and explicitly returns one of `support_llm`, `support_gold`, or `support_neither`.

**Tech Stack:** Python, OpenAI-compatible DeepSeek API, `concurrent.futures`, pytest, existing occupation retrieval QC pipeline.

---

### Task 1: Add the shared DeepSeek client with tests

**Files:**
- Create: `src/llm/deepseek_client.py`
- Create: `src/tests/test_deepseek_client.py`

- [ ] **Step 1: Write the failing test**

```python
from src.llm.deepseek_client import DeepSeekClient, DeepSeekConfig, parse_json_response


def test_parse_json_response_accepts_fenced_json():
    assert parse_json_response('```json\n{"ok": true}\n```') == {"ok": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_deepseek_client.py -v`
Expected: FAIL because the module does not exist yet.

- [ ] **Step 3: Implement the client**

Include:
- `DeepSeekConfig`
- `DeepSeekClient.complete_json(...)`
- `DeepSeekClient.map_json(...)`
- response parsing
- retries and timeout

- [ ] **Step 4: Re-run tests**

Run: `pytest src/tests/test_deepseek_client.py -v`
Expected: PASS.

### Task 2: Refactor `deepseek_round2_relabel.py` to use the shared client

**Files:**
- Modify: `src/anno_analysis/deepseek_round2_relabel.py`
- Modify: `src/tests/test_deepseek_client.py`

- [ ] **Step 1: Replace direct `OpenAI(...)` usage**

Move single-call logic onto the shared client while preserving:
- model name
- timeout
- strict JSON behavior
- retry semantics

- [ ] **Step 2: Keep threaded execution in the script**

The script may keep its task orchestration, but the underlying API call must go through the shared client.

- [ ] **Step 3: Verify**

Run: `python -m compileall -q src\anno_analysis src\llm`

### Task 3: Add real-time DeepSeek arbitration to offline QC

**Files:**
- Modify: `src/occupation_retrieval/offline_top10_qc.py`
- Modify: `src/tests/test_occupation_retrieval_offline_top10_qc.py`

- [ ] **Step 1: Add failing tests**

Cover:
- only arbitrate when `llm_selected_code != gold_code`
- prompt includes full Top10
- output supports `support_llm`, `support_gold`, or `support_neither`

- [ ] **Step 2: Implement arbitration**

Add a helper that:
- reads `job_title`, `job_requirements`, full `top10_candidates`
- highlights `llm winner`, `human gold`, and raw `top1`
- asks DeepSeek to output structured JSON with:
  - `support`
  - `reason`
  - `review_needed`

- [ ] **Step 3: Re-run QC tests**

Run: `pytest src/tests/test_occupation_retrieval_offline_top10_qc.py -v`
Expected: PASS.

### Task 4: Final verification

**Files:**
- Modify: `src/occupation_retrieval/README.md`

- [ ] **Step 1: Update docs**

Clarify that:
- `offline_top10_qc.py` now supports real-time DeepSeek arbitration
- arbitration sees full Top10 and explicitly supports LLM / gold / neither
- `src/llm/deepseek_client.py` is the shared DeepSeek entry point

- [ ] **Step 2: Run verification**

Run:
```powershell
python -m compileall -q src\llm src\anno_analysis src\occupation_retrieval
pytest src/tests/test_deepseek_client.py src/tests/test_occupation_retrieval_offline_top10_qc.py -v
python -m src.occupation_retrieval.offline_top10_qc --help
```
