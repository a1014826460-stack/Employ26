# Label Studio Task Source Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-task source-identity layer for `annotations.label_studio_tasks_v2`, persist both primary identity and candidate evidence in PostgreSQL, and switch offline QC to use the new identity mapping.

**Architecture:** Add one DB helper module for schema creation and upserts, one full-build CLI that derives task-to-source identity from current task payload text and normalized/sample tables, and update offline QC to read the new identity layer instead of directly trusting task-table `recruitment_record_id`.

**Tech Stack:** Python, SQLAlchemy, PostgreSQL, pytest, pandas

---

### Task 1: Add identity-table DB helpers

**Files:**
- Create: `src/db/label_studio_task_source_identity.py`
- Test: `src/tests/test_label_studio_task_source_identity.py`

- [ ] **Step 1: Write failing tests for table helper behavior**

- [ ] **Step 2: Implement create/upsert/query helpers**

- [ ] **Step 3: Run targeted tests**

### Task 2: Build full identity resolver CLI

**Files:**
- Create: `src/utils/build_label_studio_task_source_identity.py`
- Modify: `src/utils/repair_label_studio_recruitment_record_ids.py`
- Test: `src/tests/test_build_label_studio_task_source_identity.py`

- [ ] **Step 1: Write failing tests for resolver outputs**

- [ ] **Step 2: Implement task payload fingerprinting, candidate ranking, main-identity row generation**

- [ ] **Step 3: Implement CLI dry-run / apply flow**

- [ ] **Step 4: Run targeted tests**

### Task 3: Switch offline QC to the new identity layer

**Files:**
- Modify: `src/occupation_retrieval/common.py`
- Modify: `src/occupation_retrieval/offline_top10_qc.py`
- Modify: `src/tests/test_occupation_retrieval_offline_top10_qc.py`

- [ ] **Step 1: Write failing tests for identity-status-aware QC**

- [ ] **Step 2: Implement identity-aware task loading and QC summary/report updates**

- [ ] **Step 3: Run targeted tests**

### Task 4: Update docs and run end-to-end verification

**Files:**
- Modify: `src/occupation_retrieval/README.md`
- Output: `output/occupation_retrieval/offline_top10_qc_report.md`

- [ ] **Step 1: Document the new identity layer and QC behavior**

- [ ] **Step 2: Run full identity build against PostgreSQL**

- [ ] **Step 3: Run offline QC with the new mapping**

- [ ] **Step 4: Verify row counts and report key metrics**
