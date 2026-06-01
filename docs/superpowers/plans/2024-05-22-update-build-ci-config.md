# Task 6: Update build, CI, and dependency config Implementation Plan (Updated)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish updating build, CI, and dependency config. Renames `pipeline_v21` to `legacy_tracking` and verifies CI scripts.

**Architecture:** Rename extra in `pyproject.toml` and documentation. Update CI if necessary.

**Tech Stack:** GitHub Actions, Python (pip, pyproject.toml), Pytest, Import Linter.

---

### Task 1: Update Documentation (OPERATOR.md, QUICK_REFERENCE.md, README.md)

**Files:**
- Modify: `OPERATOR.md`
- Modify: `QUICK_REFERENCE.md`
- Modify: `README.md` (Finish updates)

- [ ] **Step 1: Replace `pipeline_v21` with `legacy_tracking` in `OPERATOR.md`**

- [ ] **Step 2: Replace `pipeline_v21` with `legacy_tracking` in `QUICK_REFERENCE.md`**

- [ ] **Step 3: Ensure `README.md` is fully updated**

---

### Task 2: Verify CI Scripts and Directories

**Files:**
- Verify: `scripts/validate_schema_docs.py`
- Verify: `tests/architecture`
- Verify: `tests/performance/test_perf_harness_smoke.py`
- Modify: `.github/workflows/ci.yml` (if needed)

- [ ] **Step 1: Check existence of files**
Run: `ls scripts/validate_schema_docs.py tests/architecture tests/performance/test_perf_harness_smoke.py`

- [ ] **Step 2: Update `ci.yml` if any are missing**

---

### Task 3: Final Validation

- [ ] **Step 1: Run tests**
Run: `.venv/bin/python -m pytest tests/ -m "not quarantine" -q`

- [ ] **Step 2: Run import linter**
Run: `.venv/bin/python -m importlinter.cli lint || lint-imports`

---

### Task 4: Stage and Commit

- [ ] **Step 1: Stage remaining changes**
Run: `git add README.md pyproject.toml src/card_capture/tracking/botsort_adapter.py tests/test_pipeline.py OPERATOR.md QUICK_REFERENCE.md` (and `.github/workflows/ci.yml` if changed)

- [ ] **Step 2: Commit with specified message**
```bash
git commit -m "ci: drop cloud/CUDA extras from install and build config

Renames pipeline_v21 extra to legacy_tracking to reflect its use
for BoT-SORT and PyAV fallbacks.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
