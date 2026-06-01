# Documentation Pointer Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up root documentation by moving files to `docs/`, updating `CLAUDE.md` to reflect the new 7-layer architecture, and turning `AGENTS.md` and `GEMINI.md` into thin pointers.

**Architecture:** Documentation reorganization to improve repository hygiene and canonicalize engineering standards in `CLAUDE.md`.

**Tech Stack:** Bash, Markdown, Makefile, Python/Pytest (for SVB).

---

### Task 1: Move Root Documentation

**Files:**
- Move: `OPERATOR.md` -> `docs/OPERATOR.md`
- Move: `QUICK_REFERENCE.md` -> `docs/QUICK_REFERENCE.md`

- [ ] **Step 1: Move files using git mv**

Run: `git mv OPERATOR.md docs/OPERATOR.md`
Run: `git mv QUICK_REFERENCE.md docs/QUICK_REFERENCE.md`

- [ ] **Step 2: Commit moves**

```bash
git commit -m "docs: move OPERATOR and QUICK_REFERENCE to docs/"
```

### Task 2: Update `CLAUDE.md` Module Map

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the "Key Modules" section**

Replace the existing "Key Modules" table with the new 7-layer layout.

```markdown
## Key Modules

src/card_capture/
  core/         # foundation layer — leaf utilities + types (models, config, gpu_utils)
  stages/       # vertical slices (sample, detect, novelty, track, refine, score, resolve, fuse, dedup, store)
  shared/       # cross-stage helpers (pipeline_utils, stage_metrics)
  ml/           # model zoo + inference (shared assets)
  training/     # all offline training logic
  runtime/      # GPU session orchestration (gpu_session, strict_gpu)
  pipeline/     # orchestration (request, runner, telemetry, runtime_local)
  data/         # DAL (connection, repositories)
  review/       # legacy Jinja review UI
```

- [ ] **Step 2: Commit update**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md module map to reflect 7-layer layout"
```

### Task 3: Update `AGENTS.md` and `GEMINI.md` to Thin Pointers

**Files:**
- Modify: `AGENTS.md`
- Modify: `GEMINI.md`

- [ ] **Step 1: Replace content of `AGENTS.md`**

Content:
```markdown
# AI Agent Mandates

This repository has been reorganized. Canonical agent context and engineering standards now live in `CLAUDE.md`. Refer to `docs/architecture/` for the module map.
```

- [ ] **Step 2: Replace content of `GEMINI.md`**

Content:
```markdown
# Project Instructions: Card Capture

This repository has been reorganized. Canonical agent context and engineering standards now live in `CLAUDE.md`. Refer to `docs/architecture/` for the module map.
```

- [ ] **Step 3: Commit changes**

```bash
git add AGENTS.md GEMINI.md
git commit -m "docs: turn AGENTS.md and GEMINI.md into thin pointers to CLAUDE.md"
```

### Task 4: Update `Makefile` References

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Search for obsolete directory references**

Run: `grep -n "out/\|card_capture_output" Makefile`

- [ ] **Step 2: Update `clean` or `test` targets to use `var/`**

If any matches are found, replace them with `var/`.

- [ ] **Step 3: Commit changes**

```bash
git add Makefile
git commit -m "build: update Makefile to use var/ for output artifacts"
```

### Task 5: Step-Value-Baseline (SVB) and Final Verification

**Files:**
- Run: `tests/`
- Run: `lint-imports`

- [ ] **Step 1: Run unit tests**

Run: `./.venv/bin/python -m pytest tests/ -m "not quarantine" -q`
Expected: Tests pass.

- [ ] **Step 2: Run import linter**

Run: `./.venv/bin/lint-imports`
Expected: No import violations.

- [ ] **Step 3: Final git status check**

Run: `git status`
Expected: Working tree clean.
