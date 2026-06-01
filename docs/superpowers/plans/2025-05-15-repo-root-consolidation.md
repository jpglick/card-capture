# Repo-root data/output consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate scratch data, output, and database files under a single `var/` directory and update code defaults.

**Architecture:** Move untracked scratch data using `mv` and tracked directories using `git mv`. Repoint CLI and app defaults to `var/`. Clean up `.gitignore`.

**Tech Stack:** Bash, Python, Git

---

### Task 1: Initialize `var/` structure and move untracked data

**Files:**
- Create: `var/output/`, `var/uploads/`, `var/db/`, `var/reports/`
- Move: `card_capture_output/*`, `card_capture_uploads/*`, `data/*.sqlite`, `data/pipeline.db`, `sample_run_2026_05_03_fixes`, `sample_run_2026_05_03_fixes_q256`, `cards.sqlite`

- [ ] **Step 1: Create directories**
```bash
mkdir -p var/output var/uploads var/db var/reports
```

- [ ] **Step 2: Move untracked files/directories**
```bash
# Move card_capture_output contents
mv card_capture_output/* var/output/ 2>/dev/null || true
# Move sample runs
mv sample_run_2026_05_03_fixes var/output/ 2>/dev/null || true
mv sample_run_2026_05_03_fixes_q256 var/output/ 2>/dev/null || true
# Move uploads
mv card_capture_uploads/* var/uploads/ 2>/dev/null || true
# Move databases
mv data/*.sqlite data/pipeline.db var/db/ 2>/dev/null || true
mv cards.sqlite var/db/ 2>/dev/null || true
```

- [ ] **Step 3: Verify moves**
```bash
ls -R var/
```

### Task 2: Move tracked directories

**Files:**
- Modify: `out/` (git mv to `var/output/`)
- Modify: `reports/` (git mv to `var/reports/`)

- [ ] **Step 1: Move `out/`**
```bash
# out/ was reported TRACKED. Its content includes cards.sqlite which we want in var/db, 
# but the instruction says move contents to var/output. 
# Wait, Task 19 says: "mv out/* var/output/". 
# But it also says: "mv out/cards.sqlite ... var/db/".
# I will follow the specific move instructions.
git mv out/* var/output/
# If cards.sqlite was in out/, move it to var/db/
mv var/output/cards.sqlite var/db/ 2>/dev/null || true
```

- [ ] **Step 2: Move `reports/`**
```bash
# reports/ was reported TRACKED. 
# We want to keep its .gitkeep behavior but move contents.
# The instruction says: "mv reports/* var/reports/ 2>/dev/null || true"
# But reports is TRACKED. Let's check if reports/ itself is tracked or just files inside.
# Inventory said "reports: TRACKED".
git mv reports/* var/reports/
```

- [ ] **Step 3: Clean up empty tracked dirs if needed**
```bash
# git mv should have handled it, but let's check
git status
```

### Task 3: Repoint default DB/output paths in code

**Files:**
- Modify: `src/card_capture/cli.py`
- Modify: `src/card_capture/review/app.py`

- [ ] **Step 1: Find strings to replace**
```bash
grep -rn "card_capture_output\|'out'\|\"out\"" src/card_capture/cli.py src/card_capture/review/app.py
```

- [ ] **Step 2: Update `src/card_capture/cli.py`**
Replace defaults like `card_capture_output/cards.sqlite` with `var/db/cards.sqlite` and `--output-dir` default `out` or `card_capture_output` with `var/output`.

- [ ] **Step 3: Update `src/card_capture/review/app.py`**
Replace defaults as found.

### Task 4: Finalize `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Remove obsolete lines**
Remove `out/`, `card_capture_output/`, `card_capture_uploads/`, `reports/`, `data/*.sqlite`, root `cards.sqlite`.

- [ ] **Step 2: Add `var/`**
Add `var/` (except perhaps `.gitkeep` files if we want to keep the structure).
Actually, we should probably ignore `var/` but keep the directory structure if desired, or just ignore everything in it.
The task says "superseded by var/". I'll add `var/` to `.gitignore`.

### Task 5: Verification

- [ ] **Step 1: Check status**
```bash
git status --porcelain | head -40
```

- [ ] **Step 2: Check preserved assets**
```bash
ls models/ golden_set/
```

- [ ] **Step 3: Run tests**
```bash
./.venv/bin/python -m pytest tests/ -m "not quarantine" -q
```

- [ ] **Step 4: Lint imports**
```bash
./.venv/bin/lint-imports
```

### Task 6: Commit

- [ ] **Step 1: Stage and commit**
```bash
git add -A
git commit -m "chore(reorg): consolidate scratch data/output/db under var/; repoint defaults"
```
