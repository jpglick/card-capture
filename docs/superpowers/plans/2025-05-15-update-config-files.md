# Update Config Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `.importlinter`, `pyproject.toml`, and `.gitignore` to reflect the new repository organization and architecture.

**Architecture:** Update the layered architecture contract in `.importlinter` to include all 7 layers. Repoint template paths in `pyproject.toml`. Add consolidated scratch directory to `.gitignore`.

**Tech Stack:** Import Linter, Setuptools (pyproject.toml), Git.

---

### Task 1: Update `.importlinter` Layered Contract

**Files:**
- Modify: `.importlinter`

- [ ] **Step 1: Replace the layered contract in `.importlinter`**

Update the `[importlinter:contract:layered]` section to include all 7 layers in the specified order.

```ini
[importlinter:contract:layered]
name = layered architecture
type = layers
containers =
    card_capture
layers =
    runtime
    pipeline
    stages
    shared
    ml
    data
    core
```

- [ ] **Step 2: Commit changes**

```bash
git add .importlinter
git commit -m "refactor(reorg): update import-linter layers to 7-layer contract"
```

---

### Task 2: Update `pyproject.toml` Package Data

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `package-data` for relocated templates**

Update the `[tool.setuptools.package-data]` entry to use the new template path.

```toml
[tool.setuptools.package-data]
card_capture = ["review/templates/*.html"]
```

- [ ] **Step 2: Commit changes**

```bash
git add pyproject.toml
git commit -m "refactor(reorg): update package-data for relocated templates"
```

---

### Task 3: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add `var/` to `.gitignore`**

Append the consolidated scratch directory to the end of the file.

```bash
printf '\n# consolidated scratch (run outputs, uploads, working DBs, reports)\nvar/\n' >> .gitignore
```

- [ ] **Step 2: Commit changes**

```bash
git add .gitignore
git commit -m "refactor(reorg): add var/ to .gitignore"
```

---

### Task 4: Verification (SVB)

- [ ] **Step 1: Run tests and import linter**

Run the following commands to verify that the changes are correct and do not introduce regressions.

```bash
./.venv/bin/python -m pytest tests/ -m "not quarantine" -q
./.venv/bin/lint-imports
```

Expected: `lint-imports` should pass with the new 7-layer contract.

- [ ] **Step 2: Final Commit (if any fixes were needed during SVB)**

If any import cycles were found and fixed, commit them.

---
