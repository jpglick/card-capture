# V5.5 Completion — Phase A: Regression Recovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pytest tests/` collect again so the default lane runs, and commit the `tests/test_unified_runtime.py` rewrite that already exists in the working tree.

**Architecture:** Two tasks. The rewrite work already exists uncommitted in the working tree (verified 2026-05-28); this plan's job is to confirm it matches the parent plan, verify behavior, and commit it.

**Tech Stack:** pytest, `card_capture.pipeline.runtime_local.LocalPipelineRuntime`.

**Parent plan:** `docs/superpowers/plans/2026-05-28-v5-5-completion.md` (Phase A section). When this plan and the parent disagree, the parent wins.

**Pre-flight context:** `tests/test_unified_runtime.py` historically imported `UnifiedRuntime` from a `card_capture.runtime` module that no longer exists (the V5.5 refactor turned `runtime` into a package containing `gpu_session.py`, `batches.py`, etc.). The working tree already contains a rewrite that targets `LocalPipelineRuntime`; this plan ensures it's correct, committed, and exercised by CI.

---

## File Structure

```text
tests/test_unified_runtime.py                Modified; rewrite already in working tree
```

---

### Task A.1: Confirm the working-tree rewrite matches the parent-plan specification

**Files:**
- Read-only: `tests/test_unified_runtime.py`

The parent plan's Task A.1 Step 2 specifies the exact body of the rewrite. Verify what's in the working tree matches that body — including imports, fixture path, schema setup, request shape, and assertions.

- [ ] **Step 1: Diff the working-tree file against the parent-plan specification**

Run:
```bash
diff <(sed -n '/Write `tests\/test_unified_runtime.py`:/,/```$/p' docs/superpowers/plans/2026-05-28-v5-5-completion.md | sed '1d;$d') tests/test_unified_runtime.py
```

Expected: no semantically meaningful diff. Whitespace and trailing-newline differences are acceptable. If the working-tree file diverges from the parent plan (e.g., wrong fixture path, wrong import, missing assertion), align it before continuing.

- [ ] **Step 2: Verify imports resolve**

Run:
```bash
python3 -c "
from card_capture.data.connection import open_connection
from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry
print('ok')
"
```

Expected: `ok`. If any import fails, the working tree is in an inconsistent state — surface the failure and stop; do not patch around it.

---

### Task A.2: Verify pytest can collect the file, then commit

**Files:**
- Modify: `tests/test_unified_runtime.py` (commit only — no edits)

- [ ] **Step 1: Confirm pytest collection succeeds**

Run:
```bash
python3 -m pytest tests/test_unified_runtime.py --collect-only -q
```

Expected: one collected item, no `ImportError`. The test will SKIP at execution time if `tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV` is absent; that's fine — collection is what matters for CI.

- [ ] **Step 2: Run the file (skip is acceptable)**

Run:
```bash
python3 -m pytest tests/test_unified_runtime.py -v
```

Expected: PASS, or SKIPPED with the reason `Golden-set video IMG_5872.MOV not present (large binary, not in repo)`.

- [ ] **Step 3: Confirm the full default lane collects**

Run:
```bash
python3 -m pytest tests/ -m 'not quarantine' --collect-only 2>&1 | tail -5
```

Expected: a line of the form `N tests collected in Xs`, no `error` lines. The full suite has 27 known regressions from prior Phase C work (15 FastAPI errors + 12 unrelated failures) that Phase C closes; this plan only requires that collection succeeds.

- [ ] **Step 4: Commit ONLY this file**

Run:
```bash
git status tests/test_unified_runtime.py
git diff --stat tests/test_unified_runtime.py
git add tests/test_unified_runtime.py
git commit -m "test(v55-phaseA): rewrite stale UnifiedRuntime smoke to LocalPipelineRuntime

Targets card_capture.pipeline.runtime_local.LocalPipelineRuntime and the
serializable PipelineRunRequest shape. Skips at runtime if the golden-set
fixture is absent; collection always succeeds.

Fixes pytest collection error introduced when card_capture.runtime became
a package without UnifiedRuntime."
```

- [ ] **Step 5: Confirm the commit landed**

Run:
```bash
git log -1 --oneline
git status tests/test_unified_runtime.py
```

Expected: top commit is the Phase A commit; `git status` reports no changes for that file.

**Phase A complete.** Default-lane collection works. The 27 unrelated test failures/errors stay in the working tree for Phase C to close.
