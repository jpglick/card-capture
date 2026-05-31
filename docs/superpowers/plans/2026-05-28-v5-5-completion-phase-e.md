# V5.5 Completion — Phase E: Flip Blocking + Final Verification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the env-var gate from the raw-SQL scanner, confirm every architecture test is blocking by default, and tag the v5.5 release.

**Architecture:** Three tasks. Phase E does no implementation work — it only flips the scanner and runs the final verification gauntlet. It assumes Phases A–D have all landed.

**Tech Stack:** pytest, Import Linter.

**Parent plan:** `docs/superpowers/plans/2026-05-28-v5-5-completion.md` (Phase E section). When this plan and the parent disagree, the parent wins.

**Acceptance:**
1. `tests/architecture/test_raw_sql_outside_data.py` no longer reads the `V55_RAW_SQL_BLOCKING` env var — the blocking test always runs.
2. Every command in the Self-Review checklist of the parent plan exits 0 (or produces empty grep output).
3. A `v55-complete` tag exists on the commit where Phase E concludes.
4. `docs/superpowers/plans/v5-5/baseline-results.md` is appended with the final completion baseline (date, test count, lint status, tag).

**Pre-flight context (verified 2026-05-28):**
- `tests/architecture/test_raw_sql_outside_data.py` currently has a docstring reading `Phase 1: advisory. Phase 4: blocking` and still imports `os` for env-var gating. Both must go.
- This plan should not be started until Phase C completes — running it earlier would either gate the test off (the existing behavior) or fail it (21 callers remain). Phase D is recommended before Phase E only because Phase D's lint changes should also be verified by the gauntlet.

---

## File Structure

```text
tests/architecture/test_raw_sql_outside_data.py            Rewritten
docs/superpowers/plans/v5-5/baseline-results.md            Appended
```

---

### Task E.1: Replace `tests/architecture/test_raw_sql_outside_data.py` with the always-blocking version

**Files:**
- Modify: `tests/architecture/test_raw_sql_outside_data.py`

- [ ] **Step 1: Read the current file to confirm the env-var gate**

```bash
grep -n 'V55_RAW_SQL_BLOCKING\|skipif\|advisory' tests/architecture/test_raw_sql_outside_data.py
```

Expected: at least one hit. If none, the file may already be in the always-blocking shape — read it, confirm, and skip to Step 4.

- [ ] **Step 2: Replace the file in full**

`tests/architecture/test_raw_sql_outside_data.py`:

```python
"""Static AST scan: raw SQL string literals outside card_capture.data and migrations.

Blocking by default at Phase E. Allowed roots are listed below; adding a new
root requires a paired plan amendment.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOTS = (
    "src/card_capture/data/",
    "migrations/",
    "tests/",            # tests may contain raw SQL fixtures
    "harness/schema.py",
)

SQL_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|PRAGMA|ALTER|DROP|WITH)\b",
    re.IGNORECASE,
)


def _iter_python_files():
    for root in ("src", "app", "pipeline", "harness"):
        for p in (REPO_ROOT / root).rglob("*.py"):
            rel = str(p.relative_to(REPO_ROOT))
            if any(rel.startswith(a) for a in ALLOWED_ROOTS):
                continue
            yield p


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if SQL_RE.match(node.value):
                out.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: raw SQL literal")
    return out


def test_no_raw_sql_outside_data() -> None:
    violations: list[str] = []
    for p in _iter_python_files():
        violations.extend(_scan(p))
    assert not violations, "\n".join(violations)
```

- [ ] **Step 3: Verify there are no `os`/`skipif`/`V55_RAW_SQL_BLOCKING` references**

```bash
grep -nE 'os|skipif|V55_RAW_SQL_BLOCKING' tests/architecture/test_raw_sql_outside_data.py
```

Expected: no output.

- [ ] **Step 4: Run the test**

```bash
python3 -m pytest tests/architecture/test_raw_sql_outside_data.py -v
```

Expected: PASS. If FAIL with a list of `path:line: raw SQL literal` violations, those are Phase C escapees — go back, migrate them, and re-run. Do NOT add files to `ALLOWED_ROOTS` to make the test pass; the allowlist is policy, not an escape hatch.

- [ ] **Step 5: Commit**

```bash
git add tests/architecture/test_raw_sql_outside_data.py
git commit -m "feat(v55-phaseE): raw-SQL scanner is blocking by default"
```

---

### Task E.2: Run the full Phase-E verification gauntlet

**Files:** none — verification only.

The gauntlet is the parent plan's "Self-review checklist" section. Run every command, capture the output, and only proceed if all six are clean.

- [ ] **Step 1: Full default lane**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q --tb=line 2>&1 | tail -10
```

Expected output ends with `N passed, M skipped, K deselected in …` — `N` should be >= 581 (the original Phase A baseline) plus the new tests added in Phases C/D (rough total >= 605). No `failed` count. No `error` count. Record the actual numbers; they go into the baseline doc in Task E.3.

- [ ] **Step 2: Architecture lane**

```bash
python3 -m pytest tests/architecture/ -v
```

Expected: every test PASS. `test_raw_sql_outside_data` (the new always-blocking version from E.1) is now in the count.

- [ ] **Step 3: Perf smoke**

```bash
python3 -m pytest tests/performance/test_perf_harness_smoke.py -v
```

Expected: PASS.

- [ ] **Step 4: Import Linter**

```bash
PYTHONPATH=src:. lint-imports
```

Expected: every contract reports KEPT. If any are BROKEN:
- `sqlite3` → return to Phase C and find the missed caller. Phase E does NOT add allowlist entries.
- `metaflow` → return to Phase 3 of the original V5.5 plan; should not appear if prior phases were clean.
- `runpod`/`beam` outside `card_capture.platforms` → check Phase D's `app/services/` migration; redirect the importer.
- `cv2.imgcodecs` in `strict_gpu` → real strict-gpu code is using file IO; fix that file, do not relax the contract.
- Layered architecture violation → a module imported across the wrong layer; this is the spec saying "stop." Fix the module, do not weaken the layer.

- [ ] **Step 5: Raw-sqlite3 grep**

```bash
grep -rn 'import sqlite3\|sqlite3.connect' --include='*.py' src/card_capture/ app/ pipeline/ harness/ | grep -v 'src/card_capture/data/\|harness/schema.py'
```

Expected: no output. If anything shows, return to Phase C, migrate it, then re-run the gauntlet.

- [ ] **Step 6: Metaflow grep**

```bash
grep -rn 'import metaflow\|from metaflow' --include='*.py' src/ app/ pipeline/ tests/ harness/ | grep -v '\.venv\|worktrees'
```

Expected: no output.

If any of the six checks fail, **stop and fix the failure**; do not advance to Task E.3. Phase E is the gate, not a checkpoint to skip.

---

### Task E.3: Append the final baseline and tag

**Files:**
- Modify: `docs/superpowers/plans/v5-5/baseline-results.md` (append)

- [ ] **Step 1: Append a completion entry**

Open `docs/superpowers/plans/v5-5/baseline-results.md` and append (do not replace):

```markdown

## V5.5 Completion Baseline — <YYYY-MM-DD>

- **Tag:** `v55-complete`
- **Default lane:** `pytest tests/ -m 'not quarantine' -q` → **<N> passed, <M> skipped, <K> deselected in <T>s**
- **Architecture lane:** `pytest tests/architecture/ -v` → **<X> passed**
- **Perf smoke:** `pytest tests/performance/test_perf_harness_smoke.py -v` → **PASS**
- **Import Linter:** `PYTHONPATH=src:. lint-imports` → **6/6 contracts kept**
- **Raw-sqlite3 outside data:** 0 callsites
- **Metaflow imports outside vendored env:** 0 callsites

All gaps identified in the 2026-05-28 verification of the original V5.5 plan are closed.
```

Replace each `<…>` placeholder with the actual numbers from the Task E.2 gauntlet output.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/v5-5/baseline-results.md
git commit -m "docs(v55-phaseE): record V5.5 completion baseline"
```

- [ ] **Step 3: Tag the release**

```bash
git tag v55-complete
```

Push only when the user requests (`git push origin v55-complete`).

- [ ] **Step 4: Final sanity check**

```bash
git log -3 --oneline
git tag -l | grep v55
```

Expected: top commit is the baseline-results doc; tag `v55-complete` is present in the list.

**Phase E complete.** V5.5 refactoring is done end-to-end: in-process runtime, strict GPU boundary, single-writer data layer, uniform platform adapters, every architectural rule statically enforced.
