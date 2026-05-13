# Wave 4 — Surface E (Foundations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the foundational gates (CI, contract drift detection, monolith/Metaflow ADR, PR conventions) that every other Wave 4 surface depends on.

**Architecture:** Single agent, ~5 PRs, serial. Each PR closes one V4_CONCERNS section and updates that section's status from §1 (open) to §2 (resolved). Surfaces A/B/C/D are blocked on E1 (CI workflow) being merged.

**Tech Stack:** GitHub Actions, pytest, Pydantic v2, Python 3.11.

**Spec:** `docs/superpowers/specs/2026-05-13-v4-wave4-hardening-design.md` §3.

**Files owned by Surface E:** `.github/**`, `docs/decisions/**`, `tests/contracts/**`, `tests/app/test_api_contract.py`, `docs/contributing.md`.

---

## Pre-flight

- [ ] **P1: Create the worktree**

Invoke `superpowers:using-git-worktrees` to create an isolated workspace on branch `wave4/e-foundations`:

```bash
git worktree add ../card-capture-wave4-e -b wave4/e-foundations origin/main
cd ../card-capture-wave4-e
```

All subsequent commands run in this worktree.

- [ ] **P2: Verify the test suite runs green locally**

Run: `pip install -e ".[harness,test]" && python -m pytest tests/ -q`

Expected: tests pass (some may be marked `xfail` or `skip`; that's fine). If anything is failing, stop and report — the CI workflow needs a green baseline before it can gate.

---

## Task 1: E1 — GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1.1: Inspect `pyproject.toml` for extras**

Read `pyproject.toml` and confirm `[project.optional-dependencies]` declares `harness` and `test` groups. Note the Python version supported (should be `>=3.10`).

- [ ] **Step 1.2: Write the workflow**

Create `.github/workflows/test.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[harness,test]"

      - name: Run pytest
        run: python -m pytest tests/ -q
```

- [ ] **Step 1.3: Push and verify the workflow runs**

```bash
git add .github/workflows/test.yml
git commit -m "ci(wave4): add pytest workflow on push and PR

Closes V4_CONCERNS §1.9.

Manual follow-up by repo owner: enable branch protection on main and
require this workflow as a status check.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push -u origin wave4/e-foundations
```

Then check `gh run list --limit 3` — the new workflow run for this branch should appear. Wait for completion: `gh run watch <run-id>`. Expected: green.

If red, fix locally and force-update the branch (`git push --force-with-lease`) — do not merge a red CI.

- [ ] **Step 1.4: Open the PR**

```bash
gh pr create --title "[Wave 4 — Surface E] CI workflow (E1)" --body "$(cat <<'EOF'
## Summary
- Adds GitHub Actions workflow running pytest on every push and PR.
- Closes V4_CONCERNS §1.9.

## Manual follow-up required (by repo owner)
The workflow runs but is not yet a required status check. Enable branch
protection on `main` and check "Require status checks to pass before
merging" → "tests".

## Test plan
- [x] new test added: workflow itself (the existing test suite is the workload)
- [x] pytest tests/ green locally
- [x] CI green (first run completes on push)
- [ ] contract docs updated: N/A
- [ ] drift gate green: N/A (E3 not yet landed)
EOF
)"
```

- [ ] **Step 1.5: Update V4_CONCERNS.md**

In the same branch, edit `V4_CONCERNS.md`:

1. Replace the body of §1.9 ("No CI gate exists; manual review is the only test signal — **High**") with: `**Resolved (see §2.12)**`.
2. Append a new §2.12 in the Resolved section:

```markdown
### 2.12 — *(was §1.9)* No CI gate exists — **Resolved** in PR #<N>

`.github/workflows/test.yml` runs `pytest tests/` on push and PR. Manual
follow-up: repo owner must enable branch protection on `main` to make
this a required status check.
```

(Use the actual PR number once `gh pr create` returns it.) Commit and push:

```bash
git add V4_CONCERNS.md
git commit -m "docs(wave4): mark §1.9 resolved by CI workflow

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

Wait for the user to merge the PR before continuing. After merge, `git fetch && git rebase origin/main` to pick up the merged state for subsequent tasks (though the worktree won't have conflicts since E owns its files).

---

## Task 2: E2 — Monolith vs Metaflow ADR

**Files:**
- Create: `docs/decisions/2026-05-13-pipeline-canonical-path.md`
- Modify: `V4_CONCERNS.md` (link from §1.1)

- [ ] **Step 2.1: Inspect both pipelines**

Read enough of each to write a credible decision:

```bash
wc -l src/card_capture/pipeline.py pipeline/card_capture_flow.py
ls pipeline/steps/
grep -l "VideoProcessor\|CardCaptureFlow" src/ app/ harness/ -r
```

Expected: `pipeline.py` ≈ 2,000 lines monolith with a `VideoProcessor` class; `card_capture_flow.py` ≈ 100 lines with a `CardCaptureFlow(FlowSpec)`; `pipeline/steps/` has 9 step modules. The grep tells you which entry points use which.

- [ ] **Step 2.2: Write the ADR**

Create `docs/decisions/2026-05-13-pipeline-canonical-path.md`:

```markdown
# ADR: Canonical Pipeline Path — Metaflow

**Date:** 2026-05-13
**Status:** Accepted
**Author:** Surface E agent
**Supersedes:** none

## Context

The Wave 1–3 implementation produced two parallel pipeline implementations:

1. `src/card_capture/pipeline.py` — the original monolithic
   `VideoProcessor` class. ~2,000 lines. All algorithmic state is held
   inside this class; entry points include the headless
   `card-capture process` CLI, the harness CLI's pipeline-execution
   path, and `app/services/pipeline_runner.py`.

2. `pipeline/card_capture_flow.py` + `pipeline/steps/*` — the Metaflow
   `CardCaptureFlow` introduced in Wave 2 (PR #45). ~100 lines of flow
   spine; nine step modules under `pipeline/steps/`. Designed for
   artifact persistence, resume, and per-step parallelism.

Wave 3 modified both paths but not equivalently — the
`feat(fusion)` commit re-enabled multi-frame fusion in the monolith;
the Wave 3 commit consolidated fusion into `MultiFrameFuser` and
touched `pipeline/steps/fuse.py`. The two paths now drift, and there
is no test that asserts they produce the same artifacts on a fixture
video. This is V4_CONCERNS §1.1.

## Decision

**Metaflow (`pipeline/card_capture_flow.py`) is the canonical path.**

Reasons:

1. The original v4 plan (`CLAUDE.md` Appendix A.1 Phase 2) chose
   Metaflow because of artifact persistence (required for the
   threshold-tuning playground, §A.5.3) and `resume` (required for
   fast iteration on Stages 4–10 without re-running the detector).
2. The Metaflow spine is small (~100 lines); each step is a focused
   module. The monolith concentrates state in one 2,000-line class
   that's hard to test, hard to reason about, and hard to extend.
3. All Wave 2 algorithmic changes (RANSAC, DINOv2+FAISS, ByteTrack,
   active learning) were primarily implemented against the Metaflow
   steps. The monolith carries lagging copies.

## Consequences

- The monolith will carry a `DeprecationWarning` from A1 onwards;
  removal is a Wave 5 task.
- Every entry point routes through `CardCaptureFlow`:
  - `src/card_capture/cli.py` (the `card-capture process` command)
  - `harness/cli.py` (the harness's pipeline-execution path, if any)
  - `app/services/pipeline_runner.py`
- A smoke test (added in A1) runs both paths on a fixture video and
  asserts equivalence until the monolith is deleted.

## Alternatives considered

**Keep the monolith, delete Metaflow.** Rejected because: artifact
persistence and resume are load-bearing for §A.5.3 (the threshold
playground, already implemented in `app/services/playground_service.py`),
and re-implementing them on top of the monolith is its own project.

**Keep both indefinitely.** Rejected because: this is exactly the
state §1.1 flags as a problem. "Both" guarantees drift.

## References

- V4_CONCERNS.md §1.1
- CLAUDE.md Appendix A.1 Phase 2, A.8
- Wave 4 spec: `docs/superpowers/specs/2026-05-13-v4-wave4-hardening-design.md` §4 (Surface A)
```

- [ ] **Step 2.3: Link from V4_CONCERNS.md**

Edit V4_CONCERNS.md §1.1: append at the end of the body:

```markdown
**ADR:** `docs/decisions/2026-05-13-pipeline-canonical-path.md` — Metaflow
selected as canonical. Execution lives with Surface A task A1.
```

- [ ] **Step 2.4: Commit and open PR**

```bash
git add docs/decisions/ V4_CONCERNS.md
git commit -m "docs(wave4): ADR — Metaflow is the canonical pipeline path

Closes V4_CONCERNS §1.1 (decision). Execution is Surface A task A1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface E] Pipeline canonical-path ADR (E2)" --body "$(cat <<'EOF'
## Summary
- Adds ADR selecting Metaflow as the canonical pipeline path.
- Wave 5 deletes the monolith; A1 handles deprecation and entry-point routing.
- Closes V4_CONCERNS §1.1 (decision-only; execution in A1).

## Test plan
- [x] ADR committed
- [x] V4_CONCERNS §1.1 links to ADR
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 3: E3 — Schema/contract drift gate

**Files:**
- Create: `tests/contracts/__init__.py`
- Create: `tests/contracts/test_drift.py`

- [ ] **Step 3.1: Write the failing test (drift detection on truth schema)**

Create `tests/contracts/__init__.py` as empty file:

```bash
mkdir -p tests/contracts
touch tests/contracts/__init__.py
```

Create `tests/contracts/test_drift.py`:

```python
"""Drift gate — assert Pydantic schemas + SQL DDL match their markdown contracts.

Closes V4_CONCERNS §1.10.

When this test fails, the offending code/doc pair is out of sync. Fix
*both* sides — the contracts under `docs/contracts/` are the canonical
shape; the Pydantic models and SQL DDL must match them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from harness.schema import ExpectedCard, TruthFile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_DIR = REPO_ROOT / "docs" / "contracts"


def _read(path: Path) -> str:
    return path.read_text()


def _pydantic_field_names(model) -> set[str]:
    return set(model.model_fields.keys())


def test_truth_schema_pydantic_fields_appear_in_contract():
    """Every TruthFile / ExpectedCard field must appear in truth-schema.md."""
    contract = _read(CONTRACTS_DIR / "truth-schema.md")
    expected_fields = _pydantic_field_names(TruthFile) | _pydantic_field_names(
        ExpectedCard
    )
    missing = [f for f in expected_fields if f"`{f}`" not in contract]
    assert not missing, (
        f"Pydantic fields missing from truth-schema.md: {missing}. "
        f"Update the markdown contract or remove the field."
    )


def test_storage_schema_columns_appear_in_contract():
    """Every column in 0001_v4_schema.sql must appear in storage-schema.md."""
    ddl = _read(REPO_ROOT / "migrations" / "0001_v4_schema.sql")
    contract = _read(CONTRACTS_DIR / "storage-schema.md")

    # Extract column names from CREATE TABLE blocks. Match identifiers at
    # line-start (after whitespace) followed by a SQL type.
    column_pattern = re.compile(
        r"^\s+([a-z_]+)\s+(INTEGER|TEXT|REAL|BLOB|NUMERIC)",
        re.MULTILINE,
    )
    columns = {m.group(1) for m in column_pattern.finditer(ddl)}
    # Exclude PRIMARY KEY pseudo-rows that don't define a column.
    columns -= {"primary"}

    missing = [c for c in columns if c not in contract]
    assert not missing, (
        f"DDL columns missing from storage-schema.md: {missing}. "
        f"Update the markdown contract or remove the column."
    )
```

- [ ] **Step 3.2: Run the test — expect PASS (current state is in sync)**

Run: `pytest tests/contracts/test_drift.py -v`
Expected: 2 passed.

If failures occur, the contracts and code are already out of sync — fix the doc to match the code (the code is what currently ships).

- [ ] **Step 3.3: Verify the gate actually catches drift (manual sanity check)**

Temporarily rename a field in `harness/schema.py` (e.g. `is_foil` → `is_foil_renamed`). Re-run the test.

```bash
sed -i.bak 's/is_foil: bool/is_foil_renamed: bool/' harness/schema.py
pytest tests/contracts/test_drift.py::test_truth_schema_pydantic_fields_appear_in_contract -v
```

Expected: FAIL with `Pydantic fields missing from truth-schema.md: ['is_foil_renamed']`.

Revert the change:

```bash
mv harness/schema.py.bak harness/schema.py
pytest tests/contracts/test_drift.py -v
```

Expected: 2 passed.

- [ ] **Step 3.4: Commit and open PR**

```bash
git add tests/contracts/
git commit -m "test(wave4): contract drift gate for truth + storage schemas

Asserts every Pydantic field on TruthFile/ExpectedCard and every column
in migrations/0001_v4_schema.sql appears in the matching markdown
contract under docs/contracts/. Fails on drift.

Verified by deliberately renaming a field and confirming the gate
fires; reverted before commit.

Closes V4_CONCERNS §1.10.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface E] Contract drift gate (E3)" --body "$(cat <<'EOF'
## Summary
- New `tests/contracts/test_drift.py` asserts Pydantic TruthFile/ExpectedCard fields and v4 SQL columns appear in their markdown contracts.
- Verified gate catches drift by deliberately renaming a field (reverted before commit).

## Test plan
- [x] new test added: tests/contracts/test_drift.py
- [x] pytest tests/ green locally
- [x] CI green
- [x] drift gate verified to catch a deliberate drift
EOF
)"
```

Wait for merge.

---

## Task 4: E4 — Contract-2 conformance hardening

**Files:**
- Modify: `tests/app/test_api_contract.py`

- [ ] **Step 4.1: Read current state of the test file**

```bash
cat tests/app/test_api_contract.py
```

Note: it currently checks (i) routes registered, (ii) routes in OpenAPI, (iii) stub routes return 501, (iv) implemented-route response shapes via `TypeAdapter` (added in §2.10). Goal of E4 is to add (v) request-shape validation on POST/PUT/PATCH and (vi) "no undocumented routes."

- [ ] **Step 4.2: Add the new tests**

Append to `tests/app/test_api_contract.py`:

```python
# ---------------------------------------------------------------------
# E4: request-shape validation on POST/PUT/PATCH routes
# ---------------------------------------------------------------------

from app.schemas.v1 import VideoCreate, LabelFB, LabelTruth, RetrainRequest

REQUEST_BODIES = {
    # (method, path): pydantic_model
    ("POST", "/api/v1/videos"): VideoCreate,
    ("POST", "/api/v1/label/fb"): LabelFB,
    ("PUT", "/api/v1/label/truth/{video_id}"): LabelTruth,
    ("POST", "/api/v1/training/retrain/{model_name}"): RetrainRequest,
}


def test_post_put_patch_request_bodies_match_pydantic_models():
    """Every documented POST/PUT/PATCH path declares a request model that
    matches the Pydantic model used in app/schemas/v1.py.
    """
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    for (method, path), model in REQUEST_BODIES.items():
        assert path in paths, f"missing path: {path}"
        method_spec = paths[path][method.lower()]
        body_ref = (
            method_spec.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("$ref")
        )
        assert body_ref, f"{method} {path} has no JSON request body schema"
        # $ref is like "#/components/schemas/VideoCreate"
        ref_name = body_ref.rsplit("/", 1)[-1]
        assert ref_name == model.__name__, (
            f"{method} {path} declares body {ref_name!r}; "
            f"expected {model.__name__!r}"
        )


# ---------------------------------------------------------------------
# E4: no undocumented routes
# ---------------------------------------------------------------------

INTERNAL_PATHS = {
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}


def test_no_undocumented_routes():
    """Every non-internal route in the FastAPI app must appear in
    ROUTES_REQUIRED or REQUEST_BODIES. Catches routes added without
    contract documentation.
    """
    client = TestClient(create_app())
    registered = {
        (list(r.methods)[0], r.path)
        for r in client.app.routes
        if hasattr(r, "methods") and r.methods
    }

    documented = {(m, p) for m, p, _ in ROUTES_REQUIRED}
    documented |= set(REQUEST_BODIES.keys())

    undocumented = {
        (m, p) for m, p in registered
        if p not in INTERNAL_PATHS and (m, p) not in documented
    }

    assert not undocumented, (
        f"Undocumented routes (add to ROUTES_REQUIRED or REQUEST_BODIES "
        f"in tests/app/test_api_contract.py, and document in "
        f"docs/contracts/v1-api.md): {sorted(undocumented)}"
    )
```

- [ ] **Step 4.3: Run the tests**

Run: `pytest tests/app/test_api_contract.py -v`

Expected: all pass. If `test_no_undocumented_routes` fails, there are routes in the app that aren't yet in Contract 2. List them, add them to `ROUTES_REQUIRED` / `REQUEST_BODIES`, and verify the corresponding Pydantic models exist in `app/schemas/v1.py`. If they don't, that's a separate bug to flag in the PR description — but the test still must be made green for E4 to close.

- [ ] **Step 4.4: Commit and open PR**

```bash
git add tests/app/test_api_contract.py
git commit -m "test(wave4): request-shape + undocumented-route guards

Extends test_api_contract.py with two new assertions:
- Every POST/PUT/PATCH route in REQUEST_BODIES declares a request model
  matching app/schemas/v1.py.
- Every non-internal route in the app appears in ROUTES_REQUIRED or
  REQUEST_BODIES — no routes can be added without contract docs.

Closes V4_CONCERNS §2.10 follow-up (request-shape validation).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface E] Contract conformance hardening (E4)" --body "$(cat <<'EOF'
## Summary
- Validates request bodies on POST/PUT/PATCH against Pydantic models.
- Asserts no undocumented routes exist in the FastAPI app.
- Closes V4_CONCERNS §2.10 follow-up.

## Test plan
- [x] new tests added: test_post_put_patch_request_bodies_match_pydantic_models, test_no_undocumented_routes
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 5: E5 — Branch / PR conventions doc

**Files:**
- Create: `docs/contributing.md`
- Modify: `V4_CONCERNS.md`

- [ ] **Step 5.1: Write the contributing doc**

Create `docs/contributing.md`:

```markdown
# Contributing to card-capture

For routine fixes and small improvements, follow your judgment. The
conventions below apply to **planned work** — anything driven by a
spec under `docs/superpowers/specs/`.

## Branch naming

- Wave 4: `wave4/{surface-letter}-{slug}` — e.g. `wave4/a-monolith-deprecation`.
- Future waves: `wave{N}/{surface-letter}-{slug}`.
- Out-of-wave fixes: `fix/{slug}` or `docs/{slug}`.

## PR title

`[Wave N — Surface X] <imperative summary>`

Example: `[Wave 4 — Surface C] FBPredictor refuses without checkpoint`

## PR description template

```
## Summary
- <1-3 bullets describing what changed and why>

Closes V4_CONCERNS §X.Y
Blocked-by: <PR # or "none">
Blocks: <PR # or "none">

## Test plan
- [ ] new test added: <test name>
- [ ] pytest tests/ green locally
- [ ] CI green
- [ ] contract docs updated (if applicable)
- [ ] drift gate green (if applicable)
```

## Closing V4_CONCERNS entries

Every Wave-4 PR closes one or more `V4_CONCERNS.md` entries.

The same branch must move the entry from `§1` (open) to `§2`
(resolved). Add the new resolved entry with the PR number. The entry's
prior body becomes the description of what was fixed.

Example (from a hypothetical PR #99):

```markdown
### 2.13 — *(was §1.X)* Brief title — **Resolved** in PR #99

What the PR did, in one paragraph.
```

## Tests

Every PR adds at least one test that exercises the change. No exceptions.

- Backend: `pytest tests/` must pass locally before push.
- Frontend: Playwright smoke test OR a documented manual smoke flow in
  the PR description.

CI (`.github/workflows/test.yml`) runs `pytest tests/` on every push and
PR. PRs are not merged with red CI.

## Contracts

If your PR adds or changes a route, schema field, or migration column:

1. Update the matching markdown contract under `docs/contracts/`.
2. Add the new shape to the relevant test in
   `tests/contracts/test_drift.py` or `tests/app/test_api_contract.py`.
3. Verify the drift gate passes.
```

- [ ] **Step 5.2: Link from V4_CONCERNS**

At the top of `V4_CONCERNS.md`, append after the existing header:

```markdown
**Contributing conventions:** see `docs/contributing.md`.
```

- [ ] **Step 5.3: Commit and open PR**

```bash
git add docs/contributing.md V4_CONCERNS.md
git commit -m "docs(wave4): branch/PR conventions + V4_CONCERNS link

Closes Surface E task E5 (spec §3).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface E] Branch + PR conventions (E5)" --body "$(cat <<'EOF'
## Summary
- Adds docs/contributing.md documenting branch naming, PR title format, PR body template, V4_CONCERNS-closing convention, test bar, and contract update flow.

## Test plan
- [x] CI green (docs-only change)
EOF
)"
```

Wait for merge.

---

## Task 6: Final verification

- [ ] **Step 6.1: Confirm Surface E is complete**

After all 5 PRs merge:

```bash
git fetch origin main
git log origin/main --oneline | head -10
```

Expected: five Wave-4 Surface-E commits visible in the log.

- [ ] **Step 6.2: Verify the drift gate fires on a deliberate violation**

This is the formal acceptance test from spec §3:

```bash
# In a fresh worktree or branch:
sed -i.bak 's/is_foil: bool/is_foil_renamed: bool/' harness/schema.py
pytest tests/contracts/test_drift.py -v
```

Expected: FAIL.

```bash
mv harness/schema.py.bak harness/schema.py
pytest tests/contracts/test_drift.py -v
```

Expected: PASS. Revert any worktree state.

- [ ] **Step 6.3: Confirm CI is gating PRs**

Open a deliberately-broken throwaway PR (e.g. add `assert False` to a test, push to a branch, open a PR). Expected: CI runs red; merge is blocked. Close the throwaway PR without merging.

If merge is *not* blocked, the manual follow-up from E1 (branch protection) has not yet been applied — flag this to the user as the gating manual step before A/B/C/D start.

- [ ] **Step 6.4: Report completion to user**

Surface E is done. Note any remaining manual user actions:
- Enable branch protection on `main` with `tests` as a required status check (one-time GitHub settings flip).

Surfaces A, B, C, D may now start in parallel.

---

## Self-Review Checklist (run before declaring Surface E done)

- [ ] All 5 PRs merged to `main`.
- [ ] `V4_CONCERNS.md` §1.9 and §1.10 moved to §2 (resolved) with PR numbers.
- [ ] §1.1 has a link to the ADR.
- [ ] CI passes on `main`.
- [ ] Drift gate verified to catch a deliberate violation.
- [ ] `docs/contributing.md` is in place.
