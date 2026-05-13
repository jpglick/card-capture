# Wave 4 — Surface A (Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the monolith-vs-Metaflow decision, consolidate the two competing Options dataclasses into one, and surface migration-runner skips with a log line.

**Architecture:** Single agent, ~3 PRs. Surface A owns pipeline orchestration and core config. Blocked-by Surface E (E1 CI; E2 ADR).

**Tech Stack:** Python 3.11, Metaflow `FlowSpec`, dataclasses, `logging`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-13-v4-wave4-hardening-design.md` §4.

**Files owned by Surface A:** `src/card_capture/pipeline.py`, `src/card_capture/config.py`, `pipeline/**`, `migrations/run_migrations.py`.

---

## Pre-flight

- [ ] **P1: Confirm E1 and E2 are merged**

```bash
git fetch origin main
ls .github/workflows/test.yml
ls docs/decisions/2026-05-13-pipeline-canonical-path.md
```

Expected: both files exist on `main`. If not, stop and wait for Surface E.

Read the ADR fully — its decision dictates A1's direction. **This plan assumes Metaflow won.** If the ADR picked the monolith instead, A1's steps invert (delete `pipeline/` instead of routing entry points through it); the rest of A2/A3 are unchanged.

- [ ] **P2: Create the worktree**

```bash
git worktree add ../card-capture-wave4-a -b wave4/a-pipeline origin/main
cd ../card-capture-wave4-a
pip install -e ".[harness,test]"
python -m pytest tests/ -q
```

Expected: tests pass.

---

## Task 1: A1 — Route every entry point through `CardCaptureFlow`

**Files:**
- Modify: `src/card_capture/cli.py`
- Modify: `app/services/pipeline_runner.py`
- Modify: `harness/cli.py` (if it executes the pipeline directly)
- Modify: `src/card_capture/pipeline.py` (add deprecation warning at module import)
- Create: `tests/pipeline/test_path_equivalence.py`

- [ ] **Step 1.1: Inventory every callsite of the monolith**

```bash
grep -rn "VideoProcessor\|from card_capture.pipeline\|card_capture\.pipeline" \
  src/ app/ harness/ pipeline/ 2>/dev/null
```

Expected output: a list of every place that instantiates or imports the
monolith. Each one needs to be re-routed to call `CardCaptureFlow`
instead (typically via `pipeline.card_capture_flow.CardCaptureFlow`).
Note: `pipeline/steps/*.py` modules are allowed to import from
`src/card_capture/...` (algorithm modules); the **entry-point** code is
what gets re-routed.

- [ ] **Step 1.2: Write a failing equivalence test**

Create `tests/pipeline/test_path_equivalence.py`:

```python
"""Verify that the monolith and Metaflow pipeline produce equivalent
artifacts on a fixture video.

Stays in place until the monolith is deleted (Wave 5). Once that
happens, this test can go too.
"""
from __future__ import annotations

import json
import subprocess
import sys
import sqlite3
from pathlib import Path

import pytest

FIXTURE_VIDEO = Path("tests/fixtures/fake_video.mov")


def _read_cards(db_path: Path) -> list[tuple]:
    """Return a sorted list of (instance_id, side, primary_hash)."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT instance_id, side, primary_hash "
            "FROM card_instances ORDER BY instance_id"
        ).fetchall()
    return rows


@pytest.mark.skipif(
    not FIXTURE_VIDEO.exists(),
    reason="fixture video not present; equivalence test is opt-in",
)
def test_monolith_and_metaflow_produce_same_cards(tmp_path):
    """Run both pipeline paths on the same fixture and assert the
    extracted card set matches.
    """
    monolith_out = tmp_path / "monolith"
    metaflow_out = tmp_path / "metaflow"

    # Monolith path — DEPRECATED; remove this test when the monolith goes.
    subprocess.run(
        [
            sys.executable, "-m", "card_capture.cli", "process",
            str(FIXTURE_VIDEO),
            "--output-dir", str(monolith_out),
            "--db", str(monolith_out / "cards.sqlite"),
            "--detector", "fake",
            "--pipeline", "monolith",
        ],
        check=True,
    )

    # Metaflow path — canonical.
    subprocess.run(
        [
            sys.executable, "-m", "card_capture.cli", "process",
            str(FIXTURE_VIDEO),
            "--output-dir", str(metaflow_out),
            "--db", str(metaflow_out / "cards.sqlite"),
            "--detector", "fake",
            "--pipeline", "metaflow",
        ],
        check=True,
    )

    mono_cards = _read_cards(monolith_out / "cards.sqlite")
    meta_cards = _read_cards(metaflow_out / "cards.sqlite")

    # Card set must match. Order is enforced by ORDER BY instance_id.
    # If instance IDs are UUIDs that differ between runs, compare on
    # primary_hash and side only.
    mono_set = {(side, ph) for _, side, ph in mono_cards}
    meta_set = {(side, ph) for _, side, ph in meta_cards}

    assert mono_set == meta_set, (
        f"Pipeline paths disagree.\n"
        f"  monolith-only: {mono_set - meta_set}\n"
        f"  metaflow-only: {meta_set - mono_set}"
    )
```

- [ ] **Step 1.3: Add `--pipeline` flag to the CLI**

Edit `src/card_capture/cli.py`. Find the `process` command. Add a flag:

```python
@click.option(
    "--pipeline",
    type=click.Choice(["metaflow", "monolith"]),
    default="metaflow",
    help="Pipeline orchestration backend. Monolith is deprecated.",
)
```

Update the command body so:

```python
if pipeline == "monolith":
    import warnings
    warnings.warn(
        "The monolithic VideoProcessor is deprecated and will be removed "
        "in Wave 5. Use --pipeline metaflow (the default).",
        DeprecationWarning,
        stacklevel=2,
    )
    from card_capture.pipeline import VideoProcessor
    VideoProcessor(...).run()
else:
    from pipeline.card_capture_flow import CardCaptureFlow
    # Invoke via the Metaflow CLI machinery. See existing
    # app/services/pipeline_runner.py for the pattern.
    CardCaptureFlow(...)
```

Match the actual constructor signatures by reading both files first.
If `pipeline_runner.py` already has a tested invocation pattern, copy it
verbatim rather than reinventing.

- [ ] **Step 1.4: Add module-level DeprecationWarning to the monolith**

Edit `src/card_capture/pipeline.py`. At module top (after imports):

```python
import warnings

warnings.warn(
    "card_capture.pipeline.VideoProcessor is deprecated; use "
    "pipeline.card_capture_flow.CardCaptureFlow instead. The monolith "
    "will be removed in Wave 5. See "
    "docs/decisions/2026-05-13-pipeline-canonical-path.md.",
    DeprecationWarning,
    stacklevel=2,
)
```

- [ ] **Step 1.5: Re-route `app/services/pipeline_runner.py` and `harness/cli.py`**

If `pipeline_runner.py` already routes through Metaflow, no change.
Otherwise, change it to do so. Same for `harness/cli.py`'s
pipeline-execution path (if it has one — search for `VideoProcessor`).

- [ ] **Step 1.6: Run the equivalence test**

```bash
pytest tests/pipeline/test_path_equivalence.py -v -W error::DeprecationWarning
```

Expected: PASS, plus the DeprecationWarning from the monolith branch
shows up in pytest's warning summary (which is exactly what we want;
`-W error` would actually fail the monolith call, so DON'T use
`-W error` in the actual CI. The line above is for local diagnostic
only; remove `-W error::DeprecationWarning` before commit if it makes
the test fail).

If `tests/fixtures/fake_video.mov` does not exist, the test is
skipped — fine for CI, but locally you should create or symlink a real
fixture and run the test once.

- [ ] **Step 1.7: Run the full test suite**

```bash
python -m pytest tests/ -q
```

Expected: pass. The DeprecationWarning will appear in warnings summary
but won't fail any test.

- [ ] **Step 1.8: Commit and open PR**

```bash
git add src/card_capture/cli.py src/card_capture/pipeline.py \
        app/services/pipeline_runner.py harness/cli.py \
        tests/pipeline/test_path_equivalence.py
git commit -m "feat(pipeline): route entry points through CardCaptureFlow

- card-capture process defaults to --pipeline metaflow; --pipeline
  monolith emits a DeprecationWarning.
- VideoProcessor import emits module-level DeprecationWarning.
- New test asserts both paths produce the same card set on a fixture
  video.

Closes V4_CONCERNS §1.1 (execution; ADR is E2).
ADR: docs/decisions/2026-05-13-pipeline-canonical-path.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push -u origin wave4/a-pipeline
gh pr create --title "[Wave 4 — Surface A] Route entry points through CardCaptureFlow (A1)" --body "$(cat <<'EOF'
## Summary
- Routes the `card-capture process` CLI, `app/services/pipeline_runner.py`, and `harness/cli.py` through `CardCaptureFlow` by default.
- Monolith path emits DeprecationWarning; removal is a Wave 5 task.
- Adds equivalence smoke test.

Closes V4_CONCERNS §1.1 (execution).
Blocked-by: E2 (#<N>).

## Test plan
- [x] new test added: tests/pipeline/test_path_equivalence.py
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 2: A2 — Two-Options consolidation

**Files:**
- Modify: `src/card_capture/pipeline.py` (delete the local `Options` dataclass)
- Modify: every caller that instantiated `pipeline.Options`
- Create: `tests/test_config.py`

- [ ] **Step 2.1: Update from main**

```bash
git fetch origin main && git rebase origin/main
```

- [ ] **Step 2.2: Inventory both dataclasses**

Read both:

```bash
grep -n "^@dataclass\|class Options\|class Config" \
  src/card_capture/config.py src/card_capture/pipeline.py
```

Make a side-by-side field list. `config.py:Options` is canonical (per spec). Any field on `pipeline.py:Options` that is NOT on `config.py:Options` must be either (a) added to `config.py:Options`, or (b) deleted as dead code. Decide one outcome per field. Common case will be that they're identical and the pipeline.py copy is just stale duplication.

- [ ] **Step 2.3: Write a failing test**

Create `tests/test_config.py`:

```python
"""Assert exactly one canonical config dataclass exists.

Closes V4_CONCERNS §4.16.
"""
from __future__ import annotations

import inspect

import card_capture.config as config_mod
import card_capture.pipeline as pipeline_mod


def test_pipeline_does_not_define_options_dataclass():
    """src/card_capture/pipeline.py must not define its own Options.
    The canonical dataclass lives in src/card_capture/config.py.
    """
    members = inspect.getmembers(pipeline_mod, inspect.isclass)
    pipeline_dataclasses = [
        name for name, cls in members
        if name == "Options" and cls.__module__ == pipeline_mod.__name__
    ]
    assert not pipeline_dataclasses, (
        f"pipeline.py still defines Options dataclass. "
        f"Move it to card_capture.config and update all callers."
    )


def test_config_options_is_importable():
    """The canonical Options is importable from card_capture.config."""
    assert hasattr(config_mod, "Options"), (
        "card_capture.config must export Options dataclass."
    )
```

- [ ] **Step 2.4: Run the test — expect FAIL**

```bash
pytest tests/test_config.py -v
```

Expected: `test_pipeline_does_not_define_options_dataclass` fails (the
dataclass is still there).

- [ ] **Step 2.5: Migrate fields and delete the duplicate**

Edit `src/card_capture/config.py:Options` — ensure every field used
anywhere in the codebase is present. If a field exists only in
`pipeline.py:Options` and is referenced elsewhere, add it to `config.py`.

Edit `src/card_capture/pipeline.py`:
1. Delete the local `@dataclass class Options:` block.
2. Replace it with `from card_capture.config import Options`.
3. Verify no `import` cycle results (run `python -c "from card_capture import pipeline"`).

Update every caller:

```bash
grep -rn "pipeline.Options\|pipeline import Options" src/ app/ harness/ pipeline/ tests/
```

For each hit, change the import to `from card_capture.config import Options` (or use the already-imported config namespace).

- [ ] **Step 2.6: Run the test — expect PASS**

```bash
pytest tests/test_config.py -v
python -m pytest tests/ -q
```

Expected: both green.

- [ ] **Step 2.7: Commit and open PR**

```bash
git add src/card_capture/config.py src/card_capture/pipeline.py \
        tests/test_config.py $(grep -rl 'pipeline.Options\|pipeline import Options' src/ app/ harness/ pipeline/ 2>/dev/null)
git commit -m "refactor(config): single canonical Options dataclass

Deletes the duplicate Options dataclass in
src/card_capture/pipeline.py and re-exports from
src/card_capture/config.py instead. All callers updated.

Closes V4_CONCERNS §4.16.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface A] Two-Options consolidation (A2)" --body "$(cat <<'EOF'
## Summary
- Deletes the duplicate Options dataclass in pipeline.py.
- Re-exports from card_capture.config (the canonical location).
- New test asserts only one Options dataclass exists.

Closes V4_CONCERNS §4.16.
Blocks: D1 (harness loads this dataclass).

## Test plan
- [x] new test added: tests/test_config.py
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 3: A3 — Migration-runner log on skip

**Files:**
- Modify: `migrations/run_migrations.py`
- Modify: `tests/migrations/test_schema.py` (or create `test_logging.py`)

- [ ] **Step 3.1: Update from main**

```bash
git fetch origin main && git rebase origin/main
```

- [ ] **Step 3.2: Write a failing test**

Append to `tests/migrations/test_schema.py`:

```python
import logging
import sqlite3
from pathlib import Path

import pytest

from migrations.run_migrations import apply_migrations


def test_migration_logs_warning_on_skip(tmp_path: Path, caplog):
    """When pipeline_events doesn't exist, the runner skips the ALTER
    and logs a warning. The migration is NOT marked applied so the
    next boot retries.
    """
    db_path = tmp_path / "cards.sqlite"
    sqlite3.connect(db_path).close()  # empty DB; no pipeline_events table

    with caplog.at_level(logging.WARNING, logger="migrations"):
        apply_migrations(db_path)

    skip_logs = [r for r in caplog.records if "skipping" in r.message.lower()]
    assert skip_logs, (
        "Expected a 'skipping' log line when ALTER on missing table is "
        "swallowed; got nothing."
    )
    # The log should mention pipeline_events specifically.
    assert any("pipeline_events" in r.message for r in skip_logs), (
        f"Skip log should mention pipeline_events. "
        f"Got: {[r.message for r in skip_logs]}"
    )
```

- [ ] **Step 3.3: Run the test — expect FAIL**

```bash
pytest tests/migrations/test_schema.py::test_migration_logs_warning_on_skip -v
```

Expected: FAIL (no log records captured).

- [ ] **Step 3.4: Add logging to the runner**

Edit `migrations/run_migrations.py`. At top of file:

```python
import logging

log = logging.getLogger("migrations")
```

In the `apply_migrations` function, inside the `if "no such table" in msg` branch, add the log line:

```python
if "no such table" in msg:
    log.warning(
        "skipping migration statement (no such table): %s — %s",
        statement[:80],
        msg,
    )
    all_ok = False
    continue
```

- [ ] **Step 3.5: Run the test — expect PASS**

```bash
pytest tests/migrations/test_schema.py::test_migration_logs_warning_on_skip -v
python -m pytest tests/ -q
```

Expected: both green.

- [ ] **Step 3.6: Commit and open PR**

```bash
git add migrations/run_migrations.py tests/migrations/test_schema.py
git commit -m "fix(migrations): log warning when skipping ALTER on missing table

apply_migrations() already correctly does not mark a migration applied
when 'no such table' forces a skip (§2.6). This commit adds a warning
log so the skip is visible — otherwise the partial migration looks like
silent success.

Closes V4_CONCERNS §1.8.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface A] Migration runner skip logging (A3)" --body "$(cat <<'EOF'
## Summary
- Adds a logging.warning() call in the 'no such table' skip branch of apply_migrations().
- New test asserts the warning fires on a fresh DB without pipeline_events.

Closes V4_CONCERNS §1.8.
Blocks: B3 (first new migration after this lands).

## Test plan
- [x] new test added: test_migration_logs_warning_on_skip
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 4: Update V4_CONCERNS.md and final verification

- [ ] **Step 4.1: Move §1.1, §1.8, §4.16 to §2 (Resolved)**

Edit `V4_CONCERNS.md`:

- §1.1 body → `**Resolved (see §2.13)**`. Add §2.13:
  ```markdown
  ### 2.13 — *(was §1.1)* Two parallel pipelines drifting asymmetrically — **Resolved** in PR #<N> (A1) and PR #<M> (E2)

  Decision: Metaflow is canonical (see ADR
  `docs/decisions/2026-05-13-pipeline-canonical-path.md`). Every entry
  point now routes through `CardCaptureFlow`. The monolith carries a
  DeprecationWarning; deletion is a Wave 5 task. Equivalence smoke
  test in `tests/pipeline/test_path_equivalence.py` guards the
  transition.
  ```
- §1.8 body → `**Resolved (see §2.14)**`. Add §2.14:
  ```markdown
  ### 2.14 — *(was §1.8)* Migration-runner silent skip — **Resolved** in PR #<N> (A3)

  `apply_migrations` now logs a warning when it skips a statement due
  to a missing table.
  ```
- §4.16 body → `**Resolved (see §2.15)**`. Add §2.15:
  ```markdown
  ### 2.15 — *(was §4.16)* Two competing Options dataclasses — **Resolved** in PR #<N> (A2)

  Deleted the duplicate in `src/card_capture/pipeline.py`. The
  canonical dataclass lives in `src/card_capture/config.py`; all
  callers updated. New `tests/test_config.py` asserts uniqueness.
  ```

```bash
git add V4_CONCERNS.md
git commit -m "docs(wave4): mark §1.1, §1.8, §4.16 resolved by Surface A

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

Either fold this into the last open PR or create a tiny standalone PR — agent's choice.

- [ ] **Step 4.2: Verify all Surface A work is on `main`**

```bash
git fetch origin main
git log origin/main --oneline | head -10
```

Expected: A1, A2, A3 commits visible.

- [ ] **Step 4.3: Sanity-check the equivalence test still runs**

```bash
pytest tests/pipeline/test_path_equivalence.py -v
```

Expected: PASS (or `skip` if the fixture video isn't present in the
CI environment — that's expected).

- [ ] **Step 4.4: Report completion**

Surface A is done. Surface D's D1 (harness config loading) was blocked
on A2 (single canonical Options) and is now unblocked.

---

## Self-Review Checklist

- [ ] A1, A2, A3 merged.
- [ ] `V4_CONCERNS.md` §1.1, §1.8, §4.16 moved to §2.
- [ ] CI green on `main`.
- [ ] Equivalence test in place and runs.
- [ ] DeprecationWarning fires on monolith import.
- [ ] Only one `Options` dataclass in the codebase.
- [ ] Migration runner logs on skip.
