# V5.5 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap identified in the 2026-05-28 verification of `docs/superpowers/plans/2026-05-28-v5-5-refactoring.md`: fix the test-collection regression, turn the static-enforcement plumbing from advisory into actually-running blocking checks, finish the Phase 4 data-access migration so zero raw SQLite calls remain outside `card_capture.data`, build the missing Phase 5 platform adapters (Local, Beam, RunPod-via-PipelineRunner) and deprecate Vast.ai, then flip the raw-SQL scanner to blocking.

**Architecture:** Five sequential sub-phases (A–E). Each ships working software on its own.

- **Phase A** stops the bleeding: rewrite the stale `tests/test_unified_runtime.py` so the default lane collects again, and confirm green.
- **Phase B** makes the dormant static checks executable: fix `.importlinter` config, install the linter by default, document and wire architecture lane CI commands.
- **Phase C** finishes Phase 4: align repositories with the production schema, then migrate the ~30 remaining raw `sqlite3` callsites to `card_capture.data.connection.read_connection` or repository methods.
- **Phase D** finishes Phase 5: reconcile `PipelineRunner` vs `RemoteRuntime`, build `LocalRunner`/`BeamRunner`, rewrite `RunpodRuntime` to the unified surface, deprecate Vast.ai end-to-end, add `manifests.py` and `failures.py`, populate `tests/platforms/`.
- **Phase E** flips raw-SQL scanner to blocking, runs every static check in CI, tags the release.

Each phase follows TDD: write the failing test → run it → implement → run it → commit.

**Tech Stack:** Python 3.9+, SQLite (WAL), Import Linter ≥2.0, pytest, OpenTelemetry Metrics SDK, RunPod/Beam SDKs.

**Spec:** `docs/superpowers/specs/2026-05-24-v5-5-refactoring-design.md` remains the source of truth. When this plan and the spec disagree, the spec wins; raise the conflict before proceeding. The original Phase plan at `docs/superpowers/plans/2026-05-28-v5-5-refactoring.md` defines the artifacts; this plan only fills gaps.

---

## File Structure

Files created, modified, or deleted across the plan:

```text
.importlinter                                              Phase B; add include_external_packages=True; drop vastai (Phase D)
.github/workflows/ci.yml                                   Phase B; add architecture lane + dev-deps install

pyproject.toml                                             Phase B/D; move import-linter to main deps; drop vastai dep

migrations/
    0013_v55_repository_schema.sql                         Phase C; add v5.5-shaped tables (card_views v55, fb_labels, telemetry_events, truth_files)

src/card_capture/
    pipeline/
        runner.py                                          Phase D; PipelineRunner protocol clarified
        runtime.py                                         Phase D; sole concrete RemoteRuntime callers updated
        remote.py                                          Phase D; DELETED (collapsed into runner.py)
    platforms/
        __init__.py                                        Phase D; exports new runners
        failures.py                                        Phase D; stable failure categories + mapping helper
        manifests.py                                       Phase D; manifest import/export helper
        local.py                                           Phase D; LocalRunner wrapping LocalPipelineRuntime
        runpod.py                                          Phase D; rewritten to implement PipelineRunner
        beam.py                                            Phase D; BeamRunner
    data/
        repositories/
            cards.py                                       Phase C; align with prod card_instances / card_views v55 schema
            runs.py                                        Phase C; align with prod pipeline_runs schema
            events.py                                      Phase C; align with prod pipeline_events schema
            videos.py                                      Phase C; align with prod videos schema
            labeling.py                                    Phase C; align with prod fb_labels / truth_files
            telemetry.py                                   Phase C; create telemetry_events table or alias to pipeline_events
            config.py                                      Phase C; NEW — config_presets table accessor
            batch.py                                       Phase C; NEW — batch jobs read/write
            training.py                                    Phase C; NEW — training_samples + fb_labels reads
            ml.py                                          Phase C; NEW — model_registry reads/writes

    cli.py                                                 Phase C; replace `import sqlite3` with read_connection
    timeline_data.py                                       Phase C; replace `import sqlite3` with read_connection
    storage.py                                             Phase C; thin wrapper over repositories OR removed
    ml/registry.py                                         Phase C; replace `import sqlite3` with read_connection + ml repo
    ml/train_fb.py                                         Phase C; replace `import sqlite3` with training repo
    ml/training/dedup_calibrate.py                         Phase C; replace `import sqlite3` with read_connection
    training/presence_trainer.py                           Phase C; replace `import sqlite3` with training repo
    training/fb_trainer.py                                 Phase C; replace `import sqlite3` with training repo

app/
    main.py                                                Phase C; remove sqlite3.connect bootstrap; use Writer
    worker_core.py                                         Phase C; replace `import sqlite3` with repositories
    runpod_handler.py                                      Phase C; replace `import sqlite3` with repositories
    api/config.py                                          Phase C; replace `import sqlite3` with config repository
    api/batch.py                                           Phase C; replace `import sqlite3` with batch repository
    services/vast_runner.py                                Phase D; DELETED
    services/vast_client.py                                Phase D; DELETED
    services/training_service.py                           Phase C; route writes via repositories (~11 sites)
    services/result_importer.py                            Phase C; route writes via repositories
    services/resource_sampler.py                           Phase C; route writes via telemetry repo
    services/cards_service.py                              Phase C; (already migrated per `cf3602a4`) — verify
    services/video_service.py                              Phase C; (already migrated) — verify

harness/
    baseline.py                                            Phase C; replace `import sqlite3` with read_connection
    cli.py                                                 Phase C; replace `import sqlite3` with read_connection
    hard_cases.py                                          Phase C; replace `import sqlite3` with read_connection
    match.py                                               Phase C; replace `import sqlite3` with read_connection
    metrics/image_quality.py                               Phase C; replace `import sqlite3` with read_connection
    metrics/dedup_accuracy.py                              Phase C; replace `import sqlite3` with read_connection

tests/
    test_unified_runtime.py                                Phase A; rewritten to LocalPipelineRuntime
    architecture/test_raw_sql_outside_data.py              Phase E; remove env-var gate (always blocking)
    architecture/test_import_linter.py                     Phase B; always-blocking by default
    data/
        test_cards_repository.py                           Phase C; updated to use production schema
        test_runs_repository.py                            Phase C; updated to use production schema
        test_events_repository.py                          Phase C; updated to use production schema
        test_videos_repository.py                          Phase C; updated to use production schema
        test_labeling_repository.py                        Phase C; updated to use production schema
        test_telemetry_repository.py                       Phase C; updated to use production schema
        test_config_repository.py                          Phase C; NEW
        test_batch_repository.py                           Phase C; NEW
        test_training_repository.py                        Phase C; NEW
        test_ml_repository.py                              Phase C; NEW
    platforms/
        __init__.py                                        Phase D; NEW
        test_local_runner.py                               Phase D; NEW
        test_runpod_runner.py                              Phase D; NEW
        test_beam_runner.py                                Phase D; NEW
        test_failures.py                                   Phase D; NEW
        test_manifests.py                                  Phase D; NEW

    app/test_vast_client.py                                Phase D; DELETED
    app/test_vast_runner.py                                Phase D; DELETED
    app/test_vastai_worker.py                              Phase D; DELETED

docs/
    superpowers/plans/v5-5/                                Phase B; ci-lane-commands.md updated
```

---

# Phase A: Regression Recovery

**Goal:** Make `pytest tests/` collect again so the default lane runs in CI.

**Acceptance:** `pytest tests/ -m 'not quarantine' -q` exits 0 with no collection errors.

---

### Task A.1: Rewrite the stale UnifiedRuntime smoke test against LocalPipelineRuntime

**Files:**
- Modify: `tests/test_unified_runtime.py`

The current file imports `card_capture.runtime.UnifiedRuntime`, which no longer exists — `card_capture.runtime` is now a package containing `gpu_session.py`, `batches.py`, etc. The V5.5 successor is `card_capture.pipeline.runtime_local.LocalPipelineRuntime`. Rewrite the smoke test against the new contract; the test must use the JSON-shaped `PipelineRunRequest`, not the old `video_path`/`db_path` shape.

- [ ] **Step 1: Confirm the regression**

Run:
```bash
python3 -m pytest tests/test_unified_runtime.py --collect-only 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'UnifiedRuntime' from 'card_capture.runtime'`.

- [ ] **Step 2: Replace the file in full**

Write `tests/test_unified_runtime.py`:

```python
"""Smoke test for the V5.5 in-process pipeline runtime.

Historically this file targeted `UnifiedRuntime`. After the V5.5 refactor
the same role is filled by `LocalPipelineRuntime` (see
`src/card_capture/pipeline/runtime_local.py`). This test exercises a
synthetic-fixture run end-to-end against the new contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.data.connection import open_connection
from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "golden_corpus"
    / "IMG_5872"
    / "IMG_5872.MOV"
)


@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="Golden-set video IMG_5872.MOV not present (large binary, not in repo)",
)
def test_local_runtime_runs_to_completion(tmp_path: Path) -> None:
    db_path = tmp_path / "cards.sqlite"
    # Synthetic schema sufficient for the smoke contract; production migrations
    # are exercised in tests/data/test_*_repository.py.
    conn = open_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_runs(
            run_id TEXT PRIMARY KEY,
            video_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            cards_extracted INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS card_instances(
            id INTEGER PRIMARY KEY,
            instance_id TEXT UNIQUE,
            video_id INTEGER NOT NULL,
            run_id TEXT,
            track_id TEXT NOT NULL,
            fused_image_path TEXT
        );
    """)
    conn.commit()
    conn.close()

    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    request = PipelineRunRequest(
        run_id="smoke-unified",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
        config={"db_path": str(db_path)},
    )

    result = runtime.run(request)

    assert result.manifest.runtime_mode == "cpu_debug"
    assert result.manifest.input_video == request.input_video

    # Stage facades must have fired.
    finished_stages = {
        e.payload["stage"] for e in telemetry.events if e.kind == "stage_finished"
    }
    expected = {
        "sample", "detect", "novelty", "track", "refine",
        "score", "resolve", "fuse", "dedup", "store",
    }
    missing = expected - finished_stages
    assert not missing, f"missing stage_finished events: {sorted(missing)}"
```

- [ ] **Step 3: Run the rewritten test**

Run:
```bash
python3 -m pytest tests/test_unified_runtime.py -v
```

Expected: PASS (it will skip if the fixture isn't present, which is fine for this task — the goal is that pytest can collect the file). If the fixture exists locally and the test fails for a real algorithmic reason, surface it and stop; do not paper over a real regression.

- [ ] **Step 4: Confirm the full default lane collects and passes**

Run:
```bash
python3 -m pytest tests/ -m 'not quarantine' -q --tb=short
```

Expected: zero collection errors; `passed` count >= 581 (the verified count from the prior run was 581 once this file was ignored); `skipped` count may grow by 1 if the fixture isn't present.

- [ ] **Step 5: Commit**

```bash
git add tests/test_unified_runtime.py
git commit -m "test(v55-phaseA): rewrite stale UnifiedRuntime smoke to LocalPipelineRuntime"
```

---

# Phase B: Activate Static Enforcement

**Goal:** Make every architecture/import check actually executable from a clean checkout and run on the CI lane. Today: `import-linter` is in `dev` extras (not in the default install), the `.importlinter` config rejects external forbidden modules, `tests/architecture/test_import_linter.py` is skipif-gated behind `V55_IMPORT_LINT_BLOCKING`, and `.github/workflows/ci.yml` does not exercise any of this.

**Acceptance:** `lint-imports` exits 0 from a fresh `pip install -e ".[dev]"`. `tests/architecture/test_import_linter.py::test_import_contracts` runs and passes by default (no env-var gate). The architecture lane is part of `ci.yml`.

---

### Task B.1: Add `include_external_packages = True` to `.importlinter`

**Files:**
- Modify: `/Users/josh/code/card-capture/.importlinter`

The config forbids external modules (`sqlite3`, `metaflow`, `runpod`, `beam`, `vastai`, `PIL`). Import Linter requires `include_external_packages = True` at the top level to scan for them.

- [ ] **Step 1: Confirm the config error**

Run:
```bash
PYTHONPATH=src:. lint-imports
```

(Or `/Users/josh/Library/Python/3.9/bin/lint-imports` if the user-site bin isn't on PATH.)

Expected output contains:
```
The top level configuration must have include_external_packages=True when there are external forbidden modules.
```

- [ ] **Step 2: Edit `.importlinter` top block**

Replace the `[importlinter]` block at the top of `.importlinter`:

```ini
[importlinter]
root_packages =
    card_capture
    app
    pipeline
include_external_packages = True
```

Leave every contract section unchanged.

- [ ] **Step 3: Re-run the linter**

Run:
```bash
PYTHONPATH=src:. lint-imports
```

Expected: the linter actually runs and reports per-contract pass/fail. There WILL be violations now (raw `sqlite3` callers, etc.); that is intended and Phase C closes them. Phase B's win condition for this task is "the tool runs and produces actionable output," not "everything is green."

- [ ] **Step 4: Commit**

```bash
git add .importlinter
git commit -m "fix(v55-phaseB): enable include_external_packages so Import Linter runs"
```

---

### Task B.2: Move `import-linter` to the default dev install + document

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/superpowers/plans/v5-5/ci-lane-commands.md`

`import-linter>=2.0` is already in `[project.optional-dependencies] dev`. The remaining issue is that the architecture test skips silently when the binary isn't on PATH, masking a missing install. Two changes: keep the package in `dev` but make the test fail loudly if the binary is missing, and document the install command in the CI lane doc.

- [ ] **Step 1: Update `tests/architecture/test_import_linter.py` to always-run**

Replace the file in full:

```python
"""Import Linter contracts run on every default pytest invocation.

The PR lane installs `[dev]` extras so `lint-imports` is available. If the
binary is missing the test fails loudly with installation instructions
rather than skipping silently.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest


_INSTALL_HINT = (
    "lint-imports binary not found on PATH. Install dev extras:\n"
    "    python3 -m pip install -e '.[dev]'\n"
    "If your user-site bin is not on PATH, add it:\n"
    "    export PATH=\"$(python3 -m site --user-base)/bin:$PATH\""
)


def test_import_contracts() -> None:
    if shutil.which("lint-imports") is None:
        pytest.fail(_INSTALL_HINT)
    result = subprocess.run(
        ["lint-imports"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.fail(
            f"Import Linter contract violations (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
```

- [ ] **Step 2: Update the CI lane doc**

Edit `docs/superpowers/plans/v5-5/ci-lane-commands.md` (create the file if missing) and replace its contents with:

```markdown
# V5.5 CI Lane Commands

## One-time setup

```bash
python3 -m pip install -e ".[dev]"
# If lint-imports is not on PATH, add user-site bin:
export PATH="$(python3 -m site --user-base)/bin:$PATH"
```

## Fast PR lane (default; no GPU, no credentials, no real videos)

```bash
python3 -m pytest tests/ -q
python3 -m pytest tests/architecture/ -q
python3 -m pytest tests/performance/test_perf_harness_smoke.py -q
PYTHONPATH=src:. lint-imports
```

The `addopts` line in `pyproject.toml` already deselects `quarantine` and `benchmark` markers.

## Optional hardware lane (CUDA)

```bash
python3 -m pytest tests/ -q -m cuda
```

## Optional hardware lane (MPS)

```bash
python3 -m pytest tests/ -q -m mps
```
```

- [ ] **Step 3: Run the test from a fresh shell (no env vars set)**

Run:
```bash
unset V55_IMPORT_LINT_BLOCKING
python3 -m pytest tests/architecture/test_import_linter.py -v
```

Expected: the test runs (no skip). It may FAIL because of unrelated contract violations Phase C will close; that is fine for this task — record the failures in the commit message body so a reader can see Phase C closes them.

If the test fails for *config* reasons (e.g., the linter cannot find `card_capture`), debug `pyproject.toml`'s `[tool.setuptools.packages.find]` or `PYTHONPATH` before continuing; configuration must be working before Phase C.

- [ ] **Step 4: Commit**

```bash
git add tests/architecture/test_import_linter.py docs/superpowers/plans/v5-5/ci-lane-commands.md
git commit -m "feat(v55-phaseB): Import Linter test runs by default; install hint on missing binary

Phase C is expected to close the remaining contract violations; this
commit only makes the violations visible."
```

---

### Task B.3: Add the architecture lane to `.github/workflows/ci.yml`

**Files:**
- Modify: `.github/workflows/ci.yml`

The existing workflow installs `pip install .` (no extras), runs `pytest tests/`, and exits. It must install `[dev]` (for `lint-imports`) and run the architecture + perf-smoke lanes.

- [ ] **Step 1: Replace `.github/workflows/ci.yml` with**

```yaml
name: CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - name: Install system dependencies
      run: sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6
    - name: Install dependencies (with dev extras for Import Linter)
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install pytest pytest-asyncio httpx pydantic anyio
    - name: Validate schema docs drift
      run: |
        python scripts/validate_schema_docs.py
    - name: Default test suite
      run: |
        pytest tests/ -q
    - name: Architecture lane (Import Linter + AST scanners)
      run: |
        pytest tests/architecture/ -q
    - name: Performance harness smoke
      run: |
        pytest tests/performance/test_perf_harness_smoke.py -q
    - name: Import Linter (blocking)
      env:
        PYTHONPATH: src:.
      run: |
        lint-imports
```

- [ ] **Step 2: Sanity-check syntax locally with `yamllint` if available**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "yaml ok"
```

Expected: `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci(v55-phaseB): run architecture lane and Import Linter on PR"
```

**Phase B complete.** The static checks now run by default; remaining contract violations are the work of Phase C.

---

# Phase C: Finish the Data-Access Layer Migration

**Goal:** Eliminate every `import sqlite3` / `sqlite3.connect` callsite outside `src/card_capture/data/`, `migrations/`, `tests/`, and `harness/schema.py`. Bring repositories in line with the production schema (currently they assume a different one — verified by reading `migrations/0001_v4_schema.sql` against `src/card_capture/data/repositories/cards.py`).

**Acceptance:**
1. `grep -rn 'import sqlite3\|sqlite3.connect' --include='*.py' src/ app/ pipeline/ harness/` returns ONLY files inside `src/card_capture/data/`, `harness/schema.py`, or tests.
2. `tests/architecture/test_raw_sql_outside_data.py::test_no_raw_sql_outside_data_blocking` passes without needing the `V55_RAW_SQL_BLOCKING` env var (Phase E flips that gate; this phase just stops adding violations).
3. `lint-imports` exits 0.
4. The repositories work against the production database schema (i.e., the schema produced by running every file in `migrations/` against an empty database).

**Pre-flight note:** The repositories at `src/card_capture/data/repositories/*.py` were written against an idealized v5.5 schema that does NOT match what `migrations/0001_v4_schema.sql` etc. produce. Concretely:

| Repo file | Repo expects | Production schema (migrations/) |
|---|---|---|
| `cards.py` | `card_instances(card_instance_id TEXT PK, run_id, front_crop, back_crop)` + `card_views(card_instance_id, metric, value)` | `card_instances(id INTEGER PK AUTOINCREMENT, instance_id TEXT UNIQUE, video_id, run_id, track_id, session_id, …)` + `card_views(id INTEGER PK, card_instance_id INTEGER FK, frame_index, timestamp_ms, image_path, …)` |
| `runs.py` | `pipeline_runs(run_id, video_id, state, started_at_ms, completed_at_ms, cards_extracted, error)` | `pipeline_runs(run_id TEXT PK, video_id INTEGER FK, status, cards_extracted, started_at TEXT, finished_at TEXT)` |
| `events.py` | `pipeline_events(run_id, video_id, stage, elapsed_ms, metadata)` | `pipeline_events(id INTEGER PK, video_id INTEGER FK, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json, artifact_ref, created_at)` |
| `videos.py` | `videos(video_id TEXT, path, registered_at_ms, metadata)` | `videos(id INTEGER PK AUTOINCREMENT, source_path, file_hash, duration_ms, width, height, status, created_at)` |
| `telemetry.py` | `telemetry_events(run_id, kind, payload, at_ms)` | TABLE DOES NOT EXIST in any migration |
| `labeling.py` | `fb_labels(source_run_id, instance_id, frame_index, side, labeler)` + `truth_files(video_id, schema_version, payload_json)` | Verify against `migrations/0005_training_samples.sql` / `0006_fb_labels_no_card.sql` |

The repositories pass their unit tests only because each test synthesizes a toy schema inline (see `tests/data/test_cards_repository.py::_init_schema`).

Approach: **rewrite the repositories to use the production schema**, then update each repository's tests to use `migrations/` to set up the schema. This makes Phase 4 actually testable end-to-end.

---

### Task C.1: Add a `RepositoryFixture` helper that applies all migrations into a tmp SQLite

**Files:**
- Create: `tests/data/conftest.py`

This fixture is shared by every Phase C repository test below; defining it once avoids re-inlining `_init_schema` in each file.

- [ ] **Step 1: Write the fixture file**

```python
"""Shared fixtures for repository tests.

`prod_db` returns a Path to a freshly-created SQLite database with every
production migration applied. Repository tests use this instead of inlining
their own toy schemas.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.data.connection import open_connection


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "migrations"


def _apply_migrations(db_path: Path) -> None:
    conn = open_connection(db_path)
    try:
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            conn.executescript(sql_file.read_text())
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def prod_db(tmp_path: Path) -> Path:
    """A SQLite database with every production migration applied."""
    db = tmp_path / "cards.sqlite"
    _apply_migrations(db)
    return db
```

- [ ] **Step 2: Smoke-test the fixture**

Run:
```bash
python3 -c "
import tempfile, pathlib, sys
sys.path.insert(0, 'tests/data')
import conftest
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / 'x.db'
    conftest._apply_migrations(p)
    from card_capture.data.connection import read_connection
    with read_connection(p) as c:
        names = [r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\")]
    print(names)
"
```

Expected: a list including `videos`, `pipeline_runs`, `pipeline_events`, `card_instances`, `card_views`. If any migration errors out, fix it BEFORE moving on; the migrations must apply cleanly to a fresh database (this was Phase 0 acceptance).

- [ ] **Step 3: Commit**

```bash
git add tests/data/conftest.py
git commit -m "test(v55-phaseC): shared prod_db fixture applies all migrations"
```

---

### Task C.2: Align `videos` repository with the production schema

**Files:**
- Modify: `src/card_capture/data/repositories/videos.py`
- Modify: `tests/data/test_videos_repository.py`

The production `videos` table uses an integer `id` primary key and has `source_path`, `file_hash`, `duration_ms`, `width`, `height`, `status`. The repository must use those columns. Callers will pass `int` video ids going forward.

- [ ] **Step 1: Write the failing test**

Replace `tests/data/test_videos_repository.py` in full:

```python
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer


def test_register_and_get(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = VideosRepository(writer=writer, db_path=prod_db)
        video_id = repo.register(
            source_path="/abs/path/IMG_5872.MOV",
            file_hash="sha256:deadbeef",
            duration_ms=12_345,
            width=3840,
            height=2160,
        )
        writer.flush()
    finally:
        writer.stop()

    assert isinstance(video_id, int) and video_id > 0
    row = repo.get(video_id)
    assert row["source_path"] == "/abs/path/IMG_5872.MOV"
    assert row["file_hash"] == "sha256:deadbeef"
    assert row["duration_ms"] == 12_345
    assert row["width"] == 3840
    assert row["height"] == 2160
    assert row["status"] == "processing"


def test_list_recent_returns_newest_first(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = VideosRepository(writer=writer, db_path=prod_db)
        ids = [
            repo.register("/a.MOV", "h1", 1, 100, 100),
            repo.register("/b.MOV", "h2", 2, 100, 100),
            repo.register("/c.MOV", "h3", 3, 100, 100),
        ]
        writer.flush()
        recent = repo.list_recent(limit=2)
    finally:
        writer.stop()

    assert [r["id"] for r in recent] == [ids[-1], ids[-2]]
```

- [ ] **Step 2: Run and confirm it fails**

```bash
python3 -m pytest tests/data/test_videos_repository.py -v
```

Expected: FAIL — the existing `VideosRepository.register` takes `(video_id, path, metadata)` and writes to non-existent columns.

- [ ] **Step 3: Rewrite the repository**

Replace `src/card_capture/data/repositories/videos.py` in full:

```python
"""videos repository — production schema (migrations/0001_v4_schema.sql)."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import open_connection, read_connection
from card_capture.data.writer import Writer, Write


class VideosRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def register(
        self,
        source_path: str,
        file_hash: str,
        duration_ms: int,
        width: int,
        height: int,
        status: str = "processing",
    ) -> int:
        """Insert a videos row and return the autoincrement id.

        Synchronous because callers immediately need the id for FK references.
        Uses a direct connection (not the writer queue) under the writer's
        thread lock to preserve single-writer semantics.
        """
        # Direct synchronous write through the writer's queue, blocking until
        # we have the row id. The writer worker is the only thread holding a
        # write connection; for inserts that must return autoincrement ids we
        # bypass the queue with the writer's lock held.
        with self._writer.serialize():  # see Task C.3 for Writer.serialize
            conn = open_connection(self._db_path)
            try:
                cur = conn.execute(
                    "INSERT INTO videos(source_path, file_hash, duration_ms, "
                    "width, height, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (source_path, file_hash, duration_ms, width, height, status),
                )
                conn.commit()
                return int(cur.lastrowid)
            finally:
                conn.close()

    def update_status(self, video_id: int, status: str) -> None:
        self._writer.submit(Write(
            sql="UPDATE videos SET status=? WHERE id=?",
            params=(status, video_id),
        ))

    def get(self, video_id: int) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, source_path, file_hash, duration_ms, width, height, "
                "status, created_at FROM videos WHERE id=?",
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        keys = ("id", "source_path", "file_hash", "duration_ms", "width",
                "height", "status", "created_at")
        return dict(zip(keys, row))

    def list_recent(self, limit: int = 50) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, source_path, file_hash, duration_ms, width, height, "
                "status, created_at FROM videos ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("id", "source_path", "file_hash", "duration_ms", "width",
                "height", "status", "created_at")
        return [dict(zip(keys, r)) for r in rows]
```

- [ ] **Step 4: Add `serialize()` to `Writer` (referenced above)**

Edit `src/card_capture/data/writer.py`. Add this method to the `Writer` class:

```python
    @contextmanager
    def serialize(self) -> Iterator[None]:
        """Acquire the writer lock for callers that must perform a direct
        synchronous write (e.g., to read back an autoincrement id).

        While the lock is held the worker thread keeps draining the queue,
        so this only protects against concurrent direct writers, not against
        the worker. Callers MUST close any connection they opened before
        releasing the lock.
        """
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()
```

Add `from contextlib import contextmanager` and `from typing import Iterator` to the file's imports if not already present.

- [ ] **Step 5: Run the new tests**

```bash
python3 -m pytest tests/data/test_videos_repository.py -v
```

Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/data/repositories/videos.py src/card_capture/data/writer.py tests/data/test_videos_repository.py
git commit -m "feat(v55-phaseC): align VideosRepository with production schema"
```

---

### Task C.3: Align `runs` repository with the production schema

**Files:**
- Modify: `src/card_capture/data/repositories/runs.py`
- Modify: `tests/data/test_runs_repository.py`

Production `pipeline_runs` is `(run_id TEXT PK, video_id INTEGER FK, status TEXT, cards_extracted INTEGER, started_at TEXT, finished_at TEXT)` — no `state`/`error`/`*_ms` columns.

- [ ] **Step 1: Replace the test file in full**

```python
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.repositories.runs import RunsRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer


def _video_id(prod_db: Path) -> int:
    writer = Writer(prod_db); writer.start()
    try:
        vid = VideosRepository(writer=writer, db_path=prod_db).register(
            "/x.MOV", "h", 1, 100, 100,
        )
        writer.flush()
        return vid
    finally:
        writer.stop()


def test_mark_started_then_completed(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db); writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=prod_db)
        repo.mark_started(run_id="r1", video_id=video_id)
        repo.mark_completed(run_id="r1", cards_extracted=7)
        writer.flush()
        row = repo.get("r1")
    finally:
        writer.stop()

    assert row["run_id"] == "r1"
    assert row["video_id"] == video_id
    assert row["status"] == "completed"
    assert row["cards_extracted"] == 7
    assert row["finished_at"] is not None


def test_mark_failed_records_status(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db); writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=prod_db)
        repo.mark_started("r2", video_id)
        repo.mark_failed("r2", error="boom")
        writer.flush()
        row = repo.get("r2")
    finally:
        writer.stop()

    assert row["status"] == "failed"
```

- [ ] **Step 2: Run and confirm failure**

```bash
python3 -m pytest tests/data/test_runs_repository.py -v
```

Expected: FAIL — repository writes nonexistent columns (`state`, `started_at_ms`, etc.).

- [ ] **Step 3: Rewrite the repository**

Replace `src/card_capture/data/repositories/runs.py`:

```python
"""pipeline_runs repository — production schema (migrations/0004_pipeline_runs.sql).

Note: production schema stores `status` (not `state`) and uses TEXT timestamps
generated by `datetime('now')` rather than millisecond integers. The `error`
column does not exist; failures record `status='failed'` only. If a richer
failure surface becomes necessary, add it as a new migration and update this
repository together.
"""
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class RunsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def mark_started(self, run_id: str, video_id: int) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT OR REPLACE INTO pipeline_runs(run_id, video_id, status,
                                                     cards_extracted, started_at)
                VALUES (?, ?, 'running', 0, datetime('now'))
            """,
            params=(run_id, video_id),
        ))

    def mark_completed(self, run_id: str, cards_extracted: int) -> None:
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs
                SET status='completed',
                    cards_extracted=?,
                    finished_at=datetime('now')
                WHERE run_id=?
            """,
            params=(cards_extracted, run_id),
        ))

    def mark_failed(self, run_id: str, error: str | None = None) -> None:
        # `error` accepted for forward-compat; current schema discards it.
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs
                SET status='failed', finished_at=datetime('now')
                WHERE run_id=?
            """,
            params=(run_id,),
        ))

    def get(self, run_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT run_id, video_id, status, cards_extracted, started_at, finished_at "
                "FROM pipeline_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        keys = ("run_id", "video_id", "status", "cards_extracted",
                "started_at", "finished_at")
        return dict(zip(keys, row))

    def list_recent(self, limit: int = 50) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, video_id, status, cards_extracted, started_at, finished_at "
                "FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("run_id", "video_id", "status", "cards_extracted",
                "started_at", "finished_at")
        return [dict(zip(keys, r)) for r in rows]
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/data/test_runs_repository.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/data/repositories/runs.py tests/data/test_runs_repository.py
git commit -m "feat(v55-phaseC): align RunsRepository with production schema"
```

---

### Task C.4: Align `events` repository with the production schema

**Files:**
- Modify: `src/card_capture/data/repositories/events.py`
- Modify: `tests/data/test_events_repository.py`

Production `pipeline_events` is `(id INTEGER PK, video_id INTEGER FK, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json, artifact_ref, created_at)`.

- [ ] **Step 1: Replace the test file**

```python
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.events import EventsRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer


def _video_id(prod_db: Path) -> int:
    writer = Writer(prod_db); writer.start()
    try:
        vid = VideosRepository(writer=writer, db_path=prod_db).register(
            "/x.MOV", "h", 1, 100, 100,
        )
        writer.flush()
        return vid
    finally:
        writer.stop()


def test_record_stage_finished_persists(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db); writer.start()
    try:
        repo = EventsRepository(writer=writer, db_path=prod_db)
        repo.record_stage_finished(
            run_id="r1",
            video_id=video_id,
            stage="detect",
            frame_index=42,
            timestamp_ms=1_700_000_000_000,
            elapsed_ms=12,
            metadata={"corners_found": 4},
        )
        writer.flush()
        events = repo.list_for_run("r1")
    finally:
        writer.stop()

    assert len(events) == 1
    e = events[0]
    assert e["stage_id"] == "detect"
    assert e["frame_index"] == 42
    assert e["event_type"] == "stage_finished"
    assert e["video_id"] == video_id
```

- [ ] **Step 2: Run, confirm fail**

```bash
python3 -m pytest tests/data/test_events_repository.py -v
```

Expected: FAIL — non-existent columns.

- [ ] **Step 3: Rewrite the repository**

Replace `src/card_capture/data/repositories/events.py`:

```python
"""pipeline_events repository — production schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class EventsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def record(
        self,
        *,
        run_id: str | None,
        video_id: int | None,
        stage_id: str,
        frame_index: int,
        timestamp_ms: int,
        event_type: str,
        data: Mapping[str, object] | None = None,
        artifact_ref: str | None = None,
    ) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT INTO pipeline_events(
                    video_id, run_id, stage_id, frame_index, timestamp_ms,
                    event_type, data_json, artifact_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params=(
                video_id, run_id, stage_id, frame_index, timestamp_ms,
                event_type, json.dumps(dict(data)) if data else None, artifact_ref,
            ),
        ))

    def record_stage_finished(
        self,
        *,
        run_id: str,
        video_id: int | None,
        stage: str,
        frame_index: int,
        timestamp_ms: int,
        elapsed_ms: int,
        metadata: Mapping[str, object],
    ) -> None:
        data = {"elapsed_ms": elapsed_ms, **dict(metadata)}
        self.record(
            run_id=run_id,
            video_id=video_id,
            stage_id=stage,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            event_type="stage_finished",
            data=data,
        )

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, video_id, run_id, stage_id, frame_index, timestamp_ms, "
                "event_type, data_json, artifact_ref, created_at "
                "FROM pipeline_events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        keys = ("id", "video_id", "run_id", "stage_id", "frame_index",
                "timestamp_ms", "event_type", "data_json", "artifact_ref",
                "created_at")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            if d["data_json"]:
                try:
                    d["data"] = json.loads(d["data_json"])
                except Exception:
                    d["data"] = None
            out.append(d)
        return out
```

- [ ] **Step 4: Update callers in the pipeline `store` stage**

Locate any uses of the old `record_stage_finished` signature in `src/card_capture/pipeline/stages/store.py` (and any other stage that calls into EventsRepository). They previously passed `(run_id, video_id, stage, elapsed_ms, metadata)` positionally; new signature is keyword-only.

Run:
```bash
grep -rn 'record_stage_finished\|EventsRepository' src/ app/ tests/
```

Expected: a handful of sites. Update each to the new keyword form. The agent must NOT skip this — leaving them on the old signature will cause runtime failures in the LocalPipelineRuntime smoke test.

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/data/test_events_repository.py tests/pipeline/test_runtime_smoke.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/data/repositories/events.py tests/data/test_events_repository.py src/card_capture/pipeline/
git commit -m "feat(v55-phaseC): align EventsRepository with production schema"
```

---

### Task C.5: Add a `0013_v55_repository_schema.sql` migration for tables the repositories need but production lacks

**Files:**
- Create: `migrations/0013_v55_repository_schema.sql`

Repositories `cards.py`, `telemetry.py`, and `labeling.py` reference tables whose v5.5 shape doesn't exist in production. Rather than rewrite each repository against an incompatible old shape, add a migration that creates the v5.5-shaped surface. The agent must check whether `fb_labels`, `truth_files`, etc. already exist in earlier migrations; if so, use the existing shape and skip creating duplicates.

- [ ] **Step 1: Audit existing tables**

Run:
```bash
grep -l 'CREATE TABLE.*\(fb_labels\|truth_files\|telemetry_events\|card_views\)' migrations/
```

Note the result so subsequent CREATE TABLE statements in the new migration use `IF NOT EXISTS` and only add what's truly missing.

- [ ] **Step 2: Write the migration**

```sql
-- migrations/0013_v55_repository_schema.sql
-- Add the v5.5-shaped tables required by the data-access-layer repositories.
-- All statements are IF NOT EXISTS so this migration is safe to re-apply.

-- Card-view metrics, one row per (instance, metric).
-- Distinct from the existing `card_views` table (which holds per-frame views);
-- name this `card_view_metrics` to avoid collision.
CREATE TABLE IF NOT EXISTS card_view_metrics (
    card_instance_id TEXT NOT NULL,
    metric           TEXT NOT NULL,
    value            REAL NOT NULL,
    PRIMARY KEY (card_instance_id, metric)
);

-- Card-instance v5.5 surface used by repositories (card_instance_id string,
-- front/back crop paths). The legacy `card_instances` table keeps its integer
-- id; we add columns that may be missing.
ALTER TABLE card_instances ADD COLUMN front_crop TEXT;
-- Re-running ALTER fails on SQLite if the column already exists; the
-- migrations harness must tolerate this (see migrations/run_migrations.py).
ALTER TABLE card_instances ADD COLUMN back_crop TEXT;

-- Telemetry events recorded by TelemetryRepository.
CREATE TABLE IF NOT EXISTS telemetry_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    kind      TEXT NOT NULL,
    payload   TEXT,
    at_ms     INTEGER NOT NULL
);

-- Front/back labels, used by LabelingRepository.
CREATE TABLE IF NOT EXISTS fb_labels (
    label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_run_id INTEGER,
    instance_id   TEXT NOT NULL,
    frame_index   INTEGER,
    side          TEXT NOT NULL,
    labeler       TEXT NOT NULL DEFAULT 'human',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Ground-truth payloads keyed by video_id, used by LabelingRepository.
CREATE TABLE IF NOT EXISTS truth_files (
    video_id       TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    payload_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
```

- [ ] **Step 3: Make `migrations/run_migrations.py` tolerate "duplicate column" errors**

The two `ALTER TABLE card_instances ADD COLUMN` statements will fail with `duplicate column name: front_crop` on second apply. The migrations runner should treat that error as a no-op only for `ADD COLUMN`.

Read `migrations/run_migrations.py` first to confirm whether such tolerance already exists. If not, add a guard:

```python
# inside the per-statement loop
import sqlite3
try:
    cursor.execute(statement)
except sqlite3.OperationalError as exc:
    msg = str(exc).lower()
    if "duplicate column name" in msg:
        continue
    raise
```

- [ ] **Step 4: Run the migrations against a fresh database**

```bash
python3 -c "
from pathlib import Path
import tempfile
from migrations.run_migrations import apply_migrations
with tempfile.TemporaryDirectory() as d:
    db = Path(d) / 'x.db'
    apply_migrations(db)
    print('ok')
"
```

Expected: `ok`.

- [ ] **Step 5: Run them twice (idempotency check)**

```bash
python3 -c "
from pathlib import Path
import tempfile
from migrations.run_migrations import apply_migrations
with tempfile.TemporaryDirectory() as d:
    db = Path(d) / 'x.db'
    apply_migrations(db)
    apply_migrations(db)
    print('ok twice')
"
```

Expected: `ok twice`.

- [ ] **Step 6: Commit**

```bash
git add migrations/0013_v55_repository_schema.sql migrations/run_migrations.py
git commit -m "feat(v55-phaseC): 0013 migration adds v5.5 tables (telemetry_events, fb_labels, truth_files, card_view_metrics)"
```

---

### Task C.6: Align `cards`, `labeling`, `telemetry` repositories with the new tables

**Files:**
- Modify: `src/card_capture/data/repositories/cards.py`
- Modify: `src/card_capture/data/repositories/labeling.py`
- Modify: `src/card_capture/data/repositories/telemetry.py`
- Modify: `tests/data/test_cards_repository.py`
- Modify: `tests/data/test_labeling_repository.py`
- Modify: `tests/data/test_telemetry_repository.py`

- [ ] **Step 1: Replace `tests/data/test_cards_repository.py`**

```python
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.cards import CardsRepository
from card_capture.data.repositories.runs import RunsRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer
from card_capture.pipeline.request import CardRecord


def _setup_run(prod_db: Path, writer: Writer) -> tuple[int, str]:
    vid = VideosRepository(writer=writer, db_path=prod_db).register(
        "/v.MOV", "h", 1, 100, 100,
    )
    writer.flush()
    RunsRepository(writer=writer, db_path=prod_db).mark_started("run_x", vid)
    writer.flush()
    return vid, "run_x"


def test_store_and_list_for_run(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        video_id, run_id = _setup_run(prod_db, writer)
        repo = CardsRepository(writer=writer, db_path=prod_db)
        repo.store_final_cards(run_id=run_id, video_id=video_id, cards=[
            CardRecord(
                card_instance_id="card_0",
                front_crop="artifact://local/run_x/crops/card_0_front.png",
                back_crop="artifact://local/run_x/crops/card_0_back.png",
                quality={"sharpness": 12.3, "glare": 0.05},
            ),
            CardRecord(
                card_instance_id="card_1",
                front_crop="artifact://local/run_x/crops/card_1_front.png",
                quality={"sharpness": 14.0},
            ),
        ])
        writer.flush()
        cards = repo.list_for_run(run_id)
    finally:
        writer.stop()

    assert len(cards) == 2
    by_id = {c["card_instance_id"]: c for c in cards}
    assert by_id["card_0"]["back_crop"].endswith("card_0_back.png")
    assert by_id["card_0"]["quality"]["sharpness"] == 12.3
    assert by_id["card_1"]["back_crop"] is None
```

- [ ] **Step 2: Rewrite `src/card_capture/data/repositories/cards.py`**

```python
"""Cards repository.

Writes to the production `card_instances` table (extended with front_crop /
back_crop columns by migration 0013) and the v5.5 `card_view_metrics` table.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write
from card_capture.pipeline.request import CardRecord


class CardsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def store_final_cards(
        self,
        *,
        run_id: str,
        video_id: int,
        cards: Iterable[CardRecord],
    ) -> None:
        for c in cards:
            # `instance_id` is the production TEXT UUID; we reuse the
            # repository's `card_instance_id` as the value.
            self._writer.submit(Write(
                sql="""
                    INSERT OR REPLACE INTO card_instances(
                        instance_id, video_id, run_id, track_id,
                        front_crop, back_crop
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                params=(
                    c.card_instance_id, video_id, run_id, c.card_instance_id,
                    c.front_crop, c.back_crop,
                ),
            ))
            for metric, value in c.quality.items():
                self._writer.submit(Write(
                    sql="""
                        INSERT OR REPLACE INTO card_view_metrics(
                            card_instance_id, metric, value
                        ) VALUES (?, ?, ?)
                    """,
                    params=(c.card_instance_id, metric, float(value)),
                ))

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT instance_id, front_crop, back_crop "
                "FROM card_instances WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
            out: list[dict] = []
            for instance_id, front, back in rows:
                quality = dict(conn.execute(
                    "SELECT metric, value FROM card_view_metrics "
                    "WHERE card_instance_id=?",
                    (instance_id,),
                ).fetchall())
                out.append({
                    "card_instance_id": instance_id,
                    "front_crop": front,
                    "back_crop": back,
                    "quality": quality,
                })
            return out

    def get(self, card_instance_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT instance_id, run_id, front_crop, back_crop "
                "FROM card_instances WHERE instance_id=?",
                (card_instance_id,),
            ).fetchone()
        if row is None:
            return None
        with read_connection(self._db_path) as conn:
            quality = dict(conn.execute(
                "SELECT metric, value FROM card_view_metrics WHERE card_instance_id=?",
                (card_instance_id,),
            ).fetchall())
        return {
            "card_instance_id": row[0],
            "run_id": row[1],
            "front_crop": row[2],
            "back_crop": row[3],
            "quality": quality,
        }
```

- [ ] **Step 3: Replace `tests/data/test_labeling_repository.py`**

```python
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.labeling import LabelingRepository
from card_capture.data.writer import Writer


def test_store_and_list_fb_labels(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = LabelingRepository(writer=writer, db_path=prod_db)
        repo.store_fb_label(instance_id="cardA", frame_index=10, side="front")
        repo.store_fb_label(instance_id="cardA", frame_index=22, side="back")
        writer.flush()
        rows = repo.list_for_instance("cardA")
    finally:
        writer.stop()

    assert {(r["frame_index"], r["side"]) for r in rows} == {(10, "front"), (22, "back")}


def test_store_and_get_truth_payload(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = LabelingRepository(writer=writer, db_path=prod_db)
        payload = {"cards": [{"id": 1}], "ts": 12345}
        repo.store_truth_payload(video_id="V001", payload=payload)
        writer.flush()
        got = repo.get_truth_payload("V001")
    finally:
        writer.stop()

    assert got == payload
```

- [ ] **Step 4: Adjust `labeling.py` only as needed**

The existing repository's SQL targets the columns created in Task C.5's migration. Verify it still passes; if `list_unlabeled` references `card_instances.fused_image_path`, leave it — that column exists in production. Run the test:

```bash
python3 -m pytest tests/data/test_labeling_repository.py -v
```

Expected: 2 PASS. If a column mismatch surfaces, fix in this commit.

- [ ] **Step 5: Replace `tests/data/test_telemetry_repository.py`**

```python
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.telemetry import TelemetryRepository
from card_capture.data.writer import Writer


def test_record_and_list(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = TelemetryRepository(writer=writer, db_path=prod_db)
        repo.record_event(run_id="r1", kind="stage_started", payload={"stage": "detect"})
        repo.record_event(run_id="r1", kind="stage_finished",
                          payload={"stage": "detect", "elapsed_ms": 9})
        writer.flush()
        events = repo.list_for_run("r1")
    finally:
        writer.stop()

    kinds = [e["kind"] for e in events]
    assert kinds == ["stage_started", "stage_finished"]
```

- [ ] **Step 6: Run every repository test**

```bash
python3 -m pytest tests/data/ -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/card_capture/data/repositories/ tests/data/
git commit -m "feat(v55-phaseC): align cards/labeling/telemetry repos with prod schema"
```

---

### Task C.7: Add four new repositories the migration callers need

**Files:**
- Create: `src/card_capture/data/repositories/config.py`
- Create: `src/card_capture/data/repositories/batch.py`
- Create: `src/card_capture/data/repositories/training.py`
- Create: `src/card_capture/data/repositories/ml.py`
- Create: `tests/data/test_config_repository.py`
- Create: `tests/data/test_batch_repository.py`
- Create: `tests/data/test_training_repository.py`
- Create: `tests/data/test_ml_repository.py`

Without these, Tasks C.10–C.14 cannot eliminate the raw `sqlite3` callers in `app/api/config.py`, `app/api/batch.py`, `app/services/training_service.py`, `src/card_capture/ml/registry.py`. Look at each existing caller to determine the exact columns needed.

For each repository:

- [ ] **Step 1: Audit the target callers**

For each (file, repository) pair below, read the file to enumerate the SELECT/INSERT/UPDATE statements it issues today. The repository must expose at least the methods needed to replace them.

| Existing file (callsites) | Target repository |
|---|---|
| `app/api/config.py` (lines 108, 143) | `repositories/config.py` |
| `app/api/batch.py` (line 73) | `repositories/batch.py` |
| `app/services/training_service.py` (11 sites) | `repositories/training.py` (use existing labeling + new training) |
| `src/card_capture/ml/registry.py` (lines 46, 59, 75) | `repositories/ml.py` |

- [ ] **Step 2: For each repository, write the failing test first**

Example template — apply to all four. Save as `tests/data/test_config_repository.py`:

```python
"""ConfigRepository read/write tests against the production schema.

The config_presets table is created by migrations/0003_config_presets.sql.
"""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.config import ConfigRepository
from card_capture.data.writer import Writer


def test_set_and_get_preset(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = ConfigRepository(writer=writer, db_path=prod_db)
        repo.upsert_preset(name="balanced", config={"corner_confidence": 0.5})
        writer.flush()
        loaded = repo.get_preset("balanced")
    finally:
        writer.stop()
    assert loaded == {"corner_confidence": 0.5}


def test_get_missing_returns_none(prod_db: Path) -> None:
    repo = ConfigRepository(writer=None, db_path=prod_db)  # reads only
    assert repo.get_preset("does-not-exist") is None
```

Repeat the pattern for `test_batch_repository.py`, `test_training_repository.py`, `test_ml_repository.py`. Each test:
1. Uses `prod_db` fixture from `conftest.py`.
2. Exercises a write through `Writer` + `repo.<method>`.
3. Reads back via `repo.<getter>`.
4. Does NOT use raw `sqlite3` in the test body (the whole point).

- [ ] **Step 3: Implement each repository**

`src/card_capture/data/repositories/config.py`:

```python
"""config_presets repository (production schema: migrations/0003_config_presets.sql)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class ConfigRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def upsert_preset(self, *, name: str, config: Mapping[str, object]) -> None:
        if self._writer is None:
            raise RuntimeError("ConfigRepository.upsert_preset requires a Writer")
        self._writer.submit(Write(
            sql="INSERT OR REPLACE INTO config_presets(name, payload_json) VALUES (?, ?)",
            params=(name, json.dumps(dict(config))),
        ))

    def get_preset(self, name: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM config_presets WHERE name=?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def list_presets(self) -> list[str]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM config_presets ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]
```

Implement the other three repositories with the same shape (Writer-or-None constructor, writes via `self._writer.submit(Write(...))`, reads via `read_connection`). The exact method surface each must expose:

**`src/card_capture/data/repositories/batch.py` — `BatchRepository`**

Required methods (derived from `app/api/batch.py:73-74`):

```python
class BatchRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None: ...

    # WRITE — enqueue a batch row; the actual SQL is determined by reading
    # `app/api/batch.py` for the existing INSERT statement.
    def enqueue(self, *, batch_id: str, video_id: int, status: str = "queued") -> None: ...

    # READ — list pending batches for the API to return.
    def list_pending(self) -> list[dict]: ...

    # WRITE — mark a batch finished.
    def mark_finished(self, *, batch_id: str, success: bool) -> None: ...
```

The agent must open `app/api/batch.py` and read the exact SQL it issues before writing this repository, then mirror those columns. If `batch.py`'s SQL targets a table created in `migrations/` other than `0013`, use that table; do not invent a new one.

**`src/card_capture/data/repositories/training.py` — `TrainingRepository`**

Required methods (derived from the 11 sqlite3 sites in `app/services/training_service.py` plus `src/card_capture/ml/train_fb.py:21`):

```python
class TrainingRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None: ...

    # WRITES
    def add_training_sample(self, *, video_id: int, frame_index: int,
                            label: str, image_path: str) -> None: ...
    def mark_sample_validated(self, *, sample_id: int) -> None: ...
    def remove_sample(self, *, sample_id: int) -> None: ...

    # READS
    def list_samples(self, *, video_id: int | None = None,
                     limit: int = 100) -> list[dict]: ...
    def get_sample(self, sample_id: int) -> dict | None: ...
    def export_dataset(self, *, video_ids: list[int]) -> list[dict]: ...
```

The agent must open `app/services/training_service.py` and `src/card_capture/ml/train_fb.py` to inventory exactly which columns each callsite reads/writes; add methods if a callsite needs one not listed above. Methods that wrap a single SELECT should accept the same parameters the caller already passes.

**`src/card_capture/data/repositories/ml.py` — `MLRepository`**

Required methods (derived from `src/card_capture/ml/registry.py:46,59,75`):

```python
class MLRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None: ...

    # READ — get the active model id for a stage.
    def get_active_model(self, *, stage: str) -> dict | None: ...

    # WRITE — register a new model version.
    def register_model(self, *, stage: str, version: str, path: str,
                       metadata: Mapping[str, object] | None = None) -> int: ...

    # WRITE — activate an existing model version (deactivates others for stage).
    def set_active(self, *, model_id: int) -> None: ...

    # READ — list every registered model for a stage.
    def list_models(self, *, stage: str) -> list[dict]: ...
```

The agent must read `src/card_capture/ml/registry.py` first to confirm the actual `model_registry` columns and adjust the methods to match. If `model_registry` does not exist yet (check `migrations/`), add it to `migrations/0013_v55_repository_schema.sql` from Task C.5 in this same commit.

Each implementation must follow the `ConfigRepository` pattern exactly:
- Writes go through `self._writer.submit(Write(sql=..., params=(...)))`.
- Reads use `with read_connection(self._db_path) as conn:` and return `dict` (or `None`/`list[dict]`).
- The constructor accepts a `Writer | None`; methods that write raise `RuntimeError("…requires a Writer")` if `self._writer is None`.
- No `import sqlite3` anywhere in any of these files.

- [ ] **Step 4: Run all the new tests**

```bash
python3 -m pytest tests/data/test_config_repository.py tests/data/test_batch_repository.py tests/data/test_training_repository.py tests/data/test_ml_repository.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/data/repositories/config.py src/card_capture/data/repositories/batch.py src/card_capture/data/repositories/training.py src/card_capture/data/repositories/ml.py tests/data/test_config_repository.py tests/data/test_batch_repository.py tests/data/test_training_repository.py tests/data/test_ml_repository.py
git commit -m "feat(v55-phaseC): ConfigRepository, BatchRepository, TrainingRepository, MLRepository"
```

---

### Task C.8: Migrate `src/card_capture/cli.py`

**Files:**
- Modify: `src/card_capture/cli.py`

Single callsite: `cli.py:290-301` opens a raw sqlite3 connection to enumerate video ids inside `_run_dataset`.

- [ ] **Step 1: Replace the raw block**

In `src/card_capture/cli.py`, find:

```python
def _run_dataset(args: argparse.Namespace) -> int:
    import sqlite3
    from .presence.training_data import export_dataset

    db_path: Path = args.db
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    if args.video_id is not None:
        video_ids = [args.video_id]
    else:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id FROM videos ORDER BY id").fetchall()
        video_ids = [r[0] for r in rows]
```

Replace with:

```python
def _run_dataset(args: argparse.Namespace) -> int:
    from .data.connection import read_connection
    from .presence.training_data import export_dataset

    db_path: Path = args.db
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    if args.video_id is not None:
        video_ids = [args.video_id]
    else:
        with read_connection(db_path) as conn:
            rows = conn.execute("SELECT id FROM videos ORDER BY id").fetchall()
        video_ids = [r[0] for r in rows]
```

- [ ] **Step 2: Verify**

```bash
grep -n 'import sqlite3\|sqlite3.connect' src/card_capture/cli.py
```

Expected: no output.

- [ ] **Step 3: Smoke-run the CLI help to confirm no syntax errors**

```bash
python3 -c "from card_capture import cli; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/cli.py
git commit -m "refactor(v55-phaseC): cli.py uses read_connection, not raw sqlite3"
```

---

### Task C.9: Migrate `src/card_capture/timeline_data.py`, `src/card_capture/storage.py`, `src/card_capture/ml/*`, `src/card_capture/training/*`

**Files:**
- Modify: `src/card_capture/timeline_data.py`
- Modify: `src/card_capture/storage.py`
- Modify: `src/card_capture/ml/registry.py`
- Modify: `src/card_capture/ml/train_fb.py`
- Modify: `src/card_capture/ml/training/dedup_calibrate.py`
- Modify: `src/card_capture/training/presence_trainer.py`
- Modify: `src/card_capture/training/fb_trainer.py`

Pattern, applied uniformly:
- Replace `import sqlite3` with `from card_capture.data.connection import read_connection` for read-only blocks.
- For files that perform writes (e.g., `ml/registry.py` writes to `model_registry`), use the relevant repository created in C.7 (MLRepository, TrainingRepository).
- For `storage.py`'s `Storage.add_video()` and friends, route through `VideosRepository` / `CardsRepository`.

- [ ] **Step 1: For each file, do the in-place edit**

For each of the seven files above:
1. Add the new import.
2. Replace each `with sqlite3.connect(...) as conn:` block with `with read_connection(...) as conn:` (reads) or with a repository method call (writes).
3. Remove the now-unused `import sqlite3` line.
4. Run `python3 -c "import <module>; print('ok')"` to confirm no NameError / ImportError.

Example for `src/card_capture/timeline_data.py:1-5`:

Before:
```python
import sqlite3

def fetch_timeline(db_path):
    conn = sqlite3.connect(db_path)
    ...
```

After:
```python
from card_capture.data.connection import read_connection

def fetch_timeline(db_path):
    with read_connection(db_path) as conn:
        ...
```

For `src/card_capture/storage.py`: the `Storage` class is invoked from several callers (FastAPI, harness, old tests). Choose between:
- **Option A (recommended):** Rewrite `Storage` as a thin wrapper that delegates each method to the new repositories + writer. Constructor takes `db_path`; methods open/close their own `Writer` per call (or accept one). Keeps every caller unchanged.
- **Option B:** Move every caller off `Storage` (more invasive). Leave for a follow-up if Option A is feasible.

Take Option A unless something blocks it. The agent must NOT delete `Storage` without first proving via grep that nothing imports it.

```bash
grep -rn 'from card_capture.storage import\|from .storage import\|card_capture.storage' --include='*.py' src/ app/ tests/ harness/
```

If the grep shows zero callers outside `tests/`, the agent may delete `storage.py` and the corresponding callers; otherwise apply Option A.

- [ ] **Step 2: Verify no raw sqlite3 in the seven files**

```bash
grep -rn 'import sqlite3\|sqlite3.connect' src/card_capture/timeline_data.py src/card_capture/storage.py src/card_capture/ml/registry.py src/card_capture/ml/train_fb.py src/card_capture/ml/training/dedup_calibrate.py src/card_capture/training/presence_trainer.py src/card_capture/training/fb_trainer.py
```

Expected: no output.

- [ ] **Step 3: Run the full test suite to catch regressions**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q
```

Expected: PASS (same count as before this task).

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/timeline_data.py src/card_capture/storage.py src/card_capture/ml/ src/card_capture/training/
git commit -m "refactor(v55-phaseC): src/card_capture/* uses data.connection + repositories"
```

---

### Task C.10: Migrate `app/main.py`, `app/worker_core.py`, `app/runpod_handler.py`, `app/api/{config,batch}.py`

**Files:**
- Modify: `app/main.py`
- Modify: `app/worker_core.py`
- Modify: `app/runpod_handler.py`
- Modify: `app/api/config.py`
- Modify: `app/api/batch.py`

`app/main.py:39` does `sqlite3.connect(db_path).close()` as a "create the file" bootstrap. That bootstrap belongs in the writer's initialization, not in app startup; remove the line and let `Writer.start()` (or repository methods that hit `open_connection`) create the file via WAL mode.

- [ ] **Step 1: app/main.py — replace the bootstrap**

Locate the block around `app/main.py:37-40`:

```python
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db_path).close()
```

Replace with:

```python
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # open_connection() applies PRAGMAs (WAL, busy_timeout) and creates
        # the file. We immediately close; subsequent reads/writes go through
        # repositories / the Writer.
        from card_capture.data.connection import open_connection
        open_connection(db_path).close()
```

Then remove the top-of-file `import sqlite3`.

- [ ] **Step 2: app/worker_core.py — replace the with-block**

Find each `with sqlite3.connect(...) as conn:` block and convert to either:
- `with read_connection(...) as conn:` for SELECT-only blocks
- A repository call for INSERT/UPDATE blocks

Remove the `import sqlite3` line.

- [ ] **Step 3: app/runpod_handler.py — replace the late import**

Find `import sqlite3 as _sqlite3` (line ~249) and the corresponding `_sqlite3.connect(...)` call. Replace with `read_connection` for reads or a repository method for writes.

- [ ] **Step 4: app/api/config.py — use ConfigRepository**

Replace both `with sqlite3.connect(str(db_path)) as conn:` blocks (lines ~108 and ~143) with the appropriate `ConfigRepository.get_preset` / `ConfigRepository.upsert_preset` / `ConfigRepository.list_presets` calls created in Task C.7.

Remove `import sqlite3`.

- [ ] **Step 5: app/api/batch.py — use BatchRepository**

Replace the inline `import sqlite3; with sqlite3.connect(db_path) as conn:` block at lines ~73-74 with a `BatchRepository` method.

- [ ] **Step 6: Verify no raw sqlite3 in the five files**

```bash
grep -rn 'import sqlite3\|sqlite3.connect' app/main.py app/worker_core.py app/runpod_handler.py app/api/config.py app/api/batch.py
```

Expected: no output.

- [ ] **Step 7: Run app tests**

```bash
python3 -m pytest tests/app/ -m 'not quarantine' -q
```

Expected: PASS (same count as before — these tests don't touch sqlite3 directly; if the count drops, a real regression was introduced).

- [ ] **Step 8: Commit**

```bash
git add app/main.py app/worker_core.py app/runpod_handler.py app/api/config.py app/api/batch.py
git commit -m "refactor(v55-phaseC): app/ uses repositories + data.connection, not raw sqlite3"
```

---

### Task C.11: Migrate `app/services/training_service.py` and `app/services/result_importer.py` and `app/services/resource_sampler.py`

**Files:**
- Modify: `app/services/training_service.py`
- Modify: `app/services/result_importer.py`
- Modify: `app/services/resource_sampler.py`

`training_service.py` has 11 sqlite3 sites (lines 44, 102, 123, 136, 164, 179, 239, 255, 281, 416). Each is a local `import sqlite3` followed by `with sqlite3.connect(...) as conn: conn.execute(...)`. Use `TrainingRepository` for writes; `read_connection` for reads.

- [ ] **Step 1: Map each callsite to a repository method**

For each of the 11 sites in `training_service.py`, look at the SQL it runs:
- Reads → `with read_connection(db_path) as conn: conn.execute(SQL).fetchall()`. (Tests / one-off reads may keep raw SQL strings; the rule is no `import sqlite3`, not "no SQL anywhere outside the repo file."  The raw-SQL scanner allows tests; for production callers the agent should still prefer a repository method.)
- Writes → `TrainingRepository.<method>` (or `LabelingRepository`/`MLRepository` for label/registry writes).

- [ ] **Step 2: Apply the same to `result_importer.py` and `resource_sampler.py`**

`result_importer.py`: imports `sqlite3` to materialize manifests from remote runs. Route through `CardsRepository.store_final_cards`, `RunsRepository.mark_started`/`mark_completed`, `EventsRepository.record`.

`resource_sampler.py`: writes `resource_samples` rows. Add a method `TelemetryRepository.record_resource_sample(...)` and route the writes through it.

- [ ] **Step 3: Verify**

```bash
grep -rn 'import sqlite3\|sqlite3.connect' app/services/training_service.py app/services/result_importer.py app/services/resource_sampler.py
```

Expected: no output.

- [ ] **Step 4: Run tests for these services**

```bash
python3 -m pytest tests/app/test_result_importer.py tests/app/test_training_api.py tests/app/test_training_endpoints.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/training_service.py app/services/result_importer.py app/services/resource_sampler.py src/card_capture/data/repositories/
git commit -m "refactor(v55-phaseC): app/services/* uses repositories, no raw sqlite3"
```

---

### Task C.12: Migrate the harness — `harness/baseline.py`, `harness/cli.py`, `harness/hard_cases.py`, `harness/match.py`, `harness/metrics/image_quality.py`, `harness/metrics/dedup_accuracy.py`

**Files:**
- Modify: `harness/baseline.py`
- Modify: `harness/cli.py`
- Modify: `harness/hard_cases.py`
- Modify: `harness/match.py`
- Modify: `harness/metrics/image_quality.py`
- Modify: `harness/metrics/dedup_accuracy.py`

Harness files do read-only analytics. Pattern: replace `import sqlite3 ... with sqlite3.connect(str(db_path)) as conn:` with `from card_capture.data.connection import read_connection ... with read_connection(db_path) as conn:`.

`harness/schema.py` is on the scanner allowlist (`tests/architecture/test_raw_sql_outside_data.py`) and may keep raw sqlite3 if it must read raw schema metadata; the agent should not modify it.

- [ ] **Step 1: Apply the pattern**

For each of the six files: replace the top-of-file `import sqlite3` and each `sqlite3.connect(...)` call.

- [ ] **Step 2: Verify**

```bash
grep -rn 'import sqlite3\|sqlite3.connect' harness/ | grep -v 'harness/schema.py'
```

Expected: no output.

- [ ] **Step 3: Run harness-adjacent tests**

```bash
python3 -m pytest tests/harness/ -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add harness/baseline.py harness/cli.py harness/hard_cases.py harness/match.py harness/metrics/
git commit -m "refactor(v55-phaseC): harness/* uses read_connection, not raw sqlite3"
```

---

### Task C.13: Lint-imports clean run

**Files:** (none — verification only)

- [ ] **Step 1: Run lint-imports**

```bash
PYTHONPATH=src:. lint-imports
```

Expected: 6/6 contracts pass. If any forbidden import remains:
- `sqlite3` violations → re-run the grep from C.9/C.10/C.11/C.12 to find the missed file; do not flip the test to "advisory" to make this pass.
- `metaflow` violations → ensure `app/services/playground_service.py` no longer imports `metaflow.Run` (it should not, per the prior verification, but re-check).
- `runpod`/`beam`/`vastai` violations → Phase D will address these inside platforms; if they appear outside `card_capture.platforms`, the call needs migration.

- [ ] **Step 2: Run the architecture lane**

```bash
python3 -m pytest tests/architecture/ -v
```

Expected: 5+ PASS, 0 FAIL, 0 unexpected SKIP. `test_raw_sql_outside_data_blocking` will still SKIP because it remains env-gated until Phase E.

- [ ] **Step 3: Commit any final touchups (no-op if clean)**

```bash
git status
# If any tidying was needed for lint to pass, commit it now.
```

---

# Phase D: Platform Adapters and Vast.ai Deprecation

**Goal:** Finish Phase 5 with a uniform `PipelineRunner` surface across `LocalRunner`, `RunpodRunner`, `BeamRunner`. Delete Vast.ai. Provide `manifests.py` and `failures.py`. Populate `tests/platforms/`.

**Acceptance:**
1. `LocalRunner`, `RunpodRunner`, `BeamRunner` all implement `card_capture.pipeline.runner.PipelineRunner` (`submit`/`wait`/`cancel`).
2. `from card_capture.platforms.failures import map_provider_failure` exists and returns one of the stable categories.
3. `from card_capture.platforms.manifests import import_manifest, export_manifest` round-trips a `RunManifest` to/from a JSON file.
4. `app/services/vast_runner.py`, `app/services/vast_client.py`, and `tests/app/test_vast*.py` are deleted. `vastai` is removed from `pyproject.toml` dependencies and from `.importlinter`'s forbidden-list (since it would now be a meaningless external).
5. `tests/platforms/` exists with at least four test files (one per runner + one for failures/manifests).
6. `lint-imports` exits 0.

---

### Task D.1: Collapse `pipeline/remote.py` into `pipeline/runner.py` with a single `PipelineRunner` Protocol

**Files:**
- Modify: `src/card_capture/pipeline/runner.py`
- Delete: `src/card_capture/pipeline/remote.py`
- Modify: `src/card_capture/pipeline/__init__.py` (re-export if needed)
- Modify: every importer of `RemoteRuntime`

Today `PipelineRunner` (submit/wait/cancel + handles) lives in `runner.py` and `RemoteRuntime` (submit/get_result, returns provider job_id strings) lives in `remote.py`. The two are incompatible surfaces. Plan acceptance says "uniform `submit/wait/cancel`." Eliminate `RemoteRuntime` and rewrite the one concrete implementation (`platforms/runpod.py`) to `PipelineRunner` in Task D.5.

- [ ] **Step 1: Audit `RemoteRuntime` imports**

```bash
grep -rn 'RemoteRuntime\|from card_capture.pipeline.remote\|from .remote import\|pipeline.remote' --include='*.py' src/ app/ tests/
```

Expected: 1–2 callers (the platforms/runpod.py and maybe a test).

- [ ] **Step 2: Delete remote.py**

```bash
git rm src/card_capture/pipeline/remote.py
```

- [ ] **Step 3: Confirm runner.py still defines the unified protocol**

`src/card_capture/pipeline/runner.py` should already contain `PipelineRunHandle`, `PipelineRunStatus`, `PipelineRunner`. No changes required unless the file is missing a docstring; if so:

```python
"""PipelineRunner: uniform submit/wait/cancel surface for local and remote backends.

Concrete implementations live in `card_capture.platforms.*`. The handle is
opaque to callers; each backend stores its provider-specific job_id in
`PipelineRunHandle.opaque`.
"""
```

- [ ] **Step 4: Smoke-import**

```bash
python3 -c "from card_capture.pipeline.runner import PipelineRunner, PipelineRunHandle, PipelineRunStatus; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add -A src/card_capture/pipeline/
git commit -m "refactor(v55-phaseD): collapse RemoteRuntime into PipelineRunner"
```

---

### Task D.2: Create `src/card_capture/platforms/failures.py`

**Files:**
- Create: `src/card_capture/platforms/failures.py`
- Create: `tests/platforms/__init__.py`
- Create: `tests/platforms/test_failures.py`

- [ ] **Step 1: Write the failing test**

`tests/platforms/__init__.py` (empty).

`tests/platforms/test_failures.py`:

```python
"""failures.py exposes stable categories and a mapping helper."""
from __future__ import annotations

import pytest

from card_capture.platforms.failures import (
    PROVIDER_FAILURE_CATEGORIES,
    ProviderFailure,
    map_provider_failure,
)


def test_categories_are_stable_strings():
    assert "preflight_failed" in PROVIDER_FAILURE_CATEGORIES
    assert "submission_failed" in PROVIDER_FAILURE_CATEGORIES
    assert "execution_failed" in PROVIDER_FAILURE_CATEGORIES
    assert "result_invalid" in PROVIDER_FAILURE_CATEGORIES
    assert "cancelled" in PROVIDER_FAILURE_CATEGORIES
    assert "unknown" in PROVIDER_FAILURE_CATEGORIES


def test_map_unknown_provider_returns_unknown_category():
    failure = map_provider_failure(provider="runpod", raw="<garbled blob>")
    assert failure.category == "unknown"
    assert failure.provider == "runpod"
    assert failure.raw == "<garbled blob>"


def test_map_well_known_runpod_phrases():
    f = map_provider_failure(provider="runpod", raw="endpoint not found")
    assert f.category == "preflight_failed"

    f = map_provider_failure(provider="runpod", raw="JOB FAILED: out of memory")
    assert f.category == "execution_failed"

    f = map_provider_failure(provider="runpod", raw="cancelled by user")
    assert f.category == "cancelled"
```

- [ ] **Step 2: Run and confirm fail**

```bash
python3 -m pytest tests/platforms/test_failures.py -v
```

Expected: FAIL (`ModuleNotFoundError: card_capture.platforms.failures`).

- [ ] **Step 3: Implement**

`src/card_capture/platforms/failures.py`:

```python
"""Stable failure categories for provider runs.

All adapters surface failures through `map_provider_failure(provider, raw)`,
returning a `ProviderFailure` with a category drawn from
`PROVIDER_FAILURE_CATEGORIES`. App-facing status code paths consume the
category, not the raw provider string.
"""
from __future__ import annotations

import dataclasses
import re
from typing import FrozenSet


PROVIDER_FAILURE_CATEGORIES: FrozenSet[str] = frozenset({
    "preflight_failed",     # provider-side rejection before job started
    "submission_failed",    # network/auth error during submit
    "execution_failed",     # provider says the job ran and failed
    "result_invalid",       # provider returned, but manifest could not be parsed
    "cancelled",            # caller-initiated cancellation
    "unknown",              # everything else
})


@dataclasses.dataclass(frozen=True)
class ProviderFailure:
    provider: str
    category: str
    raw: str

    def __post_init__(self) -> None:
        if self.category not in PROVIDER_FAILURE_CATEGORIES:
            raise ValueError(
                f"unknown category {self.category!r}; "
                f"must be one of {sorted(PROVIDER_FAILURE_CATEGORIES)}"
            )


_RUNPOD_PHRASE_TO_CATEGORY: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"endpoint\s+not\s+found", re.I), "preflight_failed"),
    (re.compile(r"unauthorized|invalid\s+api\s+key", re.I), "preflight_failed"),
    (re.compile(r"timeout\s+during\s+submit", re.I), "submission_failed"),
    (re.compile(r"job\s+failed", re.I), "execution_failed"),
    (re.compile(r"out\s+of\s+memory|oom", re.I), "execution_failed"),
    (re.compile(r"cancell?ed", re.I), "cancelled"),
)

_BEAM_PHRASE_TO_CATEGORY: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"deployment\s+not\s+found", re.I), "preflight_failed"),
    (re.compile(r"app\s+id\s+missing|missing\s+credentials", re.I), "preflight_failed"),
    (re.compile(r"timeout", re.I), "submission_failed"),
    (re.compile(r"task\s+failed|error\s+in\s+task", re.I), "execution_failed"),
    (re.compile(r"cancell?ed", re.I), "cancelled"),
)


def map_provider_failure(*, provider: str, raw: str) -> ProviderFailure:
    """Best-effort categorization of a provider failure string."""
    phrases = {
        "runpod": _RUNPOD_PHRASE_TO_CATEGORY,
        "beam": _BEAM_PHRASE_TO_CATEGORY,
        "local": (),
    }.get(provider, ())
    for pattern, category in phrases:
        if pattern.search(raw):
            return ProviderFailure(provider=provider, category=category, raw=raw)
    return ProviderFailure(provider=provider, category="unknown", raw=raw)
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python3 -m pytest tests/platforms/test_failures.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/failures.py tests/platforms/__init__.py tests/platforms/test_failures.py
git commit -m "feat(v55-phaseD): platforms.failures with stable categories + provider phrase mapping"
```

---

### Task D.3: Create `src/card_capture/platforms/manifests.py`

**Files:**
- Create: `src/card_capture/platforms/manifests.py`
- Create: `tests/platforms/test_manifests.py`

A small helper that callers (RunpodRunner, BeamRunner) use to import a `RunManifest` from a JSON file the worker dropped in object storage, and to export one for local runs.

- [ ] **Step 1: Failing test**

`tests/platforms/test_manifests.py`:

```python
from __future__ import annotations

from pathlib import Path

from card_capture.pipeline.request import RunManifest
from card_capture.platforms.manifests import export_manifest, import_manifest


def _sample_manifest() -> RunManifest:
    return RunManifest(
        run_id="r1",
        runtime_mode="cpu_debug",
        input_video="artifact://local/x.MOV",
        output_artifacts=["artifact://local/r1/cards/"],
        cards=[],
        stage_timings=[],
        contract_violations=[],
        version="0.5.5+phaseD",
    )


def test_export_then_import_roundtrip(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    path = export_manifest(manifest, tmp_path / "manifest.json")
    assert path.exists()
    loaded = import_manifest(path)
    assert loaded == manifest


def test_import_missing_file_raises(tmp_path: Path) -> None:
    import pytest as _pytest
    with _pytest.raises(FileNotFoundError):
        import_manifest(tmp_path / "nope.json")
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_manifests.py -v
```

Expected: FAIL on `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/card_capture/platforms/manifests.py`:

```python
"""Manifest import/export helpers shared by platform adapters."""
from __future__ import annotations

from pathlib import Path

from card_capture.pipeline.request import RunManifest


def export_manifest(manifest: RunManifest, path: Path | str) -> Path:
    """Write a manifest to disk as JSON. Returns the resolved path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.to_json())
    return p


def import_manifest(path: Path | str) -> RunManifest:
    """Load a manifest from disk; raises FileNotFoundError if the path is empty."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    return RunManifest.from_json(p.read_text())
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python3 -m pytest tests/platforms/test_manifests.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/manifests.py tests/platforms/test_manifests.py
git commit -m "feat(v55-phaseD): platforms.manifests import/export helpers"
```

---

### Task D.4: Implement `LocalRunner`

**Files:**
- Create: `src/card_capture/platforms/local.py`
- Create: `tests/platforms/test_local_runner.py`

A `PipelineRunner` that wraps `LocalPipelineRuntime` synchronously. `submit` invokes the runtime and stashes the manifest; `wait` returns it; `cancel` is a no-op for the synchronous path.

- [ ] **Step 1: Failing test**

`tests/platforms/test_local_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.platforms.local import LocalRunner


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


@pytest.mark.skipif(not FIXTURE.exists(), reason="tiny_synthetic.MOV fixture not present")
def test_submit_then_wait_returns_manifest(tmp_path: Path) -> None:
    runner = LocalRunner()
    req = PipelineRunRequest(
        run_id="local-1",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    handle = runner.submit(req)
    assert handle.backend == "local"
    assert handle.run_id == "local-1"
    result = runner.wait(handle)
    assert result.manifest.run_id == "local-1"


def test_cancel_is_idempotent() -> None:
    from card_capture.pipeline.runner import PipelineRunHandle
    runner = LocalRunner()
    runner.cancel(PipelineRunHandle(run_id="x", backend="local"))
    runner.cancel(PipelineRunHandle(run_id="x", backend="local"))
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_local_runner.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement**

`src/card_capture/platforms/local.py`:

```python
"""Local synchronous runner.

`submit` runs the pipeline inline and stashes the result keyed by run_id;
`wait` returns the stashed result. This implementation is intentionally
synchronous — the PipelineRunner protocol shape is preserved for parity
with remote backends, not for actual parallelism.
"""
from __future__ import annotations

import threading
from typing import Dict

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
)
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.pipeline.runtime_local import LocalPipelineRuntime


class LocalRunner(PipelineRunner):
    def __init__(self, runtime: LocalPipelineRuntime | None = None) -> None:
        self._runtime = runtime or LocalPipelineRuntime()
        self._results: Dict[str, PipelineRunResult] = {}
        self._lock = threading.Lock()

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        result = self._runtime.run(request)
        with self._lock:
            self._results[request.run_id] = result
        return PipelineRunHandle(run_id=request.run_id, backend="local")

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        with self._lock:
            result = self._results.get(handle.run_id)
        if result is None:
            raise KeyError(f"unknown run_id {handle.run_id!r}")
        return result

    def cancel(self, handle: PipelineRunHandle) -> None:
        # Synchronous local runs cannot be cancelled mid-flight; this is a
        # no-op so callers can treat all backends uniformly.
        return
```

- [ ] **Step 4: Run, PASS**

```bash
python3 -m pytest tests/platforms/test_local_runner.py -v
```

Expected: 2 PASS (one may SKIP if the fixture is missing).

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/local.py tests/platforms/test_local_runner.py
git commit -m "feat(v55-phaseD): LocalRunner implements PipelineRunner"
```

---

### Task D.5: Rewrite `RunpodRunner` to `PipelineRunner`

**Files:**
- Modify: `src/card_capture/platforms/runpod.py`
- Create: `tests/platforms/test_runpod_runner.py`

The existing `RunpodRuntime` returns provider job_id strings via `submit`/`get_result`. Rewrite as `RunpodRunner` implementing `submit`/`wait`/`cancel`. Use `failures.map_provider_failure` for error mapping and `manifests.import_manifest` when the worker writes the manifest to local disk after fetching from object storage.

- [ ] **Step 1: Failing test (using a stub runpod client)**

`tests/platforms/test_runpod_runner.py`:

```python
from __future__ import annotations

from card_capture.pipeline.request import (
    PipelineRunRequest, RunManifest,
)
from card_capture.platforms.runpod import RunpodRunner


class _StubEndpoint:
    """In-memory stand-in for runpod.Endpoint used by RunpodRunner."""

    def __init__(self) -> None:
        self._jobs: dict[str, "_StubJob"] = {}

    def run(self, payload: dict) -> "_StubJob":
        job = _StubJob(run_id=payload["run_id"])
        self._jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> "_StubJob":
        return self._jobs[job_id]


class _StubJob:
    def __init__(self, run_id: str) -> None:
        self.id = f"rp-{run_id}"
        self._run_id = run_id
        self._status = "IN_PROGRESS"

    def status(self) -> str:
        # Drive job to completion on first poll.
        if self._status == "IN_PROGRESS":
            self._status = "COMPLETED"
        return self._status

    def output(self) -> dict:
        manifest = RunManifest(
            run_id=self._run_id, runtime_mode="strict_gpu",
            input_video="artifact://s3/x.MOV", output_artifacts=[],
            cards=[], stage_timings=[], contract_violations=[],
            version="0.5.5+test",
        )
        return {"manifest_json": manifest.to_json()}

    def cancel(self) -> None:
        self._status = "CANCELLED"

    def error(self) -> str:
        return ""


def test_submit_wait_roundtrip() -> None:
    endpoint = _StubEndpoint()
    runner = RunpodRunner.__new__(RunpodRunner)
    runner._endpoint = endpoint
    runner._poll_interval = 0.0  # do not sleep in tests

    req = PipelineRunRequest(
        run_id="rp1",
        input_video="artifact://s3/x.MOV",
        output_root="artifact://s3/rp1/",
        runtime_mode="strict_gpu",
    )
    handle = runner.submit(req)
    assert handle.backend == "runpod"
    assert handle.opaque == "rp-rp1"

    result = runner.wait(handle)
    assert result.manifest.run_id == "rp1"
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_runpod_runner.py -v
```

Expected: FAIL on attribute / signature mismatch.

- [ ] **Step 3: Rewrite the runner**

Replace `src/card_capture/platforms/runpod.py` in full:

```python
"""RunPod serverless backend implementing PipelineRunner."""
from __future__ import annotations

import time

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
)
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.platforms.failures import (
    ProviderFailure,
    map_provider_failure,
)


class RunpodRunnerError(RuntimeError):
    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(f"{failure.category}: {failure.raw}")
        self.failure = failure


class RunpodRunner(PipelineRunner):
    def __init__(self, *, api_key: str, endpoint_id: str,
                 poll_interval: float = 1.0) -> None:
        import runpod  # local import keeps the SDK out of fast-path callers
        runpod.api_key = api_key
        self._endpoint = runpod.Endpoint(endpoint_id)
        self._poll_interval = poll_interval

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        try:
            job = self._endpoint.run(request.to_dict())
        except Exception as exc:  # noqa: BLE001
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod", raw=repr(exc))
            ) from exc
        return PipelineRunHandle(
            run_id=request.run_id, backend="runpod", opaque=job.id,
        )

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        job = self._endpoint.get_job(handle.opaque)
        while job.status() not in ("COMPLETED", "FAILED", "CANCELLED"):
            time.sleep(self._poll_interval)

        status = job.status()
        if status == "CANCELLED":
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod", raw="cancelled")
            )
        if status == "FAILED":
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod", raw=str(job.error() or "JOB FAILED"))
            )

        output = job.output()
        try:
            manifest = RunManifest.from_json(output["manifest_json"])
        except Exception as exc:  # noqa: BLE001
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod",
                                     raw=f"result_invalid: {exc!r}")
            ) from exc
        return PipelineRunResult(manifest=manifest)

    def cancel(self, handle: PipelineRunHandle) -> None:
        try:
            self._endpoint.get_job(handle.opaque).cancel()
        except Exception as exc:  # noqa: BLE001
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod", raw=repr(exc))
            ) from exc
```

- [ ] **Step 4: Update any callers of the old `RunpodRuntime` name**

```bash
grep -rn 'RunpodRuntime' --include='*.py' src/ app/ tests/
```

Update each to `RunpodRunner`. Update the constructor call sites — the new signature requires keyword args `api_key` and `endpoint_id`.

- [ ] **Step 5: Run, PASS**

```bash
python3 -m pytest tests/platforms/test_runpod_runner.py -v
```

Expected: 1 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/platforms/runpod.py tests/platforms/test_runpod_runner.py app/ src/
git commit -m "feat(v55-phaseD): RunpodRunner implements PipelineRunner, maps failures uniformly"
```

---

### Task D.6: Implement `BeamRunner`

**Files:**
- Create: `src/card_capture/platforms/beam.py`
- Create: `tests/platforms/test_beam_runner.py`

Beam (the inference-serverless platform) has a similar shape to RunPod: deploy an app, post a payload, poll for a result. Implement the same `PipelineRunner` surface.

- [ ] **Step 1: Failing test with a stub Beam client**

`tests/platforms/test_beam_runner.py`:

```python
from __future__ import annotations

from card_capture.pipeline.request import PipelineRunRequest, RunManifest
from card_capture.platforms.beam import BeamRunner


class _StubBeamApp:
    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}

    def run(self, payload: dict) -> dict:
        task_id = f"beam-{payload['run_id']}"
        self._tasks[task_id] = {"payload": payload, "status": "RUNNING"}
        return {"task_id": task_id}

    def status(self, task_id: str) -> str:
        # Resolve to SUCCEEDED on first read.
        s = self._tasks[task_id]["status"]
        self._tasks[task_id]["status"] = "SUCCEEDED"
        return "SUCCEEDED" if s == "RUNNING" else s

    def result(self, task_id: str) -> dict:
        payload = self._tasks[task_id]["payload"]
        manifest = RunManifest(
            run_id=payload["run_id"], runtime_mode="strict_gpu",
            input_video=payload["input_video"], output_artifacts=[],
            cards=[], stage_timings=[], contract_violations=[],
            version="0.5.5+test",
        )
        return {"manifest_json": manifest.to_json()}

    def cancel(self, task_id: str) -> None:
        self._tasks[task_id]["status"] = "CANCELLED"


def test_submit_wait_roundtrip() -> None:
    app = _StubBeamApp()
    runner = BeamRunner.__new__(BeamRunner)
    runner._app = app
    runner._poll_interval = 0.0

    req = PipelineRunRequest(
        run_id="bm1",
        input_video="artifact://s3/x.MOV",
        output_root="artifact://s3/bm1/",
        runtime_mode="strict_gpu",
    )
    handle = runner.submit(req)
    assert handle.backend == "beam"

    result = runner.wait(handle)
    assert result.manifest.run_id == "bm1"
```

- [ ] **Step 2: Confirm fail**

- [ ] **Step 3: Implement**

`src/card_capture/platforms/beam.py`:

```python
"""Beam serverless backend implementing PipelineRunner."""
from __future__ import annotations

import time

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
)
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.platforms.failures import (
    ProviderFailure,
    map_provider_failure,
)


class BeamRunnerError(RuntimeError):
    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(f"{failure.category}: {failure.raw}")
        self.failure = failure


class BeamRunner(PipelineRunner):
    def __init__(self, *, deployment_id: str, api_key: str,
                 poll_interval: float = 2.0) -> None:
        import beam  # local import; SDK kept out of fast-path callers
        self._app = beam.deployment(deployment_id, token=api_key)
        self._poll_interval = poll_interval

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        try:
            response = self._app.run(request.to_dict())
        except Exception as exc:  # noqa: BLE001
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw=repr(exc))
            ) from exc
        return PipelineRunHandle(
            run_id=request.run_id, backend="beam", opaque=response["task_id"],
        )

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        while True:
            status = self._app.status(handle.opaque)
            if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(self._poll_interval)

        if status == "CANCELLED":
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw="cancelled")
            )
        if status == "FAILED":
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw="task failed")
            )

        result = self._app.result(handle.opaque)
        try:
            manifest = RunManifest.from_json(result["manifest_json"])
        except Exception as exc:  # noqa: BLE001
            raise BeamRunnerError(
                map_provider_failure(provider="beam",
                                     raw=f"result_invalid: {exc!r}")
            ) from exc
        return PipelineRunResult(manifest=manifest)

    def cancel(self, handle: PipelineRunHandle) -> None:
        try:
            self._app.cancel(handle.opaque)
        except Exception as exc:  # noqa: BLE001
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw=repr(exc))
            ) from exc
```

- [ ] **Step 4: Run, PASS**

```bash
python3 -m pytest tests/platforms/test_beam_runner.py -v
```

Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/beam.py tests/platforms/test_beam_runner.py
git commit -m "feat(v55-phaseD): BeamRunner implements PipelineRunner"
```

---

### Task D.7: Deprecate Vast.ai end-to-end

**Files:**
- Delete: `app/services/vast_runner.py`
- Delete: `app/services/vast_client.py`
- Delete: `tests/app/test_vast_client.py`
- Delete: `tests/app/test_vast_runner.py`
- Delete: `tests/app/test_vastai_worker.py`
- Modify: `pyproject.toml` (remove `vastai` from `dependencies`)
- Modify: `.importlinter` (remove `vastai` from forbidden lists since it is no longer a dependency)
- Modify: any caller that referenced `vast_runner` / `vast_client`
- Modify: `app/api/__init__.py` and `app/services/__init__.py` if they re-exported the names

- [ ] **Step 1: Audit callers**

```bash
grep -rn 'vast_runner\|vast_client\|vastai\|VastRunner\|VastClient' --include='*.py' src/ app/ tests/ harness/
```

Expected: hits in the files listed above + any glue/UI router. Each non-test caller must be updated to either:
- Use `card_capture.platforms.runpod.RunpodRunner` if the call site was platform-agnostic.
- Be removed (if the call site was vast-specific UX, e.g., a "Run on Vast.ai" button).

For removed UX surfaces, also remove the corresponding API route and any UI artifacts.

- [ ] **Step 2: Delete files**

```bash
git rm app/services/vast_runner.py app/services/vast_client.py
git rm tests/app/test_vast_client.py tests/app/test_vast_runner.py tests/app/test_vastai_worker.py
```

- [ ] **Step 3: Remove `vastai>=0.5.0` from `pyproject.toml`'s `[project] dependencies`**

```bash
grep -n vastai pyproject.toml
```

Delete the line and the trailing comma normalization.

- [ ] **Step 4: Remove `vastai` from `.importlinter` `forbidden_modules`**

Edit `[importlinter:contract:no-provider-sdk-outside-platforms]` and drop `vastai` from `forbidden_modules` (since it's no longer present at all, a contract against it would be a no-op but kept misleading).

- [ ] **Step 5: Run the suite to surface broken imports**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q
```

Expected: PASS. Fix any `ImportError` raised by glue code that referenced the deleted modules.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(v55-phaseD): deprecate and remove Vast.ai integration"
```

---

### Task D.8: Expose new runners in `platforms/__init__.py`

**Files:**
- Modify: `src/card_capture/platforms/__init__.py`

- [ ] **Step 1: Write the package init**

```python
"""Platform adapters: uniform PipelineRunner surface over local + remote backends."""
from card_capture.platforms.failures import (
    PROVIDER_FAILURE_CATEGORIES,
    ProviderFailure,
    map_provider_failure,
)
from card_capture.platforms.local import LocalRunner
from card_capture.platforms.manifests import export_manifest, import_manifest
from card_capture.platforms.runpod import RunpodRunner, RunpodRunnerError
from card_capture.platforms.beam import BeamRunner, BeamRunnerError

__all__ = [
    "PROVIDER_FAILURE_CATEGORIES",
    "ProviderFailure",
    "map_provider_failure",
    "LocalRunner",
    "RunpodRunner",
    "RunpodRunnerError",
    "BeamRunner",
    "BeamRunnerError",
    "export_manifest",
    "import_manifest",
]
```

- [ ] **Step 2: Smoke-import**

```bash
python3 -c "from card_capture.platforms import LocalRunner, RunpodRunner, BeamRunner, map_provider_failure, export_manifest; print('ok')"
```

Expected: `ok`. (RunpodRunner/BeamRunner imports of `runpod`/`beam` are lazy inside `__init__`, so the import succeeds even without the SDK present.)

- [ ] **Step 3: Commit**

```bash
git add src/card_capture/platforms/__init__.py
git commit -m "feat(v55-phaseD): platforms package re-exports unified runner surface"
```

---

### Task D.9: Lint-imports clean, full suite green

**Files:** none — verification only.

- [ ] **Step 1: Lint**

```bash
PYTHONPATH=src:. lint-imports
```

Expected: all contracts PASS.

- [ ] **Step 2: Full suite**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q
```

Expected: PASS, count >= the post-Phase-C baseline + the new platform tests.

- [ ] **Step 3: Commit any final touchups (no-op if clean)**

---

# Phase E: Flip Blocking + Final Verification

**Goal:** Remove the env-var gate from the raw-SQL scanner, confirm every architecture test is blocking by default, and tag the v5.5 release.

**Acceptance:** `pytest tests/architecture/` runs every test with zero skips (other than capability-gated ones). `lint-imports` exits 0. Default `pytest tests/` lane is green.

---

### Task E.1: Make raw-SQL scanner blocking by default

**Files:**
- Modify: `tests/architecture/test_raw_sql_outside_data.py`

- [ ] **Step 1: Replace the file**

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

- [ ] **Step 2: Run**

```bash
python3 -m pytest tests/architecture/test_raw_sql_outside_data.py -v
```

Expected: PASS. If FAIL, the remaining call sites belong to a missed file from Phase C — find and migrate it before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/architecture/test_raw_sql_outside_data.py
git commit -m "feat(v55-phaseE): raw-SQL scanner is blocking by default"
```

---

### Task E.2: Run every verification command end-to-end

**Files:** none — verification only.

- [ ] **Step 1: Re-run the full default lane**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q
```

Record the pass count.

- [ ] **Step 2: Run architecture lane**

```bash
python3 -m pytest tests/architecture/ -v
```

Expected: every test PASS or expected-skip (capability gates only).

- [ ] **Step 3: Run perf smoke**

```bash
python3 -m pytest tests/performance/test_perf_harness_smoke.py -v
```

Expected: PASS.

- [ ] **Step 4: Run lint-imports**

```bash
PYTHONPATH=src:. lint-imports
```

Expected: all 6 contracts PASS (or 5 after vastai contract was dropped — verify by reading `.importlinter`).

- [ ] **Step 5: Confirm zero raw sqlite3 callers remain**

```bash
grep -rn 'import sqlite3\|sqlite3.connect' --include='*.py' src/card_capture/ app/ pipeline/ harness/ | grep -v 'src/card_capture/data/\|harness/schema.py'
```

Expected: no output.

- [ ] **Step 6: Confirm zero metaflow imports remain**

```bash
grep -rn 'import metaflow\|from metaflow' --include='*.py' src/ app/ pipeline/ tests/ harness/ | grep -v '\.venv\|worktrees'
```

Expected: no output.

If any of these checks fail, fix in this commit; the agent must not advance to Task E.3 with red signals.

---

### Task E.3: Tag the release

**Files:** none — tagging only.

- [ ] **Step 1: Tag**

```bash
git tag v55-complete
git push origin v55-complete
```

(If the user prefers a different tag scheme, mirror what they used previously; if there is no convention, the suggested name is `v55-complete`.)

- [ ] **Step 2: Final summary commit (docs)**

Append a short paragraph to `docs/superpowers/plans/v5-5/baseline-results.md` recording:
- Date of completion.
- Final `pytest tests/ -m 'not quarantine' -q` count.
- `lint-imports` status.
- Tag name.

```bash
git add docs/superpowers/plans/v5-5/baseline-results.md
git commit -m "docs(v55-phaseE): record V5.5 completion baseline"
```

---

## Self-review checklist

After implementing every task above, run:

```bash
python3 -m pytest tests/ -m 'not quarantine' -q
python3 -m pytest tests/architecture/ -v
python3 -m pytest tests/performance/test_perf_harness_smoke.py -q
PYTHONPATH=src:. lint-imports
grep -rn 'import sqlite3\|sqlite3.connect' --include='*.py' src/card_capture/ app/ pipeline/ harness/ | grep -v 'src/card_capture/data/\|harness/schema.py'
grep -rn 'import metaflow\|from metaflow' --include='*.py' src/ app/ pipeline/ tests/ harness/ | grep -v '\.venv\|worktrees'
```

All six commands must succeed (zero output for the greps).

If any check fails, surface the failure and propose a follow-up task; do not silently relax a scanner or contract to pass.

---

## Completion Entry (2026-05-28)

Phase E gauntlet executed on branch `feat/v5-5-refactoring` with final status green.

- `python3 -m pytest -m "not quarantine"`: `585 passed, 35 skipped, 12 deselected, 32 warnings`
- `python3 -m pytest tests/architecture -q`: pass
- `python3 -m pytest tests/performance/test_perf_harness_smoke.py -q`: pass
- `PYTHONPATH=src:. lint-imports` (with user-base bin on PATH): `5 kept, 0 broken`
- `grep -rn 'import sqlite3|sqlite3.connect' --include='*.py' src/card_capture/ app/ pipeline/ harness/ | grep -v 'src/card_capture/data/|harness/schema.py'`: no output
- `grep -rn 'import metaflow|from metaflow' --include='*.py' src/ app/ pipeline/ tests/ harness/ | grep -v '\.venv\|worktrees'`: no output
