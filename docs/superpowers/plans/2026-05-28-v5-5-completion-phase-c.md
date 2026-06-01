# V5.5 Completion — Phase C: Finish the Data-Access Layer Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate every `import sqlite3` and `sqlite3.connect` callsite outside `src/card_capture/data/`, `migrations/`, `tests/`, and `harness/schema.py`. Restore the 15 FastAPI integration tests that broke when the repositories were realigned to the production schema. Get `lint-imports` to exit 0.

**Architecture:** Six sequential sub-phases inside this plan.

- **Sub-phase C.0** salvages the uncommitted repository realignment from the working tree as atomic per-task commits, so it's durable and reviewable.
- **Sub-phase C.1** adds the missing repository unit tests for `BatchRepository`, `ConfigRepository`, `TrainingRepository`, `MLRepository` (Task C.7 Step 2 from the parent plan never landed).
- **Sub-phase C.2** restores the FastAPI integration tests by updating the services they depend on to call the realigned repositories.
- **Sub-phase C.3** migrates `src/card_capture/` raw-sqlite3 callsites (8 files).
- **Sub-phase C.4** migrates `app/services/` and `app/` raw-sqlite3 callsites (7 files).
- **Sub-phase C.5** migrates `harness/` raw-sqlite3 callsites (6 files).
- **Sub-phase C.6** lint-imports green; full default lane green.

**Tech Stack:** SQLite (WAL), Import Linter, pytest, FastAPI.

**Parent plan:** `docs/superpowers/plans/2026-05-28-v5-5-completion.md` (Phase C section). When this plan and the parent disagree, the parent wins.

**Acceptance:**
1. `grep -rn 'import sqlite3\|sqlite3.connect' --include='*.py' src/ app/ pipeline/ harness/ | grep -v 'src/card_capture/data/\|harness/schema.py'` returns zero lines.
2. `python3 -m pytest tests/ -m 'not quarantine' -q` exits 0 with no errors (skips OK).
3. `PYTHONPATH=src:. lint-imports` exits 0.
4. Every commit in this plan is atomic and references the task that produced it.

**Pre-flight context (verified 2026-05-28):**
- The repository realignment (videos/runs/events/cards/labeling/telemetry → production schema, plus four new repos config/batch/training/ml) is **done but uncommitted** — 42 files in the working tree.
- `Writer.serialize()` exists at line 72 of `src/card_capture/data/writer.py`.
- `migrations/0013_v55_repository_schema.sql` exists with `card_view_metrics`, `telemetry_events`, `batch_jobs`, plus ALTERs on `card_instances`, `fb_labels`, `truth_files`.
- `migrations/run_migrations.py` tolerates `duplicate column name` errors and tracks applied files in a `_migrations` table.
- `tests/data/conftest.py` exposes a `prod_db` fixture that applies every migration.
- **21 files** still import `sqlite3` directly outside `card_capture/data/`. Full list in Sub-phase C.3–C.5 tasks below.
- **27 tests fail or error** in the working tree: 15 FastAPI integration errors (services use stale repo signatures), 3 `test_training_api.py` failures, 1 import-linter failure (expected), and 8 unrelated.

---

## File Structure

```text
# Sub-phase C.0 — commit existing work
migrations/0013_v55_repository_schema.sql                  Committed (existing)
migrations/run_migrations.py                               Committed (existing)
tests/data/conftest.py                                     Committed (existing)
src/card_capture/data/writer.py                            Committed (existing — Writer.serialize)
src/card_capture/data/repositories/{videos,runs,events,cards,labeling,telemetry}.py    Committed (existing)
src/card_capture/data/repositories/{config,batch,training,ml}.py                       Committed (existing)
tests/data/test_{videos,runs,events,cards,labeling,telemetry}_repository.py            Committed (existing)

# Sub-phase C.1 — missing repository tests
tests/data/test_config_repository.py                       Created
tests/data/test_batch_repository.py                        Created
tests/data/test_training_repository.py                     Created
tests/data/test_ml_repository.py                           Created

# Sub-phase C.2 — restore FastAPI integration tests
app/services/cards_service.py                              Modified
app/services/runs_service.py                               Modified
app/services/video_service.py                              Modified
app/services/labeling_service.py                           Modified
# (other services touched if their integration tests still error after the four above land)

# Sub-phase C.3 — src/card_capture/ migration
src/card_capture/cli.py                                    Modified
src/card_capture/timeline_data.py                          Modified
src/card_capture/storage.py                                Modified or deleted (see Task C.3.2)
src/card_capture/ml/registry.py                            Modified
src/card_capture/ml/train_fb.py                            Modified
src/card_capture/ml/training/dedup_calibrate.py            Modified
src/card_capture/training/presence_trainer.py              Modified
src/card_capture/training/fb_trainer.py                    Modified

# Sub-phase C.4 — app/ migration
app/main.py                                                Modified
app/worker_core.py                                         Modified
app/runpod_handler.py                                      Modified
app/api/config.py                                          Modified
app/api/batch.py                                           Modified
app/services/result_importer.py                            Modified
app/services/resource_sampler.py                           Modified
app/services/presence_sampler.py                           Modified
app/services/pipeline_runner.py                            Modified
app/services/runpod_runner.py                              Modified
app/services/beam_runner.py                                Modified
app/services/training_service.py                           Modified

# Sub-phase C.5 — harness/ migration
harness/baseline.py                                        Modified
harness/cli.py                                             Modified
harness/hard_cases.py                                      Modified
harness/match.py                                           Modified
harness/metrics/image_quality.py                           Modified
harness/metrics/dedup_accuracy.py                          Modified
```

---

## Sub-phase C.0: Salvage the uncommitted repository realignment

**Goal:** Turn the 42-file working-tree dump into atomic per-task commits so the work is durable.

The repository rewrites are correct and tested (see `pytest tests/data/` in the verification report). The risk is that nothing is committed — `git reset --hard` would erase the entire effort.

---

### Task C.0.1: Commit migration 0013 and `run_migrations.py` updates

**Files:**
- New: `migrations/0013_v55_repository_schema.sql`
- Modified: `migrations/run_migrations.py`

- [ ] **Step 1: Verify both files exist and contain the expected content**

Run:
```bash
ls -la migrations/0013_v55_repository_schema.sql
grep -c 'duplicate column name' migrations/run_migrations.py
```

Expected: `0013_…sql` exists; the `duplicate column name` string occurs at least once in `run_migrations.py`. If either is missing, the working tree has diverged from the verification snapshot — surface this and stop.

- [ ] **Step 2: Run the migrations end-to-end on a tmp DB**

Run:
```bash
python3 -c "
from pathlib import Path
import tempfile
from migrations.run_migrations import apply_migrations
with tempfile.TemporaryDirectory() as d:
    db = Path(d) / 'x.db'
    apply_migrations(db)
    apply_migrations(db)  # idempotency
    print('ok')
"
```

Expected: `ok` (no errors, no exceptions).

- [ ] **Step 3: Commit**

Run:
```bash
git add migrations/0013_v55_repository_schema.sql migrations/run_migrations.py
git commit -m "feat(v55-phaseC): 0013 migration + idempotent ALTER tolerance

0013 adds card_view_metrics, telemetry_events, batch_jobs, and ALTER-ADD
front_crop/back_crop on card_instances. run_migrations.py tolerates
'duplicate column name' OperationalErrors for re-application safety and
tracks applied files in a _migrations table."
```

---

### Task C.0.2: Commit `tests/data/conftest.py`

**Files:**
- New: `tests/data/conftest.py`

- [ ] **Step 1: Verify the fixture works**

Run:
```bash
python3 -m pytest tests/data/ -q --collect-only 2>&1 | tail -5
```

Expected: collects every test under `tests/data/` without ImportError on the `prod_db` fixture.

- [ ] **Step 2: Commit**

```bash
git add tests/data/conftest.py
git commit -m "test(v55-phaseC): prod_db fixture applies every production migration"
```

---

### Task C.0.3: Commit `Writer.serialize()`

**Files:**
- Modified: `src/card_capture/data/writer.py`

- [ ] **Step 1: Verify**

Run:
```bash
grep -nA4 'def serialize' src/card_capture/data/writer.py
```

Expected: method exists around line 72 with the lock-acquire/release body.

- [ ] **Step 2: Commit**

```bash
git add src/card_capture/data/writer.py
git commit -m "feat(v55-phaseC): Writer.serialize() context for callers needing autoincrement ids"
```

---

### Task C.0.4: Commit each realigned repository + its test (one repo per commit)

**Files:** six repositories + six test files.

For each `(repo, test)` pair below, run the test, then commit both files in one atomic commit:

| Order | Repository file | Test file |
|---|---|---|
| 1 | `src/card_capture/data/repositories/videos.py` | `tests/data/test_videos_repository.py` |
| 2 | `src/card_capture/data/repositories/runs.py` | `tests/data/test_runs_repository.py` |
| 3 | `src/card_capture/data/repositories/events.py` | `tests/data/test_events_repository.py` |
| 4 | `src/card_capture/data/repositories/cards.py` | `tests/data/test_cards_repository.py` |
| 5 | `src/card_capture/data/repositories/labeling.py` | `tests/data/test_labeling_repository.py` |
| 6 | `src/card_capture/data/repositories/telemetry.py` | `tests/data/test_telemetry_repository.py` |

- [ ] **Step 1 (per pair): Run the test**

Example for `videos`:
```bash
python3 -m pytest tests/data/test_videos_repository.py -v
```

Expected: PASS. If FAIL, the working-tree implementation diverged from the parent plan's spec — align it before committing.

- [ ] **Step 2 (per pair): Commit**

```bash
git add src/card_capture/data/repositories/videos.py tests/data/test_videos_repository.py
git commit -m "feat(v55-phaseC): align VideosRepository with production schema"
```

Repeat for runs/events/cards/labeling/telemetry, updating the file paths and the commit message component.

---

### Task C.0.5: Commit the four new repositories (without tests yet — those come in Sub-phase C.1)

**Files:**
- New: `src/card_capture/data/repositories/config.py`
- New: `src/card_capture/data/repositories/batch.py`
- New: `src/card_capture/data/repositories/training.py`
- New: `src/card_capture/data/repositories/ml.py`

- [ ] **Step 1: Smoke-import each**

```bash
python3 -c "from card_capture.data.repositories.config import ConfigRepository; print('ok')"
python3 -c "from card_capture.data.repositories.batch import BatchRepository; print('ok')"
python3 -c "from card_capture.data.repositories.training import TrainingRepository; print('ok')"
python3 -c "from card_capture.data.repositories.ml import MLRepository; print('ok')"
```

Expected: four `ok` lines.

- [ ] **Step 2: Commit all four in one commit**

```bash
git add src/card_capture/data/repositories/config.py src/card_capture/data/repositories/batch.py src/card_capture/data/repositories/training.py src/card_capture/data/repositories/ml.py
git commit -m "feat(v55-phaseC): ConfigRepository, BatchRepository, TrainingRepository, MLRepository

Constructors accept Writer | None; writes raise RuntimeError if no Writer;
reads use read_connection. Tests land in Sub-phase C.1."
```

---

### Task C.0.6: Commit miscellaneous working-tree files that supporting the realignment

**Files (audit and commit individually):**

Look at `git status -sb` and triage any remaining modified files in the working tree. Likely candidates:
- `migrations/0001_v4_schema.sql` (if the working tree adjusted column defaults to enable Phase C realignment)
- `src/card_capture/pipeline/runtime_local.py`, `src/card_capture/pipeline/stages/store.py`, `src/card_capture/pipeline/runner.py` (callers of the realigned repositories — adjusting kwargs/signatures)
- `src/card_capture/data/connection.py` (any tweaks to `open_connection` / `read_connection`)
- `src/card_capture/tracking/{botsort,bytetrack}_adapter.py`, `src/card_capture/gpu_refinement.py` (likely unrelated drift — separate commit if they truly belong to V5.5; else `git checkout --` them)
- `tests/migrations/test_schema.py` (likely updated for the new 0013 file)

- [ ] **Step 1: Review `git status -sb` and the diff for each remaining file**

For each remaining file, decide: (a) belongs with the repository realignment → commit with the realignment; (b) belongs with a different phase → leave alone or commit separately; (c) is unintended drift → `git checkout --` it.

- [ ] **Step 2: Commit grouped by intent**

Use commit messages tagged `feat(v55-phaseC)` for migration callers, `refactor(v55-phaseC)` for pure refactors, or revert with `git checkout -- <file>` if drift.

**Sub-phase C.0 complete when:** `git status -sb` is clean for everything that was part of the repository realignment (Sub-phases C.1–C.5's edits will reintroduce changes).

---

## Sub-phase C.1: Add the missing repository unit tests

**Goal:** Cover the four new repositories with the same shape as the six aligned ones.

---

### Task C.1.1: `tests/data/test_config_repository.py`

**Files:**
- Create: `tests/data/test_config_repository.py`

- [ ] **Step 1: Read the existing repository to confirm method names**

Run:
```bash
grep -n '^    def ' src/card_capture/data/repositories/config.py
```

Note the exact method names and signatures. The plan listed `upsert_preset`, `get_preset`, `list_presets`; if the working-tree implementation diverged, write the test against the implemented surface.

- [ ] **Step 2: Write the test**

`tests/data/test_config_repository.py`:

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


def test_list_presets_returns_sorted(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = ConfigRepository(writer=writer, db_path=prod_db)
        repo.upsert_preset(name="b", config={})
        repo.upsert_preset(name="a", config={})
        repo.upsert_preset(name="c", config={})
        writer.flush()
        names = repo.list_presets()
    finally:
        writer.stop()
    assert names == ["a", "b", "c"]
```

- [ ] **Step 3: Run**

```bash
python3 -m pytest tests/data/test_config_repository.py -v
```

Expected: 3 PASS. If a test fails because the repository surface diverges (e.g., `list_presets` doesn't exist), either add the missing method to the repository or remove the test — but do NOT silently drop coverage; record the decision in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_config_repository.py src/card_capture/data/repositories/config.py
git commit -m "test(v55-phaseC): ConfigRepository unit tests against prod schema"
```

---

### Task C.1.2: `tests/data/test_batch_repository.py`

**Files:**
- Create: `tests/data/test_batch_repository.py`

- [ ] **Step 1: Read the repository surface**

```bash
grep -n '^    def ' src/card_capture/data/repositories/batch.py
```

- [ ] **Step 2: Write the test**

`tests/data/test_batch_repository.py`:

```python
"""BatchRepository tests.

batch_jobs table is created by migrations/0013_v55_repository_schema.sql.
"""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.batch import BatchRepository
from card_capture.data.writer import Writer


def test_enqueue_then_list_pending(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = BatchRepository(writer=writer, db_path=prod_db)
        repo.enqueue(batch_id="b1", total=3)
        repo.enqueue(batch_id="b2", total=5)
        writer.flush()
        pending = repo.list_pending()
    finally:
        writer.stop()
    by_id = {p["batch_id"]: p for p in pending}
    assert by_id["b1"]["total"] == 3
    assert by_id["b2"]["total"] == 5
    assert by_id["b1"]["status"] == "queued"


def test_mark_finished_updates_status(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = BatchRepository(writer=writer, db_path=prod_db)
        repo.enqueue(batch_id="b1", total=3)
        repo.mark_finished(batch_id="b1", success=True)
        writer.flush()
        rows = repo.list_pending()
    finally:
        writer.stop()
    # `list_pending` should NOT return the finished batch.
    assert all(r["batch_id"] != "b1" for r in rows)
```

Adjust method names/parameters if the working-tree repository uses different keys (e.g., `total_videos` instead of `total`). The test must call the actually-implemented surface.

- [ ] **Step 3: Run, commit**

```bash
python3 -m pytest tests/data/test_batch_repository.py -v
git add tests/data/test_batch_repository.py src/card_capture/data/repositories/batch.py
git commit -m "test(v55-phaseC): BatchRepository unit tests"
```

---

### Task C.1.3: `tests/data/test_training_repository.py`

**Files:**
- Create: `tests/data/test_training_repository.py`

- [ ] **Step 1: Read the surface**

```bash
grep -n '^    def ' src/card_capture/data/repositories/training.py
```

- [ ] **Step 2: Write tests against the implemented methods**

Write the test exercising at least: `add_training_sample`/equivalent, `list_samples`, `get_sample`. Use the `prod_db` fixture. Refer to Task C.1.1's structure for the pattern.

```python
"""TrainingRepository tests.

training_samples table is created by migrations/0005_training_samples.sql.
"""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.training import TrainingRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer


def _video_id(prod_db: Path) -> int:
    writer = Writer(prod_db); writer.start()
    try:
        vid = VideosRepository(writer=writer, db_path=prod_db).register(
            "/v.MOV", "h", 1, 100, 100,
        )
        writer.flush()
        return vid
    finally:
        writer.stop()


def test_add_and_list_samples(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db); writer.start()
    try:
        repo = TrainingRepository(writer=writer, db_path=prod_db)
        repo.add_training_sample(
            video_id=video_id, frame_index=42,
            label="front", image_path="/x/y/z.png",
        )
        writer.flush()
        samples = repo.list_samples(video_id=video_id)
    finally:
        writer.stop()
    assert len(samples) == 1
    assert samples[0]["frame_index"] == 42
    assert samples[0]["label"] == "front"
```

If the implementation's method signature differs (e.g., uses `class_label` instead of `label`), adjust the test to call what exists. Document any deviation in the commit message.

- [ ] **Step 3: Run, commit**

```bash
python3 -m pytest tests/data/test_training_repository.py -v
git add tests/data/test_training_repository.py src/card_capture/data/repositories/training.py
git commit -m "test(v55-phaseC): TrainingRepository unit tests"
```

---

### Task C.1.4: `tests/data/test_ml_repository.py`

**Files:**
- Create: `tests/data/test_ml_repository.py`

- [ ] **Step 1: Read the surface**

```bash
grep -n '^    def ' src/card_capture/data/repositories/ml.py
```

- [ ] **Step 2: Write tests for `register_model`, `set_active`, `get_active_model`, `list_models`**

`tests/data/test_ml_repository.py`:

```python
"""MLRepository tests against model_registry."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.ml import MLRepository
from card_capture.data.writer import Writer


def test_register_and_get_active(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = MLRepository(writer=writer, db_path=prod_db)
        mid = repo.register_model(
            stage="fb_classifier", version="v1", path="/models/fb_v1.pt",
        )
        repo.set_active(model_id=mid)
        writer.flush()
        active = repo.get_active_model(stage="fb_classifier")
    finally:
        writer.stop()
    assert active is not None
    assert active["version"] == "v1"
    assert active["path"] == "/models/fb_v1.pt"


def test_set_active_deactivates_others(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = MLRepository(writer=writer, db_path=prod_db)
        m1 = repo.register_model(stage="fb_classifier", version="v1", path="/a.pt")
        m2 = repo.register_model(stage="fb_classifier", version="v2", path="/b.pt")
        repo.set_active(model_id=m1)
        repo.set_active(model_id=m2)
        writer.flush()
        active = repo.get_active_model(stage="fb_classifier")
    finally:
        writer.stop()
    assert active["version"] == "v2"
```

If `model_registry` doesn't exist in any migration (the parent plan says to add it to 0013 in that case), check first:

```bash
grep -l 'model_registry' migrations/
```

If absent, add a `CREATE TABLE IF NOT EXISTS model_registry(...)` block to `migrations/0013_v55_repository_schema.sql` in this same commit and re-run the migrations.

- [ ] **Step 3: Run, commit**

```bash
python3 -m pytest tests/data/test_ml_repository.py -v
git add tests/data/test_ml_repository.py src/card_capture/data/repositories/ml.py migrations/0013_v55_repository_schema.sql
git commit -m "test(v55-phaseC): MLRepository unit tests"
```

---

## Sub-phase C.2: Restore the 15 broken FastAPI integration tests

**Goal:** `tests/app/test_integration.py` (TestVideos, TestRuns, TestCards, TestLabelTruth, TestLabelFB) returns to green.

The breakage came from realigning repository signatures without updating their FastAPI service callers. Walk each erroring test, identify the service it exercises, and update that service to call the realigned repository surface.

---

### Task C.2.1: Catalog every failing integration test and map to its service

**Files:**
- (read-only) `tests/app/test_integration.py`

- [ ] **Step 1: Run the failing tests with verbose tracebacks**

```bash
python3 -m pytest tests/app/test_integration.py -v --tb=short 2>&1 | tail -100
```

Expected: 15 ERRORs and 1 FAIL, each with a Python traceback. Save the output to `/tmp/c2-baseline.txt`:

```bash
python3 -m pytest tests/app/test_integration.py -v --tb=short 2>&1 > /tmp/c2-baseline.txt
```

- [ ] **Step 2: Group failures by the service module each test touches**

Read the test file and map each test class to the service file under `app/services/`:

| Test class | Likely service |
|---|---|
| `TestVideos` | `app/services/video_service.py` |
| `TestRuns` | `app/services/runs_service.py` |
| `TestCards` | `app/services/cards_service.py` |
| `TestLabelTruth` | `app/services/labeling_service.py` |
| `TestLabelFB` | `app/services/labeling_service.py` |
| `TestConfigPresets` (already failing in `test_integration.py`) | `app/services/config_service.py` if it exists; else `app/api/config.py` |

Write a short triage doc to `/tmp/c2-triage.md`:

```markdown
| Test | Service | Likely failure cause |
|---|---|---|
| TestVideos::test_get_video | video_service.py | calls VideosRepository.get with wrong arg name |
| ... |
```

This is for your own working notes — not committed.

---

### Task C.2.2: Update `app/services/video_service.py` to the realigned `VideosRepository`

**Files:**
- Modify: `app/services/video_service.py`

- [ ] **Step 1: Audit current calls**

```bash
grep -nE 'VideosRepository|videos_repo|video_service' app/services/video_service.py | head -30
grep -n 'sqlite3' app/services/video_service.py
```

- [ ] **Step 2: Identify each call to the old shape**

The realigned `VideosRepository`:
- `__init__(writer: Writer | None, db_path)` — no other args.
- `register(source_path, file_hash="unknown", duration_ms=0, width=0, height=0, status="processing") -> int` (returns the autoincrement id).
- `update_status(video_id: int, status: str)`.
- `get(video_id: int) -> dict | None` returning keys `id, source_path, file_hash, duration_ms, width, height, status, created_at`.
- `list_recent(limit: int = 50) -> list[dict]`.

If `video_service.py` calls `register(video_id=..., path=..., metadata=...)`, rewrite to the new signature. If it reads `row["video_id"]`, change to `row["id"]`. If it expects `row["path"]`, change to `row["source_path"]`.

- [ ] **Step 3: Strip raw `sqlite3` from the file**

If the service still has `import sqlite3` / `sqlite3.connect`, replace those blocks with `VideosRepository` calls or `read_connection`. Reference Phase C in the parent plan for the canonical pattern.

- [ ] **Step 4: Run the TestVideos suite**

```bash
python3 -m pytest tests/app/test_integration.py::TestVideos -v
```

Expected: all `TestVideos::*` PASS. If one still errors, dig into the exact stack trace and patch.

- [ ] **Step 5: Commit**

```bash
git add app/services/video_service.py
git commit -m "fix(v55-phaseC): video_service uses realigned VideosRepository signature"
```

---

### Task C.2.3: Repeat the same pattern for `runs_service.py`, `cards_service.py`, `labeling_service.py`

**Files:**
- Modify: `app/services/runs_service.py`
- Modify: `app/services/cards_service.py`
- Modify: `app/services/labeling_service.py`

For each service:

- [ ] **Step 1: Audit calls**

```bash
grep -nE 'Repository|sqlite3' app/services/runs_service.py
grep -nE 'Repository|sqlite3' app/services/cards_service.py
grep -nE 'Repository|sqlite3' app/services/labeling_service.py
```

- [ ] **Step 2: Update each to the realigned surfaces**

Realigned surfaces:

`RunsRepository`:
- `mark_started(run_id: str, video_id: int)`
- `mark_completed(run_id: str, cards_extracted: int)`
- `mark_failed(run_id: str, error: str | None = None)`
- `get(run_id: str) -> dict | None` with keys `run_id, video_id, status, cards_extracted, started_at, finished_at`

`CardsRepository`:
- `store_final_cards(*, run_id: str, video_id: int, cards: Iterable[CardRecord])`
- `list_for_run(run_id: str) -> list[dict]` returning rows with `card_instance_id, front_crop, back_crop, quality: {metric: value}`
- `get(card_instance_id: str) -> dict | None`

`LabelingRepository`:
- `store_fb_label(instance_id, frame_index, side, labeler="human", source_run_id=None)`
- `list_for_instance(instance_id: str) -> list[dict]`
- `store_truth_payload(video_id: str, payload, schema_version: int = 1)`
- `get_truth_payload(video_id: str) -> dict | None`
- `list_unlabeled(limit: int = 50) -> list[dict]`

- [ ] **Step 3: Run the corresponding integration tests**

```bash
python3 -m pytest tests/app/test_integration.py::TestRuns tests/app/test_integration.py::TestCards tests/app/test_integration.py::TestLabelTruth tests/app/test_integration.py::TestLabelFB -v
```

Expected: all PASS.

- [ ] **Step 4: Commit per service**

Three commits — one per service — each tagged `fix(v55-phaseC): <service>_service uses realigned <Name>Repository signatures`.

---

### Task C.2.4: Address `TestConfigPresets::test_duplicate_preset_returns_409` and `test_training_api.py` failures

**Files:**
- Modify: `app/api/config.py` (if not yet using `ConfigRepository`)
- Modify: `app/services/training_service.py` (if not yet using `TrainingRepository`/`LabelingRepository`)

- [ ] **Step 1: Run the three remaining failures verbosely**

```bash
python3 -m pytest tests/app/test_integration.py::TestConfigPresets::test_duplicate_preset_returns_409 tests/app/test_training_api.py -v --tb=short
```

- [ ] **Step 2: Patch each callsite**

For `TestConfigPresets`, the test asserts a 409 on duplicate insert. `ConfigRepository.upsert_preset` uses `INSERT OR REPLACE`, which would not raise — so either:
- The API handler must check uniqueness explicitly before calling `upsert_preset`, or
- Add a `ConfigRepository.create_preset(name, config)` method that does a strict `INSERT` and raises `sqlite3.IntegrityError` on conflict; the handler catches it and returns 409.

Choose the second option (less behavior change for callers). Add the method, write a one-test addition to `test_config_repository.py`, and patch `app/api/config.py`.

For `test_training_api.py`, the failure was reported as `sq…` (likely a `sqlite3.OperationalError`). Read the test, identify which TrainingRepository / LabelingRepository call needs to exist or change, and patch.

- [ ] **Step 3: Run, commit**

```bash
python3 -m pytest tests/app/test_integration.py::TestConfigPresets tests/app/test_training_api.py -v
git add app/api/config.py app/services/training_service.py src/card_capture/data/repositories/config.py tests/data/test_config_repository.py
git commit -m "fix(v55-phaseC): config + training APIs use repositories; duplicate preset returns 409"
```

---

### Task C.2.5: Verify the full app test suite returns to green

**Files:** none — verification only.

- [ ] **Step 1: Run the app lane**

```bash
python3 -m pytest tests/app/ -q --tb=line
```

Expected: 0 errors, 0 failures.

- [ ] **Step 2: Run the data lane**

```bash
python3 -m pytest tests/data/ -q
```

Expected: 0 errors, 0 failures.

- [ ] **Step 3: Run the full default lane**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q --tb=line | tail -10
```

Expected: passes are >= 581 (the original baseline). The architecture/test_import_linter.py will still FAIL — that's Sub-phase C.6's job.

---

## Sub-phase C.3: Migrate `src/card_capture/` raw-sqlite3 callsites

**Goal:** Zero `import sqlite3` in `src/card_capture/` outside `data/`.

Pattern, applied uniformly:
- Replace `import sqlite3` with `from card_capture.data.connection import read_connection` for read-only blocks.
- For writes, use the relevant repository created in Sub-phase C.0/C.1.

The eight files to migrate, each as its own task:

---

### Task C.3.1: `src/card_capture/cli.py`

**Files:**
- Modify: `src/card_capture/cli.py`

- [ ] **Step 1: Locate the callsite**

```bash
grep -n 'sqlite3' src/card_capture/cli.py
```

Expected: `import sqlite3` around line 290 inside `_run_dataset`, and a `with sqlite3.connect(db_path) as conn:` block.

- [ ] **Step 2: Replace the block**

In `src/card_capture/cli.py` `_run_dataset`:

Before:
```python
def _run_dataset(args: argparse.Namespace) -> int:
    import sqlite3
    from .presence.training_data import export_dataset
    ...
    if args.video_id is not None:
        video_ids = [args.video_id]
    else:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id FROM videos ORDER BY id").fetchall()
        video_ids = [r[0] for r in rows]
```

After:
```python
def _run_dataset(args: argparse.Namespace) -> int:
    from .data.connection import read_connection
    from .presence.training_data import export_dataset
    ...
    if args.video_id is not None:
        video_ids = [args.video_id]
    else:
        with read_connection(db_path) as conn:
            rows = conn.execute("SELECT id FROM videos ORDER BY id").fetchall()
        video_ids = [r[0] for r in rows]
```

- [ ] **Step 3: Verify**

```bash
grep -n 'sqlite3' src/card_capture/cli.py
```

Expected: no output.

- [ ] **Step 4: Smoke-test**

```bash
python3 -c "from card_capture import cli; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/cli.py
git commit -m "refactor(v55-phaseC): cli.py uses read_connection"
```

---

### Task C.3.2: `src/card_capture/storage.py`

**Files:**
- Modify: `src/card_capture/storage.py` (or delete — see Step 1)

`storage.py` defines a `Storage` class around line 615. The parent plan offers two options: Option A (thin wrapper over repositories) or Option B (delete after migrating callers).

- [ ] **Step 1: Audit callers**

```bash
grep -rn 'from card_capture.storage\|from .storage\|card_capture.storage' --include='*.py' src/ app/ tests/ harness/
```

If only `tests/` imports it: delete `storage.py` and update those tests to use repositories. If any non-test caller imports it, take Option A.

- [ ] **Step 2 (Option A): Rewrite `Storage` as a wrapper**

Replace `Storage._connect` and its callers in this file with `VideosRepository`/`CardsRepository`/`EventsRepository` calls. The `Storage` class remains a thin compatibility layer; methods keep their names but delegate.

Example: `Storage.add_video(...)` becomes:
```python
def add_video(self, *, source_path: str, file_hash: str, duration_ms: int,
              width: int, height: int) -> int:
    writer = Writer(self.db_path); writer.start()
    try:
        vid = VideosRepository(writer=writer, db_path=self.db_path).register(
            source_path=source_path, file_hash=file_hash,
            duration_ms=duration_ms, width=width, height=height,
        )
        writer.flush()
        return vid
    finally:
        writer.stop()
```

For chronic short-lived calls, the per-call Writer startup/teardown is wasteful but acceptable for a compatibility shim — callers wanting performance should construct their own Writer + Repository and bypass `Storage`.

- [ ] **Step 3: Strip `import sqlite3`**

- [ ] **Step 4: Verify**

```bash
grep -n 'sqlite3' src/card_capture/storage.py
python3 -c "from card_capture.storage import Storage; print('ok')"
```

Expected: no `sqlite3`; `ok`.

- [ ] **Step 5: Run the suite to catch breakage**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q --tb=line | tail -5
```

Expected: pass count >= post-C.2 baseline.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/storage.py
git commit -m "refactor(v55-phaseC): Storage wraps repositories instead of raw sqlite3"
```

---

### Tasks C.3.3 – C.3.8: Migrate the remaining six `src/card_capture/` files

For each file below, run the same pattern: audit calls, replace raw `sqlite3` with `read_connection` or repository methods, verify, commit.

| Task | File | Likely repository |
|---|---|---|
| C.3.3 | `src/card_capture/timeline_data.py` | `read_connection` only (reads) |
| C.3.4 | `src/card_capture/ml/registry.py` | `MLRepository` |
| C.3.5 | `src/card_capture/ml/train_fb.py` | `TrainingRepository` + `LabelingRepository` |
| C.3.6 | `src/card_capture/ml/training/dedup_calibrate.py` | `read_connection` only |
| C.3.7 | `src/card_capture/training/presence_trainer.py` | `TrainingRepository` |
| C.3.8 | `src/card_capture/training/fb_trainer.py` | `TrainingRepository` |

For each file, the task body is:

- [ ] **Step 1:** `grep -n 'sqlite3' <file>` to locate sites.
- [ ] **Step 2:** Replace each block with the canonical pattern (read_connection for reads, repository for writes).
- [ ] **Step 3:** `grep -n 'sqlite3' <file>` and confirm no output.
- [ ] **Step 4:** `python3 -c "import <module>; print('ok')"` to confirm imports resolve.
- [ ] **Step 5:** Run related tests (e.g., `tests/ml/`, `tests/training/`) and confirm PASS.
- [ ] **Step 6:** Commit with `refactor(v55-phaseC): <file>: read_connection / <Repo>`

**Sub-phase C.3 complete when:** `grep -n 'sqlite3' src/card_capture/ -r` returns only files inside `src/card_capture/data/`.

---

## Sub-phase C.4: Migrate `app/` raw-sqlite3 callsites

**Goal:** Zero `import sqlite3` in `app/`.

12 files. Same pattern as Sub-phase C.3. Order them so that high-traffic services migrate first (their tests are noisier) and low-traffic helpers last.

---

### Task C.4.1: `app/main.py`

**Files:**
- Modify: `app/main.py`

`app/main.py` does `sqlite3.connect(db_path).close()` as a bootstrap at line 39.

- [ ] **Step 1: Replace the bootstrap**

Locate the block around lines 37–40:

```python
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db_path).close()
```

Replace with:

```python
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        from card_capture.data.connection import open_connection
        # open_connection() applies WAL pragmas and creates the file.
        open_connection(db_path).close()
```

- [ ] **Step 2: Remove the top-of-file `import sqlite3`**

- [ ] **Step 3: Verify, commit**

```bash
grep -n 'sqlite3' app/main.py
python3 -c "from app.main import app; print('ok')"
git add app/main.py
git commit -m "refactor(v55-phaseC): app/main.py uses open_connection for bootstrap"
```

---

### Tasks C.4.2 – C.4.12: Migrate the remaining eleven `app/` files

| Task | File | Likely repository |
|---|---|---|
| C.4.2 | `app/worker_core.py` | mixed — reads use `read_connection`, writes via repositories |
| C.4.3 | `app/runpod_handler.py` | repositories |
| C.4.4 | `app/api/config.py` | `ConfigRepository` (largely done in C.2.4; verify) |
| C.4.5 | `app/api/batch.py` | `BatchRepository` |
| C.4.6 | `app/services/result_importer.py` | `CardsRepository`, `RunsRepository`, `EventsRepository` |
| C.4.7 | `app/services/resource_sampler.py` | `TelemetryRepository` (add `record_resource_sample` if missing) |
| C.4.8 | `app/services/presence_sampler.py` | `read_connection` for reads |
| C.4.9 | `app/services/pipeline_runner.py` | `RunsRepository`, `EventsRepository` |
| C.4.10 | `app/services/runpod_runner.py` | `RunsRepository` |
| C.4.11 | `app/services/beam_runner.py` | `RunsRepository` |
| C.4.12 | `app/services/training_service.py` | `TrainingRepository` + `LabelingRepository` (~11 sites; already partially addressed in C.2.4) |

Each task uses the canonical six-step pattern from C.3.

**Sub-phase C.4 complete when:** `grep -n 'sqlite3' app/ -r` returns no output.

---

## Sub-phase C.5: Migrate `harness/` raw-sqlite3 callsites

**Goal:** Zero `import sqlite3` in `harness/` except `harness/schema.py` (allowlisted).

All six are read-only analytics — use `read_connection`.

| Task | File |
|---|---|
| C.5.1 | `harness/baseline.py` |
| C.5.2 | `harness/cli.py` |
| C.5.3 | `harness/hard_cases.py` |
| C.5.4 | `harness/match.py` |
| C.5.5 | `harness/metrics/image_quality.py` |
| C.5.6 | `harness/metrics/dedup_accuracy.py` |

Same six-step pattern. After each commit, run `python3 -m pytest tests/harness/ -q` to catch regressions.

**Sub-phase C.5 complete when:** `grep -n 'sqlite3' harness/ -r | grep -v 'harness/schema.py'` returns no output.

---

## Sub-phase C.6: Lint-imports green; full default lane green

**Goal:** All Phase C acceptance criteria pass.

---

### Task C.6.1: Run the lint and confirm zero violations

- [ ] **Step 1: Lint**

```bash
PYTHONPATH=src:. lint-imports
```

Expected: 6/6 contracts KEPT. If any are BROKEN:
- `sqlite3` violation → re-run the Sub-phase grep to find the missed file.
- `metaflow` violation → check `app/services/playground_service.py` (was the historical site) and any remaining stale imports.
- `runpod`/`beam` violation → Phase D will close these inside `card_capture.platforms`; for Phase C purposes, only callsites outside `card_capture.platforms` count.

- [ ] **Step 2: Architecture lane**

```bash
python3 -m pytest tests/architecture/ -v
```

Expected: 5–6 PASS. `test_raw_sql_outside_data_blocking` may still SKIP (env-gated until Phase E).

---

### Task C.6.2: Full default lane

- [ ] **Step 1: Run**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q --tb=line | tail -5
```

Expected: passes >= 581 (the original Phase A baseline), errors = 0, failures = 0. If any unexplained failure remains, surface it and patch — do NOT advance to Phase D with red signals.

---

### Task C.6.3: Final Phase C audit grep

- [ ] **Step 1: Confirm zero remaining raw sqlite3 callsites**

```bash
grep -rn 'import sqlite3\|sqlite3.connect' --include='*.py' src/card_capture/ app/ pipeline/ harness/ | grep -v 'src/card_capture/data/\|harness/schema.py'
```

Expected: no output.

- [ ] **Step 2: Commit any final touchups; tag the milestone**

```bash
git tag v55-phaseC-complete
```

(Tag is local-only at this stage; push when the user requests.)

**Phase C complete.** Data-access layer migration is done; Phase D is unblocked.
