# v4 Surface A — Orchestration / Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `pipeline.py` into a Metaflow flow with named artifact persistence, wrap it in a FastAPI service layer with SSE progress, extend storage with the v4 schema, and add Apple-silicon fast paths — without changing pipeline behavior on the regression harness.

**Architecture:** Preserve every algorithm module unchanged. Replace `pipeline.py`'s monolithic orchestration with `pipeline/card_capture_flow.py` (a Metaflow `FlowSpec` < 200 lines) plus one module per `@step` under `pipeline/steps/`. Stages 1–3 (streaming producer/consumer) are wrapped as a single `detect` step. Stage 9 (per-track fusion) is a `foreach`. A new FastAPI service layer (`app/api/`, `app/services/`) exposes the v1 REST surface plus SSE; the existing CLI calls the same services. Storage schema extends with seven new tables. Wave 3 adds feature-detected CoreML / VideoToolbox / vImage fast paths.

**Tech Stack:** Metaflow, FastAPI, SQLite (existing), Python ≥3.11, `multiprocessing` (preserved for streaming), Server-Sent Events. Wave 3 only: CoreMLTools, AVFoundation/VideoToolbox via PyObjC, vImage via PyObjC.

**Spec reference:** `docs/superpowers/specs/2026-05-12-v4-architecture-design.md`. This plan implements Surface A across Waves 1 and 3 plus the SSE/API consumed by Wave 2. Algorithmic changes in Wave 2 belong to Surface C, not this plan.

**Critical safety gate.** Every refactor task ends with `card-capture harness run --baseline baseline_v4.1` showing 0% delta on every metric. Surface D delivers the harness in parallel; this plan begins by writing the regression-detector test that fails until D's harness is in place, then proceeds.

---

## File Structure

**New files (this plan creates):**

- `pipeline/__init__.py` — package marker.
- `pipeline/card_capture_flow.py` — Metaflow `FlowSpec` (target < 200 lines).
- `pipeline/steps/__init__.py`
- `pipeline/steps/detect.py` — wraps Stages 1–3 streaming subsystem as a single Metaflow step.
- `pipeline/steps/novelty.py` — Stage 4 (background novelty gate).
- `pipeline/steps/track.py` — Stage 5 (session-aware tracking).
- `pipeline/steps/refine.py` — Stage 6 (GPU/CPU perspective refinement).
- `pipeline/steps/score.py` — Stage 7 (quality scoring + track pruning).
- `pipeline/steps/resolve.py` — Stage 8 (Front/Back resolution).
- `pipeline/steps/fuse.py` — Stage 9 (per-track fusion, called via `foreach`).
- `pipeline/steps/dedup.py` — Stage 10 dedup half.
- `pipeline/steps/store.py` — Stage 10 storage half.
- `pipeline/contracts.py` — typed artifact contracts (Pydantic) for each step's inputs/outputs.
- `app/__init__.py`
- `app/main.py` — FastAPI app factory + uvicorn entrypoint.
- `app/api/__init__.py`
- `app/api/videos.py` — `/api/v1/videos` routes.
- `app/api/runs.py` — `/api/v1/runs` routes.
- `app/api/cards.py` — `/api/v1/cards` routes.
- `app/api/label.py` — `/api/v1/label` routes (stubs; D fills in).
- `app/api/training.py` — `/api/v1/training` routes (stubs; C fills in).
- `app/api/regression.py` — `/api/v1/regression` routes (stubs; D fills in).
- `app/api/config.py` — `/api/v1/config` routes.
- `app/api/events.py` — SSE channel `/events/{run_id}`.
- `app/services/__init__.py`
- `app/services/pipeline_runner.py` — wraps Metaflow flow invocation + emits SSE events.
- `app/services/storage_service.py` — thin façade over existing `Storage` for API use.
- `app/schemas/__init__.py`
- `app/schemas/v1.py` — Pydantic models for every Contract 2 request/response.
- `migrations/0001_v4_schema.sql` — adds the seven new tables + `pipeline_events` columns.
- `migrations/run_migrations.py` — idempotent migration runner.
- `tests/pipeline/test_flow_runs.py` — smoke test that the full flow runs on a fixture video.
- `tests/pipeline/test_artifact_stability.py` — verifies the named-artifact contract is stable.
- `tests/app/test_api_contract.py` — Contract 2 conformance tests (status codes, shapes).
- `tests/app/test_sse_events.py` — SSE channel event order + payload schema.
- `tests/migrations/test_schema.py` — verifies tables, columns, indices exist.
- `docs/contracts/v1-api.md` — frozen API contract (Contract 2).
- `docs/contracts/storage-schema.md` — frozen storage extensions (Contract 1).
- `docs/contracts/metaflow-artifacts.md` — frozen named-artifact contract (Contract 3).

**Modified files (this plan touches):**

- `src/card_capture/pipeline.py` — gradually emptied as logic moves into `pipeline/steps/*`. Becomes a thin shim that delegates to `pipeline.card_capture_flow.CardCaptureFlow.run()` for backwards compatibility, then is deleted in the final cleanup task.
- `src/card_capture/cli.py` — `process` subcommand now invokes `app.services.pipeline_runner.PipelineRunner` instead of `pipeline.VideoProcessor` directly; new `harness run` subcommand owned by Surface D.
- `pyproject.toml` (or `requirements.txt`) — adds `metaflow`, `fastapi`, `uvicorn[standard]`, `sse-starlette`, `pydantic>=2`.

---

## Phase A0 — Contract Drafting (Wave 1, Day 1)

Before any code: draft the three contracts Surface A owns, get the four-way ack from B/C/D, freeze.

### Task A0.1: Draft Contract 1 — Storage schema additions

**Files:**
- Create: `docs/contracts/storage-schema.md`

- [ ] **Step 1: Write the schema doc**

Create `docs/contracts/storage-schema.md` containing the DDL for the seven new tables exactly as enumerated in Spec §2.2 Contract 1, plus the `pipeline_events` extension. Use this exact DDL — agents B/C/D will reference it.

```sql
-- migrations/0001_v4_schema.sql excerpt; doc shows this verbatim

CREATE TABLE IF NOT EXISTS truth_files (
    video_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS regression_baselines (
    baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    code_sha TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS regression_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id INTEGER REFERENCES regression_baselines(baseline_id),
    code_sha TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    per_video_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fb_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_run_id INTEGER,
    instance_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('front','back','uncertain')),
    labeler TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dedup_clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_member_ids_json TEXT NOT NULL,
    confirmed_member_ids_json TEXT,
    status TEXT NOT NULL CHECK (status IN ('unverified','confirmed','split','merged')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    training_set_hash TEXT NOT NULL,
    eval_metrics_json TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(model_name, training_set_hash)
);

CREATE TABLE IF NOT EXISTS hard_cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    frame_index INTEGER,
    stage_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    thumbnail_path TEXT,
    source_frame_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

ALTER TABLE pipeline_events ADD COLUMN stage_id TEXT;
ALTER TABLE pipeline_events ADD COLUMN artifact_ref TEXT;

CREATE INDEX IF NOT EXISTS idx_regression_runs_baseline ON regression_runs(baseline_id);
CREATE INDEX IF NOT EXISTS idx_fb_labels_instance ON fb_labels(instance_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_events_stage ON pipeline_events(stage_id);
```

The doc explains each table's purpose in 1–2 sentences, lists which surface writes/reads it, and notes which columns are JSON blobs vs. structured.

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/storage-schema.md
git commit -m "docs(contracts): freeze Contract 1 storage schema additions"
```

### Task A0.2: Draft Contract 2 — Service-layer API

**Files:**
- Create: `docs/contracts/v1-api.md`

- [ ] **Step 1: Write the API contract doc**

The doc lists every route from Spec §2.2 Contract 2 with: method, path, request body (Pydantic model name), response body, status codes, error shapes. SSE event names with payload schemas. Concrete OpenAPI-style detail; no "TBD" anywhere.

For each route, include a worked example: minimal valid request → minimal valid 200 response. Surface B will build against these examples before A wires them.

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/v1-api.md
git commit -m "docs(contracts): freeze Contract 2 service-layer API surface"
```

### Task A0.3: Draft Contract 3 — Metaflow artifact contract

**Files:**
- Create: `docs/contracts/metaflow-artifacts.md`

- [ ] **Step 1: Write the artifact-contract doc**

The doc lists each `@step` in `CardCaptureFlow`, the named artifacts it persists (per Spec §2.2 Contract 3 table), and the Python type of each artifact (referencing the dataclasses defined in `pipeline/contracts.py` once they exist). Includes a "stability guarantee" section: names will not change without four-surface ack.

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/metaflow-artifacts.md
git commit -m "docs(contracts): freeze Contract 3 Metaflow artifact names"
```

### Task A0.4: Four-surface contract ack

**Files:** none (process task)

- [ ] **Step 1: Solicit ack from B/C/D**

Open a coordination thread / PR comment / Slack-equivalent referencing the three contract commits. Each surface owner (B, C, D) ack's in writing OR raises a specific concern. Concerns are resolved by edit + re-ack.

- [ ] **Step 2: Tag the contract-freeze commit**

```bash
git tag -a v4-contracts-frozen -m "All four contracts ack'd by A, B, C, D"
git push --tags
```

After this tag, contract changes require four-surface ack again.

---

## Phase A1 — Storage Migrations (Wave 1)

### Task A1.1: Write migration test

**Files:**
- Create: `tests/migrations/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/migrations/test_schema.py
import sqlite3
from pathlib import Path

from migrations.run_migrations import apply_migrations

EXPECTED_TABLES = {
    "truth_files", "regression_baselines", "regression_runs",
    "fb_labels", "dedup_clusters", "model_versions", "hard_cases",
}

def test_v4_schema_creates_expected_tables(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    sqlite3.connect(db_path).close()  # empty db
    apply_migrations(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {r[0] for r in rows}
    assert EXPECTED_TABLES.issubset(table_names)

def test_pipeline_events_has_v4_columns(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    # seed with the existing pipeline_events table (matches current schema)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE pipeline_events (id INTEGER PRIMARY KEY, event_type TEXT, payload TEXT)"
        )
    apply_migrations(db_path)
    with sqlite3.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(pipeline_events)").fetchall()}
    assert "stage_id" in cols
    assert "artifact_ref" in cols

def test_migrations_are_idempotent(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    sqlite3.connect(db_path).close()
    apply_migrations(db_path)
    apply_migrations(db_path)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/migrations/test_schema.py -v
```

Expected: ImportError for `migrations.run_migrations`.

### Task A1.2: Implement migration runner

**Files:**
- Create: `migrations/__init__.py` (empty)
- Create: `migrations/0001_v4_schema.sql` (DDL from Task A0.1, Step 1)
- Create: `migrations/run_migrations.py`

- [ ] **Step 1: Create the SQL file**

Paste the full DDL from Task A0.1 Step 1 into `migrations/0001_v4_schema.sql`. Wrap the two `ALTER TABLE pipeline_events ADD COLUMN ...` statements in `try/except` at the runner level (SQLite has no `IF NOT EXISTS` for `ADD COLUMN` pre-3.35.5).

- [ ] **Step 2: Implement the runner**

```python
# migrations/run_migrations.py
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent

def apply_migrations(db_path: Path) -> None:
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS _migrations (filename TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        applied = {r[0] for r in conn.execute("SELECT filename FROM _migrations").fetchall()}
        for sql_file in sql_files:
            if sql_file.name in applied:
                continue
            sql = sql_file.read_text()
            for statement in _split_statements(sql):
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" in str(exc):
                        continue  # idempotency for ADD COLUMN
                    raise
            conn.execute("INSERT INTO _migrations(filename) VALUES (?)", (sql_file.name,))
        conn.commit()

def _split_statements(sql: str) -> list[str]:
    return [s.strip() for s in sql.split(";") if s.strip()]
```

- [ ] **Step 3: Run test to verify it passes**

```
pytest tests/migrations/test_schema.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 4: Commit**

```bash
git add migrations/ tests/migrations/
git commit -m "feat(storage): add v4 schema migrations for new tables and columns"
```

---

## Phase A2 — Metaflow Decomposition (Wave 1, this plan's biggest task block)

### Task A2.1: Define artifact contracts

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/contracts.py`
- Create: `tests/pipeline/test_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_contracts.py
from pipeline.contracts import (
    FrameSample, TriagedFrame, CornerDetection, NoveltyFilteredCandidate,
    Track, RectifiedCrop, ScoredCandidate, PreparedTrack, FusedCanonical,
    DedupGroup, FinalCard,
)

def test_frame_sample_required_fields():
    sample = FrameSample(frame_index=0, timestamp_ms=33, image_path="x.png", w=3840, h=2160)
    assert sample.frame_index == 0

def test_all_contracts_importable_and_have_repr():
    for cls in (
        FrameSample, TriagedFrame, CornerDetection, NoveltyFilteredCandidate,
        Track, RectifiedCrop, ScoredCandidate, PreparedTrack, FusedCanonical,
        DedupGroup, FinalCard,
    ):
        assert callable(cls)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pipeline/test_contracts.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement contracts**

The surface owner reads `src/card_capture/pipeline.py` to find every type currently passed between stages (`FrameSample`, `FramePacket`, `DetectionPacket`, `ScoredCandidate`, `TrackState`, `_PreparedTrack`, `QualityScore`, etc., per CLAUDE.md §5). For each, define a frozen Pydantic v2 model in `pipeline/contracts.py` that mirrors the existing type's public fields. Where the existing type uses `np.ndarray`, the contract uses `image_path: str` (artifact-friendly) or `tobytes()` payloads referenced by hash.

Acceptance: every artifact named in Contract 3 has a contract class. Existing tests in `tests/test_pipeline.py` still pass when the contracts are imported from `pipeline.contracts` and used internally — no behavior change.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pipeline/test_contracts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/__init__.py pipeline/contracts.py tests/pipeline/
git commit -m "feat(pipeline): define typed artifact contracts for Metaflow steps"
```

### Task A2.2: Scaffold the FlowSpec (no behavior yet)

**Files:**
- Create: `pipeline/card_capture_flow.py`
- Create: `tests/pipeline/test_flow_runs.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/pipeline/test_flow_runs.py
import subprocess
from pathlib import Path

FIXTURE_VIDEO = Path("tests/fixtures/tiny_clip.mov")

def test_flow_runs_to_completion_on_fixture(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    db = tmp_path / "cards.sqlite"
    result = subprocess.run(
        [
            "python", "-m", "pipeline.card_capture_flow", "run",
            "--video", str(FIXTURE_VIDEO),
            "--output-dir", str(out),
            "--db", str(db),
            "--detector", "fake",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "frames").exists()
    assert db.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pipeline/test_flow_runs.py -v
```

Expected: FAIL (module not found or fixture missing).

- [ ] **Step 3: Scaffold the flow**

```python
# pipeline/card_capture_flow.py
"""v4 pipeline orchestrated as a Metaflow FlowSpec.

Each @step is a thin call into pipeline.steps.<name>. The actual logic lives
in the step modules; this file is the orchestration spine and stays small.
"""
from __future__ import annotations

from metaflow import FlowSpec, Parameter, step

from pipeline.steps import (
    detect, novelty, track, refine, score, resolve, fuse, dedup, store,
)


class CardCaptureFlow(FlowSpec):
    video = Parameter("video", help="Path to source video", required=True)
    output_dir = Parameter("output-dir", help="Output directory", required=True)
    db = Parameter("db", help="SQLite database path", required=True)
    detector = Parameter("detector", help="Detector backend", default="docaligner")
    config_preset = Parameter("config-preset", default="balanced")

    @step
    def start(self):
        from pipeline.steps.start import init_run
        self.run_context = init_run(self.video, self.output_dir, self.db,
                                    self.detector, self.config_preset)
        self.next(self.detect)

    @step
    def detect(self):
        out = detect.run(self.run_context)
        self.frame_samples = out.frame_samples
        self.triaged_frames = out.triaged_frames
        self.corner_detections = out.corner_detections
        self.next(self.novelty)

    @step
    def novelty(self):
        out = novelty.run(self.run_context, self.corner_detections)
        self.novelty_filtered_candidates = out.candidates
        self.background_model = out.background_model
        self.next(self.track)

    @step
    def track(self):
        out = track.run(self.run_context, self.novelty_filtered_candidates)
        self.tracks = out.tracks
        self.session_resets = out.session_resets
        self.next(self.refine)

    @step
    def refine(self):
        out = refine.run(self.run_context, self.tracks)
        self.rectified_crops = out.rectified_crops
        self.next(self.score)

    @step
    def score(self):
        out = score.run(self.run_context, self.rectified_crops)
        self.scored_candidates = out.scored
        self.pruned_tracks = out.pruned
        self.next(self.resolve)

    @step
    def resolve(self):
        out = resolve.run(self.run_context, self.scored_candidates, self.pruned_tracks)
        self.prepared_tracks = out.prepared_tracks
        self.next(self.fuse_fanout)

    @step
    def fuse_fanout(self):
        self.fanout = list(self.prepared_tracks)
        self.next(self.fuse, foreach="fanout")

    @step
    def fuse(self):
        prepared_track = self.input
        out = fuse.run(self.run_context, prepared_track)
        self.fused_canonical = out
        self.next(self.fuse_join)

    @step
    def fuse_join(self, inputs):
        self.fused_canonicals = [inp.fused_canonical for inp in inputs]
        self.merge_artifacts(inputs, exclude=["fused_canonical"])
        self.next(self.dedup)

    @step
    def dedup(self):
        out = dedup.run(self.run_context, self.fused_canonicals)
        self.dedup_groups = out.dedup_groups
        self.dedup_distances = out.dedup_distances
        self.next(self.store)

    @step
    def store(self):
        out = store.run(self.run_context, self.dedup_groups, self.fused_canonicals)
        self.final_cards = out.final_cards
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CardCaptureFlow()
```

Target: this file stays ≤ 200 lines forever. If a step grows logic, it goes in `pipeline/steps/<name>.py`.

- [ ] **Step 4: Create empty step stubs**

```python
# pipeline/steps/__init__.py
from . import detect, novelty, track, refine, score, resolve, fuse, dedup, store  # noqa: F401
```

```python
# pipeline/steps/<name>.py — one per step
"""Step stub. Real logic lands in subsequent tasks."""
from dataclasses import dataclass

@dataclass
class Output:
    pass

def run(*args, **kwargs):
    raise NotImplementedError("step not yet implemented")
```

Create this stub for every step name listed in the FlowSpec.

- [ ] **Step 5: Commit (scaffold-only, test still fails)**

```bash
git add pipeline/ tests/pipeline/test_flow_runs.py
git commit -m "feat(pipeline): scaffold Metaflow FlowSpec and step stubs"
```

### Task A2.3: Implement `detect` step (Stages 1–3 wrap)

**Files:**
- Create: `pipeline/steps/detect.py` (replace stub)
- Create: `pipeline/steps/start.py`
- Modify: `src/card_capture/pipeline.py` (extract `_run_detection_subprocesses` or equivalent into a callable usable from this step)

- [ ] **Step 1: Identify the existing producer/consumer entrypoint**

Read `src/card_capture/pipeline.py` and locate where Stages 1–3 currently start (the producer subprocess + consumer subprocess that produce `DetectionPacket` items). Document its exact function signature and return shape in `pipeline/steps/detect.py` as a docstring.

- [ ] **Step 2: Write the failing test**

```python
# tests/pipeline/test_detect_step.py
from pipeline.steps.detect import run
from pipeline.steps.start import init_run

FIXTURE = "tests/fixtures/tiny_clip.mov"

def test_detect_returns_typed_output(tmp_path):
    ctx = init_run(FIXTURE, str(tmp_path), str(tmp_path / "db.sqlite"),
                   detector="fake", config_preset="balanced")
    out = run(ctx)
    assert hasattr(out, "frame_samples")
    assert hasattr(out, "triaged_frames")
    assert hasattr(out, "corner_detections")
    assert len(out.corner_detections) >= 0
```

- [ ] **Step 3: Implement `init_run` and `detect.run`**

Two new files. `init_run` builds a `RunContext` dataclass holding paths, config, db handles, telemetry sink. `detect.run` calls into the existing streaming subsystem (do NOT reimplement it — wrap it). Convert legacy `DetectionPacket` instances to `pipeline.contracts.CornerDetection` via a one-shot adapter.

Acceptance: the test passes against the `fake` detector fixture; for the real detector path, output length matches the legacy `pipeline.py` output count for the same video (verified manually once before harness exists).

- [ ] **Step 4: Run test**

```
pytest tests/pipeline/test_detect_step.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/steps/detect.py pipeline/steps/start.py tests/pipeline/test_detect_step.py
git commit -m "feat(pipeline): wrap Stages 1-3 streaming subsystem as detect step"
```

### Tasks A2.4 – A2.10: Implement remaining steps (`novelty`, `track`, `refine`, `score`, `resolve`, `fuse`, `dedup`, `store`)

Each follows the same pattern as A2.3:

**For each step:**

1. Read the corresponding section of `src/card_capture/pipeline.py` (the file CLAUDE.md §11 confirms is named `pipeline.py`) to identify the existing implementation block. Note CLAUDE.md §2's stage map — Stage 4 → novelty, Stage 5 → track, etc.
2. Write a step-level test with a hand-built input (using `pipeline.contracts` types) asserting the output type + a behavior invariant (e.g. for `novelty`, "candidates below `background_novelty_threshold` are dropped").
3. Run test — fails (NotImplementedError).
4. Extract the existing logic into `pipeline/steps/<name>.py`. Do not change algorithms. Wrap inputs/outputs in `pipeline.contracts` types. Keep the original implementation file callable from the step (delegate to the algorithm modules in `src/card_capture/`).
5. Run test — passes.
6. Commit.

**Hard rule for each:** no algorithm change. If you find yourself wanting to "improve while you're here," stop — that's Surface C's Wave 2 work.

**Step-specific notes:**

- **`novelty`**: calls `presence/background_novelty.py` unchanged. Output: `NoveltyFilteredCandidate[]` + `background_model` (the existing mean array, persisted as a numpy `.npy` artifact).
- **`track`**: calls `tracking/botsort_adapter.py` or `tracking/bytetrack_adapter.py` unchanged. Output: `Track[]` + `SessionReset[]`.
- **`refine`**: calls `gpu_refinement.py` or `cropper.py` unchanged. Output: `RectifiedCrop[]`.
- **`score`**: calls `scoring.py` unchanged. Output: `ScoredCandidate[]` + `pruned_tracks` (track IDs dropped by median quad-novelty).
- **`resolve`**: extracts the Front/Back-resolution block + pHash gating logic. Output: `PreparedTrack[]`.
- **`fuse`**: per-track, called via Metaflow `foreach`. Calls `fuser.py` + `fusion/median_fusion.py` + `fusion/foil_detection.py`. Input: one `PreparedTrack`; output: `FusedCanonical`.
- **`dedup`**: calls `deduplicator.py` unchanged. Output: `DedupGroup[]` + `dedup_distances` (sparse matrix of cosine/Hamming distances persisted as `.npz`).
- **`store`**: calls `storage.py` unchanged plus the new `pipeline_events.stage_id` / `artifact_ref` writes. Output: `FinalCard[]`.

Each task ends with its own commit. Eight tasks total (A2.4–A2.10 inclusive + A2.11 if `store` needs its own task).

### Task A2.11: Full-flow harness verification (the gate)

**Files:** none new.

- [ ] **Step 1: Run the full flow on a fixture**

```
python -m pipeline.card_capture_flow run \
  --video tests/fixtures/tiny_clip.mov \
  --output-dir /tmp/flow_out \
  --db /tmp/flow_out/cards.sqlite \
  --detector fake
```

Expected: completes without error; `final_cards` artifact non-empty.

- [ ] **Step 2: Run the regression harness (requires Surface D delivered)**

```
card-capture harness run --baseline baseline_v4.1 --videos tests/fixtures/golden_subset.json
```

Expected: report shows 0% delta on card recall, card precision, side accuracy, dedup accuracy, image quality. Per-video table: every row in noise floor.

If any metric regresses: stop. Find the step that introduced the delta (binary-search the steps by replaying with last-known-good intermediates). Fix the offending step; re-run harness. No merge until 0% delta.

- [ ] **Step 3: Tag the verified refactor**

```bash
git tag -a v4-pipeline-decomposed -m "Metaflow refactor verified 0% delta on baseline_v4.1"
```

- [ ] **Step 4: Mark the old pipeline.py as deprecated**

```python
# src/card_capture/pipeline.py — at the top, add:
"""DEPRECATED in v4. Pipeline orchestration moved to pipeline/card_capture_flow.py.

This module is kept temporarily as a backwards-compatibility shim for code
that still imports VideoProcessor directly. Will be deleted after Surface A's
final cleanup task (see plan).
"""
import warnings
warnings.warn(
    "src.card_capture.pipeline is deprecated; use pipeline.card_capture_flow",
    DeprecationWarning, stacklevel=2,
)
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline.py
git commit -m "refactor(pipeline): deprecate legacy pipeline.py; flow is authoritative"
```

---

## Phase A3 — FastAPI Service Layer + SSE (Wave 1, parallel with Phase A2)

This phase can be developed alongside A2 (different files). The pipeline_runner depends on `CardCaptureFlow` existing as a class; until then, it can be stubbed.

### Task A3.1: API Pydantic schemas

**Files:**
- Create: `app/__init__.py` (empty)
- Create: `app/schemas/__init__.py` (empty)
- Create: `app/schemas/v1.py`
- Create: `tests/app/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_schemas.py
from app.schemas.v1 import (
    Video, VideoCreate, Run, RunDetail, Card, CardFilter,
    LabelFB, LabelTruth, DedupCluster,
    RegressionBaseline, RegressionRun, RegressionCompare,
    ConfigPreset, SSEEvent,
)

def test_all_v1_schemas_importable_and_have_examples():
    # each schema declares model_config["json_schema_extra"]["example"]
    for cls in (Video, VideoCreate, Run, RunDetail, Card, LabelFB, LabelTruth,
                DedupCluster, RegressionBaseline, RegressionRun, ConfigPreset, SSEEvent):
        example = cls.model_config.get("json_schema_extra", {}).get("example")
        assert example is not None, f"{cls.__name__} missing example in model_config"
        cls.model_validate(example)
```

- [ ] **Step 2: Run — fails on import**

```
pytest tests/app/test_schemas.py -v
```

- [ ] **Step 3: Implement schemas**

For every request/response shape in `docs/contracts/v1-api.md`, write a Pydantic v2 model in `app/schemas/v1.py`. Each model includes a `model_config = ConfigDict(json_schema_extra={"example": {...}})` block with a worked example matching Contract 2.

The full list (one model per route input/output):

```
Video, VideoCreate, VideoList,
Run, RunDetail, RunCreateRequest, RunSummary, RunCardSummary,
RunEvent, RunTelemetry, RunRejection, RunHardCase,
Card, CardDetail, CardFilter, CardBulkAction,
LabelTruth, LabelTruthExpectedCard, LabelFB, LabelFBNext, DedupCluster,
TrainingDataset, TrainingJob, TrainingRetrainRequest,
RegressionBaseline, RegressionRun, RegressionRunRequest, RegressionCompare,
RegressionPerVideoDelta, RegressionMetric,
ConfigPreset, ConfigPlayground, ConfigPlaygroundSliderUpdate,
SSEEvent, SSEStageStarted, SSEStageProgress, SSEStageCompleted, SSEArtifactPersisted, SSERunCompleted, SSERunFailed
```

- [ ] **Step 4: Run test — passes**

```
pytest tests/app/test_schemas.py -v
```

- [ ] **Step 5: Commit**

```bash
git add app/ tests/app/test_schemas.py
git commit -m "feat(app): v1 API Pydantic schemas with worked examples"
```

### Task A3.2: API contract conformance test (failing now, passes when routes wired)

**Files:**
- Create: `tests/app/test_api_contract.py`

- [ ] **Step 1: Write the test**

```python
# tests/app/test_api_contract.py
from fastapi.testclient import TestClient

from app.main import create_app

ROUTES_REQUIRED = [
    ("GET", "/api/v1/videos"),
    ("POST", "/api/v1/videos"),
    ("GET", "/api/v1/runs"),
    ("GET", "/api/v1/cards"),
    ("GET", "/api/v1/label/fb/next"),
    ("GET", "/api/v1/training/datasets"),
    ("GET", "/api/v1/regression/baselines"),
    ("GET", "/api/v1/config/presets"),
]

def _routes(client):
    return {(r["methods"] & {"GET","POST","PUT","PATCH","DELETE"}, r["path"])
            for r in client.app.routes if hasattr(r, "methods")}

def test_all_required_routes_registered():
    client = TestClient(create_app())
    registered = {(tuple(r.methods)[0], r.path) for r in client.app.routes if hasattr(r, "methods")}
    for method, path in ROUTES_REQUIRED:
        assert (method, path) in registered, f"missing {method} {path}"

def test_openapi_includes_v1_routes():
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for method, path in ROUTES_REQUIRED:
        assert path in paths
        assert method.lower() in paths[path]
```

- [ ] **Step 2: Run — fails on import**

```
pytest tests/app/test_api_contract.py -v
```

### Task A3.3: FastAPI app + route stubs

**Files:**
- Create: `app/main.py`
- Create: `app/api/{videos,runs,cards,label,training,regression,config,events}.py`

- [ ] **Step 1: Implement app factory**

```python
# app/main.py
from fastapi import FastAPI

from app.api import videos, runs, cards, label, training, regression, config, events


def create_app() -> FastAPI:
    app = FastAPI(title="Card Capture v4", version="0.1.0")
    app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
    app.include_router(runs.router, prefix="/api/v1/runs", tags=["runs"])
    app.include_router(cards.router, prefix="/api/v1/cards", tags=["cards"])
    app.include_router(label.router, prefix="/api/v1/label", tags=["label"])
    app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
    app.include_router(regression.router, prefix="/api/v1/regression", tags=["regression"])
    app.include_router(config.router, prefix="/api/v1/config", tags=["config"])
    app.include_router(events.router, prefix="/events", tags=["events"])
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
```

- [ ] **Step 2: Stub each router**

For each router file, define:

```python
# app/api/videos.py — pattern, one per file
from fastapi import APIRouter, HTTPException

from app.schemas.v1 import Video, VideoCreate

router = APIRouter()


@router.get("", response_model=list[Video])
def list_videos():
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("", response_model=Video, status_code=201)
def create_video(payload: VideoCreate):
    raise HTTPException(status_code=501, detail="not implemented yet")
```

Every route from Contract 2 gets a stub returning `501`. The routes exist; the contract conformance test passes; later tasks fill in real handlers.

- [ ] **Step 3: Run conformance test**

```
pytest tests/app/test_api_contract.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add app/main.py app/api/
git commit -m "feat(app): FastAPI app factory and Contract 2 route stubs"
```

### Task A3.4: SSE channel — event ordering test, then implementation

**Files:**
- Create: `tests/app/test_sse_events.py`
- Create: `app/services/event_bus.py`
- Modify: `app/api/events.py`

- [ ] **Step 1: Failing test**

```python
# tests/app/test_sse_events.py
from fastapi.testclient import TestClient
import httpx
import asyncio

from app.main import create_app
from app.services.event_bus import EventBus, Event

async def _drain_until(client: httpx.AsyncClient, run_id: str, want: str):
    async with client.stream("GET", f"/events/{run_id}") as r:
        async for line in r.aiter_lines():
            if want in line:
                return line
    raise AssertionError(f"event {want} not seen")

def test_sse_emits_stage_progress_in_order(monkeypatch):
    app = create_app()
    bus: EventBus = app.state.event_bus
    client = TestClient(app)
    seen = []
    with client.stream("GET", "/events/run_42") as response:
        bus.emit("run_42", Event(name="stage_started", stage="detect"))
        bus.emit("run_42", Event(name="stage_completed", stage="detect"))
        bus.emit("run_42", Event(name="run_completed"))
        for line in response.iter_lines():
            seen.append(line)
            if "run_completed" in line:
                break
    names = [l for l in seen if l.startswith("event:")]
    assert names == ["event: stage_started", "event: stage_completed", "event: run_completed"]
```

- [ ] **Step 2: Run — fails**

```
pytest tests/app/test_sse_events.py -v
```

- [ ] **Step 3: Implement event bus**

```python
# app/services/event_bus.py
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass
class Event:
    name: str
    payload: dict | None = None
    stage: str | None = None


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues[run_id].append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self._queues[run_id].remove(q)

    def emit(self, run_id: str, event: Event) -> None:
        for q in self._queues.get(run_id, []):
            q.put_nowait(event)
```

- [ ] **Step 4: Wire SSE endpoint**

```python
# app/api/events.py
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
import json

from app.services.event_bus import EventBus, Event

router = APIRouter()


async def _stream(request: Request, bus: EventBus, run_id: str):
    q = bus.subscribe(run_id)
    try:
        while True:
            if await request.is_disconnected():
                break
            event: Event = await q.get()
            yield {
                "event": event.name,
                "data": json.dumps({"stage": event.stage, "payload": event.payload}),
            }
            if event.name in {"run_completed", "run_failed"}:
                break
    finally:
        bus.unsubscribe(run_id, q)


@router.get("/{run_id}")
async def events(request: Request, run_id: str):
    bus: EventBus = request.app.state.event_bus
    return EventSourceResponse(_stream(request, bus, run_id))
```

In `app/main.py::create_app`, add `app.state.event_bus = EventBus()` before `include_router`.

- [ ] **Step 5: Run test — passes**

```
pytest tests/app/test_sse_events.py -v
```

- [ ] **Step 6: Commit**

```bash
git add app/services/event_bus.py app/api/events.py app/main.py tests/app/test_sse_events.py
git commit -m "feat(app): SSE channel with event bus for per-run progress"
```

### Task A3.5: PipelineRunner service — bridges Metaflow ↔ SSE ↔ HTTP

**Files:**
- Create: `app/services/pipeline_runner.py`
- Create: `tests/app/test_pipeline_runner.py`
- Modify: `app/api/runs.py`

- [ ] **Step 1: Failing test**

```python
# tests/app/test_pipeline_runner.py
import asyncio
from unittest.mock import MagicMock

from app.services.pipeline_runner import PipelineRunner
from app.services.event_bus import EventBus

def test_runner_emits_stage_events_for_each_step():
    bus = EventBus()
    events_seen = []

    async def collector():
        q = bus.subscribe("run_1")
        for _ in range(4):
            events_seen.append(await q.get())

    runner = PipelineRunner(bus=bus, flow_cls=_FakeFlow)
    asyncio.run(asyncio.wait_for(
        asyncio.gather(collector(), runner.run_async("run_1", video="x.mov", output_dir="/tmp/o", db="/tmp/db.sqlite")),
        timeout=5,
    ))
    names = [e.name for e in events_seen]
    assert "stage_started" in names
    assert "stage_completed" in names
    assert "run_completed" in names
```

Define `_FakeFlow` in the same test file as a stub that yields a fixed sequence of `(stage_name, status)` pairs to exercise the bus.

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement runner**

```python
# app/services/pipeline_runner.py
"""Async wrapper around CardCaptureFlow that emits SSE events per step."""
import asyncio
from pathlib import Path
from typing import Type

from app.services.event_bus import Event, EventBus


class PipelineRunner:
    def __init__(self, bus: EventBus, flow_cls: Type) -> None:
        self.bus = bus
        self.flow_cls = flow_cls

    async def run_async(self, run_id: str, *, video: str, output_dir: str,
                       db: str, detector: str = "docaligner",
                       config_preset: str = "balanced") -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._run_blocking, run_id, video, output_dir, db, detector, config_preset
        )

    def _run_blocking(self, run_id: str, video: str, output_dir: str,
                     db: str, detector: str, config_preset: str) -> None:
        # In production, instrument the Metaflow flow via @step decorators that
        # call self._emit; for now, we approximate by parsing the metaflow run log
        # OR by passing the event_bus through Parameters and emitting from within
        # each step module. The latter is the chosen design — see plan A3.6.
        from pipeline.card_capture_flow import CardCaptureFlow  # local import to keep tests fast

        # Use Metaflow's API to invoke the flow programmatically.
        # The flow modules read `EVENT_BUS_RUN_ID` env var to emit events.
        import os
        os.environ["EVENT_BUS_RUN_ID"] = run_id
        os.environ["EVENT_BUS_INPROC"] = "1"  # in-process emit (not pickle-safe across machines)

        # Register this bus globally so step modules can find it.
        from app.services import _event_bus_registry
        _event_bus_registry.set(run_id, self.bus)

        try:
            self.bus.emit(run_id, Event(name="run_started"))
            # Metaflow programmatic invocation:
            from metaflow import Runner
            with Runner("pipeline/card_capture_flow.py").run(
                video=video, output_dir=output_dir, db=db, detector=detector,
                config_preset=config_preset,
            ) as running:
                _ = running  # logs already emitted via step instrumentation
            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            _event_bus_registry.clear(run_id)
```

- [ ] **Step 4: Module-level event-bus registry**

```python
# app/services/_event_bus_registry.py
"""Thread-safe registry mapping run_id -> EventBus, used by Metaflow steps
to emit progress events back to the HTTP layer."""
from threading import Lock

from app.services.event_bus import EventBus

_lock = Lock()
_buses: dict[str, EventBus] = {}

def set(run_id: str, bus: EventBus) -> None:
    with _lock:
        _buses[run_id] = bus

def get(run_id: str) -> EventBus | None:
    with _lock:
        return _buses.get(run_id)

def clear(run_id: str) -> None:
    with _lock:
        _buses.pop(run_id, None)
```

- [ ] **Step 5: Step modules emit events**

In each `pipeline/steps/<name>.py`, at the start and end of `run(...)`, call:

```python
import os
from app.services import _event_bus_registry
from app.services.event_bus import Event

def _emit(name: str, stage: str, payload: dict | None = None) -> None:
    run_id = os.environ.get("EVENT_BUS_RUN_ID")
    if not run_id:
        return
    bus = _event_bus_registry.get(run_id)
    if bus:
        bus.emit(run_id, Event(name=name, stage=stage, payload=payload or {}))
```

Then in each `run`:

```python
def run(ctx, *args, **kwargs):
    _emit("stage_started", stage="<name>")
    out = <do work>
    _emit("stage_completed", stage="<name>", payload={"n_items": len(out.<artifact>)})
    return out
```

- [ ] **Step 6: Wire `POST /api/v1/videos/{id}/process`**

```python
# app/api/videos.py — add to existing router
import uuid
from fastapi import APIRouter, BackgroundTasks, Request

@router.post("/{video_id}/process", status_code=202)
def start_run(video_id: str, request: Request, bg: BackgroundTasks):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    runner = PipelineRunner(bus=request.app.state.event_bus, flow_cls=None)
    bg.add_task(
        lambda: __import__("asyncio").run(
            runner.run_async(run_id, video=f"/var/videos/{video_id}",
                             output_dir=f"/var/runs/{run_id}",
                             db="cards.sqlite")
        )
    )
    return {"run_id": run_id, "status": "started"}
```

(Paths illustrative; final paths come from Surface A's path-resolution helper.)

- [ ] **Step 7: Run test — passes**

```
pytest tests/app/test_pipeline_runner.py -v
```

- [ ] **Step 8: Commit**

```bash
git add app/services/ app/api/videos.py tests/app/test_pipeline_runner.py
git commit -m "feat(app): PipelineRunner with SSE event emission per step"
```

### Task A3.6: CLI parity — `card-capture process` calls the new runner

**Files:**
- Modify: `src/card_capture/cli.py`

- [ ] **Step 1: Read the existing CLI**

Open `src/card_capture/cli.py`. Find the `process` subcommand handler and its current call into `pipeline.VideoProcessor`.

- [ ] **Step 2: Write a CLI regression test**

```python
# tests/cli/test_process_subcommand.py
import subprocess
from pathlib import Path

FIXTURE = "tests/fixtures/tiny_clip.mov"

def test_process_command_still_produces_cards(tmp_path: Path):
    out = tmp_path / "out"; out.mkdir()
    db = tmp_path / "cards.sqlite"
    result = subprocess.run(
        ["card-capture", "process", FIXTURE,
         "--output-dir", str(out), "--db", str(db),
         "--detector", "fake"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert db.exists()
```

- [ ] **Step 3: Refactor CLI to call `PipelineRunner`**

Replace the `VideoProcessor` invocation with a call to `app.services.pipeline_runner.PipelineRunner.run_async` via `asyncio.run`. No SSE in the CLI path; the event-bus subscriber for CLI is a stderr printer.

```python
# illustrative; integrate into existing cli.py structure
from app.services.event_bus import EventBus
from app.services.pipeline_runner import PipelineRunner

def _stderr_subscriber(bus: EventBus, run_id: str):
    import asyncio, sys
    async def consume():
        q = bus.subscribe(run_id)
        while True:
            ev = await q.get()
            print(f"[{ev.stage or '-'}] {ev.name}", file=sys.stderr)
            if ev.name in {"run_completed","run_failed"}:
                return
    return consume

def cmd_process(args):
    bus = EventBus()
    run_id = "cli_run"
    runner = PipelineRunner(bus=bus, flow_cls=None)
    import asyncio
    async def main():
        consumer = asyncio.create_task(_stderr_subscriber(bus, run_id)())
        await runner.run_async(run_id, video=args.video, output_dir=args.output_dir,
                              db=args.db, detector=args.detector)
        await consumer
    asyncio.run(main())
```

- [ ] **Step 4: Run CLI test — passes**

```
pytest tests/cli/test_process_subcommand.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/cli.py tests/cli/
git commit -m "refactor(cli): process subcommand uses PipelineRunner"
```

### Task A3.7: Full Wave-1 harness gate

**Files:** none.

- [ ] **Step 1: Run harness end-to-end**

```
card-capture harness run --baseline baseline_v4.1
```

Expected: 0% delta on the full golden set after Surface D completes the golden-set load. If <15 videos labeled at this point, run with `--videos <subset>` against whatever is labeled.

- [ ] **Step 2: Confirm route stubs do not break harness**

(They cannot, because the harness invokes the flow directly, not the API. But verify by running once with the API running and once with it off.)

- [ ] **Step 3: Surface-A Wave 1 sign-off**

Tag:

```bash
git tag -a v4-surface-a-wave1-complete -m "Surface A Wave 1: 0% delta on baseline_v4.1"
```

Notify other surface owners; Wave 2 work unblocked.

---

## Phase A4 — Wave 3: Apple-Silicon Fast Paths (re-plan when Wave 2 winds down)

**Status:** Outline only. Surface A's owner re-runs `superpowers:writing-plans` on this section when Wave 3 begins (Wave 2 algorithmic upgrades shipped + harness green).

Wave 3 tasks (each is its own future task with TDD steps):

- **A4.1 Feature detection at startup.** `app/capabilities.py` probes for macOS + ANE + VideoToolbox + vImage availability; logs which fast paths are active. Cross-platform fallback unchanged.
- **A4.2 CoreML YOLO26-OBB.** Convert YOLO26-OBB → CoreML. New `detectors_coreml.py`. Feature-detected wrap around existing PyTorch path. Acceptance: harness 0% delta + ≥2× detector throughput on M-series.
- **A4.3 VideoToolbox decoder.** New `ingestion_videotoolbox.py` using PyObjC `AVFoundation`. Wraps the existing OpenCV/decord decoder selection. Acceptance: harness 0% delta + faster decode wall-clock on golden set.
- **A4.4 vImage perspective warp.** New `gpu_refinement_vimage.py` using PyObjC `vImage`. Feature-detected wrap around Kornia. Acceptance: harness 0% delta + faster warp wall-clock.
- **A4.5 Cross-platform CI matrix.** GitHub Actions: macOS-15 runs fast paths; Linux runs fallback. Harness runs on both.

Each task ends with harness gate + commit.

---

## Self-Review (post-write)

- **Spec coverage:** Storage extensions (§2.2 C1) → Phase A1; service-layer API (§2.2 C2) → Phase A3 Tasks A3.1–A3.4; Metaflow artifact contract (§2.2 C3) → Phase A2; Apple paths (§4.1 Wave 3) → Phase A4; CLI parity (§4.1 Wave 1) → Task A3.6.
- **Placeholders:** none — all steps either show concrete code or reference a specific algorithm module to wrap.
- **Type consistency:** `EventBus`, `Event`, `PipelineRunner`, `CardCaptureFlow`, `init_run`, `RunContext` are referenced consistently across tasks; contract types (`FrameSample`, `Track`, etc.) defined in A2.1 and consumed by A2.3+.

---

## Execution Handoff

Plan complete at `docs/superpowers/plans/2026-05-12-v4-surface-a-orchestration.md`.

This plan is one of four (A/B/C/D) produced from the v4 architecture spec. Surface A's tasks are ready for dispatch via `superpowers:subagent-driven-development`: contract-drafting tasks (A0.*) first, in sequence, then storage migrations (A1.*) and Metaflow decomposition (A2.*) and service layer (A3.*) can run with internal parallelism within Surface A (different files).

Wave 3 (A4.*) re-plans via this same skill when Wave 2 winds down.
