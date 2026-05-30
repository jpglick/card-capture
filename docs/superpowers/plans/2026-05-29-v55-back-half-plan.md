# V5.5 Back-Half Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take `LocalPipelineRuntime` from "orchestrates but produces zero cards" to full V4 parity with cards visible in the web UI, end-to-end tested and per-stage audited.

**Architecture:** Verbatim ports of the V4 `pipeline/steps/{score,resolve,fuse,dedup,store}.py` step bodies into `src/card_capture/pipeline/stages/*.py` and a rewrite of `stages/refine.py` to match V4 `refine.py`. All ports adapt from V4's `(ctx, prev_output) -> Output` dataclass shape to v5.5's `(state: dict, *, telemetry) -> None` mutate-in-place shape. Rectified crops live as `np.ndarray` in `state` rather than spilling to disk (v5.5 in-memory mandate); the only filesystem writes happen at the `store` boundary. All DB writes go through `CardsRepository` (no raw SQL outside `card_capture.data`).

**Tech Stack:** Python 3.9 / pytest / OpenCV / Kornia / NumPy / SQLite / FastAPI / SSE / supervision (ByteTrack/BoT-SORT) / ultralytics (YOLOv8-OBB) / DINOv2 ReID.

**Companion spec:** [`2026-05-29-v55-back-half-spec.md`](./2026-05-29-v55-back-half-spec.md) holds the design rationale, risk register, sizing analysis, and open questions. This plan is the executable companion.

**Branch:** `feat/v55-back-half-wiring` (off `fix/ui-v55-unified-runtime`).

---

## File Structure

### Files created

| Path | Responsibility | Created in |
|---|---|---|
| `src/card_capture/data/repositories/cards.py` (extended) | Repository write methods for back-half persistence | Phase 2 |
| `src/card_capture/ml/models/dino_embedder_array.py` | Array variant of `DinoEmbedder.embed_image` | Phase 3 |
| `src/card_capture/ml/inference/fb_predict_array.py` | Array variant of `FBPredictor.predict` | Phase 3 |
| `src/card_capture/ml/embeddings_array.py` | Array variant of `compute_reid_embedding` | Phase 3 |
| `tests/data/test_cards_repository_writes.py` | Unit tests for new repo methods | Phase 2 |
| `tests/ml/test_dino_embedder_array.py` | Parity test embed_array vs embed_image | Phase 3 |
| `tests/ml/test_fb_predict_array.py` | Parity test predict_array vs predict | Phase 3 |
| `tests/ml/test_reid_embeddings_array.py` | Parity test array vs path variant | Phase 3 |
| `tests/pipeline/stages/__init__.py` | Empty marker | Phase 4 |
| `tests/pipeline/stages/test_refine_stage.py` | refine stage unit tests | Phase 4 |
| `tests/pipeline/stages/test_score_stage.py` | score stage unit tests | Phase 5 |
| `tests/pipeline/stages/test_resolve_stage.py` | resolve stage unit tests | Phase 6 |
| `tests/pipeline/stages/test_fuse_stage.py` | fuse stage unit tests | Phase 7 |
| `tests/pipeline/stages/test_dedup_stage.py` | dedup stage unit tests | Phase 8 |
| `tests/pipeline/stages/test_store_stage.py` | store stage unit tests | Phase 9 |
| `tests/pipeline/conftest.py` | Synthetic two-card MOV fixture factory | Phase 10 |
| `tests/pipeline/test_back_half_e2e.py` | End-to-end run on synthetic fixture | Phase 10 |
| `tests/app/test_run_to_cards.py` | UI integration: cards endpoint + SSE | Phase 12 |
| `docs/superpowers/audits/2026-05-29-v55-back-half-audit.md` | Per-stage V4-vs-V5.5 audit | Phase 13 |
| `docs/superpowers/plans/v5-5/back-half-baseline.md` | Manual golden-set re-run metrics | Phase 14 |

### Files modified

| Path | What changes | In phase |
|---|---|---|
| `src/card_capture/config.py` | Add 10 new fields to `PipelineConfig` dataclass | Phase 1 |
| `app/services/pipeline_runner.py` | Read `PipelineConfig`, merge into `request.config` | Phase 1 |
| `src/card_capture/cli.py` | Same — pass `PipelineConfig` fields through `request.config` | Phase 1 |
| `app/worker_core.py` | Same | Phase 1 |
| `app/services/training_service.py` | Same | Phase 1 |
| `src/card_capture/data/sql_queries.py` | Add 8 new SQL constants | Phase 2 |
| `src/card_capture/ml/models/dino_embedder.py` | Refactor `embed_image` to call new `embed_array` internally | Phase 3 |
| `src/card_capture/ml/inference/fb_predict.py` | Same pattern | Phase 3 |
| `src/card_capture/ml/embeddings.py` | Same pattern | Phase 3 |
| `src/card_capture/pipeline/stages/track.py` | Convert `List[TrackState]` to V4-shape `tracks_data: List[Dict]` | Phase 4 |
| `src/card_capture/pipeline/stages/refine.py` | Full rewrite — V4 refine port | Phase 4 |
| `src/card_capture/pipeline/stages/score.py` | Full rewrite — V4 score port | Phase 5 |
| `src/card_capture/pipeline/stages/resolve.py` | Full rewrite — V4 resolve port | Phase 6 |
| `src/card_capture/pipeline/stages/fuse.py` | Full rewrite — V4 fuse port (in-process loop) | Phase 7 |
| `src/card_capture/pipeline/stages/dedup.py` | Full rewrite — V4 dedup port | Phase 8 |
| `src/card_capture/pipeline/stages/store.py` | Full rewrite — V4 store port via repositories | Phase 9 |
| `src/card_capture/pipeline/runtime_local.py` | Inject `db_path` and `output_root` (str→Path) into `state` for stages to consume | Phase 4 |
| `src/card_capture/pipeline/telemetry.py` | Add `progress(stage, pct, detail)` to Protocol + Noop + InMemory | Phase 11 |
| `app/services/pipeline_telemetry.py` | Implement `progress` on `EventBusTelemetry` (emits `stage_progress`) | Phase 11 |
| `tests/test_unified_runtime.py` | Add assertion `len(result.manifest.cards) > 0` | Phase 10 |
| `CLAUDE.md` | Update Known Weaknesses → mark stages wired | Phase 14 |

---

## Phase 1 — Codify config keys and thread `PipelineConfig` through the runtime

The back-half stages need ~10 new configurable knobs that V4 read from `RunContext`. We add them to `PipelineConfig` (which already exists and is loaded from `card_capture_config.json`) and thread the merged config into `PipelineRunRequest.config` so stages can read them.

### Task 1.1: Add new fields to `PipelineConfig`

**Files:**
- Modify: `src/card_capture/config.py`
- Test: `tests/test_config_back_half_fields.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_back_half_fields.py`:

```python
"""Phase 1 — PipelineConfig has all back-half fields with V4 defaults."""
from card_capture.config import PipelineConfig


def test_pipeline_config_has_novelty_floor():
    assert PipelineConfig().novelty_floor == 0.30


def test_pipeline_config_has_track_confidence_floor():
    assert PipelineConfig().track_confidence_floor == 0.60


def test_pipeline_config_has_stand_novelty_max():
    assert PipelineConfig().stand_novelty_max == 0.35


def test_pipeline_config_has_stand_sharpness_max():
    assert PipelineConfig().stand_sharpness_max == 0.30


def test_pipeline_config_has_foil_threshold():
    assert PipelineConfig().foil_threshold == 50.0


def test_pipeline_config_has_enable_foil_aware_fusion():
    assert PipelineConfig().enable_foil_aware_fusion is True


def test_pipeline_config_has_use_fb_classifier():
    assert PipelineConfig().use_fb_classifier is True


def test_pipeline_config_has_laplacian_scan_stride():
    assert PipelineConfig().laplacian_scan_stride == 5


def test_pipeline_config_has_max_corner_gap_frames():
    assert PipelineConfig().max_corner_gap_frames == 30


def test_pipeline_config_has_corner_refinement():
    assert PipelineConfig().corner_refinement is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_back_half_fields.py -v`
Expected: 10 FAILED with `AttributeError: 'PipelineConfig' object has no attribute 'novelty_floor'` (or similar for each field).

- [ ] **Step 3: Add the fields to `PipelineConfig`**

In `src/card_capture/config.py`, add these fields to the `@dataclass class PipelineConfig:` definition (place after the existing fields, before any methods):

```python
    # ------------------------------------------------------------------
    # V5.5 back-half stage knobs (see docs/.../2026-05-29-v55-back-half-spec.md §4.8)
    # ------------------------------------------------------------------
    novelty_floor: float = 0.30
    track_confidence_floor: float = 0.60
    stand_novelty_max: float = 0.35
    stand_sharpness_max: float = 0.30
    foil_threshold: float = 50.0
    enable_foil_aware_fusion: bool = True
    use_fb_classifier: bool = True
    laplacian_scan_stride: int = 5
    max_corner_gap_frames: int = 30
    corner_refinement: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_back_half_fields.py -v`
Expected: 10 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/config.py tests/test_config_back_half_fields.py
git commit -m "feat(v55-stages): add back-half config fields to PipelineConfig

Adds 10 fields V4 read from RunContext, with V4 defaults preserved.
Stages and runtime callers will read them via request.config in
later tasks of Phase 1."
```

### Task 1.2: Helper to convert `PipelineConfig` → dict for `request.config`

**Files:**
- Modify: `src/card_capture/config.py`
- Test: `tests/test_config_to_request_dict.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_to_request_dict.py`:

```python
"""Phase 1 — PipelineConfig.to_request_config() returns a JSON-safe dict
of all knobs stages consume."""
import json

from card_capture.config import PipelineConfig


def test_to_request_config_includes_all_back_half_fields():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    assert d["novelty_floor"] == 0.30
    assert d["track_confidence_floor"] == 0.60
    assert d["stand_novelty_max"] == 0.35
    assert d["stand_sharpness_max"] == 0.30
    assert d["foil_threshold"] == 50.0
    assert d["enable_foil_aware_fusion"] is True
    assert d["use_fb_classifier"] is True
    assert d["laplacian_scan_stride"] == 5
    assert d["max_corner_gap_frames"] == 30
    assert d["corner_refinement"] is False


def test_to_request_config_includes_detector_and_device():
    cfg = PipelineConfig()
    cfg.detector = "docaligner"
    cfg.device = "cpu"
    d = cfg.to_request_config()
    assert d["detector"] == "docaligner"
    assert d["device"] == "cpu"


def test_to_request_config_is_json_serializable():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["novelty_floor"] == 0.30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_to_request_dict.py -v`
Expected: 3 FAILED with `AttributeError: 'PipelineConfig' object has no attribute 'to_request_config'`.

- [ ] **Step 3: Implement `to_request_config` on `PipelineConfig`**

In `src/card_capture/config.py`, add this method to `PipelineConfig` (after the dataclass field declarations):

```python
    def to_request_config(self) -> dict:
        """Return the subset of fields back-half stages consume via
        ``PipelineRunRequest.config``. Stays JSON-serializable so the
        request can cross the runtime/transport boundary unchanged."""
        return {
            # Detector / device / detection
            "detector": self.detector,
            "device": self.device,
            "corner_confidence": self.corner_confidence,
            "detection_width": self.detection_width,
            # Tracker
            "tracker_backend": self.tracker_backend,
            "min_track_length": self.min_track_length,
            # Refine / fusion
            "fusion_target_frames": self.fusion_target_frames,
            "rotate_180": self.rotate_180,
            "use_kornia": True,
            "kornia_device": self.device,
            # New back-half knobs
            "novelty_floor": self.novelty_floor,
            "track_confidence_floor": self.track_confidence_floor,
            "stand_novelty_max": self.stand_novelty_max,
            "stand_sharpness_max": self.stand_sharpness_max,
            "foil_threshold": self.foil_threshold,
            "enable_foil_aware_fusion": self.enable_foil_aware_fusion,
            "use_fb_classifier": self.use_fb_classifier,
            "laplacian_scan_stride": self.laplacian_scan_stride,
            "max_corner_gap_frames": self.max_corner_gap_frames,
            "corner_refinement": self.corner_refinement,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_to_request_dict.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/config.py tests/test_config_to_request_dict.py
git commit -m "feat(v55-stages): PipelineConfig.to_request_config()

JSON-safe dict of every knob the back-half stages consume. Runtime
callers (pipeline_runner, cli, worker_core, training_service) thread
this into PipelineRunRequest.config in the next task."
```

### Task 1.3: Thread `to_request_config()` through `pipeline_runner._run_unified_inprocess`

**Files:**
- Modify: `app/services/pipeline_runner.py:155-178` (the `_run_unified_inprocess` body)
- Test: `tests/app/test_pipeline_runner_threads_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_pipeline_runner_threads_config.py`:

```python
"""Phase 1 — pipeline_runner merges PipelineConfig into request.config."""
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.event_bus import EventBus
from app.services.pipeline_runner import PipelineRunner


def test_unified_inprocess_passes_config_dict(tmp_path):
    captured: dict = {}

    def fake_runtime_run(self, request):
        captured["config"] = dict(request.config)
        result = MagicMock()
        result.manifest.contract_violations = []
        return result

    with patch(
        "card_capture.pipeline.runtime_local.LocalPipelineRuntime.run",
        new=fake_runtime_run,
    ):
        bus = EventBus()
        db = tmp_path / "cards.sqlite"
        db.touch()
        runner = PipelineRunner(bus=bus, flow_cls=None, db_path=db)
        runner._run_unified_inprocess(
            run_id="r1",
            video_id=42,
            video=str(tmp_path / "v.mov"),
            output_dir="out",
            db=str(db),
            detector="fake",
            config_preset="balanced",
        )

    assert captured["config"]["novelty_floor"] == 0.30
    assert captured["config"]["foil_threshold"] == 50.0
    assert captured["config"]["use_fb_classifier"] is True
    assert captured["config"]["detector"] == "fake"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/app/test_pipeline_runner_threads_config.py -v`
Expected: FAILED with `KeyError: 'novelty_floor'` (current runner only passes `{"detector": detector}`).

- [ ] **Step 3: Modify `_run_unified_inprocess`**

In `app/services/pipeline_runner.py`, change the `request = PipelineRunRequest(...)` block (lines 209–218 of current file) to merge `PipelineConfig`:

```python
        # Merge full PipelineConfig defaults so back-half stages read knobs
        # like novelty_floor / foil_threshold from request.config. Caller
        # overrides win (detector arg, etc.).
        from card_capture.config import load_config
        config = load_config(Path(_REPO_ROOT) / "card_capture_config.json")
        request_config = config.to_request_config()
        request_config["detector"] = detector

        runtime = LocalPipelineRuntime(telemetry=telemetry)
        request = PipelineRunRequest(
            run_id=run_id,
            input_video=f"artifact://local/{abs_video}",
            output_root=f"artifact://local/{abs_output}/",
            runtime_mode="cpu_debug",
            config=request_config,
            db_path=abs_db,
            video_id=video_id,
            config_preset=config_preset,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/app/test_pipeline_runner_threads_config.py -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add app/services/pipeline_runner.py tests/app/test_pipeline_runner_threads_config.py
git commit -m "feat(v55-stages): pipeline_runner threads PipelineConfig into request

Calls PipelineConfig.to_request_config() and passes the full dict so
back-half stages read every knob they need from request.config. CLI
arg (detector) still wins via override."
```

### Task 1.4: Same threading in `cli.py`

**Files:**
- Modify: `src/card_capture/cli.py:177-201` (the unified runtime block in `_run_process`)
- Test: covered by an extension to `tests/test_config_to_request_dict.py` — minimal CLI smoke

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_to_request_dict.py`:

```python
def test_cli_run_process_passes_full_config(tmp_path, monkeypatch):
    """The CLI's _run_process must call to_request_config()."""
    from unittest.mock import patch, MagicMock
    from card_capture.cli import _run_process

    captured = {}

    def fake_runtime_run(self, request):
        captured["config"] = dict(request.config)
        result = MagicMock()
        result.manifest.contract_violations = []
        result.manifest.to_json.return_value = "{}"
        return result

    args = MagicMock()
    args.video_path = tmp_path / "v.mov"
    args.video_path.touch()
    args.output_dir = tmp_path / "out"
    args.db = tmp_path / "cards.sqlite"
    args.db.touch()
    args.config = tmp_path / "card_capture_config.json"
    args.detector = "fake"
    args.run_id = "r1"
    # All other CLI override fields default to None
    for f in ("tracker_backend", "fast_scan_fps", "confirm_scan_fps",
              "valley_drop_ratio", "valley_min_width_frames",
              "delta_spike_ratio", "centroid_jump_ratio",
              "centroid_jump_frames", "reid_distance_threshold",
              "presence_threshold"):
        setattr(args, f, None)

    # Storage.initialize and add_video need to succeed
    with patch("card_capture.storage.Storage") as MockStorage, \
         patch("card_capture.pipeline.runtime_local.LocalPipelineRuntime.run",
               new=fake_runtime_run):
        MockStorage.return_value.add_video.return_value = 1
        _run_process(args)

    assert "novelty_floor" in captured["config"]
    assert "foil_threshold" in captured["config"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_to_request_dict.py::test_cli_run_process_passes_full_config -v`
Expected: FAILED — current CLI builds `config={"detector": ..., "corner_confidence": ..., "device": ...}` only.

- [ ] **Step 3: Modify `_run_process` in `cli.py`**

In `src/card_capture/cli.py`, change the `req = PipelineRunRequest(...)` block (lines 185–197 of current file) to:

```python
    request_config = config.to_request_config()
    request_config["detector"] = config.detector
    request_config["corner_confidence"] = config.corner_confidence
    request_config["device"] = config.device

    req = PipelineRunRequest(
        run_id=args.run_id or uuid.uuid4().hex[:12],
        input_video=f"artifact://local/{args.video_path.resolve()}",
        output_root=f"artifact://local/{args.output_dir.resolve()}/",
        runtime_mode=runtime_mode,
        config=request_config,
        db_path=str(args.db.resolve()),
        video_id=video_id,
        config_preset=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_to_request_dict.py::test_cli_run_process_passes_full_config -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/cli.py tests/test_config_to_request_dict.py
git commit -m "feat(v55-stages): cli _run_process threads full PipelineConfig

Mirror of the pipeline_runner change so 'card-capture process'
also gets every back-half knob in request.config."
```

### Task 1.5: Same threading in `worker_core.py` and `training_service.py`

**Files:**
- Modify: `app/worker_core.py:71-88` (the `request = PipelineRunRequest(...)` block in `run_pipeline`)
- Modify: `app/services/training_service.py:296-306` (the `_rerun_video` request build)
- Test: covered by extending one of the existing tests in `tests/app/test_worker_core.py` and `tests/app/test_training_service.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/app/test_worker_core.py`:

```python
def test_run_pipeline_passes_full_config(tmp_path, monkeypatch):
    from unittest.mock import patch, MagicMock
    captured = {}

    def fake_runtime_run(self, request):
        captured["config"] = dict(request.config)
        result = MagicMock()
        result.manifest.contract_violations = []
        return result

    from app.worker_core import run_pipeline
    with patch("card_capture.pipeline.runtime_local.LocalPipelineRuntime.run",
               new=fake_runtime_run):
        run_pipeline("job-1", str(tmp_path / "v.mov"), "balanced", tmp_path)

    assert captured["config"]["foil_threshold"] == 50.0
    assert captured["config"]["use_fb_classifier"] is True
```

Append to `tests/app/test_training_service.py` (create the file if it doesn't exist):

```python
"""Phase 1 — training_service threads PipelineConfig into runtime request."""
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.training_service import TrainingService


def test_rerun_video_passes_full_config(tmp_path, monkeypatch):
    captured = {}

    def fake_runtime_run(self, request):
        captured["config"] = dict(request.config)
        result = MagicMock()
        result.manifest.contract_violations = []
        return result

    db = tmp_path / "cards.sqlite"
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, cards_extracted INT)")
        conn.execute("INSERT INTO pipeline_runs VALUES ('benchmark-x', 3)")

    svc = TrainingService(db_path=db)
    video = tmp_path / "v.mov"
    video.touch()

    with patch(
        "card_capture.pipeline.runtime_local.LocalPipelineRuntime.run",
        new=fake_runtime_run,
    ):
        # We don't care about the return value; only that config got threaded
        try:
            svc._rerun_video(str(video))
        except Exception:
            pass  # tolerate the DB query at the end

    assert "foil_threshold" in captured["config"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/app/test_worker_core.py::test_run_pipeline_passes_full_config tests/app/test_training_service.py::test_rerun_video_passes_full_config -v`
Expected: 2 FAILED — current code builds `config={"detector": ...}` minimal dict.

- [ ] **Step 3: Modify `worker_core.run_pipeline`**

In `app/worker_core.py`, replace the `request = PipelineRunRequest(...)` block with:

```python
    from card_capture.config import load_config
    config = load_config(Path(__file__).parent.parent / "card_capture_config.json")
    request_config = config.to_request_config()
    # RunPod path forces CUDA detector regardless of config file
    request_config["detector"] = "cuda"
    request_config["device"] = "cuda"

    request = PipelineRunRequest(
        run_id=job_id,
        input_video=f"artifact://local/{video_path}",
        output_root=f"artifact://local/{output_dir.resolve()}/",
        runtime_mode="strict_gpu",
        config=request_config,
        db_path=str(db_path.resolve()),
        config_preset=config_preset,
    )
```

- [ ] **Step 4: Modify `training_service._rerun_video`**

In `app/services/training_service.py:_rerun_video`, replace the `request = PipelineRunRequest(...)` block with:

```python
        from card_capture.config import load_config
        config = load_config(_Path(__file__).parent.parent.parent / "card_capture_config.json")
        request_config = config.to_request_config()
        request_config["detector"] = "docaligner"

        request = PipelineRunRequest(
            run_id=run_id,
            input_video=f"artifact://local/{_Path(video_path).resolve()}",
            output_root=f"artifact://local/{out_dir.resolve()}/",
            runtime_mode="cpu_debug",
            config=request_config,
            db_path=str(_Path(self.db_path).resolve()),
            config_preset="balanced",
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/app/test_worker_core.py::test_run_pipeline_passes_full_config tests/app/test_training_service.py::test_rerun_video_passes_full_config -v`
Expected: 2 PASSED.

- [ ] **Step 6: Commit**

```bash
git add app/worker_core.py app/services/training_service.py tests/app/test_worker_core.py tests/app/test_training_service.py
git commit -m "feat(v55-stages): worker_core + training_service thread PipelineConfig

Same pattern as pipeline_runner / cli — call to_request_config()
and pass through. Caller-specific overrides (detector=cuda for RunPod,
detector=docaligner for training rerun) preserved."
```

### Phase 1 acceptance

After Phase 1, run the full test suite:

Run: `.venv/bin/python -m pytest tests/ -m "not quarantine" -q`
Expected: same pass/fail count as before Phase 1 plus the 16 new tests added in 1.1–1.5. The 9 pre-existing env failures (objc dyld + pytest-asyncio) remain unchanged.

---

## Phase 2 — `CardsRepository` write methods + SQL constants

The `store` stage (Phase 9) must write through `CardsRepository`, not raw `Storage`, to keep the `no-sqlite3-outside-data` import-linter contract green. Add 8 thin wrappers, each backed by an SQL constant in `sql_queries.py`.

### Task 2.1: Add SQL constants for back-half writes

**Files:**
- Modify: `src/card_capture/data/sql_queries.py`
- Test: `tests/data/test_back_half_sql_constants.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_back_half_sql_constants.py`:

```python
"""Phase 2 — SQL constants exist and parse as valid SQL statements."""
import sqlite3
import pytest

from card_capture.data import sql_queries as q


CONSTANTS = [
    "CARDS_ADD_INSTANCE",
    "CARDS_UPDATE_DEDUPLICATION",
    "CARDS_UPDATE_FUSION",
    "CARDS_ADD_VIEW",
    "CARDS_ADD_SAVED",
    "CARDS_ADD_TRACK_TELEMETRY",
    "CARDS_ADD_PIPELINE_EVENT",
    "CARDS_FIND_EMBEDDINGS_EXCLUDING_VIDEO",
]


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_exists(name):
    assert hasattr(q, name), f"{name} missing from sql_queries"


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_is_string(name):
    val = getattr(q, name)
    assert isinstance(val, str) and len(val) > 0


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_parses(name):
    """SQLite must accept the statement for prepare (uses a scratch in-memory db)."""
    conn = sqlite3.connect(":memory:")
    # Create minimum tables for prepare to succeed
    conn.executescript("""
        CREATE TABLE card_instances (
            id INTEGER PRIMARY KEY, video_id INTEGER, track_id TEXT,
            angle TEXT, session_id TEXT, reid_embedding BLOB, run_id TEXT,
            primary_hash TEXT, is_duplicate_of INTEGER, fused_image_path TEXT
        );
        CREATE TABLE card_views (
            id INTEGER PRIMARY KEY, card_instance_id INTEGER, frame_index INTEGER,
            timestamp_ms INTEGER, corners TEXT, confidence REAL,
            rectified_path TEXT, quality_score TEXT, is_canonical INTEGER,
            glare_x REAL, glare_y REAL, sharpness REAL, initial_confidence REAL
        );
        CREATE TABLE saved_cards (
            id INTEGER PRIMARY KEY, detection_id INTEGER,
            image_path TEXT, final_score REAL
        );
        CREATE TABLE track_telemetry (
            video_id INTEGER, instance_id TEXT, frame_index INTEGER,
            area REAL, aspect REAL, cx REAL, cy REAL
        );
        CREATE TABLE pipeline_events (
            video_id INTEGER, frame_index INTEGER, timestamp_ms INTEGER,
            event_type TEXT, data TEXT
        );
    """)
    sql = getattr(q, name)
    try:
        conn.execute(f"EXPLAIN {sql}", tuple([None] * sql.count("?")))
    except sqlite3.OperationalError as e:
        pytest.fail(f"{name} does not parse: {e}\nSQL: {sql}")
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_back_half_sql_constants.py -v`
Expected: 24 FAILED (8 constants × 3 tests each), all with `AttributeError`.

- [ ] **Step 3: Add the constants to `sql_queries.py`**

Append to `src/card_capture/data/sql_queries.py`:

```python
# ---------------------------------------------------------------------------
# Phase 2 — Back-half stage writes (consumed by CardsRepository methods that
# back the V5.5 store stage). Each constant mirrors a Storage method in
# src/card_capture/storage.py and is kept here so import-linter's
# no-sqlite3-outside-data contract stays green.
# ---------------------------------------------------------------------------

CARDS_ADD_INSTANCE = """
INSERT INTO card_instances (video_id, track_id, angle, session_id,
                            reid_embedding, run_id)
VALUES (?, ?, ?, ?, ?, ?)
"""

CARDS_UPDATE_DEDUPLICATION = """
UPDATE card_instances
   SET primary_hash = ?,
       is_duplicate_of = ?,
       reid_embedding = COALESCE(?, reid_embedding)
 WHERE id = ?
"""

CARDS_UPDATE_FUSION = """
UPDATE card_instances
   SET fused_image_path = ?
 WHERE id = ?
"""

CARDS_ADD_VIEW = """
INSERT INTO card_views (card_instance_id, frame_index, timestamp_ms,
                        corners, confidence, rectified_path,
                        quality_score, is_canonical,
                        glare_x, glare_y, sharpness, initial_confidence)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

CARDS_ADD_SAVED = """
INSERT INTO saved_cards (detection_id, image_path, final_score)
VALUES (?, ?, ?)
"""

CARDS_ADD_TRACK_TELEMETRY = """
INSERT INTO track_telemetry (video_id, instance_id, frame_index,
                              area, aspect, cx, cy)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

CARDS_ADD_PIPELINE_EVENT = """
INSERT INTO pipeline_events (video_id, frame_index, timestamp_ms,
                              event_type, data)
VALUES (?, ?, ?, ?, ?)
"""

CARDS_FIND_EMBEDDINGS_EXCLUDING_VIDEO = """
SELECT id, reid_embedding
  FROM card_instances
 WHERE reid_embedding IS NOT NULL
   AND is_duplicate_of IS NULL
   AND video_id != ?
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_back_half_sql_constants.py -v`
Expected: 24 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/data/sql_queries.py tests/data/test_back_half_sql_constants.py
git commit -m "feat(v55-stages): add 8 back-half SQL constants to sql_queries

Mirrors existing Storage methods so CardsRepository can wrap them
without leaking raw SQL into the application layer. import-linter's
no-sqlite3-outside-data contract stays green."
```

### Task 2.2: `CardsRepository.add_card_instance`

**Files:**
- Modify: `src/card_capture/data/repositories/cards.py`
- Test: `tests/data/test_cards_repository_writes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_cards_repository_writes.py`:

```python
"""Phase 2 — CardsRepository write methods used by the store stage."""
import sqlite3
from pathlib import Path

import pytest

from card_capture.data.connection import open_connection
from card_capture.data.repositories.cards import CardsRepository
from card_capture.data.writer import Writer


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "cards.sqlite"
    with open_connection(p) as conn:
        conn.executescript("""
            CREATE TABLE card_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                angle TEXT,
                session_id TEXT,
                reid_embedding BLOB,
                run_id TEXT,
                primary_hash TEXT,
                is_duplicate_of INTEGER,
                fused_image_path TEXT
            );
            CREATE TABLE card_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_instance_id INTEGER NOT NULL,
                frame_index INTEGER,
                timestamp_ms INTEGER,
                corners TEXT,
                confidence REAL,
                rectified_path TEXT,
                quality_score TEXT,
                is_canonical INTEGER,
                glare_x REAL,
                glare_y REAL,
                sharpness REAL,
                initial_confidence REAL
            );
            CREATE TABLE saved_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER,
                image_path TEXT,
                final_score REAL
            );
            CREATE TABLE track_telemetry (
                video_id INTEGER, instance_id TEXT, frame_index INTEGER,
                area REAL, aspect REAL, cx REAL, cy REAL
            );
            CREATE TABLE pipeline_events (
                video_id INTEGER, frame_index INTEGER, timestamp_ms INTEGER,
                event_type TEXT, data TEXT
            );
        """)
    return p


@pytest.fixture
def repo(db):
    w = Writer(db)
    w.start()
    yield CardsRepository(w, db)
    w.stop()


def test_add_card_instance_returns_row_id(repo, db):
    row_id = repo.add_card_instance(
        video_id=1, track_id="t-abc", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    assert isinstance(row_id, int) and row_id > 0
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT video_id, track_id, angle, run_id FROM card_instances WHERE id=?",
            (row_id,),
        ).fetchone()
    assert tuple(row) == (1, "t-abc", "Front", "r1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_cards_repository_writes.py::test_add_card_instance_returns_row_id -v`
Expected: FAILED — `CardsRepository` does not have `add_card_instance`.

- [ ] **Step 3: Implement `add_card_instance` on `CardsRepository`**

In `src/card_capture/data/repositories/cards.py`, add this method (after the existing methods, before any helper functions):

```python
    def add_card_instance(
        self,
        *,
        video_id: int,
        track_id: str,
        angle: str | None,
        session_id: str | None,
        reid_embedding: bytes | None,
        run_id: str | None,
    ) -> int:
        """Insert a card_instances row and return the new row id.

        Synchronous: blocks on the Writer queue until the INSERT is
        committed so callers (the store stage) can use the returned id
        immediately for subsequent card_views inserts.
        """
        from card_capture.data.sql_queries import CARDS_ADD_INSTANCE
        from card_capture.data.writer import Write

        future = self._writer.submit_returning(Write(
            sql=CARDS_ADD_INSTANCE,
            params=(video_id, track_id, angle, session_id, reid_embedding, run_id),
        ))
        return int(future.result())  # raises if the writer errored
```

This assumes `Writer.submit_returning` exists and returns a future yielding `lastrowid`. If it doesn't, add it in the same task — see Step 3b below.

- [ ] **Step 3b: If `Writer.submit_returning` doesn't exist, add it**

Run: `.venv/bin/python -c "from card_capture.data.writer import Writer; print(hasattr(Writer, 'submit_returning'))"`
Expected output: `True` or `False`.

If `False`, in `src/card_capture/data/writer.py`, add this method to `class Writer`:

```python
    def submit_returning(self, write: "Write") -> "concurrent.futures.Future[int]":
        """Submit a write whose ``lastrowid`` we need.

        The internal writer thread executes the statement, calls
        ``cursor.lastrowid``, and resolves the returned Future. Use
        ``.result()`` on the call site to block until done.
        """
        import concurrent.futures
        fut: concurrent.futures.Future = concurrent.futures.Future()
        # The writer thread loop must check for these "returning" writes.
        # We use a sentinel tuple (Write, Future) on the same queue.
        self._queue.put(("__returning__", write, fut))
        return fut
```

And update the writer thread loop (find the `while True:` body that pops `self._queue`) to handle the sentinel:

```python
            item = self._queue.get()
            if item is _SENTINEL_STOP:
                break
            if isinstance(item, tuple) and item and item[0] == "__returning__":
                _, write, fut = item
                try:
                    cur = self._conn.execute(write.sql, write.params)
                    self._conn.commit()
                    fut.set_result(cur.lastrowid)
                except BaseException as exc:
                    fut.set_exception(exc)
                continue
            # ... existing handling for plain Write items
```

(The exact location depends on what the file looks like; read it first with: `.venv/bin/python -m pytest tests/data/ -k "writer" -v` to find the test file that exercises it, then mirror the pattern.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_cards_repository_writes.py::test_add_card_instance_returns_row_id -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/data/repositories/cards.py src/card_capture/data/writer.py tests/data/test_cards_repository_writes.py
git commit -m "feat(v55-stages): CardsRepository.add_card_instance + Writer.submit_returning

First of 8 store-stage write methods. submit_returning blocks the
caller until lastrowid is available; needed so the store stage can
chain card_views inserts off the new card_instance row id."
```

### Task 2.3: `CardsRepository.update_instance_deduplication`

**Files:**
- Modify: `src/card_capture/data/repositories/cards.py`
- Test: `tests/data/test_cards_repository_writes.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/data/test_cards_repository_writes.py`:

```python
def test_update_instance_deduplication_writes_hash_and_dup_parent(repo, db):
    row_id = repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    repo.update_instance_deduplication(
        row_id=row_id,
        primary_hash="aabbccdd",
        cross_video_parent=None,
        reid_embedding=b"\x00" * 4,
    )
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT primary_hash, is_duplicate_of, reid_embedding FROM card_instances WHERE id=?",
            (row_id,),
        ).fetchone()
    assert row[0] == "aabbccdd"
    assert row[1] is None
    assert row[2] == b"\x00" * 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_cards_repository_writes.py::test_update_instance_deduplication_writes_hash_and_dup_parent -v`
Expected: FAILED — method missing.

- [ ] **Step 3: Implement on `CardsRepository`**

```python
    def update_instance_deduplication(
        self,
        *,
        row_id: int,
        primary_hash: str,
        cross_video_parent: int | None,
        reid_embedding: bytes | None = None,
    ) -> None:
        from card_capture.data.sql_queries import CARDS_UPDATE_DEDUPLICATION
        from card_capture.data.writer import Write
        self._writer.submit(Write(
            sql=CARDS_UPDATE_DEDUPLICATION,
            params=(primary_hash, cross_video_parent, reid_embedding, row_id),
        ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_cards_repository_writes.py::test_update_instance_deduplication_writes_hash_and_dup_parent -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/data/repositories/cards.py tests/data/test_cards_repository_writes.py
git commit -m "feat(v55-stages): CardsRepository.update_instance_deduplication"
```

### Task 2.4: `CardsRepository.update_instance_fusion`

**Files:**
- Modify: `src/card_capture/data/repositories/cards.py`
- Test: `tests/data/test_cards_repository_writes.py`

- [ ] **Step 1: Append the failing test**

```python
def test_update_instance_fusion_writes_fused_path(repo, db):
    row_id = repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    repo.update_instance_fusion(row_id=row_id, fused_image_path="/tmp/x.jpg")
    with open_connection(db) as conn:
        path = conn.execute(
            "SELECT fused_image_path FROM card_instances WHERE id=?",
            (row_id,),
        ).fetchone()[0]
    assert path == "/tmp/x.jpg"
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/data/test_cards_repository_writes.py::test_update_instance_fusion_writes_fused_path -v`

- [ ] **Step 3: Implement**

```python
    def update_instance_fusion(self, *, row_id: int, fused_image_path: str) -> None:
        from card_capture.data.sql_queries import CARDS_UPDATE_FUSION
        from card_capture.data.writer import Write
        self._writer.submit(Write(
            sql=CARDS_UPDATE_FUSION,
            params=(fused_image_path, row_id),
        ))
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(v55-stages): CardsRepository.update_instance_fusion"
```

### Task 2.5: `CardsRepository.add_card_view`

**Files:**
- Modify: `src/card_capture/data/repositories/cards.py`
- Test: `tests/data/test_cards_repository_writes.py`

- [ ] **Step 1: Append the failing test**

```python
def test_add_card_view_returns_id_and_persists(repo, db):
    inst = repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    import json
    view_id = repo.add_card_view(
        card_instance_id=inst,
        frame_index=10,
        timestamp_ms=333,
        corners=[(0.0, 0.0), (750.0, 0.0), (750.0, 1050.0), (0.0, 1050.0)],
        confidence=0.92,
        rectified_path="/tmp/v.jpg",
        quality_score={"sharpness": 0.7},
        is_canonical=True,
        glare_x=None, glare_y=None, sharpness=0.7,
        initial_confidence=0.92,
    )
    assert isinstance(view_id, int) and view_id > 0
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT frame_index, rectified_path, is_canonical, quality_score "
            "FROM card_views WHERE id=?",
            (view_id,),
        ).fetchone()
    assert row[0] == 10
    assert row[1] == "/tmp/v.jpg"
    assert row[2] == 1
    assert json.loads(row[3])["sharpness"] == 0.7
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/data/test_cards_repository_writes.py::test_add_card_view_returns_id_and_persists -v`

- [ ] **Step 3: Implement**

```python
    def add_card_view(
        self,
        *,
        card_instance_id: int,
        frame_index: int,
        timestamp_ms: int,
        corners: list,
        confidence: float,
        rectified_path: str,
        quality_score: dict | None,
        is_canonical: bool,
        glare_x: float | None,
        glare_y: float | None,
        sharpness: float | None,
        initial_confidence: float | None,
    ) -> int:
        import json
        from card_capture.data.sql_queries import CARDS_ADD_VIEW
        from card_capture.data.writer import Write
        future = self._writer.submit_returning(Write(
            sql=CARDS_ADD_VIEW,
            params=(
                card_instance_id, frame_index, timestamp_ms,
                json.dumps(corners), float(confidence), rectified_path,
                json.dumps(quality_score or {}), 1 if is_canonical else 0,
                glare_x, glare_y, sharpness, initial_confidence,
            ),
        ))
        return int(future.result())
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(v55-stages): CardsRepository.add_card_view"
```

### Task 2.6: `CardsRepository.add_saved_card`

- [ ] **Step 1: Append test**

```python
def test_add_saved_card_persists(repo, db):
    repo.add_saved_card(detection_id=42, image_path="/tmp/c.jpg", final_score=0.85)
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT detection_id, image_path, final_score FROM saved_cards"
        ).fetchone()
    assert tuple(row) == (42, "/tmp/c.jpg", 0.85)
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
    def add_saved_card(
        self, *, detection_id: int, image_path: str, final_score: float
    ) -> None:
        from card_capture.data.sql_queries import CARDS_ADD_SAVED
        from card_capture.data.writer import Write
        self._writer.submit(Write(
            sql=CARDS_ADD_SAVED,
            params=(detection_id, image_path, float(final_score)),
        ))
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(v55-stages): CardsRepository.add_saved_card"
```

### Task 2.7: `CardsRepository.add_track_telemetry`

- [ ] **Step 1: Append test**

```python
def test_add_track_telemetry_persists(repo, db):
    repo.add_track_telemetry(
        video_id=1, instance_id="t-abc", frame_index=100,
        area=750000.0, aspect=0.714, cx=1920.0, cy=1080.0,
    )
    with open_connection(db) as conn:
        row = conn.execute("SELECT * FROM track_telemetry").fetchone()
    assert row[0] == 1
    assert row[1] == "t-abc"
    assert row[2] == 100
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
    def add_track_telemetry(
        self,
        *,
        video_id: int,
        instance_id: str,
        frame_index: int,
        area: float,
        aspect: float,
        cx: float,
        cy: float,
    ) -> None:
        from card_capture.data.sql_queries import CARDS_ADD_TRACK_TELEMETRY
        from card_capture.data.writer import Write
        self._writer.submit(Write(
            sql=CARDS_ADD_TRACK_TELEMETRY,
            params=(video_id, instance_id, frame_index,
                    float(area), float(aspect), float(cx), float(cy)),
        ))
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(v55-stages): CardsRepository.add_track_telemetry"
```

### Task 2.8: `CardsRepository.add_pipeline_event`

- [ ] **Step 1: Append test**

```python
def test_add_pipeline_event_persists(repo, db):
    import json
    repo.add_pipeline_event(
        video_id=1, frame_index=0, timestamp_ms=0,
        event_type="reid_embedding_failed",
        data={"instance_id": "t-abc", "error": "FileNotFoundError"},
    )
    with open_connection(db) as conn:
        row = conn.execute("SELECT event_type, data FROM pipeline_events").fetchone()
    assert row[0] == "reid_embedding_failed"
    assert json.loads(row[1])["error"] == "FileNotFoundError"
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
    def add_pipeline_event(
        self,
        *,
        video_id: int,
        frame_index: int,
        timestamp_ms: int,
        event_type: str,
        data: dict,
    ) -> None:
        import json
        from card_capture.data.sql_queries import CARDS_ADD_PIPELINE_EVENT
        from card_capture.data.writer import Write
        self._writer.submit(Write(
            sql=CARDS_ADD_PIPELINE_EVENT,
            params=(video_id, frame_index, timestamp_ms,
                    event_type, json.dumps(data)),
        ))
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(v55-stages): CardsRepository.add_pipeline_event"
```

### Task 2.9: `CardsRepository.find_embeddings_excluding_video`

**Files:**
- Modify: `src/card_capture/data/repositories/cards.py`
- Test: `tests/data/test_cards_repository_writes.py`

- [ ] **Step 1: Append the failing test**

```python
def test_find_embeddings_excluding_video_returns_only_other_videos(repo, db):
    import numpy as np
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).tobytes()
    emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32).tobytes()
    repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=emb_a, run_id="r1",
    )
    repo.add_card_instance(
        video_id=2, track_id="t2", angle="Front",
        session_id="0", reid_embedding=emb_b, run_id="r2",
    )
    rows = repo.find_embeddings_excluding_video(video_id=1)
    assert len(rows) == 1
    assert rows[0][1] == emb_b
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/data/test_cards_repository_writes.py::test_find_embeddings_excluding_video_returns_only_other_videos -v`

- [ ] **Step 3: Implement (read path — uses `read_connection`, not writer)**

```python
    def find_embeddings_excluding_video(self, *, video_id: int) -> list[tuple[int, bytes]]:
        from card_capture.data.connection import read_connection
        from card_capture.data.sql_queries import CARDS_FIND_EMBEDDINGS_EXCLUDING_VIDEO
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                CARDS_FIND_EMBEDDINGS_EXCLUDING_VIDEO, (video_id,)
            ).fetchall()
        return [(int(r[0]), bytes(r[1])) for r in rows]
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(v55-stages): CardsRepository.find_embeddings_excluding_video"
```

### Phase 2 acceptance

Run: `.venv/bin/python -m pytest tests/data/ tests/architecture/test_raw_sql_outside_data.py -q`
Expected: all green; `test_raw_sql_outside_data` still reports 0 raw SQL outside `card_capture.data`.

---

## Phase 3 — Array-variant ML helpers

The V4 `refine`, `resolve`, and `store` steps call ML helpers with file paths. The v5.5 in-memory mandate says we shouldn't write→read JPEG round-trips just to feed these models. Add `_array` companions for the three helpers used in the back half.

### Task 3.1: `DinoEmbedder.embed_array`

**Files:**
- Modify: `src/card_capture/ml/models/dino_embedder.py`
- Test: `tests/ml/test_dino_embedder_array.py`

- [ ] **Step 1: Read the existing `DinoEmbedder.embed_image` so we know exactly what transform pipeline to mirror**

Run: `.venv/bin/python -c "import inspect; from card_capture.ml.models.dino_embedder import DinoEmbedder; print(inspect.getsource(DinoEmbedder.embed_image))"`
Note the order of operations: `cv2.imread` → `cv2.cvtColor` to RGB → `torchvision.transforms` resize+normalize → `model(...).cpu()`. The `embed_array` method must skip only the `cv2.imread` step.

- [ ] **Step 2: Write the failing parity test**

Create `tests/ml/test_dino_embedder_array.py`:

```python
"""Phase 3 — DinoEmbedder.embed_array equals embed_image(path) for the same image."""
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.skipif(
    not Path("models").exists(),
    reason="DINOv2 model weights not available locally",
)
def test_embed_array_matches_embed_image(tmp_path):
    from card_capture.ml.models.dino_embedder import DinoEmbedder
    img = (np.random.RandomState(42).rand(1050, 750, 3) * 255).astype(np.uint8)
    path = tmp_path / "x.jpg"
    cv2.imwrite(str(path), img)
    re_read = cv2.imread(str(path))  # JPEG compression — embed_array must match this, not the original ndarray

    emb = DinoEmbedder(variant="vits14")
    from_path = emb.embed_image(str(path)).cpu().numpy()
    from_array = emb.embed_array(re_read).cpu().numpy()
    assert np.allclose(from_path, from_array, atol=1e-5)
```

- [ ] **Step 3: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/ml/test_dino_embedder_array.py -v`
Expected: FAILED with `AttributeError: 'DinoEmbedder' object has no attribute 'embed_array'`.

- [ ] **Step 4: Refactor `embed_image` to delegate to `embed_array`**

In `src/card_capture/ml/models/dino_embedder.py`:

1. Add `embed_array(image: np.ndarray) -> torch.Tensor` that takes the cv2-BGR ndarray and does everything `embed_image` did **after** the `cv2.imread`.
2. Change `embed_image(path)` to `cv2.imread(path)` then delegate to `embed_array`.

Concretely (adjust to match the current `embed_image` body — only the first `cv2.imread` line moves out):

```python
    def embed_array(self, image_bgr: "np.ndarray") -> "torch.Tensor":
        """Compute the DINOv2 embedding from an in-memory BGR ndarray.

        Mirrors :meth:`embed_image` exactly except it skips the
        ``cv2.imread`` step. Used by the V5.5 in-process refine /
        store stages so we don't round-trip through disk.
        """
        import cv2
        import torch
        if image_bgr is None:
            raise ValueError("embed_array received None")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image_rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(tensor)
        return emb

    def embed_image(self, path: str) -> "torch.Tensor":
        """Load *path* from disk and embed. Thin wrapper over embed_array."""
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"DinoEmbedder.embed_image: {path!r} not readable")
        return self.embed_array(img)
```

(If the current implementation already has the transform pipeline factored out of `embed_image`, lift only the parts that come after the imread. Keep the device/grad behavior identical.)

- [ ] **Step 5: Run → PASS**

Run: `.venv/bin/python -m pytest tests/ml/test_dino_embedder_array.py -v`
Expected: PASSED (or SKIPPED if model weights aren't present — the skipif gate handles that case).

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/ml/models/dino_embedder.py tests/ml/test_dino_embedder_array.py
git commit -m "feat(v55-stages): DinoEmbedder.embed_array companion

Avoids round-tripping through JPEG so the V5.5 refine stage can pass
the in-memory canonical crop straight to DINOv2 ReID embedding."
```

### Task 3.2: `FBPredictor.predict_array`

**Files:**
- Modify: `src/card_capture/ml/inference/fb_predict.py`
- Test: `tests/ml/test_fb_predict_array.py`

- [ ] **Step 1: Write the failing parity test**

Create `tests/ml/test_fb_predict_array.py`:

```python
"""Phase 3 — FBPredictor.predict_array equals predict(path)."""
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.skipif(
    not Path("models").exists(),
    reason="FB classifier checkpoint not available locally",
)
def test_predict_array_matches_predict(tmp_path):
    from card_capture.ml.inference.fb_predict import FBPredictor
    img = (np.random.RandomState(0).rand(1050, 750, 3) * 255).astype(np.uint8)
    path = tmp_path / "x.jpg"
    cv2.imwrite(str(path), img)
    re_read = cv2.imread(str(path))

    p = FBPredictor()
    side_path, conf_path = p.predict(str(path))
    side_arr, conf_arr = p.predict_array(re_read)
    assert side_path == side_arr
    assert abs(conf_path - conf_arr) < 1e-5
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/ml/test_fb_predict_array.py -v`
Expected: FAILED with `AttributeError`.

- [ ] **Step 3: Add `predict_array` and refactor `predict`**

In `src/card_capture/ml/inference/fb_predict.py`, add:

```python
    def predict_array(self, image_bgr: "np.ndarray") -> tuple[str, float]:
        """Predict (side, confidence) from an in-memory BGR ndarray.

        Mirrors :meth:`predict` after the ``cv2.imread`` step. Used by
        the V5.5 resolve stage so we don't round-trip through disk.
        """
        import cv2
        import torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
        front_p = float(probs[self._idx_front])
        back_p = float(probs[self._idx_back])
        if front_p >= back_p:
            return "front", front_p
        return "back", back_p

    def predict(self, path: str) -> tuple[str, float]:
        """Load *path* and call :meth:`predict_array`."""
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"FBPredictor.predict: {path!r} not readable")
        return self.predict_array(img)
```

**Step 3a — discover the real attribute names first:**

Run: `.venv/bin/python -c "import inspect; from card_capture.ml.inference.fb_predict import FBPredictor; print(inspect.getsource(FBPredictor.predict))"`

This prints the current `predict` body. Read it carefully and note: (a) the input-transform pipeline (resize / normalize / to_tensor), (b) the device attribute name (`self._device` vs `self.device` vs another), (c) the model attribute, (d) how it maps the output logits to `("front"|"back", confidence)`.

**Step 3b — port verbatim:** copy the body of `predict` from `cv2.imread` onward into the new `predict_array(image_bgr)` method, but skip the `cv2.imread` line. Use the **exact same** attribute names. Then make `predict(path)` a two-line wrapper as shown above. Run the test (`Step 4`) to confirm parity.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/ml/inference/fb_predict.py tests/ml/test_fb_predict_array.py
git commit -m "feat(v55-stages): FBPredictor.predict_array companion"
```

### Task 3.3: `compute_reid_embedding_array`

**Files:**
- Modify: `src/card_capture/ml/embeddings.py`
- Test: `tests/ml/test_reid_embeddings_array.py`

- [ ] **Step 1: Write the failing parity test**

Create `tests/ml/test_reid_embeddings_array.py`:

```python
"""Phase 3 — compute_reid_embedding_array equals compute_reid_embedding(path)."""
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.skipif(
    not Path("models").exists(),
    reason="ReID model weights not available locally",
)
def test_compute_reid_embedding_array_matches_path_variant(tmp_path):
    from card_capture.ml.embeddings import (
        compute_reid_embedding, compute_reid_embedding_array,
    )
    img = (np.random.RandomState(1).rand(1050, 750, 3) * 255).astype(np.uint8)
    path = tmp_path / "x.jpg"
    cv2.imwrite(str(path), img)
    re_read = cv2.imread(str(path))

    from_path = compute_reid_embedding(str(path))
    from_array = compute_reid_embedding_array(re_read)
    assert np.allclose(from_path, from_array, atol=1e-5)
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

In `src/card_capture/ml/embeddings.py`, add:

```python
def compute_reid_embedding(path: str) -> "np.ndarray":
    """Existing entry — now delegates to compute_reid_embedding_array."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"compute_reid_embedding: {path!r} not readable")
    return compute_reid_embedding_array(img)
```

(Note: the `compute_reid_embedding_array` body is built in Step 3b below from the current source — do not commit the file without that function defined.)

- [ ] **Step 3b: Discover the current `compute_reid_embedding` body and inline it**

Run: `.venv/bin/python -c "import inspect; from card_capture.ml.embeddings import compute_reid_embedding; print(inspect.getsource(compute_reid_embedding))"`

This prints the entire body. Two structural cases to handle:

- **Case A: it delegates to a `DinoEmbedder` internally** (the body looks like `embedder = DinoEmbedder(...); return embedder.embed_image(path).cpu().numpy()...`). In that case, `compute_reid_embedding_array(img)` is:

```python
def compute_reid_embedding_array(image_bgr):
    from card_capture.ml.models.dino_embedder import DinoEmbedder
    embedder = DinoEmbedder(variant="vits14")  # use the exact variant the original passes
    return embedder.embed_array(image_bgr).cpu().numpy().astype("float32").flatten()
```

- **Case B: it uses a separate OSNet/ReID backbone** (the body imports a different model class — e.g. `from card_capture.ml.osnet import OsnetReID`). In that case, copy the exact body of `compute_reid_embedding` from after the `cv2.imread` line into `compute_reid_embedding_array`, replacing the `cv2.imread` step with `image_bgr` directly. Preserve every preprocessing call (resize, normalize, mean-subtract, etc.) verbatim.

Whichever case applies, the parity test in Step 2 is the contract that proves you got it right. The test must PASS in Step 4 before commit.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/ml/embeddings.py tests/ml/test_reid_embeddings_array.py
git commit -m "feat(v55-stages): compute_reid_embedding_array companion"
```

### Phase 3 acceptance

Run: `.venv/bin/python -m pytest tests/ml/ -q`
Expected: 3 new tests PASS (or SKIPPED if model weights absent — both are acceptable; CI will run with weights baked into the container image).

---

## Phase 4 — Port `track` and rewrite `refine`

The `track` stage currently returns `List[TrackState]` via the adapter's `.assign()`. V4 `refine` consumes `List[Dict]` with a richer per-candidate shape. We do the conversion inside the `track` stage so `refine` can be a near-verbatim V4 port.

### Task 4.1: `track` stage emits `tracks_data: List[Dict]`

**Files:**
- Modify: `src/card_capture/pipeline/stages/track.py`
- Test: `tests/pipeline/stages/test_track_stage.py`

- [ ] **Step 1: Create the test directory**

```bash
mkdir -p tests/pipeline/stages
touch tests/pipeline/stages/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/pipeline/stages/test_track_stage.py`:

```python
"""Phase 4 — track stage emits V4-shape tracks_data dicts."""
from unittest.mock import MagicMock

import pytest

from card_capture.pipeline.stages import track as track_stage


def _detection(frame_index, det_id, confidence=0.9):
    return {
        "detection_id": det_id,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 33,
        "width": 3840,
        "height": 2160,
        "corners": [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
        "confidence": confidence,
        "novelty_score": 1.0,
        "triage_metrics": {},
    }


def test_track_stage_writes_tracks_data_list_of_dicts():
    request = MagicMock()
    request.config = {"tracker_backend": "bytetrack", "min_track_length": 1}
    state = {
        "request": request,
        "sampled_frames": [],
        "novelty_scored_detections": [_detection(i, i) for i in range(5)],
    }
    track_stage.run(state, telemetry=MagicMock())
    assert "tracks_data" in state
    assert isinstance(state["tracks_data"], list)
    for t in state["tracks_data"]:
        assert "instance_id" in t and isinstance(t["instance_id"], str)
        assert "candidates" in t and isinstance(t["candidates"], list)
        for c in t["candidates"]:
            assert {"frame_index", "corners", "confidence",
                    "timestamp_ms", "width", "height",
                    "score_total", "detection_id"} <= set(c.keys())
```

- [ ] **Step 3: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_track_stage.py -v`
Expected: FAILED — current stage writes `state["tracks"]`, not `state["tracks_data"]`.

- [ ] **Step 4: Replace `stages/track.py` with the conversion-aware version**

Replace the entire body of `src/card_capture/pipeline/stages/track.py` with:

```python
"""Stage 5: Session-Aware Tracking.

Tracker backend (BoT-SORT or ByteTrack) is selected from request.config.
We emit both ``state["tracks"]`` (List[TrackState], legacy) and
``state["tracks_data"]`` (List[Dict] V4 shape) so the back-half stages
can consume the richer per-candidate dicts the V4 refine step expects.
"""
from __future__ import annotations

from card_capture.tracking.botsort_adapter import BoTSORTAdapter
from card_capture.tracking.bytetrack_adapter import ByteTrackAdapter


def run(state: dict, *, telemetry) -> None:
    cfg = state["request"].config
    backend = cfg.get("tracker_backend", "bytetrack")
    if backend == "botsort":
        tracker = BoTSORTAdapter(cfg)
    else:
        tracker = ByteTrackAdapter(cfg)

    detections = state["novelty_scored_detections"]
    frames = state["sampled_frames"]
    track_states = tracker.assign(detections, frames)
    state["tracks"] = track_states

    # Build V4-shape tracks_data: List[Dict] with rich per-candidate dicts.
    # Each ScoredCandidate is enriched with the originating detection row's
    # width/height/confidence/timestamp/corners so refine can reuse them
    # without re-decoding.
    by_det_id = {d["detection_id"]: d for d in detections}
    tracks_data: list[dict] = []
    for ts in track_states:
        candidates: list[dict] = []
        for sc in ts.candidates:
            det = by_det_id.get(sc.detection_id, {})
            candidates.append({
                "detection_id": sc.detection_id,
                "frame_index": sc.frame_index,
                "timestamp_ms": int(sc.timestamp_ms),
                "width": int(det.get("width", 0)),
                "height": int(det.get("height", 0)),
                "corners": [(float(x), float(y)) for x, y in (det.get("corners") or [])],
                "confidence": float(det.get("confidence", 0.0)),
                "novelty_score": float(det.get("novelty_score", 1.0)),
                "score_total": float(getattr(sc.score, "total", 0.0)),
                "image_path": "",
                "triage_metrics": det.get("triage_metrics", {}),
            })
        tracks_data.append({
            "instance_id": ts.instance_id,
            "track_id": int(getattr(ts, "track_id", 0) or 0),
            "angle": "Unknown",
            "session_id": int(getattr(ts, "session_id", 0) or 0),
            "first_frame_index": int(getattr(ts, "first_frame_index", -1) or -1),
            "candidates": candidates,
        })
    state["tracks_data"] = tracks_data
```

- [ ] **Step 5: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_track_stage.py -v`
Expected: PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/pipeline/stages/track.py tests/pipeline/stages/test_track_stage.py tests/pipeline/stages/__init__.py
git commit -m "feat(v55-stages): track stage emits V4-shape tracks_data dicts

Conversion from List[TrackState] -> List[Dict] happens here so the
refine stage can be a near-verbatim V4 port. state['tracks'] kept
as the legacy view for any other consumer."
```

### Task 4.2: `refine` stage — identity-carrying skeleton (no quality scoring yet)

The V4 refine step is 408 LOC. We port it in two tasks: 4.2 establishes the per-frame `frame_entry` shape with `normalized` (np.ndarray), `quality_score` (real), `visual_hash`, and `is_canonical`; 4.3 adds Laplacian scan + ReID embedding + telemetry rows.

**Files:**
- Modify: `src/card_capture/pipeline/stages/refine.py`
- Test: `tests/pipeline/stages/test_refine_stage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/stages/test_refine_stage.py`:

```python
"""Phase 4 — refine stage carries identity and writes in-memory crops."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import refine as refine_stage


def _frame(idx, w=640, h=480):
    img = (np.random.RandomState(idx).rand(h, w, 3) * 255).astype(np.uint8)
    fs = MagicMock()
    fs.frame_index = idx
    fs.image = img
    fs.width = w
    fs.height = h
    fs.timestamp_ms = idx * 33
    return fs


def _track(instance_id, frame_indices):
    return {
        "instance_id": instance_id,
        "track_id": 1,
        "angle": "Unknown",
        "session_id": 0,
        "first_frame_index": frame_indices[0],
        "candidates": [
            {
                "detection_id": idx * 10 + 1,
                "frame_index": idx,
                "timestamp_ms": idx * 33,
                "width": 640,
                "height": 480,
                "corners": [(100.0, 100.0), (300.0, 100.0),
                            (300.0, 400.0), (100.0, 400.0)],
                "confidence": 0.9,
                "novelty_score": 1.0,
                "score_total": 0.7,
                "image_path": "",
                "triage_metrics": {},
            }
            for idx in frame_indices
        ],
    }


def test_refine_carries_identity_into_frame_entries():
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,  # disables laplacian scan for this test
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-aaaaaaaa", [5, 10, 15])],
        "detections": [{"detection_id": idx * 10 + 1, "width": 640, "height": 480,
                        "novelty_score": 1.0, "triage_metrics": {}}
                       for idx in (5, 10, 15)],
        "video_id": 42,
        "db_path": "/tmp/x.sqlite",  # ignored when no telemetry writes
    }
    refine_stage.run(state, telemetry=MagicMock())

    assert "refined_tracks" in state
    refined = state["refined_tracks"]
    assert len(refined) == 1
    track = refined[0]
    assert track["instance_id"] == "inst-aaaaaaaa"
    assert "frame_entries" in track
    assert len(track["frame_entries"]) >= 1
    for fe in track["frame_entries"]:
        assert isinstance(fe["normalized"], np.ndarray)
        assert fe["normalized"].shape == (1050, 750, 3)
        assert isinstance(fe["visual_hash"], str)
        assert "quality_score" in fe and fe["quality_score"] >= 0.0


def test_refine_assigns_best_canonical_image():
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-bbbbbbbb", [5, 10, 15])],
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }
    refine_stage.run(state, telemetry=MagicMock())
    refined = state["refined_tracks"][0]
    assert isinstance(refined["best_canonical_image"], np.ndarray)
    assert refined["best_canonical_image"].shape == (1050, 750, 3)
    assert isinstance(refined["best_canonical_detection_id"], int)
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_stage.py -v`
Expected: 2 FAILED — current `refine` writes `state["crops"]`, not `state["refined_tracks"]`.

- [ ] **Step 3: Replace `stages/refine.py` with the identity-carrying port**

Replace the entire body of `src/card_capture/pipeline/stages/refine.py` with:

```python
"""Stage 6: GPU Refinement (Kornia perspective warp -> 750x1050).

V5.5 in-memory port of V4 ``pipeline/steps/refine.py``. Reads frames
from ``state["sampled_frames"]`` (never re-decodes), warps each track's
top candidates to 750x1050, attaches a QualityScore, a pHash, and glare
telemetry per frame_entry, picks a canonical set, and stashes the
best-canonical image in memory for downstream stages.

Three V4-vs-V5.5 substitutions (audited in P13):
- ``decoded_images[frame_index]`` -> ``state["sampled_frames"]`` lookup
- ``cv2.imwrite(... rectified.jpg)`` -> stored in ``frame_entry["normalized"]``
- ``embedder.embed_image(path)`` -> ``embedder.embed_array(img)``

Track-telemetry persistence and Laplacian-scan optimisation land in
Task 4.3.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from card_capture.cropper import PrecisionNormalizer
from card_capture.deduplicator import VisualDeduplicator
from card_capture.fuser import find_glare_centroid
from card_capture.gpu_refinement import KorniaNormalizer
from card_capture.models import FrameSample
from card_capture.pipeline_utils import (
    _compress_array,
    _glare_mask,
    _laplacian_heatmap,
    _select_canonical_entries,
)
from card_capture.scoring import QualityScorer
from card_capture.selector import ScoredCandidate


def _frame_index_lookup(frames) -> Dict[int, np.ndarray]:
    return {int(f.frame_index): f.image for f in frames}


def _scored_candidate_from_dict(c: dict) -> ScoredCandidate:
    from card_capture.models import QualityScore
    return ScoredCandidate(
        detection_id=int(c["detection_id"]),
        timestamp_ms=int(c.get("timestamp_ms", 0)),
        image_path="",
        score=QualityScore(total=float(c.get("score_total", 0.0)), components={}),
        corners=[(float(x), float(y)) for x, y in (c.get("corners") or [])],
        frame_index=int(c["frame_index"]),
    )


def run(state: dict, *, telemetry) -> None:
    frames = state.get("sampled_frames")
    if frames is None:
        telemetry.contract_violation(
            "refine_without_frames",
            {"hint": "sample stage must populate state['sampled_frames']"},
        )
        raise RuntimeError("refine reached without sampled_frames in state")

    config = state["request"].config
    decoded_images = _frame_index_lookup(frames)

    normalizer = PrecisionNormalizer()
    kornia_normalizer: Optional[KorniaNormalizer] = None
    if config.get("use_kornia", True):
        try:
            kornia_normalizer = KorniaNormalizer(
                width=normalizer.width,
                height=normalizer.height,
                device=config.get("kornia_device", config.get("device", "auto")),
            )
        except Exception:
            kornia_normalizer = None

    deduplicator = VisualDeduplicator()
    scorer = QualityScorer()
    rotate_180 = bool(config.get("rotate_180", False))

    refined_tracks: List[Dict[str, Any]] = []
    tracks_data = state.get("tracks_data") or []

    for track_dict in tracks_data:
        instance_id = track_dict["instance_id"]
        candidates_data = track_dict["candidates"]

        # Sort by score and take top 8 (matches V4 line 173)
        scored_candidates = sorted(
            candidates_data, key=lambda c: c.get("score_total", 0.0), reverse=True
        )[:8]

        # Batched Kornia warp for this track's candidates
        normalized_by_det: Dict[int, np.ndarray] = {}
        if kornia_normalizer is not None and scored_candidates:
            batch_items = []
            batch_ids = []
            for c in scored_candidates:
                raw = decoded_images.get(int(c["frame_index"]))
                if raw is None:
                    h = int(c.get("height", 10))
                    w = int(c.get("width", 10))
                    raw = np.zeros((h, w, 3), dtype=np.uint8)
                batch_items.append((raw, c["corners"]))
                batch_ids.append(int(c["detection_id"]))
            try:
                warped = kornia_normalizer.warp_canonical_batch(
                    batch_items, rotate_180=rotate_180
                )
                for did, img in zip(batch_ids, warped):
                    normalized_by_det[did] = img
            except Exception as exc:
                telemetry.resource_sample(
                    {"event": "kornia_warp_failed", "error": repr(exc)}
                )

        frame_entries: List[Dict[str, Any]] = []
        for c in scored_candidates:
            raw = decoded_images.get(int(c["frame_index"]))
            if raw is None:
                raw = np.zeros((int(c.get("height", 10)), int(c.get("width", 10)), 3),
                               dtype=np.uint8)
            normalized = normalized_by_det.get(int(c["detection_id"]))
            if normalized is None:
                normalized = normalizer.normalize(raw, c["corners"], rotate_180=rotate_180)

            quality_score = scorer.score(
                normalized,
                float(c.get("confidence", 0.0)),
                novelty=float(c.get("novelty_score", 1.0)),
            )
            glare_centroid = find_glare_centroid(normalized)
            glare_x, glare_y = glare_centroid if glare_centroid else (None, None)

            frame_entries.append({
                "candidate_dict": c,
                "candidate": _scored_candidate_from_dict({**c, "score_total": quality_score.total}),
                "normalized": normalized,
                "quality_score_obj": quality_score,
                "quality_score": float(quality_score.total),
                "quality_components": dict(quality_score.components),
                "visual_hash": deduplicator.compute_phash(normalized),
                "glare_x": glare_x,
                "glare_y": glare_y,
                "sharpness": float(quality_score.components.get("sharpness", 0.0)),
                "glare_mask": _compress_array(_glare_mask(normalized)),
                "laplacian_heatmap": _compress_array(_laplacian_heatmap(normalized)),
                "detection_id": int(c["detection_id"]),
                "frame_index": int(c["frame_index"]),
                "timestamp_ms": int(c.get("timestamp_ms", 0)),
                "confidence": float(c.get("confidence", 0.0)),
                "corners": [(float(x), float(y)) for x, y in (c.get("corners") or [])],
                "novelty_score": float(c.get("novelty_score", 1.0)),
                "width": int(c.get("width", 0)),
                "height": int(c.get("height", 0)),
                "triage_metrics": c.get("triage_metrics", {}),
                "is_canonical": False,
            })

        if not frame_entries:
            continue

        # Canonical selection — _select_canonical_entries expects a list of
        # dicts shaped like the V4 patched_entries (key: "candidate" is a
        # ScoredCandidate). We already set "candidate" above.
        canonical_entries = _select_canonical_entries(frame_entries, deduplicator)
        canonical_det_ids = {e["candidate"].detection_id for e in canonical_entries}
        for fe in frame_entries:
            if fe["detection_id"] in canonical_det_ids:
                fe["is_canonical"] = True
        best = max(canonical_entries, key=lambda e: e["quality_score"])

        refined_tracks.append({
            "instance_id": instance_id,
            "track_id": int(track_dict.get("track_id", 0)),
            "angle": track_dict.get("angle", "Unknown"),
            "session_id": int(track_dict.get("session_id", 0)),
            "first_frame_index": int(track_dict.get("first_frame_index", -1)),
            "frame_entries": frame_entries,
            "canonical_detection_ids": list(canonical_det_ids),
            "best_canonical_detection_id": int(best["candidate"].detection_id),
            "best_canonical_image": best["normalized"],
            "reid_embedding": None,  # populated in Task 4.3
        })

    state["refined_tracks"] = refined_tracks
```

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_stage.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/refine.py tests/pipeline/stages/test_refine_stage.py
git commit -m "feat(v55-stages): refine carries identity + scores + dedup hash

Per-track top-8 selection by score_total, Kornia warp (with CPU
fallback via PrecisionNormalizer), QualityScorer.score on each
warped crop, pHash via VisualDeduplicator, glare centroid + masks,
canonical entry selection via _select_canonical_entries.

state['refined_tracks'] now carries:
- instance_id, track_id, session_id, first_frame_index, angle
- frame_entries[] with normalized (np.ndarray), quality_score,
  visual_hash, glare_x/y, sharpness, is_canonical, corners,
  detection_id, frame_index, timestamp_ms
- best_canonical_detection_id + best_canonical_image (np.ndarray)

ReID embedding still None — added in Task 4.3."
```

### Task 4.3: `refine` — Laplacian scan + ReID embedding + track telemetry

**Files:**
- Modify: `src/card_capture/pipeline/stages/refine.py`
- Test: `tests/pipeline/stages/test_refine_stage.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/pipeline/stages/test_refine_stage.py`:

```python
def test_refine_attaches_reid_embedding_when_embedder_available(monkeypatch):
    """When DinoEmbedder is constructible, refine populates reid_embedding."""
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-cccccccc", [5, 10, 15])],
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }

    # Patch DinoEmbedder so we don't need real weights
    import card_capture.pipeline.stages.refine as ref_mod

    class _StubEmbedder:
        def embed_array(self, img):
            import torch
            return torch.tensor([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(ref_mod, "_get_embedder", lambda: _StubEmbedder())
    refine_stage.run(state, telemetry=MagicMock())
    assert state["refined_tracks"][0]["reid_embedding"] == [0.1, 0.2, 0.3]


def test_refine_records_track_telemetry_rows(tmp_path, monkeypatch):
    """refine should call CardsRepository.add_track_telemetry for each canonical."""
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    captured = []

    class _StubRepo:
        def add_track_telemetry(self, **kw):
            captured.append(kw)

    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-dddddddd", [5, 10, 15])],
        "detections": [],
        "video_id": 7,
        "db_path": str(tmp_path / "cards.sqlite"),
        "repos": {"cards": _StubRepo()},
    }
    refine_stage.run(state, telemetry=MagicMock())
    assert any(row["video_id"] == 7 and row["instance_id"] == "inst-dddddddd"
               for row in captured)
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_stage.py -v`
Expected: 2 new tests FAIL — `_get_embedder` doesn't exist; `add_track_telemetry` never called.

- [ ] **Step 3: Add embedder + telemetry block to `refine.py`**

In `src/card_capture/pipeline/stages/refine.py`, add the embedder helper near the top of the module (after the imports):

```python
_EMBEDDER_SINGLETON: object = None


def _get_embedder():
    """Return a DinoEmbedder singleton, or None if weights are missing.

    Cached at module scope so we don't reload the model per pipeline run."""
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is not None:
        return _EMBEDDER_SINGLETON
    try:
        from card_capture.ml.models.dino_embedder import DinoEmbedder
        _EMBEDDER_SINGLETON = DinoEmbedder(variant="vits14")
    except Exception:
        _EMBEDDER_SINGLETON = None
    return _EMBEDDER_SINGLETON
```

Then, inside the per-track loop in `run()`, after `refined_tracks.append({...})`, replace the `"reid_embedding": None` line with a real embedding call and add the telemetry writes. The relevant section becomes:

```python
        best_canonical_img = best["normalized"]

        # ReID embedding (v5.5: array variant, no temp file)
        reid_embedding: Optional[List[float]] = None
        embedder = _get_embedder()
        if embedder is not None:
            try:
                emb_tensor = embedder.embed_array(best_canonical_img)
                reid_embedding = emb_tensor.cpu().numpy().tolist()[0]
            except Exception as exc:
                telemetry.resource_sample(
                    {"event": "reid_embedding_failed",
                     "instance_id": instance_id, "error": repr(exc)}
                )

        # Track telemetry — per canonical entry. Repo is injected into
        # state by the runtime; tests can stub via state["repos"]["cards"].
        cards_repo = (state.get("repos") or {}).get("cards")
        if cards_repo is not None:
            for entry in canonical_entries:
                sc = entry["candidate"]
                if sc.corners:
                    try:
                        from card_capture.selector import _get_polygon_area, _aspect_ratio
                        area = _get_polygon_area(sc.corners)
                        aspect = _aspect_ratio(sc.corners)
                        cx = sum(p[0] for p in sc.corners) / 4.0
                        cy = sum(p[1] for p in sc.corners) / 4.0
                        cards_repo.add_track_telemetry(
                            video_id=int(state.get("video_id", 0)),
                            instance_id=instance_id,
                            frame_index=int(sc.frame_index),
                            area=float(area),
                            aspect=float(aspect),
                            cx=float(cx),
                            cy=float(cy),
                        )
                    except Exception:
                        pass

        refined_tracks.append({
            "instance_id": instance_id,
            "track_id": int(track_dict.get("track_id", 0)),
            "angle": track_dict.get("angle", "Unknown"),
            "session_id": int(track_dict.get("session_id", 0)),
            "first_frame_index": int(track_dict.get("first_frame_index", -1)),
            "frame_entries": frame_entries,
            "canonical_detection_ids": list(canonical_det_ids),
            "best_canonical_detection_id": int(best["candidate"].detection_id),
            "best_canonical_image": best_canonical_img,
            "reid_embedding": reid_embedding,
        })
```

(Remove the prior `refined_tracks.append({..., "reid_embedding": None})` block from Task 4.2 — it's replaced by the version above.)

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_stage.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/refine.py tests/pipeline/stages/test_refine_stage.py
git commit -m "feat(v55-stages): refine attaches ReID embedding + track telemetry

DinoEmbedder.embed_array on best_canonical_image (no temp file).
add_track_telemetry per canonical entry via CardsRepository
injected through state['repos']['cards']."
```

### Task 4.4: `runtime_local` injects `repos` and `output_root` into `state`

The refine + store stages need `state["repos"]["cards"]` and the upcoming store stage needs `state["output_root"]` as a `Path`. The runtime already creates the repos for the legacy keys; ensure those are set the same way + add `output_root`.

**Files:**
- Modify: `src/card_capture/pipeline/runtime_local.py`
- Test: extend `tests/test_unified_runtime.py`

- [ ] **Step 1: Write the failing assertion as a tiny standalone test**

Create `tests/pipeline/test_runtime_state_injection.py`:

```python
"""Phase 4 — runtime injects repos + output_root as Path into state."""
from pathlib import Path
from unittest.mock import MagicMock

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import NoopTelemetry


def test_runtime_injects_state_keys(tmp_path, monkeypatch):
    db = tmp_path / "cards.sqlite"
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE card_instances (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE pipeline_events (id INTEGER PRIMARY KEY)")
    captured = {}

    from card_capture.pipeline.stages import sample as sample_stage

    def fake_sample(state, *, telemetry):
        captured["repos"] = state.get("repos")
        captured["output_root"] = state.get("output_root")
        captured["db_path"] = state.get("db_path")
        # Short-circuit the rest by raising — runtime catches and marks violation
        raise RuntimeError("short-circuit")

    monkeypatch.setattr(sample_stage, "run", fake_sample)

    runtime = LocalPipelineRuntime(telemetry=NoopTelemetry())
    req = PipelineRunRequest(
        run_id="r1",
        input_video="artifact://local/fake.mov",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
        db_path=str(db),
        video_id=1,
    )
    try:
        runtime.run(req)
    except Exception:
        pass

    assert "cards" in captured["repos"]
    assert isinstance(captured["output_root"], Path)
    assert captured["output_root"] == tmp_path
    assert isinstance(captured["db_path"], Path)
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runtime_state_injection.py -v`
Expected: FAILED — `output_root` not currently injected as `Path`.

- [ ] **Step 3: Modify `runtime_local.py`**

In `src/card_capture/pipeline/runtime_local.py`, after the `state: dict = {...}` block, ensure `output_root` is a `Path`:

```python
        # Phase 4 — back-half stages expect output_root as a pathlib.Path
        output_root_str = str(request.output_root).replace("artifact://local/", "").rstrip("/")
        state["output_root"] = Path(output_root_str)
```

Place this line right after the existing `state = {...}` assignment (currently around line 102).

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runtime_state_injection.py -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/runtime_local.py tests/pipeline/test_runtime_state_injection.py
git commit -m "feat(v55-stages): runtime injects output_root as Path into state

Stages can read state['output_root'] without re-parsing the
artifact:// URI. repos['cards'/'runs'/'events'] already injected;
this test pins both contracts."
```

### Phase 4 acceptance

Run: `.venv/bin/python -m pytest tests/pipeline/stages/ tests/pipeline/test_runtime_state_injection.py -q`
Expected: all green. `state["refined_tracks"]` shape locked.

---

## Phase 5 — Port `score` stage (pruning gates)

V4 `score.py` does three independent prune gates: novelty (adaptive threshold from the per-track median distribution), confidence floor, and transparent-stand gate. Port verbatim.

### Task 5.1: `score` stage — novelty + confidence + stand pruning

**Files:**
- Modify: `src/card_capture/pipeline/stages/score.py`
- Test: `tests/pipeline/stages/test_score_stage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/stages/test_score_stage.py`:

```python
"""Phase 5 — score stage applies novelty / confidence / stand gates."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import score as score_stage


def _track(instance_id, frame_count, novelty=1.0, q=0.7, sharpness=0.7):
    return {
        "instance_id": instance_id,
        "frame_entries": [
            {
                "novelty_score": novelty,
                "quality_score": q,
                "score_total": q,
                "confidence": q,
                "quality_components": {"sharpness": sharpness},
            }
            for _ in range(frame_count)
        ],
    }


def test_score_passes_through_when_no_gates_active():
    """All gates off → no track pruned, scored_tracks shape preserved."""
    request = MagicMock()
    request.config = {
        "novelty_floor": 0.0,
        "track_confidence_floor": 0.0,
        "stand_novelty_max": 0.0,
        "stand_sharpness_max": 0.0,
    }
    state = {
        "request": request,
        "refined_tracks": [
            _track("a", 5, novelty=1.0, q=0.9, sharpness=0.9),
            _track("b", 5, novelty=0.05, q=0.4, sharpness=0.1),
        ],
        "bg_model": None,
    }
    score_stage.run(state, telemetry=MagicMock())
    assert len(state["scored_tracks"]) == 2
    assert all(not t["pruned"] for t in state["scored_tracks"])
    assert state["pruned_instance_ids"] == []


def test_score_confidence_floor_prunes_low_quality():
    request = MagicMock()
    request.config = {
        "novelty_floor": 0.0,
        "track_confidence_floor": 0.60,
        "stand_novelty_max": 0.0,
        "stand_sharpness_max": 0.0,
    }
    state = {
        "request": request,
        "refined_tracks": [
            _track("strong", 5, q=0.8),
            _track("weak", 5, q=0.45),
        ],
        "bg_model": None,
    }
    score_stage.run(state, telemetry=MagicMock())
    assert "weak" in state["pruned_instance_ids"]
    assert "strong" not in state["pruned_instance_ids"]


def test_score_novelty_gate_useful_requires_n5_std015_min035():
    """Gate stays off when there's no useful spread."""
    from card_capture.pipeline.stages.score import _novelty_gate_useful
    assert _novelty_gate_useful([1.0] * 10) is False  # std 0
    assert _novelty_gate_useful([0.5, 0.51, 0.52]) is False  # n < 5
    # Wide spread, low min → useful
    assert _novelty_gate_useful([0.1, 0.2, 0.7, 0.8, 0.9]) is True


def test_score_adaptive_novelty_threshold_is_largest_gap_midpoint():
    """Two-cluster novelty distribution → threshold lands in the gap."""
    request = MagicMock()
    request.config = {
        "novelty_floor": 0.30,
        "track_confidence_floor": 0.0,
        "stand_novelty_max": 0.0,
        "stand_sharpness_max": 0.0,
    }
    # Background model present, two real cards at high novelty, two phantoms at low
    state = {
        "request": request,
        "refined_tracks": [
            _track("real-1", 6, novelty=0.85, q=0.8),
            _track("real-2", 6, novelty=0.80, q=0.8),
            _track("phantom-1", 6, novelty=0.20, q=0.8),
            _track("phantom-2", 6, novelty=0.15, q=0.8),
            _track("real-3", 6, novelty=0.82, q=0.8),
        ],
        "bg_model": object(),  # truthy sentinel
    }
    score_stage.run(state, telemetry=MagicMock())
    pruned = set(state["pruned_instance_ids"])
    assert "phantom-1" in pruned and "phantom-2" in pruned
    assert "real-1" not in pruned and "real-2" not in pruned
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_score_stage.py -v`
Expected: 4 FAILED — current `score.py` is a passthrough.

- [ ] **Step 3: Replace `stages/score.py` with the V4 port**

Replace `src/card_capture/pipeline/stages/score.py` entirely:

```python
"""Stage 7: Quality Scoring + Pruning.

Port of V4 ``pipeline/steps/score.py``. Three independent prune gates,
any of which can drop a track:

1. **Novelty gate**: only active when the per-video novelty distribution
   is bimodal (background model discriminates). Adaptive threshold lands
   at the midpoint of the largest gap between per-track median novelties,
   capped at ``config['novelty_floor']``.
2. **Confidence floor**: always active when ``config['track_confidence_floor']
   > 0``. Prunes tracks whose median quality score is below it.
3. **Transparent-stand gate**: low novelty AND low sharpness. Active when
   ``bg_model`` is present AND ``config['stand_novelty_max'] > 0``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _novelty_gate_useful(scores: list) -> bool:
    """Mirrors V4 _novelty_gate_useful: ≥5 detections, std > 0.15, min < 0.35."""
    if len(scores) < 5:
        return False
    arr = np.array(scores, dtype=np.float32)
    return float(arr.std()) > 0.15 and float(arr.min()) < 0.35


def run(state: dict, *, telemetry) -> None:
    config = state["request"].config
    bg_model = state.get("bg_model")
    refined_tracks = state.get("refined_tracks") or []

    novelty_floor = float(config.get("novelty_floor", 0.30))
    conf_floor = float(config.get("track_confidence_floor", 0.60))
    stand_nov_max = float(config.get("stand_novelty_max", 0.35))
    stand_shp_max = float(config.get("stand_sharpness_max", 0.30))

    all_novelty_scores = [
        float(fe.get("novelty_score", 1.0))
        for track_dict in refined_tracks
        for fe in track_dict.get("frame_entries", [])
    ]
    gate_useful = _novelty_gate_useful(all_novelty_scores)

    if gate_useful:
        track_medians = sorted(
            float(np.median([float(fe.get("novelty_score", 1.0))
                              for fe in t.get("frame_entries", [])])
                  if t.get("frame_entries") else 1.0)
            for t in refined_tracks
        )
        if len(track_medians) >= 2:
            gaps = [track_medians[i + 1] - track_medians[i]
                    for i in range(len(track_medians) - 1)]
            gap_idx = gaps.index(max(gaps))
            adaptive = (track_medians[gap_idx] + track_medians[gap_idx + 1]) / 2
            novelty_threshold = min(adaptive, novelty_floor)
        else:
            novelty_threshold = novelty_floor
    else:
        novelty_threshold = -1.0

    scored_tracks: List[Dict[str, Any]] = []
    pruned_instance_ids: List[str] = []

    for track_dict in refined_tracks:
        frame_entries = track_dict.get("frame_entries", [])

        novelty_scores = [float(fe.get("novelty_score", 1.0)) for fe in frame_entries]
        median_novelty = float(np.median(novelty_scores)) if novelty_scores else 1.0

        quality_scores = [
            float(fe.get("score_total", fe.get("quality_score", fe.get("confidence", 1.0))))
            for fe in frame_entries
        ]
        median_quality = float(np.median(quality_scores)) if quality_scores else 1.0

        sharpness_scores = [
            float(fe.get("quality_components", {}).get("sharpness", 1.0))
            for fe in frame_entries
        ]
        median_sharpness = float(np.median(sharpness_scores)) if sharpness_scores else 1.0

        novelty_prune = (bg_model is not None) and gate_useful and (median_novelty < novelty_threshold)
        confidence_prune = conf_floor > 0 and median_quality < conf_floor
        stand_prune = (
            bg_model is not None
            and stand_nov_max > 0
            and median_novelty < stand_nov_max
            and median_sharpness < stand_shp_max
        )
        should_prune = novelty_prune or confidence_prune or stand_prune

        scored = dict(track_dict)
        scored["pruned"] = should_prune
        scored["median_novelty"] = median_novelty
        scored["median_quality"] = median_quality
        scored["median_sharpness"] = median_sharpness
        scored_tracks.append(scored)

        if should_prune:
            pruned_instance_ids.append(track_dict["instance_id"])

    state["scored_tracks"] = scored_tracks
    state["pruned_instance_ids"] = pruned_instance_ids
```

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_score_stage.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/score.py tests/pipeline/stages/test_score_stage.py
git commit -m "feat(v55-stages): score stage applies novelty/conf/stand pruning gates

Verbatim port of V4 pipeline/steps/score.py with three substitutions:
- refine_out.refined_tracks -> state['refined_tracks']
- ctx.novelty_floor etc. -> request.config dict lookups
- bg_model loaded by novelty stage -> state['bg_model']

Tracks gain median_novelty / median_quality / median_sharpness fields
and a 'pruned' boolean. pruned_instance_ids surfaced separately for
the resolve stage's active-track filter."
```

### Phase 5 acceptance

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_score_stage.py -q`
Expected: 4 passed.

---

## Phase 6 — Port `resolve` stage (F/B + same-card detection)

V4 `resolve.py` does session grouping, F/B classifier with textiness fallback, same-card detection (embedding then pHash), and hard-case capture. Largest single port (~180 LOC).

### Task 6.1: `resolve` stage — session grouping + F/B + same-card

**Files:**
- Modify: `src/card_capture/pipeline/stages/resolve.py`
- Test: `tests/pipeline/stages/test_resolve_stage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/stages/test_resolve_stage.py`:

```python
"""Phase 6 — resolve assigns Front/Back + same-card grouping."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import resolve as resolve_stage


def _scored_track(
    instance_id, frame_count=5, session_id=0, pruned=False,
    visual_hash="00ff00ff00ff00ff", embedding=None,
    best_canonical_image=None,
):
    if best_canonical_image is None:
        best_canonical_image = (np.random.RandomState(hash(instance_id) & 0xFFFFFFFF)
                                .rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "session_id": session_id,
        "pruned": pruned,
        "quality_score": 0.8,
        "frame_entries": [
            {"visual_hash": visual_hash, "triage_metrics": {"border_purity": 0.9},
             "is_canonical": True, "quality_score": 0.8}
            for _ in range(frame_count)
        ],
        "best_canonical_image": best_canonical_image,
        "best_canonical_detection_id": 1,
        "reid_embedding": list(embedding) if embedding is not None else None,
    }


def test_resolve_excludes_pruned_from_prepared_tracks():
    request = MagicMock()
    request.config = {"use_fb_classifier": False}
    state = {
        "request": request,
        "scored_tracks": [
            _scored_track("a", pruned=False),
            _scored_track("b", pruned=True),
        ],
        "db_path": "/tmp/x.sqlite",
        "video_id": 1,
        "output_root": MagicMock(),
        "observed_intra_track_distances": [],
    }
    resolve_stage.run(state, telemetry=MagicMock())
    ids = {t["instance_id"] for t in state["prepared_tracks"]}
    assert ids == {"a"}


def test_resolve_longest_track_becomes_front():
    request = MagicMock()
    request.config = {"use_fb_classifier": False}
    state = {
        "request": request,
        "scored_tracks": [
            _scored_track("short", frame_count=3, session_id=1),
            _scored_track("long", frame_count=8, session_id=1),
        ],
        "db_path": "/tmp/x.sqlite",
        "video_id": 1,
        "output_root": MagicMock(),
        "observed_intra_track_distances": [],
    }
    resolve_stage.run(state, telemetry=MagicMock())
    by_id = {t["instance_id"]: t for t in state["prepared_tracks"]}
    # "long" has more frame_entries → higher quality-weighted composite → Front
    assert by_id["long"]["angle"] == "Front"


def test_resolve_same_card_via_embedding_marks_back():
    request = MagicMock()
    request.config = {"use_fb_classifier": False}
    e_close_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_close_b = np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32)
    e_close_b = e_close_b / np.linalg.norm(e_close_b)
    state = {
        "request": request,
        "scored_tracks": [
            _scored_track("front", session_id=2, embedding=e_close_a),
            _scored_track("back", session_id=2, embedding=e_close_b),
        ],
        "db_path": "/tmp/x.sqlite",
        "video_id": 1,
        "output_root": MagicMock(),
        "observed_intra_track_distances": [],
    }
    resolve_stage.run(state, telemetry=MagicMock())
    by_id = {t["instance_id"]: t for t in state["prepared_tracks"]}
    assert "Front" in {by_id["front"]["angle"], by_id["back"]["angle"]}
    # One front, one back when embeddings are close
    assert set(by_id[k]["angle"] for k in by_id) == {"Front", "Back"}


def test_resolve_same_card_via_phash_when_embeddings_missing():
    request = MagicMock()
    request.config = {"use_fb_classifier": False}
    # Two tracks with identical pHash → must be flagged as same card
    state = {
        "request": request,
        "scored_tracks": [
            _scored_track("primary", session_id=3, visual_hash="deadbeefdeadbeef"),
            _scored_track("dup",     session_id=3, visual_hash="deadbeefdeadbeef"),
        ],
        "db_path": "/tmp/x.sqlite",
        "video_id": 1,
        "output_root": MagicMock(),
        "observed_intra_track_distances": [],
    }
    resolve_stage.run(state, telemetry=MagicMock())
    by_id = {t["instance_id"]: t for t in state["prepared_tracks"]}
    assert "Back" in {by_id["primary"]["angle"], by_id["dup"]["angle"]}
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_resolve_stage.py -v`
Expected: 4 FAILED — current stage is a passthrough.

- [ ] **Step 3: Replace `stages/resolve.py` with the V4 port**

Replace `src/card_capture/pipeline/stages/resolve.py`:

```python
"""Stage 8: Front/Back Resolution + duplicate-session grouping.

Port of V4 ``pipeline/steps/resolve.py``. Substitutions:
- ``score_out.scored_tracks`` -> ``state['scored_tracks']``
- ``ctx.use_fb_classifier`` -> ``config['use_fb_classifier']``
- ``ctx.db_path`` -> ``state['db_path']``
- ``cv2.imread(t['best_canonical_image_path'])`` -> ``t['best_canonical_image']``
- ``FBPredictor.predict(path)`` -> ``FBPredictor.predict_array(ndarray)``
- ``ctx.observed_intra_track_distances`` -> ``state.get('observed_intra_track_distances', [])``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _compute_quality_weighted_score_dict(t: Dict[str, Any], max_length: int) -> float:
    current_length = len(t.get("frame_entries", []))
    normalized_length = current_length / max(1, max_length)
    canonicals = [fe for fe in t.get("frame_entries", []) if fe.get("is_canonical")]
    if not canonicals:
        mean_quality = 0.0
    else:
        mean_quality = sum(float(fe.get("quality_score", 0.0)) for fe in canonicals) / len(canonicals)
    return 0.6 * normalized_length + 0.4 * mean_quality


def run(state: dict, *, telemetry) -> None:
    from card_capture.deduplicator import VisualDeduplicator
    from card_capture.identity.embedding_distance import embedding_same_card_score
    from card_capture.calibration.per_video_adaptive import AdaptiveThresholdComputer
    from card_capture.analysis.hard_case_capture import is_hard_case, capture_hard_case
    from card_capture.pipeline_utils import _side_textiness_score, _appearance_vector

    config = state["request"].config
    scored_tracks = state.get("scored_tracks") or []
    active_tracks = [t for t in scored_tracks if not t.get("pruned")]

    deduplicator = VisualDeduplicator()
    adaptive_computer = AdaptiveThresholdComputer()

    # Optional FB classifier
    fb_predictor = None
    if config.get("use_fb_classifier", True):
        try:
            from card_capture.ml.inference.fb_predict import FBPredictor
            from card_capture.ml.registry import get_latest
            latest_fb = get_latest(db_path=Path(state["db_path"]), model_name="fb_classifier")
            ckpt = latest_fb.checkpoint_path if latest_fb else None
            if FBPredictor.is_available(ckpt):
                fb_predictor = FBPredictor(checkpoint_path=ckpt)
        except Exception as exc:
            telemetry.resource_sample({"event": "fb_predictor_load_failed", "error": repr(exc)})

    # Group active tracks by session
    by_session: Dict[int, List[Dict[str, Any]]] = {}
    for t in active_tracks:
        by_session.setdefault(int(t.get("session_id", 0)), []).append(t)

    observed = state.get("observed_intra_track_distances", [])
    output_dir = state.get("output_root")

    for sid, session_tracks in by_session.items():
        if not session_tracks:
            continue

        session_hamming = []
        session_confidences = []
        session_purity = []

        # Compute side_score + appearance_vector from the in-memory best canonical
        for t in session_tracks:
            img = t.get("best_canonical_image")
            if img is None:
                img = np.zeros((1050, 750, 3), dtype=np.uint8)
            t["side_score"] = float(_side_textiness_score(img))
            t["appearance_vector"] = _appearance_vector(img)

        max_length = max(len(t["frame_entries"]) for t in session_tracks)

        # F/B classifier override (high confidence shifts side_score extremes)
        if fb_predictor is not None:
            for t in session_tracks:
                try:
                    side, conf = fb_predictor.predict_array(t["best_canonical_image"])
                    if conf > 0.8:
                        t["side_score"] = 0.8 + (conf * 0.2) if side == "front" else 0.2 - (conf * 0.2)
                except Exception:
                    pass

        # Sort: side_score desc, then quality-weighted composite desc
        session_tracks.sort(
            key=lambda t: (
                -float(t.get("side_score", 0.0)),
                -_compute_quality_weighted_score_dict(t, max_length),
            )
        )

        primary = session_tracks[0]
        primary["angle"] = "Front"
        primary["duplicate_track_index"] = None
        session_confidences.append(float(primary.get("quality_score", 0.0)))
        if primary.get("frame_entries"):
            session_purity.append(
                float(primary["frame_entries"][0].get("triage_metrics", {}).get("border_purity", 1.0))
            )

        primary_index_in_active = active_tracks.index(primary)
        primary_phash = (primary.get("frame_entries") or [{}])[0].get("visual_hash")

        for other in session_tracks[1:]:
            same_card = False

            emb_primary = primary.get("reid_embedding")
            emb_other = other.get("reid_embedding")
            if emb_primary is not None and emb_other is not None:
                same_card = embedding_same_card_score(
                    np.array(emb_primary), np.array(emb_other), threshold=0.5
                )

            if not same_card:
                other_phash = (other.get("frame_entries") or [{}])[0].get("visual_hash")
                if primary_phash and other_phash:
                    ham = deduplicator.hamming_distance(primary_phash, other_phash)
                    session_hamming.append(float(ham))
                    adaptive_ham = adaptive_computer.compute_hamming_threshold(
                        observed, global_threshold=15.0
                    )
                    same_card = ham <= adaptive_ham

            if same_card:
                other["duplicate_track_index"] = primary_index_in_active
                other["angle"] = "Back"
            else:
                other["duplicate_track_index"] = None
                other["angle"] = "Front"
                session_confidences.append(float(other.get("quality_score", 0.0)))
                if other.get("frame_entries"):
                    session_purity.append(
                        float(other["frame_entries"][0].get("triage_metrics", {}).get("border_purity", 1.0))
                    )

        # Hard-case capture (active learning)
        try:
            reason = is_hard_case({
                "video_id": int(state.get("video_id", 0)),
                "session_id": str(sid),
                "front_tracks": session_tracks,
                "hamming_values": session_hamming,
                "confidence_scores": session_confidences,
                "border_purity_scores": session_purity,
            })
            if reason and output_dir is not None:
                capture_hard_case(
                    {
                        "video_id": int(state.get("video_id", 0)),
                        "session_id": str(sid),
                        "front_tracks": session_tracks,
                        "hamming_values": session_hamming,
                        "confidence_scores": session_confidences,
                        "border_purity_scores": session_purity,
                    },
                    reason,
                    output_file=str(Path(output_dir) / "hard_cases.jsonl"),
                )
        except Exception as exc:
            telemetry.resource_sample({"event": "hard_case_capture_failed", "error": repr(exc)})

    state["prepared_tracks"] = active_tracks
```

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_resolve_stage.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/resolve.py tests/pipeline/stages/test_resolve_stage.py
git commit -m "feat(v55-stages): resolve stage — F/B + same-card grouping

Verbatim V4 port. Substitutions documented in module docstring:
- cv2.imread(best_canonical_image_path) -> best_canonical_image (ndarray)
- FBPredictor.predict(path) -> predict_array(ndarray)
- ctx.observed_intra_track_distances -> state lookup with default []
- ctx.db_path / ctx.output_dir -> state lookups

Session grouping, longest-track-as-front heuristic, F/B classifier
high-conf override, embedding-then-phash same-card detection, and
hard-case capture all preserved."
```

### Phase 6 acceptance

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_resolve_stage.py -q`
Expected: 4 passed.

---

## Phase 7 — Port `fuse` stage (MultiFrameFuser + foil)

V4 `fuse.py` ran once per track inside a Metaflow `foreach`. v5.5 runs the loop in-process. Same algorithm.

### Task 7.1: `fuse` stage — in-process loop with foil-aware fusion

**Files:**
- Modify: `src/card_capture/pipeline/stages/fuse.py`
- Test: `tests/pipeline/stages/test_fuse_stage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/stages/test_fuse_stage.py`:

```python
"""Phase 7 — fuse stage produces one fused_canonical per prepared track."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from card_capture.pipeline.stages import fuse as fuse_stage


def _prepared_track(instance_id, n_canonical=4):
    canonical = (np.random.RandomState(0).rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "side_score": 0.7,
        "appearance_vector": [0.1, 0.2, 0.3],
        "best_canonical_detection_id": 1,
        "duplicate_track_index": None,
        "first_frame_index": 5,
        "reid_embedding": [0.5, 0.5, 0.0, 0.0],
        "best_canonical_image": canonical,
        "frame_entries": [
            {"normalized": canonical, "is_canonical": True,
             "visual_hash": "abcd", "quality_score": 0.8,
             "image_path": ""}
            for _ in range(n_canonical)
        ],
    }


def test_fuse_emits_one_record_per_prepared_track():
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": True,
                      "fusion_target_frames": 4}
    state = {
        "request": request,
        "prepared_tracks": [_prepared_track("a"), _prepared_track("b")],
    }
    fuse_stage.run(state, telemetry=MagicMock())
    assert len(state["fused_canonicals"]) == 2
    for fc in state["fused_canonicals"]:
        assert isinstance(fc["fused_image"], np.ndarray)
        assert fc["fused_image"].shape == (1050, 750, 3)
        assert fc["primary_hash"] == "abcd"


def test_fuse_single_frame_passthrough_when_target_is_one():
    """fusion_target_frames=1 → fused_image == best_canonical_image."""
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": True,
                      "fusion_target_frames": 1}
    track = _prepared_track("c", n_canonical=4)
    state = {"request": request, "prepared_tracks": [track]}
    fuse_stage.run(state, telemetry=MagicMock())
    fc = state["fused_canonicals"][0]
    assert np.array_equal(fc["fused_image"], track["best_canonical_image"])


def test_fuse_passes_foil_threshold_when_enabled():
    """enable_foil_aware_fusion=True → MultiFrameFuser.fuse called with foil_threshold=50.0."""
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": True,
                      "fusion_target_frames": 4}
    captured = {}

    class _StubFuser:
        def fuse(self, images, foil_threshold=None):
            captured["foil_threshold"] = foil_threshold
            return images[0]

    with patch("card_capture.fuser.MultiFrameFuser", _StubFuser):
        state = {"request": request, "prepared_tracks": [_prepared_track("d")]}
        fuse_stage.run(state, telemetry=MagicMock())

    assert captured["foil_threshold"] == 50.0


def test_fuse_skips_foil_when_disabled():
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": False,
                      "fusion_target_frames": 4}
    captured = {}

    class _StubFuser:
        def fuse(self, images, foil_threshold=None):
            captured["foil_threshold"] = foil_threshold
            return images[0]

    with patch("card_capture.fuser.MultiFrameFuser", _StubFuser):
        state = {"request": request, "prepared_tracks": [_prepared_track("e")]}
        fuse_stage.run(state, telemetry=MagicMock())

    assert captured["foil_threshold"] is None
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_fuse_stage.py -v`
Expected: 4 FAILED — current `fuse.py` writes `{"fused_canonical": None}` per track.

- [ ] **Step 3: Replace `stages/fuse.py` with the in-process port**

Replace `src/card_capture/pipeline/stages/fuse.py`:

```python
"""Stage 9: Lighting-Diverse Fusion.

V5.5 change: was a Metaflow ``foreach`` (one subprocess per track, ~4–6
minutes overhead on the reference video). Now runs the fusion loop
in-process with a plain ``for``. Algorithm unchanged — see
:mod:`card_capture.fuser`.

Substitutions vs V4 ``pipeline/steps/fuse.py``:
- ``cv2.imread(fe['image_path'])`` -> ``fe['normalized']``
- ``cv2.imwrite(fused_path, fused_img)`` -> ``fused['fused_image'] = img``
- ``shutil.copy(best_path, fused_path)`` (single-frame path) -> direct
  passthrough of ``prepared_track['best_canonical_image']``
"""
from __future__ import annotations

from typing import Any, Dict, List


def run(state: dict, *, telemetry) -> None:
    config = state["request"].config
    prepared_tracks = state.get("prepared_tracks") or []

    fusion_target = int(config.get("fusion_target_frames", 1))
    enable_foil = bool(config.get("enable_foil_aware_fusion", True))
    foil_threshold = float(config.get("foil_threshold", 50.0))

    fused_canonicals: List[Dict[str, Any]] = []

    from card_capture.fuser import MultiFrameFuser

    for track in prepared_tracks:
        instance_id = track["instance_id"]
        frame_entries = track.get("frame_entries", [])
        canonical_entries = [fe for fe in frame_entries if fe.get("is_canonical")]

        if not canonical_entries:
            # Fallback: use best_canonical_image directly
            fused_img = track.get("best_canonical_image")
            primary_hash = (frame_entries[0].get("visual_hash", "") if frame_entries else "")
            quality_score = float(frame_entries[0].get("quality_score", 0.0) if frame_entries else 0.0)
        else:
            images = [fe["normalized"] for fe in canonical_entries if fe.get("normalized") is not None]
            primary_hash = str(canonical_entries[0].get("visual_hash", ""))
            quality_score = float(canonical_entries[0].get("quality_score", 0.0))

            if not images:
                fused_img = track.get("best_canonical_image")
            elif len(images) == 1 or fusion_target <= 1:
                # Single-frame passthrough
                fused_img = track.get("best_canonical_image", images[0])
            else:
                try:
                    fuser = MultiFrameFuser()
                    fused_img = fuser.fuse(
                        images,
                        foil_threshold=foil_threshold if enable_foil else None,
                    )
                except Exception as exc:
                    telemetry.resource_sample(
                        {"event": "fusion_failed",
                         "instance_id": instance_id, "error": repr(exc)}
                    )
                    fused_img = track.get("best_canonical_image", images[0])

        fused_canonicals.append({
            "instance_id": instance_id,
            "session_id": int(track.get("session_id", 0)),
            "angle": track.get("angle", "Unknown"),
            "fused_image": fused_img,
            "primary_hash": primary_hash,
            "quality_score": quality_score,
            "side_score": float(track.get("side_score", 0.0)),
            "appearance_vector": list(track.get("appearance_vector", [])),
            "best_canonical_detection_id": int(track.get("best_canonical_detection_id", 0)),
            "duplicate_track_index": track.get("duplicate_track_index"),
            "first_frame_index": int(track.get("first_frame_index", -1)),
            "reid_embedding": track.get("reid_embedding"),
        })

    state["fused_canonicals"] = fused_canonicals
```

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_fuse_stage.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/fuse.py tests/pipeline/stages/test_fuse_stage.py
git commit -m "feat(v55-stages): fuse stage in-process loop (MultiFrameFuser + foil)

V4 algorithm preserved verbatim. foreach -> for. Image IO removed
(was cv2.imread/imwrite; now in-memory ndarrays). fusion_target_frames=1
short-circuits to best_canonical_image (matches V4 shutil.copy path
in observable behavior)."
```

### Phase 7 acceptance

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_fuse_stage.py -q`
Expected: 4 passed.

---

## Phase 8 — Port `dedup` stage (pHash + DINOv2 cross-video)

### Task 8.1: `dedup` — intra-run grouping + cross-video query

**Files:**
- Modify: `src/card_capture/pipeline/stages/dedup.py`
- Test: `tests/pipeline/stages/test_dedup_stage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/stages/test_dedup_stage.py`:

```python
"""Phase 8 — dedup stage groups duplicate instances within + across runs."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import dedup as dedup_stage


def _fused(instance_id, embedding=None, primary_hash="0" * 16):
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "primary_hash": primary_hash,
        "reid_embedding": list(embedding) if embedding is not None else None,
    }


def test_dedup_intra_run_groups_by_close_embedding():
    request = MagicMock()
    e_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_b = e_a + np.array([0.01, 0.01, 0.0, 0.0], dtype=np.float32)
    e_b = e_b / np.linalg.norm(e_b)
    state = {
        "request": request,
        "fused_canonicals": [_fused("a", e_a), _fused("b", e_b)],
        "video_id": 1,
        "repos": {"cards": _StubRepoNoCrossVideo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    assert len(state["dedup_groups"]) == 1
    g = state["dedup_groups"][0]
    assert g["canonical_instance_id"] == "a"
    assert "b" in g["duplicate_instance_ids"]


def test_dedup_intra_run_groups_by_phash_when_embedding_missing():
    request = MagicMock()
    state = {
        "request": request,
        "fused_canonicals": [
            _fused("p", embedding=None, primary_hash="ffffffffffffffff"),
            _fused("q", embedding=None, primary_hash="ffffffffffffffff"),
        ],
        "video_id": 1,
        "repos": {"cards": _StubRepoNoCrossVideo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    assert len(state["dedup_groups"]) == 1
    g = state["dedup_groups"][0]
    assert "q" in g["duplicate_instance_ids"]


def test_dedup_cross_video_query_excludes_self_video_id():
    """The CardsRepository must be called with video_id=current, not zero."""
    request = MagicMock()
    captured = {}

    class _StubRepo:
        def find_embeddings_excluding_video(self, *, video_id):
            captured["video_id"] = video_id
            return []  # no cross-video matches

    state = {
        "request": request,
        "fused_canonicals": [_fused("x", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))],
        "video_id": 42,
        "repos": {"cards": _StubRepo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    assert captured["video_id"] == 42


def test_dedup_cross_video_match_sets_parent_id():
    request = MagicMock()
    e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    class _StubRepo:
        def find_embeddings_excluding_video(self, *, video_id):
            # Existing card_instance row id=99 with a very close embedding
            return [(99, e.tobytes())]

    state = {
        "request": request,
        "fused_canonicals": [_fused("new", e)],
        "video_id": 7,
        "repos": {"cards": _StubRepo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    g = state["dedup_groups"][0]
    assert g["cross_video_parent_id"] == 99


class _StubRepoNoCrossVideo:
    def find_embeddings_excluding_video(self, *, video_id):
        return []
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_dedup_stage.py -v`
Expected: 4 FAILED — current stage passes through.

- [ ] **Step 3: Replace `stages/dedup.py` with the V4 port**

Replace `src/card_capture/pipeline/stages/dedup.py`:

```python
"""Stage 10: Global Dedup.

Port of V4 ``pipeline/steps/dedup.py``. Substitution: the raw
``Storage._connect()`` query becomes
``CardsRepository.find_embeddings_excluding_video(video_id=...)`` so
no raw SQL leaves ``card_capture.data``.
"""
from __future__ import annotations

from typing import Any, Dict, List


SAME_CARD_EMB_THRESHOLD = 0.15  # DINOv2 cosine distance, identical to V4
SAME_CARD_HAMMING_MAX = 8       # pHash fallback, identical to V4


def run(state: dict, *, telemetry) -> None:
    import numpy as np
    from card_capture.deduplicator import VisualDeduplicator

    deduplicator = VisualDeduplicator()
    fused_canonicals = state.get("fused_canonicals") or []
    cards_repo = (state.get("repos") or {}).get("cards")
    current_video_id = int(state.get("video_id", 0))

    dedup_groups: List[Dict[str, Any]] = []
    processed: set = set()

    for i, f1 in enumerate(fused_canonicals):
        id1 = f1["instance_id"]
        if id1 in processed:
            continue

        group: Dict[str, Any] = {
            "canonical_instance_id": id1,
            "duplicate_instance_ids": [],
            "hamming_distances": {},
            "embedding_distances": {},
            "cross_video_parent_id": None,
        }
        processed.add(id1)
        emb1 = f1.get("reid_embedding")

        # Intra-run
        for f2 in fused_canonicals[i + 1:]:
            id2 = f2["instance_id"]
            if id2 in processed:
                continue

            same = False
            emb2 = f2.get("reid_embedding")
            if emb1 is not None and emb2 is not None:
                dist = 1.0 - float(np.dot(np.array(emb1), np.array(emb2)))
                if dist < SAME_CARD_EMB_THRESHOLD:
                    same = True
                    group["embedding_distances"][id2] = dist
            if not same:
                h1 = f1.get("primary_hash")
                h2 = f2.get("primary_hash")
                if h1 and h2:
                    ham = deduplicator.hamming_distance(h1, h2)
                    if ham <= SAME_CARD_HAMMING_MAX:
                        same = True
                        group["hamming_distances"][id2] = float(ham)

            if same:
                group["duplicate_instance_ids"].append(id2)
                processed.add(id2)

        # Cross-video
        if emb1 is not None and cards_repo is not None:
            try:
                rows = cards_repo.find_embeddings_excluding_video(video_id=current_video_id)
                best_id = None
                best_dist = 1.0
                emb1_arr = np.array(emb1, dtype=np.float32)
                for row_id, blob in rows:
                    other = np.frombuffer(blob, dtype=np.float32)
                    if other.shape != emb1_arr.shape:
                        continue
                    dist = 1.0 - float(np.dot(emb1_arr, other))
                    if dist < best_dist:
                        best_dist = dist
                        best_id = row_id
                if best_id is not None and best_dist < SAME_CARD_EMB_THRESHOLD:
                    group["cross_video_parent_id"] = int(best_id)
            except Exception as exc:
                telemetry.resource_sample(
                    {"event": "cross_video_dedup_failed", "error": repr(exc)}
                )

        dedup_groups.append(group)

    state["dedup_groups"] = dedup_groups
    state["final_cards"] = fused_canonicals  # store stage uses this
```

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_dedup_stage.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/dedup.py tests/pipeline/stages/test_dedup_stage.py
git commit -m "feat(v55-stages): dedup stage — pHash + DINOv2 cross-video

V4 algorithm preserved. Substitution: raw Storage._connect SELECT
becomes CardsRepository.find_embeddings_excluding_video. Constants
SAME_CARD_EMB_THRESHOLD (0.15) and SAME_CARD_HAMMING_MAX (8) match
V4 verbatim. cross_video_parent_id excludes self video_id."
```

### Phase 8 acceptance

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_dedup_stage.py -q`
Expected: 4 passed.

---

## Phase 9 — Port `store` stage via repositories + image-write boundary

The store stage is the only filesystem-write boundary. Writes ~16 JPEGs per track + DB rows via CardsRepository.

### Task 9.1: `store` stage — repository writes + crops directory

**Files:**
- Modify: `src/card_capture/pipeline/stages/store.py`
- Test: `tests/pipeline/stages/test_store_stage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/stages/test_store_stage.py`:

```python
"""Phase 9 — store stage writes images + DB rows; produces final_cards."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import store as store_stage


def _fused(instance_id, image=None):
    if image is None:
        image = (np.random.RandomState(hash(instance_id) & 0xFFFFFFFF)
                 .rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "fused_image": image,
        "primary_hash": "deadbeef",
        "quality_score": 0.8,
        "side_score": 0.7,
        "appearance_vector": [],
        "best_canonical_detection_id": 1,
        "duplicate_track_index": None,
        "first_frame_index": 5,
        "reid_embedding": [0.5, 0.5, 0.0, 0.0],
    }


def _prepared(instance_id):
    img = (np.random.RandomState(0).rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "best_canonical_detection_id": 1,
        "frame_entries": [
            {
                "detection_id": 1,
                "frame_index": 5,
                "timestamp_ms": 165,
                "normalized": img,
                "image_path": "",
                "is_canonical": True,
                "quality_score": 0.8,
                "quality_components": {"sharpness": 0.7},
                "confidence": 0.9,
                "corners": [(0, 0), (750, 0), (750, 1050), (0, 1050)],
                "glare_x": None, "glare_y": None,
                "sharpness": 0.7,
            },
        ],
    }


class _StubRepo:
    def __init__(self):
        self.added_instances = []
        self.added_views = []
        self.added_saved = []
        self.fusion_updates = []
        self.dedup_updates = []
        self._next_id = 100

    def add_card_instance(self, **kw):
        rid = self._next_id
        self._next_id += 1
        self.added_instances.append((rid, kw))
        return rid

    def update_instance_deduplication(self, **kw):
        self.dedup_updates.append(kw)

    def update_instance_fusion(self, **kw):
        self.fusion_updates.append(kw)

    def add_card_view(self, **kw):
        vid = self._next_id
        self._next_id += 1
        self.added_views.append((vid, kw))
        return vid

    def add_saved_card(self, **kw):
        self.added_saved.append(kw)

    def add_track_telemetry(self, **kw):
        pass

    def add_pipeline_event(self, **kw):
        pass


def _state(tmp_path, instance_id="t-aaaaaaaa"):
    request = MagicMock()
    request.config = {}
    request.run_id = "r1"
    repos = {"cards": _StubRepo(), "runs": MagicMock()}
    return {
        "request": request,
        "video_id": 42,
        "output_root": tmp_path,
        "fused_canonicals": [_fused(instance_id)],
        "prepared_tracks": [_prepared(instance_id)],
        "dedup_groups": [{
            "canonical_instance_id": instance_id,
            "duplicate_instance_ids": [],
            "hamming_distances": {},
            "embedding_distances": {},
            "cross_video_parent_id": None,
        }],
        "repos": repos,
    }


def test_store_writes_fused_image_to_crops_dir(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    crops_dir = tmp_path / "crops"
    assert crops_dir.exists()
    fused_files = list(crops_dir.glob("instance_*_fused.jpg"))
    assert len(fused_files) == 1


def test_store_writes_rectified_jpeg_per_frame_entry(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    crops_dir = tmp_path / "crops"
    rectified = list(crops_dir.glob("track_*_det_*_rectified.jpg"))
    assert len(rectified) == 1


def test_store_calls_add_card_instance_with_run_id(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    repo = state["repos"]["cards"]
    assert len(repo.added_instances) == 1
    rid, kw = repo.added_instances[0]
    assert kw["run_id"] == "r1"
    assert kw["video_id"] == 42


def test_store_best_view_points_to_fused_path(tmp_path):
    """V4 line 98 (A1): canonical best view's rectified_path = fused_image_path."""
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    repo = state["repos"]["cards"]
    best_view_kw = [kw for _, kw in repo.added_views if kw["is_canonical"]][0]
    fusion_kw = repo.fusion_updates[0]
    assert best_view_kw["rectified_path"] == fusion_kw["fused_image_path"]


def test_store_populates_final_cards_in_state(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    assert len(state["cards"]) == 1
    assert state["cards"][0]["instance_id"] == "t-aaaaaaaa"


def test_store_marks_run_completed_with_card_count(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    runs_repo = state["repos"]["runs"]
    runs_repo.mark_completed.assert_called_once_with("r1", cards_extracted=1)
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_store_stage.py -v`
Expected: 6 FAILED — current store stage hardcodes `final_cards = []`.

- [ ] **Step 3: Replace `stages/store.py` with the repository-backed port**

Replace `src/card_capture/pipeline/stages/store.py`:

```python
"""Stage 10b: Storage via repositories.

Port of V4 ``pipeline/steps/store.py``. All ``Storage`` calls replaced
with ``CardsRepository`` methods (Phase 2). Image writes happen here
and ONLY here, matching the V5.5 in-memory mandate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    run_id = request.run_id
    video_id = int(state.get("video_id", 0))
    if video_id == 0:
        telemetry.contract_violation(
            "store_without_video_id", {"hint": "runtime must inject video_id"}
        )
        raise RuntimeError("store stage reached without a real video_id")

    cards_repo = state["repos"]["cards"]
    runs_repo = state["repos"]["runs"]
    fused_canonicals: List[Dict[str, Any]] = state.get("fused_canonicals", [])
    prepared_tracks: List[Dict[str, Any]] = state.get("prepared_tracks", [])
    dedup_groups: List[Dict[str, Any]] = state.get("dedup_groups", [])

    output_root: Path = state["output_root"]
    crops_dir = output_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Write all images to disk. After this block every fused_canonical
    #    has a ``fused_image_path`` and every frame_entry has an
    #    ``image_path`` — the DB writes below reference these paths.
    # ------------------------------------------------------------------
    for fused in fused_canonicals:
        iid = fused["instance_id"]
        path = crops_dir / f"instance_{iid[:8]}_fused.jpg"
        cv2.imwrite(str(path), fused["fused_image"])
        fused["fused_image_path"] = str(path)

    fused_by_iid: Dict[str, Dict[str, Any]] = {f["instance_id"]: f for f in fused_canonicals}
    track_by_iid: Dict[str, Dict[str, Any]] = {t["instance_id"]: t for t in prepared_tracks}

    for track in prepared_tracks:
        iid = track["instance_id"]
        for fe in track.get("frame_entries", []):
            det_id = int(fe["detection_id"])
            view_path = crops_dir / f"track_{iid[:8]}_det_{det_id}_rectified.jpg"
            cv2.imwrite(str(view_path), fe["normalized"])
            fe["image_path"] = str(view_path)

    # ------------------------------------------------------------------
    # 2. Persist card_instances + card_views via the repository.
    # ------------------------------------------------------------------
    id_map: Dict[str, int] = {}
    final_cards: List[Dict[str, Any]] = []

    for fused in fused_canonicals:
        iid = fused["instance_id"]
        track = track_by_iid.get(iid, {})

        embedding_bytes: bytes | None = None
        if fused.get("reid_embedding") is not None:
            embedding_bytes = np.array(fused["reid_embedding"], dtype=np.float32).tobytes()
        else:
            # V4 fallback: compute from the fused image so the row never
            # has a NULL embedding (used by cross-video dedup in future runs).
            try:
                from card_capture.ml.embeddings import compute_reid_embedding_array
                emb = compute_reid_embedding_array(fused["fused_image"])
                embedding_bytes = emb.astype(np.float32).tobytes()
            except Exception as exc:
                cards_repo.add_pipeline_event(
                    video_id=video_id, frame_index=0, timestamp_ms=0,
                    event_type="reid_embedding_failed",
                    data={"instance_id": iid, "error": repr(exc)},
                )

        row_id = cards_repo.add_card_instance(
            video_id=video_id,
            track_id=iid,
            angle=fused["angle"],
            session_id=str(fused["session_id"]),
            reid_embedding=embedding_bytes,
            run_id=run_id,
        )
        id_map[iid] = row_id

        cards_repo.update_instance_deduplication(
            row_id=row_id,
            primary_hash=fused["primary_hash"],
            cross_video_parent=None,
            reid_embedding=embedding_bytes,
        )
        cards_repo.update_instance_fusion(
            row_id=row_id,
            fused_image_path=fused["fused_image_path"],
        )

        best_det_id = int(track.get("best_canonical_detection_id", -1))
        for fe in track.get("frame_entries", []):
            is_best = int(fe["detection_id"]) == best_det_id
            view_path = fused["fused_image_path"] if is_best else fe["image_path"]

            view_id = cards_repo.add_card_view(
                card_instance_id=row_id,
                frame_index=int(fe["frame_index"]),
                timestamp_ms=int(fe["timestamp_ms"]),
                corners=fe["corners"],
                confidence=float(fe["confidence"]),
                rectified_path=view_path,
                quality_score=fe.get("quality_components", {}),
                is_canonical=bool(fe.get("is_canonical", False)),
                glare_x=fe.get("glare_x"),
                glare_y=fe.get("glare_y"),
                sharpness=fe.get("sharpness"),
                initial_confidence=float(fe["confidence"]),
            )
            if fe.get("is_canonical") and is_best:
                cards_repo.add_saved_card(
                    detection_id=view_id,
                    image_path=view_path,
                    final_score=float(fe["quality_score"]),
                )

        final_cards.append({
            "instance_id": iid,
            "row_id": row_id,
            "video_id": video_id,
            "run_id": run_id,
            "angle": fused["angle"],
            "fused_image_path": fused["fused_image_path"],
        })

    # ------------------------------------------------------------------
    # 3. Persist dedup links (intra-run + cross-video).
    # ------------------------------------------------------------------
    for group in dedup_groups:
        canonical_iid = group["canonical_instance_id"]
        if canonical_iid not in id_map:
            continue
        canonical_row_id = id_map[canonical_iid]

        cross_parent = group.get("cross_video_parent_id")
        if cross_parent:
            cards_repo.update_instance_deduplication(
                row_id=canonical_row_id,
                primary_hash=fused_by_iid[canonical_iid]["primary_hash"],
                cross_video_parent=int(cross_parent),
            )
        for dup_iid in group["duplicate_instance_ids"]:
            if dup_iid not in id_map:
                continue
            cards_repo.update_instance_deduplication(
                row_id=id_map[dup_iid],
                primary_hash=fused_by_iid[dup_iid]["primary_hash"],
                cross_video_parent=canonical_row_id,
            )

    state["cards"] = final_cards
    state["output_artifacts"] = [str(crops_dir / "*.jpg")]
    runs_repo.mark_completed(run_id, cards_extracted=len(final_cards))
```

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_store_stage.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/store.py tests/pipeline/stages/test_store_stage.py
git commit -m "feat(v55-stages): store stage via CardsRepository + image writes

Verbatim V4 port. Substitutions:
- Storage.add_card_instance/update_*/add_card_view -> CardsRepository
- cv2.imwrite happens here once and only once (in-memory mandate)
- compute_reid_embedding(path) fallback -> compute_reid_embedding_array

V4 line 98 (A1) preserved: canonical best view's rectified_path
equals fused_image_path. Cross-video dedup links applied last.
runs_repo.mark_completed fires with the real cards count."
```

### Phase 9 acceptance

After Phase 9 the pipeline is functionally complete end-to-end.

Run: `.venv/bin/python -m pytest tests/pipeline/stages/ tests/data/test_cards_repository_writes.py -q`
Expected: all green.

---

## Phase 10 — Synthetic e2e fixture + assert `cards > 0`

### Task 10.1: Generate a deterministic 2-card synthetic MOV

**Files:**
- Create: `tests/pipeline/conftest.py`
- Test: `tests/pipeline/test_back_half_e2e.py`

- [ ] **Step 1: Write the fixture**

Create `tests/pipeline/conftest.py`:

```python
"""Synthetic test fixtures for the v5.5 back-half e2e tests.

We synthesise a tiny 480p MOV where two static rectangles ("cards")
appear in front of a checkerboard background. Deterministic via a
fixed seed; cached on disk between test runs.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


_CACHE = Path(__file__).parent / "fixtures"
_CACHE.mkdir(exist_ok=True)


def _make_checkerboard(h: int, w: int, square: int = 60) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, square):
        for x in range(0, w, square):
            if ((x // square) + (y // square)) % 2 == 0:
                img[y:y + square, x:x + square] = (180, 180, 180)
            else:
                img[y:y + square, x:x + square] = (60, 60, 60)
    return img


def _make_card(color, label, h=300, w=210) -> np.ndarray:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.putText(img, label, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 3)
    # Sprinkle texture so QualityScorer's sharpness component fires
    rng = np.random.RandomState(42)
    noise = rng.randint(-15, 15, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture(scope="session")
def synthetic_two_cards_mov() -> Path:
    """A 4-second 480x640 MOV with two cards held in succession."""
    out = _CACHE / "synthetic_two_cards.mov"
    if out.exists():
        return out

    w, h, fps, secs = 640, 480, 30, 4
    n_frames = fps * secs
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open mp4v encoder on this platform")

    bg = _make_checkerboard(h, w)
    card_a = _make_card((40, 80, 200), "A")    # red-ish
    card_b = _make_card((200, 120, 40), "B")   # blue-ish

    for i in range(n_frames):
        frame = bg.copy()
        # First half: card A near top-left; second half: card B near center
        if i < n_frames // 2:
            card, x, y = card_a, 120, 80
        else:
            card, x, y = card_b, 220, 100
        frame[y:y + card.shape[0], x:x + card.shape[1]] = card
        writer.write(frame)

    writer.release()
    return out
```

- [ ] **Step 2: Write the e2e test**

Create `tests/pipeline/test_back_half_e2e.py`:

```python
"""Phase 10 — synthetic e2e: cards > 0 after a full run."""
from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


def _init_db(path: Path) -> None:
    import sqlite3
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE pipeline_runs (
                run_id TEXT PRIMARY KEY, video_id INTEGER, status TEXT,
                cards_extracted INTEGER DEFAULT 0, started_at TEXT, finished_at TEXT
            );
            CREATE TABLE card_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL, track_id TEXT NOT NULL,
                angle TEXT, session_id TEXT,
                reid_embedding BLOB, run_id TEXT, primary_hash TEXT,
                is_duplicate_of INTEGER, fused_image_path TEXT
            );
            CREATE TABLE card_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_instance_id INTEGER NOT NULL, frame_index INTEGER,
                timestamp_ms INTEGER, corners TEXT, confidence REAL,
                rectified_path TEXT, quality_score TEXT, is_canonical INTEGER,
                glare_x REAL, glare_y REAL, sharpness REAL, initial_confidence REAL
            );
            CREATE TABLE saved_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT, detection_id INTEGER,
                image_path TEXT, final_score REAL
            );
            CREATE TABLE track_telemetry (
                video_id INTEGER, instance_id TEXT, frame_index INTEGER,
                area REAL, aspect REAL, cx REAL, cy REAL
            );
            CREATE TABLE pipeline_events (
                video_id INTEGER, frame_index INTEGER, timestamp_ms INTEGER,
                event_type TEXT, data TEXT
            );
        """)


def test_back_half_e2e_produces_cards(synthetic_two_cards_mov, tmp_path):
    db = tmp_path / "cards.sqlite"
    _init_db(db)

    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id="e2e-1",
        input_video=f"artifact://local/{synthetic_two_cards_mov}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
        config={
            "detector": "fake",   # Synthesises 2 corner detections per frame
            "device": "cpu",
            "use_kornia": True,
            "kornia_device": "cpu",
            "rotate_180": False,
            "tracker_backend": "bytetrack",
            "min_track_length": 2,
            "fusion_target_frames": 1,
            "novelty_floor": 0.0,
            "track_confidence_floor": 0.0,
            "stand_novelty_max": 0.0,
            "stand_sharpness_max": 0.0,
            "use_fb_classifier": False,
            "enable_foil_aware_fusion": False,
            "laplacian_scan_stride": 0,
            "max_corner_gap_frames": 30,
            "corner_refinement": False,
        },
        db_path=str(db),
        video_id=1,
    )
    result = runtime.run(req)

    # Stages all fired
    finished = {e.payload["stage"] for e in telemetry.events if e.kind == "stage_finished"}
    expected = {"sample", "detect", "novelty", "track", "refine",
                "score", "resolve", "fuse", "dedup", "store"}
    assert expected <= finished, f"missing stages: {expected - finished}"

    # At least one card persisted
    import sqlite3
    with sqlite3.connect(db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM card_instances WHERE run_id=?", ("e2e-1",)
        ).fetchone()[0]
    assert count >= 1, "store stage did not persist any card_instances"

    # crops/ has fused images
    crops = list((tmp_path / "crops").glob("instance_*_fused.jpg"))
    assert len(crops) >= 1

    # Run marked completed
    with sqlite3.connect(db) as conn:
        status, cards = conn.execute(
            "SELECT status, cards_extracted FROM pipeline_runs WHERE run_id=?", ("e2e-1",)
        ).fetchone()
    assert status == "completed"
    assert cards >= 1
```

- [ ] **Step 3: Run the e2e test**

Run: `.venv/bin/python -m pytest tests/pipeline/test_back_half_e2e.py -v`
Expected: PASSED (the test may take ~10s due to YOLO/Kornia initialization — adjust the synthetic detector if it's slower).

Note: if the `FakeCardDetector` doesn't emit detections from the synthetic frames (its current behavior depends on `state["sampled_frames"]` shape), update `card_capture.detectors.FakeCardDetector` to emit two static rectangles per frame matching the synthetic card positions. Add this small change in the same commit.

- [ ] **Step 4: Commit**

```bash
git add tests/pipeline/conftest.py tests/pipeline/test_back_half_e2e.py
git commit -m "test(v55-stages): synthetic 2-card MOV e2e — asserts cards>0

Deterministic 4-second 480p clip with two rectangles on a
checkerboard. End-to-end runs LocalPipelineRuntime and asserts:
- all 10 stages fired
- card_instances has >=1 row for the run
- crops/ has the fused JPEG
- pipeline_runs.status='completed' with cards_extracted>=1

This is the regression guard against future stub regressions."
```

### Task 10.2: Upgrade `tests/test_unified_runtime.py` to assert cards > 0

**Files:**
- Modify: `tests/test_unified_runtime.py`

- [ ] **Step 1: Append assertion**

In `tests/test_unified_runtime.py`, after the existing `assert expected <= finished_stages` block, add:

```python
    # Phase 10 — back-half is wired; expect at least one persisted card.
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM card_instances").fetchone()[0]
    assert count >= 1, "back-half stages did not produce any cards"
```

- [ ] **Step 2: Run → PASS (because the gating skipif still skips when IMG_5872 is absent)**

Run: `.venv/bin/python -m pytest tests/test_unified_runtime.py -v`
Expected: PASSED or SKIPPED (golden video absent).

- [ ] **Step 3: Commit**

```bash
git add tests/test_unified_runtime.py
git commit -m "test(v55-stages): smoke test asserts cards>0 after run"
```

---

## Phase 11 — Mid-stage progress events via telemetry

### Task 11.1: Add `progress()` to `PipelineTelemetry`

**Files:**
- Modify: `src/card_capture/pipeline/telemetry.py`
- Modify: `app/services/pipeline_telemetry.py`
- Test: `tests/pipeline/test_progress_event.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_progress_event.py`:

```python
"""Phase 11 — PipelineTelemetry.progress emits a stage_progress event."""
from card_capture.pipeline.telemetry import InMemoryTelemetry, NoopTelemetry


def test_noop_progress_is_noop():
    NoopTelemetry().progress("refine", 42, "track 3/7")


def test_inmemory_progress_records_event():
    tel = InMemoryTelemetry()
    tel.progress("refine", 42, "track 3/7")
    progresses = [e for e in tel.events if e.kind == "progress"]
    assert len(progresses) == 1
    p = progresses[0]
    assert p.payload["stage"] == "refine"
    assert p.payload["pct"] == 42
    assert p.payload["detail"] == "track 3/7"


def test_event_bus_telemetry_emits_stage_progress():
    """EventBusTelemetry.progress fires an Event with name=stage_progress."""
    from app.services.event_bus import EventBus, Event
    from app.services.pipeline_telemetry import EventBusTelemetry

    bus = EventBus()
    captured = []
    bus._subscribers["r1"]  # initialize defaultdict entry
    # Subscribe at the dict level so we don't need an event loop
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        q = bus.subscribe("r1")
    finally:
        asyncio.set_event_loop(None)

    tel = EventBusTelemetry(bus, "r1")
    tel.progress("refine", 50, "track 5/10")

    loop.run_until_complete(asyncio.sleep(0))  # flush call_soon_threadsafe
    ev = q.get_nowait()
    assert ev.name == "stage_progress"
    assert ev.payload == {"stage_id": "refine", "pct": 50, "detail": "track 5/10"}
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/test_progress_event.py -v`
Expected: 3 FAILED — method missing.

- [ ] **Step 3: Add to `telemetry.py`**

In `src/card_capture/pipeline/telemetry.py`, extend `PipelineTelemetry` Protocol + `NoopTelemetry` + `InMemoryTelemetry`:

```python
class PipelineTelemetry(Protocol):
    # ... existing methods unchanged ...
    def progress(self, stage: str, pct: int, detail: str) -> None: ...


class NoopTelemetry:
    # ... existing methods unchanged ...
    def progress(self, stage: str, pct: int, detail: str) -> None: ...


class InMemoryTelemetry:
    # ... existing __init__ + methods ...
    def progress(self, stage: str, pct: int, detail: str) -> None:
        self.events.append(TelemetryEvent(
            "progress", {"stage": stage, "pct": int(pct), "detail": detail}
        ))
```

Also extend `OtelMetricsTelemetry` if present with a no-op `progress` (we don't track progress as a metric):

```python
class OtelMetricsTelemetry:
    # ... existing methods ...
    def progress(self, stage: str, pct: int, detail: str) -> None:
        pass
```

- [ ] **Step 4: Add to `EventBusTelemetry`**

In `app/services/pipeline_telemetry.py`, add:

```python
    def progress(self, stage: str, pct: int, detail: str) -> None:
        self._bus.emit(self._run_id, Event(name="stage_progress", payload={
            "stage_id": stage, "pct": int(pct), "detail": detail,
        }))
```

- [ ] **Step 5: Run → PASS**

Run: `.venv/bin/python -m pytest tests/pipeline/test_progress_event.py -v`
Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/pipeline/telemetry.py app/services/pipeline_telemetry.py tests/pipeline/test_progress_event.py
git commit -m "feat(v55-stages): PipelineTelemetry.progress contract + EventBus adapter

Adds progress(stage, pct, detail) to the Protocol + NoopTelemetry +
InMemoryTelemetry + OtelMetricsTelemetry + EventBusTelemetry.
EventBusTelemetry emits the legacy 'stage_progress' SSE event so the
existing SPA consumer keeps working unchanged."
```

### Task 11.2: Wire mid-stage progress emissions

**Files:**
- Modify: `src/card_capture/pipeline/stages/refine.py`
- Modify: `src/card_capture/pipeline/stages/score.py`
- Modify: `src/card_capture/pipeline/stages/resolve.py`
- Modify: `src/card_capture/pipeline/stages/fuse.py`
- Modify: `src/card_capture/pipeline/stages/dedup.py`
- Modify: `src/card_capture/pipeline/stages/store.py`
- Test: `tests/pipeline/stages/test_progress_emission.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/stages/test_progress_emission.py`:

```python
"""Phase 11 — each track-iterating stage emits progress events."""
from unittest.mock import MagicMock

import numpy as np

from card_capture.pipeline.telemetry import InMemoryTelemetry


def _track(iid, frame_count=3):
    return {
        "instance_id": iid,
        "frame_entries": [{"novelty_score": 1.0, "quality_score": 0.8, "score_total": 0.8,
                            "confidence": 0.8, "quality_components": {"sharpness": 0.7}}
                          for _ in range(frame_count)],
    }


def test_score_emits_progress_per_track():
    from card_capture.pipeline.stages import score as score_stage
    request = MagicMock()
    request.config = {"novelty_floor": 0.0, "track_confidence_floor": 0.0,
                      "stand_novelty_max": 0.0, "stand_sharpness_max": 0.0}
    state = {"request": request, "refined_tracks": [_track(f"t{i}") for i in range(4)],
             "bg_model": None}
    tel = InMemoryTelemetry()
    score_stage.run(state, telemetry=tel)
    progresses = [e for e in tel.events if e.kind == "progress" and e.payload["stage"] == "score"]
    assert len(progresses) >= 4
    # Monotonic non-decreasing pct
    pcts = [p.payload["pct"] for p in progresses]
    assert pcts == sorted(pcts)
    assert pcts[-1] == 100
```

(Mirror this test for each of the 6 stages — same shape, different stage names.)

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_progress_emission.py -v`
Expected: FAILED — no stage emits progress yet.

- [ ] **Step 3: Add progress emission to each track-iterating stage**

For each of `score.py`, `resolve.py`, `fuse.py`, `dedup.py`, `store.py`: at the top of the per-track / per-session loop, compute `pct = int(100 * (i + 1) / max(1, total))` and call `telemetry.progress(stage_name, pct, detail)`. Example diff for `score.py`:

```python
    for i, track_dict in enumerate(refined_tracks):
        # ... existing body ...

        if (i + 1) % max(1, len(refined_tracks) // 10) == 0 or i + 1 == len(refined_tracks):
            telemetry.progress(
                "score",
                int(100 * (i + 1) / max(1, len(refined_tracks))),
                f"track {i + 1}/{len(refined_tracks)}",
            )
```

For `refine.py`, emit per-track during the warp+score loop (same pattern). For `dedup.py`, emit every 25% across the outer pairwise loop.

- [ ] **Step 4: Run → PASS for each stage**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_progress_emission.py -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/ tests/pipeline/stages/test_progress_emission.py
git commit -m "feat(v55-stages): emit mid-stage progress (refine/score/resolve/fuse/dedup/store)

UI gets stepping progress bars instead of only stage_started/finished.
pct is monotonic non-decreasing within a stage; final emission is 100."
```

### Phase 11 acceptance

Run: `.venv/bin/python -m pytest tests/pipeline/test_progress_event.py tests/pipeline/stages/test_progress_emission.py -q`
Expected: all green.

---

## Phase 12 — UI integration tests

### Task 12.1: `/api/runs/{id}/cards` populated after a synthetic run

**Files:**
- Test: `tests/app/test_run_to_cards.py`

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_run_to_cards.py`:

```python
"""Phase 12 — UI integration: cards endpoint + SSE progress + harness."""
from pathlib import Path

import pytest


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Build a FastAPI TestClient against a freshly-migrated db."""
    from fastapi.testclient import TestClient
    db = tmp_path / "cards.sqlite"
    # Run the real migrations so card_instances etc. exist
    from migrations.run_migrations import run_migrations
    run_migrations(db)
    monkeypatch.setenv("CARD_CAPTURE_DB", str(db))

    from app.main import create_app
    app = create_app(db_path=db)
    return TestClient(app), db


def test_full_run_populates_cards_endpoint(app_client, synthetic_two_cards_mov, tmp_path):
    client, db = app_client
    # Register the synthetic video
    r = client.post("/api/videos", json={"file_path": str(synthetic_two_cards_mov)})
    assert r.status_code == 201
    video_id = r.json()["id"]

    # Kick off processing — bg task runs the unified runtime in-process
    r = client.post(f"/api/videos/{video_id}/process")
    assert r.status_code == 202
    run_id = r.json()["run_id"]

    # Poll until completed (timeout 60s)
    import time
    deadline = time.time() + 60.0
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}")
        if r.json().get("status") in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert r.json()["status"] == "completed", r.json()

    # Cards endpoint returns the persisted instances
    r = client.get(f"/api/runs/{run_id}/cards")
    assert r.status_code == 200
    cards = r.json()
    assert isinstance(cards, list)
    assert len(cards) >= 1
    assert "fused_image_path" in cards[0]
    assert "angle" in cards[0]
```

(Note: the exact route shapes — `/api/runs/{id}/cards` etc. — must match the current `app/api/` shape. Read `app/api/` first and align the URLs and fields. If the endpoint doesn't exist yet, add a minimal one in this task that returns `card_instances` rows for the run.)

- [ ] **Step 2: Run → FAIL (likely missing route)**

Run: `.venv/bin/python -m pytest tests/app/test_run_to_cards.py::test_full_run_populates_cards_endpoint -v`
Expected: FAILED — either the route doesn't exist or it returns empty.

- [ ] **Step 3: Add `/api/runs/{run_id}/cards` if missing**

Read `app/api/runs.py`. If there's no `cards` sub-route, add it (it must read `card_instances` via `CardsRepository` — no raw SQL):

```python
@router.get("/{run_id}/cards")
def list_run_cards(run_id: str, request: Request):
    from card_capture.data.connection import read_connection
    db_path = request.app.state.db_path
    with read_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, video_id, track_id, angle, session_id, "
            "fused_image_path FROM card_instances WHERE run_id=?",
            (run_id,),
        ).fetchall()
    keys = ("id", "video_id", "track_id", "angle", "session_id", "fused_image_path")
    return [dict(zip(keys, r)) for r in rows]
```

(If the raw SQL violates the import-linter contract, add this as a `CardsRepository.list_by_run_id` method first, then the route delegates.)

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add tests/app/test_run_to_cards.py app/api/runs.py
git commit -m "test(v55-stages): UI cards endpoint populated after synthetic run

End-to-end: upload synthetic MOV -> process -> poll until completed
-> GET /api/runs/{id}/cards -> non-empty card_instances list."
```

### Task 12.2: SSE `stage_progress` events arrive in order

**Files:**
- Test: `tests/app/test_run_to_cards.py`

- [ ] **Step 1: Append failing test**

```python
def test_sse_emits_stage_progress(app_client, synthetic_two_cards_mov):
    """SSE stream emits at least one stage_progress event per back-half stage
    with monotonically non-decreasing pct within each stage."""
    client, _ = app_client
    r = client.post("/api/videos", json={"file_path": str(synthetic_two_cards_mov)})
    video_id = r.json()["id"]
    r = client.post(f"/api/videos/{video_id}/process")
    run_id = r.json()["run_id"]

    events_per_stage: dict[str, list[int]] = {}
    with client.stream("GET", f"/api/runs/{run_id}/events", timeout=60.0) as resp:
        import json
        for raw_line in resp.iter_lines():
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                if payload.get("name") == "stage_progress":
                    p = payload["payload"]
                    events_per_stage.setdefault(p["stage_id"], []).append(int(p["pct"]))
                if payload.get("name") in ("run_completed", "run_failed"):
                    break

    for stage in ("refine", "score", "resolve", "fuse", "dedup", "store"):
        assert stage in events_per_stage, f"no stage_progress for {stage}"
        pcts = events_per_stage[stage]
        assert pcts == sorted(pcts), f"{stage} pct not monotonic: {pcts}"
```

- [ ] **Step 2: Run → FAIL or PASS depending on SSE wiring**

Run: `.venv/bin/python -m pytest tests/app/test_run_to_cards.py::test_sse_emits_stage_progress -v`
Expected: PASS if Phase 11 emissions are correct and the SSE route forwards events untouched. If the SSE route serializes the `Event` dataclass differently, adjust the test's JSON-decode path to match.

- [ ] **Step 3: If failing, align SSE payload**

Read `app/api/runs.py`'s SSE route. Confirm it forwards `Event.payload` under the key `payload` (or whatever the route uses). Either align the test or align the route — but lock the contract.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add tests/app/test_run_to_cards.py
git commit -m "test(v55-stages): SSE stage_progress events monotonic per stage

End-to-end SSE stream during a synthetic run. Asserts each back-half
stage emits at least one progress event and pct is non-decreasing
within a stage."
```

### Task 12.3: Regression harness CLI runs against the synthetic fixture

**Files:**
- Test: `tests/app/test_run_to_cards.py`

- [ ] **Step 1: Append failing test**

```python
def test_regression_harness_runs(synthetic_two_cards_mov, tmp_path):
    """`card-capture harness run` against synthetic fixture should produce
    a parseable metrics row."""
    import subprocess, sys
    db = tmp_path / "cards.sqlite"
    from migrations.run_migrations import run_migrations
    run_migrations(db)
    # Process the synthetic video first
    result = subprocess.run(
        [sys.executable, "-m", "card_capture.cli", "process",
         str(synthetic_two_cards_mov),
         "--output-dir", str(tmp_path / "out"),
         "--db", str(db)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    # Run harness — synthetic ground truth: 2 cards.
    # (Adjust the actual harness CLI form to whatever `card-capture harness` expects.)
    result = subprocess.run(
        [sys.executable, "-m", "card_capture.cli", "harness", "run",
         "--db", str(db),
         "--video-id", "1"],
        capture_output=True, text=True, timeout=60,
    )
    # We don't require specific metric values, only that the harness completes
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2–5: Run → adjust → PASS → commit**

```bash
git add tests/app/test_run_to_cards.py
git commit -m "test(v55-stages): regression harness runs against synthetic"
```

### Phase 12 acceptance

Run: `.venv/bin/python -m pytest tests/app/test_run_to_cards.py -q`
Expected: 3 passed.

Also execute the manual UI smoke checklist from §12.3 of the companion spec:

```
[ ] uvicorn app.main:app --reload starts cleanly
[ ] http://127.0.0.1:8000/ loads the SPA
[ ] Upload synthetic_two_cards.MOV -> appears as 'pending'
[ ] Process -> 'processing', SSE opens
[ ] Stage progress bars step through all 10 stages
[ ] Completes; cards_extracted >= 1
[ ] Review page lists cards with thumbnail
[ ] Labeling page accepts F/B tags
[ ] curl /api/runs/<id>/cards returns the expected JSON
```

Paste the executed checklist into the PR description before merging.

---

## Phase 13 — Per-stage V4 audit document

### Task 13.1: Write `2026-05-29-v55-back-half-audit.md`

**Files:**
- Create: `docs/superpowers/audits/2026-05-29-v55-back-half-audit.md`

- [ ] **Step 1: Create the audit doc**

Create `docs/superpowers/audits/2026-05-29-v55-back-half-audit.md` with one section per ported stage. Template (fill in concretely for each of `refine`, `score`, `resolve`, `fuse`, `dedup`, `store`):

```markdown
# V5.5 Back-Half Stage Audit

Date: 2026-05-29
Auditor: <author>
Ported across commits: <list of phase commit SHAs>

---

## Stage: refine

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/refine.py` (408 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/refine.py` (~250 LOC)
**Ported in commits:** P4.2, P4.3 (commit SHAs <fill in>)

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Sort candidates by `score_total` desc; take top 8 | identical | ✅ | line refine.py:173 -> stages/refine.py:<line> |
| Kornia warp batch per track | identical | ✅ |  |
| `PrecisionNormalizer` CPU fallback | identical | ✅ |  |
| `QualityScorer.score(normalized, conf, novelty=...)` per frame_entry | identical | ✅ |  |
| `find_glare_centroid` + glare_mask + laplacian_heatmap | identical | ✅ |  |
| `_select_canonical_entries` for canonical flag | identical | ✅ |  |
| Per-track Laplacian scan via `_laplacian_select_frames` | identical (uses in-memory frame dict) | ✅ |  |
| Persist `add_track_telemetry` rows per canonical | uses `CardsRepository.add_track_telemetry` | ✅ |  |
| DINOv2 ReID embedding via `DinoEmbedder.embed_image(path)` | `DinoEmbedder.embed_array(ndarray)` | ⚠️ deviation | See Deviations §1 |
| `cv2.imwrite` rectified crops to `crops_dir` | crops kept as ndarray in `frame_entries[*]['normalized']` | ⚠️ deviation | See Deviations §2 |
| Corner refinement when `ctx.corner_refinement` | identical | ✅ |  |

### Deviations

1. **DINOv2 input is in-memory ndarray, not file path.** V4 wrote the rectified JPEG and called `embed_image(path)`. V5.5 calls `embed_array(ndarray)`. **Reason:** V5.5 in-memory mandate. **Risk:** Low. JPEG compression artifacts are absent → embedding is on cleaner pixels. P3 parity test confirms outputs match within 1e-5 when given the same post-JPEG image.

2. **Rectified crops live in-memory until `store`.** **Reason:** V5.5 in-memory mandate. **Risk:** Memory peak (~180MB for reference video; see spec §10.4). Mitigation: spill to disk between refine and score, follow-up if needed.

### Removed (with reason)

- V4 wrote `frame_entries[*]['image_path']` mid-stage. Now empty string until `store` populates it. No external consumer reads `image_path` before `store`.

### Test coverage

- Unit: `tests/pipeline/stages/test_refine_stage.py` (4 tests — identity, normalization shape, embedding attach, telemetry rows)
- E2E: covered by `tests/pipeline/test_back_half_e2e.py`
- Golden-set regression: covered in P14 — `card_recall` / `card_precision` / `image_quality(SSIM)` within ±0.05

---

## Stage: score

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/score.py` (180 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/score.py`
**Ported in commits:** P5.1

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| `_novelty_gate_useful` returns True iff n≥5, std>0.15, min<0.35 | identical | ✅ | (constants confirmed) |
| Adaptive novelty threshold = midpoint of largest gap, capped at `ctx.novelty_floor` | identical, capped at `config['novelty_floor']` | ✅ | |
| Gate stays off when bg_model is None | identical | ✅ | |
| Confidence-floor prune when `ctx.track_confidence_floor > 0` AND median_quality < floor | identical with `config['track_confidence_floor']` | ✅ | |
| Stand prune requires bg_model AND stand_nov_max > 0 AND median_novelty < stand_nov_max AND median_sharpness < stand_shp_max | identical | ✅ | (both preconditions present) |
| Append `pruned`, `median_novelty`, `median_quality`, `median_sharpness` to track dict | identical | ✅ | |
| Emit `pruned_instance_ids: List[str]` | identical | ✅ | |
| Stage-level print line summarising counts | replaced by `telemetry.resource_sample` | ⚠️ deviation | See Deviations §1 |

### Deviations

1. **Stage summary log line.** V4 printed `[Stage: Score] | …` to stdout; V5.5 routes equivalent counts through telemetry. **Reason:** PipelineRunner already captures telemetry; stdout would duplicate. **Risk:** None.

### Removed

- V4 imported `from pipeline.steps.start import RunContext`; V5.5 doesn't need that since `state["request"].config` carries all knobs.

### Test coverage

- Unit: `tests/pipeline/stages/test_score_stage.py` (4 tests: passthrough when off, confidence-floor prune, novelty-gate-useful, adaptive threshold)
- E2E: `tests/pipeline/test_back_half_e2e.py`
- Golden-set: Phase 14 covers card_recall / precision deltas; ±0.05 gates

---

## Stage: resolve

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/resolve.py` (234 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/resolve.py`
**Ported in commits:** P6.1

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Filter out `pruned` tracks (`active_tracks = [t for t in scored if not t['pruned']]`) | identical | ✅ | |
| Group active tracks by `session_id` | identical | ✅ | |
| Compute `side_score` via `_side_textiness_score(img)` per track | identical (img from state, not disk) | ✅ | |
| Compute `appearance_vector` via `_appearance_vector(img)` | identical | ✅ | |
| F/B classifier override: side="front", conf>0.8 → `side_score = 0.8 + conf*0.2` | identical magic numbers | ✅ | |
| F/B classifier override: side="back",  conf>0.8 → `side_score = 0.2 - conf*0.2` | identical magic numbers | ✅ | |
| Primary sort: `(-side_score, -_compute_quality_weighted_score_dict(t, max_length))` | identical | ✅ | |
| `_compute_quality_weighted_score_dict` weights: 0.6 * normalized_length + 0.4 * mean_quality_of_canonicals | identical | ✅ | |
| First sorted track → angle="Front", duplicate_track_index=None | identical | ✅ | |
| Same-card via embedding `threshold=0.5` (cosine distance, not score) | identical | ✅ | |
| pHash fallback via `deduplicator.hamming_distance` + `AdaptiveThresholdComputer` with `global_threshold=15.0` | identical | ✅ | |
| Same-card → angle="Back", `duplicate_track_index = active_tracks.index(primary)` | identical | ✅ | |
| Hard-case capture via `is_hard_case` / `capture_hard_case` to `output_dir/hard_cases.jsonl` | identical (uses `state["output_root"]` as path) | ✅ | (gracefully no-ops if output_root is None) |
| Returns `prepared_tracks` with side metadata attached | written into `state["prepared_tracks"]` | ✅ | |
| `cv2.imread(t["best_canonical_image_path"])` | `t["best_canonical_image"]` ndarray | ⚠️ deviation | See Deviations §1 |
| `FBPredictor.predict(path)` | `FBPredictor.predict_array(ndarray)` | ⚠️ deviation | See Deviations §2 |
| `ctx.observed_intra_track_distances` | `state.get("observed_intra_track_distances", [])` | ⚠️ deviation | See Deviations §3 |

### Deviations

1. **Best-canonical image is in-memory ndarray, not file path.** **Reason:** V5.5 in-memory mandate. **Risk:** None.
2. **F/B classifier consumes ndarray.** P3 parity test confirms `predict_array(re_read) == predict(path)`. **Reason:** V5.5 in-memory mandate. **Risk:** None.
3. **`observed_intra_track_distances` always empty.** V4 populated this incrementally during refine; we don't currently surface it. **Reason:** Out of scope this phase; AdaptiveThresholdComputer degrades to its `global_threshold=15.0` default. **Risk:** Low — same fallback behavior V4 used when running on a fresh DB.

### Removed

- The internal `_capture_hard_cases` helper from V4 — `capture_hard_case` is the public function.

### Test coverage

- Unit: `tests/pipeline/stages/test_resolve_stage.py` (4 tests: pruned excluded, longest→Front, embedding same-card→Back, pHash fallback)
- E2E: `tests/pipeline/test_back_half_e2e.py`
- Golden-set: side_accuracy ±0.05 in Phase 14

---

## Stage: fuse

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/fuse.py` (121 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/fuse.py`
**Ported in commits:** P7.1

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Per-track loop (one Metaflow subprocess each in V4; in-process `for` here) | identical algorithm | ✅ | |
| Filter `frame_entries` by `is_canonical` | identical | ✅ | |
| `cv2.imread(fe["image_path"])` for each canonical | replaced by `fe["normalized"]` (ndarray) | ⚠️ deviation | See Deviations §1 |
| Single-frame passthrough when `len(images)==1` or `fusion_target_frames <= 1` | identical (returns `best_canonical_image`) | ✅ | |
| `MultiFrameFuser().fuse(images, foil_threshold=...)` | identical (helper already accepts ndarrays) | ✅ | |
| `foil_threshold = ctx.foil_threshold if ctx.enable_foil_aware_fusion else None` | identical with `config[...]` lookups | ✅ | |
| On fusion exception: fallback to single-frame, log via telemetry | identical (V4 used `print`; V5.5 uses telemetry.resource_sample) | ✅ | |
| Output dict fields: `instance_id`, `session_id`, `angle`, `fused_image_path`, `primary_hash`, `quality_score`, `side_score`, `appearance_vector`, `best_canonical_detection_id`, `duplicate_track_index`, `first_frame_index`, `reid_embedding` | identical fields; `fused_image_path` → `fused_image` (ndarray) until store writes it | ⚠️ deviation | See Deviations §2 |
| `cv2.imwrite(fused_path, fused_img)` | deferred to store stage | ⚠️ deviation | See Deviations §3 |

### Deviations

1. **Canonical-frame inputs are ndarrays, not file paths.** **Reason:** V5.5 in-memory mandate. **Risk:** None.
2. **Field renamed from `fused_image_path` → `fused_image`.** **Reason:** No path exists yet; store writes the file. **Risk:** Downstream consumers must read `fused_image_path` only after `store` runs (the dedup stage uses `primary_hash` + `reid_embedding`, not the path, so it's safe).
3. **Image write deferred to store stage.** **Reason:** Single filesystem boundary per V5.5 mandate. **Risk:** None.

### Removed

- V4's `shutil.copy(best_path, fused_path)` for single-frame path. V5.5 just sets `fused["fused_image"] = best_canonical_image`. Observable behavior identical.

### Test coverage

- Unit: `tests/pipeline/stages/test_fuse_stage.py` (4 tests: 1-per-track, single-frame passthrough, foil enabled, foil disabled)
- E2E: `tests/pipeline/test_back_half_e2e.py`
- Golden-set: image_quality(SSIM) and (PSNR) deltas in Phase 14

---

## Stage: dedup

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/dedup.py` (127 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/dedup.py`
**Ported in commits:** P8.1

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| **Constant `SAME_CARD_EMB_THRESHOLD = 0.15`** | identical (defined at module scope) | ✅ | (constant explicit in audit per spec §13.1) |
| **Constant `SAME_CARD_HAMMING_MAX = 8`** | identical | ✅ | (constant explicit in audit per spec §13.1) |
| Outer `for i, f1 in enumerate(fused_canonicals):` with `processed: set` | identical | ✅ | |
| Group fields: `canonical_instance_id`, `duplicate_instance_ids`, `hamming_distances`, `embedding_distances`, `cross_video_parent_id` | identical | ✅ | |
| Intra-run: try embedding first (1.0 - dot product), fall back to pHash | identical | ✅ | |
| Cross-video: `SELECT id, reid_embedding FROM card_instances WHERE reid_embedding IS NOT NULL AND is_duplicate_of IS NULL AND video_id != ?` | replaced by `CardsRepository.find_embeddings_excluding_video(video_id=current)` | ⚠️ deviation | See Deviations §1 |
| Cross-video: track best (min dist) and set `cross_video_parent_id` iff `min_dist < SAME_CARD_EMB_THRESHOLD` | identical | ✅ | |
| Print line on cross-video match | dropped (telemetry already captures stage) | ✅ | |
| **Cross-video query excludes current video_id** | identical (test `test_dedup_cross_video_query_excludes_self_video_id` proves it) | ✅ | (critical per spec §13.1) |

### Deviations

1. **Cross-video query uses CardsRepository, not raw SQL.** **Reason:** V5.5 `no-sqlite3-outside-data` import-linter contract. **Risk:** None — same WHERE clause.

### Removed

- V4 returned `DedupOutput(dedup_groups=..., dedup_distances={})`. The `dedup_distances` field was unused (always `{}`). V5.5 stage doesn't emit it.

### Test coverage

- Unit: `tests/pipeline/stages/test_dedup_stage.py` (4 tests: intra-run by embedding, intra-run by pHash, cross-video query excludes self, cross-video match sets parent_id)
- E2E: `tests/pipeline/test_back_half_e2e.py`

---

## Stage: store

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/store.py` (155 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/store.py`
**Ported in commits:** P9.1

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Build `id_map`, `fused_map`, `track_map` dicts | identical | ✅ | |
| For each fused: try `f["reid_embedding"]`, fall back to `compute_reid_embedding(fused_image_path)` (V4) → `compute_reid_embedding_array(fused_image)` (V5.5) | identical semantic; in-memory input | ⚠️ deviation | See Deviations §1 |
| On embedding failure: `storage.add_pipeline_event("reid_embedding_failed", {...})` | `cards_repo.add_pipeline_event(...)` | ⚠️ deviation | See Deviations §2 |
| `Storage.add_card_instance(video_id, track_id, angle, session_id, reid_embedding, run_id)` returns row_id | `CardsRepository.add_card_instance(**kwargs)` returns row_id | ⚠️ deviation | See Deviations §2 |
| `Storage.update_instance_deduplication(row_id, primary_hash, None, reid_embedding=embedding_bytes)` | `CardsRepository.update_instance_deduplication(**kwargs)` | ⚠️ deviation | See Deviations §2 |
| `Storage.update_instance_fusion(row_id, fused_image_path)` | `CardsRepository.update_instance_fusion(**kwargs)` | ⚠️ deviation | See Deviations §2 |
| For each frame_entry in track: `is_best = (det_id == best_det_id)` | identical | ✅ | |
| **V4 line 98 (A1): if `is_best`, `view_path = f["fused_image_path"]`, else `view_path = fe["image_path"]`** | identical (critical behavior; test `test_store_best_view_points_to_fused_path` pins it) | ✅ | (explicit confirmation per spec §13.1) |
| `Storage.add_card_view(...)` returns view_id | `CardsRepository.add_card_view(**kwargs)` returns view_id | ⚠️ deviation | See Deviations §2 |
| **`add_saved_card` only when `fe["is_canonical"] AND is_best`** | identical (both conditions present) | ✅ | (explicit confirmation per spec §13.1) |
| For dedup_groups with `cross_video_parent_id`: `Storage.update_instance_deduplication(canonical_row_id, primary_hash, cross_video_parent)` | identical via repository | ⚠️ deviation | See Deviations §2 |
| For each `duplicate_instance_id` in group: `Storage.update_instance_deduplication(dup_row_id, primary_hash, canonical_row_id)` | identical via repository | ⚠️ deviation | See Deviations §2 |
| `Storage.update_video_status(video_id, "completed")` as the last DB write | replaced by `runs_repo.mark_completed(run_id, cards_extracted=len(final_cards))` | ⚠️ deviation | See Deviations §3 |
| Image writes: `crops/instance_<iid>_fused.jpg` + `crops/track_<iid>_det_<id>_rectified.jpg` | identical filenames | ✅ | (test `test_store_writes_*` pins both) |

### Deviations

1. **`compute_reid_embedding` fallback consumes ndarray.** **Reason:** in-memory mandate. **Risk:** None — P3 parity test.
2. **All `Storage.*` calls replaced by `CardsRepository.*`.** **Reason:** `no-sqlite3-outside-data` import-linter contract. **Risk:** None — repository methods are thin wrappers (Phase 2).
3. **Video status transition moved.** V4 stage called `storage.update_video_status(video_id, "completed")` at the end. V5.5 stage calls `runs_repo.mark_completed(run_id, cards_extracted=N)`. The video's status is updated by the surrounding `PipelineRunner._set_video_status` after `runtime.run()` returns successfully (already wired in PR #60). **Risk:** None — same end state in DB; cleaner separation of concerns (runs vs videos).

### Removed

- V4 used `CornerDetection(corners=..., confidence=...)` as the `detection` arg to `add_card_view`. V5.5 passes corners + confidence as separate kwargs (matching the repository signature in Phase 2).

### Test coverage

- Unit: `tests/pipeline/stages/test_store_stage.py` (6 tests: fused JPEG, rectified JPEG, add_card_instance with run_id, best-view-points-to-fused (A1), final_cards populated, mark_completed with count)
- E2E: `tests/pipeline/test_back_half_e2e.py` (DB rows + crops directory + run status)
- Golden-set: `cards_extracted` count compared to V4 baseline in Phase 14

---

## Cross-stage audit

| V4 invariant | V5.5 status | Note |
|---|---|---|
| Source video opened ONCE | ✅ identical (sample stage) |  |
| YOLO model loaded ONCE | ✅ identical (detect stage caches) |  |
| All in-memory pixel data is BGR uint8 | ✅ unchanged |  |
| All warped crops are 1050x750 uint8 BGR | ✅ unchanged |  |
| Storage writes go through DAL | ✅ stages 9-10 use CardsRepository |  |
| Metaflow imports outside vendored env: 0 | ✅ unchanged |  |
| Raw sqlite3 outside data/: 0 | ✅ unchanged |  |
```

- [ ] **Step 2: Fill in each stage's audit table by reading the V4 source and the new stage side-by-side**

For each stage:
1. Open the V4 file in `.worktrees/ci-fixes/pipeline/steps/<stage>.py`
2. Open the V5.5 file in `src/card_capture/pipeline/stages/<stage>.py`
3. For every behavior in the V4 file, add a row to the table marking ✅ / ⚠️ / ❌
4. For every ⚠️/❌ row, write a Deviations bullet with reason + risk

- [ ] **Step 3: Commit**

```bash
mkdir -p docs/superpowers/audits
git add docs/superpowers/audits/2026-05-29-v55-back-half-audit.md
git commit -m "docs(v55-stages): per-stage V4-vs-V5.5 audit

Each back-half stage row in the table marks every V4 card-producing
behavior as identical / deviation / removed. Deviations carry reasons
and risk notes. The audit gates merge per §15 sign-off."
```

### Phase 13 acceptance

Every card-producing behavior in §13.1 of the spec is explicitly marked in the audit. No `TODO` markers remain.

---

## Phase 14 — Manual golden-set re-run + CLAUDE.md update

This phase requires real hardware (a machine with `IMG_5872.MOV` + YOLO weights). Cannot run in CI.

### Task 14.1: Run the V4 baseline against `IMG_5872.MOV`

**Files:**
- Create: `docs/superpowers/plans/v5-5/back-half-baseline.md`

- [ ] **Step 1: Process the golden video**

```bash
.venv/bin/python -m card_capture.cli process tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV \
    --output-dir card_capture_output/back-half-baseline \
    --db card_capture_output/back-half-baseline/cards.sqlite \
    --detector docaligner
```

Expected: process completes successfully; `cards.sqlite` contains `card_instances` rows.

- [ ] **Step 2: Run the harness**

```bash
.venv/bin/python -m card_capture.cli harness run \
    --baseline v1 \
    --db card_capture_output/back-half-baseline/cards.sqlite \
    --truth-dir tests/fixtures/golden_corpus/IMG_5872/
```

Note all 5 metrics (`card_recall`, `card_precision`, `side_accuracy`, `image_quality (SSIM)`, `image_quality (PSNR)`).

- [ ] **Step 3: Compare against the V4 baseline**

Open `docs/superpowers/plans/v5-5/baseline-results.md`. Compute deltas:

| Metric | V4 baseline | V5.5 ported | Δ | Within gate? |
|---|---|---|---|---|
| card_recall | 0.1667 | <filled> | <±> | ±0.05 |
| card_precision | 1.0000 | <filled> | <±> | ±0.05 |
| side_accuracy | 1.0000 | <filled> | <±> | ±0.05 |
| SSIM | 0.4964 | <filled> | <±> | ±0.05 |
| PSNR | 8.0904 | <filled> | <±> | ±0.5 |

(Note: the V4 numbers shown were from a structural `fake` detector run per the existing baseline doc. Use the real-detector run when comparing — re-establish baseline if needed.)

- [ ] **Step 4: Write the doc**

Create `docs/superpowers/plans/v5-5/back-half-baseline.md`:

```markdown
# V5.5 Back-Half Baseline

Date: 2026-05-29
Tag (proposed): `v55-back-half-complete`
Video: `tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV`
Detector: `docaligner`
Git SHA: <fill in HEAD>

## Metrics

| Metric | V4 baseline | V5.5 back-half | Δ | Gate (±) | Pass? |
|---|---|---|---|---|---|
| card_recall      | <v4> | <v55> | <Δ> | 0.05 | ✅ / ❌ |
| card_precision   | <v4> | <v55> | <Δ> | 0.05 | ✅ / ❌ |
| side_accuracy    | <v4> | <v55> | <Δ> | 0.05 | ✅ / ❌ |
| SSIM             | <v4> | <v55> | <Δ> | 0.05 | ✅ / ❌ |
| PSNR             | <v4> | <v55> | <Δ> | 0.5  | ✅ / ❌ |

## Notes

- Run on <machine model>.
- Wallclock total: <seconds>.
- crops/ count: <n>.
- Notable behaviors observed (e.g., specific cards missed / front-back swaps): <fill in>.

## Gate verdict

<PASS / FAIL>. <If FAIL: link to follow-up issue.>
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/v5-5/back-half-baseline.md
git commit -m "docs(v55-stages): record back-half manual golden-set baseline

Real-video metrics from IMG_5872.MOV with docaligner detector.
Δ against V4 baseline per metric with the pass/fail gate marked.
This is the merge-gating evidence per §15 of the plan."
```

### Task 14.2: Update `CLAUDE.md` Known Weaknesses

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Edit Known Weaknesses**

In `CLAUDE.md`, find the `## Known Weaknesses (v5.5)` section. Remove the bullet about "F/B classifier fallback uses longest-track heuristic" if it now works (it does via Phase 6) — or rephrase to reflect actual state. Add a positive note about the wired stages.

Replace:

```markdown
## Known Weaknesses (v5.5)

- In-process telemetry coverage in `UnifiedRuntime` is incomplete (TODOs in code).
- F/B classifier fallback uses longest-track heuristic (classifier needs training update).
- Eager warping in worker thread uses CPU fallback if Kornia is not strictly enforced.
```

with:

```markdown
## Known Weaknesses (v5.5)

- F/B classifier fallback uses longest-track heuristic when the classifier is unavailable (resolve stage gracefully degrades; classifier weights are an optional artifact).
- Eager warping uses CPU fallback (`PrecisionNormalizer`) if Kornia construction fails for the requested device.
- In-memory peak (~180 MB for the reference video) scales with concurrent active tracks; mitigation (selective spill between refine/score) is a tracked follow-up.

## Recent baseline

V5.5 back-half wired and verified against IMG_5872.MOV — see
[docs/superpowers/plans/v5-5/back-half-baseline.md](docs/superpowers/plans/v5-5/back-half-baseline.md).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md Known Weaknesses post back-half wiring"
```

### Phase 14 acceptance

```
[ ] Real-video run completed without errors
[ ] All 5 gates ✅ in back-half-baseline.md, OR any ❌ has a logged follow-up issue
[ ] CLAUDE.md reflects current state
[ ] All sign-off boxes in §15 of the spec checked
```

---

## Plan-level acceptance

After all 14 phases:

Run: `.venv/bin/python -m pytest tests/ -m "not quarantine" -q`
Expected: 591+ existing tests + ~80 new tests added across phases 1–14 pass. The 9 pre-existing env failures (objc dyld + pytest-asyncio) remain unchanged.

Run: `.venv/bin/python -m pytest tests/architecture/ -v`
Expected: 5/5 contracts kept (raw sqlite3 outside data: 0; Metaflow outside vendored: 0; vast.ai: 0; layered: pass; strict-gpu: pass).

Run: `PYTHONPATH=src:. lint-imports`
Expected: 5 kept, 0 broken.

If all above + the §14 manual gates pass, open the PR and request review.

