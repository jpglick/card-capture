# Training Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified `/training` page where uploading a video automatically populates labeling queues for all three models (presence, front/back, YOLO corners), keyboard-driven one-at-a-time labeling flows handle all tasks, a single Retrain button trains all models, and a benchmark button re-runs the last N videos to show before/after improvement.

**Architecture:** Backend adds a sampling service (triggered after each pipeline run), two new DB tables, and new API endpoints extending the existing TrainingService. Frontend replaces the stub `/training` page with a four-panel hub and three labeling sub-routes, reusing the existing Hotkeys component pattern.

**Tech Stack:** Python/FastAPI, SQLite, MobileNetV3-Small (torchvision), Svelte 5, existing `Hotkeys` + `VerdictButtons` component patterns.

---

## File Map

**Create:**
- `migrations/0005_training_samples.sql` — presence_samples, corner_samples, benchmark_snapshots tables
- `app/services/presence_sampler.py` — re-scan video post-run, save balanced 192px frames
- `tests/app/test_presence_sampler.py`
- `tests/app/test_training_api.py`
- `src/card_capture/training/presence_trainer.py` — real MobileNetV3 training for presence model
- `src/card_capture/training/fb_trainer.py` — real MobileNetV3 training for FB model
- `app/web/src/routes/training/presence/+page.svelte` — Y/N labeling flow
- `app/web/src/routes/training/fb/+page.svelte` — F/B/U/X labeling flow (replaces `/label/fb`)
- `app/web/src/routes/training/corners/+page.svelte` — overlay + drag-to-adjust flow
- `app/web/src/lib/stores/training.svelte.ts` — shared reactive store for hub stats

**Modify:**
- `migrations/run_migrations.py` — assert migration count (auto-picks up new .sql file)
- `pipeline/steps/detect.py` — write borderline corner detections to corner_samples after detection
- `app/services/pipeline_runner.py` — call presence_sampler after run completes
- `app/services/training_service.py` — add presence/corner queue methods, stats, benchmark, real trainers
- `app/api/training.py` — add presence, corners, stats, benchmark endpoints
- `app/web/src/routes/training/+page.svelte` — replace stub with four-panel hub
- `app/web/src/routes/label/fb/+page.svelte` — redirect to `/training/fb`

---

## Task 1: DB Migration

**Files:**
- Create: `migrations/0005_training_samples.sql`

- [ ] **Write the migration file**

```sql
-- 0005_training_samples.sql

CREATE TABLE IF NOT EXISTS presence_samples (
    id           INTEGER PRIMARY KEY,
    run_id       TEXT    NOT NULL,
    video_id     INTEGER NOT NULL,
    frame_index  INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    image_path   TEXT    NOT NULL,
    label        TEXT    CHECK (label IN ('present', 'absent')),
    labeled_at   TEXT,
    created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_presence_unlabeled
    ON presence_samples (label) WHERE label IS NULL;

CREATE TABLE IF NOT EXISTS corner_samples (
    id                INTEGER PRIMARY KEY,
    run_id            TEXT    NOT NULL,
    video_id          INTEGER NOT NULL,
    frame_index       INTEGER NOT NULL,
    image_path        TEXT    NOT NULL,
    predicted_corners TEXT    NOT NULL,  -- JSON: [[x,y],[x,y],[x,y],[x,y]]
    confidence        REAL    NOT NULL,
    label             TEXT    CHECK (label IN ('correct', 'adjusted', 'negative')),
    corrected_corners TEXT,              -- JSON: [[x,y],...] when label='adjusted'
    labeled_at        TEXT,
    created_at        TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_corner_unlabeled
    ON corner_samples (label) WHERE label IS NULL;

CREATE TABLE IF NOT EXISTS benchmark_snapshots (
    id              INTEGER PRIMARY KEY,
    job_id          TEXT    NOT NULL,   -- TrainingJob.job_id at retrain time
    run_id          TEXT    NOT NULL,   -- pipeline_runs.run_id being snapshotted
    cards_extracted INTEGER NOT NULL,
    snapshotted_at  TEXT    DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **Apply migration and verify**

```bash
python3 -c "
from migrations.run_migrations import apply_migrations
from pathlib import Path
apply_migrations(Path('card_capture_output/cards.sqlite'))
"
```

Expected output includes: `Applied migration: 0005_training_samples.sql`

```bash
sqlite3 card_capture_output/cards.sqlite ".tables" | grep -E "presence_samples|corner_samples|benchmark_snapshots"
```

Expected: all three table names printed.

- [ ] **Commit**

```bash
git add migrations/0005_training_samples.sql
git commit -m "feat(training): add presence_samples, corner_samples, benchmark_snapshots tables"
```

---

## Task 2: Presence Sampling Utility

**Files:**
- Create: `app/services/presence_sampler.py`
- Create: `tests/app/test_presence_sampler.py`

- [ ] **Write the failing test**

```python
# tests/app/test_presence_sampler.py
import sqlite3, tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest
from app.services.presence_sampler import sample_presence_frames, SAMPLES_PER_RUN

def _make_fake_video(path: Path, n_frames: int = 120, fps: float = 30.0):
    """Write a minimal grayscale .avi the sampler can open."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    for i in range(n_frames):
        frame = np.full((480, 640, 3), i % 255, dtype=np.uint8)
        out.write(frame)
    out.release()

def _make_db(path: Path):
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS presence_samples (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                video_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                label TEXT,
                labeled_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

def test_sample_presence_frames_inserts_rows():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = tmp / "test.avi"
        db = tmp / "cards.sqlite"
        _make_fake_video(video)
        _make_db(db)

        n = sample_presence_frames(
            video_path=video,
            run_id="run_test",
            video_id=1,
            output_dir=tmp,
            db_path=db,
        )

        assert n == SAMPLES_PER_RUN
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute("SELECT * FROM presence_samples").fetchall()
        assert len(rows) == SAMPLES_PER_RUN

def test_sample_saves_192px_jpegs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = tmp / "test.avi"
        db = tmp / "cards.sqlite"
        _make_fake_video(video)
        _make_db(db)
        sample_presence_frames(video, "run_test", 1, tmp, db)
        jpegs = list((tmp / "presence_samples").glob("*.jpg"))
        assert len(jpegs) == SAMPLES_PER_RUN
        img = cv2.imread(str(jpegs[0]))
        assert img.shape[1] == 192
```

- [ ] **Run test to confirm it fails**

```bash
python3 -m pytest tests/app/test_presence_sampler.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError` for `presence_sampler`.

- [ ] **Implement the sampler**

```python
# app/services/presence_sampler.py
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

SAMPLES_PER_RUN = 20
_SCAN_FPS = 15.0
_SCAN_WIDTH = 192


def sample_presence_frames(
    video_path: Path,
    run_id: str,
    video_id: int,
    output_dir: Path,
    db_path: Path,
) -> int:
    """Re-scan video at 192px/15fps, save SAMPLES_PER_RUN balanced frames.

    Returns the number of rows inserted into presence_samples.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return 0

    target_present, target_absent = _balance_targets(db_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(source_fps / _SCAN_FPS)))

    scan_frames: list[tuple[int, int, np.ndarray]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            h, w = frame.shape[:2]
            scaled_h = max(1, int(round(h * _SCAN_WIDTH / w)))
            small = cv2.resize(frame, (_SCAN_WIDTH, scaled_h))
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            scan_frames.append((frame_idx, ts_ms, small))
        frame_idx += 1
    cap.release()

    if not scan_frames:
        return 0

    total_target = target_present + target_absent
    if len(scan_frames) <= total_target:
        selected = scan_frames
    else:
        step = len(scan_frames) / total_target
        selected = [scan_frames[int(i * step)] for i in range(total_target)]

    out_dir = output_dir / "presence_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    inserted = 0
    with sqlite3.connect(str(db_path)) as conn:
        for fi, ts_ms, small in selected:
            fname = f"{run_id}_{fi}.jpg"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), small)
            conn.execute(
                """INSERT INTO presence_samples
                   (run_id, video_id, frame_index, timestamp_ms, image_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, video_id, fi, ts_ms, str(fpath)),
            )
            inserted += 1
        conn.commit()
    return inserted


def _balance_targets(db_path: Path) -> tuple[int, int]:
    """Return (target_present, target_absent) for this run's sample."""
    half = SAMPLES_PER_RUN // 2
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT label, COUNT(*) FROM presence_samples "
                "WHERE label IS NOT NULL GROUP BY label"
            ).fetchall()
    except Exception:
        return half, half

    counts = dict(rows)
    present = counts.get("present", 0)
    absent = counts.get("absent", 0)

    if present > absent * 3:
        return SAMPLES_PER_RUN // 4, 3 * SAMPLES_PER_RUN // 4
    if absent > present * 3:
        return 3 * SAMPLES_PER_RUN // 4, SAMPLES_PER_RUN // 4
    return half, half
```

- [ ] **Run tests to confirm they pass**

```bash
python3 -m pytest tests/app/test_presence_sampler.py -v
```

Expected: both tests PASS.

- [ ] **Commit**

```bash
git add app/services/presence_sampler.py tests/app/test_presence_sampler.py
git commit -m "feat(training): presence sampling utility — 20 balanced 192px frames per run"
```

---

## Task 3: Corner Sampling from Detect Step

**Files:**
- Modify: `pipeline/steps/detect.py`

- [ ] **Add `_save_corner_samples` to detect.py**

Add this import at the top of `pipeline/steps/detect.py`:

```python
import json as _json
import sqlite3 as _sqlite3
```

Add this function at the bottom of `detect.py` (before the `_build_sampler_detector` block or after it):

```python
def _save_corner_samples(ctx: RunContext, detection_rows: list, output_dir: Path) -> None:
    """Persist borderline YOLO detections (0.50–0.70 conf) for corner labeling."""
    if not ctx.db_path:
        return
    borderline = [d for d in detection_rows if 0.50 <= d["confidence"] <= 0.70]
    if not borderline:
        return
    try:
        with _sqlite3.connect(ctx.db_path) as conn:
            for d in borderline:
                conn.execute(
                    """INSERT OR IGNORE INTO corner_samples
                       (run_id, video_id, frame_index, image_path,
                        predicted_corners, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        ctx.ui_run_id or "",
                        ctx.video_id or 0,
                        d["frame_index"],
                        d.get("source_frame_path", ""),
                        _json.dumps(d["corners"]),
                        d["confidence"],
                    ),
                )
            conn.commit()
    except Exception as exc:
        print(f"[detect] corner sampling failed: {exc}", flush=True)
```

- [ ] **Call it at the end of `run()`**

In `detect.py`'s `run()` function, after `detection_rows` is built and before the `return DetectOutput(...)` line, add:

```python
    _save_corner_samples(ctx, detection_rows, output_dir)
```

- [ ] **Verify no import errors**

```bash
python3 -c "from pipeline.steps.detect import run; print('ok')"
```

Expected: `ok`

- [ ] **Commit**

```bash
git add pipeline/steps/detect.py
git commit -m "feat(training): write borderline YOLO detections to corner_samples"
```

---

## Task 4: Trigger Presence Sampling After Pipeline Run

**Files:**
- Modify: `app/services/pipeline_runner.py`

- [ ] **Add the sampling call after run completes**

In `pipeline_runner.py`, locate the line `self.bus.emit(run_id, Event(name="run_completed"))`. After `self._record_run_finish(run_id, "completed")`, add:

```python
            self._sample_presence_frames(run_id, video_id, video)
```

Then add the method to `PipelineRunner`:

```python
    def _sample_presence_frames(self, run_id: str, video_id: int, video_path: str) -> None:
        if not self.db_path:
            return
        try:
            from app.services.presence_sampler import sample_presence_frames
            from pathlib import Path as _Path
            base_output = _Path(self.db_path).parent
            n = sample_presence_frames(
                video_path=_Path(video_path),
                run_id=run_id,
                video_id=video_id,
                output_dir=base_output,
                db_path=_Path(self.db_path),
            )
            print(f"[{run_id}] queued {n} presence frames for labeling", flush=True)
        except Exception as exc:
            print(f"[{run_id}] presence sampling skipped: {exc}", flush=True)
```

- [ ] **Verify pipeline_runner imports cleanly**

```bash
python3 -c "from app.services.pipeline_runner import PipelineRunner; print('ok')"
```

Expected: `ok`

- [ ] **Commit**

```bash
git add app/services/pipeline_runner.py
git commit -m "feat(training): trigger presence frame sampling after each pipeline run"
```

---

## Task 5: Training Service — Queue Methods, Stats, and Benchmark

**Files:**
- Modify: `app/services/training_service.py`

- [ ] **Add methods for presence queue, corner queue, stats, and benchmark**

Add the following methods to `TrainingService`. Insert them after `list_datasets()`:

```python
    # ------------------------------------------------------------------
    # Presence queue
    # ------------------------------------------------------------------

    def next_presence_sample(self) -> Optional[dict]:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, image_path, frame_index FROM presence_samples "
                "WHERE label IS NULL ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            pending = conn.execute(
                "SELECT COUNT(*) FROM presence_samples WHERE label IS NULL"
            ).fetchone()[0]
        return {
            "sample_id": row["id"],
            "image_url": self._to_url(row["image_path"]),
            "frame_index": row["frame_index"],
            "pending_count": pending,
        }

    def label_presence(self, sample_id: int, label: str) -> None:
        assert label in ("present", "absent"), f"invalid label: {label!r}"
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE presence_samples SET label=?, labeled_at=datetime('now') WHERE id=?",
                (label, sample_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Corner queue
    # ------------------------------------------------------------------

    def next_corner_sample(self) -> Optional[dict]:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, image_path, frame_index, predicted_corners, confidence "
                "FROM corner_samples WHERE label IS NULL ORDER BY confidence LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            pending = conn.execute(
                "SELECT COUNT(*) FROM corner_samples WHERE label IS NULL"
            ).fetchone()[0]
        return {
            "sample_id": row["id"],
            "image_url": self._to_url(row["image_path"]),
            "frame_index": row["frame_index"],
            "predicted_corners": row["predicted_corners"],
            "confidence": row["confidence"],
            "pending_count": pending,
        }

    def label_corner(
        self,
        sample_id: int,
        label: str,
        corrected_corners: Optional[str] = None,
    ) -> None:
        assert label in ("correct", "adjusted", "negative"), f"invalid label: {label!r}"
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """UPDATE corner_samples
                   SET label=?, corrected_corners=?, labeled_at=datetime('now')
                   WHERE id=?""",
                (label, corrected_corners, sample_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            presence_pending = conn.execute(
                "SELECT COUNT(*) FROM presence_samples WHERE label IS NULL"
            ).fetchone()[0]
            fb_pending = conn.execute(
                """SELECT COUNT(*) FROM card_instances ci
                   WHERE NOT EXISTS (
                       SELECT 1 FROM fb_labels fl WHERE fl.instance_id = ci.track_id
                   )"""
            ).fetchone()[0]
            corner_pending = conn.execute(
                "SELECT COUNT(*) FROM corner_samples WHERE label IS NULL"
            ).fetchone()[0]

            # Accuracy from model_versions (most recent per model)
            accuracies = {}
            for model in ("presence", "fb_classifier"):
                row = conn.execute(
                    "SELECT eval_metrics_json FROM model_versions "
                    "WHERE model_name=? ORDER BY created_at DESC LIMIT 1",
                    (model,),
                ).fetchone()
                if row and row["eval_metrics_json"]:
                    import json
                    m = json.loads(row["eval_metrics_json"])
                    accuracies[model] = m.get("accuracy")

            # Accuracy history for chart
            history_rows = conn.execute(
                "SELECT model_name, eval_metrics_json, created_at FROM model_versions "
                "ORDER BY created_at ASC"
            ).fetchall()
            history = []
            for r in history_rows:
                if r["eval_metrics_json"]:
                    import json
                    m = json.loads(r["eval_metrics_json"])
                    history.append({
                        "model": r["model_name"],
                        "accuracy": m.get("accuracy"),
                        "created_at": r["created_at"],
                    })

        return {
            "pending": {
                "presence": presence_pending,
                "fb": fb_pending,
                "corners": corner_pending,
            },
            "accuracy": accuracies,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def snapshot_baseline(self, job_id: str, n: int = 3) -> None:
        """Snapshot pipeline output for last N runs before retraining."""
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            runs = conn.execute(
                "SELECT run_id, cards_extracted FROM pipeline_runs "
                "ORDER BY started_at DESC LIMIT ?",
                (n,),
            ).fetchall()
            for run_id, cards in runs:
                conn.execute(
                    "INSERT INTO benchmark_snapshots (job_id, run_id, cards_extracted) "
                    "VALUES (?, ?, ?)",
                    (job_id, run_id, cards),
                )
            conn.commit()

    def get_benchmark_baseline(self, job_id: str) -> list[dict]:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, cards_extracted FROM benchmark_snapshots WHERE job_id=?",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_url(self, abs_path: str) -> str:
        from pathlib import Path
        p = Path(abs_path)
        output_dir = Path(self.db_path).parent
        try:
            rel = p.relative_to(output_dir)
            return "/files/" + str(rel)
        except ValueError:
            return "/files/" + p.name
```

- [ ] **Verify the service imports cleanly**

```bash
python3 -c "from app.services.training_service import TrainingService; print('ok')"
```

Expected: `ok`

- [ ] **Commit**

```bash
git add app/services/training_service.py
git commit -m "feat(training): add presence/corner queue methods, stats, and benchmark snapshot to TrainingService"
```

---

## Task 6: API Routes

**Files:**
- Modify: `app/api/training.py`

- [ ] **Write failing API tests**

```python
# tests/app/test_training_api.py
import sqlite3, tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

def _bootstrap_app(tmp: Path):
    """Create a minimal app with the training API and an empty DB."""
    from app.main import create_app
    db = tmp / "cards.sqlite"
    from migrations.run_migrations import apply_migrations
    apply_migrations(db)
    app = create_app(db_path=db)
    return TestClient(app), db

def test_presence_next_empty_returns_204():
    with tempfile.TemporaryDirectory() as tmp:
        client, _ = _bootstrap_app(Path(tmp))
        r = client.get("/api/v1/training/presence/next")
        assert r.status_code == 204

def test_presence_label_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        client, db = _bootstrap_app(Path(tmp))
        # Insert a sample directly
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "INSERT INTO presence_samples (run_id, video_id, frame_index, timestamp_ms, image_path) "
                "VALUES ('r1', 1, 0, 0, '/tmp/x.jpg')"
            )
            conn.commit()
        r = client.get("/api/v1/training/presence/next")
        assert r.status_code == 200
        sample_id = r.json()["sample_id"]

        r2 = client.post("/api/v1/training/presence/label",
                         json={"sample_id": sample_id, "label": "present"})
        assert r2.status_code == 204

        # Queue should now be empty
        r3 = client.get("/api/v1/training/presence/next")
        assert r3.status_code == 204

def test_stats_returns_pending_counts():
    with tempfile.TemporaryDirectory() as tmp:
        client, db = _bootstrap_app(Path(tmp))
        r = client.get("/api/v1/training/stats")
        assert r.status_code == 200
        data = r.json()
        assert "pending" in data
        assert "presence" in data["pending"]
        assert "fb" in data["pending"]
        assert "corners" in data["pending"]
```

- [ ] **Run tests to confirm they fail**

```bash
python3 -m pytest tests/app/test_training_api.py -v 2>&1 | head -30
```

Expected: failures because new endpoints don't exist yet.

- [ ] **Add new endpoints to `app/api/training.py`**

Add these routes after the existing `get_job` route:

```python
from app.schemas.v1 import PresenceLabelRequest, CornerLabelRequest


@router.get("/presence/next")
def get_next_presence(request: Request):
    sample = _svc(request).next_presence_sample()
    if sample is None:
        from fastapi import Response
        return Response(status_code=204)
    return sample


@router.post("/presence/label", status_code=204)
def post_presence_label(body: PresenceLabelRequest, request: Request):
    _svc(request).label_presence(body.sample_id, body.label)


@router.get("/corners/next")
def get_next_corner(request: Request):
    sample = _svc(request).next_corner_sample()
    if sample is None:
        from fastapi import Response
        return Response(status_code=204)
    return sample


@router.post("/corners/label", status_code=204)
def post_corner_label(body: CornerLabelRequest, request: Request):
    _svc(request).label_corner(
        body.sample_id, body.label, body.corrected_corners
    )


@router.get("/stats")
def get_training_stats(request: Request):
    return _svc(request).get_stats()
```

- [ ] **Add the request schemas to `app/schemas/v1.py`**

Open `app/schemas/v1.py` and append:

```python
from typing import Optional

class PresenceLabelRequest(BaseModel):
    sample_id: int
    label: str  # 'present' | 'absent'

class CornerLabelRequest(BaseModel):
    sample_id: int
    label: str              # 'correct' | 'adjusted' | 'negative'
    corrected_corners: Optional[str] = None  # JSON string when label='adjusted'
```

- [ ] **Run tests to confirm they pass**

```bash
python3 -m pytest tests/app/test_training_api.py -v
```

Expected: all three tests PASS.

- [ ] **Commit**

```bash
git add app/api/training.py app/schemas/v1.py tests/app/test_training_api.py
git commit -m "feat(training): add presence, corners, and stats API endpoints"
```

---

## Task 7: Presence Trainer

**Files:**
- Create: `src/card_capture/training/presence_trainer.py`

- [ ] **Write the trainer**

```python
# src/card_capture/training/presence_trainer.py
"""Train the presence classifier (MobileNetV3-Small) from presence_samples."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

_LABEL_MAP = {"present": 1, "absent": 0}


class _PresenceDataset(Dataset):
    def __init__(self, rows: list[dict], tx):
        self.rows = rows
        self.tx = tx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(row["image_path"])
        if img is None:
            img = np.zeros((192, 192, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tx(rgb)
        y = torch.tensor(_LABEL_MAP[row["label"]], dtype=torch.long)
        return x, y


def train_presence(
    db_path: Path,
    output_path: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    progress_cb=None,
) -> dict:
    """Train and save the presence classifier. Returns eval metrics dict."""
    rows = _load_labeled_rows(db_path)
    if len(rows) < 10:
        raise ValueError(f"Need at least 10 labeled samples, got {len(rows)}")

    # Deterministic 80/20 split: id % 5 == 0 → validation
    val_rows = [r for r in rows if r["id"] % 5 == 0]
    train_rows = [r for r in rows if r["id"] % 5 != 0]

    tx_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tx_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = _PresenceDataset(train_rows, tx_train)
    val_ds = _PresenceDataset(val_rows, tx_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = _get_device()
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        acc = _evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if progress_cb:
            progress_cb({"epoch": epoch, "total_epochs": epochs, "val_accuracy": acc})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive = output_path.parent / "archive"
    archive.mkdir(exist_ok=True)
    if output_path.exists():
        import shutil, time
        shutil.copy(output_path, archive / f"presence_classifier_{int(time.time())}.pt")

    torch.save({"state_dict": best_state}, str(output_path))
    metrics = {"accuracy": round(best_acc, 4), "val_samples": len(val_rows)}
    return metrics


def _load_labeled_rows(db_path: Path) -> list[dict]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, image_path, label FROM presence_samples WHERE label IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

- [ ] **Wire it into TrainingService**

In `training_service.py`, find the stub in the `_run_job` method:

```python
            if job.model_name == "fb_classifier":
                # Real training would happen here
```

Replace the entire `_run_job` method body with:

```python
    def _run_job(self, job: TrainingJob) -> None:
        from datetime import datetime
        try:
            with self._lock:
                job.status = "running"

            def _progress(p: dict):
                with self._lock:
                    job.progress = p

            if job.model_name == "presence":
                from card_capture.training.presence_trainer import train_presence
                metrics = train_presence(
                    db_path=self.db_path,
                    output_path=Path("models/presence_classifier.pt"),
                    epochs=job.epochs,
                    progress_cb=_progress,
                )
            elif job.model_name == "fb_classifier":
                from card_capture.training.fb_trainer import train_fb
                metrics = train_fb(
                    db_path=self.db_path,
                    output_path=Path("models/fb_classifier.pt"),
                    epochs=job.epochs,
                    progress_cb=_progress,
                )
            else:
                raise ValueError(f"unknown model: {job.model_name!r}")

            self._record_model_version(job.model_name, metrics)

            with self._lock:
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
                job.progress = metrics

        except Exception as exc:
            logger.exception("Training job %s failed", job.job_id)
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = datetime.now().isoformat()

    def _record_model_version(self, model_name: str, metrics: dict) -> None:
        import sqlite3, json
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO model_versions (model_name, training_set_hash, eval_metrics_json, checkpoint_path) "
                "VALUES (?, ?, ?, ?)",
                (
                    model_name,
                    "",
                    json.dumps(metrics),
                    f"models/{model_name}.pt",
                ),
            )
            conn.commit()
```

Also add `epochs: int = 30` to `TrainingJob`:
```python
    epochs: int = 30
```

And update `start_retrain` to accept and pass epochs:
```python
    def start_retrain(self, model_name: str, epochs: int = 30, learning_rate: float = 1e-3) -> TrainingJob:
        ...
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            epochs=epochs,
            ...
        )
```

- [ ] **Verify the trainer imports cleanly**

```bash
python3 -c "from card_capture.training.presence_trainer import train_presence; print('ok')"
```

Expected: `ok`

- [ ] **Commit**

```bash
git add src/card_capture/training/presence_trainer.py app/services/training_service.py
git commit -m "feat(training): real MobileNetV3 presence trainer wired into TrainingService"
```

---

## Task 8: FB Trainer

**Files:**
- Create: `src/card_capture/training/fb_trainer.py`

- [ ] **Write the FB trainer**

```python
# src/card_capture/training/fb_trainer.py
"""Train the Front/Back classifier (MobileNetV3-Small) from fb_labels."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# 'uncertain' and 'no_card' are excluded — not useful for the FB task
_LABEL_MAP = {"front": 0, "back": 1}


class _FBDataset(Dataset):
    def __init__(self, rows: list[dict], tx):
        self.rows = rows
        self.tx = tx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(row["image_path"])
        if img is None:
            img = np.zeros((750, 1050, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tx(rgb)
        y = torch.tensor(_LABEL_MAP[row["label"]], dtype=torch.long)
        return x, y


def train_fb(
    db_path: Path,
    output_path: Path,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    progress_cb=None,
) -> dict:
    """Train and save the FB classifier. Returns eval metrics dict."""
    rows = _load_labeled_rows(db_path)
    if len(rows) < 10:
        raise ValueError(f"Need at least 10 labeled front/back samples, got {len(rows)}")

    val_rows = [r for r in rows if r["id"] % 5 == 0]
    train_rows = [r for r in rows if r["id"] % 5 != 0]

    tx_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tx_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = _FBDataset(train_rows, tx_train)
    val_ds = _FBDataset(val_rows, tx_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = _get_device()
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        acc = _evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if progress_cb:
            progress_cb({"epoch": epoch, "total_epochs": epochs, "val_accuracy": acc})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive = output_path.parent / "archive"
    archive.mkdir(exist_ok=True)
    if output_path.exists():
        import shutil, time
        shutil.copy(output_path, archive / f"fb_classifier_{int(time.time())}.pt")

    torch.save({"state_dict": best_state}, str(output_path))
    return {"accuracy": round(best_acc, 4), "val_samples": len(val_rows)}


def _load_labeled_rows(db_path: Path) -> list[dict]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        # Join fb_labels → card_views to get image paths
        # Only keep front/back — exclude uncertain and no_card
        rows = conn.execute(
            """SELECT fl.id, cv.rectified_path AS image_path, fl.side AS label
               FROM fb_labels fl
               JOIN card_instances ci ON ci.track_id = fl.instance_id
               JOIN card_views cv ON cv.instance_id = ci.id
                   AND cv.frame_index = fl.frame_index
               WHERE fl.side IN ('front', 'back')
               ORDER BY fl.id"""
        ).fetchall()
    return [dict(r) for r in rows]


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

- [ ] **Verify the trainer imports cleanly**

```bash
python3 -c "from card_capture.training.fb_trainer import train_fb; print('ok')"
```

Expected: `ok`

- [ ] **Commit**

```bash
git add src/card_capture/training/fb_trainer.py
git commit -m "feat(training): real MobileNetV3 FB trainer"
```

---

## Task 9: Training Hub Page

**Files:**
- Modify: `app/web/src/routes/training/+page.svelte`
- Create: `app/web/src/lib/stores/training.svelte.ts`

- [ ] **Create the shared stats store**

```typescript
// app/web/src/lib/stores/training.svelte.ts
import { api } from '$lib/api/client';

export interface TrainingStats {
  pending: { presence: number; fb: number; corners: number };
  accuracy: Record<string, number | null>;
  history: Array<{ model: string; accuracy: number; created_at: string }>;
}

export function createTrainingStore() {
  let stats = $state<TrainingStats | null>(null);
  let loading = $state(true);

  async function refresh() {
    loading = true;
    try {
      const r = await api.get('/training/stats');
      stats = await r.json();
    } finally {
      loading = false;
    }
  }

  return { get stats() { return stats; }, get loading() { return loading; }, refresh };
}
```

- [ ] **Replace the training hub page**

```svelte
<!-- app/web/src/routes/training/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { api } from '$lib/api/client';
  import { createTrainingStore } from '$lib/stores/training.svelte';

  const store = createTrainingStore();

  let retraining = $state(false);
  let benchmarking = $state(false);
  let benchmarkResult = $state<any>(null);
  let jobId = $state<string | null>(null);

  onMount(() => store.refresh());

  async function retrain(model: string) {
    retraining = true;
    const r = await api.post(`/training/retrain/${model}`, { epochs: 30, learning_rate: 0.001 });
    const job = await r.json();
    jobId = job.job_id;
    pollJob(job.job_id);
  }

  async function retrainAll() {
    retraining = true;
    for (const model of ['presence', 'fb_classifier']) {
      const r = await api.post(`/training/retrain/${model}`, { epochs: 30, learning_rate: 0.001 });
      const job = await r.json();
      await pollJob(job.job_id);
    }
    await store.refresh();
    retraining = false;
  }

  async function pollJob(id: string): Promise<void> {
    return new Promise((resolve) => {
      const interval = setInterval(async () => {
        const r = await api.get(`/training/jobs/${id}`);
        const job = await r.json();
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(interval);
          resolve();
        }
      }, 2000);
    });
  }

  function pct(v: number | null | undefined) {
    return v != null ? `${Math.round(v * 100)}%` : '—';
  }
</script>

<div class="hub">
  <h1>Training</h1>

  <div class="panels">
    <div class="panel" onclick={() => goto('/training/presence')}>
      <div class="panel-title">Presence</div>
      <div class="pending">{store.stats?.pending.presence ?? '…'} pending</div>
      <div class="acc">acc: {pct(store.stats?.accuracy['presence'])}</div>
      <button class="label-btn" onclick|stopPropagation={() => goto('/training/presence')}>
        Label now →
      </button>
    </div>

    <div class="panel" onclick={() => goto('/training/fb')}>
      <div class="panel-title">Front / Back</div>
      <div class="pending">{store.stats?.pending.fb ?? '…'} pending</div>
      <div class="acc">acc: {pct(store.stats?.accuracy['fb_classifier'])}</div>
      <button class="label-btn" onclick|stopPropagation={() => goto('/training/fb')}>
        Label now →
      </button>
    </div>

    <div class="panel" onclick={() => goto('/training/corners')}>
      <div class="panel-title">YOLO Corners</div>
      <div class="pending">{store.stats?.pending.corners ?? '…'} pending</div>
      <div class="acc">acc: —</div>
      <button class="label-btn" onclick|stopPropagation={() => goto('/training/corners')}>
        Label now →
      </button>
    </div>

    <div class="panel benchmark">
      <div class="panel-title">Benchmark</div>
      <button class="retrain-btn" disabled={retraining} onclick={retrainAll}>
        {retraining ? 'Training…' : 'Retrain all'}
      </button>
    </div>
  </div>
</div>

<style>
  .hub { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: 1.5rem; }
  .panels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
  .panel {
    background: #1e1e2e; border-radius: 12px; padding: 1.25rem;
    cursor: pointer; transition: background 0.15s;
    display: flex; flex-direction: column; gap: 0.5rem;
  }
  .panel:hover { background: #2a2a3e; }
  .panel-title { font-weight: 700; font-size: 1rem; }
  .pending { font-size: 1.4rem; font-weight: 700; }
  .acc { color: #aaa; font-size: 0.85rem; }
  .label-btn {
    margin-top: auto; background: #6366f1; color: white;
    border: none; border-radius: 8px; padding: 0.5rem 1rem;
    cursor: pointer; font-size: 0.85rem;
  }
  .retrain-btn {
    background: #0acf97; color: white; border: none;
    border-radius: 8px; padding: 0.6rem 1.2rem; cursor: pointer;
    font-weight: 600;
  }
  .retrain-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .benchmark { cursor: default; }
  .benchmark:hover { background: #1e1e2e; }
</style>
```

- [ ] **Start dev server and verify the hub loads**

```bash
cd app/web && npm run dev
```

Open `http://localhost:5173/training` — four panels should be visible with pending counts loading.

- [ ] **Commit**

```bash
git add app/web/src/routes/training/+page.svelte app/web/src/lib/stores/training.svelte.ts
git commit -m "feat(training): training hub page with four panels"
```

---

## Task 10: Presence Labeler

**Files:**
- Create: `app/web/src/routes/training/presence/+page.svelte`

- [ ] **Create the presence labeling page**

```svelte
<!-- app/web/src/routes/training/presence/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Hotkeys from '$lib/components/Hotkeys.svelte';
  import { api } from '$lib/api/client';

  let sample = $state<any>(null);
  let done = $state(false);
  let labeled = $state(0);

  async function fetchNext() {
    const r = await api.get('/training/presence/next');
    if (r.status === 204) { done = true; return; }
    sample = await r.json();
  }

  async function submit(label: 'present' | 'absent') {
    if (!sample) return;
    await api.post('/training/presence/label', { sample_id: sample.sample_id, label });
    labeled++;
    await fetchNext();
  }

  async function skip() { await fetchNext(); }

  onMount(fetchNext);

  const keys = {
    y: () => submit('present'),
    n: () => submit('absent'),
    s: skip,
  };
</script>

<Hotkeys {keys}>
  <div class="container">
    <header>
      <button onclick={() => goto('/training')} class="back">← Training</button>
      <span class="counter">
        {#if sample}
          {sample.pending_count} remaining · {labeled} labeled this session
        {/if}
      </span>
    </header>

    {#if done}
      <div class="empty">
        <p>Queue is empty.</p>
        <button onclick={() => goto('/training')}>Back to hub</button>
      </div>
    {:else if sample}
      <div class="label-area">
        <div class="frame-box">
          <img src={sample.image_url} alt="scan frame" class="scan-frame" />
        </div>

        <div class="question">Is there a card in this frame?</div>

        <div class="actions">
          <button class="yes" onclick={() => submit('present')}>
            <span class="label">Card present</span>
            <span class="key">Y</span>
          </button>
          <button class="no" onclick={() => submit('absent')}>
            <span class="label">No card</span>
            <span class="key">N</span>
          </button>
          <button class="skip" onclick={skip}>
            <span class="label">Skip</span>
            <span class="key">S</span>
          </button>
        </div>

        <div class="progress">
          <div class="bar" style="width: {Math.max(0, 100 - (sample.pending_count / (sample.pending_count + labeled)) * 100)}%"></div>
        </div>
      </div>
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
</Hotkeys>

<style>
  .container { max-width: 600px; margin: 2rem auto; padding: 0 1rem; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
  .back { background: none; border: none; color: #aaa; cursor: pointer; font-size: 0.9rem; }
  .counter { color: #aaa; font-size: 0.85rem; }
  .label-area { display: flex; flex-direction: column; gap: 1.5rem; align-items: center; }
  .frame-box { background: #111; border-radius: 8px; padding: 1rem; }
  .scan-frame { image-rendering: pixelated; width: 384px; height: auto; display: block; }
  .question { font-size: 1.1rem; font-weight: 600; }
  .actions { display: flex; gap: 1rem; }
  button { display: flex; flex-direction: column; align-items: center; padding: 0.75rem 1.5rem;
    border: none; border-radius: 8px; cursor: pointer; min-width: 100px; }
  .yes { background: #0acf97; color: white; }
  .no { background: #fa5c7c; color: white; }
  .skip { background: #444; color: white; }
  .label { font-weight: 600; }
  .key { font-size: 0.75rem; opacity: 0.7; }
  .progress { width: 100%; height: 4px; background: #333; border-radius: 2px; }
  .bar { height: 100%; background: #6366f1; border-radius: 2px; transition: width 0.3s; }
  .empty { text-align: center; margin-top: 4rem; }
</style>
```

- [ ] **Verify page loads at `/training/presence`**

Open `http://localhost:5173/training/presence` — should show the scan frame labeler or "Queue is empty."

- [ ] **Commit**

```bash
git add app/web/src/routes/training/presence/+page.svelte
git commit -m "feat(training): presence labeler page — Y/N/S keyboard flow"
```

---

## Task 11: Front/Back Labeler

**Files:**
- Create: `app/web/src/routes/training/fb/+page.svelte`
- Modify: `app/web/src/routes/label/fb/+page.svelte` (redirect)

- [ ] **Create the new FB labeler at `/training/fb`**

```svelte
<!-- app/web/src/routes/training/fb/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Hotkeys from '$lib/components/Hotkeys.svelte';
  import { api } from '$lib/api/client';

  let sample = $state<any>(null);
  let done = $state(false);
  let labeled = $state(0);

  async function fetchNext() {
    const r = await api.get('/label/fb/next');
    if (r.status === 204) { done = true; return; }
    sample = await r.json();
  }

  async function submit(side: string) {
    if (!sample) return;
    await api.post('/label/fb', { instance_id: sample.instance_id, frame_index: sample.frame_index, side });
    labeled++;
    await fetchNext();
  }

  async function skip() { await fetchNext(); }

  onMount(fetchNext);

  const keys = {
    f: () => submit('front'),
    b: () => submit('back'),
    u: () => submit('uncertain'),
    x: () => submit('no_card'),
    s: skip,
  };
</script>

<Hotkeys {keys}>
  <div class="container">
    <header>
      <button onclick={() => goto('/training')} class="back">← Training</button>
      <span class="counter">{labeled} labeled this session</span>
    </header>

    {#if done}
      <div class="empty">
        <p>Queue is empty.</p>
        <button onclick={() => goto('/training')}>Back to hub</button>
      </div>
    {:else if sample}
      <div class="label-area">
        <div class="card-box">
          <img src={sample.canonical_url} alt="card" class="card-img" />
        </div>

        <div class="question">Which side is this?</div>

        <div class="actions">
          <button class="front" onclick={() => submit('front')}>
            <span class="label">Front</span><span class="key">F</span>
          </button>
          <button class="back" onclick={() => submit('back')}>
            <span class="label">Back</span><span class="key">B</span>
          </button>
          <button class="uncertain" onclick={() => submit('uncertain')}>
            <span class="label">Unsure</span><span class="key">U</span>
          </button>
          <button class="nocard" onclick={() => submit('no_card')}>
            <span class="label">Not a card</span><span class="key">X</span>
          </button>
          <button class="skip" onclick={skip}>
            <span class="label">Skip</span><span class="key">S</span>
          </button>
        </div>
      </div>
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
</Hotkeys>

<style>
  .container { max-width: 500px; margin: 2rem auto; padding: 0 1rem; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
  .back { background: none; border: none; color: #aaa; cursor: pointer; }
  .counter { color: #aaa; font-size: 0.85rem; }
  .label-area { display: flex; flex-direction: column; gap: 1.5rem; align-items: center; }
  .card-box { background: #111; border-radius: 8px; padding: 0.75rem; }
  .card-img { width: 280px; height: auto; display: block; border-radius: 4px; }
  .question { font-size: 1.1rem; font-weight: 600; }
  .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; }
  button { display: flex; flex-direction: column; align-items: center; padding: 0.65rem 1.2rem;
    border: none; border-radius: 8px; cursor: pointer; }
  .front { background: #6366f1; color: white; }
  .back { background: #8b5cf6; color: white; }
  .uncertain { background: #f59e0b; color: white; }
  .nocard { background: #fa5c7c; color: white; }
  .skip { background: #444; color: white; }
  .label { font-weight: 600; font-size: 0.9rem; }
  .key { font-size: 0.7rem; opacity: 0.7; }
  .empty { text-align: center; margin-top: 4rem; }
</style>
```

- [ ] **Redirect old `/label/fb` to `/training/fb`**

Replace the contents of `app/web/src/routes/label/fb/+page.svelte` with:

```svelte
<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  onMount(() => goto('/training/fb', { replaceState: true }));
</script>
```

- [ ] **Verify both routes work**

Open `http://localhost:5173/training/fb` — should show the FB labeler with 5 buttons.
Open `http://localhost:5173/label/fb` — should immediately redirect to `/training/fb`.

- [ ] **Commit**

```bash
git add app/web/src/routes/training/fb/+page.svelte app/web/src/routes/label/fb/+page.svelte
git commit -m "feat(training): FB labeler at /training/fb with No Card option; redirect /label/fb"
```

---

## Task 12: Corner Labeler

**Files:**
- Create: `app/web/src/routes/training/corners/+page.svelte`

- [ ] **Create the corner labeling page**

```svelte
<!-- app/web/src/routes/training/corners/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Hotkeys from '$lib/components/Hotkeys.svelte';
  import { api } from '$lib/api/client';

  let sample = $state<any>(null);
  let done = $state(false);
  let labeled = $state(0);
  let adjusting = $state(false);
  let corners = $state<[number, number][]>([]);
  let dragging = $state<number | null>(null);
  let imgEl = $state<HTMLImageElement | null>(null);
  let imgNaturalW = $state(1);
  let imgNaturalH = $state(1);
  let imgDisplayW = $state(1);
  let imgDisplayH = $state(1);

  async function fetchNext() {
    adjusting = false;
    const r = await api.get('/training/corners/next');
    if (r.status === 204) { done = true; return; }
    sample = await r.json();
    corners = JSON.parse(sample.predicted_corners);
  }

  async function submit(label: string, corrected: [number,number][] | null = null) {
    if (!sample) return;
    const body: any = { sample_id: sample.sample_id, label };
    if (corrected) body.corrected_corners = JSON.stringify(corrected);
    await api.post('/training/corners/label', body);
    labeled++;
    await fetchNext();
  }

  function startAdjust() { adjusting = true; }

  function scaleToNatural(x: number, y: number): [number, number] {
    const sx = imgNaturalW / imgDisplayW;
    const sy = imgNaturalH / imgDisplayH;
    return [x * sx, y * sy];
  }

  function scaleToDisplay(x: number, y: number): [number, number] {
    return [x * (imgDisplayW / imgNaturalW), y * (imgDisplayH / imgNaturalH)];
  }

  function onMouseDown(i: number) { dragging = i; }

  function onMouseMove(e: MouseEvent) {
    if (dragging === null || !imgEl) return;
    const rect = imgEl.getBoundingClientRect();
    const dx = e.clientX - rect.left;
    const dy = e.clientY - rect.top;
    const [nx, ny] = scaleToNatural(dx, dy);
    corners = corners.map((c, i) => i === dragging ? [nx, ny] : c) as [number,number][];
  }

  function onMouseUp() { dragging = null; }

  function confirmAdjust() { submit('adjusted', corners); }

  function onImgLoad() {
    if (!imgEl) return;
    imgNaturalW = imgEl.naturalWidth;
    imgNaturalH = imgEl.naturalHeight;
    imgDisplayW = imgEl.clientWidth;
    imgDisplayH = imgEl.clientHeight;
  }

  onMount(fetchNext);

  const keys = {
    y: () => !adjusting && submit('correct'),
    n: () => !adjusting && submit('negative'),
    e: () => !adjusting && startAdjust(),
    ' ': () => adjusting && confirmAdjust(),
    Escape: () => { adjusting = false; corners = JSON.parse(sample.predicted_corners); },
    s: () => !adjusting && fetchNext(),
  };
</script>

<Hotkeys {keys}>
  <div class="container">
    <header>
      <button onclick={() => goto('/training')} class="back">← Training</button>
      <span class="counter">
        {#if sample}
          conf {(sample.confidence * 100).toFixed(0)}% · {sample.pending_count} remaining · {labeled} labeled
        {/if}
      </span>
    </header>

    {#if done}
      <div class="empty">
        <p>Queue is empty.</p>
        <button onclick={() => goto('/training')}>Back to hub</button>
      </div>
    {:else if sample}
      <div class="label-area">
        <div
          class="img-wrap"
          onmousemove={onMouseMove}
          onmouseup={onMouseUp}
          role="img"
          aria-label="frame with corner overlay"
        >
          <img
            bind:this={imgEl}
            src={sample.image_url}
            alt="detection frame"
            class="frame-img"
            onload={onImgLoad}
          />
          <svg class="overlay" width={imgDisplayW} height={imgDisplayH}>
            <polygon
              points={corners.map(([x, y]) => scaleToDisplay(x, y).join(',')).join(' ')}
              fill="rgba(0,255,120,0.15)"
              stroke="#00ff78"
              stroke-width="2"
            />
            {#if adjusting}
              {#each corners as [cx, cy], i}
                {@const [dx, dy] = scaleToDisplay(cx, cy)}
                <circle
                  cx={dx} cy={dy} r="8"
                  fill="#00ff78" stroke="white" stroke-width="2"
                  style="cursor: grab"
                  onmousedown={() => onMouseDown(i)}
                  role="button"
                  aria-label={`corner ${i}`}
                  tabindex={i}
                />
              {/each}
            {/if}
          </svg>
        </div>

        {#if adjusting}
          <div class="hint">Drag corners to adjust · <kbd>Space</kbd> confirm · <kbd>Esc</kbd> cancel</div>
          <div class="actions">
            <button class="confirm" onclick={confirmAdjust}>Confirm (Space)</button>
            <button class="skip" onclick={() => { adjusting = false; }}>Cancel (Esc)</button>
          </div>
        {:else}
          <div class="question">Do the highlighted corners look right?</div>
          <div class="actions">
            <button class="yes" onclick={() => submit('correct')}>
              <span class="label">Correct</span><span class="key">Y</span>
            </button>
            <button class="adjust" onclick={startAdjust}>
              <span class="label">Adjust</span><span class="key">E</span>
            </button>
            <button class="no" onclick={() => submit('negative')}>
              <span class="label">No card</span><span class="key">N</span>
            </button>
            <button class="skip" onclick={fetchNext}>
              <span class="label">Skip</span><span class="key">S</span>
            </button>
          </div>
        {/if}
      </div>
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
</Hotkeys>

<style>
  .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
  .back { background: none; border: none; color: #aaa; cursor: pointer; }
  .counter { color: #aaa; font-size: 0.85rem; }
  .label-area { display: flex; flex-direction: column; gap: 1.25rem; align-items: center; }
  .img-wrap { position: relative; display: inline-block; }
  .frame-img { max-width: 100%; max-height: 60vh; display: block; }
  .overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
  .overlay circle { pointer-events: all; }
  .question { font-size: 1.05rem; font-weight: 600; }
  .hint { color: #aaa; font-size: 0.85rem; }
  .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; }
  button { display: flex; flex-direction: column; align-items: center; padding: 0.65rem 1.2rem;
    border: none; border-radius: 8px; cursor: pointer; }
  .yes { background: #0acf97; color: white; }
  .adjust { background: #f59e0b; color: white; }
  .no { background: #fa5c7c; color: white; }
  .skip { background: #444; color: white; }
  .confirm { background: #6366f1; color: white; }
  .label { font-weight: 600; font-size: 0.9rem; }
  .key { font-size: 0.7rem; opacity: 0.7; }
  .empty { text-align: center; margin-top: 4rem; }
  kbd { background: #333; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.8rem; }
</style>
```

- [ ] **Verify page loads at `/training/corners`**

Open `http://localhost:5173/training/corners` — should show "Queue is empty" (no borderline detections yet) or a frame with green polygon overlay.

- [ ] **Commit**

```bash
git add app/web/src/routes/training/corners/+page.svelte
git commit -m "feat(training): YOLO corner labeler with SVG overlay and drag-to-adjust"
```

---

## Task 13: Benchmark Panel and Accuracy Chart

**Files:**
- Modify: `app/web/src/routes/training/+page.svelte` (add benchmark section below panels)

- [ ] **Add benchmark and chart section to the hub page**

In `app/web/src/routes/training/+page.svelte`, extend the script block and template. Add to `<script>`:

```typescript
  interface BenchmarkRow { video: string; before: number; after: number; delta: number; }
  let benchmarkRows = $state<BenchmarkRow[]>([]);
  let benchmarking = $state(false);
  let lastJobId = $state<string | null>(null);

  async function runBenchmark() {
    benchmarking = true;
    benchmarkRows = [];
    // snapshot happens server-side before retrain; here we just re-run last N videos
    // For now, call the benchmark endpoint which re-runs and compares
    const r = await api.post('/training/benchmark', { n: 3 });
    const job = await r.json();
    lastJobId = job.job_id;
    await pollBenchmarkJob(job.job_id);
    benchmarking = false;
  }

  async function pollBenchmarkJob(id: string) {
    return new Promise<void>((resolve) => {
      const interval = setInterval(async () => {
        const r = await api.get(`/training/benchmark/${id}`);
        const j = await r.json();
        if (j.status === 'completed') {
          benchmarkRows = j.rows ?? [];
          clearInterval(interval);
          resolve();
        } else if (j.status === 'failed') {
          clearInterval(interval);
          resolve();
        }
      }, 3000);
    });
  }
```

Add below the `.panels` div in the template:

```svelte
  <!-- Benchmark section -->
  <div class="benchmark-section">
    <div class="bm-header">
      <h2>Benchmark</h2>
      <button class="bm-btn" disabled={benchmarking} onclick={runBenchmark}>
        {benchmarking ? 'Running…' : 'Run pipeline on last 3 videos'}
      </button>
    </div>

    {#if benchmarkRows.length > 0}
      <table class="bm-table">
        <thead><tr><th>Video</th><th>Before</th><th>After</th><th>Δ</th></tr></thead>
        <tbody>
          {#each benchmarkRows as row}
            <tr>
              <td>{row.video}</td>
              <td>{row.before} cards</td>
              <td>{row.after} cards</td>
              <td class:positive={row.delta > 0} class:neutral={row.delta === 0}>
                {row.delta > 0 ? '+' : ''}{row.delta} {row.delta > 0 ? '✓' : '→'}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  <!-- Accuracy history chart -->
  {#if store.stats?.history?.length}
    <div class="chart-section">
      <h2>Accuracy over time</h2>
      <div class="chart">
        {#each ['presence', 'fb_classifier'] as model}
          {@const points = store.stats.history.filter(h => h.model === model && h.accuracy != null)}
          {#if points.length > 1}
            <div class="series">
              <span class="series-label">{model === 'presence' ? 'Presence' : 'Front/Back'}</span>
              <svg viewBox="0 0 200 60" class="sparkline">
                <polyline
                  points={points.map((p, i) =>
                    `${(i / (points.length - 1)) * 190 + 5},${55 - (p.accuracy ?? 0) * 50}`
                  ).join(' ')}
                  fill="none"
                  stroke={model === 'presence' ? '#6366f1' : '#0acf97'}
                  stroke-width="2"
                />
              </svg>
              <span class="series-pct">{pct(points[points.length - 1]?.accuracy)}</span>
            </div>
          {/if}
        {/each}
      </div>
    </div>
  {/if}
```

Add to `<style>`:

```css
  .benchmark-section { margin-top: 2rem; }
  .bm-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
  .bm-header h2 { margin: 0; }
  .bm-btn {
    background: #6366f1; color: white; border: none;
    border-radius: 8px; padding: 0.5rem 1.2rem; cursor: pointer;
  }
  .bm-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .bm-table { width: 100%; border-collapse: collapse; }
  .bm-table th, .bm-table td { padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #333; }
  .positive { color: #0acf97; font-weight: 600; }
  .neutral { color: #aaa; }
  .chart-section { margin-top: 2rem; }
  .chart { display: flex; flex-direction: column; gap: 0.75rem; }
  .series { display: flex; align-items: center; gap: 1rem; }
  .series-label { width: 90px; font-size: 0.85rem; color: #aaa; }
  .sparkline { width: 200px; height: 60px; }
  .series-pct { font-weight: 700; width: 40px; }
```

- [ ] **Add the benchmark API endpoint to `app/api/training.py`**

```python
@router.post("/benchmark", status_code=202)
def start_benchmark(body: dict, request: Request):
    n = int(body.get("n", 3))
    job = _svc(request).start_benchmark(n)
    return {"job_id": job.job_id}

@router.get("/benchmark/{job_id}")
def get_benchmark(job_id: str, request: Request):
    return _svc(request).get_benchmark_job(job_id)
```

- [ ] **Add `start_benchmark` and `get_benchmark_job` to `TrainingService`**

```python
    def start_benchmark(self, n: int = 3) -> "TrainingJob":
        from datetime import datetime
        job_id = f"benchmark-{int(datetime.now().timestamp())}"
        job = TrainingJob(
            job_id=job_id,
            model_name="benchmark",
            status="queued",
            created_at=datetime.now().isoformat(),
        )
        with self._lock:
            self._jobs[job_id] = job
        t = threading.Thread(target=self._run_benchmark_job, args=(job, n), daemon=True)
        t.start()
        return job

    def _run_benchmark_job(self, job: "TrainingJob", n: int) -> None:
        from datetime import datetime
        import sqlite3
        try:
            with self._lock:
                job.status = "running"

            # Find last N completed runs
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                runs = conn.execute(
                    "SELECT run_id, video_id, cards_extracted FROM pipeline_runs "
                    "WHERE status='completed' ORDER BY started_at DESC LIMIT ?",
                    (n,),
                ).fetchall()
                videos = {
                    r["id"]: r["source_path"]
                    for r in conn.execute("SELECT id, source_path FROM videos").fetchall()
                }

            rows = []
            for run in runs:
                video_path = videos.get(run["video_id"], "")
                before = run["cards_extracted"]
                # Re-run pipeline (fire and wait synchronously)
                after = self._rerun_video(video_path)
                video_name = Path(video_path).name
                rows.append({
                    "video": video_name,
                    "before": before,
                    "after": after,
                    "delta": after - before,
                })

            with self._lock:
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
                job.progress = {"rows": rows}

        except Exception as exc:
            logger.exception("Benchmark job %s failed", job.job_id)
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = datetime.now().isoformat()

    def _rerun_video(self, video_path: str) -> int:
        """Re-run the pipeline on a video and return cards_extracted count."""
        import subprocess, sys, uuid, sqlite3, time
        from pathlib import Path as _Path
        run_id = f"benchmark-{uuid.uuid4().hex[:8]}"
        out_dir = _Path(self.db_path).parent / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "pipeline.card_capture_flow",
            "--no-pylint", "run",
            "--video", video_path,
            "--output-dir", str(out_dir),
            "--db", str(self.db_path),
            "--detector", "docaligner",
            "--config-preset", "balanced",
            "--ui-run-id", run_id,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline failed: {proc.stderr[-500:]}")
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT cards_extracted FROM pipeline_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return row[0] if row else 0

    def get_benchmark_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "rows": job.progress.get("rows", []) if job.progress else [],
            "error": job.error,
        }
```

- [ ] **Verify hub page shows benchmark section and chart**

Open `http://localhost:5173/training` — the benchmark section and sparklines should appear below the four panels.

- [ ] **Commit**

```bash
git add app/web/src/routes/training/+page.svelte app/api/training.py app/services/training_service.py
git commit -m "feat(training): benchmark panel and accuracy sparklines on training hub"
```

---

## Self-Review Checklist

After all tasks are complete:

- [ ] Run all backend tests: `python3 -m pytest tests/app/test_presence_sampler.py tests/app/test_training_api.py -v`
- [ ] Process a new video through the UI and verify presence_samples rows appear in DB: `sqlite3 card_capture_output/cards.sqlite "SELECT COUNT(*) FROM presence_samples;"`
- [ ] Open `/training` and verify all four pending counts load
- [ ] Label 5 presence frames — confirm queue count decrements
- [ ] Label 5 FB frames — confirm new "Not a card" option works
- [ ] Hit "Retrain all" — confirm progress updates appear and job completes
- [ ] After retrain, verify accuracy numbers update in the hub panels
- [ ] Process a video, then hit "Run pipeline on last 3 videos" — verify before/after table appears
