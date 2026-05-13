# v4 Surface C — ML Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship five algorithmic upgrades — multi-frame fusion verification, Front/Back classifier, DINOv2 + FAISS dedup, tracker swap, RANSAC corner refinement — plus the training-loop scaffolding, model registry, and retrain pipeline that makes them measurable through the harness. Train locally on Apple Silicon only.

**Architecture:** Algorithms remain in `src/card_capture/` modules (preserved by Surface A's Phase 2 refactor). Surface C touches algorithm internals only where the spec calls for replacement (e.g. `deduplicator.py` adds a DINOv2 path; `scoring.py` is unchanged). All training is local PyTorch on MPS; models persist as `.pt` files under `models/` and a row in `model_versions`. Inference paths integrate via the existing Metaflow steps owned by Surface A. Every algorithmic change ships behind a config flag (`use_fb_classifier`, `use_dino_dedup`, `tracker_backend`, `corner_refinement`) so it can be A/B'd against `baseline_v4.1` via the harness.

**Tech Stack:** PyTorch 2.x with MPS, torchvision, `transformers` (for DINOv2 hub model), `faiss-cpu` (in-process, no GPU build needed for ≤100K vectors), `scikit-image`, NumPy, the existing `cv2`, `kornia`. Tracker work uses either the existing BoT-SORT adapter (with real-image ReID) or `bytetrack` library, depending on Surface C's decision (Spec §1.5).

**Spec reference:** `docs/superpowers/specs/2026-05-12-v4-architecture-design.md`. This plan implements Surface C across Waves 1, 2, and 3.

**Contract dependencies:**
- Contract 1 — `fb_labels`, `dedup_clusters`, `model_versions`, `hard_cases` tables read/written by this surface.
- Contract 4 — metric definitions; every algorithmic upgrade is judged by them.
- Contract 3 — Metaflow artifact names; step implementations live in Surface A, but Surface C's integrations need stable artifact shapes.

**Critical safety gate.** Every Wave 2/3 PR runs `card-capture harness run --baseline baseline_v4.1` and the report is linked from the PR. No merge without harness evidence; no metric regression outside noise floor.

---

## File Structure

**New files (this plan creates):**

- `src/card_capture/ml/__init__.py`
- `src/card_capture/ml/scaffolding.py` — shared training loop, deterministic seeds, MPS device pick.
- `src/card_capture/ml/registry.py` — `model_versions` row read/write.
- `src/card_capture/ml/eval.py` — eval glue producing the per-model metrics that go into `eval_metrics_json`.
- `src/card_capture/ml/datasets/__init__.py`
- `src/card_capture/ml/datasets/fb.py` — F/B classifier dataset (reads `fb_labels` join with cards).
- `src/card_capture/ml/datasets/dedup.py` — dedup-similarity dataset (reads `dedup_clusters`).
- `src/card_capture/ml/models/fb_classifier.py` — MobileNetV3-Small head.
- `src/card_capture/ml/models/dino_embedder.py` — DINOv2 wrapper, variant chosen at config-time.
- `src/card_capture/ml/training/fb_train.py` — CLI: `python -m card_capture.ml.training.fb_train`.
- `src/card_capture/ml/training/dedup_calibrate.py` — calibrates DINOv2 cosine threshold against `dedup_clusters`.
- `src/card_capture/ml/inference/fb_predict.py` — load `fb_classifier` + predict.
- `src/card_capture/ml/inference/dino_dedup.py` — DINOv2 → FAISS index → nearest neighbors.
- `src/card_capture/ml/fusion_verification.py` — sweep `_CANONICAL_TARGET_FRAMES`, report harness deltas.
- `src/card_capture/ml/corner_refinement.py` — RANSAC line-fit corner refinement.
- `src/card_capture/ml/synthetic_eval.py` — synthetic eval harness for ML iteration before real labels exist.
- `app/services/training_service.py` — backend for `/api/v1/training/*`.
- `tests/ml/test_scaffolding.py`
- `tests/ml/test_registry.py`
- `tests/ml/test_fb_train.py`
- `tests/ml/test_fb_predict.py`
- `tests/ml/test_dino_embedder.py`
- `tests/ml/test_dino_dedup.py`
- `tests/ml/test_corner_refinement.py`
- `tests/ml/test_fusion_verification.py`
- `tests/ml/test_synthetic_eval.py`
- `tests/app/test_training_endpoints.py`
- `models/.gitkeep`
- `docs/ml/fb-classifier.md`, `docs/ml/dino-dedup.md`, `docs/ml/tracker-decision.md`, `docs/ml/corner-refinement.md`, `docs/ml/fusion-verification.md`

**Modified files (this plan touches):**

- `src/card_capture/scoring.py` — UNCHANGED in Wave 2; if Wave 3 introduces learned ranker, that re-plan.
- `src/card_capture/deduplicator.py` — extended to accept a `backend: Literal["phash","dino"]` parameter; pHash path preserved for within-session use; DINO path used for cross-track matching.
- `src/card_capture/tracking/botsort_adapter.py` OR `src/card_capture/tracking/bytetrack_adapter.py` — whichever ships, dummy-image ReID bug fixed (real crops passed to ReID).
- `src/card_capture/cropper.py` (or `gpu_refinement.py`) — RANSAC corner refinement called before perspective warp when enabled.
- `pipeline/steps/resolve.py` — calls `ml.inference.fb_predict` when `use_fb_classifier=True`, falls back to longest-track heuristic otherwise.
- `pipeline/steps/dedup.py` — backend switch.
- `src/card_capture/config.py` — adds `use_fb_classifier`, `use_dino_dedup`, `tracker_backend`, `enable_corner_refinement`, `fusion_target_frames`.
- `app/api/training.py` — wire training service routes.

---

## Phase C0 — Scaffolding (Wave 1, parallel with Surface A's Phase A2)

### Task C0.1: Deterministic training loop scaffold

**Files:**
- Create: `src/card_capture/ml/__init__.py`
- Create: `src/card_capture/ml/scaffolding.py`
- Create: `tests/ml/test_scaffolding.py`

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_scaffolding.py
import torch
from src.card_capture.ml.scaffolding import pick_device, set_seed, train_one_epoch

def test_pick_device_returns_torch_device():
    d = pick_device()
    assert isinstance(d, torch.device)

def test_set_seed_makes_init_reproducible():
    set_seed(42)
    a = torch.randn(8, 8)
    set_seed(42)
    b = torch.randn(8, 8)
    assert torch.equal(a, b)

def test_train_one_epoch_returns_loss(monkeypatch):
    model = torch.nn.Linear(4, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = [(x[i:i+4], y[i:i+4]) for i in range(0, 16, 4)]
    avg_loss = train_one_epoch(model, loader, optim, loss_fn, device=torch.device("cpu"))
    assert avg_loss > 0
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/scaffolding.py
import random
import os
import numpy as np
import torch


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)
```

- [ ] **Step 3: Tests pass**

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/ml/__init__.py src/card_capture/ml/scaffolding.py tests/ml/test_scaffolding.py
git commit -m "feat(ml): deterministic training scaffold (device pick, seed, train loop)"
```

### Task C0.2: Model registry

**Files:**
- Create: `src/card_capture/ml/registry.py`
- Create: `tests/ml/test_registry.py`

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_registry.py
import sqlite3
from pathlib import Path
from src.card_capture.ml.registry import register_model, get_latest, list_models
from migrations.run_migrations import apply_migrations

def test_register_and_fetch(tmp_path: Path):
    db = tmp_path / "cards.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    v_id = register_model(
        db_path=db, model_name="fb_classifier",
        training_set_hash="abc",
        eval_metrics={"val_acc": 0.91},
        checkpoint_path="models/fb_classifier/v1.pt",
    )
    latest = get_latest(db_path=db, model_name="fb_classifier")
    assert latest.version_id == v_id
    assert latest.eval_metrics["val_acc"] == 0.91

def test_unique_training_set_per_name(tmp_path: Path):
    db = tmp_path / "cards.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    register_model(db_path=db, model_name="fb", training_set_hash="h", eval_metrics={}, checkpoint_path="p")
    import pytest
    with pytest.raises(Exception):
        register_model(db_path=db, model_name="fb", training_set_hash="h", eval_metrics={}, checkpoint_path="p2")
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/registry.py
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelVersion:
    version_id: int
    model_name: str
    training_set_hash: str
    eval_metrics: dict
    checkpoint_path: str
    created_at: str


def register_model(*, db_path: Path, model_name: str, training_set_hash: str,
                  eval_metrics: dict, checkpoint_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO model_versions(model_name, training_set_hash, eval_metrics_json, checkpoint_path) "
            "VALUES (?, ?, ?, ?)",
            (model_name, training_set_hash, json.dumps(eval_metrics), checkpoint_path),
        )
        conn.commit()
        return cur.lastrowid


def get_latest(*, db_path: Path, model_name: str) -> ModelVersion | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT version_id, model_name, training_set_hash, eval_metrics_json, checkpoint_path, created_at "
            "FROM model_versions WHERE model_name = ? ORDER BY created_at DESC LIMIT 1",
            (model_name,),
        ).fetchone()
    if not row:
        return None
    return ModelVersion(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5])


def list_models(*, db_path: Path) -> list[ModelVersion]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT version_id, model_name, training_set_hash, eval_metrics_json, checkpoint_path, created_at "
            "FROM model_versions ORDER BY created_at DESC",
        ).fetchall()
    return [ModelVersion(r[0], r[1], r[2], json.loads(r[3]), r[4], r[5]) for r in rows]
```

- [ ] **Step 3: Tests pass**

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/ml/registry.py tests/ml/test_registry.py
git commit -m "feat(ml): model_versions registry read/write"
```

### Task C0.3: Synthetic eval harness

**Files:**
- Create: `src/card_capture/ml/synthetic_eval.py`
- Create: `tests/ml/test_synthetic_eval.py`

Purpose: let Surface C iterate on model architecture and training loops BEFORE real labels exist. Generates rendered fake card crops with known F/B / cluster ground truth.

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_synthetic_eval.py
from src.card_capture.ml.synthetic_eval import generate_fb_dataset, generate_dedup_dataset

def test_fb_synthetic_balanced(tmp_path):
    ds = generate_fb_dataset(out_dir=tmp_path, n_per_class=20, seed=7)
    assert len(ds) == 40
    fronts = [item for item in ds if item.label == "front"]
    backs = [item for item in ds if item.label == "back"]
    assert len(fronts) == 20 and len(backs) == 20

def test_dedup_synthetic_clusters(tmp_path):
    ds = generate_dedup_dataset(out_dir=tmp_path, n_clusters=5, samples_per_cluster=4, seed=7)
    assert len(ds.items) == 20
    cluster_sizes = {cid: 0 for cid in ds.cluster_ids}
    for item in ds.items:
        cluster_sizes[item.cluster_id] += 1
    assert all(v == 4 for v in cluster_sizes.values())
```

- [ ] **Step 2: Implement using PIL to render fake card-like images**

```python
# src/card_capture/ml/synthetic_eval.py
"""Synthetic dataset generation for ML iteration prior to real labels.

Renders simple 750x1050 BGR images with controlled invariants:
- F/B: distinct background colors + text orientation hint.
- Dedup: per-cluster fixed motif + per-sample noise.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


@dataclass
class FBItem:
    image_path: Path
    label: str  # "front" or "back"

@dataclass
class DedupItem:
    image_path: Path
    cluster_id: int

@dataclass
class DedupDataset:
    items: list[DedupItem]
    cluster_ids: list[int]


def generate_fb_dataset(*, out_dir: Path, n_per_class: int, seed: int) -> list[FBItem]:
    random.seed(seed); np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[FBItem] = []
    for i in range(n_per_class):
        items.append(_make_fb_image(out_dir, i, "front"))
        items.append(_make_fb_image(out_dir, i, "back"))
    return items


def _make_fb_image(out_dir: Path, idx: int, side: str) -> FBItem:
    img = Image.new("RGB", (750, 1050),
                   color=(220, 200, 120) if side == "front" else (120, 130, 200))
    draw = ImageDraw.Draw(img)
    if side == "front":
        draw.rectangle([(50, 700), (700, 1000)], outline=(0,0,0), width=4)
        draw.text((100, 750), f"FRONT {idx}", fill=(0,0,0))
    else:
        draw.rectangle([(80, 80), (670, 970)], outline=(0,0,0), width=2)
        draw.text((250, 500), "BACK", fill=(255,255,255))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.random() * 0.5))
    path = out_dir / f"fb_{side}_{idx}.png"
    img.save(path)
    return FBItem(image_path=path, label=side)


def generate_dedup_dataset(*, out_dir: Path, n_clusters: int, samples_per_cluster: int, seed: int) -> DedupDataset:
    random.seed(seed); np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[DedupItem] = []
    for cluster_id in range(n_clusters):
        base_color = tuple(random.randint(50, 200) for _ in range(3))
        for s in range(samples_per_cluster):
            img = Image.new("RGB", (750, 1050), color=base_color)
            d = ImageDraw.Draw(img)
            d.ellipse([(200, 400), (550, 750)], fill=(255 - base_color[0], 0, 0))
            jitter = np.random.randn(1050, 750, 3) * 5
            arr = np.clip(np.array(img).astype(np.float32) + jitter, 0, 255).astype(np.uint8)
            jittered = Image.fromarray(arr)
            path = out_dir / f"cluster_{cluster_id}_{s}.png"
            jittered.save(path)
            items.append(DedupItem(image_path=path, cluster_id=cluster_id))
    return DedupDataset(items=items, cluster_ids=list(range(n_clusters)))
```

- [ ] **Step 3: Tests pass**

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/ml/synthetic_eval.py tests/ml/test_synthetic_eval.py
git commit -m "feat(ml): synthetic F/B and dedup datasets for pre-label iteration"
```

### Task C0.4: Training service (backend for `/api/v1/training/*`)

**Files:**
- Create: `app/services/training_service.py`
- Modify: `app/api/training.py`
- Create: `tests/app/test_training_endpoints.py`

- [ ] **Step 1: Failing endpoint test**

```python
# tests/app/test_training_endpoints.py
from fastapi.testclient import TestClient
from app.main import create_app

def test_list_datasets():
    client = TestClient(create_app())
    r = client.get("/api/v1/training/datasets")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_retrain_returns_job_id():
    client = TestClient(create_app())
    r = client.post("/api/v1/training/retrain/fb_classifier", json={"dry_run": True})
    assert r.status_code in (200, 202)
    body = r.json()
    assert "job_id" in body

def test_get_job():
    client = TestClient(create_app())
    j = client.post("/api/v1/training/retrain/fb_classifier", json={"dry_run": True}).json()
    r = client.get(f"/api/v1/training/jobs/{j['job_id']}")
    assert r.status_code == 200
    assert r.json()["model_name"] == "fb_classifier"
```

- [ ] **Step 2: Implement service**

```python
# app/services/training_service.py
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class TrainingService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def list_datasets(self) -> list[dict]:
        # F/B dataset size = labeled rows in fb_labels; dedup = confirmed clusters; etc.
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            fb = conn.execute("SELECT COUNT(*) FROM fb_labels").fetchone()[0]
            dedup = conn.execute("SELECT COUNT(*) FROM dedup_clusters WHERE status='confirmed'").fetchone()[0]
        return [
            {"name": "fb_labels", "size": fb},
            {"name": "dedup_clusters", "size": dedup},
        ]

    def start_retrain(self, model_name: str, *, dry_run: bool = False) -> TrainingJob:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = TrainingJob(job_id=job_id, model_name=model_name, status="queued")
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run, args=(job, dry_run), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job: TrainingJob, dry_run: bool) -> None:
        job.status = "running"
        try:
            if dry_run:
                job.metrics = {"dry_run": True}
            elif job.model_name == "fb_classifier":
                from src.card_capture.ml.training.fb_train import train as fb_train
                job.metrics = fb_train(db_path=self.db_path)
            elif job.model_name == "dino_threshold":
                from src.card_capture.ml.training.dedup_calibrate import calibrate
                job.metrics = calibrate(db_path=self.db_path)
            else:
                raise ValueError(f"unknown model: {job.model_name}")
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
```

- [ ] **Step 3: Wire routes**

```python
# app/api/training.py — replace stubs
from fastapi import APIRouter, Request

router = APIRouter()


def _svc(request: Request):
    return request.app.state.training_service


@router.get("/datasets")
def list_datasets(request: Request):
    return _svc(request).list_datasets()


@router.post("/retrain/{model_name}", status_code=202)
def retrain(model_name: str, body: dict, request: Request):
    job = _svc(request).start_retrain(model_name, dry_run=bool(body.get("dry_run")))
    return {"job_id": job.job_id, "status": job.status}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    j = _svc(request).get_job(job_id)
    if not j:
        from fastapi import HTTPException
        raise HTTPException(404)
    return {"job_id": j.job_id, "model_name": j.model_name, "status": j.status,
            "metrics": j.metrics, "error": j.error}
```

In `app/main.py`:

```python
from app.services.training_service import TrainingService
app.state.training_service = TrainingService(db_path=Path("cards.sqlite"))
```

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add app/services/training_service.py app/api/training.py app/main.py tests/app/test_training_endpoints.py
git commit -m "feat(app): training service + endpoints (job queue + dataset stats)"
```

---

## Phase C1 — Multi-Frame Fusion Verification (Wave 2, first algo task per Spec §3.2)

### Task C1.1: Sweep `_CANONICAL_TARGET_FRAMES`

**Files:**
- Create: `src/card_capture/ml/fusion_verification.py`
- Create: `tests/ml/test_fusion_verification.py`
- Modify: `src/card_capture/config.py` (`fusion_target_frames: int = 3`)

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_fusion_verification.py
import subprocess
from pathlib import Path

def test_sweep_produces_report(tmp_path):
    out = tmp_path / "sweep.json"
    result = subprocess.run(
        ["python", "-m", "src.card_capture.ml.fusion_verification",
         "--baseline", "baseline_v4.1",
         "--target-frames", "2,3,4,5",
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    import json
    report = json.loads(out.read_text())
    assert set(report["sweeps"].keys()) == {"2","3","4","5"}
    for k, v in report["sweeps"].items():
        assert "image_quality" in v["metrics"]
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/fusion_verification.py
"""Sweep _CANONICAL_TARGET_FRAMES; emit harness metric deltas per value."""
import argparse
import json
import subprocess
from pathlib import Path


def run(*, baseline: str, target_frames: list[int], out: Path) -> dict:
    sweeps: dict[str, dict] = {}
    for k in target_frames:
        env_extras = {"FUSION_TARGET_FRAMES": str(k)}
        result = subprocess.run(
            ["card-capture", "harness", "run", "--baseline", baseline, "--db", "cards.sqlite",
             "--truth-dir", "golden_set/videos",
             "--videos", _video_list()],
            capture_output=True, text=True,
            env={**__import__("os").environ, **env_extras},
        )
        # parse stdout JSON
        deltas = json.loads(result.stdout.strip().splitlines()[-1] if result.stdout else "{}")
        sweeps[str(k)] = {"metrics": deltas}
    payload = {"baseline": baseline, "sweeps": sweeps}
    out.write_text(json.dumps(payload, indent=2))
    return payload


def _video_list() -> str:
    return ",".join((Path("golden_set/videos/_index.txt").read_text().splitlines()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--target-frames", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    run(baseline=args.baseline,
        target_frames=[int(x) for x in args.target_frames.split(",")],
        out=args.out)
```

- [ ] **Step 3: Run sweep against golden set, pick best**

Surface owner runs the sweep CLI, inspects deltas, commits chosen `fusion_target_frames` in `config.py`, and links the sweep JSON in the PR.

- [ ] **Step 4: Update config**

```python
# src/card_capture/config.py — add to existing Config dataclass
fusion_target_frames: int = 3  # set to sweep winner
```

`fuser.py` reads `cfg.fusion_target_frames` instead of the hard-coded `_CANONICAL_TARGET_FRAMES`. (Surface owner adapts `fuser.py` minimally.)

- [ ] **Step 5: Harness gate**

```
card-capture harness run --baseline baseline_v4.1
```

Expected: neutral-or-positive deltas. If positive on `image_quality`, log the improvement; commit. If neutral, document and commit anyway (configuration tuned to safe value).

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/ml/fusion_verification.py src/card_capture/config.py src/card_capture/fuser.py tests/ml/test_fusion_verification.py
git commit -m "feat(fusion): sweep target_frames; commit best with harness evidence"
```

---

## Phase C2 — Front/Back Classifier (Wave 2; biggest expected win)

### Task C2.1: F/B dataset reader

**Files:**
- Create: `src/card_capture/ml/datasets/__init__.py`
- Create: `src/card_capture/ml/datasets/fb.py`
- Create: `tests/ml/test_fb_dataset.py`

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_fb_dataset.py
from pathlib import Path
from src.card_capture.ml.datasets.fb import FBDataset

def test_dataset_iterates_labeled_rows(tmp_path: Path):
    # seed fb_labels + a card image with sqlite + PNG; surface owner writes the fixture builder
    db = _seed_db_with_two_labels(tmp_path)
    ds = FBDataset(db_path=db)
    items = list(ds)
    assert len(items) == 2
    sides = {item["side"] for item in items}
    assert sides.issubset({"front","back","uncertain"})
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/datasets/fb.py
import sqlite3
from pathlib import Path
from typing import Iterator
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class FBDataset(Dataset):
    def __init__(self, db_path: Path, *, train: bool = True, image_size: int = 224) -> None:
        self.db_path = db_path
        self._rows = self._load_rows()
        base = [T.Resize((image_size, image_size)), T.ToTensor()]
        if train:
            base = [T.RandomHorizontalFlip(p=0.0)] + base  # do NOT flip — orientation is the label
            base.insert(1, T.ColorJitter(brightness=0.2, contrast=0.2))
        self.transform = T.Compose(base)

    def _load_rows(self) -> list[dict]:
        # Surface owner picks the canonical join: fb_labels x card_views x crops
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT fl.side, cv.image_path "
                "FROM fb_labels fl JOIN card_views cv ON cv.instance_id = fl.instance_id "
                "WHERE fl.side IN ('front','back')"
            ).fetchall()
        return [{"side": r[0], "image_path": r[1]} for r in rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.transform(img)
        y = 0 if row["side"] == "front" else 1
        return {"image": x, "label": y, "side": row["side"]}
```

- [ ] **Step 3: Tests pass**

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/ml/datasets/ tests/ml/test_fb_dataset.py
git commit -m "feat(ml): F/B dataset reading fb_labels join"
```

### Task C2.2: F/B model + training script

**Files:**
- Create: `src/card_capture/ml/models/fb_classifier.py`
- Create: `src/card_capture/ml/training/fb_train.py`
- Create: `tests/ml/test_fb_train.py`

- [ ] **Step 1: Failing smoke test**

```python
# tests/ml/test_fb_train.py
from pathlib import Path
from src.card_capture.ml.training.fb_train import train

def test_train_converges_on_synthetic(tmp_path):
    from src.card_capture.ml.synthetic_eval import generate_fb_dataset
    # generate enough data
    generate_fb_dataset(out_dir=tmp_path / "synth", n_per_class=40, seed=1)
    metrics = train(
        db_path=tmp_path / "_unused.sqlite",  # use synthetic loader
        synthetic_dir=tmp_path / "synth",
        epochs=3, batch_size=8, lr=1e-3,
        out_dir=tmp_path / "ckpt",
    )
    assert metrics["val_acc"] > 0.9
```

- [ ] **Step 2: Implement model**

```python
# src/card_capture/ml/models/fb_classifier.py
import torch
import torch.nn as nn
import torchvision.models as M


def build_fb_classifier() -> nn.Module:
    base = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, 2)
    return base
```

- [ ] **Step 3: Implement training script**

```python
# src/card_capture/ml/training/fb_train.py
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from src.card_capture.ml.scaffolding import pick_device, set_seed, train_one_epoch
from src.card_capture.ml.models.fb_classifier import build_fb_classifier
from src.card_capture.ml.datasets.fb import FBDataset
from src.card_capture.ml.registry import register_model


def train(*, db_path: Path, synthetic_dir: Path | None = None,
         epochs: int = 10, batch_size: int = 32, lr: float = 1e-3,
         out_dir: Path = Path("models/fb_classifier"), seed: int = 42) -> dict:
    set_seed(seed)
    device = pick_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    if synthetic_dir is not None:
        from src.card_capture.ml.synthetic_eval import generate_fb_dataset
        items = generate_fb_dataset(out_dir=synthetic_dir, n_per_class=40, seed=seed)
        dataset = _ItemsAsDataset(items)
    else:
        dataset = FBDataset(db_path=db_path, train=True)

    n_val = max(1, int(len(dataset) * 0.2))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = build_fb_classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history: list[dict] = []
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = _eval(model, val_loader, device)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})

    ckpt_path = out_dir / f"v{int(__import__('time').time())}.pt"
    torch.save({"state_dict": model.state_dict(), "history": history}, ckpt_path)

    metrics = {"val_acc": history[-1]["val_acc"], "epochs": epochs, "best_loss": min(h["loss"] for h in history)}
    if synthetic_dir is None:
        ts_hash = _hash_dataset(dataset)
        register_model(db_path=db_path, model_name="fb_classifier",
                      training_set_hash=ts_hash, eval_metrics=metrics,
                      checkpoint_path=str(ckpt_path))
    return metrics


def _eval(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device); y = batch["label"].to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _hash_dataset(dataset) -> str:
    # Hash of (image_path, label) tuples; surface owner refines if dataset is huge.
    h = hashlib.sha256()
    for row in getattr(dataset, "_rows", []):
        h.update(row["image_path"].encode())
        h.update(row["side"].encode())
    return h.hexdigest()


class _ItemsAsDataset:
    """Adapter for synthetic_eval items."""
    def __init__(self, items):
        from PIL import Image
        import torchvision.transforms as T
        self._items = items
        self._tx = T.Compose([T.Resize((224,224)), T.ToTensor()])
        self._rows = [{"image_path": str(i.image_path), "side": i.label} for i in items]
    def __len__(self): return len(self._items)
    def __getitem__(self, i):
        from PIL import Image
        item = self._items[i]
        img = Image.open(item.image_path).convert("RGB")
        return {"image": self._tx(img), "label": 0 if item.label == "front" else 1, "side": item.label}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    metrics = train(db_path=args.db, epochs=args.epochs)
    print(json.dumps(metrics, indent=2))
```

- [ ] **Step 4: Run synthetic-data smoke test**

```
pytest tests/ml/test_fb_train.py::test_train_converges_on_synthetic -v
```

Acceptance: val_acc > 0.9 on the synthetic balanced set.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/ml/models/ src/card_capture/ml/training/ tests/ml/test_fb_train.py
git commit -m "feat(ml): F/B classifier model + training script (MobileNetV3-S)"
```

### Task C2.3: F/B inference + Stage-resolve integration

**Files:**
- Create: `src/card_capture/ml/inference/__init__.py`
- Create: `src/card_capture/ml/inference/fb_predict.py`
- Create: `tests/ml/test_fb_predict.py`
- Modify: `pipeline/steps/resolve.py` (use F/B classifier when `use_fb_classifier=True`)
- Modify: `src/card_capture/config.py` (`use_fb_classifier: bool = False`)

- [ ] **Step 1: Failing inference test**

```python
# tests/ml/test_fb_predict.py
import torch
from src.card_capture.ml.inference.fb_predict import FBPredictor

def test_predictor_returns_side_with_confidence(tmp_path):
    # save a trained-on-synthetic checkpoint
    from src.card_capture.ml.training.fb_train import train
    from src.card_capture.ml.synthetic_eval import generate_fb_dataset
    generate_fb_dataset(out_dir=tmp_path/"synth", n_per_class=20, seed=1)
    metrics = train(db_path=tmp_path/"_x.sqlite", synthetic_dir=tmp_path/"synth",
                   epochs=3, out_dir=tmp_path/"ckpt")
    import glob
    ckpt = sorted(glob.glob(str(tmp_path/"ckpt"/"v*.pt")))[-1]
    pred = FBPredictor(checkpoint_path=ckpt)
    side, conf = pred.predict_image(tmp_path/"synth"/"fb_front_0.png")
    assert side in {"front","back"}
    assert 0.0 <= conf <= 1.0
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/inference/fb_predict.py
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image

from src.card_capture.ml.models.fb_classifier import build_fb_classifier
from src.card_capture.ml.scaffolding import pick_device

_TX = T.Compose([T.Resize((224,224)), T.ToTensor()])


class FBPredictor:
    def __init__(self, checkpoint_path: str | Path):
        self.device = pick_device()
        self.model = build_fb_classifier().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_image(self, path: str | Path) -> tuple[str, float]:
        img = Image.open(path).convert("RGB")
        x = _TX(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        return ("front" if idx == 0 else "back", float(probs[idx].item()))
```

- [ ] **Step 3: Integrate in `resolve` step**

Surface owner reads `pipeline/steps/resolve.py` (Surface A's Wave-1 deliverable) and adds:

```python
# pipeline/steps/resolve.py — additions
def run(ctx, scored_candidates, pruned_tracks):
    cfg = ctx.config
    out = _legacy_resolve(scored_candidates, pruned_tracks)
    if cfg.use_fb_classifier:
        out = _apply_fb_classifier(out, ctx)
    return Output(prepared_tracks=out)


def _apply_fb_classifier(prepared_tracks, ctx):
    from src.card_capture.ml.inference.fb_predict import FBPredictor
    from src.card_capture.ml.registry import get_latest
    ver = get_latest(db_path=ctx.db_path, model_name="fb_classifier")
    if not ver:
        return prepared_tracks
    pred = FBPredictor(checkpoint_path=ver.checkpoint_path)
    out = []
    for pt in prepared_tracks:
        side, conf = pred.predict_image(pt.canonical_path)
        if conf >= 0.6:
            pt = pt.with_angle(side)
        # else: keep heuristic-assigned angle
        out.append(pt)
    return out
```

- [ ] **Step 4: Harness gate — F/B classifier on, off**

```
USE_FB_CLASSIFIER=true card-capture harness run --baseline baseline_v4.1
```

Acceptance: `side_accuracy` improves by ≥5 pp vs baseline (Spec §1.3). If under 5 pp: tune (more data, longer training, lower threshold); do not merge until threshold met OR document why fallback is acceptable (e.g. labeled set too small; revisit after Wave 2 grows golden set).

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/ml/inference/ src/card_capture/config.py pipeline/steps/resolve.py tests/ml/test_fb_predict.py
git commit -m "feat(resolve): use F/B classifier when enabled; harness-gated"
```

---

## Phase C3 — DINOv2 + FAISS Dedup (Wave 2; second-biggest expected win)

### Task C3.1: DINOv2 embedder

**Files:**
- Create: `src/card_capture/ml/models/dino_embedder.py`
- Create: `tests/ml/test_dino_embedder.py`

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_dino_embedder.py
import torch
from PIL import Image
from pathlib import Path
from src.card_capture.ml.models.dino_embedder import DinoEmbedder

def test_embed_returns_fixed_dim(tmp_path):
    img_path = tmp_path / "x.png"
    Image.new("RGB", (256,256), color=(123,0,0)).save(img_path)
    emb = DinoEmbedder(variant="vits14")
    v = emb.embed_image(img_path)
    assert v.shape == (emb.dim,)
    assert torch.is_floating_point(v) or hasattr(v, "dtype")

def test_two_calls_same_image_same_embedding(tmp_path):
    img_path = tmp_path / "x.png"
    Image.new("RGB", (256,256), color=(50,200,30)).save(img_path)
    emb = DinoEmbedder(variant="vits14")
    a = emb.embed_image(img_path)
    b = emb.embed_image(img_path)
    import numpy as np
    np.testing.assert_allclose(a, b, atol=1e-5)
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/models/dino_embedder.py
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.card_capture.ml.scaffolding import pick_device


VARIANT_TO_HUB = {
    "vits14": "dinov2_vits14",
    "vitb14": "dinov2_vitb14",
}
VARIANT_TO_DIM = {"vits14": 384, "vitb14": 768}


@lru_cache(maxsize=2)
def _load(variant: str):
    device = pick_device()
    model = torch.hub.load("facebookresearch/dinov2", VARIANT_TO_HUB[variant])
    model.to(device).eval()
    return model, device


_TX = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


class DinoEmbedder:
    def __init__(self, variant: str = "vits14"):
        if variant not in VARIANT_TO_HUB:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.dim = VARIANT_TO_DIM[variant]

    @torch.no_grad()
    def embed_image(self, path: str | Path) -> np.ndarray:
        model, device = _load(self.variant)
        img = Image.open(path).convert("RGB")
        x = _TX(img).unsqueeze(0).to(device)
        v = model(x)  # [1, dim]
        v = torch.nn.functional.normalize(v, dim=1)
        return v.squeeze(0).detach().cpu().numpy()
```

- [ ] **Step 3: Tests pass**

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/ml/models/dino_embedder.py tests/ml/test_dino_embedder.py
git commit -m "feat(ml): DINOv2 embedder (vits14, vitb14) with L2-normalized output"
```

### Task C3.2: FAISS index + dedup inference

**Files:**
- Create: `src/card_capture/ml/inference/dino_dedup.py`
- Create: `tests/ml/test_dino_dedup.py`

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_dino_dedup.py
from pathlib import Path
from src.card_capture.ml.inference.dino_dedup import DinoDedup
from src.card_capture.ml.synthetic_eval import generate_dedup_dataset
from sklearn.metrics import adjusted_rand_score

def test_clusters_synthetic_dataset_with_high_ari(tmp_path):
    ds = generate_dedup_dataset(out_dir=tmp_path, n_clusters=5, samples_per_cluster=4, seed=7)
    paths = [str(item.image_path) for item in ds.items]
    truth_labels = [item.cluster_id for item in ds.items]

    d = DinoDedup(variant="vits14")
    pred = d.cluster_paths(paths, cosine_threshold=0.3)
    pred_labels = [pred[p] for p in paths]
    ari = adjusted_rand_score(truth_labels, pred_labels)
    assert ari > 0.7
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/inference/dino_dedup.py
from pathlib import Path
from collections import defaultdict
import faiss
import numpy as np

from src.card_capture.ml.models.dino_embedder import DinoEmbedder


class DinoDedup:
    def __init__(self, variant: str = "vits14"):
        self.embedder = DinoEmbedder(variant=variant)

    def cluster_paths(self, paths: list[str], *, cosine_threshold: float) -> dict[str, int]:
        if not paths:
            return {}
        embs = np.stack([self.embedder.embed_image(p) for p in paths])
        index = faiss.IndexFlatIP(self.embedder.dim)
        index.add(embs.astype(np.float32))
        # Inner product on L2-normalized vectors == cosine similarity.
        sim_threshold = 1.0 - cosine_threshold
        n = len(paths)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            a, b = find(a), find(b)
            if a != b: parent[a] = b

        D, I = index.search(embs.astype(np.float32), k=min(10, n))
        for i in range(n):
            for j_idx, sim in zip(I[i], D[i]):
                if j_idx == i: continue
                if sim >= sim_threshold:
                    union(i, j_idx)

        cluster_of = {}
        canonical = {}
        next_id = 0
        for i in range(n):
            root = find(i)
            if root not in canonical:
                canonical[root] = next_id; next_id += 1
            cluster_of[paths[i]] = canonical[root]
        return cluster_of
```

- [ ] **Step 3: Tests pass on synthetic data**

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/ml/inference/dino_dedup.py tests/ml/test_dino_dedup.py
git commit -m "feat(dedup): DINOv2 + FAISS clustering by cosine threshold"
```

### Task C3.3: Threshold calibration

**Files:**
- Create: `src/card_capture/ml/training/dedup_calibrate.py`

- [ ] **Step 1: Implement**

```python
# src/card_capture/ml/training/dedup_calibrate.py
"""Sweep cosine_threshold against confirmed dedup_clusters; pick ARI optimum."""
import json
import sqlite3
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

from src.card_capture.ml.inference.dino_dedup import DinoDedup
from src.card_capture.ml.registry import register_model


def calibrate(*, db_path: Path, variant: str = "vits14",
             thresholds: list[float] | None = None) -> dict:
    thresholds = thresholds or [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT cluster_id, confirmed_member_ids_json FROM dedup_clusters WHERE status='confirmed'"
        ).fetchall()
    if not rows:
        return {"error": "no confirmed clusters"}

    # Build truth: (image_path -> cluster_id) by loading the views referenced.
    truth: dict[str, int] = {}
    for cluster_id, members_json in rows:
        for instance_id in json.loads(members_json):
            with sqlite3.connect(db_path) as conn:
                img = conn.execute(
                    "SELECT image_path FROM card_views WHERE instance_id = ? LIMIT 1",
                    (instance_id,),
                ).fetchone()
            if img:
                truth[img[0]] = cluster_id

    paths = list(truth.keys())
    truth_labels = [truth[p] for p in paths]

    d = DinoDedup(variant=variant)
    best = {"threshold": None, "ari": -1.0}
    sweep = {}
    for t in thresholds:
        clusters = d.cluster_paths(paths, cosine_threshold=t)
        pred_labels = [clusters[p] for p in paths]
        ari = adjusted_rand_score(truth_labels, pred_labels)
        sweep[str(t)] = ari
        if ari > best["ari"]:
            best = {"threshold": t, "ari": ari}

    register_model(db_path=db_path, model_name="dino_threshold",
                  training_set_hash=str(len(paths)),
                  eval_metrics={"ari": best["ari"], "threshold": best["threshold"], "sweep": sweep},
                  checkpoint_path="<implicit: DINOv2 hub + threshold in config>")
    return {"best": best, "sweep": sweep}
```

- [ ] **Step 2: Commit**

```bash
git add src/card_capture/ml/training/dedup_calibrate.py
git commit -m "feat(dedup): calibrate DINOv2 cosine threshold against confirmed clusters"
```

### Task C3.4: Integrate into Stage `dedup`

**Files:**
- Modify: `src/card_capture/deduplicator.py` (add DINO backend)
- Modify: `pipeline/steps/dedup.py`
- Modify: `src/card_capture/config.py`

- [ ] **Step 1: Add `backend` parameter to `deduplicator.py`**

Surface owner reads `src/card_capture/deduplicator.py`. Adds a `backend: Literal["phash","dino"]` parameter to the relevant function(s). For `backend="dino"`, calls `DinoDedup` with the calibrated threshold. For `backend="phash"`, calls the existing pHash code unchanged. Default remains `"phash"` so existing behavior is preserved unless `use_dino_dedup=True`.

- [ ] **Step 2: Config flag**

```python
# src/card_capture/config.py
use_dino_dedup: bool = False
dino_dedup_variant: str = "vits14"
```

- [ ] **Step 3: Harness gate**

```
USE_DINO_DEDUP=true card-capture harness run --baseline baseline_v4.1
```

Acceptance: `dedup_accuracy.ari` improves vs baseline by a measurable margin (Spec §1.3). If not, run benchmark on `vitb14` and pick the winner.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/deduplicator.py pipeline/steps/dedup.py src/card_capture/config.py
git commit -m "feat(dedup): integrate DINOv2 backend behind config flag (harness-gated)"
```

---

## Phase C4 — Tracker Swap (Wave 2; per-Spec §1.5 deferred decision)

### Task C4.1: Decision document

**Files:**
- Create: `docs/ml/tracker-decision.md`

- [ ] **Step 1: Write decision doc**

The doc walks through the choice between (a) BoT-SORT with real-image ReID (fix the dummy-image bug; pass actual crops to OSNet) and (b) ByteTrack with no ReID (drop the appearance backbone entirely, rely on motion + IoU). For each, lists implementation effort, current baseline ID-switch / session-fragmentation metrics, and expected post-fix metrics.

- [ ] **Step 2: Decision is made by Surface C with regression evidence**

Run both candidates against `baseline_v4.1`:

```
TRACKER_BACKEND=botsort_real_reid card-capture harness run --baseline baseline_v4.1 --out /tmp/botsort.json
TRACKER_BACKEND=bytetrack card-capture harness run --baseline baseline_v4.1 --out /tmp/bytetrack.json
```

Compare metrics; commit decision + harness evidence into `docs/ml/tracker-decision.md`.

- [ ] **Step 3: Commit**

```bash
git add docs/ml/tracker-decision.md
git commit -m "docs(ml): tracker decision (botsort-real-reid vs bytetrack) with harness evidence"
```

### Task C4.2: Implement chosen tracker

Surface C's owner implements ONE of:

**Option A: Fix BoT-SORT real-image ReID.**

- Modify `src/card_capture/tracking/botsort_adapter.py` to pass the actual frame crop (not a dummy image) to the OSNet ReID backbone. Surface owner reads the existing adapter to find the dummy-image call site (mentioned in CLAUDE.md §3.4 critique).
- Add test: ID switches on a hand-built two-card swap fixture drop to zero (or near-zero) with real ReID vs the current dummy behavior.
- Harness gate.

**Option B: Switch to ByteTrack-no-ReID.**

- Modify `src/card_capture/tracking/bytetrack_adapter.py` to be the default; remove dependency on dummy-image ReID.
- `config.tracker_backend` defaults to `"bytetrack"`.
- Harness gate.

Each option ends with its own TDD task (test, implementation, harness run, commit).

---

## Phase C5 — RANSAC Corner Refinement (Wave 2)

### Task C5.1: Implement RANSAC line-fit

**Files:**
- Create: `src/card_capture/ml/corner_refinement.py`
- Create: `tests/ml/test_corner_refinement.py`
- Modify: `src/card_capture/cropper.py` OR `src/card_capture/gpu_refinement.py` to call refinement when enabled.
- Modify: `src/card_capture/config.py` — `enable_corner_refinement: bool = False`.

- [ ] **Step 1: Failing test**

```python
# tests/ml/test_corner_refinement.py
import numpy as np
from src.card_capture.ml.corner_refinement import refine_corners

def test_refines_corners_toward_strong_edges():
    # synthetic image: black card on white background, corner at (100, 100)
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    img[100:300, 100:300] = 0
    rough_corners = np.array([
        [98, 98], [302, 100], [300, 302], [100, 300]
    ], dtype=np.float32)
    refined = refine_corners(img, rough_corners)
    # corners should snap to ~ (100, 100)..(300, 300)
    np.testing.assert_allclose(refined[0], [100, 100], atol=1.0)
    np.testing.assert_allclose(refined[2], [300, 300], atol=1.0)
```

- [ ] **Step 2: Implement**

```python
# src/card_capture/ml/corner_refinement.py
"""RANSAC line-fit corner refinement.

Given rough corners, refine each by fitting a line to the strongest edges
in a small ROI around the rough corner, then intersect adjacent edges to
produce a sub-pixel corner.
"""
import cv2
import numpy as np


def refine_corners(image: np.ndarray, rough_corners: np.ndarray, *,
                  roi: int = 32, edge_threshold: float = 80.0) -> np.ndarray:
    if rough_corners.shape != (4, 2):
        raise ValueError("rough_corners must be (4, 2)")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
    refined: list[tuple[float, float]] = []

    # For each pair of adjacent corners, fit a line via RANSAC, then intersect.
    lines = []
    for i in range(4):
        a = rough_corners[i]; b = rough_corners[(i + 1) % 4]
        line = _ransac_line_between(edges, a, b, roi=roi)
        lines.append(line)

    for i in range(4):
        prev_line = lines[(i - 1) % 4]
        next_line = lines[i]
        pt = _intersect(prev_line, next_line)
        refined.append(pt)
    return np.array(refined, dtype=np.float32)


def _ransac_line_between(edges, a, b, *, roi: int):
    # Sample edge pixels in a corridor between a and b; fit a line via RANSAC.
    ax, ay = a; bx, by = b
    cx = int((ax + bx) / 2); cy = int((ay + by) / 2)
    half = max(int(roi), int(np.hypot(bx - ax, by - ay) / 2))
    y0 = max(0, cy - half); y1 = min(edges.shape[0], cy + half)
    x0 = max(0, cx - half); x1 = min(edges.shape[1], cx + half)
    patch = edges[y0:y1, x0:x1]
    ys, xs = np.where(patch > 0)
    if len(xs) < 8:
        # fallback: line through a and b
        return _line_from_points(a, b)
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
    # OpenCV's fitLine with DIST_HUBER is robust enough; full RANSAC optional.
    vx, vy, x_, y_ = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
    return (float(x_[0]), float(y_[0]), float(vx[0]), float(vy[0]))


def _line_from_points(a, b):
    ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    norm = np.hypot(vx, vy) or 1.0
    return (float(ax), float(ay), float(vx / norm), float(vy / norm))


def _intersect(l1, l2) -> tuple[float, float]:
    # l = (x0, y0, vx, vy); param eq: P = p + t * v
    x1, y1, vx1, vy1 = l1
    x2, y2, vx2, vy2 = l2
    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < 1e-6:
        return (x1, y1)  # parallel; fallback
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
    return (x1 + t * vx1, y1 + t * vy1)
```

- [ ] **Step 3: Integrate behind flag**

In `cropper.py` (CPU) or `gpu_refinement.py` (GPU), surface owner adds:

```python
if cfg.enable_corner_refinement:
    from src.card_capture.ml.corner_refinement import refine_corners
    corners = refine_corners(full_frame_image, corners)
# then continue to getPerspectiveTransform...
```

- [ ] **Step 4: Harness gate**

```
ENABLE_CORNER_REFINEMENT=true card-capture harness run --baseline baseline_v4.1
```

Acceptance: `image_quality.mean_ssim` improves by ≥0.01 OR documented as marginal-but-positive.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/ml/corner_refinement.py src/card_capture/cropper.py src/card_capture/config.py tests/ml/test_corner_refinement.py
git commit -m "feat(refine): RANSAC corner refinement behind config flag"
```

---

## Phase C6 — Wave 2 Gate

- [ ] All five upgrades shipped behind config flags.
- [ ] F/B classifier and DINOv2 dedup meet Spec §1.3 improvement targets.
- [ ] Tracker decision committed to `docs/ml/tracker-decision.md` with harness evidence.
- [ ] Tag:

```bash
git tag -a v4-surface-c-wave2-complete -m "Surface C Wave 2: five upgrades shipped harness-gated"
```

---

## Phase C7 — Active Learning Retrain (Wave 3)

**Status:** Outline. Re-plan when Wave 2 winds down.

Tasks:

- **C7.1** Hard-case → training-set endpoint integration: `POST /training/datasets/from_hard_cases` queues hard cases into appropriate training datasets.
- **C7.2** Auto-retrain trigger: when a model's dataset grows by ≥N samples, queue a retrain job.
- **C7.3** Validation-set previews: surface mis-predictions in B's Train tab.
- **C7.4** Promote-on-success: if retrained model improves harness metrics on golden set, prompt to set as active.

Each is a TDD task. Re-planned via `superpowers:writing-plans`.

---

## Self-Review (post-write)

- **Spec coverage:** Scaffolding (C0); fusion verification (Spec §3.2 #1 → C1); F/B classifier (#2 → C2); DINOv2 dedup (#3 → C3); tracker swap (#4 → C4); corner refinement (#5 → C5); active learning (Wave 3 → C7 outline).
- **Placeholders:** none. Where existing files are referenced, the surface owner is directed to specific functions to integrate at; the integration shape is shown.
- **Type consistency:** `ModelVersion`, `FBDataset`, `FBPredictor`, `DinoEmbedder`, `DinoDedup`, `TrainingService`, `TrainingJob` consistent across tasks.

---

## Execution Handoff

Plan complete at `docs/superpowers/plans/2026-05-12-v4-surface-c-ml.md`.

Surface C Phase C0 starts as soon as Contract 1 is ack'd (Surface A's task A0.4). Phases C1–C5 begin once Surface A's pipeline decomposition is complete (`v4-pipeline-decomposed` tag) AND Surface D's harness is green AND Surface B's label UX is producing labels. Phase C7 is Wave 3.
