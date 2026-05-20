# GPU Provider Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract shared pipeline-execution logic into `worker_core.py`, define a `GPURunner` protocol, and add `BeamRunner` and `RunPodRunner` as switchable GPU backends alongside the existing `VastAIRunner`.

**Architecture:** A `GPURunner` Protocol provides structural typing so all runners are interchangeable without inheritance. `worker_core.py` holds the pipeline-execution functions that all three provider workers call. Each provider runner handles its own file transfer using provider-native storage (Beam Volumes, RunPod S3).

**Tech Stack:** Python asyncio, httpx (already a dep), boto3 (new), beam-sdk (new), runpod (new), pytest-asyncio

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `app/worker_core.py` | **Create** | Shared pipeline execution: apply/restore CUDA config, run pipeline subprocess, package results as bytes |
| `app/vastai_worker.py` | **Modify** | Thin wrapper: import from worker_core, keep FastAPI app + job queue |
| `app/services/gpu_runner.py` | **Create** | `GPURunner` Protocol definition |
| `app/services/beam_runner.py` | **Create** | BeamRunner: Beam Volume upload, endpoint invocation, polling, result download |
| `app/services/runpod_runner.py` | **Create** | RunPodRunner: RunPod S3 upload, job submission, polling, result download |
| `app/beam_handler.py` | **Create** | Beam endpoint handler (deployed to Beam, not run locally) |
| `app/runpod_handler.py` | **Create** | RunPod serverless handler (deployed to RunPod, not run locally) |
| `app/api/videos.py` | **Modify** | `_build_runner`: add "beam" and "runpod" backend cases |
| `app/api/config.py` | **Modify** | Add Beam + RunPod fields to `_COMPUTE_FIELDS` and `_COMPUTE_DEFAULTS` |
| `pyproject.toml` | **Modify** | Add `boto3`, `runpod`, `beam-sdk` to `[project.dependencies]` |
| `tests/app/test_worker_core.py` | **Create** | Unit tests for worker_core functions |
| `tests/app/test_beam_runner.py` | **Create** | Unit tests for BeamRunner (all HTTP mocked) |
| `tests/app/test_runpod_runner.py` | **Create** | Unit tests for RunPodRunner (all HTTP + S3 mocked) |

---

## Task 1: Add new dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add boto3, runpod, and beam-sdk to dependencies**

Open `pyproject.toml` and add to the `dependencies` list:

```toml
dependencies = [
  "numpy",
  "opencv-python",
  "Pillow>=10.4.0",
  "supervision>=0.21,<0.28",
  "torch>=2.6.0",
  "torchvision",
  "vastai>=0.5.0",
  "httpx>=0.27.0",
  "python-dotenv>=1.0.0",
  "boto3>=1.34.0",
  "runpod>=1.7.0",
  "beam-sdk>=0.8.0",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add boto3, runpod, beam-sdk dependencies"
```

---

## Task 2: Create worker_core.py and tests

**Files:**
- Create: `app/worker_core.py`
- Create: `tests/app/test_worker_core.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_worker_core.py`:

```python
"""Tests for worker_core — shared GPU pipeline execution logic."""
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.worker_core import (
    CUDA_CONFIG_OVERRIDES,
    apply_cuda_config,
    restore_config,
    package_results,
)


def test_apply_cuda_config_writes_overrides(tmp_path, monkeypatch):
    config_file = tmp_path / "card_capture_config.json"
    config_file.write_text(json.dumps({"corner_confidence": 0.5}))
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", config_file)

    original = apply_cuda_config()

    written = json.loads(config_file.read_text())
    assert written["device"] == "cuda"
    assert written["pipeline_backend"] == "cuda"
    assert written["corner_confidence"] == 0.5  # untouched key preserved
    assert original["device"] is None  # key was absent before


def test_apply_cuda_config_returns_original_values(tmp_path, monkeypatch):
    config_file = tmp_path / "card_capture_config.json"
    config_file.write_text(json.dumps({"device": "mps", "cuda_stride": 1}))
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", config_file)

    original = apply_cuda_config()

    assert original["device"] == "mps"
    assert original["cuda_stride"] == 1


def test_restore_config_restores_originals(tmp_path, monkeypatch):
    config_file = tmp_path / "card_capture_config.json"
    config_file.write_text(json.dumps({"device": "cuda", "corner_confidence": 0.5}))
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", config_file)

    restore_config({"device": "mps", "cuda_stride": None})

    written = json.loads(config_file.read_text())
    assert written["device"] == "mps"
    assert written["cuda_stride"] is None
    assert written["corner_confidence"] == 0.5  # untouched


def test_restore_config_noop_when_no_file(tmp_path, monkeypatch):
    missing = tmp_path / "nonexistent.json"
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", missing)
    restore_config({"device": "mps"})  # must not raise


def test_package_results_returns_valid_gzipped_tarball(tmp_path):
    output_dir = tmp_path / "output"
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True)
    (crops_dir / "card1.jpg").write_bytes(b"fake_image_data")
    db_path = tmp_path / "cards.sqlite"  # does not exist — cards list will be empty

    result = package_results("job123", output_dir, db_path)

    assert isinstance(result, bytes)
    assert len(result) > 0
    buf = io.BytesIO(result)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        names = tar.getnames()
    assert "export.json" in names
    assert any(n.startswith("crops") for n in names)


def test_package_results_export_json_is_empty_list_when_no_db(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    db_path = tmp_path / "cards.sqlite"  # does not exist

    result = package_results("job999", output_dir, db_path)

    buf = io.BytesIO(result)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        export_member = tar.getmember("export.json")
        f = tar.extractfile(export_member)
        cards = json.loads(f.read())
    assert cards == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/app/test_worker_core.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.worker_core'`

- [ ] **Step 3: Create app/worker_core.py**

```python
"""Shared GPU pipeline execution logic used by all provider workers."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

CUDA_CONFIG_OVERRIDES: dict = {
    "detector": "cuda",
    "device": "cuda",
    "cuda_stride": 2,
    "cuda_batch_size": 32,
    "pipeline_backend": "cuda",
}

_CONFIG_PATH = Path(__file__).parent.parent / "card_capture_config.json"


def apply_cuda_config() -> dict:
    """Write CUDA overrides to config; return original values for restore."""
    cfg: dict = {}
    if _CONFIG_PATH.exists():
        try:
            cfg = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
    original = {k: cfg.get(k) for k in CUDA_CONFIG_OVERRIDES}
    cfg.update(CUDA_CONFIG_OVERRIDES)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return original


def restore_config(original: dict) -> None:
    """Restore config values that were overridden by apply_cuda_config."""
    if not _CONFIG_PATH.exists():
        return
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        cfg.update(original)
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


def run_pipeline(job_id: str, video_path: str, config_preset: str, output_dir: Path) -> Path:
    """Run the Metaflow pipeline subprocess; return path to the output db."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "cards.sqlite"
    repo_root = Path(__file__).parent.parent
    cmd = [
        sys.executable, "-m", "pipeline.card_capture_flow",
        "--no-pylint", "run",
        "--video", video_path,
        "--output-dir", str(output_dir),
        "--db", str(db_path),
        "--config-preset", config_preset,
        "--ui-run-id", job_id,
    ]
    env = os.environ.copy()
    env.setdefault("USERNAME", "root")
    env.setdefault("USER", "root")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:] or result.stdout[-500:])
    return db_path


def package_results(job_id: str, output_dir: Path, db_path: Path) -> bytes:
    """Bundle crops + export.json into a gzipped tarball; return as bytes."""
    cards: list[dict] = []
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT track_id, session_id, fused_image_path, angle"
                    " FROM card_instances WHERE run_id=?",
                    (job_id,),
                ).fetchall()
                cards = [dict(r) for r in rows]
        except Exception:
            pass

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        crops_dir = output_dir / "crops"
        if crops_dir.exists():
            tar.add(crops_dir, arcname="crops")
        export_bytes = json.dumps(cards).encode()
        info = tarfile.TarInfo(name="export.json")
        info.size = len(export_bytes)
        tar.addfile(info, io.BytesIO(export_bytes))
    return buf.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/app/test_worker_core.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/worker_core.py tests/app/test_worker_core.py
git commit -m "feat: add worker_core.py — shared GPU pipeline execution logic"
```

---

## Task 3: Thin vastai_worker.py to use worker_core

**Files:**
- Modify: `app/vastai_worker.py`

No new tests needed — `tests/app/test_vastai_worker.py` already covers the worker endpoints and must keep passing.

- [ ] **Step 1: Replace the four functions and constant in vastai_worker.py**

Replace the block from `_CUDA_CONFIG_OVERRIDES` through `_shutdown` with imports from `worker_core` and a thinned `_run_pipeline`. The FastAPI app, endpoints, `_worker_loop`, and `_shutdown` are unchanged.

At the top of the file, add the import:

```python
from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results
```

Remove these entire blocks from `vastai_worker.py`:
- `_CUDA_CONFIG_OVERRIDES` dict
- `_CONFIG_PATH` assignment
- `_apply_cuda_config()` function
- `_restore_config()` function
- The body of `_run_pipeline()` (keep the signature)
- `_package_results()` function

Replace the body of `_run_pipeline` with:

```python
def _run_pipeline(job: dict) -> None:
    job_id = job["job_id"]
    video_path = job["video_path"]
    config_preset = job.get("config_preset", "balanced")
    output_dir = _OUTPUT_DIR / job_id

    original = apply_cuda_config()
    try:
        db_path = run_pipeline(job_id, video_path, config_preset, output_dir)
        tarball_bytes = package_results(job_id, output_dir, db_path)
    finally:
        restore_config(original)

    tarball = _OUTPUT_DIR / f"{job_id}.tar.gz"
    tarball.write_bytes(tarball_bytes)
```

- [ ] **Step 2: Run existing vastai_worker tests to verify nothing broke**

```bash
python3 -m pytest tests/app/test_vastai_worker.py -v
```

Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add app/vastai_worker.py
git commit -m "refactor: thin vastai_worker.py — delegate pipeline logic to worker_core"
```

---

## Task 4: Create GPURunner Protocol

**Files:**
- Create: `app/services/gpu_runner.py`
- Test inline in test_vast_runner.py (add one test)

- [ ] **Step 1: Create app/services/gpu_runner.py**

```python
"""GPURunner — structural protocol for switchable GPU compute backends."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GPURunner(Protocol):
    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **kw,
    ) -> None: ...

    async def run_batch_async(self, jobs: list[dict]) -> None: ...

    async def destroy_instance(self) -> None: ...
```

- [ ] **Step 2: Add protocol conformance test to test_vast_runner.py**

Append to `tests/app/test_vast_runner.py`:

```python
def test_vastai_runner_satisfies_gpu_runner_protocol(tmp_path):
    from app.services.gpu_runner import GPURunner
    runner = _make_runner(tmp_path)
    assert isinstance(runner, GPURunner)
```

- [ ] **Step 3: Run to verify**

```bash
python3 -m pytest tests/app/test_vast_runner.py::test_vastai_runner_satisfies_gpu_runner_protocol -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add app/services/gpu_runner.py tests/app/test_vast_runner.py
git commit -m "feat: add GPURunner protocol; verify VastAIRunner satisfies it"
```

---

## Task 5: Add Beam + RunPod config fields

**Files:**
- Modify: `app/api/config.py`

- [ ] **Step 1: Extend _COMPUTE_FIELDS and _COMPUTE_DEFAULTS**

In `app/api/config.py`, replace:

```python
_COMPUTE_FIELDS = {
    "pipeline_backend": str,
    "cuda_gpu_type": str,
    "vast_template_id": str,
    "cuda_idle_timeout_s": int,
}

_COMPUTE_DEFAULTS = {
    "pipeline_backend": "mps",
    "cuda_gpu_type": "RTX 4090",
    "vast_template_id": "",
    "cuda_idle_timeout_s": 600,
}
```

With:

```python
_COMPUTE_FIELDS = {
    "pipeline_backend": str,
    "cuda_gpu_type": str,
    "vast_template_id": str,
    "cuda_idle_timeout_s": int,
    # Beam
    "beam_api_key": str,
    "beam_volume_id": str,
    "beam_endpoint_id": str,
    # RunPod
    "runpod_api_key": str,
    "runpod_endpoint_id": str,
    "runpod_s3_bucket": str,
    "runpod_s3_access_key_id": str,
    "runpod_s3_secret_access_key": str,
}

_COMPUTE_DEFAULTS = {
    "pipeline_backend": "mps",
    "cuda_gpu_type": "RTX 4090",
    "vast_template_id": "",
    "cuda_idle_timeout_s": 600,
    "beam_api_key": "",
    "beam_volume_id": "",
    "beam_endpoint_id": "",
    "runpod_api_key": "",
    "runpod_endpoint_id": "",
    "runpod_s3_bucket": "",
    "runpod_s3_access_key_id": "",
    "runpod_s3_secret_access_key": "",
}
```

- [ ] **Step 2: Run config API tests to verify nothing broke**

```bash
python3 -m pytest tests/app/test_config_presets.py tests/app/test_api_contract.py -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add app/api/config.py
git commit -m "feat: add Beam and RunPod config fields to compute config"
```

---

## Task 6: Create BeamRunner and tests

**Files:**
- Create: `app/services/beam_runner.py`
- Create: `tests/app/test_beam_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_beam_runner.py`:

```python
"""Tests for BeamRunner — all HTTP calls mocked."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import pytest

from app.services.beam_runner import BeamRunner


def _make_runner(tmp_path):
    bus = MagicMock()
    bus.emit = MagicMock()
    return BeamRunner(
        bus=bus,
        db_path=tmp_path / "cards.sqlite",
        output_base=tmp_path,
        api_key="beam-test-key",
        volume_id="vol-abc123",
        endpoint_id="ep-xyz789",
    )


@pytest.mark.asyncio
async def test_run_async_emits_started_and_completed(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")

    runner._importer = MagicMock()
    runner._importer.import_tarball.return_value = 5

    runner._upload_to_volume = AsyncMock()
    runner._invoke_endpoint = AsyncMock(return_value="task-001")
    runner._poll_task = AsyncMock(return_value="complete")
    runner._download_from_volume = AsyncMock(
        side_effect=lambda key, dest: dest.write_bytes(b"fake_tarball")
    )
    runner._cleanup_volume = AsyncMock()

    db = str(tmp_path / "cards.sqlite")
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_runs "
            "(run_id TEXT PRIMARY KEY, video_id INTEGER, status TEXT, "
            "cards_extracted INTEGER, finished_at TEXT)"
        )

    await runner.run_async(
        "run-1",
        video=str(tmp_path / "video.mov"),
        output_dir=str(tmp_path / "out"),
        db=db,
        config_preset="balanced",
    )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_started" in emitted_names
    assert "run_completed" in emitted_names
    runner._upload_to_volume.assert_called_once()
    runner._invoke_endpoint.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_emits_run_failed_on_error(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")

    runner._upload_to_volume = AsyncMock(side_effect=RuntimeError("upload failed"))
    runner._cleanup_volume = AsyncMock()

    db = str(tmp_path / "cards.sqlite")
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_runs "
            "(run_id TEXT PRIMARY KEY, video_id INTEGER, status TEXT, "
            "cards_extracted INTEGER, finished_at TEXT)"
        )

    with pytest.raises(RuntimeError, match="upload failed"):
        await runner.run_async(
            "run-fail",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_failed" in emitted_names


@pytest.mark.asyncio
async def test_run_async_raises_on_beam_task_failure(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")

    runner._upload_to_volume = AsyncMock()
    runner._invoke_endpoint = AsyncMock(return_value="task-002")
    runner._poll_task = AsyncMock(return_value="failed")
    runner._cleanup_volume = AsyncMock()

    db = str(tmp_path / "cards.sqlite")
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_runs "
            "(run_id TEXT PRIMARY KEY, video_id INTEGER, status TEXT, "
            "cards_extracted INTEGER, finished_at TEXT)"
        )

    with pytest.raises(RuntimeError, match="task-002"):
        await runner.run_async(
            "run-fail2",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )


def test_destroy_instance_is_noop(tmp_path):
    runner = _make_runner(tmp_path)
    asyncio.get_event_loop().run_until_complete(runner.destroy_instance())
    # No exception = pass


def test_satisfies_gpu_runner_protocol(tmp_path):
    from app.services.gpu_runner import GPURunner
    runner = _make_runner(tmp_path)
    assert isinstance(runner, GPURunner)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/app/test_beam_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.services.beam_runner'`

- [ ] **Step 3: Create app/services/beam_runner.py**

```python
"""BeamRunner — orchestrates Beam endpoint invocation for GPU pipeline runs."""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Optional

import httpx

from app.services.event_bus import Event, EventBus
from app.services.result_importer import ResultImporter
from app.services import _event_bus_registry

_BEAM_BASE = "https://app.beam.cloud"


class BeamRunner:
    def __init__(
        self,
        bus: EventBus,
        db_path: Path,
        output_base: Path,
        api_key: str,
        volume_id: str,
        endpoint_id: str,
    ) -> None:
        self.bus = bus
        self._api_key = api_key
        self._volume_id = volume_id
        self._endpoint_id = endpoint_id
        self._importer = ResultImporter(db_path=db_path, output_base=output_base)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **kw,
    ) -> None:
        video_id: Optional[int] = kw.get("video_id")
        _event_bus_registry.set(run_id, self.bus)

        video_key = f"runs/{run_id}/input.mov"
        results_key = f"runs/{run_id}/results.tar.gz"

        try:
            self._record_run_start(run_id, video_id, db)
            self.bus.emit(run_id, Event(name="run_started"))

            self.bus.emit(run_id, Event(name="log", payload={"line": "Uploading video to Beam…"}))
            print(f"[{run_id}] beam: uploading video…", flush=True)
            await self._upload_to_volume(video_key, Path(video))

            self.bus.emit(run_id, Event(name="log", payload={"line": "Invoking Beam endpoint…"}))
            task_id = await self._invoke_endpoint(run_id, video_key, results_key, config_preset)
            print(f"[{run_id}] beam: task {task_id} submitted", flush=True)

            poll_count = 0
            while True:
                status = await self._poll_task(task_id)
                if status == "complete":
                    break
                if status in ("failed", "error", "cancelled"):
                    raise RuntimeError(f"Beam task {task_id} ended with status: {status}")
                if poll_count % 10 == 0:
                    msg = "Processing on Beam GPU…"
                    self.bus.emit(run_id, Event(name="log", payload={"line": msg}))
                    print(f"[{run_id}] beam: {msg}", flush=True)
                poll_count += 1
                await asyncio.sleep(3)

            self.bus.emit(run_id, Event(name="log", payload={"line": "Downloading results…"}))
            tarball = Path(output_dir) / f"{run_id}_results.tar.gz"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            await self._download_from_volume(results_key, tarball)
            n_cards = self._importer.import_tarball(tarball, run_id)
            tarball.unlink(missing_ok=True)

            self._record_run_finish(run_id, n_cards, db)
            print(f"[{run_id}] beam: done — {n_cards} cards imported", flush=True)
            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            print(f"[{run_id}] beam: FAILED — {exc}", flush=True)
            self._record_run_fail(run_id, db)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            _event_bus_registry.clear(run_id)
            await self._cleanup_volume(run_id)

    async def run_batch_async(self, jobs: list[dict]) -> None:
        for job in jobs:
            await self.run_async(**job)

    async def destroy_instance(self) -> None:
        pass  # Beam manages container lifecycle

    async def _upload_to_volume(self, key: str, path: Path) -> None:
        url = f"{_BEAM_BASE}/volume/files/{self._volume_id}/{key}"
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.put(
                url,
                content=path.read_bytes(),
                headers={**self._headers, "Content-Type": "application/octet-stream"},
            )
            r.raise_for_status()

    async def _download_from_volume(self, key: str, dest: Path) -> None:
        url = f"{_BEAM_BASE}/volume/files/{self._volume_id}/{key}"
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            dest.write_bytes(r.content)

    async def _invoke_endpoint(
        self, run_id: str, video_key: str, results_key: str, config_preset: str
    ) -> str:
        url = f"{_BEAM_BASE}/endpoint/{self._endpoint_id}/"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                url,
                json={
                    "run_id": run_id,
                    "video_volume_path": f"/volumes/{self._volume_id}/{video_key}",
                    "results_volume_path": f"/volumes/{self._volume_id}/{results_key}",
                    "config_preset": config_preset,
                },
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()["task_id"]

    async def _poll_task(self, task_id: str) -> str:
        url = f"{_BEAM_BASE}/task/{task_id}/status/"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            return r.json().get("status", "unknown")

    async def _cleanup_volume(self, run_id: str) -> None:
        for key in [f"runs/{run_id}/input.mov", f"runs/{run_id}/results.tar.gz"]:
            try:
                url = f"{_BEAM_BASE}/volume/files/{self._volume_id}/{key}"
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.delete(url, headers=self._headers)
            except Exception:
                pass

    def _record_run_start(self, run_id: str, video_id: Optional[int], db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status)"
                    " VALUES (?, ?, 'running')",
                    (run_id, video_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, n_cards: int, db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='completed', cards_extracted=?,"
                    " finished_at=datetime('now') WHERE run_id=?",
                    (n_cards, run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _record_run_fail(self, run_id: str, db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='failed',"
                    " finished_at=datetime('now') WHERE run_id=?",
                    (run_id,),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run failure: {exc}", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/app/test_beam_runner.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/beam_runner.py tests/app/test_beam_runner.py
git commit -m "feat: add BeamRunner for Beam serverless GPU backend"
```

---

## Task 7: Create beam_handler.py

**Files:**
- Create: `app/beam_handler.py`

No unit tests — this file is a deployment artifact; its logic is covered by `test_worker_core.py`.

- [ ] **Step 1: Create app/beam_handler.py**

```python
"""Beam endpoint handler — deployed to Beam, not run locally.

Deploy with:
    beam deploy app/beam_handler.py:process_video

The Volume ID and GPU type are configured at deploy time.
"""
from __future__ import annotations

from pathlib import Path

import beam

from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results


@beam.endpoint(
    cpu=4,
    memory="16Gi",
    gpu="A10G",
)
def process_video(
    run_id: str,
    video_volume_path: str,
    results_volume_path: str,
    config_preset: str = "balanced",
) -> dict:
    """Run the card-capture pipeline on a video file stored in a Beam Volume."""
    output_dir = Path(f"/tmp/cc_output/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = apply_cuda_config()
    try:
        db_path = run_pipeline(run_id, video_volume_path, config_preset, output_dir)
        tarball_bytes = package_results(run_id, output_dir, db_path)
    finally:
        restore_config(original)

    results_path = Path(results_volume_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_bytes(tarball_bytes)
    return {"status": "complete", "results_path": results_volume_path}
```

- [ ] **Step 2: Commit**

```bash
git add app/beam_handler.py
git commit -m "feat: add Beam endpoint handler for deployment"
```

---

## Task 8: Create RunPodRunner and tests

**Files:**
- Create: `app/services/runpod_runner.py`
- Create: `tests/app/test_runpod_runner.py`

> **Note:** Before implementing, confirm the RunPod S3 endpoint URL from https://docs.runpod.io/storage/s3-api and update `_RUNPOD_S3_ENDPOINT` in the file if it differs from `https://storage.runpod.io`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_runpod_runner.py`:

```python
"""Tests for RunPodRunner — all HTTP and S3 calls mocked."""
import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.runpod_runner import RunPodRunner


def _make_runner(tmp_path):
    bus = MagicMock()
    bus.emit = MagicMock()
    return RunPodRunner(
        bus=bus,
        db_path=tmp_path / "cards.sqlite",
        output_base=tmp_path,
        api_key="rp-test-key",
        endpoint_id="ep-runpod-001",
        s3_bucket="cc-runpod-bucket",
        s3_access_key_id="AKIATEST",
        s3_secret_access_key="secret",
    )


def _make_db(tmp_path):
    db = str(tmp_path / "cards.sqlite")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_runs "
            "(run_id TEXT PRIMARY KEY, video_id INTEGER, status TEXT, "
            "cards_extracted INTEGER, finished_at TEXT)"
        )
    return db


@pytest.mark.asyncio
async def test_run_async_emits_started_and_completed(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._importer = MagicMock()
    runner._importer.import_tarball.return_value = 3
    runner._upload_video = MagicMock()
    runner._download_results = MagicMock(
        side_effect=lambda key, dest: dest.write_bytes(b"fake_tarball")
    )
    runner._submit_job = AsyncMock(return_value="rp-job-001")
    runner._poll_job = AsyncMock(return_value="COMPLETED")
    runner._cleanup_s3 = AsyncMock()

    await runner.run_async(
        "run-rp-1",
        video=str(tmp_path / "video.mov"),
        output_dir=str(tmp_path / "out"),
        db=db,
        config_preset="balanced",
    )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_started" in emitted_names
    assert "run_completed" in emitted_names
    runner._submit_job.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_emits_run_failed_on_upload_error(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._upload_video = MagicMock(side_effect=RuntimeError("S3 upload failed"))
    runner._cleanup_s3 = AsyncMock()

    with pytest.raises(RuntimeError, match="S3 upload failed"):
        await runner.run_async(
            "run-rp-fail",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_failed" in emitted_names


@pytest.mark.asyncio
async def test_run_async_raises_on_runpod_job_failure(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._upload_video = MagicMock()
    runner._submit_job = AsyncMock(return_value="rp-job-002")
    runner._poll_job = AsyncMock(return_value="FAILED")
    runner._cleanup_s3 = AsyncMock()

    with pytest.raises(RuntimeError, match="rp-job-002"):
        await runner.run_async(
            "run-rp-fail2",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )


def test_destroy_instance_is_noop(tmp_path):
    runner = _make_runner(tmp_path)
    asyncio.get_event_loop().run_until_complete(runner.destroy_instance())


def test_satisfies_gpu_runner_protocol(tmp_path):
    from app.services.gpu_runner import GPURunner
    runner = _make_runner(tmp_path)
    assert isinstance(runner, GPURunner)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/app/test_runpod_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.services.runpod_runner'`

- [ ] **Step 3: Create app/services/runpod_runner.py**

> **Before writing:** confirm the RunPod S3 endpoint URL at https://docs.runpod.io/storage/s3-api and set `_RUNPOD_S3_ENDPOINT` accordingly.

```python
"""RunPodRunner — orchestrates RunPod serverless endpoint for GPU pipeline runs."""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Optional

import boto3
import httpx

from app.services.event_bus import Event, EventBus
from app.services.result_importer import ResultImporter
from app.services import _event_bus_registry

_RUNPOD_API = "https://api.runpod.io/v2"
# Confirm this URL from https://docs.runpod.io/storage/s3-api before deploying
_RUNPOD_S3_ENDPOINT = "https://storage.runpod.io"


class RunPodRunner:
    def __init__(
        self,
        bus: EventBus,
        db_path: Path,
        output_base: Path,
        api_key: str,
        endpoint_id: str,
        s3_bucket: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
    ) -> None:
        self.bus = bus
        self._api_key = api_key
        self._endpoint_id = endpoint_id
        self._s3_bucket = s3_bucket
        self._s3 = boto3.client(
            "s3",
            endpoint_url=_RUNPOD_S3_ENDPOINT,
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_secret_access_key,
        )
        self._importer = ResultImporter(db_path=db_path, output_base=output_base)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **kw,
    ) -> None:
        video_id: Optional[int] = kw.get("video_id")
        _event_bus_registry.set(run_id, self.bus)

        video_key = f"runs/{run_id}/input.mov"
        results_key = f"runs/{run_id}/results.tar.gz"

        try:
            self._record_run_start(run_id, video_id, db)
            self.bus.emit(run_id, Event(name="run_started"))

            self.bus.emit(run_id, Event(name="log", payload={"line": "Uploading video to RunPod storage…"}))
            print(f"[{run_id}] runpod: uploading video…", flush=True)
            await asyncio.get_event_loop().run_in_executor(
                None, self._upload_video, video_key, Path(video)
            )

            self.bus.emit(run_id, Event(name="log", payload={"line": "Submitting RunPod job…"}))
            job_id = await self._submit_job(run_id, video_key, results_key, config_preset)
            print(f"[{run_id}] runpod: job {job_id} submitted", flush=True)

            poll_count = 0
            while True:
                status = await self._poll_job(job_id)
                if status == "COMPLETED":
                    break
                if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    raise RuntimeError(f"RunPod job {job_id} ended with status: {status}")
                if poll_count % 10 == 0:
                    msg = "Processing on RunPod GPU…"
                    self.bus.emit(run_id, Event(name="log", payload={"line": msg}))
                    print(f"[{run_id}] runpod: {msg}", flush=True)
                poll_count += 1
                await asyncio.sleep(3)

            self.bus.emit(run_id, Event(name="log", payload={"line": "Downloading results…"}))
            tarball = Path(output_dir) / f"{run_id}_results.tar.gz"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            await asyncio.get_event_loop().run_in_executor(
                None, self._download_results, results_key, tarball
            )
            n_cards = self._importer.import_tarball(tarball, run_id)
            tarball.unlink(missing_ok=True)

            self._record_run_finish(run_id, n_cards, db)
            print(f"[{run_id}] runpod: done — {n_cards} cards imported", flush=True)
            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            print(f"[{run_id}] runpod: FAILED — {exc}", flush=True)
            self._record_run_fail(run_id, db)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            _event_bus_registry.clear(run_id)
            await self._cleanup_s3(run_id)

    async def run_batch_async(self, jobs: list[dict]) -> None:
        for job in jobs:
            await self.run_async(**job)

    async def destroy_instance(self) -> None:
        pass  # RunPod manages container lifecycle

    def _upload_video(self, key: str, path: Path) -> None:
        self._s3.upload_file(str(path), self._s3_bucket, key)

    def _download_results(self, key: str, dest: Path) -> None:
        self._s3.download_file(self._s3_bucket, key, str(dest))

    async def _submit_job(
        self, run_id: str, video_key: str, results_key: str, config_preset: str
    ) -> str:
        url = f"{_RUNPOD_API}/{self._endpoint_id}/run"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                url,
                json={
                    "input": {
                        "run_id": run_id,
                        "video_s3_key": video_key,
                        "results_s3_key": results_key,
                        "bucket": self._s3_bucket,
                        "config_preset": config_preset,
                    }
                },
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()["id"]

    async def _poll_job(self, job_id: str) -> str:
        url = f"{_RUNPOD_API}/{self._endpoint_id}/status/{job_id}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            return r.json().get("status", "UNKNOWN")

    async def _cleanup_s3(self, run_id: str) -> None:
        loop = asyncio.get_event_loop()
        for key in [f"runs/{run_id}/input.mov", f"runs/{run_id}/results.tar.gz"]:
            try:
                await loop.run_in_executor(
                    None,
                    lambda k=key: self._s3.delete_object(Bucket=self._s3_bucket, Key=k),
                )
            except Exception:
                pass

    def _record_run_start(self, run_id: str, video_id: Optional[int], db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status)"
                    " VALUES (?, ?, 'running')",
                    (run_id, video_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, n_cards: int, db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='completed', cards_extracted=?,"
                    " finished_at=datetime('now') WHERE run_id=?",
                    (n_cards, run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _record_run_fail(self, run_id: str, db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='failed',"
                    " finished_at=datetime('now') WHERE run_id=?",
                    (run_id,),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run failure: {exc}", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/app/test_runpod_runner.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/runpod_runner.py tests/app/test_runpod_runner.py
git commit -m "feat: add RunPodRunner for RunPod serverless GPU backend"
```

---

## Task 9: Create runpod_handler.py

**Files:**
- Create: `app/runpod_handler.py`

No unit tests — deployment artifact; logic covered by `test_worker_core.py`.

- [ ] **Step 1: Create app/runpod_handler.py**

> **Before writing:** confirm the RunPod S3 endpoint URL and update `_RUNPOD_S3_ENDPOINT` if needed.

```python
"""RunPod serverless handler — deployed to RunPod, not run locally.

Deploy by building a Docker image with this file as the entrypoint:
    CMD ["python", "-m", "app.runpod_handler"]

RunPod injects AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as env vars
for the built-in S3 storage.
"""
from __future__ import annotations

from pathlib import Path

import boto3
import runpod

from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results

# Confirm from https://docs.runpod.io/storage/s3-api before deploying
_RUNPOD_S3_ENDPOINT = "https://storage.runpod.io"


def handler(job: dict) -> dict:
    """RunPod calls this for each submitted job."""
    inp = job["input"]
    run_id = inp["run_id"]
    video_key = inp["video_s3_key"]
    results_key = inp["results_s3_key"]
    bucket = inp["bucket"]
    config_preset = inp.get("config_preset", "balanced")

    # Credentials are injected by RunPod as standard AWS env vars
    s3 = boto3.client("s3", endpoint_url=_RUNPOD_S3_ENDPOINT)

    video_path = Path(f"/tmp/{run_id}_input.mov")
    s3.download_file(bucket, video_key, str(video_path))

    output_dir = Path(f"/tmp/cc_output/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = apply_cuda_config()
    try:
        db_path = run_pipeline(run_id, str(video_path), config_preset, output_dir)
        tarball_bytes = package_results(run_id, output_dir, db_path)
    finally:
        restore_config(original)

    s3.put_object(Bucket=bucket, Key=results_key, Body=tarball_bytes)
    return {"status": "complete", "results_key": results_key}


runpod.serverless.start({"handler": handler})
```

- [ ] **Step 2: Commit**

```bash
git add app/runpod_handler.py
git commit -m "feat: add RunPod serverless handler for deployment"
```

---

## Task 10: Wire runner selection in videos.py

**Files:**
- Modify: `app/api/videos.py`

- [ ] **Step 1: Extend _build_runner to handle beam and runpod backends**

Replace the `_build_runner` function in `app/api/videos.py`:

```python
def _build_runner(request: Request):
    """Return the appropriate GPU runner based on pipeline_backend config."""
    import json
    import os
    from pathlib import Path as _Path

    _cfg = _Path(__file__).parent.parent.parent / "card_capture_config.json"
    cfg: dict = {}
    if _cfg.exists():
        try:
            cfg = json.loads(_cfg.read_text())
        except Exception:
            pass

    bus = request.app.state.event_bus
    db_path = request.app.state.db_path
    backend = cfg.get("pipeline_backend", "mps")

    if backend == "cuda":
        api_key = os.environ.get("VAST_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="VAST_API_KEY environment variable is not set")
        from app.services.vast_runner import VastAIRunner
        return VastAIRunner(
            bus=bus,
            db_path=db_path,
            output_base=db_path.parent,
            api_key=api_key,
            gpu_type=cfg.get("cuda_gpu_type", "RTX 4090"),
            template_id=cfg.get("vast_template_id", ""),
            idle_timeout_s=int(cfg.get("cuda_idle_timeout_s", 600)),
        )

    if backend == "beam":
        api_key = cfg.get("beam_api_key") or os.environ.get("BEAM_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="beam_api_key config or BEAM_API_KEY env var is not set")
        volume_id = cfg.get("beam_volume_id", "")
        endpoint_id = cfg.get("beam_endpoint_id", "")
        if not volume_id or not endpoint_id:
            raise HTTPException(status_code=500, detail="beam_volume_id and beam_endpoint_id must be configured")
        from app.services.beam_runner import BeamRunner
        return BeamRunner(
            bus=bus,
            db_path=db_path,
            output_base=db_path.parent,
            api_key=api_key,
            volume_id=volume_id,
            endpoint_id=endpoint_id,
        )

    if backend == "runpod":
        api_key = cfg.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="runpod_api_key config or RUNPOD_API_KEY env var is not set")
        endpoint_id = cfg.get("runpod_endpoint_id", "")
        s3_bucket = cfg.get("runpod_s3_bucket", "")
        s3_access_key_id = cfg.get("runpod_s3_access_key_id", "")
        s3_secret_access_key = cfg.get("runpod_s3_secret_access_key", "")
        if not endpoint_id or not s3_bucket:
            raise HTTPException(
                status_code=500,
                detail="runpod_endpoint_id and runpod_s3_bucket must be configured",
            )
        from app.services.runpod_runner import RunPodRunner
        return RunPodRunner(
            bus=bus,
            db_path=db_path,
            output_base=db_path.parent,
            api_key=api_key,
            endpoint_id=endpoint_id,
            s3_bucket=s3_bucket,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
        )

    return PipelineRunner(bus=bus, flow_cls=None, db_path=db_path)
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

```bash
python3 -m pytest tests/app/ -q --ignore=tests/app/test_beam_runner.py --ignore=tests/app/test_runpod_runner.py --ignore=tests/app/test_worker_core.py
```

Expected: same pass/fail counts as before this task

- [ ] **Step 3: Run the full new test suite**

```bash
python3 -m pytest tests/app/test_worker_core.py tests/app/test_beam_runner.py tests/app/test_runpod_runner.py -v
```

Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add app/api/videos.py
git commit -m "feat: wire Beam and RunPod backend selection in _build_runner"
```

---

## Task 11: Full test suite verification

- [ ] **Step 1: Run all tests (excluding known pre-existing failures)**

```bash
python3 -m pytest tests/ -q \
  --ignore=tests/pipeline/test_path_equivalence.py
```

Expected: no new failures beyond the pre-existing ones documented in CLAUDE.md (`test_migrations_are_idempotent`, several in `test_wave1/2_robustness.py`).

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "chore: GPU provider abstraction complete — Beam + RunPod + worker_core"
```
