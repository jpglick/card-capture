# Vast.ai Integration Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vast.ai GPU cloud backend to card-capture so a single toggle in Settings switches from local MPS processing to a remote RTX 4090 instance, with batch mode and automatic instance spin-down.

**Architecture:** A new `VastAIRunner` service mirrors the `PipelineRunner` interface and is selected per-request by reading `pipeline_backend` from `card_capture_config.json`. The runner provisions a vast.ai instance via the vastai CLI, submits jobs to an instance-side FastAPI worker (`vastai_worker.py`), downloads a results tarball, imports cards into local SQLite, and destroys the instance. Batch mode reuses one instance across multiple videos.

**Tech Stack:** Python (vastai CLI via subprocess, httpx, tarfile), FastAPI (instance-side worker), Svelte (settings + batch UI), SQLite.

**Spec:** `docs/superpowers/specs/2026-05-17-vastai-integration-design.md`

---

## File Map

**New — Python services (Mac side):**
- `app/services/vast_client.py` — thin vastai CLI wrapper
- `app/services/worker_client.py` — HTTP client for instance worker API
- `app/services/result_importer.py` — tarball unpack + SQLite import
- `app/services/vast_runner.py` — orchestration (provision → run → destroy)

**New — API:**
- `app/api/batch.py` — POST /batch, GET /batch/{id}

**New — Instance side:**
- `app/vastai_worker.py` — FastAPI worker that runs on the GPU instance

**New — Frontend:**
- `app/web/src/routes/batch/+page.svelte` — batch UI

**Modified:**
- `app/api/videos.py` — `_build_runner()` selector
- `app/api/config.py` — expose vast.ai config fields
- `app/main.py` — wire batch router
- `app/web/src/lib/api/client.ts` — batch + compute config API calls
- `app/web/src/lib/api/types.ts` — BatchJob, VastConfig types
- `app/web/src/routes/settings/+page.svelte` — Compute section
- `card_capture_config.json` — new fields

**New — Tests:**
- `tests/app/test_vast_client.py`
- `tests/app/test_worker_client.py`
- `tests/app/test_result_importer.py`
- `tests/app/test_vast_runner.py`
- `tests/app/test_vastai_worker.py`
- `tests/app/test_batch_api.py`

---

### Task 1: Install vastai + add config fields

**Files:**
- Modify: `pyproject.toml` (add vastai dependency)
- Modify: `card_capture_config.json`
- Modify: `app/api/config.py`

- [ ] **Step 1: Add vastai to dependencies**

In `pyproject.toml`, add to `[project.dependencies]`:
```
"vastai>=5.0.0",
"httpx>=0.27.0",
```

Run: `pip install vastai httpx`

- [ ] **Step 2: Add new fields to card_capture_config.json**

Read the file first. Add the new fields:
```json
{
  "pipeline_backend": "mps",
  "cuda_gpu_type": "RTX 4090",
  "vast_template_id": "",
  "cuda_idle_timeout_s": 300,
  "active_vast_instance": null
}
```
(Merge with any existing keys already in the file.)

- [ ] **Step 3: Expose compute config fields in the config API**

In `app/api/config.py`, add a new dict alongside `_PIPELINE_FIELDS`:

```python
_COMPUTE_FIELDS = {
    "pipeline_backend": str,
    "cuda_gpu_type": str,
    "vast_template_id": str,
    "cuda_idle_timeout_s": int,
}
```

Add two new routes at the bottom of the file:

```python
@router.get("/compute")
def get_compute_config(_request: Request):
    """Return the current compute backend config."""
    try:
        data = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: data.get(k, _COMPUTE_DEFAULTS[k]) for k in _COMPUTE_FIELDS}

@router.patch("/compute")
def patch_compute_config(body: dict, _request: Request):
    """Update compute backend fields in card_capture_config.json."""
    unknown = set(body) - set(_COMPUTE_FIELDS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown fields: {sorted(unknown)}")
    try:
        data = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
        for key, cast in _COMPUTE_FIELDS.items():
            if key in body:
                data[key] = cast(body[key])
        _CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: data.get(k) for k in _COMPUTE_FIELDS}
```

Also add this constant before the routes:

```python
_COMPUTE_DEFAULTS = {
    "pipeline_backend": "mps",
    "cuda_gpu_type": "RTX 4090",
    "vast_template_id": "",
    "cuda_idle_timeout_s": 300,
}
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml card_capture_config.json app/api/config.py
git commit -m "feat(vastai): add compute config fields and API endpoints"
```

---

### Task 2: VastAIClient — vastai CLI wrapper

**Files:**
- Create: `app/services/vast_client.py`
- Create: `tests/app/test_vast_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/app/test_vast_client.py`:

```python
"""Tests for VastAIClient — mocks subprocess so no real API calls needed."""
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.services.vast_client import VastAIClient, GPU_TYPE_QUERIES


def _mock_run(stdout_data):
    """Return a mock CompletedProcess with JSON stdout."""
    m = MagicMock()
    m.stdout = json.dumps(stdout_data)
    return m


def test_search_offers_returns_list():
    client = VastAIClient(api_key="test-key")
    offers = [{"id": 1, "dph_total": 0.5}, {"id": 2, "dph_total": 0.3}]
    with patch("subprocess.run", return_value=_mock_run(offers)) as mock_run:
        result = client.search_offers("RTX 4090")
    assert result == offers
    call_args = mock_run.call_args[0][0]
    assert "RTX 4090" in " ".join(call_args)


def test_provision_returns_instance_dict():
    client = VastAIClient(api_key="test-key")
    instance = {"id": 42, "status": "created"}
    with patch("subprocess.run", return_value=_mock_run(instance)):
        result = client.provision(offer_id=1, template_id="pytorch/pytorch:latest")
    assert result["id"] == 42


def test_destroy_calls_vastai():
    client = VastAIClient(api_key="test-key")
    with patch("subprocess.run", return_value=_mock_run({"success": True})) as mock_run:
        client.destroy(instance_id=42)
    call_args = mock_run.call_args[0][0]
    assert "destroy" in call_args
    assert "42" in call_args


def test_get_instance_ip_found():
    client = VastAIClient(api_key="test-key")
    instances = [{"id": 42, "public_ipaddr": "1.2.3.4"}, {"id": 99, "public_ipaddr": "5.6.7.8"}]
    with patch("subprocess.run", return_value=_mock_run(instances)):
        ip = client.get_instance_ip(42)
    assert ip == "1.2.3.4"


def test_get_instance_ip_not_found():
    client = VastAIClient(api_key="test-key")
    with patch("subprocess.run", return_value=_mock_run([])):
        ip = client.get_instance_ip(999)
    assert ip is None


def test_gpu_type_queries_has_all_options():
    for key in ["RTX 4090", "Flagship", "RTX 5060 Ti"]:
        assert key in GPU_TYPE_QUERIES
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/app/test_vast_client.py -v 2>&1 | tail -10
```
Expected: ImportError — `cannot import name 'VastAIClient'`

- [ ] **Step 3: Implement VastAIClient**

Create `app/services/vast_client.py`:

```python
"""Thin wrapper around the vastai CLI for instance provisioning."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

GPU_TYPE_QUERIES: dict[str, str] = {
    "RTX 4090": "gpu_name=RTX_4090 num_gpus=1 reliability>0.95",
    "Flagship": "num_gpus=1 reliability>0.99",   # sorted by TFLOPS at provision time
    "RTX 5060 Ti": "gpu_name=RTX_5060_Ti num_gpus=1 reliability>0.95",
}

_BOOT_SCRIPT = (
    "cd /workspace/card-capture && "
    "git pull origin {branch} -q && "
    "pip install -e '.[app]' -q && "
    "nohup uvicorn app.vastai_worker:app --host 0.0.0.0 --port 8765 &"
)


class VastAIClient:
    """Wraps the vastai CLI. All calls require VAST_API_KEY in the environment."""

    def __init__(self, api_key: str) -> None:
        self._env = {**os.environ, "VAST_API_KEY": api_key}

    def _run(self, *args: str) -> object:
        result = subprocess.run(
            ["vastai", *args, "--raw"],
            capture_output=True, text=True, env=self._env, check=True,
        )
        return json.loads(result.stdout)

    def search_offers(self, gpu_type: str) -> list[dict]:
        """Return available offers matching the GPU type, cheapest first."""
        query = GPU_TYPE_QUERIES.get(gpu_type, gpu_type)
        offers = self._run("search", "offers", query)
        if isinstance(offers, list):
            offers.sort(key=lambda o: o.get("dph_total", 999))
        return offers if isinstance(offers, list) else []

    def provision(
        self,
        offer_id: int,
        template_id: str,
        branch: str = "main",
    ) -> dict:
        """Launch an instance. Returns the instance dict with at least {"id": int}."""
        script = _BOOT_SCRIPT.format(branch=branch)
        result = self._run(
            "create", "instance", str(offer_id),
            "--image", template_id,
            "--onstart", script,
            "--ports", "8765",
        )
        return result if isinstance(result, dict) else {"id": result}

    def destroy(self, instance_id: int) -> None:
        """Destroy a running instance. Billing stops immediately."""
        self._run("destroy", "instance", str(instance_id))

    def get_instance_ip(self, instance_id: int) -> Optional[str]:
        """Return the public IP of a running instance, or None if not yet assigned."""
        try:
            instances = self._run("show", "instances")
        except subprocess.CalledProcessError:
            return None
        if not isinstance(instances, list):
            return None
        for inst in instances:
            if inst.get("id") == instance_id:
                return inst.get("public_ipaddr") or None
        return None
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/app/test_vast_client.py -v 2>&1 | tail -10
```
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/vast_client.py tests/app/test_vast_client.py
git commit -m "feat(vastai): VastAIClient — vastai CLI wrapper"
```

---

### Task 3: InstanceWorkerClient — HTTP client for instance API

**Files:**
- Create: `app/services/worker_client.py`
- Create: `tests/app/test_worker_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/app/test_worker_client.py`:

```python
"""Tests for InstanceWorkerClient — uses httpx mock transport."""
import json
from pathlib import Path
import httpx
import pytest

from app.services.worker_client import InstanceWorkerClient


def _make_transport(responses: dict):
    """Build a mock httpx transport from {path: (status, body)} dict."""
    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            key = f"{request.method} {request.url.path}"
            status, body = responses.get(key, (404, b"not found"))
            return httpx.Response(status, content=body)
    return MockTransport()


@pytest.mark.asyncio
async def test_health_check_returns_true():
    t = _make_transport({"GET /health": (200, b'{"status":"ok"}')})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    assert await client.health_check() is True
    await client.close()


@pytest.mark.asyncio
async def test_health_check_returns_false_on_error():
    t = _make_transport({})  # no routes → 404
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    assert await client.health_check() is False
    await client.close()


@pytest.mark.asyncio
async def test_submit_job():
    t = _make_transport({"POST /jobs": (200, b'{"job_id":"run-1"}')})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    await client.submit_job("run-1", "/tmp/video.mp4", {"config_preset": "balanced"})
    await client.close()


@pytest.mark.asyncio
async def test_poll_status_returns_dict():
    body = json.dumps({"status": "running", "progress_pct": 50}).encode()
    t = _make_transport({"GET /jobs/run-1": (200, body)})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    status = await client.poll_status("run-1")
    assert status["status"] == "running"
    await client.close()


@pytest.mark.asyncio
async def test_confirm_downloaded():
    t = _make_transport({"DELETE /jobs/run-1": (200, b'{"deleted":"run-1"}')})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    await client.confirm_downloaded("run-1")
    await client.close()
```

- [ ] **Step 2: Verify they fail**

```bash
python3 -m pytest tests/app/test_worker_client.py -v 2>&1 | tail -8
```
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `app/services/worker_client.py`:

```python
"""HTTP client for the instance-side vastai_worker FastAPI app."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx


class InstanceWorkerClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    async def health_check(self) -> bool:
        try:
            r = await self._client.get("/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def upload_video(self, video_path: Path) -> str:
        """Upload the video file; returns the server-side path string."""
        with open(video_path, "rb") as f:
            r = await self._client.post(
                "/upload",
                files={"file": (video_path.name, f, "video/mp4")},
                timeout=600.0,
            )
        r.raise_for_status()
        return r.json()["path"]

    async def submit_job(self, job_id: str, server_video_path: str, params: dict) -> None:
        r = await self._client.post(
            "/jobs",
            json={"job_id": job_id, "video_path": server_video_path, **params},
        )
        r.raise_for_status()

    async def poll_status(self, job_id: str) -> dict:
        r = await self._client.get(f"/jobs/{job_id}")
        r.raise_for_status()
        return r.json()

    async def download_results(self, job_id: str, dest: Path) -> None:
        async with self._client.stream("GET", f"/jobs/{job_id}/results", timeout=300.0) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)

    async def confirm_downloaded(self, job_id: str) -> None:
        r = await self._client.delete(f"/jobs/{job_id}")
        r.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/app/test_worker_client.py -v 2>&1 | tail -8
```
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/worker_client.py tests/app/test_worker_client.py
git commit -m "feat(vastai): InstanceWorkerClient — HTTP client for instance API"
```

---

### Task 4: ResultImporter — tarball unpack and SQLite import

**Files:**
- Create: `app/services/result_importer.py`
- Create: `tests/app/test_result_importer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/app/test_result_importer.py`:

```python
"""Tests for ResultImporter — uses synthetic tarballs."""
import io
import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import pytest

from app.services.result_importer import ResultImporter


def _make_tarball(cards: list[dict], crop_filenames: list[str]) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    with tarfile.open(tmp.name, "w:gz") as tar:
        # Add fake crop files
        for fname in crop_filenames:
            data = b"JPEG"
            info = tarfile.TarInfo(name=f"crops/{fname}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        # Add export.json
        export_data = json.dumps(cards).encode()
        info = tarfile.TarInfo(name="export.json")
        info.size = len(export_data)
        tar.addfile(info, io.BytesIO(export_data))
    return Path(tmp.name)


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "cards.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE card_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, track_id TEXT, session_id INTEGER,
            fused_image_path TEXT, angle TEXT,
            hidden INTEGER DEFAULT 0
        )""")
    return db


def test_import_cards_are_inserted(tmp_path):
    cards = [
        {"track_id": "abc", "session_id": 0, "fused_image_path": "crops/instance_abc_fused.jpg", "side": "Front"},
        {"track_id": "def", "session_id": 1, "fused_image_path": "crops/instance_def_fused.jpg", "side": "Back"},
    ]
    tarball = _make_tarball(cards, ["instance_abc_fused.jpg", "instance_def_fused.jpg"])
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)

    count = importer.import_tarball(tarball, "run-1")

    assert count == 2
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT track_id FROM card_instances WHERE run_id='run-1'").fetchall()
    assert {r[0] for r in rows} == {"abc", "def"}


def test_crops_are_copied(tmp_path):
    cards = [{"track_id": "x", "session_id": 0, "fused_image_path": "crops/instance_x_fused.jpg", "side": "Front"}]
    tarball = _make_tarball(cards, ["instance_x_fused.jpg"])
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)

    importer.import_tarball(tarball, "run-2")

    assert (tmp_path / "run-2" / "crops" / "instance_x_fused.jpg").exists()


def test_duplicate_import_is_idempotent(tmp_path):
    cards = [{"track_id": "dup", "session_id": 0, "fused_image_path": "crops/x.jpg", "side": "Front"}]
    tarball = _make_tarball(cards, ["x.jpg"])
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)
    importer.import_tarball(tarball, "run-3")
    importer.import_tarball(tarball, "run-3")  # second call — idempotent

    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM card_instances WHERE run_id='run-3'").fetchone()[0]
    assert count == 1
```

- [ ] **Step 2: Verify they fail**

```bash
python3 -m pytest tests/app/test_result_importer.py -v 2>&1 | tail -8
```
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `app/services/result_importer.py`:

```python
"""Unpack a results tarball from the instance and import cards into local SQLite."""
from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from pathlib import Path


class ResultImporter:
    def __init__(self, db_path: Path, output_base: Path) -> None:
        self.db_path = db_path
        self.output_base = output_base

    def import_tarball(self, tarball_path: Path, run_id: str) -> int:
        """Unpack tarball, copy crops, import card rows. Returns count of new cards."""
        run_dir = self.output_base / run_id
        crops_dir = run_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tarball_path, "r:gz") as tar:
            # Extract crop files into run_dir/crops/
            for member in tar.getmembers():
                if member.name.startswith("crops/") and not member.isdir():
                    fname = member.name.split("/", 1)[1]
                    data = tar.extractfile(member)
                    if data:
                        (crops_dir / fname).write_bytes(data.read())

            # Load export.json
            export_f = tar.extractfile("export.json")
            if not export_f:
                raise ValueError("Tarball missing export.json")
            cards: list[dict] = json.loads(export_f.read())

        count = 0
        with sqlite3.connect(self.db_path) as conn:
            for card in cards:
                fname = Path(card["fused_image_path"]).name
                local_path = str(crops_dir / fname)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO card_instances
                           (run_id, track_id, session_id, fused_image_path, angle)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            run_id,
                            card.get("track_id", ""),
                            card.get("session_id", 0),
                            local_path,
                            card.get("side", "Front"),
                        ),
                    )
                    count += conn.execute("SELECT changes()").fetchone()[0]
                except sqlite3.Error:
                    pass
            conn.commit()
        return count
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/app/test_result_importer.py -v 2>&1 | tail -8
```
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/result_importer.py tests/app/test_result_importer.py
git commit -m "feat(vastai): ResultImporter — tarball unpack + SQLite import"
```

---

### Task 5: VastAIRunner — orchestration

**Files:**
- Create: `app/services/vast_runner.py`
- Create: `tests/app/test_vast_runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/app/test_vast_runner.py`:

```python
"""Tests for VastAIRunner — all external calls mocked."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.vast_runner import VastAIRunner


def _make_runner(tmp_path):
    bus = MagicMock()
    bus.emit = MagicMock()
    return VastAIRunner(
        bus=bus,
        db_path=tmp_path / "cards.sqlite",
        output_base=tmp_path,
        api_key="test-key",
        gpu_type="RTX 4090",
        template_id="pytorch/pytorch:latest",
    )


@pytest.mark.asyncio
async def test_run_async_emits_started_and_completed(tmp_path):
    runner = _make_runner(tmp_path)

    # Mock collaborators
    runner._client = MagicMock()
    runner._client.search_offers.return_value = [{"id": 1, "dph_total": 0.5}]
    runner._client.provision.return_value = {"id": 42}
    runner._client.get_instance_ip.return_value = "1.2.3.4"

    worker = AsyncMock()
    worker.health_check.return_value = True
    worker.upload_video.return_value = "/tmp/video.mp4"
    worker.poll_status.return_value = {"status": "complete"}
    worker.download_results = AsyncMock()
    worker.confirm_downloaded = AsyncMock()
    worker.close = AsyncMock()

    importer = MagicMock()
    importer.import_tarball.return_value = 3
    runner._importer = importer

    with patch("app.services.vast_runner.InstanceWorkerClient", return_value=worker):
        with patch("app.services.vast_runner._save_active_instance"):
            with patch("app.services.vast_runner._clear_active_instance"):
                (tmp_path / "run-1").mkdir()
                await runner.run_async(
                    "run-1",
                    video=str(tmp_path / "video.mp4"),
                    output_dir=str(tmp_path / "run-1"),
                    db=str(tmp_path / "cards.sqlite"),
                )

    event_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_started" in event_names
    assert "run_completed" in event_names


@pytest.mark.asyncio
async def test_run_async_emits_failed_on_error(tmp_path):
    runner = _make_runner(tmp_path)
    runner._client = MagicMock()
    runner._client.search_offers.side_effect = RuntimeError("no offers")

    with patch("app.services.vast_runner._save_active_instance"):
        with patch("app.services.vast_runner._clear_active_instance"):
            with pytest.raises(RuntimeError):
                await runner.run_async(
                    "run-fail",
                    video="/tmp/v.mp4",
                    output_dir=str(tmp_path),
                    db=str(tmp_path / "cards.sqlite"),
                )

    event_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_failed" in event_names


@pytest.mark.asyncio
async def test_destroy_calls_client(tmp_path):
    runner = _make_runner(tmp_path)
    runner._instance_id = 42
    runner._client = MagicMock()
    runner._worker = AsyncMock()
    runner._worker.close = AsyncMock()

    with patch("app.services.vast_runner._clear_active_instance"):
        await runner.destroy_instance()

    runner._client.destroy.assert_called_once_with(42)
    assert runner._instance_id is None
```

- [ ] **Step 2: Verify they fail**

```bash
python3 -m pytest tests/app/test_vast_runner.py -v 2>&1 | tail -8
```
Expected: ImportError.

- [ ] **Step 3: Implement VastAIRunner**

Create `app/services/vast_runner.py`:

```python
"""Orchestrates vast.ai instance lifecycle + job execution."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from app.services.event_bus import Event, EventBus
from app.services.vast_client import VastAIClient
from app.services.worker_client import InstanceWorkerClient
from app.services.result_importer import ResultImporter

_CONFIG_PATH = Path(__file__).parent.parent.parent / "card_capture_config.json"


def _save_active_instance(instance_id: int) -> None:
    data: dict = {}
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
    data["active_vast_instance"] = instance_id
    _CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _clear_active_instance() -> None:
    if not _CONFIG_PATH.exists():
        return
    try:
        data = json.loads(_CONFIG_PATH.read_text())
        data["active_vast_instance"] = None
        _CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


class VastAIRunner:
    """Mirrors PipelineRunner.run_async interface for drop-in substitution."""

    def __init__(
        self,
        bus: EventBus,
        db_path: Path,
        output_base: Path,
        api_key: str,
        gpu_type: str = "RTX 4090",
        template_id: str = "",
        branch: str = "main",
        idle_timeout_s: int = 300,
    ) -> None:
        self.bus = bus
        self._client = VastAIClient(api_key)
        self._gpu_type = gpu_type
        self._template_id = template_id
        self._branch = branch
        self._idle_timeout_s = idle_timeout_s
        self._instance_id: Optional[int] = None
        self._worker: Optional[InstanceWorkerClient] = None
        self._importer = ResultImporter(db_path=db_path, output_base=output_base)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **_kw,
    ) -> None:
        """Provision instance (if needed), run job, download results, destroy."""
        try:
            await self._ensure_instance()
            self.bus.emit(run_id, Event(name="run_started"))

            server_path = await self._worker.upload_video(Path(video))
            await self._worker.submit_job(run_id, server_path, {"config_preset": config_preset})

            # Poll until complete or failed
            while True:
                status = await self._worker.poll_status(run_id)
                if status["status"] == "complete":
                    break
                if status["status"] == "failed":
                    raise RuntimeError(f"Remote job failed: {status.get('error', 'unknown')}")
                await asyncio.sleep(3)

            # Download and import results
            tarball = Path(output_dir) / f"{run_id}_results.tar.gz"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            await self._worker.download_results(run_id, tarball)
            await self._worker.confirm_downloaded(run_id)
            self._importer.import_tarball(tarball, run_id)
            tarball.unlink(missing_ok=True)

            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            self.bus.emit(run_id, Event(name="run_failed", data={"error": str(exc)}))
            raise
        finally:
            await self.destroy_instance()

    async def run_batch_async(self, jobs: list[dict]) -> None:
        """Process multiple videos on one instance, destroy when all done."""
        try:
            await self._ensure_instance()
            for job in jobs:
                await self.run_async(**job)
        finally:
            await self.destroy_instance()

    async def destroy_instance(self) -> None:
        if self._instance_id is None:
            return
        if self._worker:
            await self._worker.close()
            self._worker = None
        self._client.destroy(self._instance_id)
        _clear_active_instance()
        self._instance_id = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_instance(self) -> None:
        if self._worker is not None:
            return

        # Find cheapest offer
        offers = self._client.search_offers(self._gpu_type)
        if not offers:
            raise RuntimeError(f"No vast.ai offers found for GPU type: {self._gpu_type}")
        offer_id = offers[0]["id"]

        # Provision
        result = self._client.provision(offer_id, self._template_id, self._branch)
        self._instance_id = result["id"]
        _save_active_instance(self._instance_id)

        # Wait for IP (up to 2 minutes)
        ip: Optional[str] = None
        for _ in range(24):
            await asyncio.sleep(5)
            ip = self._client.get_instance_ip(self._instance_id)
            if ip:
                break
        if not ip:
            raise RuntimeError("Instance did not receive an IP within 2 minutes")

        self._worker = InstanceWorkerClient(f"http://{ip}:8765")

        # Wait for health (up to 3 more minutes)
        for _ in range(36):
            if await self._worker.health_check():
                return
            await asyncio.sleep(5)
        raise RuntimeError("Instance worker did not become healthy within 5 minutes")
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/app/test_vast_runner.py -v 2>&1 | tail -8
```
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/vast_runner.py tests/app/test_vast_runner.py
git commit -m "feat(vastai): VastAIRunner — full instance lifecycle orchestration"
```

---

### Task 6: Backend selector in video process route

**Files:**
- Modify: `app/api/videos.py`

- [ ] **Step 1: Read the current process route**

Read `app/api/videos.py` lines 64–85. The key lines are:

```python
runner = PipelineRunner(bus=request.app.state.event_bus, flow_cls=None, db_path=db_path)
```

- [ ] **Step 2: Add `_build_runner` and update the route**

In `app/api/videos.py`, add this helper function before the route handlers (after the imports):

```python
def _build_runner(request: Request):
    """Return VastAIRunner or PipelineRunner based on pipeline_backend config."""
    import json, os
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

    if cfg.get("pipeline_backend") == "cuda":
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
            idle_timeout_s=int(cfg.get("cuda_idle_timeout_s", 300)),
        )

    return PipelineRunner(bus=bus, flow_cls=None, db_path=db_path)
```

Then replace the line `runner = PipelineRunner(...)` in the process route with:

```python
runner = _build_runner(request)
```

- [ ] **Step 3: Run existing integration tests**

```bash
python3 -m pytest tests/app/ -q -k "not vast" 2>&1 | tail -5
```
Expected: no new failures.

- [ ] **Step 4: Commit**

```bash
git add app/api/videos.py
git commit -m "feat(vastai): backend selector in video process route"
```

---

### Task 7: Instance-side vastai_worker.py

**Files:**
- Create: `app/vastai_worker.py`
- Create: `tests/app/test_vastai_worker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/app/test_vastai_worker.py`:

```python
"""Tests for vastai_worker — uses FastAPI TestClient."""
import asyncio
import io
import json
import tarfile
from fastapi.testclient import TestClient

# Import lazily to avoid triggering startup in other tests
def _app():
    import importlib, sys
    if "app.vastai_worker" in sys.modules:
        del sys.modules["app.vastai_worker"]
    from app.vastai_worker import app
    return app


def test_health_returns_ok():
    client = TestClient(_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_submit_job_enqueues():
    client = TestClient(_app())
    r = client.post("/jobs", json={
        "job_id": "test-job-1",
        "video_path": "/tmp/fake.mp4",
        "config_preset": "balanced",
    })
    assert r.status_code == 200
    assert r.json()["job_id"] == "test-job-1"


def test_status_pending_after_submit():
    client = TestClient(_app())
    client.post("/jobs", json={"job_id": "test-job-2", "video_path": "/tmp/fake.mp4"})
    r = client.get("/jobs/test-job-2")
    assert r.status_code == 200
    assert r.json()["status"] in ("pending", "running")


def test_status_404_for_unknown_job():
    client = TestClient(_app())
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def test_results_409_when_not_complete():
    client = TestClient(_app())
    client.post("/jobs", json={"job_id": "not-done", "video_path": "/tmp/fake.mp4"})
    r = client.get("/jobs/not-done/results")
    assert r.status_code == 409


def test_confirm_deletes_job():
    client = TestClient(_app())
    client.post("/jobs", json={"job_id": "to-delete", "video_path": "/tmp/fake.mp4"})
    r = client.delete("/jobs/to-delete")
    assert r.status_code == 200
    assert client.get("/jobs/to-delete").status_code == 404
```

- [ ] **Step 2: Verify they fail**

```bash
python3 -m pytest tests/app/test_vastai_worker.py -v 2>&1 | tail -10
```
Expected: ImportError or 5 fail.

- [ ] **Step 3: Implement vastai_worker.py**

Create `app/vastai_worker.py`:

```python
"""
Instance-side FastAPI worker — runs on the vast.ai GPU instance.

Start with:  uvicorn app.vastai_worker:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="card-capture vast.ai worker")

_UPLOAD_DIR = Path(os.environ.get("WORKER_UPLOAD_DIR", "/tmp/cc_uploads"))
_OUTPUT_DIR = Path(os.environ.get("WORKER_OUTPUT_DIR", "/tmp/cc_output"))
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory state — single-process, no persistence needed
_jobs: dict[str, dict[str, Any]] = {}
_queue: asyncio.Queue = asyncio.Queue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_video(file: UploadFile):
    """Receive a video file; return its server-side path."""
    dest = _UPLOAD_DIR / (file.filename or "video.mp4")
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"path": str(dest)}


@app.post("/jobs")
async def submit_job(body: dict):
    """Enqueue a processing job."""
    job_id = body["job_id"]
    _jobs[job_id] = {"status": "pending", "progress_pct": 0}
    await _queue.put(body)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/jobs/{job_id}/results")
def job_results(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if _jobs[job_id]["status"] != "complete":
        raise HTTPException(status_code=409, detail="Job not complete yet")
    tarball = _OUTPUT_DIR / f"{job_id}.tar.gz"
    if not tarball.exists():
        raise HTTPException(status_code=404, detail="Results tarball missing")
    return FileResponse(str(tarball), media_type="application/gzip",
                        filename=f"{job_id}_results.tar.gz")


@app.delete("/jobs/{job_id}")
def confirm_job(job_id: str):
    """Mac Mini calls this after downloading results. Triggers idle shutdown."""
    _jobs.pop(job_id, None)
    (_OUTPUT_DIR / f"{job_id}.tar.gz").unlink(missing_ok=True)
    # Shutdown when queue drained and no running jobs
    running = any(j["status"] == "running" for j in _jobs.values())
    if _queue.empty() and not running:
        asyncio.get_event_loop().call_later(3, _shutdown)
    return {"deleted": job_id}


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    asyncio.create_task(_worker_loop())


async def _worker_loop():
    while True:
        job = await _queue.get()
        job_id = job["job_id"]
        _jobs[job_id]["status"] = "running"
        try:
            await asyncio.get_event_loop().run_in_executor(None, _run_pipeline, job)
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["progress_pct"] = 100
        except Exception as exc:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
        finally:
            _queue.task_done()


def _run_pipeline(job: dict) -> None:
    job_id = job["job_id"]
    video_path = job["video_path"]
    config_preset = job.get("config_preset", "balanced")
    output_dir = _OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "cards.sqlite"

    # Sub-project B will replace this with the CUDA-native pipeline.
    # For now, invoke the existing Metaflow pipeline via subprocess.
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
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:] or result.stdout[-500:])

    _package_results(job_id, output_dir, db_path)


def _package_results(job_id: str, output_dir: Path, db_path: Path) -> None:
    """Bundle crops + export.json into a gzipped tarball."""
    import sqlite3

    cards: list[dict] = []
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT track_id, session_id, fused_image_path, angle FROM card_instances WHERE run_id=?",
                (job_id,),
            ).fetchall()
            cards = [dict(r) for r in rows]

    tarball = _OUTPUT_DIR / f"{job_id}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        crops_dir = output_dir / "crops"
        if crops_dir.exists():
            tar.add(crops_dir, arcname="crops")
        export_bytes = json.dumps(cards).encode()
        info = tarfile.TarInfo(name="export.json")
        info.size = len(export_bytes)
        tar.addfile(info, io.BytesIO(export_bytes))


def _shutdown() -> None:
    os.kill(os.getpid(), 15)  # SIGTERM — clean exit, billing stops
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/app/test_vastai_worker.py -v 2>&1 | tail -10
```
Expected: all 7 pass.

- [ ] **Step 5: Commit**

```bash
git add app/vastai_worker.py tests/app/test_vastai_worker.py
git commit -m "feat(vastai): instance-side worker FastAPI app"
```

---

### Task 8: Batch API + wiring

**Files:**
- Create: `app/api/batch.py`
- Modify: `app/main.py`

- [ ] **Step 1: Write failing tests**

Create `tests/app/test_batch_api.py`:

```python
"""Tests for POST /api/v1/runs/batch."""
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import pytest


def _make_client():
    from app.main import create_app
    from pathlib import Path
    import tempfile, sqlite3
    tmp = tempfile.mkdtemp()
    db = Path(tmp) / "cards.sqlite"
    sqlite3.connect(db).close()
    app = create_app(db_path=db)
    return TestClient(app)


def test_batch_requires_video_ids():
    client = _make_client()
    r = client.post("/api/v1/runs/batch", json={})
    assert r.status_code == 422


def test_batch_returns_batch_id():
    client = _make_client()
    with patch("app.api.batch.asyncio") as mock_asyncio:
        mock_asyncio.create_task = MagicMock()
        r = client.post("/api/v1/runs/batch", json={"video_ids": [1, 2]})
    assert r.status_code == 202
    assert "batch_id" in r.json()


def test_batch_status_404_unknown():
    client = _make_client()
    r = client.get("/api/v1/runs/batch/does-not-exist")
    assert r.status_code == 404
```

- [ ] **Step 2: Verify tests fail**

```bash
python3 -m pytest tests/app/test_batch_api.py -v 2>&1 | tail -8
```
Expected: 404 on /api/v1/runs/batch (route not registered yet).

- [ ] **Step 3: Implement batch.py**

Create `app/api/batch.py`:

```python
"""Batch run routes — /api/v1/runs/batch."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

# In-memory batch state (keyed by batch_id)
_batches: dict[str, dict[str, Any]] = {}


class BatchRequest(BaseModel):
    video_ids: list[str]
    config_preset: str = "balanced"


@router.post("", status_code=202)
async def create_batch(body: BatchRequest, request: Request):
    """Enqueue a batch of videos for cloud GPU processing."""
    if not body.video_ids:
        raise HTTPException(status_code=422, detail="video_ids must not be empty")

    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    _batches[batch_id] = {
        "batch_id": batch_id,
        "status": "queued",
        "total": len(body.video_ids),
        "completed": 0,
        "failed": 0,
        "jobs": [
            {"video_id": vid, "status": "pending", "run_id": None}
            for vid in body.video_ids
        ],
    }

    asyncio.create_task(_run_batch(batch_id, body, request))
    return {"batch_id": batch_id}


@router.get("/{batch_id}")
def get_batch(batch_id: str):
    if batch_id not in _batches:
        raise HTTPException(status_code=404, detail="Batch not found")
    return _batches[batch_id]


async def _run_batch(batch_id: str, body: BatchRequest, request: Request) -> None:
    """Background task: process each video sequentially on one instance."""
    import json, os
    from pathlib import Path
    from app.api.videos import _build_runner

    _batches[batch_id]["status"] = "running"
    runner = None

    try:
        runner = _build_runner(request)
        db_path = request.app.state.db_path
        output_base = db_path.parent

        for i, job in enumerate(_batches[batch_id]["jobs"]):
            video_id = job["video_id"]
            run_id = f"batch_{batch_id}_v{video_id}"
            job["run_id"] = run_id
            job["status"] = "running"

            # Look up video path from DB
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT source_path FROM videos WHERE id=?", (video_id,)
                ).fetchone()
            if not row:
                job["status"] = "failed"
                job["error"] = f"Video {video_id} not found"
                _batches[batch_id]["failed"] += 1
                continue

            output_dir = str(output_base / run_id)
            try:
                await runner.run_async(
                    run_id,
                    video=row[0],
                    output_dir=output_dir,
                    db=str(db_path),
                    config_preset=body.config_preset,
                )
                job["status"] = "complete"
                _batches[batch_id]["completed"] += 1
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = str(exc)
                _batches[batch_id]["failed"] += 1

    except Exception as exc:
        _batches[batch_id]["status"] = "failed"
        _batches[batch_id]["error"] = str(exc)
        return
    finally:
        if runner is not None:
            try:
                await runner.destroy_instance()
            except Exception:
                pass

    total = _batches[batch_id]["total"]
    failed = _batches[batch_id]["failed"]
    _batches[batch_id]["status"] = "failed" if failed == total else (
        "partial" if failed > 0 else "complete"
    )
```

- [ ] **Step 4: Wire batch router in app/main.py**

Read `app/main.py`. Find where other routers are included (e.g. `app.include_router(runs.router, ...)`). Add:

```python
from app.api import batch
app.include_router(batch.router, prefix="/api/v1/runs/batch", tags=["batch"])
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/app/test_batch_api.py -v 2>&1 | tail -8
```
Expected: all 3 pass.

- [ ] **Step 6: Commit**

```bash
git add app/api/batch.py app/main.py tests/app/test_batch_api.py
git commit -m "feat(vastai): batch API — POST /runs/batch + status endpoint"
```

---

### Task 9: TypeScript types and API client methods

**Files:**
- Modify: `app/web/src/lib/api/types.ts`
- Modify: `app/web/src/lib/api/client.ts`

- [ ] **Step 1: Add types**

In `app/web/src/lib/api/types.ts`, append:

```typescript
export interface VastConfig {
    pipeline_backend: 'mps' | 'cuda';
    cuda_gpu_type: 'RTX 4090' | 'Flagship' | 'RTX 5060 Ti';
    vast_template_id: string;
    cuda_idle_timeout_s: number;
}

export interface BatchJob {
    video_id: string;
    status: 'pending' | 'running' | 'complete' | 'failed';
    run_id: string | null;
    error?: string;
}

export interface BatchStatus {
    batch_id: string;
    status: 'queued' | 'running' | 'complete' | 'partial' | 'failed';
    total: number;
    completed: number;
    failed: number;
    jobs: BatchJob[];
    error?: string;
}
```

- [ ] **Step 2: Add API client methods**

In `app/web/src/lib/api/client.ts`, add to the `api` export object:

```typescript
    compute: {
        get: () => req<T.VastConfig>('GET', '/config/compute'),
        patch: (body: Partial<T.VastConfig>) => req<T.VastConfig>('PATCH', '/config/compute', body),
    },
    batch: {
        create: (video_ids: string[], config_preset?: string) =>
            req<{ batch_id: string }>('POST', '/runs/batch', { video_ids, config_preset }),
        status: (batch_id: string) => req<T.BatchStatus>('GET', `/runs/batch/${batch_id}`),
    },
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd app/web && npx tsc --noEmit 2>&1 | tail -10
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add app/web/src/lib/api/types.ts app/web/src/lib/api/client.ts
git commit -m "feat(vastai): TS types and API client for compute config + batch"
```

---

### Task 10: Settings UI — Compute section

**Files:**
- Modify: `app/web/src/routes/settings/+page.svelte`

- [ ] **Step 1: Add Compute section**

Read `app/web/src/routes/settings/+page.svelte` first to find the end of the `<script>` block and end of the template.

In the `<script>` block, add after existing state declarations:

```javascript
    // Compute backend config
    import type { VastConfig } from '$lib/api/types';
    let computeConfig = $state<VastConfig>({
        pipeline_backend: 'mps',
        cuda_gpu_type: 'RTX 4090',
        vast_template_id: '',
        cuda_idle_timeout_s: 300,
    });
    let computeSaving = $state(false);
    let apiKeyInput = $state('');
    let apiKeyStatus = $state<'unchecked' | 'ok' | 'error'>('unchecked');

    async function loadComputeConfig() {
        try {
            computeConfig = await api.compute.get();
        } catch { /* keep defaults */ }
    }

    async function saveComputeConfig() {
        computeSaving = true;
        try {
            computeConfig = await api.compute.patch(computeConfig);
            saveSuccess = 'Compute config saved.';
        } finally {
            computeSaving = false;
        }
    }
```

Add `loadComputeConfig()` call inside the existing `onMount` (or whichever load function runs on mount).

In the template, add a new `<section>` after the existing sections (before the closing style tag):

```svelte
<section>
    <h2>Compute</h2>
    <p class="section-desc">Choose where heavy pipeline work runs.</p>

    <div class="form-row">
        <label>Pipeline</label>
        <select bind:value={computeConfig.pipeline_backend}>
            <option value="mps">Local (MPS — Mac mini)</option>
            <option value="cuda">Cloud GPU (vast.ai)</option>
        </select>
    </div>

    {#if computeConfig.pipeline_backend === 'cuda'}
        <div class="form-row">
            <label>GPU type</label>
            <select bind:value={computeConfig.cuda_gpu_type}>
                <option value="RTX 4090">RTX 4090 (recommended)</option>
                <option value="Flagship">Flagship (best available)</option>
                <option value="RTX 5060 Ti">RTX 5060 Ti (budget)</option>
            </select>
        </div>

        <div class="form-row">
            <label>Vast template ID</label>
            <input type="text" bind:value={computeConfig.vast_template_id}
                   placeholder="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel" />
        </div>

        <div class="form-row">
            <label>Idle timeout (s)</label>
            <input type="number" bind:value={computeConfig.cuda_idle_timeout_s}
                   min="60" max="3600" step="60" />
        </div>

        <div class="form-row">
            <label>VAST_API_KEY</label>
            <span class="key-hint">Set via environment variable — never stored in config.</span>
        </div>
    {/if}

    <button class="btn-primary" onclick={saveComputeConfig} disabled={computeSaving}>
        {computeSaving ? 'Saving…' : 'Save Compute Config'}
    </button>
</section>
```

- [ ] **Step 2: Verify app builds**

```bash
cd app/web && npm run build 2>&1 | tail -10
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add app/web/src/routes/settings/+page.svelte
git commit -m "feat(vastai): Compute section in Settings — pipeline toggle + GPU selector"
```

---

### Task 11: Batch UI

**Files:**
- Create: `app/web/src/routes/batch/+page.svelte`

- [ ] **Step 1: Create the batch page**

Create `app/web/src/routes/batch/+page.svelte`:

```svelte
<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Video, BatchStatus } from '$lib/api/types';

    let videos = $state<Video[]>([]);
    let selected = $state<Set<string>>(new Set());
    let loading = $state(true);
    let submitting = $state(false);
    let batchStatus = $state<BatchStatus | null>(null);
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    onMount(async () => {
        try {
            videos = await api.videos.list();
        } catch (e) {
            console.error(e);
        } finally {
            loading = false;
        }
    });

    function toggle(id: string) {
        const s = new Set(selected);
        s.has(id) ? s.delete(id) : s.add(id);
        selected = s;
    }

    function toggleAll() {
        selected = selected.size === videos.length
            ? new Set<string>()
            : new Set(videos.map(v => v.video_id));
    }

    async function submitBatch() {
        if (selected.size === 0) return;
        submitting = true;
        try {
            const { batch_id } = await api.batch.create([...selected]);
            batchStatus = await api.batch.status(batch_id);
            pollTimer = setInterval(async () => {
                batchStatus = await api.batch.status(batch_id);
                if (['complete', 'failed', 'partial'].includes(batchStatus.status)) {
                    clearInterval(pollTimer!);
                    pollTimer = null;
                }
            }, 3000);
        } catch (e) {
            console.error(e);
        } finally {
            submitting = false;
        }
    }

    function statusColor(s: string) {
        return s === 'complete' ? '#28a745' : s === 'failed' ? '#dc3545' : s === 'running' ? '#007bff' : '#6c757d';
    }
</script>

<h1>Batch Process</h1>

{#if batchStatus}
    <section class="batch-status">
        <h2>Batch {batchStatus.batch_id} — <span style="color:{statusColor(batchStatus.status)}">{batchStatus.status}</span></h2>
        <p>{batchStatus.completed}/{batchStatus.total} complete · {batchStatus.failed} failed</p>
        <ul class="job-list">
            {#each batchStatus.jobs as job}
                <li>
                    <span class="job-id">Video {job.video_id}</span>
                    <span class="job-status" style="color:{statusColor(job.status)}">{job.status}</span>
                    {#if job.run_id}<a href="/runs/{job.run_id}">view run</a>{/if}
                    {#if job.error}<span class="err">{job.error}</span>{/if}
                </li>
            {/each}
        </ul>
        <button onclick={() => batchStatus = null}>Start new batch</button>
    </section>
{:else}
    <p class="section-desc">Select videos to process on the cloud GPU, then click Process Batch.</p>

    <div class="toolbar">
        <button onclick={toggleAll} class="btn-secondary">
            {selected.size === videos.length ? 'Deselect all' : 'Select all'}
        </button>
        <button onclick={submitBatch} class="btn-primary"
                disabled={selected.size === 0 || submitting}>
            {submitting ? 'Submitting…' : `Process Batch (${selected.size})`}
        </button>
    </div>

    {#if loading}
        <p>Loading videos…</p>
    {:else if videos.length === 0}
        <p class="empty">No videos uploaded yet.</p>
    {:else}
        <table class="video-table">
            <thead><tr><th></th><th>Filename</th><th>Status</th></tr></thead>
            <tbody>
                {#each videos as video}
                    <tr class:selected={selected.has(video.video_id)}
                        onclick={() => toggle(video.video_id)}>
                        <td><input type="checkbox" checked={selected.has(video.video_id)}
                                   onchange={() => toggle(video.video_id)} /></td>
                        <td>{video.filename ?? video.source_path?.split('/').pop()}</td>
                        <td>{video.status ?? '—'}</td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}
{/if}

<style>
    .toolbar { display: flex; gap: 1rem; margin-bottom: 1rem; align-items: center; }
    .video-table { width: 100%; border-collapse: collapse; }
    .video-table th, .video-table td { padding: 0.5rem 0.75rem; text-align: left;
        border-bottom: 1px solid #dee2e6; }
    .video-table tbody tr { cursor: pointer; }
    .video-table tbody tr:hover, .video-table tbody tr.selected { background: #f0f1ff; }
    .job-list { list-style: none; padding: 0; }
    .job-list li { display: flex; gap: 1rem; padding: 0.4rem 0; border-bottom: 1px solid #dee2e6; }
    .job-id { font-family: monospace; }
    .err { color: #dc3545; font-size: 0.8rem; }
    .empty { color: #6c757d; }
    .section-desc { color: #6c757d; margin-bottom: 1rem; }
</style>
```

- [ ] **Step 2: Add batch to navigation**

In `app/web/src/routes/+layout.svelte`, after line 17 (`<li><a href="/settings">Settings</a></li>`), add:

```svelte
            <li><a href="/batch">Batch</a></li>
```

- [ ] **Step 3: Verify app builds**

```bash
cd app/web && npm run build 2>&1 | tail -10
```
Expected: no errors.

- [ ] **Step 4: Commit and push**

```bash
git add app/web/src/routes/batch/+page.svelte
git commit -m "feat(vastai): batch UI — checkbox video selection + Process Batch button"
git push origin main
```
