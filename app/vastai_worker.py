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
