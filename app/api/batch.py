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

            # Look up video path from DB using the repository
            video_row = request.app.state.videos_repo.get(video_id)
            if not video_row:
                job["status"] = "failed"
                job["error"] = f"Video {video_id} not found"
                _batches[batch_id]["failed"] += 1
                continue

            video_path = video_row["source_path"]
            output_dir = str(output_base / run_id)
            try:
                await runner.run_async(
                    run_id,
                    video=video_path,
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
