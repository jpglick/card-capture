"""Videos routes — `/api/v1/videos`."""
from __future__ import annotations

import datetime
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.schemas.v1 import RunSummary, Video, VideoCreate
from app.services.pipeline_runner import PipelineRunner

router = APIRouter()


@router.get("", response_model=list[Video])
def list_videos():
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("", response_model=Video, status_code=201)
def create_video(payload: VideoCreate):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{video_id}", response_model=Video)
def get_video(video_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.delete("/{video_id}", status_code=204)
def delete_video(video_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("/{video_id}/process", response_model=RunSummary, status_code=202)
async def start_run(video_id: str, request: Request, bg: BackgroundTasks):
    """Enqueue a pipeline run for *video_id*, returning a run_id immediately."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    runner = PipelineRunner(bus=request.app.state.event_bus, flow_cls=None)
    bg.add_task(
        runner.run_async,
        run_id,
        video=f"/var/videos/{video_id}",
        output_dir=f"/var/runs/{run_id}",
        db="cards.sqlite",
    )
    return RunSummary(
        run_id=run_id,
        video_id=video_id,
        status="pending",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
