"""Videos routes — `/api/v1/videos`."""
from __future__ import annotations

import datetime
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, File

from app.schemas.v1 import RunSummary, Video, VideoCreate
from app.services.pipeline_runner import PipelineRunner

UPLOADS_DIR = Path("card_capture_uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


def _svc(request: Request):
    return request.app.state.video_service


@router.get("", response_model=list[Video])
def list_videos(request: Request):
    return _svc(request).list_videos()


@router.post("", response_model=Video, status_code=201)
def create_video(payload: VideoCreate, request: Request):
    path = payload.file_path or payload.filename
    video_id = _svc(request).add_video(path)
    return _svc(request).get_video(video_id)


@router.post("/upload", response_model=Video, status_code=201)
async def upload_video(request: Request, file: UploadFile = File(...)):
    """Accept a video file upload, persist it to the managed uploads directory,
    and create the corresponding video record."""
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()
    video_id = _svc(request).add_video(str(dest))
    return _svc(request).get_video(video_id)


@router.get("/{video_id}", response_model=Video)
def get_video(video_id: int, request: Request):
    video = _svc(request).get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.delete("/{video_id}", status_code=204)
def delete_video(video_id: int, request: Request):
    _svc(request).delete_video(video_id)


@router.post("/{video_id}/process", response_model=RunSummary, status_code=202)
async def start_run(video_id: int, request: Request, bg: BackgroundTasks):
    """Enqueue a pipeline run for *video_id*, returning a run_id immediately."""
    video = _svc(request).get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    runner = PipelineRunner(bus=request.app.state.event_bus, flow_cls=None)
    
    # Use output dir relative to video or in a central location
    output_dir = Path("card_capture_output") / run_id
    
    bg.add_task(
        runner.run_async,
        run_id,
        video=video["source_path"],
        output_dir=str(output_dir),
        db=str(request.app.state.db_path),
    )
    
    return RunSummary(
        run_id=run_id,
        video_id=str(video_id),
        status="pending",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
