"""Videos routes — `/api/v1/videos`."""
from __future__ import annotations

import datetime
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, File

from app.schemas.v1 import RunSummary, Video, VideoCreate
from app.services.pipeline_runner import PipelineRunner, _REPO_ROOT

# Absolute path so the stored source_path is always resolvable regardless of cwd
UPLOADS_DIR = (Path(__file__).parent.parent.parent / "card_capture_uploads").resolve()
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


def _svc(request: Request):
    return request.app.state.video_service


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
    db_path = request.app.state.db_path
    runner = _build_runner(request)
    _svc(request).update_status(video_id, "processing")

    output_dir = Path(_REPO_ROOT) / "card_capture_output" / run_id

    bg.add_task(
        runner.run_async,
        run_id,
        video_id=video_id,
        video=video["source_path"],
        output_dir=str(output_dir),
        db=str(db_path.resolve()),
    )
    
    return RunSummary(
        run_id=run_id,
        video_id=str(video_id),
        status="pending",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
