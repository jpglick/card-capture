"""Training routes — `/api/v1/training`.

Handlers obtain the :class:`~app.services.training_service.TrainingService`
instance from ``request.app.state.training_service``.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.services.training_service import TrainingService

router = APIRouter()


def _svc(request: Request) -> TrainingService:
    return request.app.state.training_service


@router.get("/datasets")
def list_datasets(request: Request):
    """Return dataset sizes for each model type."""
    return _svc(request).list_datasets()


@router.post("/retrain/{model_name}", status_code=202)
def retrain(model_name: str, request: Request):
    """Enqueue a retrain job and return ``{"job_id": ..., "status": "queued"}``."""
    job = _svc(request).start_retrain(model_name)
    return {"job_id": job.job_id, "status": job.status}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    """Return the current state of a training job."""
    job = _svc(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return {
        "job_id": job.job_id,
        "model_name": job.model_name,
        "status": job.status,
        "metrics": job.metrics,
        "error": job.error,
    }
