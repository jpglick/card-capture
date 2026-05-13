"""FastAPI router for ``/api/v1/training/*`` endpoints.

Mount this router in :mod:`app.main` under the prefix ``/api/v1/training``.
All handlers obtain the :class:`~app.services.training_service.TrainingService`
instance from ``request.app.state.training_service``.
"""
from fastapi import APIRouter, HTTPException, Request
from app.schemas.v1 import DatasetSummary, RetrainRequest, TrainingJobSummary, TrainingJobDetail
from app.services.training_service import TrainingService

router = APIRouter()


def _svc(request: Request) -> TrainingService:
    return request.app.state.training_service


@router.get("/datasets", response_model=list[DatasetSummary])
def list_datasets(request: Request):
    """Return dataset sizes for each model type."""
    return _svc(request).list_datasets()


@router.post("/retrain/{model_name}", status_code=202, response_model=TrainingJobSummary)
def retrain(model_name: str, body: RetrainRequest, request: Request):
    """Enqueue a retrain job."""
    job = _svc(request).start_retrain(model_name, epochs=body.epochs, learning_rate=body.learning_rate)
    return {
        "job_id": job.job_id,
        "model_name": job.model_name,
        "status": job.status,
        "created_at": job.created_at,
    }


@router.get("/jobs/{job_id}", response_model=TrainingJobDetail)
def get_job(job_id: str, request: Request):
    """Return the current state of a training job."""
    job = _svc(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return {
        "job_id": job.job_id,
        "model_name": job.model_name,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }
