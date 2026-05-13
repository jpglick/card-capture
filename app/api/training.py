"""Training routes — `/api/v1/training`.

Stubs only — Surface C fills in real implementations.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.v1 import TrainingDataset, TrainingJob, TrainingRetrainRequest

router = APIRouter()


@router.get("/datasets", response_model=list[TrainingDataset])
def list_datasets():
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("/retrain/{model_name}", response_model=TrainingJob, status_code=202)
def retrain(model_name: str, payload: TrainingRetrainRequest):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/jobs/{job_id}", response_model=TrainingJob)
def get_job(job_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")
