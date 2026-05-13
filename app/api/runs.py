"""Runs routes — `/api/v1/runs`."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.v1 import RunDetail, RunEvent, RunHardCase, RunRejection, RunSummary, RunTelemetry

router = APIRouter()


@router.get("", response_model=list[RunSummary])
def list_runs():
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{run_id}/cards")
def get_run_cards(run_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{run_id}/events", response_model=list[RunEvent])
def get_run_events(run_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{run_id}/telemetry", response_model=RunTelemetry)
def get_run_telemetry(run_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{run_id}/rejection_log", response_model=list[RunRejection])
def get_rejection_log(run_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{run_id}/hard_cases", response_model=list[RunHardCase])
def get_hard_cases(run_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")
