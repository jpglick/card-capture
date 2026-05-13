"""Runs routes — `/api/v1/runs`."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from app.schemas.v1 import RunDetail, RunEvent, RunHardCase, RunRejection, RunSummary, RunTelemetry

router = APIRouter()


def _svc(request: Request):
    return request.app.state.run_service


@router.get("", response_model=list[RunSummary])
def list_runs(request: Request, video_id: Optional[int] = None):
    return _svc(request).list_runs(video_id=video_id)


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, request: Request):
    run = _svc(request).get_run_details(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/cards")
def get_run_cards(run_id: str, request: Request):
    return request.app.state.card_service.list_cards(run_id=run_id)


@router.get("/{run_id}/events", response_model=list[RunEvent])
def get_run_events(run_id: str, request: Request):
    run = _svc(request).get_run_details(run_id)
    return run["events"] if run else []
