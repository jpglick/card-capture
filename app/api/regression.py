"""Regression routes — `/api/v1/regression`.

Stubs only — Surface D fills in real implementations.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.v1 import (
    RegressionBaseline,
    RegressionBaselineCreate,
    RegressionCompare,
    RegressionRun,
    RegressionRunRequest,
)

router = APIRouter()


@router.get("/baselines", response_model=list[RegressionBaseline])
def list_baselines():
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("/baselines", response_model=RegressionBaseline, status_code=201)
def create_baseline(payload: RegressionBaselineCreate):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("/run", response_model=RegressionRun, status_code=202)
def start_regression_run(payload: RegressionRunRequest):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/runs/{run_id}", response_model=RegressionRun)
def get_regression_run(run_id: int):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/compare", response_model=RegressionCompare)
def compare_runs(a: int, b: int):
    raise HTTPException(status_code=501, detail="not implemented yet")
