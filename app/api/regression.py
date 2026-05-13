"""API endpoints for regression reporting.
"""
from fastapi import APIRouter, Request
from app.schemas.v1 import Baseline, RegressionCompare

router = APIRouter()


def _svc(request: Request):
    return request.app.state.regression_service


@router.get("/baselines", response_model=list[Baseline])
def list_baselines(request: Request):
    return _svc(request).list_baselines()


@router.get("/compare", response_model=RegressionCompare)
def compare(a: int, b: int, request: Request):
    return _svc(request).compare(a, b)
