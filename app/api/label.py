"""Label routes — `/api/v1/label`.

Stubs only — Surface D fills in real implementations.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.v1 import (
    DedupCluster,
    DedupClusterUpdate,
    LabelFB,
    LabelFBNext,
    LabelFBResult,
    LabelTruth,
)

router = APIRouter()


@router.get("/truth/{video_id}", response_model=LabelTruth)
def get_truth(video_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.put("/truth/{video_id}", response_model=LabelTruth)
def put_truth(video_id: str, payload: LabelTruth):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/fb/next", response_model=LabelFBNext)
def get_fb_next():
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("/fb", response_model=LabelFBResult, status_code=201)
def post_fb_label(payload: LabelFB):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/clusters", response_model=list[DedupCluster])
def list_clusters(status: str | None = None, page: int = 1, page_size: int = 20):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.patch("/clusters/{cluster_id}", response_model=DedupCluster)
def update_cluster(cluster_id: int, payload: DedupClusterUpdate):
    raise HTTPException(status_code=501, detail="not implemented yet")
