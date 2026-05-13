"""API endpoints for labeling operations.
"""
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from app.schemas.v1 import LabelTruth, LabelFB, DedupCluster

router = APIRouter()


def _svc(request: Request):
    return request.app.state.labeling_service


@router.get("/truth/{video_id}", response_model=Optional[LabelTruth])
def get_truth(video_id: str, request: Request):
    payload = _svc(request).get_truth(video_id)
    return payload


@router.put("/truth/{video_id}", status_code=204)
def put_truth(video_id: str, body: LabelTruth, request: Request):
    if body.video_id and body.video_id != video_id:
        raise HTTPException(status_code=400, detail="video_id mismatch")
    _svc(request).put_truth(video_id, body.model_dump())


@router.get("/fb/next")
def get_next_fb(request: Request):
    return _svc(request).next_fb_candidate() or {}


@router.get("/clusters", response_model=list[DedupCluster])
def list_clusters(request: Request, status: Optional[str] = None):

    label_id = _svc(request).post_fb_label(
        instance_id=body.instance_id,
        frame_index=body.frame_index,
        side=body.side,
    )
    return {"label_id": label_id}


@router.get("/clusters", response_model=list[DedupCluster])
def list_clusters(request: Request, status: Optional[str] = None):
    return _svc(request).list_clusters(status=status)


@router.patch("/clusters/{cluster_id}", status_code=204)
def patch_cluster(cluster_id: int, body: dict, request: Request):
    _svc(request).patch_cluster(
        cluster_id,
        status=body.get("status"),
        confirmed=body.get("confirmed"),
    )
