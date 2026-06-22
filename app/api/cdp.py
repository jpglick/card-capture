"""CardDealerPro integration routes — `/api/v1/cdp`."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


def _svc(request: Request):
    return request.app.state.cdp_service


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SubmitCardRequest(BaseModel):
    batch_name: Optional[str] = None


class BulkSubmitRunRequest(BaseModel):
    run_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/submissions/{instance_id}")
def get_submission(instance_id: int, request: Request) -> dict[str, Any]:
    """Get the CDP submission status for a single card instance."""
    sub = _svc(request).get_submission(instance_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="No CDP submission for this card")
    return sub


@router.get("/submissions/run/{run_id}")
def get_run_submissions(run_id: str, request: Request) -> dict[str, Any]:
    """Get all CDP submission statuses for cards in a run.

    Returns a dict keyed by instance_id (string).
    """
    return _svc(request).get_run_submissions(run_id)


@router.post("/submit/{instance_id}")
def submit_card(instance_id: int, request: Request, body: SubmitCardRequest = SubmitCardRequest()) -> dict[str, Any]:
    """Submit a single card to CardDealerPro."""
    try:
        return _svc(request).submit_card(instance_id, batch_name=body.batch_name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CDP error: {exc}")


@router.post("/submit/run/{run_id}")
def bulk_submit_run(run_id: str, request: Request) -> dict[str, Any]:
    """Submit all visible (non-hidden) cards in a run to a single CDP batch."""
    try:
        return _svc(request).bulk_submit_run(run_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CDP error: {exc}")


@router.post("/poll/{instance_id}")
def poll_card(instance_id: int, request: Request) -> dict[str, Any]:
    """Poll CDP for identification/pricing status of a single card."""
    try:
        return _svc(request).poll_submission(instance_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CDP error: {exc}")


@router.post("/poll/batch/{batch_id}")
def poll_batch(batch_id: str, request: Request) -> list[dict[str, Any]]:
    """Poll CDP for all cards in a batch and update their statuses."""
    try:
        return _svc(request).poll_batch(batch_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CDP error: {exc}")
