"""Cards routes — `/api/v1/cards`."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from app.schemas.v1 import Card, CardBulkAction, CardDetail, CardFilter

router = APIRouter()


def _svc(request: Request):
    return request.app.state.card_service


@router.get("", response_model=dict[str, Any])
def list_cards(
    request: Request,
    run_id: Optional[str] = None,
    video_id: Optional[int] = None,
    dedup_group_id: Optional[int] = None,
    review_state: Optional[str] = None,
    side: Optional[str] = None,
    is_foil: Optional[bool] = None,
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
    page: int = 1,
    page_size: int = 50,
):
    return _svc(request).list_cards(
        run_id=run_id, 
        video_id=video_id,
        page=page,
        page_size=page_size
    )


@router.get("/{card_id}", response_model=CardDetail)
def get_card(card_id: int, request: Request):
    card = _svc(request).get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    # For now CardDetail = Card, later we add more info
    return card


@router.patch("/{card_id}", response_model=CardDetail)
def update_card(card_id: int, payload: dict, request: Request):
    # TODO: implement real update
    card = _svc(request).get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return card


@router.post("/bulk")
def bulk_update_cards(payload: CardBulkAction, request: Request):
    # TODO: implement real bulk update
    return {"updated": 0}
