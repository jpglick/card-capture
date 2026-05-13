"""Cards routes — `/api/v1/cards`."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.v1 import Card, CardBulkAction, CardDetail, CardFilter

router = APIRouter()


@router.get("", response_model=list[Card])
def list_cards(
    run_id: str | None = None,
    video_id: str | None = None,
    dedup_group_id: int | None = None,
    review_state: str | None = None,
    side: str | None = None,
    is_foil: bool | None = None,
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    page: int = 1,
    page_size: int = 50,
):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/{card_id}", response_model=CardDetail)
def get_card(card_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.patch("/{card_id}", response_model=CardDetail)
def update_card(card_id: str, payload: dict):
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.post("/bulk")
def bulk_update_cards(payload: CardBulkAction):
    raise HTTPException(status_code=501, detail="not implemented yet")
