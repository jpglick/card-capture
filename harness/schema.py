"""Pydantic models for the truth.json schema (Contract 4, schema_version=1).

See docs/contracts/truth-schema.md for the full specification.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExpectedCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    card_id: str
    front_present: bool
    back_present: bool
    physical_card_key: str
    is_foil: bool
    approx_front_window_ms: Optional[tuple[int, int]] = None
    approx_back_window_ms: Optional[tuple[int, int]] = None
    notes: Optional[str] = None

    @field_validator("approx_front_window_ms", "approx_back_window_ms", mode="before")
    @classmethod
    def _ordered_window(cls, v: Any) -> Optional[tuple[int, int]]:
        if v is None:
            return v
        start, end = v
        if start > end:
            raise ValueError(
                f"window must be (start, end) with start <= end; got {list(v)!r}"
            )
        return (int(start), int(end))


class TruthFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    schema_version: int = Field(ge=1)
    expected_cards: list[ExpectedCard]


def from_legacy(payload: dict[str, Any]) -> TruthFile:
    """Adapt a schema_version==0 truth file (from templates/labeling.html)
    to a TruthFile with schema_version==1.

    Legacy shape produced by labeling.html::

        {
          "video_id": "video_<id>",
          "video_path": "...",
          "labeled_at": "YYYY-MM-DD",
          "expected_cards": [
            {
              "card_id": "card_001",
              "front_present": true,
              "back_present": false,
              "approx_front_window_ms": [start, end],   # optional
              "approx_back_window_ms":  [start, end],   # optional
              "physical_card_key": "topps_42"           # optional
            }
          ]
        }

    Differences vs. schema_version=1:
    - No ``schema_version`` field (treated as 0).
    - No ``is_foil`` field (defaults to False).
    - ``physical_card_key`` is optional (defaults to ``card_id``).
    - Extra fields (``video_path``, ``labeled_at``) are ignored.
    """
    if payload.get("schema_version", 0) == 1:
        return TruthFile.model_validate(payload)

    adapted_cards = []
    for card in payload.get("expected_cards", []):
        adapted_cards.append(
            {
                "card_id": card["card_id"],
                "front_present": card.get("front_present", False),
                "back_present": card.get("back_present", False),
                # Legacy files often lack physical_card_key; fall back to card_id.
                "physical_card_key": card.get("physical_card_key") or card["card_id"],
                # Legacy files never had is_foil; default to False.
                "is_foil": card.get("is_foil", False),
                "approx_front_window_ms": card.get("approx_front_window_ms"),
                "approx_back_window_ms": card.get("approx_back_window_ms"),
                "notes": card.get("notes"),
            }
        )

    return TruthFile.model_validate(
        {
            "video_id": payload.get("video_id", ""),
            "schema_version": 1,
            "expected_cards": adapted_cards,
        }
    )
