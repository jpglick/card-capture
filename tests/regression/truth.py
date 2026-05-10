from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


class TruthValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExpectedCard:
    card_id: str
    front_present: bool
    back_present: bool
    approx_front_window_ms: Optional[Tuple[int, int]] = None
    approx_back_window_ms: Optional[Tuple[int, int]] = None
    physical_card_key: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class GroundTruth:
    video_id: str
    video_path: str
    expected_cards: Tuple[ExpectedCard, ...]
    labeled_at: str = ""
    labeled_by: str = ""


def _coerce_window(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise TruthValidationError(f"window must be [start_ms, end_ms], got {value!r}")
    return (int(value[0]), int(value[1]))


def load_truth(path: Path) -> GroundTruth:
    raw = json.loads(Path(path).read_text())

    for required in ("video_id", "video_path", "expected_cards"):
        if required not in raw:
            raise TruthValidationError(f"missing required field: {required}")

    cards = []
    for entry in raw["expected_cards"]:
        for required in ("card_id", "front_present", "back_present"):
            if required not in entry:
                raise TruthValidationError(f"card missing required field: {required}")
        cards.append(
            ExpectedCard(
                card_id=str(entry["card_id"]),
                front_present=bool(entry["front_present"]),
                back_present=bool(entry["back_present"]),
                approx_front_window_ms=_coerce_window(entry.get("approx_front_window_ms")),
                approx_back_window_ms=_coerce_window(entry.get("approx_back_window_ms")),
                physical_card_key=entry.get("physical_card_key"),
                notes=str(entry.get("notes", "")),
            )
        )

    return GroundTruth(
        video_id=str(raw["video_id"]),
        video_path=str(raw["video_path"]),
        expected_cards=tuple(cards),
        labeled_at=str(raw.get("labeled_at", "")),
        labeled_by=str(raw.get("labeled_by", "")),
    )
