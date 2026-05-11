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
    scene_type: str = "single_card"  # "single_card", "multi_card", "rapid_swap"
    foil_label: Optional[str] = None  # "foil", "holo", None
    is_occluded: bool = False
    occlusion_type: Optional[str] = None  # "finger", "sticker", "adjacent_card", "lens_flare", None


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
        for bool_field in ("front_present", "back_present"):
            if not isinstance(entry[bool_field], bool):
                raise TruthValidationError(
                    f"card field {bool_field!r} must be a boolean, got {entry[bool_field]!r}"
                )
        cards.append(
            ExpectedCard(
                card_id=str(entry["card_id"]),
                front_present=entry["front_present"],
                back_present=entry["back_present"],
                approx_front_window_ms=_coerce_window(entry.get("approx_front_window_ms")),
                approx_back_window_ms=_coerce_window(entry.get("approx_back_window_ms")),
                physical_card_key=entry.get("physical_card_key"),
                notes=str(entry.get("notes", "")),
                scene_type=str(entry.get("scene_type", "single_card")),
                foil_label=entry.get("foil_label"),
                is_occluded=bool(entry.get("is_occluded", False)),
                occlusion_type=entry.get("occlusion_type"),
            )
        )

    return GroundTruth(
        video_id=str(raw["video_id"]),
        video_path=str(raw["video_path"]),
        expected_cards=tuple(cards),
        labeled_at=str(raw.get("labeled_at", "")),
        labeled_by=str(raw.get("labeled_by", "")),
    )
