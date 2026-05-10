from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .pipeline_runner import HarnessInstance
from .truth import ExpectedCard


@dataclass(frozen=True)
class MatchedPair:
    truth_card: ExpectedCard
    side: str  # "F" or "B"
    instance: HarnessInstance


@dataclass(frozen=True)
class MatchResult:
    matched: Tuple[MatchedPair, ...]
    unmatched_truth: Tuple[ExpectedCard, ...]
    phantom_instances: Tuple[HarnessInstance, ...]


def _windows_overlap(a: Tuple[int, int], b: Tuple[int, int], tolerance_ms: int) -> bool:
    a_start, a_end = a
    b_start, b_end = b
    return (a_start - tolerance_ms) <= b_end and (b_start - tolerance_ms) <= a_end


def _instance_overlaps_window(inst: HarnessInstance, window: Optional[Tuple[int, int]], tol: int) -> bool:
    if window is None:
        return False
    return _windows_overlap((inst.start_ms, inst.end_ms), window, tol)


def match_instances_to_truth(
    instances: Sequence[HarnessInstance],
    truth: Sequence[ExpectedCard],
    tolerance_ms: int = 500,
) -> MatchResult:
    """Greedy temporal match: each truth side claims one best-overlapping instance."""
    remaining = list(instances)
    matched: List[MatchedPair] = []
    unmatched: List[ExpectedCard] = []

    for card in truth:
        sides_to_match = []
        if card.front_present and card.approx_front_window_ms is not None:
            sides_to_match.append(("F", card.approx_front_window_ms))
        if card.back_present and card.approx_back_window_ms is not None:
            sides_to_match.append(("B", card.approx_back_window_ms))

        any_side_matched = False
        for side, window in sides_to_match:
            best_idx = -1
            best_overlap: Optional[int] = None
            for idx, inst in enumerate(remaining):
                if not _instance_overlaps_window(inst, window, tolerance_ms):
                    continue
                ovl = min(inst.end_ms, window[1]) - max(inst.start_ms, window[0])
                if best_overlap is None or ovl > best_overlap:
                    best_overlap = ovl
                    best_idx = idx
            if best_idx >= 0:
                matched.append(MatchedPair(truth_card=card, side=side, instance=remaining.pop(best_idx)))
                any_side_matched = True

        if not any_side_matched:
            unmatched.append(card)

    return MatchResult(
        matched=tuple(matched),
        unmatched_truth=tuple(unmatched),
        phantom_instances=tuple(remaining),
    )
