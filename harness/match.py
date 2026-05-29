"""Match detected card instances (from cards.sqlite) to ground-truth cards
(from truth.json) for a given video.

Used by card_recall, card_precision, and side_accuracy to ensure consistent
pair assignment across all three metrics.

Matching rules
--------------
1. **Window-based (preferred):** If a GT card has ``approx_front_window_ms``
   or ``approx_back_window_ms``, match it to a card_instance whose temporal
   extent (min/max timestamp from card_views) overlaps the GT window AND
   whose ``angle`` matches the expected side (Front/Back).

2. **Order-based (fallback):** If the GT card has no time windows, match by
   detection order within the video (first unmatched detection → first GT card
   without a window, etc.).

Matching is 1-to-1: a card_instance can match at most one GT slot, and each
GT slot can be satisfied by at most one card_instance.

Return value
-----------
A list of :class:`MatchPair` covering:

- All GT cards (each appears as ``gt_card_id`` in exactly one pair).
- All detected card_instances (each appears as ``detection_id`` in exactly
  one pair; phantoms have ``gt_card_id=None``).

``side_match`` is ``True`` when the matched detection's ``angle`` equals the
expected side of its GT slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from card_capture.data.connection import read_connection
from harness.schema import ExpectedCard, TruthFile


@dataclass(frozen=True)
class MatchPair:
    """One matched (or unmatched) pairing between a GT card and a detection.

    - Both ids present → a real match.
    - ``detection_id`` is None → GT card was missed (no detection found).
    - ``gt_card_id`` is None → phantom detection (no GT card matched).
    - ``side_match`` is True only when both ids are present AND the detection's
      angle equals the expected side for this GT slot.
    """

    gt_card_id: Optional[str]
    detection_id: Optional[int]  # card_instances.id
    side_match: bool
    expected_side: Optional[str] = None


@dataclass
class _Detection:
    """Internal representation of one detected card_instance."""

    instance_id: int
    angle: Optional[str]
    start_ms: int  # min timestamp across card_views
    end_ms: int  # max timestamp across card_views


@dataclass
class _Slot:
    """Internal representation of one GT matching slot."""

    gt_card_id: str
    expected_side: Optional[str]  # "Front", "Back", or None (ordering fallback)
    window: Optional[tuple[int, int]]  # (start_ms, end_ms) or None


def match_detections_to_truth(
    db_path: Path, truth: TruthFile, video_id: str
) -> list[MatchPair]:
    """Compute the full list of MatchPairs for one video.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite``.
    truth:
        Parsed ``TruthFile`` (schema_version 1).
    video_id:
        String video identifier from ``truth.json``; matched against
        ``videos.source_path`` using a ``LIKE '%<video_id>%'`` filter so that
        fixture filenames like ``"all_matched.mov"`` match ``"all_matched"``.
    """
    detections = _load_detections(db_path, video_id)
    slots = _build_slots(truth.expected_cards)
    return _match(detections, slots)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_detections(db_path: Path, video_id: str) -> list[_Detection]:
    """Query card_instances + card_views for the given video."""
    with read_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                ci.id            AS instance_id,
                ci.angle         AS angle,
                MIN(cv.timestamp_ms) AS start_ms,
                MAX(cv.timestamp_ms) AS end_ms
            FROM card_instances ci
            JOIN card_views cv ON cv.card_instance_id = ci.id
            JOIN videos v ON v.id = ci.video_id
            WHERE v.source_path LIKE ?
            GROUP BY ci.id
            ORDER BY start_ms
            """,
            (f"%{video_id}%",),
        ).fetchall()
    return [
        _Detection(
            instance_id=int(row[0]),
            angle=row[1],
            start_ms=int(row[2]),
            end_ms=int(row[3]),
        )
        for row in rows
    ]


def _build_slots(expected_cards: list[ExpectedCard]) -> list[_Slot]:
    """Convert GT expected_cards into matching slots.

    One slot is created for each (GT card, expected side) combination:
    - If the card has ``approx_front_window_ms`` → one Front slot.
    - If the card has ``approx_back_window_ms`` → one Back slot.
    - If neither window is present → one slot using the card's primary side
      (Front if ``front_present``, else Back, else None) with no window
      constraint (ordering fallback).

    For a card with **both** front and back windows, two slots are produced
    (one Front, one Back).  For recall/precision calculations the caller
    aggregates by unique ``gt_card_id``.
    """
    slots: list[_Slot] = []
    for card in expected_cards:
        has_front_window = card.approx_front_window_ms is not None
        has_back_window = card.approx_back_window_ms is not None

        if has_front_window:
            slots.append(
                _Slot(
                    gt_card_id=card.card_id,
                    expected_side="Front",
                    window=card.approx_front_window_ms,
                )
            )
        if has_back_window:
            slots.append(
                _Slot(
                    gt_card_id=card.card_id,
                    expected_side="Back",
                    window=card.approx_back_window_ms,
                )
            )
        if not has_front_window and not has_back_window:
            # No windows → ordering fallback
            side: str | None = None
            if card.front_present:
                side = "Front"
            elif card.back_present:
                side = "Back"
            slots.append(_Slot(gt_card_id=card.card_id, expected_side=side, window=None))

    return slots


def _overlaps(det: _Detection, window: tuple[int, int]) -> bool:
    """True if the detection's temporal extent overlaps the GT window."""
    return det.start_ms <= window[1] and det.end_ms >= window[0]


def _match(detections: list[_Detection], slots: list[_Slot]) -> list[MatchPair]:
    """Greedy 1-to-1 matching of detections to GT slots.

    Phase 1 — window-constrained slots (in window-start order):
        For each windowed slot, pick the unmatched detection whose timestamp
        overlaps the window AND whose angle matches the expected side.  If
        multiple candidates exist, prefer the one with the smallest ``start_ms``.
        If no match is found, the slot is left unmatched (missed GT).

    Phase 2 — ordering fallback for windowless slots:
        Pair remaining unmatched detections to windowless slots in order.

    Phase 3 — remaining detections are phantoms.
    """
    available = list(detections)  # detections not yet assigned
    pairs: list[MatchPair] = []

    # Split slots into windowed and windowless, ordered by window start.
    windowed = sorted(
        [s for s in slots if s.window is not None],
        key=lambda s: s.window[0],  # type: ignore[index]
    )
    windowless = [s for s in slots if s.window is None]

    # Phase 1 — windowed matching
    for slot in windowed:
        assert slot.window is not None
        # Match by time overlap ONLY — do not filter on side here so that
        # mis-assigned sides are still counted (needed for side_accuracy).
        candidates = [d for d in available if _overlaps(d, slot.window)]
        if candidates:
            best = min(candidates, key=lambda d: d.start_ms)
            available.remove(best)
            side_match = slot.expected_side is None or best.angle == slot.expected_side
            pairs.append(
                MatchPair(
                    gt_card_id=slot.gt_card_id,
                    detection_id=best.instance_id,
                    side_match=side_match,
                    expected_side=slot.expected_side,
                )
            )
        else:
            pairs.append(
                MatchPair(
                    gt_card_id=slot.gt_card_id,
                    detection_id=None,
                    side_match=False,
                    expected_side=slot.expected_side,
                )
            )

    # Phase 2 — windowless ordering fallback
    for slot in windowless:
        if available:
            det = available.pop(0)
            side_match = slot.expected_side is None or det.angle == slot.expected_side
            pairs.append(
                MatchPair(
                    gt_card_id=slot.gt_card_id,
                    detection_id=det.instance_id,
                    side_match=side_match,
                    expected_side=slot.expected_side,
                )
            )
        else:
            pairs.append(
                MatchPair(
                    gt_card_id=slot.gt_card_id,
                    detection_id=None,
                    side_match=False,
                    expected_side=slot.expected_side,
                )
            )

    # Phase 3 — remaining detections are phantoms
    for det in available:
        pairs.append(MatchPair(gt_card_id=None, detection_id=det.instance_id, side_match=False))

    return pairs
