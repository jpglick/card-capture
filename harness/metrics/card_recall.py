"""card_recall metric.

card_recall = |{GT cards matched to a detection}| / |{GT cards}|

Returns None when there are 0 GT cards (undefined, not 1.0).
"""
from __future__ import annotations

from pathlib import Path

from typing import Optional

from harness.match import match_detections_to_truth
from harness.schema import TruthFile


def card_recall(*, db_path: Path, truth_path: Path, video_id: str) -> Optional[float]:
    """Compute card recall for one video.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite`` from a pipeline run.
    truth_path:
        Path to ``truth.json`` (schema_version 1 or legacy).
    video_id:
        String video identifier; must match ``truth.video_id`` and be present
        in ``videos.source_path`` of the database.

    Returns
    -------
    float in [0, 1], or ``None`` when there are no GT cards.
    """
    truth = TruthFile.model_validate_json(truth_path.read_text())
    if not truth.expected_cards:
        return None

    pairs = match_detections_to_truth(db_path, truth, video_id)

    # Count unique GT cards that were matched to at least one detection.
    matched_gt_ids = {
        p.gt_card_id for p in pairs if p.gt_card_id is not None and p.detection_id is not None
    }
    return len(matched_gt_ids) / len(truth.expected_cards)
