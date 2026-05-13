"""side_accuracy metric.

side_accuracy = |{matched detections with correct side}| / |{matched detections}|

Computed only over detections matched to a GT card.  Returns None when there
are 0 matched detections.
"""
from __future__ import annotations

from pathlib import Path

from harness.match import match_detections_to_truth
from harness.schema import TruthFile


def side_accuracy(*, db_path: Path, truth_path: Path, video_id: str) -> float | None:
    """Compute front/back side accuracy for one video.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite`` from a pipeline run.
    truth_path:
        Path to ``truth.json``.
    video_id:
        String video identifier.

    Returns
    -------
    float in [0, 1], or ``None`` when there are no matched detections.
    """
    truth = TruthFile.model_validate_json(truth_path.read_text())
    pairs = match_detections_to_truth(db_path, truth, video_id)

    matched = [p for p in pairs if p.gt_card_id is not None and p.detection_id is not None]
    if not matched:
        return None

    correct = sum(1 for p in matched if p.side_match)
    return correct / len(matched)
