"""card_precision metric.

card_precision = |{detections matched to a GT card}| / |{all detections}|

Returns None when there are 0 detections (undefined).
"""
from __future__ import annotations

from pathlib import Path

from harness.match import match_detections_to_truth
from harness.metrics.types import MetricResult
from harness.schema import TruthFile


def card_precision(*, db_path: Path, truth_path: Path, video_id: str) -> MetricResult:
    """Compute card precision for one video.

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
    MetricResult with value in [0, 1], or value=None when there are no detections.
    """
    truth = TruthFile.model_validate_json(truth_path.read_text())
    pairs = match_detections_to_truth(db_path, truth, video_id)

    # All pairs that contain a detection (matched or phantom).
    all_detections = [p for p in pairs if p.detection_id is not None]
    if not all_detections:
        return MetricResult(value=None)

    matched = [p for p in all_detections if p.gt_card_id is not None]
    return MetricResult(value=len(matched) / len(all_detections))
