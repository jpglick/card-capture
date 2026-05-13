"""Step 1 — detect: run Stages 1–3 (sampler + triage + YOLO-OBB detection).

Wraps the existing ``_run_pipeline_workers`` producer/consumer subsystem.
Returns serialisable lists so that Metaflow can pickle the artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.steps.start import RunContext


@dataclass
class DetectOutput:
    """Outputs of the detect step — all JSON-serialisable."""

    frame_count: int
    accepted_frame_count: int
    # (frame_index, timestamp_ms, is_present) tuples
    accepted_frame_presence: List[Tuple[int, int, bool]]
    # Flat list of detection dicts (one per CornerDetection)
    detection_rows: List[Dict[str, Any]]
    # Sampler telemetry dict (all values must be JSON-serialisable)
    sampler_telemetry: Dict[str, Any]
    video_id: int


def run(ctx: RunContext) -> DetectOutput:
    """Execute Stages 1–3 and return serialised detection data.

    Delegates entirely to ``_run_pipeline_workers`` from the legacy
    ``pipeline.py`` module — no algorithm is changed here.

    Args:
        ctx: Populated ``RunContext`` (video_id must already be set).

    Returns:
        A fully-populated ``DetectOutput``.
    """
    from pathlib import Path as _Path
    from card_capture.pipeline import (
        ProcessingOptions,
        _run_pipeline_workers,
    )

    video_path = _Path(ctx.video_path)
    output_dir = _Path(ctx.output_dir)
    frame_dir = _Path(ctx.frame_dir)

    sampler, detector = _build_sampler_detector(ctx)

    options = _ctx_to_options(ctx, output_dir)

    stats, raw_rows = _run_pipeline_workers(
        video_path=video_path,
        video_id=ctx.video_id,
        frame_dir=frame_dir,
        sampler=sampler,
        detector=detector,
        options=options,
    )

    # Serialise _DetectionEnvelope → plain dict
    detection_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        dp = row.detection_packet
        cd = dp.corner_detection
        detection_rows.append(
            {
                "detection_id": idx,
                "frame_index": dp.frame_index,
                "timestamp_ms": dp.timestamp_ms,
                "width": dp.width,
                "height": dp.height,
                "corners": [(float(p[0]), float(p[1])) for p in cd.corners],
                "confidence": float(cd.confidence),
                "source_frame_path": row.source_frame_path,
                "triage_metrics": dict(row.triage_metrics),
            }
        )

    # Make sampler_telemetry JSON-serialisable
    sampler_telemetry = _serialise_telemetry(stats.sampler_telemetry)

    return DetectOutput(
        frame_count=stats.frame_count,
        accepted_frame_count=stats.accepted_frame_count,
        accepted_frame_presence=list(stats.accepted_frame_presence),
        detection_rows=detection_rows,
        sampler_telemetry=sampler_telemetry,
        video_id=ctx.video_id,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sampler_detector(ctx: RunContext):
    """Construct the sampler and detector from RunContext.detector."""
    from card_capture.detectors import FakeCardDetector, CardcaptorUltralyticsDetector
    from card_capture.sampler import SyntheticSampler, AdaptivePresenceSampler

    if ctx.detector == "fake":
        sampler = SyntheticSampler()
        detector = FakeCardDetector()
    else:
        from pathlib import Path as _Path
        weights = _Path("models/presence_classifier.pt")
        sampler = AdaptivePresenceSampler(
            video_path=_Path(ctx.video_path),
            reader_backend="auto",
            device="auto",
            presence_weights_path=weights if weights.exists() else None,
        )
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=ctx.corner_confidence_threshold,
            detection_width=640,
            device="auto",
        )
    return sampler, detector


def _ctx_to_options(ctx: RunContext, output_dir: Path):
    """Build a ``ProcessingOptions`` from a ``RunContext``."""
    from card_capture.pipeline import ProcessingOptions
    return ProcessingOptions(
        output_dir=output_dir,
        queue_size=ctx.queue_size,
        inference_batch_size=ctx.inference_batch_size,
        corner_confidence_threshold=ctx.corner_confidence_threshold,
        blur_threshold=ctx.blur_threshold,
        variance_threshold=ctx.variance_threshold,
        empty_pixel_threshold=ctx.empty_pixel_threshold,
        group_gap_ms=ctx.group_gap_ms,
        background_frames=ctx.background_frames,
        background_threshold=ctx.background_threshold,
        null_patience_frames=ctx.null_patience_frames,
        min_track_length=ctx.min_track_length,
        use_kornia=ctx.use_kornia,
        kornia_device=ctx.kornia_device,
        triage_keep_percentile=ctx.triage_keep_percentile,
        rotate_180=ctx.rotate_180,
        tracker_backend=ctx.tracker_backend,
        centroid_jump_ratio=ctx.centroid_jump_ratio,
        centroid_jump_frames=ctx.centroid_jump_frames,
        foil_threshold=ctx.foil_threshold,
        enable_foil_aware_fusion=ctx.enable_foil_aware_fusion,
    )


def _serialise_telemetry(telemetry: dict) -> dict:
    """Convert sampler_telemetry values to JSON-serialisable types."""
    import numpy as np

    result: dict = {}
    for k, v in telemetry.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        elif isinstance(v, list):
            result[k] = [
                int(x) if isinstance(x, np.integer) else
                float(x) if isinstance(x, np.floating) else x
                for x in v
            ]
        else:
            result[k] = v
    return result
