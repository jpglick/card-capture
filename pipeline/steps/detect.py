"""Step 1 — detect: run Stages 1–3 (sampler + triage + YOLO-OBB detection).

Wraps the existing ``_run_pipeline_workers`` producer/consumer subsystem.
Returns serialisable lists so that Metaflow can pickle the artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json as _json
import sqlite3 as _sqlite3

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
    # Diagnostic snapshot for the detect stage (frames, YOLO device/timing, triage rate)
    detect_telemetry: Dict[str, Any] = field(default_factory=dict)


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
    from card_capture.workers import (
        ProcessingOptions,
        _run_pipeline_workers,
    )

    video_path = _Path(ctx.video_path)
    output_dir = _Path(ctx.output_dir)
    frame_dir = _Path(ctx.frame_dir)

    sampler, detector = _build_sampler_detector(ctx)

    options = _ctx_to_options(ctx, output_dir)

    if ctx.detector == "cuda":
        detect_out = _run_cuda_inference(ctx, sampler, detector, output_dir, frame_dir)
        _save_corner_samples(ctx, detect_out.detection_rows, output_dir)
        return detect_out

    stats, consumer_stats, raw_rows = _run_pipeline_workers(
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
    sampler_telemetry.setdefault("target_yolo_fps", ctx.target_yolo_fps)

    _save_corner_samples(ctx, detection_rows, output_dir)

    triage_pass_rate = (
        stats.accepted_frame_count / stats.frame_count
        if stats.frame_count > 0 else 0.0
    )
    detect_telemetry: Dict[str, Any] = {
        "frame_count": stats.frame_count,
        "accepted_frame_count": stats.accepted_frame_count,
        "triage_pass_rate": round(triage_pass_rate, 4),
        "yolo_frames": consumer_stats.yolo_frames,
        "yolo_batches": consumer_stats.yolo_batches,
        "yolo_elapsed_s": round(consumer_stats.yolo_elapsed_s, 2),
        "yolo_device": consumer_stats.device_resolved,
        "presence_windows": stats.sampler_telemetry.get("last_presence_window_count"),
        "sampler_type": stats.sampler_telemetry.get("sampler_type"),
    }

    return DetectOutput(
        frame_count=stats.frame_count,
        accepted_frame_count=stats.accepted_frame_count,
        accepted_frame_presence=list(stats.accepted_frame_presence),
        detection_rows=detection_rows,
        sampler_telemetry=sampler_telemetry,
        video_id=ctx.video_id,
        detect_telemetry=detect_telemetry,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_corner_samples(ctx: RunContext, detection_rows: list, output_dir: Path) -> None:
    """Persist borderline YOLO detections (0.50–0.70 conf) for corner labeling."""
    if not ctx.db_path:
        return
    borderline = [d for d in detection_rows if 0.50 <= d["confidence"] <= 0.70]
    if not borderline:
        return
    try:
        with _sqlite3.connect(ctx.db_path) as conn:
            for d in borderline:
                conn.execute(
                    """INSERT OR IGNORE INTO corner_samples
                       (run_id, video_id, frame_index, image_path,
                        predicted_corners, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        getattr(ctx, 'ui_run_id', None) or "",
                        ctx.video_id or 0,
                        d["frame_index"],
                        d.get("source_frame_path", ""),
                        _json.dumps(d["corners"]),
                        d["confidence"],
                    ),
                )
            conn.commit()
    except Exception as exc:
        print(f"[detect] corner sampling failed: {exc}", flush=True)


def _build_sampler_detector(ctx: RunContext):
    """Construct the sampler and detector from RunContext.detector."""
    from card_capture.detectors import FakeCardDetector, CardcaptorUltralyticsDetector
    from card_capture.sampler import SyntheticSampler, AdaptivePresenceSampler

    if ctx.detector == "fake":
        sampler = SyntheticSampler()
        detector = FakeCardDetector()
    elif ctx.detector == "cuda":
        from card_capture.sampler.cuda_sampler import CudaSampler
        sampler = CudaSampler(
            video_path=_Path(ctx.video_path),
            stride=ctx.cuda_stride,
            opening_scan_s=ctx.opening_scan_s,
        )
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=ctx.corner_confidence,
            detection_width=640,
            device="cuda",   # explicit — hard-fail if CUDA unavailable on instance
        )
    else:
        from pathlib import Path as _Path
        weights = _Path("models/presence_classifier.pt")
        sampler = AdaptivePresenceSampler(
            video_path=_Path(ctx.video_path),
            reader_backend="auto",
            device="auto",
            presence_weights_path=weights if weights.exists() else None,
            presence_threshold=ctx.presence_threshold,
            target_yolo_fps=ctx.target_yolo_fps,
            opening_scan_s=ctx.opening_scan_s,
        )
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=ctx.corner_confidence,
            detection_width=640,
            device="auto",
        )
    return sampler, detector


def _run_cuda_inference(
    ctx: RunContext,
    sampler: "CudaSampler",
    detector: "CardcaptorUltralyticsDetector",
    output_dir: Path,
    frame_dir: Path,
) -> "DetectOutput":
    """Single-process CUDA inference path: decode → batch YOLO → DetectOutput.

    Used when ctx.detector == "cuda". Bypasses the multiprocessing
    producer/consumer model — frames are decoded by NVDEC and fed to YOLO
    in large batches without subprocess overhead.
    """
    from card_capture.models import FramePacket

    frames = list(sampler.sample())   # list[FrameSample] — already decoded
    batch_size = ctx.cuda_batch_size

    detection_rows: list[dict] = []
    accepted_frame_presence: list[tuple[int, int, bool]] = []
    det_id = 0

    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start:batch_start + batch_size]

        # Convert FrameSample → FramePacket for detect_batch
        packets_in = [
            FramePacket(
                frame_index=f.frame_index,
                timestamp_ms=f.timestamp_ms,
                image=f.image,
                width=f.width,
                height=f.height,
                triage_metrics={},
            )
            for f in batch
        ]

        packets_out = detector.detect_batch(
            packets_in, detector.confidence_threshold
        )

        for f in batch:
            accepted_frame_presence.append((f.frame_index, f.timestamp_ms, True))

        for pkt in packets_out:
            cd = pkt.corner_detection
            detection_rows.append(
                {
                    "detection_id": det_id,
                    "frame_index": pkt.frame_index,
                    "timestamp_ms": pkt.timestamp_ms,
                    "width": pkt.width,
                    "height": pkt.height,
                    "corners": [(float(p[0]), float(p[1])) for p in cd.corners],
                    "confidence": float(cd.confidence),
                    "source_frame_path": "",
                    "triage_metrics": {},
                }
            )
            det_id += 1

    return DetectOutput(
        frame_count=len(frames),
        accepted_frame_count=len(frames),
        accepted_frame_presence=accepted_frame_presence,
        detection_rows=detection_rows,
        sampler_telemetry={
            "sampler_type": "CudaSampler",
            "last_selected_frame_count": sampler.last_selected_frame_count,
            "last_source_fps": sampler.last_source_fps,
            "cuda_stride": ctx.cuda_stride,
            "target_yolo_fps": None,
        },
        video_id=ctx.video_id,
        detect_telemetry={
            "frame_count": len(frames),
            "accepted_frame_count": len(frames),
            "triage_pass_rate": 1.0,
            "yolo_frames": len(frames),
            "yolo_batches": (len(frames) + batch_size - 1) // batch_size,
            "yolo_device": "cuda",
            "sampler_type": "CudaSampler",
        }
    )


def _ctx_to_options(ctx: RunContext, output_dir: Path):
    """Build a ``ProcessingOptions`` from a ``RunContext``."""
    from card_capture.workers import ProcessingOptions
    return ProcessingOptions(
        output_dir=output_dir,
        queue_size=ctx.queue_size,
        inference_batch_size=ctx.inference_batch_size,
        corner_confidence_threshold=ctx.corner_confidence,
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
