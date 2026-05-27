"""Fused CUDA hot path — decode + YOLO + eager warp + track + refine in one step.

Runs the entire CUDA detection/refinement path inside a single Metaflow @step so
the in-memory crop cache (~2.4GB of 750x1050 normalized crops) never crosses a
Metaflow artifact boundary. Stages 4 (novelty) and 5 (track) are reused unchanged
— they touch only detection metadata. refine.run is reused with the crop cache so
it never re-decodes the video.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from pipeline.steps import novelty, track, refine
from pipeline.steps.detect import _build_sampler_detector, _run_fused_inference, _save_corner_samples
from pipeline.steps.refine import RefineOutput
from pipeline.steps.start import RunContext


def run(ctx: RunContext) -> RefineOutput:
    """Execute the fused detection/refinement path and return a RefineOutput."""
    sampler, detector = _build_sampler_detector(ctx)
    output_dir = Path(ctx.output_dir)
    frame_dir = Path(ctx.frame_dir)

    crop_cache: Dict[int, Any] = {}
    _t_infer = time.time()
    detect_out = _run_fused_inference(
        ctx, sampler, detector, output_dir, frame_dir, crop_cache=crop_cache,
    )
    _infer_elapsed = time.time() - _t_infer

    _save_corner_samples(ctx, detect_out.detection_rows, output_dir)

    novelty_out = novelty.run(ctx, detect_out)
    track_out = track.run(ctx, novelty_out)

    # CRITICAL: Prune crop cache to only keep detections that survived tracking.
    # In long 4K videos, YOLO can produce thousands of candidates, and keeping
    # them all in RAM will cause OOM.
    active_det_ids = set()
    for track_data in track_out.tracks_data:
        for cand in track_data.get("candidates", []):
            active_det_ids.add(cand["detection_id"])
    
    original_count = len(crop_cache)
    crop_cache = {k: v for k, v in crop_cache.items() if k in active_det_ids}
    pruned_count = original_count - len(crop_cache)
    if pruned_count > 0:
        print(f"[fused] Pruned {pruned_count} crops from cache (remaining: {len(crop_cache)})", flush=True)

    refine_out = refine.run(ctx, track_out, decoded_crops=crop_cache)

    # Surface fused-path telemetry alongside refine's own op breakdown so the
    # handler diagnostic still shows where time went.
    existing = getattr(refine_out, "refine_telemetry", None) or {}
    existing.update({
        "fused": True,
        "fused_inference_s": round(_infer_elapsed, 3),
        "crops_cached": len(crop_cache),
        "detect_telemetry": detect_out.detect_telemetry,
    })
    refine_out.refine_telemetry = existing  # type: ignore[attr-defined]
    return refine_out
