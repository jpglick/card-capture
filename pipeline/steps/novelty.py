"""Step 2 — novelty: Stage 4 background novelty gate.

Scores every candidate against the workspace baseline and collects
novelty scores for adaptive-threshold computation.  Candidates are NOT
hard-dropped here (the guard is disabled in v4.1); the step adds a
``novelty_score`` field to each detection dict and persists the
background model to disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.steps.start import RunContext
from pipeline.steps.detect import DetectOutput


@dataclass
class NoveltyOutput:
    """Outputs of the novelty step."""

    detection_rows: List[Dict[str, Any]]  # same rows, novelty_score added
    sampler_telemetry: Dict[str, Any]
    bg_model_path: Optional[str]          # path to saved .npy background model
    accepted_frame_presence: List[Tuple[int, int, bool]]
    frame_count: int
    accepted_frame_count: int
    video_id: int


def run(ctx: RunContext, detect_out: DetectOutput) -> NoveltyOutput:
    """Score candidates against the workspace background model.

    Args:
        ctx:        RunContext from the start step.
        detect_out: Output from the detect step.

    Returns:
        ``NoveltyOutput`` with novelty scores added to each detection row.
    """
    import numpy as np
    from card_capture.presence.background_novelty import BackgroundModel, quad_novelty

    bg_paths = detect_out.sampler_telemetry.get("background_proxies_paths", [])
    bg_images = detect_out.sampler_telemetry.get("background_proxies", [])
    bg_model: Optional[BackgroundModel] = None
    bg_model_path: Optional[str] = None

    if bg_paths:
        bg_model = BackgroundModel.from_source_frame_paths(bg_paths, n=30)
    elif bg_images:
        bg_model = BackgroundModel.from_frames(bg_images)
    
    if bg_model is not None:
        # Persist the background model so downstream steps can use it
        bg_model_path = str(Path(ctx.output_dir) / "bg_model.npy")
        np.save(bg_model_path, bg_model.bgr if bg_model.bgr is not None else bg_model.gray)

    scored_rows: List[Dict[str, Any]] = []
    for row in detect_out.detection_rows:
        scored_row = dict(row)
        
        # Respect existing novelty_score if present (MPS fast path)
        if "novelty_score" in row:
            scored_rows.append(scored_row)
            ctx.observed_novelty_scores.append(float(row["novelty_score"]))
            continue

        scored_row["novelty_score"] = 1.0  # default: fully novel

        # The MPS cheap-score path never writes per-detection frames to disk
        # (source_frame_path is ""), so skip the imread — novelty stays inert
        # (score 1.0 = fully novel, nothing pruned) instead of spamming decode errors.
        src = row.get("source_frame_path")
        if bg_model is not None and row.get("corners") and src:
            import cv2
            img = cv2.imread(src)
            if img is not None:
                corners = [(float(x), float(y)) for x, y in row["corners"]]
                novelty_val = quad_novelty(
                    img, corners, bg_model,
                    color_space="lab", lab_weights=(1.0, 0.5, 0.5),
                )
                scored_row["novelty_score"] = float(novelty_val)
                # SOLE COLLECTION POINT: Collect novelty scores for adaptive thresholding
                ctx.observed_novelty_scores.append(float(novelty_val))

        scored_rows.append(scored_row)

    if bg_model is not None:
        print(
            f"[Stage: NoveltyGate] | scored {len(scored_rows)} candidates "
            "(hard novelty pruning disabled)"
        )
    else:
        print("[Stage: NoveltyGate] | skipped (no background proxies found)")

    return NoveltyOutput(
        detection_rows=scored_rows,
        sampler_telemetry=detect_out.sampler_telemetry,
        bg_model_path=bg_model_path,
        accepted_frame_presence=detect_out.accepted_frame_presence,
        frame_count=detect_out.frame_count,
        accepted_frame_count=detect_out.accepted_frame_count,
        video_id=detect_out.video_id,
    )
