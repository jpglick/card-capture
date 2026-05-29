"""Stage 6: GPU Refinement (Kornia perspective warp -> 750x1050).

CRITICAL V5.5 CHANGE: reads frames from state["sampled_frames"], NOT from
disk. The V4 code re-decoded the source video here; V5.5 must never do
that. If state["sampled_frames"] is missing, that is a contract violation,
not a fallback path.
"""
from __future__ import annotations

from card_capture.gpu_refinement import KorniaNormalizer


def run(state: dict, *, telemetry) -> None:
    frames = state.get("sampled_frames")
    if frames is None:
        telemetry.contract_violation(
            "refine_without_frames",
            {"hint": "sample stage must populate state['sampled_frames']"},
        )
        raise RuntimeError("refine stage reached without sampled_frames in state")
        
    config = state["request"].config
    normalizer = KorniaNormalizer(
        width=config.get("detection_width", 750), # Canonical width
        height=config.get("detection_height", 1050),
        device=config.get("device", "auto")
    )
    
    # Prepare batch data: list of (image, corners)
    batch_data = []
    # In a real implementation we would extract the best candidate per track.
    # For this stub, we just process all novelty_scored_detections.
    for det in state.get("novelty_scored_detections", []):
        frame_idx = det["frame_index"]
        corners = det["corners"]
        frame = next((f.image for f in frames if f.frame_index == frame_idx), None)
        if frame is not None:
            batch_data.append((frame, corners))
            
    if batch_data:
        crops = normalizer.warp_canonical_batch(batch_data)
    else:
        crops = []
        
    state["crops"] = crops
