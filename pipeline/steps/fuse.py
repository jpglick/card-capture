"""Step 7 — fuse: fuse multiple canonical crops into one high-quality fused image.

Note: In v4.1, multi-frame median fusion is disabled due to ghosting issues.
This step currently just returns the single best canonical frame as the 'fused'
image, preserving the pipeline architecture for future fusion improvements.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict
from .start import RunContext

@dataclass
class FuseOutput:
    """Outputs of the fuse step."""
    fused_canonical: Dict[str, Any]

def run(ctx: RunContext, prepared_track: Dict[str, Any]) -> FuseOutput:
    """Fuse multiple canonical frames for a track.

    Args:
        ctx:            RunContext from the start step.
        prepared_track: Single track dict from the resolve step.

    Returns:
        ``FuseOutput`` containing the fused image path and metadata.
    """
    import cv2
    import numpy as np
    from pathlib import Path
    from card_capture.fuser import MultiFrameFuser
    
    instance_id = prepared_track["instance_id"]
    frame_entries = prepared_track.get("frame_entries", [])
    canonical_entries = [fe for fe in frame_entries if fe.get("is_canonical")]
    
    if not canonical_entries:
        # Fallback to best canonical image if no entries marked as canonical
        best_path = prepared_track.get("best_canonical_image_path")
        if not best_path and frame_entries:
            best_path = frame_entries[0]["image_path"]
        
        return _return_single_frame(ctx, instance_id, best_path, prepared_track)

    # Load images for fusion
    images = []
    for fe in canonical_entries:
        p = fe["image_path"]
        if Path(p).exists():
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
    
    if not images:
        best_path = prepared_track.get("best_canonical_image_path", "")
        return _return_single_frame(ctx, instance_id, best_path, prepared_track)

    if len(images) == 1 or ctx.fusion_target_frames <= 1:
        return _return_single_frame(ctx, instance_id, canonical_entries[0]["image_path"], prepared_track)

    # Perform multi-frame fusion
    fuser = MultiFrameFuser()
    try:
        # We pass the foil_threshold from context if enabled
        foil_t = ctx.foil_threshold if ctx.enable_foil_aware_fusion else None
        fused_img = fuser.fuse(images, foil_threshold=foil_t)
    except Exception as e:
        print(f"Fusion failed for {instance_id}: {e}")
        return _return_single_frame(ctx, instance_id, canonical_entries[0]["image_path"], prepared_track)

    fused_dir = Path(ctx.crops_dir)
    fused_path = fused_dir / f"instance_{instance_id[:8]}_fused.jpg"
    cv2.imwrite(str(fused_path), fused_img)
    
    return FuseOutput(
        fused_canonical={
            "instance_id": instance_id,
            "session_id": prepared_track.get("session_id", 0),
            "angle": prepared_track["angle"],
            "fused_image_path": str(fused_path),
            "primary_hash": str(prepared_track["frame_entries"][0]["visual_hash"]), # Matches monolith
            "quality_score": prepared_track["frame_entries"][0]["quality_score"], # simplified


            "side_score": prepared_track.get("side_score", 0.0),
            "appearance_vector": prepared_track.get("appearance_vector", []),
            "best_canonical_detection_id": prepared_track["best_canonical_detection_id"],
            "duplicate_track_index": prepared_track.get("duplicate_track_index"),
            "first_frame_index": prepared_track.get("first_frame_index", -1),
            "reid_embedding": prepared_track.get("reid_embedding"),
        }
    )

def _return_single_frame(ctx: RunContext, instance_id: str, best_path: str, prepared_track: Dict[str, Any]) -> FuseOutput:
    import shutil
    from pathlib import Path
    
    fused_dir = Path(ctx.crops_dir)
    fused_path = fused_dir / f"instance_{instance_id[:8]}_fused.jpg"
    
    if best_path and Path(best_path).exists():
        shutil.copy(best_path, fused_path)
    
    return FuseOutput(
        fused_canonical={
            "instance_id": instance_id,
            "session_id": prepared_track.get("session_id", 0),
            "angle": prepared_track["angle"],
            "fused_image_path": str(fused_path),
            "primary_hash": str(prepared_track["frame_entries"][0]["visual_hash"]) if prepared_track.get("frame_entries") else "",
            "quality_score": prepared_track["frame_entries"][0]["quality_score"] if prepared_track.get("frame_entries") else 0.0,
            "side_score": prepared_track.get("side_score", 0.0),
            "appearance_vector": prepared_track.get("appearance_vector", []),
            "best_canonical_detection_id": prepared_track.get("best_canonical_detection_id"),
            "duplicate_track_index": prepared_track.get("duplicate_track_index"),
            "first_frame_index": prepared_track.get("first_frame_index", -1),
            "reid_embedding": prepared_track.get("reid_embedding"),
        }
    )
