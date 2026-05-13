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
    """Select the best canonical frame for a track.

    Args:
        ctx:            RunContext from the start step.
        prepared_track: Single track dict from the resolve step.

    Returns:
        ``FuseOutput`` containing the fused image path and metadata.
    """
    # In v4.1, we just use the best canonical image path
    best_path = prepared_track["best_canonical_image_path"]
    
    # We'll create a new "fused" path for consistency, though it's just a copy
    # or we can just point to the same file. 
    # For now, let's just copy it to 'fused_...' to follow the pattern.
    import shutil
    from pathlib import Path
    
    fused_dir = Path(ctx.crops_dir)
    instance_id = prepared_track["instance_id"]
    fused_path = fused_dir / f"instance_{instance_id[:8]}_fused.jpg"
    
    if Path(best_path).exists():
        shutil.copy(best_path, fused_path)
    
    return FuseOutput(
        fused_canonical={
            "instance_id": instance_id,
            "session_id": prepared_track.get("session_id", 0),
            "angle": prepared_track["angle"],
            "fused_image_path": str(fused_path),
            "primary_hash": str(prepared_track["frame_entries"][0]["visual_hash"]), # simplified
            "quality_score": prepared_track["frame_entries"][0]["quality_score"], # simplified
            "side_score": prepared_track.get("side_score", 0.0),
            "appearance_vector": prepared_track.get("appearance_vector", []),
            "best_canonical_detection_id": prepared_track["best_canonical_detection_id"],
            "duplicate_track_index": prepared_track.get("duplicate_track_index"),
            "first_frame_index": prepared_track.get("first_frame_index", -1),
        }
    )
