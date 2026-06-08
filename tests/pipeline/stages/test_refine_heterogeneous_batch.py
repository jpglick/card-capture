import numpy as np
import pytest
from unittest.mock import MagicMock
from card_capture.stages import refine

def _frame(idx, w=200, h=300):
    img = (np.random.RandomState(idx).rand(h, w, 3) * 255).astype(np.uint8)
    fs = MagicMock()
    fs.frame_index = idx
    fs.image = img
    fs.width = w
    fs.height = h
    fs.timestamp_ms = idx * 33
    return fs

def test_refine_handles_heterogeneous_crop_sizes_with_kornia():
    # Two candidates with completely different quad sizes
    # This will result in different crop shapes from _crop_and_rebase
    track = {
        "instance_id": "inst-a",
        "track_id": 1,
        "angle": "Unknown",
        "session_id": 0,
        "first_frame_index": 5,
        "candidates": [
            {
                "detection_id": 1,
                "frame_index": 5,
                "timestamp_ms": 165,
                "width": 200,
                "height": 300,
                "corners": [(10.0, 10.0), (50.0, 10.0), (50.0, 90.0), (10.0, 90.0)], # Small (40x80)
                "confidence": 0.9,
                "score_total": 0.9,
            },
            {
                "detection_id": 2,
                "frame_index": 6,
                "timestamp_ms": 198,
                "width": 200,
                "height": 300,
                "corners": [(20.0, 20.0), (150.0, 20.0), (150.0, 250.0), (20.0, 250.0)], # Large (130x230)
                "confidence": 0.8,
                "score_total": 0.8,
            }
        ]
    }
    
    request = MagicMock()
    # Force use_kornia=True to trigger the warp_canonical_batch path
    request.config = {
        "device": "cpu", "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0, "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1, "refine_crop_margin_px": 8, "refine_min_available_mb": 0.0,
    }
    
    state = {
        "request": request,
        "sampled_frames": [_frame(5), _frame(6)],
        "tracks_data": [track],
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }
    
    telemetry = MagicMock()
    
    # Should not raise ValueError from np.stack inside kornia batching
    refine.run(state, telemetry=telemetry)
    assert state.get("refined_tracks")
