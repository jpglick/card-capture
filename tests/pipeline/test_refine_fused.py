# tests/pipeline/test_refine_fused.py
"""Fused-mode refine: when given a pre-warped crop cache, it must not decode."""
import numpy as np
import pytest
from typing import Any, Dict, List, Optional, Tuple


def _make_ctx(tmp_path):
    from pipeline.steps.start import RunContext
    crops = tmp_path / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    return RunContext(
        video_path="/nonexistent/video.MOV",
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"),
        detector="cuda",
        config_preset="balanced",
        crops_dir=str(crops),
        frame_dir=str(tmp_path / "frames"),
        use_kornia=False,          # force CPU-safe path; cache is pre-warped anyway
        corner_refinement=False,
        video_id=1,
    )


def _make_track_out():
    from pipeline.steps.track import TrackOutput
    corners = [[0.0, 0.0], [100.0, 0.0], [100.0, 140.0], [0.0, 140.0]]
    candidate = {
        "detection_id": 0,
        "frame_index": 5,
        "timestamp_ms": 166,
        "image_path": "",
        "confidence": 0.9,
        "score_total": 0.9,
        "score_components": {},
        "corners": corners,
    }
    track = {
        "instance_id": "inst-aaaaaaaa",
        "track_id": 1,
        "angle": 0.0,
        "candidate_detection_ids": [0],
        "first_frame_index": 5,
        "session_id": 1,
        "candidates": [candidate],
        "reid_embedding": None,
    }
    det_row = {
        "detection_id": 0, "frame_index": 5, "timestamp_ms": 166,
        "width": 3840, "height": 2160, "corners": corners,
        "confidence": 0.9, "source_frame_path": "", "triage_metrics": {},
        "novelty_score": 1.0,
    }
    return TrackOutput(
        tracks_data=[track],
        frame_to_session={"5": 1},
        tracker_events=[],
        detection_rows=[det_row],
        sampler_telemetry={"last_source_fps": 30.0},
        bg_model_path=None,
        accepted_frame_presence=[(5, 166, True)],
        frame_count=1,
        accepted_frame_count=1,
        video_id=1,
    )


def test_refine_fused_does_not_decode(tmp_path, monkeypatch):
    from pipeline.steps import refine
    from card_capture.storage import Storage
    # Ensure database directory exists
    (tmp_path / "cards.sqlite").parent.mkdir(parents=True, exist_ok=True)
    Storage(tmp_path / "cards.sqlite").initialize()

    # decode_frames_gpu must never be called in fused mode.
    import card_capture.pipeline_utils as pu
    monkeypatch.setattr(
        pu, "decode_frames_gpu",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decode_frames_gpu called in fused mode")),
    )

    # A pre-warped 750x1050 BGR crop keyed by detection_id.
    crop = np.random.randint(0, 256, (1050, 750, 3), dtype=np.uint8)
    decoded_crops = {0: crop}

    ctx = _make_ctx(tmp_path)
    track_out = _make_track_out()

    # This should fail initially because run() doesn't accept decoded_crops
    out = refine.run(ctx, track_out, decoded_crops=decoded_crops)

    assert len(out.refined_tracks) == 1
    entries = out.refined_tracks[0]["frame_entries"]
    assert len(entries) == 1
    # The cached crop was scored and written to disk (not re-warped from video).
    assert entries[0]["image_path"].endswith("_rectified.jpg")
