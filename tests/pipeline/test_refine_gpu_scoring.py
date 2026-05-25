# tests/pipeline/test_refine_gpu_scoring.py
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

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
        use_kornia=True,
        kornia_device="cpu", # Use CPU for tests
        corner_refinement=False,
        video_id=1,
    )

def _make_track_out():
    from pipeline.steps.track import TrackOutput
    corners = [[0.0, 0.0], [100.0, 0.0], [100.0, 140.0], [0.0, 140.0]]
    candidate0 = {
        "detection_id": 0,
        "frame_index": 5,
        "timestamp_ms": 166,
        "image_path": "",
        "confidence": 0.9,
        "score_total": 0.9,
        "score_components": {},
        "corners": corners,
    }
    candidate1 = {
        "detection_id": 1,
        "frame_index": 6,
        "timestamp_ms": 200,
        "image_path": "",
        "confidence": 0.8,
        "score_total": 0.8,
        "score_components": {},
        "corners": corners,
    }
    track = {
        "instance_id": "inst-aaaaaaaa",
        "track_id": 1,
        "angle": 0.0,
        "candidate_detection_ids": [0, 1],
        "first_frame_index": 5,
        "session_id": 1,
        "candidates": [candidate0, candidate1],
        "reid_embedding": None,
    }
    det_rows = [
        {
            "detection_id": 0, "frame_index": 5, "timestamp_ms": 166,
            "width": 3840, "height": 2160, "corners": corners,
            "confidence": 0.9, "source_frame_path": "", "triage_metrics": {},
            "novelty_score": 1.0,
        },
        {
            "detection_id": 1, "frame_index": 6, "timestamp_ms": 200,
            "width": 3840, "height": 2160, "corners": corners,
            "confidence": 0.8, "source_frame_path": "", "triage_metrics": {},
            "novelty_score": 1.0,
        }
    ]
    return TrackOutput(
        tracks_data=[track],
        frame_to_session={"5": 1, "6": 1},
        tracker_events=[],
        detection_rows=det_rows,
        sampler_telemetry={"last_source_fps": 30.0},
        bg_model_path=None,
        accepted_frame_presence=[(5, 166, True), (6, 200, True)],
        frame_count=2,
        accepted_frame_count=2,
        video_id=1,
    )

def test_refine_uses_batched_gpu_scoring(tmp_path, monkeypatch):
    from pipeline.steps import refine
    from card_capture.storage import Storage
    from card_capture.scoring import QualityScorer
    from card_capture.ml import gpu_ops

    # Ensure database directory exists
    (tmp_path / "cards.sqlite").parent.mkdir(parents=True, exist_ok=True)
    Storage(tmp_path / "cards.sqlite").initialize()

    # Create dummy GPU tensors (on CPU for test)
    # Refine expects (H, W, 3) BGR uint8
    crop0 = torch.zeros((1050, 750, 3), dtype=torch.uint8)
    crop1 = torch.ones((1050, 750, 3), dtype=torch.uint8)
    decoded_crops = {0: crop0, 1: crop1}

    ctx = _make_ctx(tmp_path)
    track_out = _make_track_out()

    # Spies
    score_batch_called = []
    original_score_batch = QualityScorer.score_batch
    def wrapped_score_batch(self, *args, **kwargs):
        score_batch_called.append((args, kwargs))
        return original_score_batch(self, *args, **kwargs)

    score_called = []
    original_score = QualityScorer.score
    def wrapped_score(self, *args, **kwargs):
        score_called.append((args, kwargs))
        return original_score(self, *args, **kwargs)

    # For gpu_ops, they are functions, so MagicMock(side_effect=...) is fine
    phash_batch_spy = MagicMock(side_effect=gpu_ops.phash_batch)
    glare_mask_batch_spy = MagicMock(side_effect=gpu_ops.glare_mask_batch)
    glare_centroid_batch_spy = MagicMock(side_effect=gpu_ops.glare_centroid_batch)

    monkeypatch.setattr(QualityScorer, "score_batch", wrapped_score_batch)
    monkeypatch.setattr(QualityScorer, "score", wrapped_score)
    monkeypatch.setattr(gpu_ops, "phash_batch", phash_batch_spy)
    monkeypatch.setattr(gpu_ops, "glare_mask_batch", glare_mask_batch_spy)
    monkeypatch.setattr(gpu_ops, "glare_centroid_batch", glare_centroid_batch_spy)

    # Run refine
    out = refine.run(ctx, track_out, decoded_crops=decoded_crops)

    # Assertions
    assert len(score_batch_called) > 0, "QualityScorer.score_batch should be called"
    assert phash_batch_spy.called, "gpu_ops.phash_batch should be called"
    assert glare_mask_batch_spy.called, "gpu_ops.glare_mask_batch should be called"
    assert glare_centroid_batch_spy.called, "gpu_ops.glare_centroid_batch should be called"
    
    # Crucially, per-image score should NOT be called for these candidates
    assert len(score_called) == 0, "QualityScorer.score should NOT be called in batched mode"

    assert len(out.refined_tracks) == 1
    entries = out.refined_tracks[0]["frame_entries"]
    assert len(entries) == 2
    for entry in entries:
        assert "quality_score" in entry
        assert "visual_hash" in entry
        assert "glare_x" in entry
        assert "glare_y" in entry
        assert "glare_mask" in entry
        assert entry["image_path"].endswith("_rectified.jpg")
        # Verify file was written
        assert (tmp_path / "crops" / entry["image_path"].split("/")[-1]).exists()
