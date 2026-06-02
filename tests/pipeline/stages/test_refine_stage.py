"""Phase 4 — refine stage carries identity and writes in-memory crops."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.stages import refine


def _frame(idx, w=640, h=480):
    img = (np.random.RandomState(idx).rand(h, w, 3) * 255).astype(np.uint8)
    fs = MagicMock()
    fs.frame_index = idx
    fs.image = img
    fs.width = w
    fs.height = h
    fs.timestamp_ms = idx * 33
    return fs


def _track(instance_id, frame_indices):
    return {
        "instance_id": instance_id,
        "track_id": 1,
        "angle": "Unknown",
        "session_id": 0,
        "first_frame_index": frame_indices[0],
        "candidates": [
            {
                "detection_id": idx * 10 + 1,
                "frame_index": idx,
                "timestamp_ms": idx * 33,
                "width": 640,
                "height": 480,
                "corners": [(100.0, 100.0), (300.0, 100.0),
                            (300.0, 400.0), (100.0, 400.0)],
                "confidence": 0.9,
                "novelty_score": 1.0,
                "score_total": 0.7,
                "image_path": "",
                "triage_metrics": {},
            }
            for idx in frame_indices
        ],
    }


def test_refine_carries_identity_into_frame_entries():
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,  # disables laplacian scan for this test
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-aaaaaaaa", [5, 10, 15])],
        "detections": [{"detection_id": idx * 10 + 1, "width": 640, "height": 480,
                        "novelty_score": 1.0, "triage_metrics": {}}
                       for idx in (5, 10, 15)],
        "video_id": 42,
        "db_path": "/tmp/x.sqlite",  # ignored when no telemetry writes
    }
    refine.run(state, telemetry=MagicMock())

    assert "refined_tracks" in state
    refined = state["refined_tracks"]
    assert len(refined) == 1
    track = refined[0]
    assert track["instance_id"] == "inst-aaaaaaaa"
    assert "frame_entries" in track
    assert len(track["frame_entries"]) >= 1
    for fe in track["frame_entries"]:
        assert isinstance(fe["normalized"], np.ndarray)
        assert fe["normalized"].shape == (1050, 750, 3)
        assert isinstance(fe["visual_hash"], str)
        assert "quality_score" in fe and fe["quality_score"] >= 0.0


def test_refine_assigns_best_canonical_image():
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-bbbbbbbb", [5, 10, 15])],
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }
    refine.run(state, telemetry=MagicMock())
    refined = state["refined_tracks"][0]
    assert isinstance(refined["best_canonical_image"], np.ndarray)
    assert refined["best_canonical_image"].shape == (1050, 750, 3)
    assert isinstance(refined["best_canonical_detection_id"], int)


def test_refine_attaches_reid_embedding_when_embedder_available(monkeypatch):
    """When DinoEmbedder is constructible, refine populates reid_embedding."""
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-cccccccc", [5, 10, 15])],
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }

    # Patch DinoEmbedder so we don't need real weights
    import card_capture.stages.refine

    class _StubEmbedder:
        def embed_array(self, img):
            import torch
            return torch.tensor([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(refine, "_get_embedder", lambda: _StubEmbedder())
    refine.run(state, telemetry=MagicMock())
    assert state["refined_tracks"][0]["reid_embedding"] == pytest.approx([0.1, 0.2, 0.3])


def test_refine_records_track_telemetry_rows(tmp_path, monkeypatch):
    """refine should call CardsRepository.add_track_telemetry for each canonical."""
    request = MagicMock()
    request.config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0,
        "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1,
    }
    captured = []

    class _StubRepo:
        def add_track_telemetry(self, **kw):
            captured.append(kw)

    state = {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, 20)],
        "tracks_data": [_track("inst-dddddddd", [5, 10, 15])],
        "detections": [],
        "video_id": 7,
        "db_path": str(tmp_path / "cards.sqlite"),
        "repos": {"cards": _StubRepo()},
    }
    refine.run(state, telemetry=MagicMock())
    assert any(row["video_id"] == 7 and row["instance_id"] == "inst-dddddddd"
               for row in captured)
