import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.core.config import PipelineConfig
from card_capture.stages import refine
from card_capture.stages.refine.cropper import PrecisionNormalizer


def test_config_exposes_crop_knobs_in_request_dict():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    assert d["refine_crop_margin_px"] == 8
    assert d["refine_min_available_mb"] == 2048.0


def test_quad_bbox_clamps_to_frame_and_applies_margin():
    # quad inside a 200x300 (w x h) frame; margin expands, clamped at 0/edge.
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    assert refine._quad_bbox(corners, 200, 300, margin=8) == (30, 42, 168, 248)


def test_quad_bbox_clamps_when_quad_touches_edge():
    corners = [(2.0, 1.0), (199.0, 0.0), (198.0, 299.0), (1.0, 298.0)]
    assert refine._quad_bbox(corners, 200, 300, margin=8) == (0, 0, 200, 300)


def test_crop_and_rebase_returns_a_copy_not_a_view():
    frame = (np.random.RandomState(1).rand(300, 200, 3) * 255).astype(np.uint8)
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    crop, _ = refine._crop_and_rebase(frame, corners, margin=8)
    assert crop.base is None  # a real copy, so freeing `frame` releases its memory


def test_crop_and_rebase_warp_equivalence():
    """warp(full, corners) is pixel-identical to warp(crop, rebased corners)."""
    frame = (np.random.RandomState(0).rand(300, 200, 3) * 255).astype(np.uint8)
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    norm = PrecisionNormalizer()

    full = norm.normalize(frame, corners, rotate_180=False)
    crop, local = refine._crop_and_rebase(frame, corners, margin=8)
    cropped = norm.normalize(crop, local, rotate_180=False)

    assert np.array_equal(full, cropped)


def _candidate(detection_id, frame_index, score):
    return {
        "detection_id": detection_id,
        "frame_index": frame_index,
        "score_total": score,
        "corners": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    }


def test_plan_crop_candidates_keeps_top_n_per_track_keyed_by_detection():
    tracks = [
        {"candidates": [_candidate(1, 5, 0.9), _candidate(2, 6, 0.1), _candidate(3, 7, 0.5)]},
    ]
    wanted = refine._plan_crop_candidates(tracks, top_n=2)
    # Top-2 by score = detections 1 (0.9) and 3 (0.5); detection 2 dropped.
    assert set(wanted.keys()) == {1, 3}
    assert wanted[1] == (5, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])


def _frame(idx, w=200, h=300):
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
                "width": 200,
                "height": 300,
                "corners": [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)],
                "confidence": 0.9,
                "novelty_score": 1.0,
                "score_total": 0.7,
                "image_path": "",
                "triage_metrics": {},
            }
            for idx in frame_indices
        ],
    }


class _CapturingTelemetry:
    def __init__(self):
        self.samples = []

    def resource_sample(self, payload):
        self.samples.append(payload)

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _state(tracks, n_frames=20):
    request = MagicMock()
    request.config = {
        "device": "cpu", "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0, "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1, "refine_crop_margin_px": 8, "refine_min_available_mb": 0.0,
    }
    return {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(n_frames)],
        "tracks_data": tracks,
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }


def test_refine_frees_frames_in_prepass_and_still_refines():
    state = _state([_track("inst-a", [5, 10, 15])])
    tele = _CapturingTelemetry()
    refine.run(state, telemetry=tele)

    assert state.get("refined_tracks")
    assert state["refined_tracks"][0]["frame_entries"]
    assert not state.get("sampled_frames")  # raw buffer released
    cropped = [s for s in tele.samples if s.get("event") == "refine_cropped"]
    assert cropped and cropped[0]["frames_freed"] == 3  # frames 5, 10, 15


def test_available_memory_mb_returns_inf_when_psutil_unavailable(monkeypatch):
    import psutil

    def _boom():
        raise RuntimeError("no psutil")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    assert refine._available_memory_mb() == math.inf


def test_refine_aborts_cleanly_when_below_memory_floor(monkeypatch):
    monkeypatch.setattr(refine, "_available_memory_mb", lambda: 100.0)
    state = _state([_track("inst-a", [5, 10, 15])])
    state["request"].config["refine_min_available_mb"] = 4096.0
    with pytest.raises(RuntimeError, match="memory"):
        refine.run(state, telemetry=MagicMock())
    assert not state.get("sampled_frames")  # buffer released on the failure path
