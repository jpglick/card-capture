"""Refine memory-safety: on a 16GB box the full 4K frame buffer stays resident
while MPS warps allocate on top, so refine could trip an OS memory-pressure
SIGKILL (the run then lingers as 'running' forever). These tests pin the real
fix:

  1. abort on a low *available-physical-memory* floor (what jetsam acts on),
     not a virtual_memory().percent ceiling that reads ~82% at kill time;
  2. shrink the per-warp spike via a smaller, configurable chunk_size;
  3. release each decoded frame as soon as the last track using it is done,
     and prune untracked frames up front, so the resident buffer shrinks
     across the loop instead of staying ~GB resident the whole time.
"""
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.stages import refine
from card_capture.stages.refine.gpu_refinement import KorniaNormalizer


def _frame(idx, w=320, h=240):
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
                "width": 320,
                "height": 240,
                "corners": [(50.0, 50.0), (250.0, 50.0), (250.0, 200.0), (50.0, 200.0)],
                "confidence": 0.9,
                "novelty_score": 1.0,
                "score_total": 0.7,
                "image_path": "",
                "triage_metrics": {},
            }
            for idx in frame_indices
        ],
    }


def _state(tracks=None, n_frames=20, extra_config=None):
    request = MagicMock()
    config = {
        "device": "cpu", "detection_width": 750, "detection_height": 1050,
        "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0, "max_corner_gap_frames": 30,
        "corner_refinement": False, "fusion_target_frames": 1,
        # Gate off by default so memory-agnostic tests never abort on a loaded
        # CI box; the two gate tests set this floor explicitly.
        "refine_min_available_mb": 0.0,
    }
    if extra_config:
        config.update(extra_config)
    request.config = config
    return {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(0, n_frames)],
        "tracks_data": tracks if tracks is not None else [_track("inst-aaaaaaaa", [5, 10, 15])],
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }


class _CapturingTelemetry:
    """Records resource_sample payloads; no-ops every other telemetry call."""

    def __init__(self):
        self.samples = []

    def resource_sample(self, payload):
        self.samples.append(payload)

    def __getattr__(self, _name):
        return lambda *a, **k: None


# --- (3) resident-buffer release ------------------------------------------

def test_last_track_using_frame_maps_each_frame_to_its_final_track():
    tracks = [_track("a", [2, 4]), _track("b", [4, 6])]
    assert refine._last_track_using_frame(tracks) == {2: 0, 4: 1, 6: 1}


def test_refine_releases_raw_frames_when_done():
    """refine is the last consumer of sampled_frames; it must drop them so the
    ~GB frame buffer is freed before the back-half stages run."""
    state = _state()
    refine.run(state, telemetry=MagicMock())
    assert not state.get("sampled_frames")


def test_refine_frees_frames_progressively_and_prunes_untracked():
    """Untracked frames are dropped up front; each track's frames are freed
    once it finishes, so the resident buffer shrinks monotonically to zero."""
    tracks = [
        _track("a", [2, 4]),
        _track("b", [6, 8]),
        _track("c", [10, 12]),
    ]
    state = _state(tracks=tracks, n_frames=20)
    tele = _CapturingTelemetry()
    refine.run(state, telemetry=tele)

    resident = [
        s["resident_frames"] for s in tele.samples
        if s.get("event") == "refine_frame_buffer"
    ]
    # 20 frames pruned to the 6 referenced; then 2 freed after each of 3 tracks.
    assert resident == [4, 2, 0]


# --- (1) available-memory abort gate --------------------------------------

def test_refine_aborts_when_available_memory_below_floor(monkeypatch):
    """When *available* physical memory is below the floor, refine raises a
    clean failure instead of letting the OS SIGKILL the process mid-warp."""
    monkeypatch.setattr(refine, "_available_memory_mb", lambda: 100.0)
    state = _state(extra_config={"refine_min_available_mb": 4096.0})
    with pytest.raises(RuntimeError, match="memory"):
        refine.run(state, telemetry=MagicMock())
    # Buffers released on the way out so the failure path frees memory.
    assert not state.get("sampled_frames")


def test_refine_proceeds_when_available_memory_ok(monkeypatch):
    monkeypatch.setattr(refine, "_available_memory_mb", lambda: 9000.0)
    state = _state(extra_config={"refine_min_available_mb": 4096.0})
    refine.run(state, telemetry=MagicMock())
    assert state.get("refined_tracks")


def test_available_memory_mb_returns_inf_when_psutil_unavailable(monkeypatch):
    """If memory can't be read, the gate must never abort spuriously."""
    import psutil

    def _boom():
        raise RuntimeError("no psutil")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    assert refine._available_memory_mb() == math.inf


# --- (2) configurable warp chunk size -------------------------------------

def test_refine_forwards_configured_warp_chunk_size(monkeypatch):
    captured = {}

    class _FakeKornia:
        def __init__(self, *a, **k):
            pass

        def warp_canonical_batch(self, batch_items, rotate_180=True, chunk_size=8):
            captured["chunk_size"] = chunk_size
            return [np.zeros((1050, 750, 3), dtype=np.uint8) for _ in batch_items]

        def release_cache(self):
            pass

    monkeypatch.setattr(refine, "KorniaNormalizer", _FakeKornia)
    state = _state(extra_config={"refine_warp_chunk_size": 3})
    refine.run(state, telemetry=MagicMock())
    assert captured["chunk_size"] == 3


# --- existing invariant ----------------------------------------------------

def test_kornia_normalizer_release_cache_is_safe_on_cpu():
    norm = KorniaNormalizer(width=750, height=1050, device="cpu")
    assert norm.release_cache() is None
