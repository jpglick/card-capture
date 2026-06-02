"""Phase 7 — fuse stage produces one fused_canonical per prepared track."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from card_capture.stages import fuse


def _prepared_track(instance_id, n_canonical=4):
    canonical = (np.random.RandomState(0).rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "side_score": 0.7,
        "appearance_vector": [0.1, 0.2, 0.3],
        "best_canonical_detection_id": 1,
        "duplicate_track_index": None,
        "first_frame_index": 5,
        "reid_embedding": [0.5, 0.5, 0.0, 0.0],
        "best_canonical_image": canonical,
        "frame_entries": [
            {"normalized": canonical, "is_canonical": True,
             "visual_hash": "abcd", "quality_score": 0.8,
             "image_path": ""}
            for _ in range(n_canonical)
        ],
    }


def test_fuse_emits_one_record_per_prepared_track():
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": True,
                      "fusion_target_frames": 4}
    state = {
        "request": request,
        "prepared_tracks": [_prepared_track("a"), _prepared_track("b")],
    }
    fuse.run(state, telemetry=MagicMock())
    assert len(state["fused_canonicals"]) == 2
    for fc in state["fused_canonicals"]:
        assert isinstance(fc["fused_image"], np.ndarray)
        assert fc["fused_image"].shape == (1050, 750, 3)
        assert fc["primary_hash"] == "abcd"


def test_fuse_single_frame_passthrough_when_target_is_one():
    """fusion_target_frames=1 → fused_image == best_canonical_image."""
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": True,
                      "fusion_target_frames": 1}
    track = _prepared_track("c", n_canonical=4)
    state = {"request": request, "prepared_tracks": [track]}
    fuse.run(state, telemetry=MagicMock())
    fc = state["fused_canonicals"][0]
    assert np.array_equal(fc["fused_image"], track["best_canonical_image"])


def test_fuse_passes_foil_threshold_when_enabled():
    """enable_foil_aware_fusion=True → MultiFrameFuser.fuse called with foil_threshold=50.0."""
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": True,
                      "fusion_target_frames": 4}
    captured = {}

    class _StubFuser:
        def fuse(self, images, foil_threshold=None):
            captured["foil_threshold"] = foil_threshold
            return images[0]

    with patch("card_capture.stages.fuse.fuser.MultiFrameFuser", _StubFuser):
        state = {"request": request, "prepared_tracks": [_prepared_track("d")]}
        fuse.run(state, telemetry=MagicMock())

    assert captured["foil_threshold"] == 50.0


def test_fuse_skips_foil_when_disabled():
    request = MagicMock()
    request.config = {"foil_threshold": 50.0, "enable_foil_aware_fusion": False,
                      "fusion_target_frames": 4}
    captured = {}

    class _StubFuser:
        def fuse(self, images, foil_threshold=None):
            captured["foil_threshold"] = foil_threshold
            return images[0]

    with patch("card_capture.stages.fuse.fuser.MultiFrameFuser", _StubFuser):
        state = {"request": request, "prepared_tracks": [_prepared_track("e")]}
        fuse.run(state, telemetry=MagicMock())

    assert captured["foil_threshold"] is None
