"""Phase 9 — store stage writes images + DB rows; produces final_cards."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import store as store_stage


def _fused(instance_id, image=None):
    if image is None:
        image = (np.random.RandomState(hash(instance_id) & 0xFFFFFFFF)
                 .rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "fused_image": image,
        "primary_hash": "deadbeef",
        "quality_score": 0.8,
        "side_score": 0.7,
        "appearance_vector": [],
        "best_canonical_detection_id": 1,
        "duplicate_track_index": None,
        "first_frame_index": 5,
        "reid_embedding": [0.5, 0.5, 0.0, 0.0],
    }


def _prepared(instance_id):
    img = (np.random.RandomState(0).rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "best_canonical_detection_id": 1,
        "frame_entries": [
            {
                "detection_id": 1,
                "frame_index": 5,
                "timestamp_ms": 165,
                "normalized": img,
                "image_path": "",
                "is_canonical": True,
                "quality_score": 0.8,
                "quality_components": {"sharpness": 0.7},
                "confidence": 0.9,
                "corners": [(0, 0), (750, 0), (750, 1050), (0, 1050)],
                "glare_x": None, "glare_y": None,
                "sharpness": 0.7,
            },
        ],
    }


class _StubRepo:
    def __init__(self):
        self.added_instances = []
        self.added_views = []
        self.added_saved = []
        self.fusion_updates = []
        self.dedup_updates = []
        self._next_id = 100

    def add_card_instance(self, **kw):
        rid = self._next_id
        self._next_id += 1
        self.added_instances.append((rid, kw))
        return rid

    def update_instance_deduplication(self, **kw):
        self.dedup_updates.append(kw)

    def update_instance_fusion(self, **kw):
        self.fusion_updates.append(kw)

    def add_card_view(self, **kw):
        vid = self._next_id
        self._next_id += 1
        self.added_views.append((vid, kw))
        return vid

    def add_saved_card(self, **kw):
        self.added_saved.append(kw)

    def add_track_telemetry(self, **kw):
        pass

    def add_pipeline_event(self, **kw):
        pass


def _state(tmp_path, instance_id="t-aaaaaaaa"):
    request = MagicMock()
    request.config = {}
    request.run_id = "r1"
    repos = {"cards": _StubRepo(), "runs": MagicMock()}
    return {
        "request": request,
        "video_id": 42,
        "output_root": tmp_path,
        "fused_canonicals": [_fused(instance_id)],
        "prepared_tracks": [_prepared(instance_id)],
        "dedup_groups": [{
            "canonical_instance_id": instance_id,
            "duplicate_instance_ids": [],
            "hamming_distances": {},
            "embedding_distances": {},
            "cross_video_parent_id": None,
        }],
        "repos": repos,
    }


def test_store_writes_fused_image_to_crops_dir(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    crops_dir = tmp_path / "crops"
    assert crops_dir.exists()
    fused_files = list(crops_dir.glob("instance_*_fused.jpg"))
    assert len(fused_files) == 1


def test_store_writes_rectified_jpeg_per_frame_entry(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    crops_dir = tmp_path / "crops"
    rectified = list(crops_dir.glob("track_*_det_*_rectified.jpg"))
    assert len(rectified) == 1


def test_store_calls_add_card_instance_with_run_id(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    repo = state["repos"]["cards"]
    assert len(repo.added_instances) == 1
    rid, kw = repo.added_instances[0]
    assert kw["run_id"] == "r1"
    assert kw["video_id"] == 42


def test_store_best_view_points_to_fused_path(tmp_path):
    """V4 line 98 (A1): canonical best view's rectified_path = fused_image_path."""
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    repo = state["repos"]["cards"]
    best_view_kw = [kw for _, kw in repo.added_views if kw["is_canonical"]][0]
    fusion_kw = repo.fusion_updates[0]
    assert best_view_kw["rectified_path"] == fusion_kw["fused_image_path"]


def test_store_populates_final_cards_in_state(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    assert len(state["cards"]) == 1
    assert state["cards"][0]["instance_id"] == "t-aaaaaaaa"


def test_store_marks_run_completed_with_card_count(tmp_path):
    state = _state(tmp_path)
    store_stage.run(state, telemetry=MagicMock())
    runs_repo = state["repos"]["runs"]
    runs_repo.mark_completed.assert_called_once_with("r1", cards_extracted=1)
