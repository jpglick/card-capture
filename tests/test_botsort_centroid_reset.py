from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class _FakeBoTSORT:
    def __init__(self, **kwargs):
        self.active_tracks = []

    def update(self, det_input, img=None, embs=None):
        self.active_tracks = [SimpleNamespace(track_id=1, smooth_feat=np.ones(4, dtype=np.float32))]
        if len(det_input) == 0:
            return np.empty((0, 8), dtype=np.float32)
        d = det_input[0]
        return np.asarray([[d[0], d[1], d[2], d[3], 1, d[4], d[5], 0.0]], dtype=np.float32)


def _cand(det_id: int, frame_index: int, x: int, y: int):
    return {
        "detection_id": det_id,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 33,
        "width": 1000,
        "height": 100,
        "corners": [(x, y), (x + 40, y), (x + 40, y + 40), (x, y + 40)],
        "confidence": 0.9,
        "novelty_score": 1.0,
        "triage_metrics": {},
    }


def _frame(frame_index: int):
    return SimpleNamespace(
        frame_index=frame_index,
        image=np.zeros((100, 1000, 3), dtype=np.uint8),
        width=1000,
        height=100,
        timestamp_ms=frame_index * 33,
    )


def test_centroid_jump_is_telemetry_only():
    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch("card_capture.tracking.botsort_adapter._import_botsort", return_value=_FakeBoTSORT):
            with patch("card_capture.tracking.botsort_adapter._get_shared_embedder", return_value=None):
                from card_capture.tracking.botsort_adapter import BoTSORTAdapter

                adapter = BoTSORTAdapter(min_track_length=1, centroid_jump_ratio=0.30, centroid_jump_frames=3)
                tracks = adapter.assign(
                    [_cand(1, 0, 10, 10), _cand(2, 1, 560, 10)],
                    [_frame(0), _frame(1)],
                )

    assert adapter.centroid_jump_count == 1
    assert adapter.last_reset_count == 0
    session_ids = sorted({t.session_id for t in tracks})
    assert session_ids == [0]
