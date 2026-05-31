from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from card_capture.models import QualityScore, ScoredCandidate
import card_capture.tracking.botsort_adapter as botsort_mod


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _StubEmbedder:
    dim = 4

    def embed_array(self, arr):
        return _FakeTensor(np.ones((1, self.dim), dtype=np.float32))


class _FakeBoTSORT:
    calls: list[dict] = []
    captured: dict[str, object] = {}

    def __init__(self, **kwargs):
        _FakeBoTSORT.calls.append(kwargs)
        self.active_tracks = []

    def update(self, det_input, img=None, embs=None):
        _FakeBoTSORT.captured = {
            "img_shape": tuple(img.shape) if img is not None else None,
            "embs": embs,
        }
        n = len(det_input)
        self.active_tracks = [
            SimpleNamespace(track_id=i + 1, smooth_feat=np.full(4, i + 1, dtype=np.float32))
            for i in range(n)
        ]
        rows = []
        for i, det in enumerate(det_input):
            rows.append([det[0], det[1], det[2], det[3], i + 1, det[4], det[5], 0.0])
        return np.asarray(rows, dtype=np.float32)


def _cand(det_id: int, frame_index: int, x: int, y: int, w: int = 20, h: int = 20):
    return {
        "detection_id": det_id,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 33,
        "width": 64,
        "height": 32,
        "corners": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        "confidence": 0.9,
        "novelty_score": 1.0,
        "triage_metrics": {},
    }


def _frame(frame_index: int):
    return SimpleNamespace(
        frame_index=frame_index,
        image=np.full((32, 64, 3), 50, dtype=np.uint8),
        width=64,
        height=32,
        timestamp_ms=frame_index * 33,
    )


def _make_adapter():
    with patch.object(botsort_mod, "_import_botsort", return_value=_FakeBoTSORT):
        return botsort_mod.BoTSORTAdapter(min_track_length=1)


def test_assign_passes_real_frame_and_embeddings_to_boxmot():
    adapter = _make_adapter()
    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch("card_capture.tracking.botsort_adapter._get_shared_embedder", return_value=_StubEmbedder()):
            tracks = adapter.assign(
                [_cand(1, 0, 2, 2), _cand(2, 0, 30, 2)],
                [_frame(0)],
            )
    assert tracks
    assert _FakeBoTSORT.captured["img_shape"] == (32, 64, 3)
    embs = _FakeBoTSORT.captured["embs"]
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 4)


def test_assign_locks_down_one_embedding_pass_per_frame():
    adapter = _make_adapter()
    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch("card_capture.tracking.botsort_adapter._get_shared_embedder", return_value=_StubEmbedder()):
            with patch.object(adapter, "_embed_candidates", wraps=adapter._embed_candidates) as mock_embed:
                tracks = adapter.assign(
                    [_cand(1, 0, 2, 2), _cand(2, 0, 30, 2), _cand(3, 1, 2, 2)],
                    [_frame(0), _frame(1)],
                )
    
    # Called once for frame 0, once for frame 1
    assert mock_embed.call_count == 2
    assert tracks
