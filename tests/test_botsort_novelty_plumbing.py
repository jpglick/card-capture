from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import card_capture.stages.track.botsort_adapter as botsort_mod


class _FakeBoTSORT:
    def __init__(self, **kwargs):
        self.active_tracks = []

    def update(self, det_input, img=None, embs=None):
        if len(det_input) == 0:
            return np.empty((0, 8), dtype=np.float32)
        self.active_tracks = [
            SimpleNamespace(track_id=1, smooth_feat=np.ones(4, dtype=np.float32))
        ]
        d = det_input[0]
        return np.asarray([[d[0], d[1], d[2], d[3], 1, d[4], d[5], 0.0]], dtype=np.float32)


def _det(det_id: int, frame_index: int, novelty: float) -> dict:
    return {
        "detection_id": det_id,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 33,
        "width": 1000,
        "height": 100,
        "corners": [(10, 10), (50, 10), (50, 50), (10, 50)],
        "confidence": 0.9,
        "novelty_score": novelty,
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


def test_equal_length_holder_bridge_is_suppressed_only_via_real_novelty():
    """A holder plateau that is the SAME length as the neighboring cards but has
    higher novelty must be suppressed.

    The duration gate cannot fire (equal lengths), so suppression depends
    entirely on per-detection novelty reaching the sessionizer. ScoredCandidate
    carries no novelty_score, so this passes only if assign() plumbs the
    detection dict's novelty_score through to the AppearanceObservation.
    """
    basis = np.eye(5, dtype=np.float32)
    card_a, holder, card_b, card_c, card_d = (basis[i] for i in range(5))

    # A, HOLDER, B, HOLDER, C, HOLDER, D — every plateau 5 frames long.
    layout = [
        (card_a, 0.10), (holder, 0.30), (card_b, 0.10),
        (holder, 0.30), (card_c, 0.10), (holder, 0.30), (card_d, 0.10),
    ]
    embeddings: list[np.ndarray] = []
    novelties: list[float] = []
    for emb, nov in layout:
        embeddings.extend([emb] * 5)
        novelties.extend([nov] * 5)

    detections = [_det(i, i, novelties[i]) for i in range(len(embeddings))]
    frames = [_frame(i) for i in range(len(embeddings))]

    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch.object(botsort_mod, "_import_botsort", return_value=_FakeBoTSORT):
            from card_capture.stages.track.botsort_adapter import BoTSORTAdapter

            adapter = BoTSORTAdapter(min_track_length=1)
            with patch.object(adapter, "_embed_candidates") as mock_embed:
                mock_embed.side_effect = [[emb] for emb in embeddings]
                tracks = adapter.assign(detections, frames)

    sessions = sorted({t.session_id for t in tracks})
    assert sessions == [0, 1, 2, 3]
    assert adapter.sessionization_metrics["appearance_bridge_plateaus_suppressed"] == 3
