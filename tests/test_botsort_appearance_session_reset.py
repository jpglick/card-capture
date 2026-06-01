from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from card_capture.core.models import ScoredCandidate, QualityScore
import card_capture.tracking.botsort_adapter as botsort_mod

class _FakeBoTSORT:
    def __init__(self, **kwargs):
        self.active_tracks = []

    def update(self, det_input, img=None, embs=None):
        if len(det_input) == 0:
            return np.empty((0, 8), dtype=np.float32)
        
        # Always return track_id 1 for simplicity in these tests
        self.active_tracks = [SimpleNamespace(track_id=1, smooth_feat=np.ones(4, dtype=np.float32))]
        d = det_input[0]
        return np.asarray([[d[0], d[1], d[2], d[3], 1, d[4], d[5], 0.0]], dtype=np.float32)

def _cand(det_id: int, frame_index: int, x: int = 10, y: int = 10):
    return ScoredCandidate(
        detection_id=str(det_id),
        frame_index=frame_index,
        timestamp_ms=frame_index * 33,
        image_path="",
        score=QualityScore(total=0.9, components={}),
        corners=[(x, y), (x + 40, y), (x + 40, y + 40), (x, y + 40)],
    )

def _frame(frame_index: int):
    return SimpleNamespace(
        frame_index=frame_index,
        image=np.zeros((100, 1000, 3), dtype=np.uint8),
        width=1000,
        height=100,
        timestamp_ms=frame_index * 33,
    )

def test_direct_appearance_replacement_resets_once_after_confirmation():
    """3 frames of A, 3 frames of B -> 2 sessions."""
    emb_a = np.array([1, 0, 0, 0], dtype=np.float32)
    emb_b = np.array([0, 1, 0, 0], dtype=np.float32)
    
    embeddings = [emb_a] * 3 + [emb_b] * 3
    
    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch.object(botsort_mod, "_import_botsort", return_value=_FakeBoTSORT):
            from card_capture.tracking.botsort_adapter import BoTSORTAdapter
            adapter = BoTSORTAdapter(min_track_length=1)
            
            with patch.object(adapter, "_embed_candidates") as mock_embed:
                mock_embed.side_effect = [[emb] for emb in embeddings]
                
                candidates = [_cand(i, i) for i in range(6)]
                frames = [_frame(i) for i in range(6)]
                
                tracks = adapter.assign(candidates, frames)
    
    session_ids = sorted([t.session_id for t in tracks])
    # Expectation: 2 sessions because of appearance change
    assert len(set(session_ids)) == 2
    assert session_ids == [0, 1]

def test_recurrent_holder_plateaus_are_not_emitted_as_sessions():
    """A-B-C-B-D-B-E where B is short (bridge) -> 4 sessions (B's suppressed)."""
    # Five mutually-orthogonal cards. emb_e must be distinct from the holder
    # (emb_b): a near-holder vector would land in the ambiguous transition band
    # and be dropped rather than forming a 5th card.
    emb_a = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    emb_b = np.array([0, 1, 0, 0, 0], dtype=np.float32)
    emb_c = np.array([0, 0, 1, 0, 0], dtype=np.float32)
    emb_d = np.array([0, 0, 0, 1, 0], dtype=np.float32)
    emb_e = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    
    # A(10), B(3), C(10), B(3), D(10), B(3), E(10)
    embeddings = ([emb_a] * 10 + [emb_b] * 3 + [emb_c] * 10 + [emb_b] * 3 + [emb_d] * 10 + [emb_b] * 3 + [emb_e] * 10)
    
    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch.object(botsort_mod, "_import_botsort", return_value=_FakeBoTSORT):
            from card_capture.tracking.botsort_adapter import BoTSORTAdapter
            adapter = BoTSORTAdapter(min_track_length=1)
            
            with patch.object(adapter, "_embed_candidates") as mock_embed:
                mock_embed.side_effect = [[emb] for emb in embeddings]
                
                candidates = [_cand(i, i) for i in range(len(embeddings))]
                frames = [_frame(i) for i in range(len(embeddings))]
                
                tracks = adapter.assign(candidates, frames)
    
    session_ids = [t.session_id for t in tracks]
    unique_sessions = sorted(set(session_ids))
    
    # Retained: A, C, D, E -> 4 sessions.
    # Suppressed: B (occurs 3 times, all interior, all neighbors different, shorter than neighbors).
    assert len(unique_sessions) == 4

def test_identical_fronts_in_distinct_plateaus_remain_distinct_sessions():
    """A-B-A where B is NOT a bridge -> 3 sessions."""
    emb_a = np.array([1, 0, 0, 0], dtype=np.float32)
    emb_b = np.array([0, 1, 0, 0], dtype=np.float32)
    
    # B is long enough or novelty is low enough to not be a bridge.
    # Let's just make it long.
    embeddings = [emb_a] * 10 + [emb_b] * 10 + [emb_a] * 10
    
    with patch.dict("sys.modules", {"supervision": SimpleNamespace(Detections=object)}):
        with patch.object(botsort_mod, "_import_botsort", return_value=_FakeBoTSORT):
            from card_capture.tracking.botsort_adapter import BoTSORTAdapter
            adapter = BoTSORTAdapter(min_track_length=1)
            
            with patch.object(adapter, "_embed_candidates") as mock_embed:
                mock_embed.side_effect = [[emb] for emb in embeddings]
                
                candidates = [_cand(i, i) for i in range(len(embeddings))]
                frames = [_frame(i) for i in range(len(embeddings))]
                
                tracks = adapter.assign(candidates, frames)
    
    session_ids = sorted(list(set([t.session_id for t in tracks])))
    assert session_ids == [0, 1, 2]
