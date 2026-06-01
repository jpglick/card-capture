"""Phase 8 — dedup stage groups duplicate instances within + across runs."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import dedup as dedup_stage


def _fused(instance_id, embedding=None, primary_hash="0" * 16):
    return {
        "instance_id": instance_id,
        "session_id": 0,
        "angle": "Front",
        "primary_hash": primary_hash,
        "reid_embedding": list(embedding) if embedding is not None else None,
    }


def test_dedup_intra_run_groups_by_close_embedding():
    request = MagicMock()
    e_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_b = e_a + np.array([0.01, 0.01, 0.0, 0.0], dtype=np.float32)
    e_b = e_b / np.linalg.norm(e_b)
    state = {
        "request": request,
        "fused_canonicals": [_fused("a", e_a), _fused("b", e_b)],
        "video_id": 1,
        "repos": {"cards": _StubRepoNoCrossVideo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    assert len(state["dedup_groups"]) == 1
    g = state["dedup_groups"][0]
    assert g["canonical_instance_id"] == "a"
    assert "b" in g["duplicate_instance_ids"]


def test_dedup_intra_run_groups_by_phash_when_embedding_missing():
    request = MagicMock()
    state = {
        "request": request,
        "fused_canonicals": [
            _fused("p", embedding=None, primary_hash="ffffffffffffffff"),
            _fused("q", embedding=None, primary_hash="ffffffffffffffff"),
        ],
        "video_id": 1,
        "repos": {"cards": _StubRepoNoCrossVideo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    assert len(state["dedup_groups"]) == 1
    g = state["dedup_groups"][0]
    assert "q" in g["duplicate_instance_ids"]


def test_dedup_cross_video_query_excludes_self_video_id():
    """The CardsRepository must be called with video_id=current, not zero."""
    request = MagicMock()
    captured = {}

    class _StubRepo:
        def find_embeddings_excluding_video(self, *, video_id):
            captured["video_id"] = video_id
            return []  # no cross-video matches

    state = {
        "request": request,
        "fused_canonicals": [_fused("x", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))],
        "video_id": 42,
        "repos": {"cards": _StubRepo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    assert captured["video_id"] == 42


def test_dedup_cross_video_match_sets_parent_id():
    request = MagicMock()
    e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    class _StubRepo:
        def find_embeddings_excluding_video(self, *, video_id):
            # Existing card_instance row id=99 with a very close embedding
            return [(99, e.tobytes())]

    state = {
        "request": request,
        "fused_canonicals": [_fused("new", e)],
        "video_id": 7,
        "repos": {"cards": _StubRepo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    g = state["dedup_groups"][0]
    assert g["cross_video_parent_id"] == 99


class _StubRepoNoCrossVideo:
    def find_embeddings_excluding_video(self, *, video_id):
        return []


def test_intra_run_visual_duplicates_remain_in_final_cards():
    request = MagicMock()
    # Identical embeddings -> should be grouped
    e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state = {
        "request": request,
        "fused_canonicals": [
            _fused("physical-a", e),
            _fused("physical-b", e),
        ],
        "video_id": 1,
        "repos": {"cards": _StubRepoNoCrossVideo()},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    
    # Check they are grouped
    assert len(state["dedup_groups"]) == 1
    assert "physical-b" in state["dedup_groups"][0]["duplicate_instance_ids"]
    
    # Check they BOTH remain in final_cards (the invariant we are testing)
    instance_ids = [card["instance_id"] for card in state["final_cards"]]
    assert "physical-a" in instance_ids
    assert "physical-b" in instance_ids
