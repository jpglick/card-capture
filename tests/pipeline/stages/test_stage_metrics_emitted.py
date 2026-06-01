from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from card_capture.core.models import QualityScore, ScoredCandidate, TrackState
from card_capture.stages import dedup, fuse, resolve, score, refine, track, novelty, detect
from card_capture.stages import sample


class _EventsRepo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_stage_metrics(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _Producer:
    def __init__(self, frames) -> None:
        self._frames = list(frames)

    def __iter__(self):
        return iter(self._frames)

    def stop(self) -> None:
        return None


def test_detect_emits_stage_metrics(monkeypatch):
    events = _EventsRepo()
    frame = SimpleNamespace(
        frame_index=0,
        timestamp_ms=0,
        image=np.zeros((20, 30, 3), dtype=np.uint8),
        width=30,
        height=20,
    )

    class _Detector:
        def detect_batch(self, packets, conf):
            pkt = packets[0]
            return [
                SimpleNamespace(
                    frame_index=pkt.frame_index,
                    timestamp_ms=pkt.timestamp_ms,
                    width=pkt.width,
                    height=pkt.height,
                    corner_detection=SimpleNamespace(
                        corners=[(1.0, 1.0), (10.0, 1.0), (10.0, 10.0), (1.0, 10.0)],
                        confidence=0.9,
                    ),
                )
            ]

    monkeypatch.setattr(detect, "FakeCardDetector", _Detector)
    state = {
        "request": SimpleNamespace(run_id="r1", config={"detector": "fake"}),
        "frame_producer": _Producer([frame]),
        "sampled_frames": [],
        "estimated_frame_total": 1,
        "repos": {"events": events},
        "video_id": 1,
    }
    detect.run(state, telemetry=SimpleNamespace(progress=lambda *a, **k: None, resource_sample=lambda *a, **k: None))
    assert events.calls[-1]["stage"] == "detect"
    assert events.calls[-1]["metrics"]["detections"] == 1


def test_novelty_emits_stage_metrics():
    events = _EventsRepo()
    state = {
        "request": SimpleNamespace(run_id="r1"),
        "detections": [{"detection_id": 1, "frame_index": 0, "corners": [], "novelty_score": 1.0}],
        "sampled_frames": [],
        "repos": {"events": events},
        "video_id": 1,
    }
    novelty.run(state, telemetry=SimpleNamespace())
    assert events.calls[-1]["stage"] == "novelty"
    assert events.calls[-1]["metrics"]["scored"] == 1


def test_track_emits_stage_metrics(monkeypatch):
    events = _EventsRepo()

    class _Tracker:
        def __init__(self, **kwargs):
            self.sessionization_metrics = {
                "appearance_raw_jumps": 4,
                "appearance_plateaus_confirmed": 3,
                "appearance_bridge_plateaus_suppressed": 1,
                "appearance_presentations_retained": 2,
                "appearance_boundary_frames": [20],
            }

        def assign(self, detections, frames):
            cand = ScoredCandidate(
                detection_id=1,
                timestamp_ms=0,
                image_path="",
                score=QualityScore(total=0.9, components={}),
                corners=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
                frame_index=0,
            )
            return [TrackState(instance_id="i1", candidates=[cand], session_id=0)]

    monkeypatch.setattr(track, "BoTSORTAdapter", _Tracker)
    monkeypatch.setattr(track, "ByteTrackAdapter", _Tracker)
    state = {
        "request": SimpleNamespace(run_id="r1", config={"tracker_backend": "botsort", "min_track_length": 1}),
        "sampled_frames": [],
        "novelty_scored_detections": [{
            "detection_id": 1,
            "frame_index": 0,
            "timestamp_ms": 0,
            "width": 10,
            "height": 10,
            "corners": [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            "confidence": 0.9,
            "novelty_score": 1.0,
            "triage_metrics": {},
        }],
        "repos": {"events": events},
        "video_id": 1,
    }
    track.run(state, telemetry=SimpleNamespace())
    assert events.calls[-1]["stage"] == "track"
    metrics = events.calls[-1]["metrics"]
    assert metrics["tracks_final"] == 1
    assert metrics["tracks_data"] == 1
    assert metrics["appearance_raw_jumps"] == 4
    assert metrics["appearance_plateaus_confirmed"] == 3


def test_all_stage_modules_call_emit_stage_metrics():
    from card_capture.stages import store

    expected = {
        sample: 'stage="sample"',
        detect: 'stage="detect"',
        novelty: 'stage="novelty"',
        track: 'stage="track"',
        refine: 'stage="refine"',
        score: 'stage="score"',
        resolve: 'stage="resolve"',
        fuse: 'stage="fuse"',
        dedup: 'stage="dedup"',
        store: 'stage="store"',
    }
    for module, stage_literal in expected.items():
        src = inspect.getsource(module)
        assert "emit_stage_metrics(" in src
        assert stage_literal in src

