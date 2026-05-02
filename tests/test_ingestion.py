from __future__ import annotations

import numpy as np

from card_capture.ingestion import FrameTriageFilter, _resolve_reader_backend


def test_frame_triage_filter_rejects_empty_frame() -> None:
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    triage = FrameTriageFilter(empty_ratio_threshold=0.95, variance_threshold=1.0)

    accepted, metrics = triage.evaluate(frame)

    assert accepted is False
    assert metrics["empty_ratio"] >= 0.95
    assert set(metrics.keys()) == {"blur", "variance", "empty_ratio"}


def test_frame_triage_filter_accepts_textured_frame() -> None:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    triage = FrameTriageFilter(empty_ratio_threshold=0.95, variance_threshold=100.0)

    accepted, metrics = triage.evaluate(frame)

    assert accepted is True
    assert metrics["variance"] > 100.0
    assert set(metrics.keys()) == {"blur", "variance", "empty_ratio"}


def test_resolve_reader_backend_auto_falls_back_to_pyav_when_decord_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr("card_capture.ingestion._decord_available", lambda: False)

    backend = _resolve_reader_backend("auto")

    assert backend == "pyav"
