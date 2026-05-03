from __future__ import annotations

import builtins

import numpy as np

from card_capture.ingestion import (
    FrameTriageFilter,
    _decord_available,
    _resolve_reader_backend,
)


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


def test_resolve_reader_backend_decord_returns_decord() -> None:
    assert _resolve_reader_backend("decord") == "decord"


def test_resolve_reader_backend_pyav_returns_pyav() -> None:
    assert _resolve_reader_backend("pyav") == "pyav"


def test_resolve_reader_backend_invalid_raises_value_error() -> None:
    try:
        _resolve_reader_backend("opencv")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_decord_available_returns_false_on_runtime_import_exception(monkeypatch) -> None:
    original_import = builtins.__import__

    def _import_with_decord_runtime_failure(name, *args, **kwargs):
        if name == "decord":
            raise RuntimeError("binary mismatch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_with_decord_runtime_failure)

    assert _decord_available() is False

def test_rolling_window_triage():
    from card_capture.ingestion import RollingWindowTriage
    triage = RollingWindowTriage(window_size=10, keep_percentile=0.5)
    scores = list(range(15))
    kept = []
    for i, score in enumerate(scores):
        if triage.evaluate_score(i, score):
            kept.append(i)
    assert kept == [9, 10, 11, 12, 13, 14]
