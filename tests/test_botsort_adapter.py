import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from card_capture.selector import ScoredCandidate
from card_capture.models import QualityScore


def _candidate(detection_id, frame_index, x, y, conf=0.9, w=200, h=300):
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return ScoredCandidate(
        detection_id=detection_id,
        timestamp_ms=frame_index * 33,
        image_path="x.jpg",
        score=QualityScore(total=conf, components={"sharpness": conf, "blur": 0.0, "area": 0.5}),
        corners=corners,
        frame_index=frame_index,
    )


def _make_mock_botsort():
    """Create a mock BoT-SORT tracker that assigns sequential track IDs."""
    mock_tracker = MagicMock()
    call_count = [0]

    def mock_update(det, img=None):
        # Simulate: assign tracker_id 1 to all detections
        n = len(det.xyxy)
        det.tracker_id = np.ones(n, dtype=int)
        call_count[0] += 1

    mock_tracker.update_with_detections.side_effect = mock_update
    return mock_tracker, call_count


def _make_mock_detections_class():
    """Create a mock supervision.Detections class that works like the real one."""
    class MockDetections:
        def __init__(self, xyxy, confidence, class_id):
            self.xyxy = xyxy
            self.confidence = confidence
            self.class_id = class_id
            self.tracker_id = None

    return MockDetections


@pytest.fixture
def adapter():
    """BoTSORTAdapter with boxmot and supervision mocked out."""
    MockDetections = _make_mock_detections_class()
    mock_supervision = MagicMock()
    mock_supervision.Detections = MockDetections

    with patch.dict("sys.modules", {
        "boxmot": MagicMock(),
        "boxmot.trackers": MagicMock(),
        "boxmot.trackers.botsort": MagicMock(),
        "boxmot.trackers.botsort.botsort": MagicMock(),
        "supervision": mock_supervision,
    }):
        with patch("card_capture.tracking.botsort_adapter._import_botsort") as mock_import:
            MockBoTSORT = MagicMock()
            mock_tracker, _ = _make_mock_botsort()
            MockBoTSORT.return_value = mock_tracker
            mock_import.return_value = MockBoTSORT
            from card_capture.tracking.botsort_adapter import BoTSORTAdapter
            yield BoTSORTAdapter(min_track_length=1)


def test_botsort_adapter_importable():
    """Module must be importable without boxmot installed."""
    import importlib
    import card_capture.tracking.botsort_adapter  # must not raise


def test_process_returns_list(adapter):
    cand = _candidate(0, 0, x=100, y=100)
    result = adapter.process([cand])
    assert isinstance(result, list)


def test_process_empty_input_returns_empty(adapter):
    result = adapter.process([])
    assert result == []


def test_pending_splits_starts_empty(adapter):
    assert adapter.pending_splits == []


def test_botsort_internal_confirmation_does_not_hide_short_sessions(adapter):
    assert adapter._BoTSORT.call_args.kwargs["min_hits"] == 1
    assert adapter._BoTSORT.call_args.kwargs["new_track_thresh"] == adapter._track_activation_threshold
    assert adapter._BoTSORT.call_args.kwargs["track_low_thresh"] == 0.1
    adapter.reset()
    assert adapter._BoTSORT.call_args.kwargs["min_hits"] == 1
    assert adapter._BoTSORT.call_args.kwargs["new_track_thresh"] == adapter._track_activation_threshold
    assert adapter._BoTSORT.call_args.kwargs["track_low_thresh"] == 0.1


def test_finalize_returns_track_states(adapter):
    for i in range(3):
        adapter.process([_candidate(i, i, x=100, y=100)])
    tracks = adapter.finalize()
    assert isinstance(tracks, list)


def test_reset_clears_state(adapter):
    for i in range(3):
        adapter.process([_candidate(i, i, x=100, y=100)])
    adapter.reset()
    assert adapter.pending_splits == []


def test_finalized_tracks_returns_list(adapter):
    result = adapter.finalized_tracks()
    assert isinstance(result, list)


def test_import_botsort_resolves_new_or_old_api():
    """The adapter must import BoT-SORT regardless of boxmot major version."""
    from card_capture.tracking.botsort_adapter import _import_botsort
    klass = _import_botsort()
    # Class name varies across versions: 'BoTSORT' (≤0.16) or 'BotSort' (≥0.17).
    assert klass.__name__ in {"BoTSORT", "BotSort"}, klass.__name__
