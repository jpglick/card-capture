from tests.regression.matcher import MatchResult, MatchedPair
from tests.regression.metrics import VideoMetrics, compute_video_metrics
from tests.regression.pipeline_runner import HarnessInstance
from tests.regression.truth import ExpectedCard


def _inst(iid, angle, start=0, end=1000):
    return HarnessInstance(
        instance_id=iid, video_id=1, session_id=1, angle=angle,
        duplicate_of=None, fused_image_path=None,
        start_ms=start, end_ms=end, detection_count=5, phash=None,
    )


def _exp(card_id, front=True, back=True):
    return ExpectedCard(
        card_id=card_id, front_present=front, back_present=back,
        approx_front_window_ms=(0, 1000) if front else None,
        approx_back_window_ms=(2000, 3000) if back else None,
    )


def test_metrics_perfect_video():
    truth = (_exp("c1"),)
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),
        MatchedPair(truth_card=truth[0], side="B", instance=_inst(11, "Back", 2000, 3000)),
    )
    result = MatchResult(matched=matched, unmatched_truth=(), phantom_instances=())

    m = compute_video_metrics(result, truth)
    assert isinstance(m, VideoMetrics)
    assert m.expected_cards == 1
    assert m.detected_cards == 1
    assert m.recall == 1.0
    assert m.phantom_rate == 0.0
    assert m.fb_correct == 2
    assert m.fb_total == 2
    assert m.fb_accuracy == 1.0


def test_metrics_recall_partial():
    truth = (_exp("c1"), _exp("c2"))
    matched = (MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),)
    result = MatchResult(matched=matched, unmatched_truth=(truth[1],), phantom_instances=())

    m = compute_video_metrics(result, truth)
    assert m.recall == 0.5


def test_metrics_phantom_rate():
    truth = (_exp("c1"),)
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),
    )
    phantoms = (_inst(99, "Front", 5000, 6000),)
    result = MatchResult(matched=matched, unmatched_truth=(), phantom_instances=phantoms)

    m = compute_video_metrics(result, truth)
    assert m.phantom_rate == 0.5


def test_metrics_fb_inversion():
    truth = (_exp("c1"),)
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Back", 0, 1000)),
        MatchedPair(truth_card=truth[0], side="B", instance=_inst(11, "Front", 2000, 3000)),
    )
    result = MatchResult(matched=matched, unmatched_truth=(), phantom_instances=())

    m = compute_video_metrics(result, truth)
    assert m.fb_correct == 0
    assert m.fb_accuracy == 0.0


from tests.regression.metrics import count_id_switches


def test_id_switches_counts_track_changes_per_session():
    events = [
        {"event_type": "tracking", "session_id": 1, "track_id": "a", "timestamp_ms": 100},
        {"event_type": "tracking", "session_id": 1, "track_id": "a", "timestamp_ms": 200},
        {"event_type": "tracking", "session_id": 1, "track_id": "b", "timestamp_ms": 300},
        {"event_type": "tracking", "session_id": 2, "track_id": "c", "timestamp_ms": 5000},
        {"event_type": "tracking", "session_id": 2, "track_id": "d", "timestamp_ms": 5100},
        {"event_type": "tracking", "session_id": 2, "track_id": "d", "timestamp_ms": 5200},
    ]
    assert count_id_switches(events) == 2


def test_id_switches_ignores_non_tracking_events():
    events = [
        {"event_type": "session_reset", "session_id": 1, "timestamp_ms": 0},
        {"event_type": "tracking", "session_id": 1, "track_id": "a", "timestamp_ms": 100},
    ]
    assert count_id_switches(events) == 0
