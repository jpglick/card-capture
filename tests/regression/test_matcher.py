from tests.regression.matcher import match_instances_to_truth, MatchResult
from tests.regression.pipeline_runner import HarnessInstance
from tests.regression.truth import ExpectedCard


def _inst(iid, video_id, session_id, angle, start, end, **kw):
    return HarnessInstance(
        instance_id=iid, video_id=video_id, session_id=session_id, angle=angle,
        duplicate_of=kw.get("duplicate_of"), fused_image_path=None,
        start_ms=start, end_ms=end, detection_count=10, phash=None,
    )


def _exp(card_id, front_window=None, back_window=None, key=None):
    return ExpectedCard(
        card_id=card_id,
        front_present=front_window is not None,
        back_present=back_window is not None,
        approx_front_window_ms=front_window,
        approx_back_window_ms=back_window,
        physical_card_key=key,
    )


def test_match_pairs_overlapping_windows():
    truth = (_exp("c1", front_window=(1000, 3000), back_window=(3500, 5000)),)
    instances = [
        _inst(10, 1, 1, "Front", 1100, 2900),
        _inst(11, 1, 1, "Back", 3600, 4900),
    ]
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)

    assert isinstance(result, MatchResult)
    assert len(result.matched) == 2
    matched_ids = {pair.instance.instance_id for pair in result.matched}
    assert matched_ids == {10, 11}
    assert len(result.unmatched_truth) == 0
    assert len(result.phantom_instances) == 0


def test_phantom_when_no_truth_overlaps():
    truth = (_exp("c1", front_window=(1000, 2000)),)
    instances = [
        _inst(10, 1, 1, "Front", 1100, 1900),
        _inst(11, 1, 2, "Front", 8000, 9000),
    ]
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)
    assert {p.instance.instance_id for p in result.matched} == {10}
    assert len(result.phantom_instances) == 1
    assert result.phantom_instances[0].instance_id == 11


def test_unmatched_truth_when_no_instance_overlaps():
    truth = (
        _exp("c1", front_window=(1000, 2000)),
        _exp("c2", front_window=(5000, 6000)),
    )
    instances = [_inst(10, 1, 1, "Front", 1000, 2000)]
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)
    assert len(result.unmatched_truth) == 1
    assert result.unmatched_truth[0].card_id == "c2"


def test_tolerance_allows_window_drift():
    truth = (_exp("c1", front_window=(1000, 2000)),)
    instances = [_inst(10, 1, 1, "Front", 2300, 2800)]
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)
    assert len(result.matched) == 1
