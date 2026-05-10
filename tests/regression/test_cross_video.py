from tests.regression.cross_video import compute_dedup_f1, DedupMetrics
from tests.regression.matcher import MatchedPair
from tests.regression.pipeline_runner import HarnessInstance
from tests.regression.truth import ExpectedCard


def _exp(card_id, key):
    return ExpectedCard(card_id=card_id, front_present=True, back_present=False,
                       approx_front_window_ms=(0, 1000), physical_card_key=key)


def _inst(iid, dup_of=None):
    return HarnessInstance(
        instance_id=iid, video_id=1, session_id=1, angle="Front",
        duplicate_of=dup_of, fused_image_path=None,
        start_ms=0, end_ms=1000, detection_count=1, phash=None,
    )


def test_dedup_perfect():
    pairs_video_1 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(10, dup_of=None))]
    pairs_video_2 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(20, dup_of=10))]

    m = compute_dedup_f1(matched_pairs_per_video=[pairs_video_1, pairs_video_2])
    assert isinstance(m, DedupMetrics)
    assert m.true_positives == 1
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.f1 == 1.0


def test_dedup_missed_duplicate():
    pairs_video_1 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(10, dup_of=None))]
    pairs_video_2 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(20, dup_of=None))]

    m = compute_dedup_f1(matched_pairs_per_video=[pairs_video_1, pairs_video_2])
    assert m.true_positives == 0
    assert m.false_negatives == 1
    assert m.f1 == 0.0


def test_dedup_false_positive():
    pairs_video_1 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(10, dup_of=None))]
    pairs_video_2 = [MatchedPair(truth_card=_exp("c2", "Y"), side="F", instance=_inst(20, dup_of=10))]

    m = compute_dedup_f1(matched_pairs_per_video=[pairs_video_1, pairs_video_2])
    assert m.false_positives == 1
    assert m.f1 == 0.0
