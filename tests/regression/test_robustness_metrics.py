"""Tests for robustness metrics."""
from tests.regression.matcher import MatchResult, MatchedPair
from tests.regression.metrics import compute_video_metrics
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


def test_card_recall_counts_matched_cards():
    """Test that card recall = matched_cards / total_truth_cards."""
    # 3 truth cards, 2 matched
    truth = (_exp("c1"), _exp("c2"), _exp("c3"))
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),
        MatchedPair(truth_card=truth[1], side="F", instance=_inst(11, "Front", 0, 1000)),
    )
    result = MatchResult(matched=matched, unmatched_truth=(truth[2],), phantom_instances=())

    # Import and use the robustness metrics
    from card_capture.metrics.robustness_pack import RobustnessMetrics

    metrics = RobustnessMetrics(matched=matched, unmatched_truth=(truth[2],),
                                 phantom_instances=(), truth_cards=truth)
    recall = metrics.card_recall()

    # 2 out of 3 cards matched
    assert recall == 2 / 3, f"Expected recall 2/3, got {recall}"


def test_front_back_f1_measures_angle_accuracy():
    """Test that front/back F1 measures angle accuracy."""
    # 2 cards, both matched correctly
    truth = (_exp("c1"), _exp("c2"))
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),
        MatchedPair(truth_card=truth[0], side="B", instance=_inst(11, "Back", 2000, 3000)),
        MatchedPair(truth_card=truth[1], side="F", instance=_inst(12, "Front", 0, 1000)),
        # c2 back is wrong angle
        MatchedPair(truth_card=truth[1], side="B", instance=_inst(13, "Front", 2000, 3000)),
    )
    result = MatchResult(matched=matched, unmatched_truth=(), phantom_instances=())

    from card_capture.metrics.robustness_pack import RobustnessMetrics

    metrics = RobustnessMetrics(matched=matched, unmatched_truth=(),
                                 phantom_instances=(), truth_cards=truth)
    f1 = metrics.front_back_f1()

    # 3 correct out of 4 total
    # Precision = 3/4, Recall = 3/4, F1 = 0.75
    assert f1 == 0.75, f"Expected F1 0.75, got {f1}"
