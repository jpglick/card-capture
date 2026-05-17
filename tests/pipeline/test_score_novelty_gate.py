import pytest


def _gate(scores):
    from pipeline.steps.score import _novelty_gate_useful
    return _novelty_gate_useful(scores)


def test_bimodal_distribution_activates_gate():
    """Stand-style video: low-novelty stand detections + high-novelty card detections."""
    scores = [0.05, 0.08, 0.85, 0.90, 0.92]
    assert _gate(scores) is True


def test_all_high_scores_disables_gate():
    """Hand-held video: everything is novel vs background — std too low."""
    scores = [0.82, 0.88, 0.91, 0.87, 0.85]
    assert _gate(scores) is False


def test_too_few_samples_disables_gate():
    """Fewer than 5 detections — not enough data to judge distribution."""
    scores = [0.05, 0.90]
    assert _gate(scores) is False


def test_high_std_but_min_not_low_enough_disables_gate():
    """Spread exists but nothing scores below 0.35 — no background-like detections."""
    scores = [0.40, 0.90, 0.91, 0.40, 0.88]
    assert _gate(scores) is False


def test_empty_scores_disables_gate():
    """No detections at all — gate must not fire."""
    assert _gate([]) is False


def test_exactly_five_samples_bimodal_activates():
    """Boundary: exactly 5 samples with bimodal distribution."""
    scores = [0.10, 0.12, 0.80, 0.85, 0.88]
    assert _gate(scores) is True


# ---------------------------------------------------------------------------
# Confidence floor logic (tested via the score step's run() directly)
# ---------------------------------------------------------------------------

_track_counter = 0

def _make_track(score_totals, novelty_scores=None):
    """Helper: build a minimal refined_track dict."""
    global _track_counter
    _track_counter += 1
    entries = []
    for i, st in enumerate(score_totals):
        entry = {"score_total": st}
        entry["novelty_score"] = novelty_scores[i] if novelty_scores else 1.0
        entries.append(entry)
    return {"instance_id": f"track-{_track_counter}", "frame_entries": entries}


def _make_ctx(conf_floor=0.60, novelty_floor=0.30):
    from pipeline.steps.start import RunContext
    ctx = RunContext(
        video_path="/fake/video.MOV",
        output_dir="/tmp/fake",
        db_path="/tmp/fake.sqlite",
        detector="fake",
        config_preset="balanced",
    )
    ctx.track_confidence_floor = conf_floor
    ctx.novelty_floor = novelty_floor
    ctx.observed_novelty_scores = []
    return ctx


def _make_refine_out(tracks):
    from unittest.mock import MagicMock
    refine_out = MagicMock()
    refine_out.refined_tracks = tracks
    refine_out.bg_model_path = None
    refine_out.tracker_events = []
    refine_out.detection_rows = []
    refine_out.sampler_telemetry = {}
    refine_out.accepted_frame_presence = []
    refine_out.frame_count = 0
    refine_out.accepted_frame_count = 0
    refine_out.video_id = 1
    return refine_out


def test_confidence_floor_prunes_low_quality_track():
    """Acrylic stand track (median quality 0.546) is pruned when floor is 0.60."""
    from pipeline.steps.score import run
    stand_track = _make_track([0.546, 0.537, 0.548, 0.553])
    card_track = _make_track([0.791, 0.778, 0.785, 0.780])

    ctx = _make_ctx(conf_floor=0.60)
    refine_out = _make_refine_out([stand_track, card_track])

    result = run(ctx, refine_out)
    pruned = {t["instance_id"] for t in result.scored_tracks if t["pruned"]}
    assert stand_track["instance_id"] in pruned
    assert card_track["instance_id"] not in pruned


def test_confidence_floor_zero_disables_pruning():
    """Setting track_confidence_floor=0 disables the confidence gate entirely."""
    from pipeline.steps.score import run
    low_track = _make_track([0.40, 0.42, 0.41])

    ctx = _make_ctx(conf_floor=0.0)
    refine_out = _make_refine_out([low_track])

    result = run(ctx, refine_out)
    assert not any(t["pruned"] for t in result.scored_tracks)
