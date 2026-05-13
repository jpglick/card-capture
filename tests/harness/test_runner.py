"""Tests for harness.runner — aggregate metric runner."""
import pytest
from pathlib import Path

from harness.runner import Report, PerVideoReport, run_metrics
from harness.metrics.dedup_accuracy import DedupAccuracy
from harness.metrics.image_quality import ImageQuality

FIX = Path("tests/harness/fixtures/runs")


def test_runner_returns_all_metrics():
    """Runner produces all five metric keys in both aggregate and per-video."""
    report = run_metrics(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_dir=FIX / "all_matched",
        videos=["all_matched"],
    )
    assert isinstance(report, Report)
    assert "card_recall" in report.metrics
    assert "card_precision" in report.metrics
    assert "side_accuracy" in report.metrics
    assert "dedup_accuracy" in report.metrics
    assert "image_quality" in report.metrics
    assert len(report.per_video) == 1
    assert report.per_video[0].video_id == "all_matched"


def test_runner_per_video_has_all_keys():
    report = run_metrics(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_dir=FIX / "all_matched",
        videos=["all_matched"],
    )
    pv = report.per_video[0]
    for key in ("card_recall", "card_precision", "side_accuracy", "dedup_accuracy", "image_quality"):
        assert key in pv.metrics


def test_runner_recall_value():
    """all_matched: 2 of 3 GT cards matched → recall = 2/3."""
    report = run_metrics(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_dir=FIX / "all_matched",
        videos=["all_matched"],
    )
    assert report.metrics["card_recall"] == pytest.approx(2 / 3)


def test_runner_aggregate_averages_across_videos():
    """With two runs of the same all_matched fixture, aggregate = per-video value."""
    report = run_metrics(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_dir=FIX / "all_matched",
        videos=["all_matched"],
    )
    pv_recall = report.per_video[0].metrics["card_recall"]
    assert report.metrics["card_recall"] == pytest.approx(pv_recall)


def test_runner_raises_when_truth_not_found(tmp_path):
    """Missing truth.json raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_metrics(
            db_path=FIX / "all_matched" / "cards.sqlite",
            truth_dir=tmp_path,
            videos=["no_such_video"],
        )


def test_runner_dedup_accuracy_is_dataclass():
    report = run_metrics(
        db_path=FIX / "dedup" / "cards.sqlite",
        truth_dir=FIX / "dedup",
        videos=["dedup"],
    )
    assert isinstance(report.metrics["dedup_accuracy"], DedupAccuracy)


def test_runner_image_quality_is_dataclass():
    report = run_metrics(
        db_path=FIX / "image_quality" / "cards.sqlite",
        truth_dir=FIX / "image_quality",
        videos=["image_quality"],
    )
    assert isinstance(report.metrics["image_quality"], ImageQuality)
