"""Aggregate harness runner.

Calls all five metrics for each video in the given list and returns a
:class:`Report` with per-video results and a cross-video aggregate.

Usage::

    from pathlib import Path
    from harness.runner import run_metrics

    report = run_metrics(
        db_path=Path("cards.sqlite"),
        truth_dir=Path("golden_set/videos/run_2026_05"),
        videos=["practice_session_03", "practice_session_04"],
    )
    print(report.metrics)          # aggregate across all videos
    print(report.per_video[0])     # per-video breakdown

Truth file lookup
-----------------
For each ``video_id`` the runner looks for a truth file in this order:

1. ``<truth_dir>/<video_id>.truth.json``
2. ``<truth_dir>/<video_id>/truth.json``
3. ``<truth_dir>/truth.json``  (single-file fixtures used in unit tests)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.metrics.card_precision import card_precision
from harness.metrics.card_recall import card_recall
from harness.metrics.dedup_accuracy import DedupAccuracy, dedup_accuracy
from harness.metrics.image_quality import ImageQuality, image_quality
from harness.metrics.side_accuracy import side_accuracy


@dataclass
class PerVideoReport:
    """Metric results for a single video."""

    video_id: str
    metrics: dict[str, Any]  # float | None | DedupAccuracy | ImageQuality


@dataclass
class Report:
    """Aggregate report across all videos in a run."""

    metrics: dict[str, Any]  # aggregate (averaged) metric values
    per_video: list[PerVideoReport] = field(default_factory=list)


def run_metrics(
    *,
    db_path: Path,
    truth_dir: Path,
    videos: list[str],
) -> Report:
    """Run all metrics across the specified videos.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite``.
    truth_dir:
        Directory containing per-video truth files.
    videos:
        List of ``video_id`` strings to evaluate.

    Returns
    -------
    Report with per-video metrics and a cross-video aggregate.
    """
    per_video: list[PerVideoReport] = []
    for video_id in videos:
        truth_path = _find_truth(truth_dir, video_id)
        m: dict[str, Any] = {
            "card_recall": card_recall(
                db_path=db_path, truth_path=truth_path, video_id=video_id
            ),
            "card_precision": card_precision(
                db_path=db_path, truth_path=truth_path, video_id=video_id
            ),
            "side_accuracy": side_accuracy(
                db_path=db_path, truth_path=truth_path, video_id=video_id
            ),
            "dedup_accuracy": dedup_accuracy(
                db_path=db_path, truth_path=truth_path, video_id=video_id
            ),
            "image_quality": image_quality(
                db_path=db_path, truth_path=truth_path, video_id=video_id
            ),
        }
        per_video.append(PerVideoReport(video_id=video_id, metrics=m))

    aggregate = _aggregate(per_video)
    return Report(metrics=aggregate, per_video=per_video)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_truth(truth_dir: Path, video_id: str) -> Path:
    """Locate a truth.json file for the given video_id.

    Search order:
    1. ``truth_dir/<video_id>.truth.json``
    2. ``truth_dir/<video_id>/truth.json``
    3. ``truth_dir/truth.json``
    """
    candidates = [
        truth_dir / f"{video_id}.truth.json",
        truth_dir / video_id / "truth.json",
        truth_dir / "truth.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No truth.json found for video_id '{video_id}' in {truth_dir}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _avg_or_none(vals: list[float | None]) -> float | None:
    non_null = [v for v in vals if v is not None]
    return sum(non_null) / len(non_null) if non_null else None


def _aggregate(per_video: list[PerVideoReport]) -> dict[str, Any]:
    """Compute cross-video aggregates.

    Simple metrics (float | None): averaged across videos that have a value.
    Dataclass metrics (DedupAccuracy, ImageQuality): averaged field-by-field.
    """
    out: dict[str, Any] = {}

    for key in ("card_recall", "card_precision", "side_accuracy"):
        vals = [pv.metrics[key] for pv in per_video]
        out[key] = _avg_or_none(vals)  # type: ignore[arg-type]

    # DedupAccuracy — average ari and pair_f1 fields
    dedup_results: list[DedupAccuracy] = [
        pv.metrics["dedup_accuracy"]
        for pv in per_video
        if isinstance(pv.metrics.get("dedup_accuracy"), DedupAccuracy)
    ]
    if dedup_results:
        out["dedup_accuracy"] = DedupAccuracy(
            ari=_avg_or_none([r.ari for r in dedup_results]),
            pair_f1=_avg_or_none([r.pair_f1 for r in dedup_results]),
        )
    else:
        out["dedup_accuracy"] = DedupAccuracy(ari=None, pair_f1=None)

    # ImageQuality — average mean_ssim, mean_psnr, coverage
    iq_results: list[ImageQuality] = [
        pv.metrics["image_quality"]
        for pv in per_video
        if isinstance(pv.metrics.get("image_quality"), ImageQuality)
    ]
    if iq_results:
        out["image_quality"] = ImageQuality(
            mean_ssim=_avg_or_none([r.mean_ssim for r in iq_results]),
            mean_psnr=_avg_or_none([r.mean_psnr for r in iq_results]),
            coverage=float(
                sum(r.coverage for r in iq_results) / len(iq_results)
            ),
        )
    else:
        out["image_quality"] = ImageQuality(mean_ssim=None, mean_psnr=None, coverage=0.0)

    return out
