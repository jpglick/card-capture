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
2. ``<truth_dir>/<video_id>/truth.json``  (deprecated)
3. ``<truth_dir>/truth.json``  (deprecated; single-file fixtures used in unit tests)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from harness.metrics.card_precision import card_precision
from harness.metrics.card_recall import card_recall
from harness.metrics.dedup_accuracy import dedup_accuracy
from harness.metrics.image_quality import image_quality
from harness.metrics.side_accuracy import side_accuracy
from harness.metrics.types import MetricResult


@dataclass
class PerVideoReport:
    """Metric results for a single video."""

    video_id: str
    metrics: dict[str, MetricResult]


@dataclass
class Report:
    """Aggregate report across all videos in a run."""

    metrics: dict[str, MetricResult]
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
        m: dict[str, MetricResult] = {
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
    1. ``truth_dir/<video_id>.truth.json``   (canonical)
    2. ``truth_dir/<video_id>/truth.json``    (legacy subdirectory form — deprecated)
    3. ``truth_dir/truth.json``              (legacy single-file form — deprecated)
    """
    canonical = truth_dir / f"{video_id}.truth.json"
    if canonical.exists():
        return canonical

    subdirectory = truth_dir / video_id / "truth.json"
    if subdirectory.exists():
        warnings.warn(
            f"Truth file at '{subdirectory}' uses the deprecated subdirectory layout. "
            f"Rename to '{canonical}' to suppress this warning.",
            DeprecationWarning,
            stacklevel=3,
        )
        return subdirectory

    single_file = truth_dir / "truth.json"
    if single_file.exists():
        warnings.warn(
            f"Truth file at '{single_file}' uses the deprecated single-file layout. "
            f"Create per-video files named '<video_id>.truth.json' to suppress this warning.",
            DeprecationWarning,
            stacklevel=3,
        )
        return single_file

    raise FileNotFoundError(
        f"No truth.json found for video_id '{video_id}' in {truth_dir}. "
        f"Expected: '{canonical}'"
    )


def _avg_or_none(vals: list[Optional[float]]) -> Optional[float]:
    non_null = [v for v in vals if v is not None]
    return sum(non_null) / len(non_null) if non_null else None


def _aggregate(per_video: list[PerVideoReport]) -> dict[str, MetricResult]:
    """Compute cross-video aggregates.

    Simple metrics (value is float | None): averaged across videos that have a value.
    Compound metrics (breakdown dict): averaged field-by-field.
    """
    out: dict[str, MetricResult] = {}

    for key in ("card_recall", "card_precision", "side_accuracy"):
        vals = [pv.metrics[key].value for pv in per_video]
        out[key] = MetricResult(value=_avg_or_none(vals))

    # dedup_accuracy — average ari and pair_f1 breakdown keys
    dedup_breakdowns = [pv.metrics["dedup_accuracy"].breakdown for pv in per_video]
    out["dedup_accuracy"] = MetricResult(
        breakdown={
            "ari": _avg_or_none([b.get("ari") for b in dedup_breakdowns]),
            "pair_f1": _avg_or_none([b.get("pair_f1") for b in dedup_breakdowns]),
        }
    )

    # image_quality — average mean_ssim, mean_psnr, coverage breakdown keys
    iq_breakdowns = [pv.metrics["image_quality"].breakdown for pv in per_video]
    out["image_quality"] = MetricResult(
        breakdown={
            "mean_ssim": _avg_or_none([b.get("mean_ssim") for b in iq_breakdowns]),
            "mean_psnr": _avg_or_none([b.get("mean_psnr") for b in iq_breakdowns]),
            "coverage": _avg_or_none([b.get("coverage") for b in iq_breakdowns]),
        }
    )

    return out
