from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .cross_video import DedupMetrics
from .metrics import VideoMetrics


@dataclass(frozen=True)
class AggregateReport:
    git_sha: str
    per_video: Tuple[VideoMetrics, ...]
    dedup: DedupMetrics


def _aggregate_recall(per_video: Sequence[VideoMetrics]) -> float:
    expected = sum(v.expected_cards for v in per_video)
    detected = sum(v.detected_cards for v in per_video)
    return (detected / expected) if expected else 1.0


def _aggregate_phantom_rate(per_video: Sequence[VideoMetrics]) -> float:
    output = sum(v.pipeline_output_count for v in per_video)
    phantoms = sum(v.phantom_count for v in per_video)
    return (phantoms / output) if output else 0.0


def _aggregate_fb_accuracy(per_video: Sequence[VideoMetrics]) -> float:
    total = sum(v.fb_total for v in per_video)
    correct = sum(v.fb_correct for v in per_video)
    return (correct / total) if total else 1.0


def write_json_report(report: AggregateReport, path: Path) -> None:
    payload = {
        "git_sha": report.git_sha,
        "aggregates": {
            "recall": _aggregate_recall(report.per_video),
            "phantom_rate": _aggregate_phantom_rate(report.per_video),
            "fb_accuracy": _aggregate_fb_accuracy(report.per_video),
        },
        "per_video": [asdict(v) for v in report.per_video],
        "dedup": asdict(report.dedup),
    }
    path.write_text(json.dumps(payload, indent=2))


def _delta(current: float, baseline: Optional[float]) -> str:
    if baseline is None:
        return ""
    diff = current - baseline
    sign = "+" if diff >= 0 else ""
    return f" ({sign}{diff:.3f})"


def write_markdown_report(
    report: AggregateReport,
    path: Path,
    baseline: Optional[AggregateReport] = None,
) -> None:
    lines = [
        f"# Harness report — {report.git_sha}",
        "",
        "## Aggregates",
        "",
    ]
    cur_recall = _aggregate_recall(report.per_video)
    cur_phantom = _aggregate_phantom_rate(report.per_video)
    cur_fb = _aggregate_fb_accuracy(report.per_video)
    base_recall = _aggregate_recall(baseline.per_video) if baseline else None
    base_phantom = _aggregate_phantom_rate(baseline.per_video) if baseline else None
    base_fb = _aggregate_fb_accuracy(baseline.per_video) if baseline else None

    lines.append(f"- Recall: **{cur_recall:.3f}**{_delta(cur_recall, base_recall)}")
    lines.append(f"- Phantom rate: **{cur_phantom:.3f}**{_delta(cur_phantom, base_phantom)}")
    lines.append(f"- F/B accuracy: **{cur_fb:.3f}**{_delta(cur_fb, base_fb)}")
    lines.append(f"- Dedup F1: **{report.dedup.f1:.3f}**")
    lines.append("")

    lines.append("## Per video")
    lines.append("")
    lines.append("| video | recall | phantom_rate | fb_acc | id_switches | wall_s |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    base_by_id = {v.video_id: v for v in baseline.per_video} if baseline else {}
    for v in report.per_video:
        b = base_by_id.get(v.video_id)
        recall_cell = f"{v.recall:.3f}{_delta(v.recall, b.recall if b else None)}"
        phantom_cell = f"{v.phantom_rate:.3f}{_delta(v.phantom_rate, b.phantom_rate if b else None)}"
        fb_cell = f"{v.fb_accuracy:.3f}{_delta(v.fb_accuracy, b.fb_accuracy if b else None)}"
        lines.append(
            f"| {v.video_id} | {recall_cell} | {phantom_cell} | {fb_cell} | {v.id_switches} | {v.wall_clock_s:.1f} |"
        )

    path.write_text("\n".join(lines) + "\n")
