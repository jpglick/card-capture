from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .matcher import MatchResult
from .truth import ExpectedCard


@dataclass(frozen=True)
class VideoMetrics:
    video_id: str
    expected_cards: int
    detected_cards: int
    recall: float
    phantom_count: int
    pipeline_output_count: int
    phantom_rate: float
    fb_correct: int
    fb_total: int
    fb_accuracy: float
    id_switches: int = 0
    sharpness_mean: float = 0.0
    wall_clock_s: float = 0.0
    peak_memory_mb: float = 0.0


def _angle_to_side(angle: str) -> str:
    a = angle.strip().lower()
    if a in {"front", "f"}:
        return "F"
    if a in {"back", "b"}:
        return "B"
    return "?"


def compute_video_metrics(
    match: MatchResult,
    truth: Sequence[ExpectedCard],
    *,
    video_id: str = "",
    id_switches: int = 0,
    sharpness_mean: float = 0.0,
    wall_clock_s: float = 0.0,
    peak_memory_mb: float = 0.0,
) -> VideoMetrics:
    expected = len(truth)
    detected_card_ids = {pair.truth_card.card_id for pair in match.matched}
    detected = len(detected_card_ids)
    recall = (detected / expected) if expected else 1.0

    pipeline_output = len(match.matched) + len(match.phantom_instances)
    phantom_rate = (len(match.phantom_instances) / pipeline_output) if pipeline_output else 0.0

    fb_total = len(match.matched)
    fb_correct = sum(
        1 for pair in match.matched
        if _angle_to_side(pair.instance.angle) == pair.side
    )
    fb_accuracy = (fb_correct / fb_total) if fb_total else 1.0

    return VideoMetrics(
        video_id=video_id,
        expected_cards=expected,
        detected_cards=detected,
        recall=recall,
        phantom_count=len(match.phantom_instances),
        pipeline_output_count=pipeline_output,
        phantom_rate=phantom_rate,
        fb_correct=fb_correct,
        fb_total=fb_total,
        fb_accuracy=fb_accuracy,
        id_switches=id_switches,
        sharpness_mean=sharpness_mean,
        wall_clock_s=wall_clock_s,
        peak_memory_mb=peak_memory_mb,
    )


def count_id_switches(events: Iterable[dict]) -> int:
    by_session: dict = {}
    for ev in events:
        if ev.get("event_type") != "tracking":
            continue
        sid = ev.get("session_id")
        tid = ev.get("track_id")
        if sid is None or tid is None:
            continue
        by_session.setdefault(sid, []).append((ev.get("timestamp_ms", 0), tid))

    switches = 0
    for _sid, entries in by_session.items():
        entries.sort()
        prev = None
        for _ts, tid in entries:
            if prev is not None and tid != prev:
                switches += 1
            prev = tid
    return switches
