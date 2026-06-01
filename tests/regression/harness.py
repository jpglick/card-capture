from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .cross_video import compute_dedup_f1
from .matcher import MatchedPair, match_instances_to_truth
from .metrics import VideoMetrics, compute_video_metrics, count_id_switches
from .pipeline_runner import HarnessInstance, load_instances_for_video
from .report import AggregateReport, write_json_report, write_markdown_report
from .truth import GroundTruth, load_truth


@dataclass(frozen=True)
class HarnessConfig:
    corpus_dir: Path
    output_dir: Path
    git_sha: str
    tolerance_ms: int = 500
    db_path: Path = Path("card_capture_output/cards.sqlite")
    presence_threshold: float = 0.5


def _peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def run_pipeline_for_video(
    video_path: Path,
    db_path: Path,
    output_dir: Path,
    presence_threshold: float = 0.5,
) -> Tuple[List[HarnessInstance], float, float, list]:
    """Run the real pipeline against a video and return harness records.

    Returns (instances, wall_clock_s, peak_memory_mb, events).
    This function is monkeypatched in tests.
    """
    from card_capture.cli import _run_process
    import argparse

    args = argparse.Namespace(
        video_path=Path(video_path),
        output_dir=Path(output_dir),
        db=Path(db_path),
        config=Path("card_capture_config.json"),
        presence_threshold=presence_threshold,
    )

    start = time.perf_counter()
    rc = _run_process(args)
    wall = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"pipeline returned non-zero exit code {rc} for {video_path}")

    from card_capture.stages.store.storage import Storage
    import json as _json

    storage = Storage(db_path)
    storage.initialize()
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT id FROM videos WHERE source_path = ? ORDER BY id DESC LIMIT 1",
            (str(video_path),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"no video row found for {video_path} after pipeline run")
        video_db_id = int(row["id"])

        events_rows = conn.execute(
            "SELECT event_type, data_json FROM pipeline_events WHERE video_id = ?",
            (video_db_id,),
        ).fetchall()

    parsed_events = []
    for e in events_rows:
        d = _json.loads(e["data_json"]) if e["data_json"] else {}
        parsed_events.append({"event_type": e["event_type"], **d})

    instances = load_instances_for_video(db_path, video_db_id)
    return instances, wall, _peak_memory_mb(), parsed_events


def run_corpus(cfg: HarnessConfig) -> AggregateReport:
    truth_files = sorted(Path(cfg.corpus_dir).glob("*/*.truth.json"))
    if not truth_files:
        raise RuntimeError(f"no truth.json files found under {cfg.corpus_dir}")

    per_video: List[VideoMetrics] = []
    matched_per_video: List[Sequence[MatchedPair]] = []

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    for truth_path in truth_files:
        truth: GroundTruth = load_truth(truth_path)
        video_path = Path(truth.video_path)

        instances, wall, peak_mb, events = run_pipeline_for_video(
            video_path=video_path,
            db_path=cfg.db_path,
            output_dir=cfg.output_dir,
            presence_threshold=cfg.presence_threshold,
        )
        match = match_instances_to_truth(instances, truth.expected_cards, tolerance_ms=cfg.tolerance_ms)

        vm = compute_video_metrics(
            match, truth.expected_cards,
            video_id=truth.video_id,
            id_switches=count_id_switches(events),
            sharpness_mean=0.0,
            wall_clock_s=wall,
            peak_memory_mb=peak_mb,
        )
        per_video.append(vm)
        matched_per_video.append(match.matched)

    dedup = compute_dedup_f1(matched_per_video)
    return AggregateReport(git_sha=cfg.git_sha, per_video=tuple(per_video), dedup=dedup)
