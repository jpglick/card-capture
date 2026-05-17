"""Step 5 — score: Stage 7 quality scoring and track pruning.

Adds quality scores to each frame entry (already computed in refine)
and prunes tracks whose median quad-novelty matches the empty workspace.
Returns scored candidates and the list of pruned track IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.steps.start import RunContext
from pipeline.steps.refine import RefineOutput


def _novelty_gate_useful(scores: list) -> bool:
    """Return True only when the background model discriminates well enough to prune.

    Requires all three:
    - At least 5 detections (enough data)
    - std > 0.15 (model is actually separating high from low novelty)
    - min < 0.35 (at least some detections look like background)
    """
    if len(scores) < 5:
        return False
    import numpy as np
    arr = np.array(scores, dtype=np.float32)
    return float(arr.std()) > 0.15 and float(arr.min()) < 0.35


@dataclass
class ScoreOutput:
    """Outputs of the score step."""

    scored_tracks: List[Dict[str, Any]]   # refined_tracks with pruned flag
    pruned_instance_ids: List[str]        # instance_ids dropped by novelty pruning
    tracker_events: List[Dict[str, Any]]
    detection_rows: List[Dict[str, Any]]
    sampler_telemetry: Dict[str, Any]
    bg_model_path: Optional[str]
    accepted_frame_presence: List[Tuple[int, int, bool]]
    frame_count: int
    accepted_frame_count: int
    video_id: int


def run(ctx: RunContext, refine_out: RefineOutput) -> ScoreOutput:
    """Prune empty-workspace tracks and attach quality scores.

    Two independent pruning gates, either of which can prune a track:

    1. Novelty gate: activates when the per-video novelty score distribution is
       bimodal (background model discriminates). Prunes tracks whose median
       novelty falls below ctx.novelty_floor (default 0.30). Stays off for
       hand-held/transparent-stand videos where all scores cluster similarly.

    2. Confidence floor: always active when ctx.track_confidence_floor > 0.
       Prunes tracks whose median frame quality score is below the floor
       (default 0.60). Catches acrylic/transparent stand false positives that
       score 0.54–0.57 while real cards score 0.74–0.79.

    Args:
        ctx:        RunContext from the start step.
        refine_out: Output from the refine step.

    Returns:
        ``ScoreOutput`` with scored / pruned track lists.
    """
    import numpy as np

    bg_model = None
    if refine_out.bg_model_path and Path(refine_out.bg_model_path).exists():
        from card_capture.presence.background_novelty import BackgroundModel
        mean_bgr = np.load(refine_out.bg_model_path)
        bg_model = BackgroundModel.__new__(BackgroundModel)
        bg_model.mean_bgr = mean_bgr

    # Collect novelty scores from frame entries (ctx.observed_novelty_scores is
    # not propagated across Metaflow step boundaries; refine_out is reliable)
    all_novelty_scores = [
        float(fe.get("novelty_score", 1.0))
        for track_dict in refine_out.refined_tracks
        for fe in track_dict.get("frame_entries", [])
    ]
    gate_useful = _novelty_gate_useful(all_novelty_scores)
    novelty_threshold = ctx.novelty_floor if gate_useful else -1.0
    conf_floor = ctx.track_confidence_floor

    scored_tracks: List[Dict[str, Any]] = []
    pruned_instance_ids: List[str] = []

    for track_dict in refine_out.refined_tracks:
        frame_entries = track_dict.get("frame_entries", [])

        novelty_scores = [
            float(fe.get("novelty_score", 1.0))
            for fe in frame_entries
        ]
        median_novelty = float(np.median(novelty_scores)) if novelty_scores else 1.0

        quality_scores = [
            float(fe.get("score_total", fe.get("confidence", 1.0)))
            for fe in frame_entries
        ]
        median_quality = float(np.median(quality_scores)) if quality_scores else 1.0

        novelty_prune = (bg_model is not None) and gate_useful and (median_novelty < novelty_threshold)
        confidence_prune = conf_floor > 0 and median_quality < conf_floor
        should_prune = novelty_prune or confidence_prune

        track_out = dict(track_dict)
        track_out["pruned"] = should_prune
        track_out["median_novelty"] = median_novelty
        track_out["median_quality"] = median_quality

        scored_tracks.append(track_out)
        if should_prune:
            pruned_instance_ids.append(track_dict["instance_id"])

    active_count = sum(1 for t in scored_tracks if not t["pruned"])
    print(
        f"[Stage: Score] | {len(scored_tracks)} tracks scored"
        f" | {len(pruned_instance_ids)} pruned | {active_count} active"
        f" | novelty_gate={'on' if gate_useful else 'off'}"
        f" | conf_floor={conf_floor:.2f}"
    )

    return ScoreOutput(
        scored_tracks=scored_tracks,
        pruned_instance_ids=pruned_instance_ids,
        tracker_events=refine_out.tracker_events,
        detection_rows=refine_out.detection_rows,
        sampler_telemetry=refine_out.sampler_telemetry,
        bg_model_path=refine_out.bg_model_path,
        accepted_frame_presence=refine_out.accepted_frame_presence,
        frame_count=refine_out.frame_count,
        accepted_frame_count=refine_out.accepted_frame_count,
        video_id=refine_out.video_id,
    )
