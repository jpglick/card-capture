"""Stage 7: Quality Scoring + Pruning.

Port of V4 ``pipeline/steps/score.py``. Three independent prune gates,
any of which can drop a track:

1. **Novelty gate**: only active when the per-video novelty distribution
   is bimodal (background model discriminates). Adaptive threshold lands
   at the midpoint of the largest gap between per-track median novelties,
   capped at ``config['novelty_floor']``.
2. **Confidence floor**: always active when ``config['track_confidence_floor']
   > 0``. Prunes tracks whose median quality score is below it.
3. **Transparent-stand gate**: low novelty AND low sharpness. Active when
   ``bg_model`` is present AND ``config['stand_novelty_max'] > 0``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from card_capture.pipeline.stage_metrics import emit_stage_metrics


def _novelty_gate_useful(scores: list) -> bool:
    """Mirrors V4 _novelty_gate_useful: ≥5 detections, std > 0.15, min < 0.35."""
    if len(scores) < 5:
        return False
    arr = np.array(scores, dtype=np.float32)
    return float(arr.std()) > 0.15 and float(arr.min()) < 0.35


def run(state: dict, *, telemetry) -> None:
    config = state["request"].config
    bg_model = state.get("bg_model")
    refined_tracks = state.get("refined_tracks") or []

    novelty_floor = float(config.get("novelty_floor", 0.30))
    conf_floor = float(config.get("track_confidence_floor", 0.60))
    stand_nov_max = float(config.get("stand_novelty_max", 0.35))
    stand_shp_max = float(config.get("stand_sharpness_max", 0.30))

    all_novelty_scores = [
        float(fe.get("novelty_score", 1.0))
        for track_dict in refined_tracks
        for fe in track_dict.get("frame_entries", [])
    ]
    gate_useful = _novelty_gate_useful(all_novelty_scores)

    if gate_useful:
        track_medians = sorted(
            float(np.median([float(fe.get("novelty_score", 1.0))
                              for fe in t.get("frame_entries", [])])
                  if t.get("frame_entries") else 1.0)
            for t in refined_tracks
        )
        if len(track_medians) >= 2:
            gaps = [track_medians[i + 1] - track_medians[i]
                    for i in range(len(track_medians) - 1)]
            gap_idx = gaps.index(max(gaps))
            adaptive = (track_medians[gap_idx] + track_medians[gap_idx + 1]) / 2
            novelty_threshold = min(adaptive, novelty_floor)
        else:
            novelty_threshold = novelty_floor
    else:
        novelty_threshold = -1.0

    scored_tracks: List[Dict[str, Any]] = []
    pruned_instance_ids: List[str] = []

    for track_dict in refined_tracks:
        frame_entries = track_dict.get("frame_entries", [])

        novelty_scores = [float(fe.get("novelty_score", 1.0)) for fe in frame_entries]
        median_novelty = float(np.median(novelty_scores)) if novelty_scores else 1.0

        quality_scores = [
            float(fe.get("score_total", fe.get("quality_score", fe.get("confidence", 1.0))))
            for fe in frame_entries
        ]
        median_quality = float(np.median(quality_scores)) if quality_scores else 1.0

        sharpness_scores = [
            float(fe.get("quality_components", {}).get("sharpness", 1.0))
            for fe in frame_entries
        ]
        median_sharpness = float(np.median(sharpness_scores)) if sharpness_scores else 1.0

        novelty_prune = (bg_model is not None) and gate_useful and (median_novelty < novelty_threshold)
        confidence_prune = conf_floor > 0 and median_quality < conf_floor
        stand_prune = (
            bg_model is not None
            and stand_nov_max > 0
            and median_novelty < stand_nov_max
            and median_sharpness < stand_shp_max
        )
        should_prune = novelty_prune or confidence_prune or stand_prune

        scored = dict(track_dict)
        scored["pruned"] = should_prune
        scored["median_novelty"] = median_novelty
        scored["median_quality"] = median_quality
        scored["median_sharpness"] = median_sharpness
        scored_tracks.append(scored)

        if should_prune:
            pruned_instance_ids.append(track_dict["instance_id"])

    state["scored_tracks"] = scored_tracks
    state["pruned_instance_ids"] = pruned_instance_ids
    emit_stage_metrics(
        state,
        stage="score",
        metrics={"scored_tracks": len(scored_tracks), "pruned_tracks": len(pruned_instance_ids)},
    )
