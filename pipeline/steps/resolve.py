"""Step 6 — resolve: resolve session-aware tracks into PreparedTrack items.

For each session, labels tracks using side_score (textiness) as primary sort.
Identifies duplicate tracks within the same session (e.g. Front and Back of the same card).
Returns a list of PreparedTrack-like dicts for the fuse step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.steps.start import RunContext
from pipeline.steps.score import ScoreOutput


@dataclass
class ResolveOutput:
    """Outputs of the resolve step."""

    prepared_tracks: List[Dict[str, Any]]
    tracker_events: List[Dict[str, Any]]
    detection_rows: List[Dict[str, Any]]
    sampler_telemetry: Dict[str, Any]
    accepted_frame_presence: List[Tuple[int, int, bool]]
    frame_count: int
    accepted_frame_count: int
    video_id: int


def run(ctx: RunContext, score_out: ScoreOutput) -> ResolveOutput:
    """Resolve session tracks and prepare for fusion.

    Args:
        ctx:       RunContext from the start step.
        score_out: Output from the score step.

    Returns:
        ``ResolveOutput`` with tracks ready for fusion.
    """
    import numpy as np
    from card_capture.deduplicator import VisualDeduplicator
    from card_capture.identity.embedding_distance import embedding_same_card_score
    from card_capture.calibration.per_video_adaptive import AdaptiveThresholdComputer
    from card_capture.analysis.hard_case_capture import is_hard_case, capture_hard_case
    from card_capture.pipeline import (
        _compute_quality_weighted_score,
        _side_textiness_score,
        _appearance_vector,
    )

    # Filter out pruned tracks
    active_tracks = [t for t in score_out.scored_tracks if not t["pruned"]]
    
    # We need a deduplicator for Hamming distances (fallback)
    deduplicator = VisualDeduplicator()
    adaptive_computer = AdaptiveThresholdComputer()

    # Sort tracks into sessions
    by_session: Dict[int, List[Dict[str, Any]]] = {}
    for track_dict in active_tracks:
        sid = track_dict.get("session_id", 0)
        by_session.setdefault(sid, []).append(track_dict)

    prepared_tracks: List[Dict[str, Any]] = []

    for sid, session_tracks in by_session.items():
        if not session_tracks:
            continue

        # Data for hard-case capture
        session_hamming_values = []
        session_confidences = []
        session_purity_scores = []

        # Compute side_score and appearance_vector for each track if missing
        # (Usually they are derived from the best canonical image)
        for t in session_tracks:
            # We need the actual image to compute textiness/appearance
            # but they were not serialised. We'll load the best_canonical_image_path.
            import cv2
            img = cv2.imread(t["best_canonical_image_path"])
            if img is None:
                img = np.zeros((1050, 750, 3), dtype=np.uint8)
            
            t["side_score"] = float(_side_textiness_score(img))
            t["appearance_vector"] = _appearance_vector(img)

        # Find max candidates in session for normalized_length computation
        max_length = max(len(t["frame_entries"]) for t in session_tracks)

        # Primary sort: by side_score (descending) - high textiness first.
        # Secondary tie-breaker: quality-weighted composite score (descending)
        session_tracks.sort(
            key=lambda t: (
                -t["side_score"],
                -_compute_quality_weighted_score_dict(t, max_length)
            )
        )

        # The first track in the sorted session is our primary "Front"
        primary = session_tracks[0]
        primary["angle"] = "Front"
        primary["duplicate_track_index"] = None
        
        # Collect confidence and purity for hard-case detection
        session_confidences.append(primary.get("quality_score", 0.0))
        if primary.get("frame_entries"):
            purity = primary["frame_entries"][0].get("triage_metrics", {}).get("border_purity", 1.0)
            session_purity_scores.append(purity)

        # Compare others to the primary
        for other in session_tracks[1:]:
            same_card = False
            
            # 1. Try embedding-based same-card detection
            emb1 = primary.get("reid_embedding")
            emb2 = other.get("reid_embedding")
            if emb1 is not None and emb2 is not None:
                same_card = embedding_same_card_score(
                    np.array(emb1), np.array(emb2), threshold=0.5
                )
            
            # 2. Fallback to pHash if embeddings unavailable
            if not same_card:
                phash1 = primary.get("frame_entries", [{}])[0].get("visual_hash")
                phash2 = other.get("frame_entries", [{}])[0].get("visual_hash")
                if phash1 and phash2:
                    ham = deduplicator.hamming_distance(phash1, phash2)
                    session_hamming_values.append(float(ham))
                    
                    # Use ADAPTIVE threshold if available
                    adaptive_ham = adaptive_computer.compute_hamming_threshold(
                        ctx.observed_intra_track_distances,
                        global_threshold=15.0 # _SAME_CARD_HAMMING_MAX
                    )
                    same_card = ham <= adaptive_ham

            if same_card:
                # Same card → it's a "Back" (or another Front if textiness too high, but usually Back)
                other["duplicate_track_index"] = active_tracks.index(primary)
                other["angle"] = "Back"
            else:
                # Different card? In the same session? 
                other["duplicate_track_index"] = None
                other["angle"] = "Front"
                
                # Collect confidence and purity for the new "Front" candidate
                session_confidences.append(other.get("quality_score", 0.0))
                if other.get("frame_entries"):
                    purity = other["frame_entries"][0].get("triage_metrics", {}).get("border_purity", 1.0)
                    session_purity_scores.append(purity)

        # Active Learning: Check if this session is a "hard case"
        session_data = {
            "video_id": score_out.video_id,
            "session_id": str(sid),
            "front_tracks": session_tracks,
            "hamming_values": session_hamming_values,
            "confidence_scores": session_confidences,
            "border_purity_scores": session_purity_scores,
        }
        reason = is_hard_case(session_data)
        if reason:
            capture_hard_case(session_data, reason, output_file=str(Path(ctx.output_dir) / "hard_cases.jsonl"))

    return ResolveOutput(
        prepared_tracks=active_tracks,
        tracker_events=score_out.tracker_events,
        detection_rows=score_out.detection_rows,
        sampler_telemetry=score_out.sampler_telemetry,
        accepted_frame_presence=score_out.accepted_frame_presence,
        frame_count=score_out.frame_count,
        accepted_frame_count=score_out.accepted_frame_count,
        video_id=score_out.video_id,
    )


def _compute_quality_weighted_score_dict(t: Dict[str, Any], max_length: int) -> float:
    """Port of _compute_quality_weighted_score for dicts."""
    current_length = len(t["frame_entries"])
    normalized_length = current_length / max(1, max_length)
    
    # mean_quality_score of canonical entries
    canonicals = [fe for fe in t["frame_entries"] if fe["is_canonical"]]
    if not canonicals:
        mean_quality = 0.0
    else:
        mean_quality = sum(fe["quality_score"] for fe in canonicals) / len(canonicals)
        
    return 0.6 * normalized_length + 0.4 * mean_quality


def _capture_hard_cases(tracks: List[Dict[str, Any]], video_id: int, output_dir: Path):
    """Port of _capture_hard_cases_from_sessions."""
    # This logic would normally write to the hard_cases table
    # but for now we just log it or rely on the store step.
    pass
