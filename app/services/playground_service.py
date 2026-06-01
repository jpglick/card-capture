"""Service layer for the Threshold Playground.

Allows recomputing pipeline stages in-memory using v5.5 local state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from card_capture.pipeline.stages import novelty, track, score, resolve, fuse


class PlaygroundService:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_run_artifacts(self, run_id: str) -> Dict[str, Any]:
        """Load artifacts from a completed run."""
        raise NotImplementedError("Metaflow artifacts are no longer supported in V5.5.")

    def recompute(self, run_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recompute novelty -> score stages with new thresholds."""
        artifacts = self.get_run_artifacts(run_id)
        ctx = artifacts["run_context"]
        detect_out = artifacts["detect_out"]

        # Apply overrides to context (in-memory only)
        for k, v in overrides.items():
            if hasattr(ctx, k):
                setattr(ctx, k, v)
        
        # 1. Re-run Novelty
        novelty_out = novelty.run(ctx, detect_out)
        
        # 2. Re-run Track
        track_out = track.run(ctx, novelty_out)
        
        # 3. Re-run Score
        # (Note: Score currently depends on Refine results being in track_out, 
        # but if we only change track thresholds, we might need a more complex
        # state management. For the playground, we focus on Gating.)
        
        # Simplified metrics for the playground
        metrics = {
            "card_count": len(track_out.tracks_data),
            "tracker_t_high": track_out.sampler_telemetry.get("tracker_t_high"),
            "tracker_t_low": track_out.sampler_telemetry.get("tracker_t_low"),
        }
        
        return {
            "run_id": run_id,
            "metrics": metrics,
            "tracks": track_out.tracks_data[:20], # Sample for UI
        }
