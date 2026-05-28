"""Stage 7: Quality Scoring."""
from __future__ import annotations


def run(state: dict, *, telemetry) -> None:
    # In a full implementation, we would use QualityScorer or to_cpu_for_score
    # For now, just pass through the tracks as scored
    state["scored"] = state.get("tracks", {})
