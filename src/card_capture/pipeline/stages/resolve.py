"""Stage 8: Front/Back Resolution."""
from __future__ import annotations


def run(state: dict, *, telemetry) -> None:
    # In a full implementation, we'd use resolve_front_back.
    # For now, just pass through the scored tracks.
    scored = state.get("scored", {})
    resolved = {}
    for track_id, track in scored.items() if isinstance(scored, dict) else enumerate(scored):
        resolved[track_id] = track
    state["resolved"] = resolved
