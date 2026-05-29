"""Stage 10: Global Dedup."""
from __future__ import annotations


def run(state: dict, *, telemetry) -> None:
    # In a full implementation we would use dedupe_fused
    state["final_cards"] = state.get("fused", [])
