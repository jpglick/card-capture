"""Stage 10b: Storage via repository."""
from __future__ import annotations


def run(state: dict, *, telemetry) -> None:
    cards_repo = state["repos"]["cards"]
    runs_repo = state["repos"]["runs"]

    # In a full implementation we would generate actual CardRecords from state["final_cards"]
    # For now we just pass an empty list or placeholders to satisfy the API
    final_cards = []
    
    cards_repo.store_final_cards(
        run_id=state["request"].run_id,
        video_id=state.get("video_id", 0),  # fallback to 0 if not set
        cards=final_cards
    )
    runs_repo.mark_completed(state["request"].run_id, cards_extracted=len(final_cards))
    state["cards"] = final_cards
    state["output_artifacts"] = []  # populated by export-boundary helpers, not store
