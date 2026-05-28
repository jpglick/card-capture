"""Stage 10b: Storage.

In Phase 3 this stage still writes via card_capture.storage. Phase 4
will replace direct SQL with card_capture.data repositories.
"""
from __future__ import annotations


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    
    # In a full implementation we would use store_cards
    # out_paths = store_cards(state["final_cards"], request.output_root, config=request.config)
    
    state["cards"] = state.get("final_cards", [])
    state["output_artifacts"] = []
