"""Stage 4: Background Novelty Gate."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np

from card_capture.stages.novelty.background_novelty import BackgroundModel, quad_novelty
from card_capture.shared.stage_metrics import emit_stage_metrics


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    detect_out = state["detections"]
    sampler = state.get("sampler")
    
    bg_images = getattr(sampler, "background_proxies", []) if sampler else []
    bg_model: Optional[BackgroundModel] = None

    if bg_images:
        bg_model = BackgroundModel.from_frames(bg_images)
        state["bg_model"] = bg_model

    scored_rows: List[Dict[str, Any]] = []
    
    for row in detect_out:
        scored_row = dict(row)
        
        # Respect existing novelty_score if present (MPS fast path)
        if "novelty_score" in row:
            scored_rows.append(scored_row)
            continue

        scored_row["novelty_score"] = 1.0  # default: fully novel

        # In-process runtime has frames in memory
        frame_idx = row.get("frame_index")
        frames = state.get("sampled_frames", [])
        source_frame = next((f.image for f in frames if f.frame_index == frame_idx), None)

        if bg_model is not None and row.get("corners") and source_frame is not None:
            corners = [(float(x), float(y)) for x, y in row["corners"]]
            novelty_val = quad_novelty(
                source_frame, corners, bg_model,
                color_space="lab", lab_weights=(1.0, 0.5, 0.5),
            )
            scored_row["novelty_score"] = float(novelty_val)

        scored_rows.append(scored_row)

    state["novelty_scored_detections"] = scored_rows
    emit_stage_metrics(state, stage="novelty", metrics={"scored": len(scored_rows)})
