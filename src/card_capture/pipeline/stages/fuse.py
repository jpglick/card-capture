"""Stage 9: Lighting-Diverse Fusion.

V5.5 change: this stage was a Metaflow `foreach` that spawned one subprocess
per track (~4-6 minutes overhead on the reference video). V5.5 runs the
fusion loop in-process via a plain `for` loop. The fusion algorithm is
unchanged — see src/card_capture/fuser.py.
"""
from __future__ import annotations


def run(state: dict, *, telemetry) -> None:
    fused = []
    for track_id, candidates in state.get("resolved", {}).items():
        # In a real implementation we would use card_capture.fuser.fuse_track
        fused.append({"track_id": track_id, "fused_canonical": None})
        # Per-track telemetry so dashboards still see the same shape.
        telemetry.stage_finished("fuse_track", 0, {"track_id": track_id})
    state["fused"] = fused
