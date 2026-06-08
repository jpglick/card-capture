"""Helpers for writing per-stage diagnostic metrics."""
from __future__ import annotations

from typing import Mapping


def emit_stage_metrics(state: dict, *, stage: str, metrics: Mapping[str, object]) -> None:
    # Buffer for the telemetry bridge: LocalPipelineRuntime drains this into the
    # stage_finished(...) metadata. Written unconditionally so the bridge works
    # even when no events repo is wired (e.g. CLI / tests).
    state.setdefault("stage_metrics", {}).setdefault(stage, {}).update(metrics)

    repos = state.get("repos") or {}
    events_repo = repos.get("events")
    request = state.get("request")
    run_id = getattr(request, "run_id", None)
    if events_repo is None or not run_id:
        return

    events_repo.record_stage_metrics(
        run_id=str(run_id),
        video_id=state.get("video_id"),
        stage=stage,
        metrics=metrics,
    )

