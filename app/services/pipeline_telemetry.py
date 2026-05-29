"""EventBus adapter for the unified-runtime PipelineTelemetry contract.

Bridges stage events emitted by ``card_capture.pipeline.runtime_local`` to the
UI's SSE EventBus. Also funnels them to ``logging`` so the existing log
persistence path (which scrapes stdout in the metaflow days) still gets
something useful written to ``pipeline_run_logs``.

All emit calls are thread-safe — EventBus.emit handles the thread→loop hop
internally — so this adapter is safe to invoke from the worker thread the
runtime executes stages on.
"""
from __future__ import annotations

import logging
import time
from typing import Mapping

from app.services.event_bus import Event, EventBus


_STAGES = (
    "sample", "detect", "novelty", "track", "refine",
    "score", "resolve", "fuse", "dedup", "store",
)


class EventBusTelemetry:
    """Implements ``card_capture.pipeline.telemetry.PipelineTelemetry``."""

    def __init__(
        self,
        bus: EventBus,
        run_id: str,
        *,
        logger: logging.Logger | None = None,
        log_sink: "callable | None" = None,
    ) -> None:
        self._bus = bus
        self._run_id = run_id
        self._logger = logger or logging.getLogger(f"pipeline.{run_id}")
        # Optional: called with each user-facing log line so callers (the UI
        # PipelineRunner) can persist them to pipeline_run_logs alongside the
        # SSE emission. Failures here must never sink the pipeline.
        self._log_sink = log_sink
        self._started_at: float | None = None
        self._completed: set[str] = set()

    # --- PipelineTelemetry protocol ---

    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None:
        if self._started_at is None:
            self._started_at = time.monotonic()
        line = f"[stage] {stage} started"
        self._log(line)
        self._bus.emit(self._run_id, Event(name="log", payload={"line": line}))
        # Use the same "stage_progress" event the SSE client already listens
        # to. We can't report a real percentage mid-stage, but firing 0% at
        # start + 100% at finish keeps the UI's stage list stepping.
        self._emit_progress(stage, pct=0, detail="started")

    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None:
        self._completed.add(stage)
        line = f"[stage] {stage} finished in {elapsed_ms}ms"
        self._log(line)
        self._bus.emit(self._run_id, Event(name="log", payload={"line": line}))
        self._emit_progress(stage, pct=100, detail=f"elapsed_ms={elapsed_ms}")
        # Also emit an overall pipeline-progress signal based on stage count.
        overall = int(100 * len(self._completed) / len(_STAGES))
        self._bus.emit(
            self._run_id,
            Event(name="stage_progress", payload={
                "stage_id": "pipeline",
                "pct": overall,
                "detail": f"{len(self._completed)}/{len(_STAGES)} stages",
            }),
        )

    def resource_sample(self, sample: Mapping[str, object]) -> None:
        # Resource samples are noisy; only forward as logs at debug level.
        self._logger.debug("resource_sample %s", dict(sample))

    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None:
        line = f"[violation] {code} {dict(metadata)}"
        self._log(line, level=logging.ERROR)
        self._bus.emit(self._run_id, Event(name="log", payload={"line": line}))

    # --- helpers ---

    def _emit_progress(self, stage: str, *, pct: int, detail: str) -> None:
        self._bus.emit(
            self._run_id,
            Event(name="stage_progress", payload={
                "stage_id": stage,
                "pct": pct,
                "detail": detail,
            }),
        )

    def _log(self, line: str, *, level: int = logging.INFO) -> None:
        self._logger.log(level, line)
        if self._log_sink is not None:
            try:
                self._log_sink(line)
            except Exception:
                # Persisting telemetry must never kill the pipeline.
                self._logger.debug("log_sink failed for line=%r", line, exc_info=True)
