"""Application-facing telemetry contract.

Implementations include a no-op for tests, an in-memory recorder for tests/
debugging, and (added in Task 1.4) an OpenTelemetry Metrics adapter.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Mapping, Protocol

logger = logging.getLogger(__name__)

# OpenTelemetry is a core dependency, but keep the import soft so that importing
# this module (NoopTelemetry/InMemoryTelemetry/CompositeTelemetry/DbTelemetry)
# never hard-fails on an environment where it is absent. Only
# ``OpenTelemetryAdapter`` actually requires it.
try:  # pragma: no cover - exercised indirectly
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status, StatusCode, set_span_in_context
except Exception:  # opentelemetry not installed
    _otel_metrics = None
    _otel_trace = None
    Status = StatusCode = set_span_in_context = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class TelemetryEvent:
    kind: str
    payload: Mapping[str, object]


class PipelineTelemetry(Protocol):
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
    def progress(self, stage_id: str, pct: int, detail: str) -> None: ...
    def resource_sample(self, sample: Mapping[str, object]) -> None: ...
    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None: ...


class NoopTelemetry:
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
    def progress(self, stage_id: str, pct: int, detail: str) -> None: ...
    def resource_sample(self, sample: Mapping[str, object]) -> None: ...
    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None: ...


class InMemoryTelemetry:
    """For tests. Not thread-safe; use one instance per run."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("stage_started", {"stage": stage, **metadata}))

    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None:
        self.events.append(
            TelemetryEvent("stage_finished", {"stage": stage, "elapsed_ms": elapsed_ms, **metadata})
        )
        
    def progress(self, stage_id: str, pct: int, detail: str) -> None:
        self.events.append(TelemetryEvent("progress", {"stage": stage_id, "pct": pct, "detail": detail}))

    def resource_sample(self, sample: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("resource_sample", dict(sample)))

    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("contract_violation", {"code": code, **metadata}))


class OpenTelemetryAdapter:
    """Publishes stage timings via OpenTelemetry Metrics and Traces.

    A run-level parent span (``pipeline.run``) is opened on construction; every
    stage span is a child of it and is stamped with ``run_id`` so a backend can
    correlate the stages of one run. Call :meth:`shutdown` exactly once at the
    end of a run (including the failure path) to end the run span, close any
    stage span left dangling by a crash, and flush the exporters.
    """

    def __init__(
        self,
        tracer: "opentelemetry.trace.Tracer",
        meter: "opentelemetry.metrics.Meter",
        run_id: str | None = None,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._run_id = run_id
        self._active_spans: dict[str, object] = {}
        self._shutdown = False

        self._stage_duration = meter.create_histogram(
            name="card_capture.pipeline.stage.duration_ms",
            description="Per-stage elapsed wall time",
            unit="ms",
        )
        self._violation_counter = meter.create_counter(
            name="card_capture.pipeline.contract_violations",
            description="Strict-contract violations recorded by the runtime",
        )
        self._resource_sample = meter.create_histogram(
            name="card_capture.pipeline.resource_sample",
            description="Generic resource sample (free-form payload via attributes)",
        )

        run_attrs = {"run_id": str(run_id)} if run_id is not None else {}
        self._run_span = tracer.start_span("pipeline.run", attributes=run_attrs)
        self._run_context = set_span_in_context(self._run_span)

    def _stage_attrs(self, metadata: Mapping[str, object]) -> dict[str, str]:
        attrs = {k: str(v) for k, v in metadata.items()}
        if self._run_id is not None:
            attrs["run_id"] = str(self._run_id)
        return attrs

    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None:
        span = self._tracer.start_span(
            stage, context=self._run_context, attributes=self._stage_attrs(metadata)
        )
        self._active_spans[stage] = span

    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None:
        # Duration is an aggregated metric: keep its attributes low-cardinality
        # (stage only). Per-stage counts vary run-to-run and would explode the
        # metric time-series, so they ride on the span (below) instead.
        self._stage_duration.record(elapsed_ms, attributes={"stage": stage})

        span = self._active_spans.pop(stage, None)
        if span:
            span.set_attributes({k: str(v) for k, v in metadata.items()})
            span.set_attribute("elapsed_ms", elapsed_ms)
            span.set_status(Status(StatusCode.OK))
            span.end()

    def progress(self, stage_id: str, pct: int, detail: str) -> None:
        pass

    def resource_sample(self, sample: Mapping[str, object]) -> None:
        numeric = next((v for v in sample.values() if isinstance(v, (int, float))), 1)
        attrs = {k: str(v) for k, v in sample.items()}
        self._resource_sample.record(numeric, attributes=attrs)

    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None:
        attrs = {"code": code, **{k: str(v) for k, v in metadata.items()}}
        self._violation_counter.add(1, attributes=attrs)
        if self._run_span is not None:
            self._run_span.add_event("contract_violation", attributes=attrs)

    def shutdown(self) -> None:
        """End the run span (and any leaked stage spans) and flush exporters."""
        if self._shutdown:
            return
        self._shutdown = True

        # A stage span still open means the stage never reported finished — i.e.
        # the run crashed mid-stage. Mark it (and the run) as failed.
        failed = bool(self._active_spans)
        for span in self._active_spans.values():
            span.set_status(Status(StatusCode.ERROR, "stage did not finish"))
            span.end()
        self._active_spans.clear()

        if self._run_span is not None:
            self._run_span.set_status(
                Status(StatusCode.ERROR) if failed else Status(StatusCode.OK)
            )
            self._run_span.end()
            self._run_span = None

        self._force_flush()

    def _force_flush(self) -> None:
        # BatchSpanProcessor / PeriodicExportingMetricReader export off-thread;
        # a short run can finish before they fire, so flush explicitly.
        providers = []
        if _otel_trace is not None:
            providers.append(_otel_trace.get_tracer_provider())
        if _otel_metrics is not None:
            providers.append(_otel_metrics.get_meter_provider())
        for provider in providers:
            flush = getattr(provider, "force_flush", None)
            if callable(flush):
                try:
                    flush()
                except Exception:  # flushing must never sink a run
                    logger.debug("telemetry force_flush failed", exc_info=True)


class CompositeTelemetry:
    """Broadcasts telemetry events to multiple underlying sinks."""

    def __init__(self, sinks: list[PipelineTelemetry]) -> None:
        self._sinks = sinks

    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None:
        for sink in self._sinks:
            try:
                sink.stage_started(stage, metadata)
            except Exception as e:
                logger.warning("Telemetry sink %s failed: %s", type(sink).__name__, e)

    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None:
        for sink in self._sinks:
            try:
                sink.stage_finished(stage, elapsed_ms, metadata)
            except Exception as e:
                logger.warning("Telemetry sink %s failed: %s", type(sink).__name__, e)
            
    def progress(self, stage_id: str, pct: int, detail: str) -> None:
        for sink in self._sinks:
            try:
                sink.progress(stage_id, pct, detail)
            except Exception as e:
                logger.warning("Telemetry sink %s failed: %s", type(sink).__name__, e)

    def resource_sample(self, sample: Mapping[str, object]) -> None:
        for sink in self._sinks:
            try:
                sink.resource_sample(sample)
            except Exception as e:
                logger.warning("Telemetry sink %s failed: %s", type(sink).__name__, e)

    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None:
        for sink in self._sinks:
            try:
                sink.contract_violation(code, metadata)
            except Exception as e:
                logger.warning("Telemetry sink %s failed: %s", type(sink).__name__, e)

