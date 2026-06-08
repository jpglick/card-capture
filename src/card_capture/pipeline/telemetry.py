"""Application-facing telemetry contract.

Implementations include a no-op for tests, an in-memory recorder for tests/
debugging, and (added in Task 1.4) an OpenTelemetry Metrics adapter.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Mapping, Protocol

logger = logging.getLogger(__name__)


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


class OtelMetricsTelemetry:
    """Publishes stage timings and counters via OpenTelemetry Metrics.

    Traces and span sampling are intentionally out of scope (see spec
    Non-Goals). This adapter records histograms/counters only.
    """

    def __init__(self, meter: "opentelemetry.metrics.Meter") -> None:
        self._meter = meter
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

    def stage_started(self, stage: str, metadata: Mapping[str, object]):
        # Stage start is metadata only; durations are recorded on finish.
        pass

    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]):
        attrs = {"stage": stage, **{k: str(v) for k, v in metadata.items()}}
        self._stage_duration.record(elapsed_ms, attributes=attrs)
        
    def progress(self, stage_id: str, pct: int, detail: str) -> None:
        pass

    def resource_sample(self, sample: Mapping[str, object]):
        # Record any single numeric field if present; otherwise a count of 1.
        numeric = next((v for v in sample.values() if isinstance(v, (int, float))), 1)
        attrs = {k: str(v) for k, v in sample.items()}
        self._resource_sample.record(numeric, attributes=attrs)

    def contract_violation(self, code: str, metadata: Mapping[str, object]):
        attrs = {"code": code, **{k: str(v) for k, v in metadata.items()}}
        self._violation_counter.add(1, attributes=attrs)


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

