"""Application-facing telemetry contract.

Implementations include a no-op for tests, an in-memory recorder for tests/
debugging, and (added in Task 1.4) an OpenTelemetry Metrics adapter.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping, Protocol


@dataclasses.dataclass(frozen=True)
class TelemetryEvent:
    kind: str
    payload: Mapping[str, object]


class PipelineTelemetry(Protocol):
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
    def resource_sample(self, sample: Mapping[str, object]) -> None: ...
    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None: ...


class NoopTelemetry:
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
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

    def resource_sample(self, sample: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("resource_sample", dict(sample)))

    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("contract_violation", {"code": code, **metadata}))
