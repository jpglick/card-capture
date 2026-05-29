"""PipelineTelemetry callers can swap implementations without code changes."""
from __future__ import annotations

from card_capture.pipeline.telemetry import (
    PipelineTelemetry,
    NoopTelemetry,
    InMemoryTelemetry,
)


def test_noop_satisfies_protocol():
    sink: PipelineTelemetry = NoopTelemetry()
    sink.stage_started("detect", {})
    sink.stage_finished("detect", 1234, {"frames": 100})
    sink.resource_sample({"vram_mb": 4096})
    sink.contract_violation("cpu_read_in_strict", {"call_site": "f.py:1"})


def test_inmemory_records_events_in_order():
    sink = InMemoryTelemetry()
    sink.stage_started("detect", {})
    sink.stage_finished("detect", 1234, {"frames": 100})
    sink.resource_sample({"vram_mb": 4096})
    sink.contract_violation("cpu_read_in_strict", {})
    kinds = [e.kind for e in sink.events]
    assert kinds == ["stage_started", "stage_finished", "resource_sample", "contract_violation"]
