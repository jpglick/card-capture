"""OtelMetricsTelemetry records measurements via an InMemoryMetricReader."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.metrics")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from card_capture.pipeline.telemetry import OtelMetricsTelemetry


def test_stage_finished_emits_histogram():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    sink = OtelMetricsTelemetry(meter=provider.get_meter("card_capture.pipeline"))
    sink.stage_started("detect", {})
    sink.stage_finished("detect", elapsed_ms=42, metadata={"frames": 100})

    metrics = reader.get_metrics_data()
    names = []
    for rm in metrics.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                names.append(m.name)
    assert "card_capture.pipeline.stage.duration_ms" in names
