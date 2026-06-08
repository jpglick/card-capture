"""OpenTelemetryAdapter records traces and measurements."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.metrics")
pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from card_capture.pipeline.telemetry import OpenTelemetryAdapter

def test_stage_finished_emits_histogram():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    tracer_provider = TracerProvider()
    
    sink = OpenTelemetryAdapter(
        tracer=tracer_provider.get_tracer("card_capture"),
        meter=provider.get_meter("card_capture.pipeline")
    )
    sink.stage_started("detect", {})
    sink.stage_finished("detect", elapsed_ms=42, metadata={"frames": 100})

    metrics = reader.get_metrics_data()
    names = []
    for rm in metrics.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                names.append(m.name)
    assert "card_capture.pipeline.stage.duration_ms" in names

def test_stage_emits_trace_span():
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    
    sink = OpenTelemetryAdapter(
        tracer=tracer_provider.get_tracer("card_capture"),
        meter=meter_provider.get_meter("card_capture.pipeline")
    )
    
    sink.stage_started("detect", {"custom": "data"})
    sink.stage_finished("detect", elapsed_ms=100, metadata={"result": "ok"})
    
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "detect"
