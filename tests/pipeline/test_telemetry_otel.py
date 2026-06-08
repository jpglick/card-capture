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
from opentelemetry.trace.status import StatusCode

from card_capture.pipeline.telemetry import OpenTelemetryAdapter


def _harness():
    """Return (tracer, meter, span_exporter) backed by in-memory readers."""
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return (
        tracer_provider.get_tracer("card_capture"),
        meter_provider.get_meter("card_capture.pipeline"),
        span_exporter,
    )


def test_stage_finished_emits_histogram():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    tracer_provider = TracerProvider()

    sink = OpenTelemetryAdapter(
        tracer=tracer_provider.get_tracer("card_capture"),
        meter=provider.get_meter("card_capture.pipeline"),
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
    tracer, meter, span_exporter = _harness()
    sink = OpenTelemetryAdapter(tracer=tracer, meter=meter)

    sink.stage_started("detect", {"custom": "data"})
    sink.stage_finished("detect", elapsed_ms=100, metadata={"result": "ok"})

    spans = span_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["detect"]


def test_stage_spans_are_children_of_run_span_and_carry_run_id():
    tracer, meter, span_exporter = _harness()
    sink = OpenTelemetryAdapter(tracer=tracer, meter=meter, run_id="run_42")

    sink.stage_started("detect", {})
    sink.stage_finished("detect", elapsed_ms=10, metadata={})
    sink.shutdown()

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert "pipeline.run" in spans
    assert "detect" in spans
    run_span = spans["pipeline.run"]
    detect = spans["detect"]
    # detect span is a child of the run span
    assert detect.parent is not None
    assert detect.parent.span_id == run_span.context.span_id
    assert detect.context.trace_id == run_span.context.trace_id
    # run_id is stamped on both spans for correlation
    assert detect.attributes["run_id"] == "run_42"
    assert run_span.attributes["run_id"] == "run_42"


def test_dangling_stage_span_is_ended_with_error_on_shutdown():
    tracer, meter, span_exporter = _harness()
    sink = OpenTelemetryAdapter(tracer=tracer, meter=meter, run_id="run_1")

    sink.stage_started("refine", {})
    # Simulate a stage crash: stage_finished is never called.
    sink.shutdown()

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    # The unfinished stage span must still be ended (no leak) and marked failed.
    assert spans["refine"].status.status_code == StatusCode.ERROR
    # The run span inherits the failure.
    assert spans["pipeline.run"].status.status_code == StatusCode.ERROR


def test_shutdown_is_idempotent():
    tracer, meter, span_exporter = _harness()
    sink = OpenTelemetryAdapter(tracer=tracer, meter=meter, run_id="run_1")
    sink.stage_started("detect", {})
    sink.stage_finished("detect", elapsed_ms=5, metadata={})
    sink.shutdown()
    sink.shutdown()  # second call must not raise or double-end the run span

    run_spans = [s for s in span_exporter.get_finished_spans() if s.name == "pipeline.run"]
    assert len(run_spans) == 1


def test_counts_go_on_span_not_on_duration_histogram_attributes():
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    sink = OpenTelemetryAdapter(
        tracer_provider.get_tracer("card_capture"),
        meter_provider.get_meter("card_capture.pipeline"),
        run_id="run_1",
    )
    sink.stage_started("detect", {})
    sink.stage_finished("detect", elapsed_ms=12, metadata={"detections": 42})

    # The count is recorded on the span (an event — rich attributes are fine).
    span = {s.name: s for s in span_exporter.get_finished_spans()}["detect"]
    assert span.attributes["detections"] == "42"

    # The duration histogram is an aggregated metric: the count must NOT be an
    # attribute (it would explode time-series cardinality). Only `stage`.
    points = []
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == "card_capture.pipeline.stage.duration_ms":
                    points.extend(m.data.data_points)
    assert points, "no duration_ms data points"
    attrs = dict(points[0].attributes)
    assert attrs.get("stage") == "detect"
    assert "detections" not in attrs
