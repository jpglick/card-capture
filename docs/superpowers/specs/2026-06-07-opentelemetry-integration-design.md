# OpenTelemetry Integration Design

## Overview
This document outlines the plan to integrate rich OpenTelemetry (OTel) instrumentation (spans and metrics) into the `card-capture` pipeline, alongside fulfilling the original design of local SQLite telemetry storage. Currently, telemetry events are sent to the EventBus for the UI, but OTel logic was stubbed out or disabled, and the `telemetry_events` SQLite table remains unpopulated.

## Architectural Changes

### 1. `CompositeTelemetry` Adapter
To allow multiple sinks (EventBus, OTel, local DB) to receive telemetry simultaneously without coupling the core pipeline to any single implementation, we will introduce a `CompositeTelemetry` multiplexer.
- **Location:** `src/card_capture/pipeline/telemetry.py`
- **Behavior:** Implements the `PipelineTelemetry` protocol. Accepts a list of `PipelineTelemetry` instances in its constructor and synchronously broadcasts all events (`stage_started`, `stage_finished`, `progress`, `resource_sample`, `contract_violation`) to every registered adapter.

### 2. `OpenTelemetryAdapter`
We will expand the existing, unused `OtelMetricsTelemetry` class into a full metrics and tracing adapter.
- **Rename:** `OtelMetricsTelemetry` will be renamed to `OpenTelemetryAdapter` in `src/card_capture/pipeline/telemetry.py`.
- **Initialization:** It will be updated to require both an `opentelemetry.metrics.Meter` and an `opentelemetry.trace.Tracer`.
- **Tracing Logic:**
  - `stage_started`: Starts a new trace span (e.g., `tracer.start_span(stage)`). The active span will be stored in an internal dictionary (`dict[str, Span]`) keyed by the stage name.
  - `stage_finished`: Pops the span for the given stage from the dictionary, sets attributes (like `elapsed_ms` and other metadata), and ends the span.
- **Metrics Logic:** Existing metric histograms and counters for durations, contract violations, and resource samples will be preserved.

### 3. `DbTelemetry` Adapter
We will create a new adapter to write telemetry to the pre-existing SQLite `telemetry_events` table.
- **Location:** Likely in `app/services/pipeline_telemetry.py` or alongside repositories.
- **Behavior:** Implements the `PipelineTelemetry` protocol. Wraps the existing `TelemetryRepository.record_event` method.
- **Data Mapping:** Each protocol method will call `record_event(run_id, kind, payload)`, translating the specific telemetry events (e.g., `kind="stage_started"`, `payload={"stage": stage, **metadata}`).

### 4. Integration in `PipelineRunner`
The web application's pipeline runner will be updated to wire all these pieces together.
- **Location:** `app/services/pipeline_runner.py` (specifically `_run_unified_inprocess`).
- **Wiring:** Instead of passing a single `EventBusTelemetry` instance to the `LocalPipelineRuntime`, we will instantiate a `CompositeTelemetry` that includes:
  - `EventBusTelemetry` (existing)
  - `OpenTelemetryAdapter` (new, initialized with the global tracer/meter)
  - `DbTelemetry` (new, initialized with the run's DB context)
- Global OTel provider initialization will be handled at the FastAPI app lifecycle level (`app/main.py`).

## Testing Strategy
- Update existing telemetry tests in `tests/pipeline/test_telemetry_otel.py` to cover tracing behavior and the rename.
- Add unit tests for `CompositeTelemetry` to ensure it broadcasts correctly and handles exceptions gracefully (if one sink fails, it shouldn't crash the pipeline).
- Add unit tests for `DbTelemetry`.
