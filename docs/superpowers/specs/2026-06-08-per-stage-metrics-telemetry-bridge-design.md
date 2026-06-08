# Per-Stage Metrics → Telemetry Bridge

**Date:** 2026-06-08
**Status:** Design approved, pending spec review

## Problem

Every pipeline stage already computes count-style metrics and persists them via
`emit_stage_metrics(state, stage=…, metrics={…})`, which writes a
`stage_metrics` row to the `pipeline_events` table. But those metrics never reach
the `PipelineTelemetry` contract: `LocalPipelineRuntime.run` calls
`stage_finished(name, elapsed_ms, {})` with empty metadata. As a result OTel
spans and `telemetry_events` rows carry only `stage` + `elapsed_ms` — the counts
the pipeline already knows are invisible on the telemetry/trace path.

Existing per-stage metrics (all via `emit_stage_metrics`):

| Stage | Metrics |
|-------|---------|
| sample | `estimated_frames` |
| detect | `detections` |
| novelty | `scored` |
| track | (metrics dict) |
| refine | `refined_tracks` |
| score | `scored_tracks`, `pruned_tracks` |
| resolve | `sessions`, `prepared_tracks` |
| fuse | `fused_canonicals` |
| dedup | `dedup_groups`, `final_cards` |
| store | `final_cards` |

## Goal / Non-Goals

**Goal:** Route the metrics each stage already emits into `stage_finished`
metadata so they enrich OTel span attributes and `telemetry_events` rows.

**Non-Goals:**
- No new metric types (memory, throughput, device) — counts only, as they exist
  today. (Tracked as a possible follow-up.)
- No change to the existing `pipeline_events` `stage_metrics` path (no
  regression to current consumers).
- No new per-stage code; stages keep calling `emit_stage_metrics` unchanged.
- No UI/SSE changes.

## Design

Approach: **buffer in `emit_stage_metrics`, drain in the runtime.** Three small
changes.

### 1. `shared/stage_metrics.emit_stage_metrics` — buffer into state

After the existing `events_repo.record_stage_metrics(...)` call, additionally
stash the metrics in an in-memory buffer on `state`:

```python
state.setdefault("stage_metrics", {}).setdefault(stage, {}).update(metrics)
```

- Merges if a stage emits more than once (last value wins per key).
- Pure in-memory; no new dependency; the events-table write is untouched.

### 2. `LocalPipelineRuntime.run` — pass buffered metrics to `stage_finished`

Replace the empty metadata:

```python
metrics = state.get("stage_metrics", {}).get(name, {})
self._telemetry.stage_finished(name, elapsed_ms, metrics)
```

`stage_started` stays `{}` — counts are not known at stage start. If a stage
emits no metrics, metadata is `{}` (unchanged behavior).

### 3. `OpenTelemetryAdapter.stage_finished` — keep the duration metric low-cardinality

The bridge forces one correctness fix. Today the duration **histogram** is
recorded with `{"stage", **metadata}`. Once metadata carries varying counts
(`detections=42`, …), each distinct value spawns a new metric time-series —
cardinality explosion in any metrics backend. Fix:

- **Span:** carries the full metadata (spans are per-event; rich attributes are
  fine). Unchanged.
- **Histogram:** record with only the low-cardinality `{"stage": stage}`.

## Data Flow

```
stage.run(state)
  └─ emit_stage_metrics(state, stage, metrics)
        ├─ events_repo.record_stage_metrics(...)   # pipeline_events (existing)
        └─ state["stage_metrics"][stage].update(metrics)   # new buffer

LocalPipelineRuntime.run (after stage returns)
  └─ stage_finished(name, elapsed_ms, state["stage_metrics"][name])
        └─ CompositeTelemetry fans out:
             ├─ OpenTelemetryAdapter → span attrs (rich) + duration histogram (stage only)
             ├─ DbTelemetry          → telemetry_events row (payload = metrics)
             └─ EventBusTelemetry     → SSE (metadata ignored; unchanged)
```

## Edge Cases

- **Stage emits no metrics:** metadata `{}`, same as today.
- **Stage fails mid-run:** `stage_finished` is not called (exception path);
  OTel span is closed as ERROR by `shutdown()`. No change.
- **Multiple emits per stage:** merged via `dict.update`.
- **Non-scalar metric values:** current metrics are JSON-able scalars; OTel
  stringifies, `DbTelemetry` json-dumps. No special handling needed.
- **Thread safety:** stages run sequentially in the runtime loop; the `state`
  buffer is mutated only on that thread.

## Testing

- `emit_stage_metrics` writes metrics into `state["stage_metrics"][stage]` and
  merges across calls (unit).
- `OpenTelemetryAdapter.stage_finished`: counts appear as span attributes but
  the duration histogram's attributes are limited to `stage` (unit, in-memory
  span/metric readers).
- Runtime bridge: a stub stage that calls `emit_stage_metrics` results in the
  metrics reaching `stage_finished`, asserted via `InMemoryTelemetry`.

## Out of Scope / Follow-ups

- New per-stage resource metrics (peak/available memory, throughput fps,
  device) — valuable for the known refine-OOM and novelty-bottleneck issues,
  but a separate effort.
- Reconciling the two persistence paths (`pipeline_events.stage_metrics` vs
  `telemetry_events`) — both remain for now.
