# Runs Page Overhaul & Neural Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the runs page with a synchronized waterfall chart for stage progress andcorrelated hardware utilization (CPU, GPU, ANE).

**Architecture:** 
1. **Backend:** Extend `PipelineTelemetry` protocol with `progress(pct)`, update `EventBusTelemetry` for SSE streaming, and enhance `ResourceSampler` with Apple Neural Engine (ANE) monitoring via `powermetrics`.
2. **Frontend:** Create a new SVG `WaterfallChart` component and assemble a synchronized dashboard on the run detail page that correlates stages with utilization.

**Tech Stack:** Python 3.9, SQLite, FastAPI, SvelteKit 5, SVG.

---

### Task 1: Schema & Protocol Update

**Files:**
- Create: `migrations/0014_resource_samples_neural.sql`
- Modify: `src/card_capture/pipeline/telemetry.py`
- Test: `tests/pipeline/test_telemetry_protocol.py`

- [ ] **Step 1: Write failing test for Protocol**
```python
from card_capture.pipeline.telemetry import PipelineTelemetry

def test_telemetry_has_progress_method():
    # This will fail if PipelineTelemetry doesn't define progress()
    assert hasattr(PipelineTelemetry, "progress")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `.venv/bin/python -m pytest tests/pipeline/test_telemetry_protocol.py -v`

- [ ] **Step 3: Update PipelineTelemetry Protocol**
```python
class PipelineTelemetry(Protocol):
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
    def progress(self, stage_id: str, pct: int, detail: str) -> None: ...  # Added
    def resource_sample(self, sample: Mapping[str, object]) -> None: ...
    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None: ...
```
Also update `NoopTelemetry` and `InMemoryTelemetry` in the same file.

- [ ] **Step 4: Create migration for neural_pct**
```sql
-- migrations/0014_resource_samples_neural.sql
ALTER TABLE run_resource_samples ADD COLUMN neural_pct REAL;
```

- [ ] **Step 5: Run tests and commit**
```bash
git add migrations/0014_resource_samples_neural.sql src/card_capture/pipeline/telemetry.py
git commit -m "feat(telemetry): add progress() to protocol and neural_pct to schema"
```

---

### Task 2: EventBus Implementation & Progress SSE

**Files:**
- Modify: `app/services/pipeline_telemetry.py`
- Test: `tests/app/test_pipeline_telemetry.py`

- [ ] **Step 1: Write test for progress SSE emission**
```python
def test_event_bus_telemetry_emits_progress_event():
    from unittest.mock import MagicMock
    from app.services.pipeline_telemetry import EventBusTelemetry
    bus = MagicMock()
    tel = EventBusTelemetry(bus, "run-1")
    tel.progress("detect", 50, "batch 5/10")
    
    # Verify bus.emit was called with stage_progress
    bus.emit.assert_called()
    args = bus.emit.call_args[0]
    assert args[0] == "run-1"
    assert args[1].name == "stage_progress"
    assert args[1].payload["pct"] == 50
```

- [ ] **Step 2: Implement progress in EventBusTelemetry**
```python
    def progress(self, stage_id: str, pct: int, detail: str) -> None:
        self._emit_progress(stage_id, pct=pct, detail=detail)
```

- [ ] **Step 3: Run tests and commit**
```bash
git add app/services/pipeline_telemetry.py
git commit -m "feat(telemetry): implement progress() in EventBusTelemetry"
```

---

### Task 3: Neural Engine Sampling (macOS)

**Files:**
- Modify: `app/services/resource_sampler.py`
- Test: `tests/app/test_resource_sampler_ane.py`

- [ ] **Step 1: Write test for ANE parsing**
```python
def test_parse_powermetrics_ane():
    from app.services.resource_sampler import _parse_ane_pct
    # Mock powermetrics output snippet
    sample = "ANE Energy: 1000 mW\nANE Resampler: 25.5%"
    assert _parse_ane_pct(sample) == 25.5
```

- [ ] **Step 2: Implement powermetrics sampler**
Add `_sample_ane()` to `ResourceSampler` which runs `powermetrics` (best effort).
Add `neural_pct` to the SQL insert in `_sample()`.

- [ ] **Step 3: Run tests and commit**
```bash
git add app/services/resource_sampler.py
git commit -m "feat(sampler): collect Apple Neural Engine utilization via powermetrics"
```

---

### Task 4: Pipeline Stage Instrumentation

**Files:**
- Modify: `src/card_capture/pipeline/stages/detect.py`
- Modify: `src/card_capture/pipeline/stages/refine.py`
- Modify: `src/card_capture/pipeline/stages/fuse.py`

- [ ] **Step 1: Instrument Detect Stage**
Inside the batch loop in `detect.py`:
```python
    for i, batch in enumerate(batches):
        # ... existing code ...
        pct = int(100 * (i + 1) / len(batches))
        telemetry.progress("detect", pct, f"batch {i+1}/{len(batches)}")
```

- [ ] **Step 2: Instrument Refine & Fuse Stages**
Repeat similar `telemetry.progress` calls in the main loops of `refine.py` and `fuse.py`.

- [ ] **Step 3: Commit**
```bash
git commit -a -m "feat(pipeline): add progress reporting to detect, refine, and fuse stages"
```

---

### Task 5: Frontend API Alignment

**Files:**
- Modify: `app/services/runs_service.py`
- Modify: `app/web/src/lib/api/types.ts`

- [ ] **Step 1: Update RunService resource query**
Add `neural_pct` to `RUN_RESOURCE_SAMPLES` SQL query and the `get_run_resources` result dictionary.

- [ ] **Step 2: Update TypeScript interfaces**
Add `neural_pct?: number | null;` to `ResourceSample` in `types.ts`.

- [ ] **Step 3: Commit**
```bash
git add app/services/runs_service.py app/web/src/lib/api/types.ts
git commit -m "feat(api): expose neural_pct in run resources"
```

---

### Task 6: SVG Waterfall Chart Component

**Files:**
- Create: `app/web/src/lib/components/WaterfallChart.svelte`

- [ ] **Step 1: Implement SVG Waterfall**
Create a component that takes `stages` (start/end times) and renders horizontal bars.
Support "filling" bars for live runs.

- [ ] **Step 2: Commit**
```bash
git add app/web/src/lib/components/WaterfallChart.svelte
git commit -m "feat(ui): add WaterfallChart component for pipeline visualization"
```

---

### Task 7: Dashboard Assembly

**Files:**
- Modify: `app/web/src/routes/runs/[run_id]/+page.svelte`

- [ ] **Step 1: Synchronize Charts**
Replace existing charts with a layout where `WaterfallChart` and multiple `ResourceChart`s share the same X-axis scale.
Add a shared "hover playhead" line.

- [ ] **Step 2: Final Verification & Commit**
Verify live progress updates correctly during a run.
```bash
git add app/web/src/routes/runs/[run_id]/+page.svelte
git commit -m "feat(ui): overhaul run detail page with synchronized dashboard"
```
