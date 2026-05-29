# Design Spec: Unified Run Timeline & Neural Telemetry

**Date:** 2026-05-29  
**Topic:** Overhauling the runs page with synchronized stage-vs-utilization visualizations and Apple Neural Engine (ANE) monitoring.

## 1. Problem Statement
The current runs page displays stage progress as a simple list of progress bars that flip from 0% to 100% with no mid-stage fidelity. Hardware utilization graphs (CPU/GPU) are disconnected from the pipeline timeline, making it difficult to correlate performance bottlenecks (e.g., "Why did the GPU stall here?"). Additionally, Neural Engine utilization—critical for YOLO inference on Apple Silicon—is not tracked.

## 2. Goals
- **Synchronized Visualization:** A waterfall/Gantt chart for stages aligned horizontally with resource utilization graphs.
- **Mid-Stage Fidelity:** Pipeline stages report real-time progress percentages (e.g., batch 5/20).
- **ANE Monitoring:** Best-effort collection of Apple Neural Engine utilization on macOS.
- **Improved UI Density:** Reduce vertical whitespace by consolidating progress and utilization into a single dashboard.

## 3. Architecture

### 3.1 Telemetry Layer (`PipelineTelemetry`)
- **Protocol Update:** Add `progress(stage_id: str, pct: int, detail: str)` to the protocol.
- **EventBus Implementation:** `EventBusTelemetry` will fire `stage_progress` events immediately to the UI via SSE.
- **Overall Progress:** Re-calculate "Total Pipeline Progress" as a weighted average of stage percentages (or a simple sum of progress increments).

### 3.2 Resource Sampling (`ResourceSampler`)
- **Schema Update:** Add `neural_pct` column to the `run_resource_samples` table.
- **ANE Collection (macOS):** 
    - Launch `powermetrics` as a non-blocking subprocess if possible.
    - Fallback: If `sudo` is required and missing, report `neural_pct` as `null` (N/A in UI).
- **GPU Synchronization:** Ensure `elapsed_s` used for resource samples perfectly matches the `created_at` timestamps of stage events.

### 3.3 UI Dashboard (`app/web`)
- **Waterfall Component:** A new SVG-based timeline where:
    - Rows = Stages (Sample, Detect, etc.).
    - X-Axis = Time (Seconds).
    - Bars = Colored segments representing stage duration, with internal fills showing progress.
- **Utilization Component:** Line charts for CPU, GPU, ANE, and RAM sharing the **exact same X-axis scale** as the Waterfall chart.
- **Interaction:** A vertical playhead/scrubber that highlights the data values across all charts simultaneously.

## 4. Proposed Approaches for ANE Collection
- **Best Effort Powermetrics:** Spawn `sudo powermetrics -i 2000 --samplers resampler -n 1`. If it fails, the sampler continues without ANE data.
- **UI Fallback:** If ANE data is missing, the "Neural Engine" chart is hidden or replaced with a "N/A" placeholder.

## 5. Success Criteria
- [ ] UI displays a single, synchronized timeline containing both stage bars and utilization lines.
- [ ] Stages (Detect, Refine, Fuse) report progress smoothly from 0-100% during execution.
- [ ] On supported M-series hardware with permissions, the ANE utilization graph is populated.
- [ ] Overall pipeline progress percentage reflects actual stage completion rather than just a step function.

## 6. Testing Strategy
- **Unit Tests:** Verify `EventBusTelemetry.progress` correctly formats SSE payloads.
- **Integration Tests:** Run the `test_back_half_e2e` and verify `run_resource_samples` contains valid percentages (and `neural_pct` if on Mac).
- **UI Validation:** Mock SSE events with varied percentages and durations to ensure the Waterfall chart renders correctly.
