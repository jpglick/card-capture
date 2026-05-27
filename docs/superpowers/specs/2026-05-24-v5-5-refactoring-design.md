# V5.5 Refactoring Design

**Date:** 2026-05-24  
**Status:** Draft  
**Scope:** Software engineering refactors for pipeline correctness, performance enforcement, and runtime portability

---

## Problem

The V4 pipeline has accumulated optimization work inside boundaries that do not enforce the properties we care about. Metaflow is used as a subprocess-oriented orchestration shell, the app runner parses logs for timing data, CPU/GPU behavior is selected through scattered flags and conditional code paths, and runtime-specific concerns leak across app, worker, and pipeline modules.

The result is that performance regressions are too easy to introduce. Hidden CPU reads, implicit tensor copies, silent fallback behavior, and repeated decode/load work can all appear as local implementation details instead of contract violations. V5.5 should make those mistakes structurally difficult.

This document focuses on software engineering refactoring. It does not propose new card-detection, tracking, scoring, fusion, deduplication, or OCR algorithms.

---

## Goals

- Make pipeline execution contracts explicit enough that hidden CPU work and repeated load/decode work fail fast.
- Treat Metaflow as orchestration and artifact lineage, not as the enforcement layer for GPU memory residency.
- Replace scattered CPU/GPU conditionals with runtime interfaces and backend implementations.
- Replace scattered SQL access with an explicit data access layer and ownership rules.
- Separate runner, platform, telemetry, and pipeline concerns so Beam, RunPod, local CUDA, and CPU debug can share the same contract.
- Preserve current algorithmic behavior while improving architecture, testability, and operational clarity.
- Produce an architectural standards document that AI agents and human contributors must follow when changing the pipeline.

---

## Non-Goals

- No algorithm redesigns.
- No quality metric or threshold tuning.
- No new model training work.
- No requirement to pass GPU tensors as Metaflow artifacts.
- No attempt to make CPU fallback transparent in production.

---

## Design Principles

- **Contracts over conventions.** If production GPU execution must stay on GPU, represent that through types, interfaces, guards, and tests.
- **Explicit debug modes.** CPU debug should be a selected backend, not a silent fallback path.
- **Coarse orchestration, strict execution.** Metaflow should call well-defined execution units and receive serializable artifacts, while runtime-specific execution rules are enforced inside those units.
- **Single ownership of telemetry.** Timing and resource metrics should be emitted through structured telemetry, not scraped from stdout.
- **Platform adapters at the edge.** Beam, RunPod, Vast.ai, and local execution should differ at runner and transport boundaries, not inside pipeline logic.
- **Agent-safe architecture.** Rules that are easy for AI agents to violate must be written as explicit standards and backed by tests where possible.

---

## Deliverable: Architectural Standards Document

V5.5 should produce a durable standards document that is treated as binding project guidance for AI agents and human contributors.

Proposed file:

```text
docs/architecture/standards.md
```

This document should be concise, prescriptive, and written in "must / must not" language. It should not be a long design essay. Its job is to answer: "When an agent edits this codebase, what architecture rules must it obey?"

### Required Contents

The standards document should cover:

- Pipeline orchestration rules.
- Runtime backend rules.
- Strict GPU execution rules.
- CPU debug rules.
- Data access rules.
- Telemetry rules.
- Platform adapter rules.
- Testing expectations for architectural boundaries.
- CI/architecture-lint expectations.
- A short "when unsure" escalation rule for agents.

### Example Standards

Initial standards should include rules such as:

- Metaflow flow files must orchestrate steps and artifacts only; they must not contain algorithmic logic, direct SQL, provider upload/download logic, or stdout timing parsers.
- Production GPU execution must use the strict GPU runtime; it must not silently fallback to CPU.
- CPU execution must be selected through an explicit CPU debug backend.
- GPU hot-path modules must not call `cv2.VideoCapture`, `cv2.imread`, `PIL.Image.open`, `torch.Tensor.cpu`, or `torch.Tensor.numpy` except through approved boundary modules.
- App services, pipeline steps, platform runners, and Metaflow flows must not open SQLite connections directly.
- Raw SQL must live in the data access layer, migrations/schema code, or allowlisted test helpers.
- Platform-specific code must live in provider adapters and must communicate through runner contracts and manifests.
- Timing and resource metrics must be emitted through structured telemetry interfaces, not parsed from logs.

### Enforcement

The standards document should not rely on memory or reviewer discipline alone. Each rule should indicate one of:

- **Static enforcement:** import lint, raw SQL scan, forbidden API scan, package boundary checks.
- **Runtime enforcement:** strict GPU guard, telemetry contract checks, manifest validation.
- **Review enforcement:** rules that require human judgment until automated checks exist.

### Agent Usage

AI agents working in this repo should be instructed to read the standards document before making changes that touch:

- `pipeline/`
- `src/card_capture/pipeline_utils.py`
- GPU/runtime modules.
- app runner or provider modules.
- storage/data access code.
- CI or test architecture.

If a requested change appears to violate a standard, the agent should stop and surface the conflict rather than silently working around the rule.

---

## Section 1: Metaflow Usage Refactor

### Current Issues

- The app invokes Metaflow as a subprocess and parses stdout for step timings.
- Stage timing is duplicated between flow code, runner diagnostics, and handler diagnostics.
- The flow mostly behaves as a linear script wrapper, so Metaflow's graph, artifact, and runtime metadata capabilities are underused.
- Runtime decisions such as CPU debug versus GPU execution are represented indirectly through config files, environment variables, and detector names.

### Target Shape

Metaflow should own pipeline orchestration:

- Step graph.
- Resume and retry boundaries.
- Serializable artifacts.
- Structured step-level metadata.
- Conditional branch selection for coarse execution modes.

Metaflow should not own:

- GPU memory residency.
- CPU/GPU fallback policy inside hot-path compute.
- Provider-specific upload/download mechanics.
- App-facing telemetry formatting.

### Proposed Refactor

Introduce a thin Metaflow flow that delegates to explicit execution services:

```text
CardCaptureFlow
    start
    choose_runtime
        strict_gpu
        cpu_debug
    import_results
    end
```

The flow should select a runtime branch from an artifact such as `runtime_mode`, not from hidden environment state. The selected branch calls a runtime implementation that owns the execution contract.

Artifacts passed between Metaflow steps should remain serializable:

- Run IDs.
- Input and output paths.
- Manifest paths.
- Metrics and telemetry summaries.
- Final result references.

They should not include CUDA tensors, model objects, open video handles, worker processes, or other process-local resources.

### Structured Telemetry

Replace stdout timing parsing with a single telemetry interface:

```python
class PipelineTelemetry:
    def stage_started(self, stage: str, metadata: dict) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: dict) -> None: ...
    def resource_sample(self, sample: dict) -> None: ...
    def contract_violation(self, code: str, metadata: dict) -> None: ...
```

The Metaflow flow, app runner, and platform handlers can all consume the same telemetry stream. SQLite events may remain the app compatibility layer, but they should be written by telemetry adapters rather than ad hoc timing calls in each step.

### Open Questions

- Minimum Metaflow version required for conditional branching.
- Whether Metaflow should continue to run inside remote workers or only wrap remote submissions from the app side.
- Whether current per-stage step names should be preserved for operator familiarity.

---

## Section 2: GPU/CPU Execution Contract Refactor

### Current Issues

- CPU/GPU behavior is spread across sampler, detector, refinement, scoring, fusion, utility, and runner code.
- Several functions silently fall back to CPU when CUDA is unavailable or when a GPU operation fails.
- Hidden CPU work can be introduced by reading images from paths, calling OpenCV operations, converting tensors to NumPy arrays, or writing intermediate images.
- The codebase lacks a single boundary that says "from here until final export, data is GPU-resident."

### Target Shape

Introduce explicit execution backends:

```text
PipelineRuntime
    StrictGpuRuntime
    CpuDebugRuntime
    FutureBeamGpuRuntime
```

Production GPU execution uses `StrictGpuRuntime`. CPU execution uses `CpuDebugRuntime`. These are different implementations of the same interface, not fallback branches inside the same implementation.

### GPU Session Capability

Strict GPU execution should require a session/capability object:

```python
class GpuSession:
    device: torch.device
    strict: bool
    telemetry: PipelineTelemetry
```

Hot-path code should require `GpuSession` explicitly. If a function can decode, transfer, run inference, warp, embed, or score GPU-resident data, it should not be callable without the session.

### Device-Tagged Data Types

Introduce small wrapper types for GPU-owned data:

```python
class GpuFrameBatch:
    tensor: torch.Tensor

class GpuCropBatch:
    tensor: torch.Tensor

class GpuEmbeddingBatch:
    tensor: torch.Tensor
```

These wrappers enforce:

- `tensor.is_cuda`.
- Known layout and dtype.
- No implicit `.cpu()`, `.numpy()`, or image-path roundtrip.
- Explicit conversion only at approved export boundaries.

### Strict Runtime Guard

Add a strict mode that fails on forbidden operations inside GPU execution:

- `cv2.VideoCapture`
- `cv2.imread`
- `cv2.imwrite`, except final export boundaries
- `PIL.Image.open`
- `torch.Tensor.cpu`
- `torch.Tensor.numpy`
- CPU-only Laplacian/sharpness helpers
- Any frame reread from a path that should already be resident

Initial implementation can be monkeypatch-based in tests and optionally enabled at runtime with `CC_GPU_STRICT=1`. The first goal is detection, not elegance.

### Fallback Policy

Production GPU execution should not silently fallback to CPU.

Allowed modes:

- `strict_gpu`: fail fast on missing CUDA, missing NVDEC, CPU-only helper usage, or tensor host transfer.
- `cpu_debug`: run a CPU implementation intentionally for local debugging and deterministic tests.
- `mixed_compat`: optional temporary mode for migration only; must emit contract warnings and should not be used in production.

### Open Questions

- Exact package boundary for strict GPU code.
- Whether strict guards should run in production always or only in canary/test jobs.
- How much of current intermediate image writing is truly required for UI/debugging versus historical convenience.

---

## Section 3: Execution and Platform Abstractions

### Current Issues

- Local, RunPod, Vast.ai, and Beam execution have different transport and lifecycle behavior but do not share a clean execution contract.
- Provider-specific details leak into app services and worker entrypoints.
- Filling out Beam requires touching broad pipeline and result-handling surfaces instead of adding a targeted adapter.

### Target Shape

Define one runner-facing contract:

```python
class PipelineRunner:
    async def submit(self, request: PipelineRunRequest) -> PipelineRunHandle: ...
    async def poll(self, handle: PipelineRunHandle) -> PipelineRunStatus: ...
    async def fetch_results(self, handle: PipelineRunHandle) -> PipelineRunResult: ...
    async def cancel(self, handle: PipelineRunHandle) -> None: ...
```

Provider implementations own upload, invocation, polling, download, cleanup, and provider-specific error mapping.

Pipeline execution owns input manifest, runtime mode, telemetry, and output manifest.

### Result Manifests

Every runtime should produce a manifest with:

- Run ID.
- Runtime mode.
- Input video reference.
- Output artifact references.
- Final card records.
- Structured telemetry summary.
- Contract violations or warnings.
- Version/build metadata.

The app imports from the manifest rather than reconstructing provider-specific filesystem assumptions.

---

## Section 4: Repo Structure

Stub for follow-up.

Topics to define:

- Pipeline orchestration package boundaries.
- Runtime backend package boundaries.
- GPU-only package boundary and forbidden imports.
- Shared contracts location.
- App runner versus remote worker ownership.
- Where platform adapters live.
- Migration path for legacy modules.

---

## Section 5: Data Access Layer

### Current Issues

SQL is used directly across multiple layers of the codebase:

- Pipeline orchestration writes timing and event rows directly.
- Pipeline steps and storage helpers write result data.
- App services update run status and query labeling data directly.
- Harness utilities and tests query application tables directly.
- Schema ownership is distributed across implementation files rather than expressed through one clear module boundary.

The presence of `Storage` helps, but it is not a complete data access layer. There are no obvious rules for where raw SQL is allowed, which module owns each table, how migrations/schema changes are coordinated, or which query shapes are part of the application contract.

This makes refactoring risky. A table change can require broad search-and-edit work, tests can accidentally depend on private schema details, and runtime/platform code can mutate database state without going through shared invariants.

### Target Shape

Introduce a small data access layer with explicit ownership:

```text
card_capture.data
    connection.py        database connection/session helpers
    schema.py            schema creation and migrations
    repositories/
        runs.py          pipeline_runs ownership
        events.py        pipeline_events ownership
        videos.py        video metadata ownership
        cards.py         card_instances/card_views ownership
        labeling.py      truth/labeling ownership
        telemetry.py     structured telemetry persistence
```

Application, pipeline, runner, and harness code should depend on repository methods or query services, not open SQLite connections directly.

### SQL Boundary Rules

Raw SQL should be allowed only in:

- Data access layer modules.
- Migration/schema modules.
- Narrow test helpers that explicitly validate schema behavior.
- One-off development scripts, if clearly marked and not imported by production code.

Raw SQL should not appear in:

- Metaflow flow definitions.
- Pipeline step implementations.
- Runtime/platform runners.
- App API handlers.
- General service modules.
- Algorithmic modules.

### Repository Contracts

Repositories should expose intention-revealing methods:

```python
runs.mark_started(run_id, video_id)
runs.mark_completed(run_id, cards_extracted)
runs.mark_failed(run_id, error)
events.record_stage_finished(run_id, video_id, stage, elapsed_ms, metadata)
cards.store_final_cards(run_id, final_cards)
labeling.get_truth_payload(video_id)
```

The goal is not a large ORM. The goal is to make table ownership, invariants, transactions, and query shapes explicit.

### Migration Direction

Start by wrapping the highest-churn direct SQL:

1. Pipeline run status updates.
2. Pipeline event/timing writes.
3. Final card storage.
4. Labeling/truth queries.
5. Harness read models.

After wrappers exist, add an architecture test that fails on new `sqlite3.connect` or raw SQL usage outside allowlisted modules.

### Open Questions

- Whether to keep a lightweight hand-written repository layer or adopt SQLModel/SQLAlchemy Core.
- Whether app-facing read models should be separate from write repositories.
- How much direct SQL should remain in tests versus test-only repository helpers.
- Whether migration/versioning belongs in the same package or a separate operational tool.

---

## Section 6: Testing

Stub for follow-up.

Topics to define:

- Contract tests for `StrictGpuRuntime`.
- Architecture tests that forbid raw SQL outside the data access layer.
- CPU debug parity tests.
- Static import tests for GPU-only packages.
- Runtime monkeypatch guards for forbidden CPU operations.
- Metaflow graph smoke tests.
- Provider adapter tests with fake transport.
- Manifest import/export tests.
- Performance regression tests and minimum telemetry assertions.

---

## Section 7: CI

Stub for follow-up.

Topics to define:

- CPU-only fast test lane.
- Static architecture lint lane.
- Optional CUDA lane.
- Nightly or manual GPU integration lane.
- Metaflow smoke lane.
- Provider contract tests with mocked Beam/RunPod/Vast.ai APIs.
- Performance baseline reporting.

---

## Section 8: Runtime Platforms

Stub for follow-up.

Topics to define:

- Local CPU debug.
- Local CUDA.
- RunPod serverless.
- Beam endpoint.
- Vast.ai lifecycle support or deprecation.
- Artifact storage and transfer model per platform.
- GPU driver/NVDEC preflight contract.
- Platform-specific telemetry and failure mapping.

---

## Migration Strategy

### Phase 1: Contracts and Telemetry

- Add `docs/architecture/standards.md` with the initial binding rules from this design.
- Add runtime interfaces and result manifest contracts.
- Add structured telemetry abstraction.
- Stop adding new stdout parsing dependencies.
- Keep current pipeline behavior otherwise unchanged.

### Phase 2: Strict GPU Boundary

- Introduce `GpuSession` and device-tagged batch types.
- Add strict runtime guard in tests.
- Move GPU hot-path code behind the new session boundary.
- Convert silent CPU fallbacks into explicit backend selection.

### Phase 3: Metaflow Slimming

- Move timing writes out of flow step bodies.
- Replace hidden runtime selection with explicit branch selection.
- Make Metaflow artifacts point to manifests and structured outputs.

### Phase 4: Platform Adapter Cleanup

- Make Beam, RunPod, local, and any remaining Vast.ai runner implement the same runner contract.
- Import results from manifests.
- Remove provider-specific result assumptions from app-facing code.

---

## Acceptance Criteria

- A production GPU run cannot silently use CPU fallback inside strict execution.
- Hidden host transfers and frame rereads fail in strict tests.
- Metaflow no longer needs stdout parsing for step timing.
- Beam can be completed by implementing a platform adapter, not by editing core pipeline logic.
- Raw SQL is restricted to the data access layer, migration/schema code, and explicit test helpers.
- Pipeline, app service, and platform code use repository/query interfaces for database access.
- `docs/architecture/standards.md` exists and states the binding architecture rules AI agents must follow.
- At least the highest-risk standards have automated enforcement through tests, lint, or runtime guards.
- CPU debug remains available as an explicit runtime backend.
- Existing algorithmic outputs are preserved unless a later algorithm-specific design says otherwise.

---

## Notes for Expansion

This draft intentionally leaves repo structure, testing, CI, and runtime platforms as stubs. The main architectural decisions captured here are:

- Metaflow is orchestration, not GPU residency enforcement.
- GPU/CPU selection is a runtime/backend decision, not scattered conditionals.
- Database access is a repository/query-layer concern, not an incidental implementation detail in any module.
- The final V5.5 output includes an architectural standards document for AI agents and contributors.
- Strict GPU execution is enforced with session capabilities, device-tagged types, guards, and tests.
- Platform differences belong in runner adapters and manifests.

---

## Amendment (2026-05-27): Collapse the single-machine path to one in-process runtime

**Status:** Proposed, backed by profiling.
**Supersedes:** the implicit assumption in Section 1 that Metaflow remains the per-stage execution graph on a single machine.

### Motivation: Metaflow's per-step subprocess model is a net negative on one machine

The original draft already observes that "the flow mostly behaves as a linear script wrapper" and that Metaflow "should not own" GPU residency, CPU/GPU policy, provider mechanics, or telemetry. Profiling the MPS path on real 4K footage (`IMG_5922.MOV`, ~27 cards, ~2.2k sampled frames) makes the stronger conclusion unavoidable: **on a single machine the per-step subprocess boundary costs more than it provides, and it structurally forces wasted work.**

Each Metaflow `@step` runs as its own subprocess that re-imports torch/cv2/CoreML/kornia and reloads models. Measured consequences:

- **Per-step process + reload tax.** Stages that do almost nothing (novelty, score, resolve) each cost ~2-3s of pure subprocess/import overhead.
- **`fuse` fan-out catastrophe.** `fuse_fanout` → `foreach` spawned **one subprocess per track** (128 on this video), each booting Python to perform a single file copy — **~4-6.5 minutes** of pure overhead. (Now collapsed to a 1.5s in-process loop.)
- **The detect→refine re-decode.** Because frames cannot cross a subprocess boundary as in-memory or GPU-resident data, `refine` re-decodes nearly the entire 4K video to recover canonical frames — **~52-84s** that exists *solely* because of the step boundary. The CUDA path already proves the fix: `fused_refine` runs detect + warp in one process and keeps crops in memory.
- **Artifact serialization.** Detection rows, track data, and frame entries are pickled to the datastore between every step.

After collapsing `fuse` in-process, moving cheap metrics off the GPU, and using `grab()/retrieve()` decode, a full run dropped from **>12 min (did not complete in test)** to **~3 min**. The remaining dominant stage is `refine` (~112s), of which ~52s is the re-decode — i.e. the largest single remaining cost is a direct artifact of the step boundary.

### Amended target shape

The local / single-machine runtime **must** execute the full stage sequence **in one process via direct function calls**, not as a Metaflow step-per-stage graph. The runtime owns the in-memory pipeline; Metaflow does not sit between stages.

```python
class PipelineRuntime:                     # StrictGpuRuntime | CpuDebugRuntime
    def run(self, request: PipelineRunRequest) -> PipelineRunResult:  # manifest
        ...
```

Within one `run()` call the stages share, as plain in-memory objects:

- **Loaded models** — YOLO/CoreML and DINOv2 are loaded **once per run**, not once per stage.
- **Decoded frames** — frames decoded in detect are reused by refine; **the re-decode is eliminated**, not optimized.
- **GPU-resident tensors** — crops/embeddings stay on-device across stages; no `.cpu()` roundtrip and no pickling. (Generalizes the existing `fused_refine` precedent to all runtimes and all stages.)

Metaflow, if retained at all, is reduced to one of:

- **(a) a one-step wrapper** that calls `runtime.run(request)` and records the returned manifest as its single artifact; or
- **(b) dropped for local execution entirely**, and used only as a remote *submission* shell — which the `PipelineRunner` contract in Section 3 already covers.

The `choose_runtime → {strict_gpu | cpu_debug}` branch in Section 1 stays, but each branch is a **single execution unit**, not an 11-node graph.

### What this amendment does NOT change

The rest of V5.5 is orthogonal and still wanted — none of it depends on Metaflow's graph:

- Strict GPU runtime, `GpuSession`, device-tagged batch types, and forbidden-op guards (Section 2).
- The `PipelineRunner` contract and result manifests (Section 3).
- The data access layer and SQL boundary rules (Section 5).
- Structured `PipelineTelemetry` (Section 1) — which also removes the stdout timing parsing that this very investigation had to rely on.

### Trade-offs accepted

Dropping the per-stage graph gives up: per-stage resume/retry, per-stage artifact lineage, and the Metaflow step UI. For a single-machine run measured at ~3 min (and headed lower), that is not worth ~10-15s/run of process tax plus the re-decode plus pickling. Resume becomes **coarse (per-run)**; remote orchestration is handled by the runner contract and manifests, not by an in-pipeline graph.

### Added acceptance criteria

- A local run executes all stages in a single process; no stage is a separate OS process.
- Models are instantiated at most once per run.
- `refine` (and any post-detect stage) consumes decoded frames produced by `detect`; no module re-opens the source video after detect except an explicit, telemetered fallback.
- Metaflow, if present, holds only serializable artifacts (run id, manifest path, telemetry summary) and contains no per-stage algorithmic step bodies.

### Migration note

This reorders the draft's Phase 3 ("Metaflow Slimming") ahead of where it sits today and makes it concrete: the single-process `PipelineRuntime` is the unit Phase 2's strict-GPU boundary should wrap, and the `fused_refine` CUDA path is the reference implementation to generalize.
