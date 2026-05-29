# V5.5 Refactoring Design

**Date:** 2026-05-24
**Last revised:** 2026-05-27
**Status:** Plan-of-record
**Scope:** Software engineering refactors for pipeline correctness, performance enforcement, and runtime portability

## Decision Posture

This document is a working collection of thoughts from developing, profiling, and running the system. Everything here is open to debate, revision, or rejection as new evidence appears.

The project owner is the final authority for V5.5 decisions. External architecture patterns, framework conventions, agent recommendations, and implementation plans are inputs to evaluate, not obligations. The deciding standard is what best serves this system's correctness, performance, maintainability, and operating reality.

---

## Problem

The V4 pipeline has accumulated optimization work inside boundaries that do not enforce the properties we care about. Metaflow is used as a per-stage subprocess shell, the app runner parses logs for timing data, CPU/GPU behavior is selected through scattered flags and conditional code paths, and runtime-specific concerns leak across app, worker, and pipeline modules.

The result is that performance regressions are too easy to introduce. Hidden CPU reads, implicit tensor copies, silent fallback behavior, and repeated decode/load work can all appear as local implementation details instead of contract violations.

Profiling on real 4K footage (`IMG_5922.MOV`, ~27 cards, ~2.2k sampled frames) confirms the cost is not theoretical:

- **Per-step process + reload tax.** Stages that do almost nothing (novelty, score, resolve) each cost ~2-3s of pure subprocess/import overhead — ~10-15s/run baseline.
- **`fuse` fan-out catastrophe.** `fuse_fanout` → `foreach` previously spawned one subprocess per track (128 on this video), each booting Python to perform a single file copy — **~4-6.5 minutes** of pure overhead.
- **The detect→refine re-decode.** Because frames cannot cross a subprocess boundary as in-memory or GPU-resident data, `refine` re-decodes nearly the entire 4K video to recover canonical frames — **~52-84s** that exists *solely* because of the step boundary.
- **Artifact serialization.** Detection rows, track data, and frame entries are pickled to the datastore between every step.

After collapsing `fuse` in-process, moving cheap metrics off the GPU, and using `grab()/retrieve()` decode, a full run dropped from **>12 min (timed out in test)** to **~3 min**. The remaining dominant cost (`refine` ~112s, of which ~52s is the re-decode) is a direct artifact of the per-stage process boundary.

V5.5 must make these mistakes structurally difficult. This document focuses on software engineering refactoring. It does not propose new card-detection, tracking, scoring, fusion, deduplication, or OCR algorithms.

---

## Goals

- Make pipeline execution contracts explicit enough that hidden CPU work and repeated load/decode work fail fast.
- Remove Metaflow from the codebase and replace it with a lightweight runtime/runner contract that is small enough to enforce.
- Replace scattered CPU/GPU conditionals with runtime interfaces and backend implementations.
- Replace scattered SQL access with an explicit data access layer and ownership rules.
- Separate runner, platform, telemetry, and pipeline concerns so Beam, RunPod, local CUDA, and CPU debug can share the same contract.
- Preserve current algorithmic behavior while improving architecture, testability, and operational clarity.
- Enforce architecture rules statically where possible (import direction, raw SQL location, forbidden GPU call sites) and at runtime where static checks cannot reach.
- Produce an architectural standards document that AI agents and human contributors must follow when changing the pipeline.

---

## Non-Goals

- No algorithm redesigns.
- No quality metric or threshold tuning.
- No new model training work.
- No new general-purpose workflow/DAG framework unless a future decision explicitly chooses one.
- No attempt to make CPU fallback transparent in production.
- No distributed tracing or span sampling — single-process batch runs do not produce trace volume that benefits from sampling.

---

## Design Principles

- **Contracts over conventions.** If production GPU execution must stay on GPU, represent that through types, interfaces, guards, and tests.
- **Explicit debug modes.** CPU debug should be a selected backend, not a silent fallback path.
- **Small contracts over workflow frameworks.** The replacement for Metaflow should be a small collection of interfaces, dataclasses, and CLI entrypoints, not another orchestration layer.
- **Duplicate across backend contexts when it clarifies ownership.** Prefer local duplication in CPU debug, CUDA, MPS/CoreML, RunPod, and Beam code over shared conditionals that bend one backend around another.
- **Telemetry is first-class.** Timing, resource metrics, contract violations, and result summaries should be emitted through structured telemetry/metrics APIs, not scraped from stdout.
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
- CI / architecture-lint expectations.
- A short "when unsure" escalation rule for agents.

### Rule Schema

Each rule carries two markers:

- **Phase marker** (`advisory:phase-N` or `blocking:phase-N`) — indicates when the rule must be obeyed and when it becomes CI-enforced. Rules may start advisory and tighten as the phase plan completes; the doc must not claim a rule is blocking before the codebase actually complies.
- **Enforcement marker** (`static` / `runtime` / `review`):
  - **Static**: import lint, raw SQL scan, forbidden API scan, package boundary checks.
  - **Runtime**: strict GPU guard, telemetry contract checks, manifest validation.
  - **Review**: rules that require human judgment until automated checks exist.

### Example Standards

Initial standards should include rules such as:

- Metaflow must be removed from production code. New orchestration code must use the lightweight runtime/runner contracts instead of a workflow framework.
- Production GPU execution must use the strict GPU runtime; it must not silently fall back to CPU.
- CPU execution must be selected through an explicit CPU debug backend.
- Backend-specific code may duplicate small amounts of logic when the alternative is conditionals that obscure runtime ownership.
- GPU hot-path modules must not call `cv2.VideoCapture`, `cv2.imread`, `PIL.Image.open`, `torch.Tensor.cpu`, or `torch.Tensor.numpy` except through the approved CPU-export boundary helpers (Section 2).
- App services, pipeline runtime code, and platform runners must not open SQLite connections directly.
- Raw SQL must live in the data access layer, migrations/schema code, or allowlisted test helpers.
- Platform-specific code must live in provider adapters and must communicate through runner contracts and manifests.
- Timing and resource metrics must be emitted through structured telemetry interfaces, not parsed from logs.

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

## Section 1: Single-Process Pipeline Runtime

### Current Issues

- The app invokes Metaflow as a subprocess and parses stdout for step timings.
- Stage timing is duplicated between flow code, runner diagnostics, and handler diagnostics.
- The flow mostly behaves as a linear script wrapper, so Metaflow adds process boundaries, import/model reloads, datastore serialization, and graph semantics without providing value proportional to that cost.
- Per-step subprocess re-imports of torch/cv2/CoreML/Kornia and per-step model reloads accumulate to ~10-15s/run of pure process tax.
- The `fuse` foreach previously spawned one subprocess per track, each booting Python to perform a single file copy.
- Because frames cannot cross a subprocess boundary as in-memory or GPU-resident data, `refine` re-decodes nearly the entire 4K video to recover canonical frames — work that exists *solely* because of the step boundary.
- Runtime decisions such as CPU debug versus GPU execution are represented indirectly through config files, environment variables, and detector names.

### Target Shape

The local / single-machine runtime **must** execute the full stage sequence **in one process via direct function calls**, not as a per-stage workflow graph. The runtime owns the in-memory pipeline; Metaflow is removed from production code.

Replacement surface:

```python
class PipelineRuntime(Protocol):
    def run(self, request: PipelineRunRequest) -> PipelineRunResult: ...

class PipelineRunner(Protocol):
    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle: ...
    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult: ...
    def cancel(self, handle: PipelineRunHandle) -> None: ...

class PipelineTelemetry(Protocol):
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
    def resource_sample(self, sample: Mapping[str, object]) -> None: ...
    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None: ...
```

Within one `run()` call the stages share, as plain in-memory objects:

- **Loaded models** — YOLO/CoreML and DINOv2 are loaded **once per run**, not once per stage.
- **Decoded frames** — frames decoded in detect are reused by refine; **the re-decode is eliminated**, not optimized.
- **GPU-resident tensors** — crops/embeddings stay on-device across stages; no `.cpu()` roundtrip and no pickling between stages.

Runtime mode is an explicit field in `PipelineRunRequest` (`runtime_mode = strict_gpu | cpu_debug`), not hidden environment state.

Values passed between runtime, runner, app import, and performance harness code must remain serializable: run IDs, input and output paths, manifest paths, metrics summaries, final result references. They must not include CUDA tensors, model objects, open video handles, worker processes, or other process-local resources.

The replacement does not own GPU memory residency, CPU/GPU fallback policy inside hot-path compute, provider-specific upload/download mechanics, app-facing telemetry formatting, retry semantics beyond explicit coarse per-run retry, or a general DAG / step scheduler / artifact datastore.

### Reference Implementation

The CUDA `fused_refine` path — which runs detect + warp in one process and keeps crops in memory — is the **precedent for the pattern (single process, frames in memory)**, not a portable reference implementation. CUDA-specific machinery (streams, page-locked memory, NVDEC) does not generalize to MPS or CPU debug. Each backend must produce its own in-process implementation of the same logical contract.

### Resume Contract

Resume is **per-run, not per-stage**. A failed run is rerun from the input video; the runtime does not persist mid-run checkpoints to disk. This trade-off is acceptable at the measured ~3-minute single-machine cost. Remote orchestration concerns (retry across machines, partial-output cleanup) are owned by the platform adapter contracts in Section 3, not by an in-pipeline graph.

### Backend Duplication Policy

Do not force all backends through one implementation when their constraints differ. It is acceptable, and often preferable, to duplicate small pieces of code in separate backend modules when the shared alternative would introduce conditionals such as:

- `if cuda`.
- `if mps`.
- `if cpu_debug`.
- `if remote_provider`.
- `if has_gpu_decode`.

Shared code should be limited to stable contracts, pure geometry/math utilities, manifest/telemetry schemas, and genuinely backend-neutral helpers. Runtime-specific decode, model loading, frame caching, memory transfer, and export rules should live with the backend that owns them.

### Structured Telemetry

Replace stdout timing parsing with a single telemetry and metrics surface. The project-local `PipelineTelemetry` facade remains the application-facing contract; the implementation publishes through OpenTelemetry Metrics.

Direction:

- Use OpenTelemetry Metrics as the primary instrumentation API for counters, histograms, gauges, and resource attributes (`MeterProvider`, `Meter`, instruments). Reference: <https://opentelemetry-python.readthedocs.io/en/latest/api/metrics.html>.
- **Distributed tracing and span sampling are out of scope for V5.5.** A single-process ~3-minute batch run does not produce trace volume that benefits from head/tail sampling. If distributed remote runners later produce per-frame spans worth sampling, revisit then.
- Keep the project-local `PipelineTelemetry` facade so app code, runtime code, and tests do not depend directly on one vendor/exporter shape.
- Write telemetry to at least two sinks: structured JSONL output for run debugging (one event per line, replayable into the manifest), and a metrics exporter for local/performance/production analysis.
- MLflow Tracking is **deferred** — the manifest + JSONL pair covers experiment storage at the scale we run. Revisit only if cross-branch / cross-machine experiment comparison becomes intractable.

The runtime, app runner, platform handlers, and performance harness consume the same telemetry stream. SQLite events may remain the app compatibility layer, but they are written by telemetry adapters rather than ad hoc timing calls in each stage.

### Trade-offs Accepted

Dropping the per-stage graph gives up per-stage resume/retry, per-stage artifact lineage, and the Metaflow step UI. For a single-machine run measured at ~3 min (and headed lower), that cost is not worth ~10-15s/run of process tax plus the re-decode plus pickling.

### Open Questions

- Whether current per-stage names should be preserved for operator familiarity in telemetry, even though stage execution is no longer a workflow graph.
- Whether the metrics exporter should default to local JSON, Prometheus/OpenMetrics, or OTLP.

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

- `tensor.is_cuda` (or `.is_mps`, per backend).
- Known layout and dtype.
- No implicit `.cpu()`, `.numpy()`, or image-path roundtrip.
- Explicit conversion only at approved export boundaries.

### Approved CPU-Export Boundaries

GPU-resident tensors may cross to CPU **only** at the following explicit, telemetered boundaries. Every other `.cpu()` / `.numpy()` / `cv2.imwrite` call inside strict GPU code is a contract violation:

| Boundary | Purpose | Output type |
| --- | --- | --- |
| Stage 7 quality scoring | Reduce GPU tensors to scalar `QualityScore` fields | CPU dict (sharpness, glare, aspect_ratio, size, complexity, border_purity, confidence) |
| Stage 8 front/back resolution | pHash + rotation tolerance comparison | CPU hex digests |
| Stage 9 fusion candidate selection | Quadrant glare statistics for selection | CPU scalars |
| Stage 10 pHash + ReID dedup | Hash and embedding comparisons against persisted state | CPU `bytes` / `numpy.float32` arrays |
| Final export | 750×1050 PNG write to `crops/`, manifest write | `cv2.imwrite` / file write |
| Telemetry flush | Per-stage timing, resource samples, contract violations | OTel metrics record |

Each boundary is implemented as a named helper under `card_capture.runtime.batches` (`to_cpu_for_score`, `to_cpu_for_phash`, `to_cpu_for_export`, ...). The strict guard's allowlist references these helpers by qualified name. Other call sites of `.cpu()` / `.numpy()` are forbidden inside files tagged GPU-resident.

### Strict-Mode Enforcement Mechanism

Strict mode is implemented through **two distinct mechanisms** that share one logical contract:

- **Tests** (`tests/runtime/`) use `pytest`'s `monkeypatch.context()` to temporarily overwrite `torch.Tensor.cpu`, `torch.Tensor.numpy`, `cv2.imread`, `cv2.VideoCapture`, `PIL.Image.open`, and the cheap-CPU Laplacian/sharpness helpers within the scope of a single test block. The original behavior is restored on context exit, so strict tests do not break CPU debug tests run later in the same session.
- **Production** uses a **wrapper runtime** (`StrictGpuRuntime`) that exposes only the safe device-tagged batch APIs to stage code; forbidden imports inside strict stage modules are caught by the static AST pass below before reaching runtime. Monkeypatching third-party libraries in production is forbidden — third-party code (Kornia, model exporters) calls these methods legitimately and patching globally would break them.

The `CC_GPU_STRICT=1` runtime flag enables additional **assertion checks** in the wrapper runtime (device tags, batch invariants) but does **not** enable global monkeypatching.

### Static Enforcement (Zero-Copy Lint)

A small custom `ast.NodeVisitor` scanner runs in CI. It is **narrowly scoped**:

- Reads a glob list from `pyproject.toml` (`[tool.gpu_strict_lint] files = [...]`) of files tagged as GPU-resident — e.g. `src/card_capture/runtime/strict_gpu/**`, `src/card_capture/pipeline/stages/gpu_*.py`.
- For each file in scope, fails on AST nodes that resolve to forbidden calls: `cv2.VideoCapture`, `cv2.imread`, `cv2.imwrite` (except in export-boundary modules), `PIL.Image.open`, `torch.Tensor.cpu`, `torch.Tensor.numpy`, the cheap-CPU sharpness helpers.
- Allowlists are file-path-based, not per-call-site, to keep the scanner small.

Out-of-scope files (CPU debug, scoring helpers, app services, tests) are not scanned. Quality scoring legitimately reduces to CPU and would otherwise trip the guard.

Import-direction concerns (which modules may import which packages) are handled by Import Linter, not this AST pass — see Section 4.

### Fallback Policy

Production GPU execution must not silently fall back to CPU.

Allowed modes:

- `strict_gpu`: fail fast on missing CUDA/MPS, missing decode backend, CPU-only helper usage, or tensor host transfer outside approved export boundaries.
- `cpu_debug`: run a CPU implementation intentionally for local debugging and deterministic tests.
- `mixed_compat`: optional temporary mode for migration only; must emit contract warnings and must not be used in production.

### Open Questions

- Exact glob list for the AST scanner — finalized in `standards.md` as part of Phase 2.
- Whether the wrapper-runtime assertion checks (`CC_GPU_STRICT=1`) should default on or off in production runs.

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

The repo should make the V5.5 boundaries obvious from file paths. The current split between top-level `pipeline/`, `app/services`, and broad `src/card_capture/*` helpers makes it too easy to put runtime, platform, storage, and algorithmic code in whichever module is closest.

### Target Shape

Use package boundaries that map to ownership:

```text
pipeline/
    contracts.py                  transitional imports only; move stable contracts below

src/card_capture/
    pipeline/
        runtime.py                PipelineRuntime interface and in-process stage sequence
        request.py                PipelineRequest / PipelineResult / RunManifest models
        telemetry.py              PipelineTelemetry interface and event schema
        stages/                   stage orchestration facades, not low-level algorithms
    runtime/
        cpu_debug.py              intentional CPU backend
        strict_gpu.py             strict GPU backend
        gpu_session.py            GpuSession and strict guard lifecycle
        batches.py                device-tagged batch/result types + approved export helpers
        guards.py                 forbidden-op guard implementation
    platforms/
        local.py                  local in-process runner
        runpod.py                 RunPod adapter
        beam.py                   Beam adapter
        vastai.py                 Vast.ai adapter, if retained
        manifests.py              manifest read/write and artifact reference helpers
    data/
        connection.py
        schema.py
        writer.py                 single-writer queue (see Section 5)
        repositories/
```

Existing algorithm modules (detection, sampling, tracking, scoring, fusion, ML inference) may stay under their current domain packages initially. The refactor should wrap them behind `card_capture.pipeline.stages` and runtime backends before moving files. File moves happen only after tests make ownership clear.

### Boundary Rules

- `pipeline/` is orchestration/submission only. It may create a `PipelineRequest`, call a runner/runtime, and persist a manifest reference. It must not contain algorithmic stage bodies.
- `card_capture.pipeline` owns the logical pipeline sequence, request/result contracts, telemetry contracts, and stage facades.
- `card_capture.runtime` owns backend selection, GPU session lifecycle, strict GPU guards, CPU debug implementations, and device-residency types.
- `card_capture.platforms` owns provider transport, preflight, artifact upload/download, remote job lifecycle, and provider failure mapping.
- `card_capture.data` owns database connections, migrations/schema helpers, repositories, the single-writer queue, and app-facing query services.
- `app/services` owns HTTP/API-facing workflow state and user-facing status mapping. It must call `PipelineRunner` or data repositories rather than importing provider clients or opening SQLite directly.
- Remote worker entrypoints must be thin shells that deserialize a request, call the same runtime contract, and write a manifest. They must not fork their own pipeline semantics.

### Import-Direction Enforcement

Boundary rules are enforced by **Import Linter** contracts checked into the repo at `.importlinter`. Contracts express:

- **Layers contract**: `app/services` and platform adapters sit at the top; `card_capture.pipeline` and `card_capture.runtime` are middle layers; `card_capture.data` is the bottom layer. Lower layers must not import from higher layers.
- **Forbidden contract**: no module outside `card_capture.data`, migration modules, or allowlisted test helpers may import `sqlite3`.
- **Forbidden contract**: no module under `card_capture.runtime.strict_gpu` may import `app.*`, `card_capture.platforms.*`, PIL, or OpenCV file/image-IO helpers.
- **Forbidden contract**: no module outside `card_capture.platforms` may import provider SDKs (`runpod`, `beam`, vast.ai clients).
- **Forbidden contract**: no module may import `metaflow` once Phase 3 completes.

Import Linter replaces the hand-rolled import scanner sketched in earlier drafts. It does **not** replace the AST scanners in Section 2 (forbidden GPU call sites) or the raw-SQL string scanner in Section 6 — those operate at call-site granularity, which import lint cannot see.

### GPU-Only Boundary

Strict GPU code needs a boundary that is enforceable by tests:

- GPU hot-path modules must require `GpuSession` or a device-tagged batch type for entry.
- GPU hot-path modules may import Torch/Kornia/CUDA-or-MPS decode helpers and runtime guards.
- GPU hot-path modules must not import OpenCV image/file IO helpers, PIL image loaders, `sqlite3`, app services, or platform adapters.
- CPU debug modules may share pure math/geometry utilities, but must not be imported by strict GPU modules.
- Shared dataclasses live in contract modules with no heavyweight runtime side effects.

### Migration Direction

1. Add the new packages as wrappers around existing code.
2. Move shared request/result/manifest/telemetry contracts out of top-level `pipeline/contracts.py` once stable.
3. Route local execution through `card_capture.pipeline.runtime.PipelineRuntime` before moving provider adapters.
4. Move direct provider logic out of app services into `card_capture.platforms`.
5. Move database writes and reads behind `card_capture.data`.
6. Add architecture tests and Import Linter contracts for the highest-risk boundaries before doing large file moves.
7. Delete Metaflow modules, tests, and dependencies after call sites have moved and the replacement runtime contract is covered.

### Open Questions

- Whether the top-level `pipeline/` package should remain long-term or become a compatibility layer only.
- Whether provider adapters should live in `card_capture.platforms` or under `app/services` until the API boundary settles.
- Whether current worker modules should be split by platform or collapsed behind one worker entrypoint.

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

### Concurrency: Single-Writer Discipline

SQLite WAL allows concurrent readers but **one writer at a time**. The pipeline (which writes events, results, and telemetry), the FastAPI app (which writes run status and label data), and the harness (which writes baseline metrics) are all writers against the same database file. Historical `database is locked` errors are a direct consequence.

V5.5 must pick one of the following and document it in `standards.md`:

- **(a) Cross-process single writer (recommended)**: a dedicated writer owns all writes. Pipeline runtime, FastAPI handlers, and harness submit writes through `card_capture.data.writer` — an in-process queue when colocated, an IPC queue when separate. A pipeline-internal queue alone does **not** solve the problem because the FastAPI side opens its own connection.
- **(b) Split databases by ownership**: `runtime.db` (pipeline writes only) and `app.db` (app writes only); cross-references use IDs rather than foreign keys.
- **(c) Retry-with-timeout**: `PRAGMA busy_timeout=5000` plus bounded retry on `OperationalError: database is locked`. Only acceptable if writer contention is rare enough that retry latency is invisible.

The default recommendation is **(a)**. Whichever option is chosen, both the pipeline runtime and the FastAPI handlers must route writes through it; a queue inside `card_capture.data` is insufficient without app-side adoption.

### Target Shape

Introduce a small data access layer with explicit ownership:

```text
card_capture.data
    connection.py        database connection/session helpers
    schema.py            schema creation and migrations
    writer.py            single-writer queue (see above)
    repositories/
        runs.py          pipeline_runs ownership
        events.py        pipeline_events ownership
        videos.py        video metadata ownership
        cards.py         card_instances/card_views ownership
        labeling.py      truth/labeling ownership
        telemetry.py     structured telemetry persistence
```

Application, pipeline, runner, and harness code depend on repository methods or query services, not open SQLite connections directly.

### SQL Boundary Rules

Raw SQL is allowed only in:

- Data access layer modules.
- Migration/schema modules.
- Narrow test helpers that explicitly validate schema behavior.
- One-off development scripts, if clearly marked and not imported by production code.

Raw SQL must not appear in:

- Legacy orchestration wrappers.
- Pipeline step implementations.
- Runtime/platform runners.
- App API handlers.
- General service modules.
- Algorithmic modules.

Import Linter enforces the `sqlite3` import boundary. A separate AST pass (`tests/architecture/test_raw_sql_outside_data.py`) scans for raw SQL string literals outside allowlisted modules — the import lint can't catch a hand-written `connection.execute("UPDATE ...")` if the connection object came from somewhere else.

### Repository Contracts

Repositories expose intention-revealing methods:

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

After wrappers exist, the raw-SQL AST scanner tightens from advisory to blocking.

### Open Questions

- Whether to keep a lightweight hand-written repository layer or adopt SQLModel/SQLAlchemy Core.
- Whether app-facing read models should be separate from write repositories.
- How much direct SQL should remain in tests versus test-only repository helpers.
- Whether migration/versioning belongs in the same package or a separate operational tool.

---

## Section 6: Testing

Testing for V5.5 should prove the architecture boundaries, not only the happy-path card outputs. The current risk is that a future change can reintroduce host transfers, raw SQL, provider leakage, or per-stage process overhead while still passing narrow algorithm tests.

### Test Categories

- `tests/architecture/`: Import Linter integration, AST-based forbidden-call and raw-SQL scanners.
- `tests/runtime/`: `PipelineRuntime`, `StrictGpuRuntime`, `CpuDebugRuntime`, `GpuSession`, device-tagged batches, and forbidden-op guards.
- `tests/platforms/`: local, RunPod, Beam, and Vast.ai adapters using fake transports and fake artifact stores.
- `tests/data/`: repository contracts, migration/schema behavior, single-writer queue, and transactional invariants.
- `tests/pipeline/`: stage sequencing, manifest generation, telemetry emission, and runtime contract smoke tests.
- `tests/performance/`: benchmark harness tests, telemetry artifact validation, and sample-video fixture checks.
- `tests/regression/`: golden corpus and performance baselines.

### Architecture Tests

Architecture enforcement is split across three mechanisms, all running in the fast PR lane:

- **Import Linter** (`.importlinter` contracts) for module-import direction and forbidden cross-package imports.
- **Narrow AST scanner** (`tests/architecture/test_gpu_strict_calls.py`) for forbidden call sites inside files tagged GPU-resident.
- **Raw-SQL AST scanner** (`tests/architecture/test_raw_sql_outside_data.py`) for SQL string literals outside `card_capture.data`, migrations, and allowlisted test helpers.

Combined, these fail on:

- Raw SQL or `sqlite3.connect` outside `card_capture.data`, migrations, and allowlisted schema tests.
- Provider SDK/client imports outside `card_capture.platforms` and explicit app composition roots.
- App service imports from GPU hot-path modules.
- GPU hot-path imports from app, platform, data, PIL, or OpenCV file IO modules.
- Any Metaflow import, flow definition, or workflow framework dependency after the Phase 3 removal completes.
- Forbidden GPU-export call sites (`.cpu()`, `.numpy()`, `cv2.imread`, `cv2.imwrite` outside the named export-boundary helpers) inside files tagged GPU-resident.

### Strict-Guard Tests (Runtime Contract)

Strict guard tests use `monkeypatch.context()` (per Section 2) to temporarily redefine forbidden calls within a single test block, then assert the strict runtime raises `ContractViolation` and records the violation in the manifest. Tests must not patch outside their context — that would corrupt unrelated tests in the same session.

Assertions:

- `strict_gpu` fails when CUDA/MPS/decode requirements are missing unless the test explicitly uses a fake capability.
- Hot-path functions reject CPU tensors, path-only frame references, or batches without a `GpuSession`.
- Forbidden operations such as `cv2.VideoCapture`, `cv2.imread`, `PIL.Image.open`, `.cpu()`, and `.numpy()` fail inside guarded strict sections.
- Approved CPU-export helpers (Section 2 table) succeed and emit telemetry events.
- Contract violations appear in the run manifest and telemetry stream with stable codes.

### Equivalence Contract: CPU Debug vs Strict GPU

`cpu_debug` and `strict_gpu` produce the same `RunManifest` shape but **not** byte-identical numerical outputs. Tests assert equivalence using a documented field-class contract:

| Field class | Match level | Examples |
| --- | --- | --- |
| Identity / schema | Exact | run_id, video_id, card_instance_id, manifest version, stage list, runtime_mode |
| Counts | Exact | cards_extracted, tracks_resolved, stage durations present |
| Geometric | Tolerance | corner coordinates ± 2px, warp output corner pixels |
| Quality scores | Tolerance | sharpness ± 5%, glare ± 5%, total score ± 0.02 |
| Hashes | Hamming distance | pHash within 4 bits |
| Embeddings | Cosine | ReID embedding cosine similarity ≥ 0.95 |

`tests/runtime/test_cpu_debug_strict_gpu_equivalence.py` applies the contract to a small fixture run. Larger golden-corpus runs use the same contract.

CPU debug tests additionally assert:

- CPU debug is selected explicitly through runtime mode, not by missing GPU dependencies.
- CPU debug returns the same manifest shape as strict GPU (per the table above).

### Telemetry and Manifest Tests

Every runtime and platform adapter has tests that verify:

- A `RunManifest` round-trips through JSON without provider-specific assumptions.
- Stage timing, runtime mode, backend capability, warnings, and contract violations are present when expected.
- The app can import results only from the manifest and repositories.
- The telemetry adapter can persist SQLite events without pipeline steps writing SQL directly.

### Performance Regression Tests

Add a lightweight performance harness with explicit tolerances and artifacts. It should be easy for an agent to run an experiment, inspect telemetry, change code, rerun, and compare results.

Required shape:

- Store one or more benchmark videos under a documented local path or fixture fetch mechanism. At least one represents the real 4K phone HEVC workload.
- Provide one command, e.g. `python -m harness.performance run --profile v5_5_local --video IMG_5922.MOV --out reports/perf/<run_id>`.
- Emit `run_manifest.json`, telemetry JSON/JSONL, machine/runtime capability metadata, git SHA, command/config params, and a compact comparison report.
- Record decode count, model-load count, frame reread count, GPU/CPU utilization samples, memory samples, stage timings, cards produced, and quality/regression summary fields.
- Assert one runtime process, one model load per model per run, and no post-detect source-video reopen except an explicit fallback.
- Allow advisory thresholds at first, then tighten into blocking thresholds once the baseline stabilizes.

Research direction:

- Use the OpenAI Evals pattern as inspiration: dataset/cases, runner, recorder, aggregate report. The useful part is the shape, not the dependency: <https://github.com/openai/evals>.
- Use `pytest-benchmark` for microbenchmarks where pytest integration and compare mode are enough: <https://pytest-benchmark.readthedocs.io/>.
- Consider ASV only for isolated Python microbenchmarks and long-term trend dashboards: <https://asv.readthedocs.io/>.

### Skipped Test Policy

Skipped tests are allowed only when they represent unavailable external capability (missing CUDA hardware, missing provider credentials, etc.), and the skip reason must name the capability. Skips that hide known failures are converted into explicit failing tests or moved behind a clearly named quarantine marker with an issue reference.

V5.5 includes a cleanup pass that removes stale skipped tests, turns planned-but-missing tests into real tests, and documents any remaining hardware/provider skips.

### Open Questions

- Which test markers identify CUDA, MPS, provider, benchmark, and slow real-video tests.
- Whether architecture tests should also run as a pre-commit hook.
- Which real video becomes the canonical performance baseline for V5.5.

---

## Section 7: CI

CI should separate fast architectural feedback from slower hardware and provider validation. The default PR lane should be cheap and deterministic; GPU/provider lanes should run when relevant, on demand, or nightly.

### Required PR Lanes

Run on every PR:

- Formatting and ordinary lint/type checks, using the repo's existing tools.
- **Import Linter contract validation.**
- **GPU-strict AST scan** (forbidden call sites in tagged files).
- **Raw-SQL AST scan** (SQL strings outside allowlisted modules).
- CPU-only unit tests, including CPU debug runtime contract tests.
- Architecture boundary tests (anything Import Linter / AST scans cannot express).
- Manifest serialization tests.
- Data access layer tests against temporary SQLite databases, including the single-writer queue.
- Provider adapter contract tests with fake transports only.
- App runner/importer tests that prove app code consumes `PipelineRunner`, manifests, and repositories.
- Skip audit that fails on unexplained skips or skips masking known failures.

This lane must not require CUDA, provider credentials, local model downloads, or large video assets.

### Optional Hardware Lanes

Run on labels, manual dispatch, or branches that touch GPU/runtime code:

- CUDA strict-runtime tests on a CUDA runner with NVDEC capability preflight.
- MPS/CoreML smoke tests on Apple Silicon, if such a runner is available outside GitHub-hosted CI.
- GPU guard tests that monkeypatch forbidden CPU operations and assert strict sections fail correctly.
- Small real-video smoke runs that verify no post-detect reread and no repeated model load.

Hardware lanes publish manifests, telemetry summaries, and capability reports as artifacts even when they fail.

### Nightly / Manual Integration Lanes

- Real-video performance baseline against the selected V5.5 benchmark video.
- Provider smoke tests for RunPod and Beam using test credentials.
- Vast.ai smoke test only if Vast.ai remains a supported runtime platform.
- End-to-end app submission/import test using a fake or disposable artifact store.
- Golden-corpus regression report.

Nightly jobs trend timings, decode counts, model-load counts, frame reread counts, and card-output deltas. CI fails only on clear contract violations at first; performance thresholds tighten after the baseline is stable.

### CI Artifacts

Each relevant lane uploads:

- `run_manifest.json`.
- Structured telemetry summary.
- Runtime capability/preflight report.
- Architecture-test report.
- Performance baseline report, when run.

These artifacts are part of the debugging contract. The goal is to diagnose CI failures from structured output, not stdout scraping.

### CI Prerequisite

Before beginning the V5.5 refactor, fix existing CI failures. The refactor must not start on top of a red baseline because it will be impossible to distinguish architecture regressions from existing breakage.

Prerequisite work:

- Make the default CI lane green.
- Remove or justify skipped tests.
- Convert planned tests that already describe desired confidence into real tests before changing the architecture they protect.
- Add architecture tests (Import Linter + AST scanners) for the highest-risk boundaries before moving code.
- Add telemetry/manifest tests before replacing runner behavior.
- Add the first performance harness smoke test before optimizing the single-process runtime.

### Open Questions

- Which CI provider can supply CUDA/NVDEC and, separately, Apple Silicon for MPS/CoreML validation.
- Whether provider smoke tests should run in GitHub Actions or in a scheduled external runner.
- What threshold turns performance reporting from advisory into blocking.

---

## Section 8: Runtime Platforms

Runtime platforms should differ only at the runner, transport, artifact, and preflight boundaries. They all execute the same `PipelineRequest` contract and return the same `RunManifest` shape.

### Platform Matrix

| Platform | Purpose | Runtime mode | Execution shape | Artifact model |
| --- | --- | --- | --- | --- |
| Local CPU debug | Deterministic local development and CI | `cpu_debug` | In-process `PipelineRuntime` | Local paths |
| Local CUDA | Primary single-machine production path | `strict_gpu` | In-process `PipelineRuntime` | Local paths or configured artifact root |
| Local MPS | Apple Silicon single-machine path | `strict_gpu` | In-process `PipelineRuntime` | Local paths |
| RunPod serverless | Remote GPU execution | `strict_gpu` | Worker deserializes request, runs runtime once, uploads manifest | Object storage / signed URLs |
| Beam endpoint | Remote GPU execution | `strict_gpu` | Endpoint/job adapter implements `PipelineRunner` | Beam volume/object references |
| Vast.ai | Legacy/manual GPU capacity unless retained | `strict_gpu` | Explicit lifecycle adapter or deprecated | SSH/rsync/object storage, if supported |

### Shared Platform Contract

Every platform adapter implements:

- Validate runtime capability before accepting work.
- Materialize or reference the input video.
- Execute exactly one runtime call per run.
- Persist outputs and a manifest.
- Return provider-neutral status, telemetry, warnings, and failure codes.
- Avoid writing database state directly; app-side import owns persistence through repositories.

Provider adapters may own upload/download, polling, cancellation, credential handling, and failure translation. They must not own stage ordering, CPU/GPU fallback policy, scoring thresholds, storage schema, or app-facing result shaping.

### Preflight Contract

Strict GPU platforms must report:

- CUDA / MPS availability and device name.
- Driver/runtime versions.
- NVDEC or chosen decode backend availability.
- Torch/Kornia/model runtime availability.
- Model artifact presence and version.
- Writable output/artifact location.
- Expected memory floor for the configured video size.

Preflight failures map to stable codes: `missing_cuda`, `missing_mps`, `missing_nvdec`, `missing_model`, `insufficient_vram`, `artifact_store_unavailable`, `unsupported_runtime_mode`.

### Artifact Transfer

Manifests refer to artifacts through typed references rather than provider-specific paths:

```text
artifact://local/run_id/cards/...
artifact://s3/bucket/key
artifact://beam/volume/path
artifact://runpod/job_id/path
```

The app imports a completed run by reading the manifest and asking the platform/artifact layer to resolve references. It must not infer provider paths from run IDs.

### Failure and Telemetry Mapping

All platforms map provider failures into common categories:

- `preflight_failed`
- `submission_failed`
- `input_transfer_failed`
- `runtime_contract_failed`
- `runtime_execution_failed`
- `output_transfer_failed`
- `result_import_failed`
- `cancelled`
- `timeout`

Original provider payloads may be retained in debug metadata, but app logic and user-facing status use the common categories.

### Platform Decisions

- Local CPU debug stays first-class but explicit.
- Local CUDA is the reference implementation for strict GPU behavior and single-process execution; local MPS implements the same logical contract with its own backend code.
- RunPod and Beam are normal adapters, not special cases in app or pipeline code.
- Vast.ai is either brought behind the same adapter contract or deprecated. A half-supported lifecycle path is worse than no official support because it encourages provider-specific assumptions.

### Open Questions

- Whether Beam or RunPod is the first remote adapter completed against the new contract.
- Whether artifact storage should standardize on object storage for all remote platforms.
- Whether Vast.ai is still worth maintaining after RunPod/Beam support stabilizes.
- How cancellation and partial-output cleanup behave across providers.

---

## Migration Strategy

### Phase 0: CI and Confidence Baseline

- Fix existing CI failures.
- Remove stale skipped tests or convert them into explicit hardware/provider skips.
- Implement the planned contract, telemetry, manifest, architecture, and performance-harness smoke tests needed to protect the refactor.
- Establish the first real-video performance baseline and store its manifest/telemetry artifacts.

### Phase 1: Contracts, Telemetry, and Static Enforcement

- Add `docs/architecture/standards.md` with the initial binding rules from this design, **phase-marked** (advisory vs blocking per phase).
- Add runtime interfaces and result manifest contracts.
- Add structured telemetry abstraction; OpenTelemetry Metrics as the primary instrumentation API. Traces and sampling out of scope.
- Add Import Linter with the contracts in Section 4.
- Add the narrow AST scanners (GPU-strict calls, raw SQL) — initially as advisory PR-lane reports.
- Stop adding new stdout parsing dependencies.
- Keep current pipeline behavior otherwise unchanged.

### Phase 2: Strict GPU Boundary

- Introduce `GpuSession` and device-tagged batch types.
- Implement the approved CPU-export boundary helpers (Section 2 table).
- Add `monkeypatch.context()`-based strict-guard tests; the wrapper `StrictGpuRuntime` exposes only safe APIs.
- Move GPU hot-path code behind the new session boundary; update Import Linter / AST scopes accordingly.
- Convert silent CPU fallbacks into explicit backend selection.
- Tighten Phase-1 advisory checks into blocking.

### Phase 3: Single-Process Runtime and Metaflow Removal

This phase **must complete before** further MPS-perf optimizations bake in assumptions about the per-stage subprocess shape. The current `codex/v5-5-document-stubs` branch's perf work depends on the new in-process runtime; do not lock in optimizations that assume Metaflow stages.

- Move timing writes out of legacy flow step bodies.
- Replace hidden runtime selection with explicit branch selection.
- Route local execution through one in-process runtime call instead of per-stage Metaflow subprocesses. Generalize the `fused_refine` pattern (single process, frames in memory) per backend — the CUDA implementation is precedent, not a portable reference.
- Delete Metaflow flow code, tests, dependencies, and documentation once the replacement runtime/runner path is covered.
- Add the Metaflow-forbidden Import Linter contract.

### Phase 4: Data Access and Single-Writer

- Move database writes and reads behind `card_capture.data` repositories.
- Implement the single-writer discipline (Section 5) covering both pipeline runtime and FastAPI app writers.
- Tighten the raw-SQL AST scanner from advisory to blocking.

### Phase 5: Platform Adapter Cleanup

- Make Beam, RunPod, local, and any remaining Vast.ai runner implement the same runner contract.
- Import results from manifests.
- Remove provider-specific result assumptions from app-facing code.

---

## Acceptance Criteria

### Single-process runtime

- A local run executes all stages in a single process; no stage is a separate OS process.
- Models are instantiated at most once per run.
- `refine` (and any post-detect stage) consumes decoded frames produced by `detect`; no module re-opens the source video after detect except an explicit, telemetered fallback.
- Metaflow imports, flow definitions, dependency declarations, and tests are removed.
- Resume is per-run only; a failed run reruns from the input video.

### Strict GPU execution

- A production GPU run cannot silently use CPU fallback inside strict execution.
- Hidden host transfers and frame rereads fail in strict tests (via `monkeypatch.context()`) and in CI (via the AST scanner).
- The approved CPU-export boundary list in Section 2 is implemented as named helpers and is the only `.cpu()`/`.numpy()` call path inside strict GPU code.
- `strict_gpu` and `cpu_debug` runs satisfy the equivalence contract (Section 6) on a fixture run.

### Static enforcement

- Import Linter contracts encode the package-boundary rules and run in the PR lane.
- The GPU-strict AST scanner enforces forbidden call sites inside files tagged GPU-resident.
- The raw-SQL AST scanner enforces the data-layer boundary.
- `docs/architecture/standards.md` exists, states the binding architecture rules AI agents must follow, and phase-marks each rule.
- At least the highest-risk standards have automated enforcement through tests, lint, or runtime guards.

### Data access

- Raw SQL is restricted to the data access layer, migration/schema code, and explicit test helpers.
- Pipeline, app service, and platform code use repository/query interfaces for database access.
- SQLite writes from pipeline, FastAPI app, and harness are serialized through a documented single-writer mechanism; `database is locked` errors are no longer expected under normal load.

### Platforms

- Beam can be completed by implementing a platform adapter, not by editing core pipeline logic.
- All supported platforms accept the same request contract and return the same manifest shape.
- Provider-specific failures are mapped to stable, provider-neutral categories before app-facing status handling.
- CPU debug remains available as an explicit runtime backend.

### CI and operations

- The default CI lane is green before the refactor begins.
- Stale skipped tests are removed or converted into explicit hardware/provider skips.
- CI has a fast CPU-only lane that validates contracts and architecture without provider credentials or GPU hardware.
- CUDA/provider/performance lanes emit manifests, telemetry summaries, and capability reports as artifacts when they run.
- A documented performance harness can run at least one real-video benchmark and produce comparable telemetry artifacts.

### Behavior

- Existing algorithmic outputs are preserved unless a later algorithm-specific design says otherwise.
