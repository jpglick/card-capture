# V5.5 Refactoring Design

**Date:** 2026-05-24  
**Status:** Draft  
**Scope:** Software engineering refactors for pipeline correctness, performance enforcement, and runtime portability

## Decision Posture

This document is a working collection of thoughts from developing, profiling, and running the system. Everything here is open to debate, revision, or rejection as new evidence appears.

The project owner is the final authority for V5.5 decisions. External architecture patterns, framework conventions, agent recommendations, and implementation plans are inputs to evaluate, not obligations. The deciding standard is what best serves this system's correctness, performance, maintainability, and operating reality.

---

## Problem

The V4 pipeline has accumulated optimization work inside boundaries that do not enforce the properties we care about. Metaflow is used as a subprocess-oriented orchestration shell, the app runner parses logs for timing data, CPU/GPU behavior is selected through scattered flags and conditional code paths, and runtime-specific concerns leak across app, worker, and pipeline modules.

The result is that performance regressions are too easy to introduce. Hidden CPU reads, implicit tensor copies, silent fallback behavior, and repeated decode/load work can all appear as local implementation details instead of contract violations. V5.5 should make those mistakes structurally difficult.

This document focuses on software engineering refactoring. It does not propose new card-detection, tracking, scoring, fusion, deduplication, or OCR algorithms.

---

## Goals

- Make pipeline execution contracts explicit enough that hidden CPU work and repeated load/decode work fail fast.
- Remove Metaflow from the codebase and replace it with a lightweight runtime/runner contract that is small enough to enforce.
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
- No new general-purpose workflow/DAG framework unless a future decision explicitly chooses one.
- No attempt to make CPU fallback transparent in production.

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
- CI/architecture-lint expectations.
- A short "when unsure" escalation rule for agents.

### Example Standards

Initial standards should include rules such as:

- Metaflow must be removed from production code. New orchestration code must use the lightweight runtime/runner contracts instead of a workflow framework.
- Production GPU execution must use the strict GPU runtime; it must not silently fallback to CPU.
- CPU execution must be selected through an explicit CPU debug backend.
- Backend-specific code may duplicate small amounts of logic when the alternative is conditionals that obscure runtime ownership.
- GPU hot-path modules must not call `cv2.VideoCapture`, `cv2.imread`, `PIL.Image.open`, `torch.Tensor.cpu`, or `torch.Tensor.numpy` except through approved boundary modules.
- App services, pipeline runtime code, and platform runners must not open SQLite connections directly.
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

## Section 1: Metaflow Removal and Lightweight Runtime Contract

### Current Issues

- The app invokes Metaflow as a subprocess and parses stdout for step timings.
- Stage timing is duplicated between flow code, runner diagnostics, and handler diagnostics.
- The flow mostly behaves as a linear script wrapper, so Metaflow adds process boundaries, import/model reloads, datastore serialization, and graph semantics without providing value proportional to that cost.
- Runtime decisions such as CPU debug versus GPU execution are represented indirectly through config files, environment variables, and detector names.
- Metaflow has been a hindrance for this system because it makes the fast local path harder: state cannot stay in memory, frames cannot stay decoded, and GPU-resident work cannot cross step boundaries.

### Target Shape

Remove Metaflow from the production code path. The replacement should be deliberately small:

- A `PipelineRuntime` protocol that executes one run in one process.
- A `PipelineRunner` protocol that handles local or remote submission.
- A `PipelineTelemetry` / metrics sink that receives structured events and samples.
- A `RunManifest` model that is the only interchange format between runtime, platform adapters, and app import.
- A small CLI entrypoint for local runs, smoke runs, and performance harness runs.

The local stage sequence should run inside one `PipelineRuntime` process so decoded frames, loaded models, GPU sessions, and process-local caches can be reused across stages. Remote platforms can run the same request contract inside their worker process and return the same manifest shape.

The replacement should not own:

- GPU memory residency.
- CPU/GPU fallback policy inside hot-path compute.
- Provider-specific upload/download mechanics.
- App-facing telemetry formatting.
- Retry semantics beyond explicit, coarse per-run retry.
- A general DAG, step scheduler, or artifact datastore.

### Proposed Refactor

Introduce a tiny runtime surface:

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

Runtime mode should be an explicit field in `PipelineRunRequest`, not hidden environment state. The selected runtime implementation owns the full execution contract in one process.

Values passed between runtime, runner, app import, and performance harness code should remain serializable:

- Run IDs.
- Input and output paths.
- Manifest paths.
- Metrics and telemetry summaries.
- Final result references.

They should not include CUDA tensors, model objects, open video handles, worker processes, or other process-local resources.

### Backend Duplication Policy

Do not force all backends through one implementation when their constraints differ. It is acceptable, and often preferable, to duplicate small pieces of code in separate backend modules when the shared alternative would introduce conditionals such as:

- `if cuda`.
- `if mps`.
- `if cpu_debug`.
- `if remote_provider`.
- `if has_gpu_decode`.

Shared code should be limited to stable contracts, pure geometry/math utilities, manifest/telemetry schemas, and genuinely backend-neutral helpers. Runtime-specific decode, model loading, frame caching, memory transfer, and export rules should live with the backend that owns them.

### Structured Telemetry

Replace stdout timing parsing with a single telemetry and metrics surface. The custom `PipelineTelemetry` protocol should remain the application-facing contract, while the implementation can publish metrics through a real library.

Candidate direction:

- Use OpenTelemetry Metrics as the primary instrumentation API for counters, histograms, gauges, and resource attributes. The OpenTelemetry Python metrics API is built around a `MeterProvider`, `Meter`, and instruments for recording measurements: <https://opentelemetry-python.readthedocs.io/en/latest/api/metrics.html>.
- Keep a project-local `PipelineTelemetry` facade so app code, runtime code, and tests do not depend directly on one vendor/exporter shape.
- Write telemetry to at least two sinks: structured JSON/manifest output for run debugging, and a metrics exporter for local/performance/production analysis.
- Evaluate MLflow Tracking for performance experiments because it already models runs, params, metrics, and artifacts: <https://mlflow.org/docs/latest/ml/tracking>.

The runtime, app runner, platform handlers, and performance harness should consume the same telemetry stream. SQLite events may remain the app compatibility layer, but they should be written by telemetry adapters rather than ad hoc timing calls in each stage.

### Open Questions

- Whether OpenTelemetry alone is enough, or whether MLflow should be added specifically for experiment tracking.
- Whether the metrics exporter should default to local JSON, Prometheus/OpenMetrics, OTLP, or a combination.
- Whether current per-stage names should be preserved for operator familiarity in telemetry, even though stage execution is no longer a workflow graph.

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
        batches.py                device-tagged batch/result types
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
        repositories/
```

Existing algorithm modules such as detection, sampling, tracking, scoring, fusion, and ML inference can stay under their current domain packages initially. The refactor should wrap them behind `card_capture.pipeline.stages` and runtime backends before moving files. File moves should happen only after tests make ownership clear.

### Boundary Rules

- `pipeline/` is orchestration/submission only. It may create a `PipelineRequest`, call a runner/runtime, and persist a manifest reference. It should not contain algorithmic stage bodies.
- `card_capture.pipeline` owns the logical pipeline sequence, request/result contracts, telemetry contracts, and stage facades.
- `card_capture.runtime` owns backend selection, GPU session lifecycle, strict GPU guards, CPU debug implementations, and device-residency types.
- `card_capture.platforms` owns provider transport, preflight, artifact upload/download, remote job lifecycle, and provider failure mapping.
- `card_capture.data` owns database connections, migrations/schema helpers, repositories, and app-facing query services.
- `app/services` owns HTTP/API-facing workflow state and user-facing status mapping. It should call `PipelineRunner` or data repositories rather than importing provider clients or opening SQLite directly.
- Remote worker entrypoints should be thin shells that deserialize a request, call the same runtime contract, and write a manifest. They should not fork their own pipeline semantics.

### GPU-Only Boundary

Strict GPU code needs a boundary that is enforceable by tests:

- GPU hot-path modules must require `GpuSession` or a device-tagged batch type for entry.
- GPU hot-path modules may import Torch/Kornia/CUDA decode helpers and runtime guards.
- GPU hot-path modules must not import OpenCV image/file IO helpers, PIL image loaders, `sqlite3`, app services, or platform adapters.
- CPU debug modules may share pure math/geometry utilities, but should not be imported by strict GPU modules.
- Shared dataclasses should live in contract modules with no heavyweight runtime side effects.

### Migration Direction

1. Add the new packages as wrappers around existing code.
2. Move shared request/result/manifest/telemetry contracts out of top-level `pipeline/contracts.py` once stable.
3. Route local execution through `card_capture.pipeline.runtime.PipelineRuntime` before moving provider adapters.
4. Move direct provider logic out of app services into `card_capture.platforms`.
5. Move database writes and reads behind `card_capture.data`.
6. Add architecture tests for import direction before doing large file moves.
7. Delete Metaflow modules, tests, and dependencies after call sites have moved and the replacement runtime contract is covered.

### Open Questions

- Whether the top-level `pipeline/` package should remain long term or become a compatibility layer only.
- Whether provider adapters should live in `card_capture.platforms` or under `app/services` until the API boundary settles.
- Whether current worker modules should be split by platform or collapsed behind one worker entrypoint.
- Whether top-level `pipeline/contracts.py` should be deleted immediately or kept as a short-lived compatibility import while the new contract package is introduced.

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

- Legacy orchestration wrappers.
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

Testing for V5.5 should prove the architecture boundaries, not only the happy-path card outputs. The current risk is that a future change can reintroduce host transfers, raw SQL, provider leakage, or per-stage process overhead while still passing narrow algorithm tests.

### Test Categories

Add focused suites:

- `tests/architecture/`: static import and raw-SQL boundary checks.
- `tests/runtime/`: `PipelineRuntime`, `StrictGpuRuntime`, `CpuDebugRuntime`, `GpuSession`, device-tagged batches, and forbidden-op guards.
- `tests/platforms/`: local, RunPod, Beam, and Vast.ai adapters using fake transports and fake artifact stores.
- `tests/data/`: repository contracts, migration/schema behavior, and transactional invariants.
- `tests/pipeline/`: stage sequencing, manifest generation, telemetry emission, and runtime contract smoke tests.
- `tests/performance/`: benchmark harness tests, telemetry artifact validation, and sample-video fixture checks.
- `tests/regression/`: golden corpus and performance baselines.

### Contract Tests

Strict runtime tests should assert:

- `strict_gpu` fails when CUDA/NVDEC requirements are missing unless the test explicitly uses a fake capability.
- Hot-path functions reject CPU tensors, path-only frame references, or batches without a `GpuSession`.
- Forbidden operations such as `cv2.VideoCapture`, `cv2.imread`, `PIL.Image.open`, `.cpu()`, and `.numpy()` fail inside guarded strict sections.
- Final export boundaries are explicit and telemetered.
- Contract violations appear in the run manifest and telemetry stream with stable codes.

CPU debug tests should assert:

- CPU debug is selected explicitly through runtime mode, not by missing GPU dependencies.
- CPU debug returns the same manifest shape as strict GPU.
- A small fixture run preserves stage ordering, card identity fields, and output schema even if numerical quality metrics differ slightly.

### Architecture Tests

Static tests should fail on:

- Raw SQL or `sqlite3.connect` outside `card_capture.data`, migrations, and allowlisted schema tests.
- Provider SDK/client imports outside `card_capture.platforms` and explicit app composition roots.
- App service imports from GPU hot-path modules.
- GPU hot-path imports from app, platform, data, PIL image loading, or OpenCV file IO modules.
- Any Metaflow import, flow definition, or workflow framework dependency after the removal phase.

These tests should be simple AST/import scanners with allowlists checked into the repo. The goal is to stop architectural drift early, not to build a general linter.

### Telemetry and Manifest Tests

Every runtime and platform adapter should have tests that verify:

- A `RunManifest` round-trips through JSON without provider-specific assumptions.
- Stage timing, runtime mode, backend capability, warnings, and contract violations are present when expected.
- The app can import results only from the manifest and repositories.
- The telemetry adapter can persist SQLite events without pipeline steps writing SQL directly.

### Performance Regression Tests

Add a lightweight performance harness with explicit tolerances and artifacts. It should be easy for an agent to run an experiment, inspect telemetry, change code, rerun, and compare results.

Required shape:

- Store one or more benchmark videos under a documented local path or fixture fetch mechanism. At least one should represent the real 4K phone HEVC workload.
- Provide one command, for example `python -m harness.performance run --profile v5_5_local --video IMG_5922.MOV --out reports/perf/<run_id>`.
- Emit `run_manifest.json`, telemetry JSON/JSONL, machine/runtime capability metadata, git SHA, command/config params, and a compact comparison report.
- Record decode count, model-load count, frame reread count, GPU/CPU utilization samples, memory samples, stage timings, cards produced, and quality/regression summary fields.
- Assert one runtime process, one model load per model per run, and no post-detect source-video reopen except an explicit fallback.
- Allow advisory thresholds at first, then tighten into blocking thresholds once the baseline stabilizes.

Research direction:

- Use the OpenAI Evals pattern as inspiration: dataset/cases, runner, recorder, and aggregate report. The useful part is the shape, not the dependency: <https://github.com/openai/evals>.
- Use `pytest-benchmark` for microbenchmarks where pytest integration and compare mode are enough: <https://pytest-benchmark.readthedocs.io/>.
- Consider ASV only for isolated Python microbenchmarks and long-term trend dashboards; full-video GPU experiments probably need the custom harness because they depend on hardware, videos, artifacts, and telemetry: <https://asv.readthedocs.io/>.
- Consider MLflow Tracking for experiment/run storage if local JSON reports become too hard to compare across branches and machines: <https://mlflow.org/docs/latest/ml/tracking>.

### Skipped Test Policy

Skipped tests are allowed only when they represent unavailable external capability, such as missing CUDA hardware or missing provider credentials, and the skip reason must name the capability. Skips that hide known failures should be converted into explicit failing tests or moved behind a clearly named quarantine marker with an issue/reference.

V5.5 should include a cleanup pass that removes stale skipped tests, turns planned-but-missing tests into real tests, and documents any remaining hardware/provider skips.

### Open Questions

- Which test markers should identify CUDA, MPS, provider, benchmark, and slow real-video tests.
- Whether architecture tests should live under pytest only or also run as a pre-commit hook.
- Which real video becomes the canonical performance baseline for V5.5.
- Whether the performance harness should store experiment history as local files only or use MLflow from the start.

---

## Section 7: CI

CI should separate fast architectural feedback from slower hardware and provider validation. The default PR lane should be cheap and deterministic; GPU/provider lanes should run when relevant, on demand, or nightly.

### Required PR Lanes

Run on every PR:

- Formatting and ordinary lint/type checks, using the repo's existing tools.
- CPU-only unit tests, including CPU debug runtime contract tests.
- Architecture boundary tests for imports, raw SQL, provider leakage, and forbidden Metaflow/workflow framework usage.
- Manifest serialization tests.
- Data access layer tests against temporary SQLite databases.
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

Hardware lanes should publish manifests, telemetry summaries, and capability reports as artifacts even when they fail.

### Nightly / Manual Integration Lanes

Run slower validation outside the default PR path:

- Real-video performance baseline against the selected V5.5 benchmark video.
- Provider smoke tests for RunPod and Beam using test credentials.
- Vast.ai smoke test only if Vast.ai remains a supported runtime platform.
- End-to-end app submission/import test using a fake or disposable artifact store.
- Golden-corpus regression report.

Nightly jobs should trend timings, decode counts, model-load counts, frame reread counts, and card-output deltas. CI should fail only on clear contract violations at first; performance thresholds can tighten after the baseline is stable.

### CI Artifacts

Each relevant lane should upload:

- `run_manifest.json`.
- Structured telemetry summary.
- Runtime capability/preflight report.
- Architecture-test report.
- Performance baseline report, when run.

These artifacts are part of the debugging contract. The goal is to diagnose CI failures from structured output, not stdout scraping.

### CI Prerequisite

Before beginning the V5.5 refactor, fix existing CI failures. The refactor should not start on top of a red baseline because it will be impossible to distinguish architecture regressions from existing breakage.

Prerequisite work:

- Make the default CI lane green.
- Remove or justify skipped tests.
- Convert planned tests that already describe desired confidence into real tests before changing the architecture they protect.
- Add architecture tests for the highest-risk boundaries before moving code.
- Add telemetry/manifest tests before replacing runner behavior.
- Add the first performance harness smoke test before optimizing the single-process runtime.

### Open Questions

- Which CI provider can supply CUDA/NVDEC and, separately, Apple Silicon for MPS/CoreML validation.
- Whether provider smoke tests should run in GitHub Actions or in a scheduled external runner.
- What threshold turns performance reporting from advisory into blocking.

---

## Section 8: Runtime Platforms

Runtime platforms should differ only at the runner, transport, artifact, and preflight boundaries. They should all execute the same `PipelineRequest` contract and return the same `RunManifest` shape.

### Platform Matrix

| Platform | Purpose | Runtime mode | Execution shape | Artifact model |
| --- | --- | --- | --- | --- |
| Local CPU debug | Deterministic local development and CI | `cpu_debug` | In-process `PipelineRuntime` | Local paths |
| Local CUDA | Primary single-machine production path | `strict_gpu` | In-process `PipelineRuntime` | Local paths or configured artifact root |
| RunPod serverless | Remote GPU execution | `strict_gpu` | Worker deserializes request, runs runtime once, uploads manifest | Object storage / signed URLs |
| Beam endpoint | Remote GPU execution | `strict_gpu` | Endpoint/job adapter implements `PipelineRunner` | Beam volume/object references |
| Vast.ai | Legacy/manual GPU capacity unless retained | `strict_gpu` | Explicit lifecycle adapter or deprecated | SSH/rsync/object storage, if supported |

### Shared Platform Contract

Every platform adapter should implement the same responsibilities:

- Validate runtime capability before accepting work.
- Materialize or reference the input video.
- Execute exactly one runtime call per run.
- Persist outputs and a manifest.
- Return provider-neutral status, telemetry, warnings, and failure codes.
- Avoid writing database state directly; app-side import owns persistence through repositories.

Provider adapters may own upload/download, polling, cancellation, credential handling, and failure translation. They should not own stage ordering, CPU/GPU fallback policy, scoring thresholds, storage schema, or app-facing result shaping.

### Preflight Contract

Strict GPU platforms must report:

- CUDA availability and device name.
- Driver/runtime versions.
- NVDEC or chosen decode backend availability.
- Torch/Kornia/model runtime availability.
- Model artifact presence and version.
- Writable output/artifact location.
- Expected memory floor for the configured video size.

Preflight failures should map to stable codes such as `missing_cuda`, `missing_nvdec`, `missing_model`, `insufficient_vram`, `artifact_store_unavailable`, and `unsupported_runtime_mode`.

### Artifact Transfer

Manifests should refer to artifacts through typed references rather than provider-specific paths:

```text
artifact://local/run_id/cards/...
artifact://s3/bucket/key
artifact://beam/volume/path
artifact://runpod/job_id/path
```

The app imports a completed run by reading the manifest and asking the platform/artifact layer to resolve references. It should not infer provider paths from run IDs.

### Failure and Telemetry Mapping

All platforms should map provider failures into common categories:

- `preflight_failed`
- `submission_failed`
- `input_transfer_failed`
- `runtime_contract_failed`
- `runtime_execution_failed`
- `output_transfer_failed`
- `result_import_failed`
- `cancelled`
- `timeout`

The original provider payload can be retained in debug metadata, but app logic and user-facing status should use the common categories.

### Platform Decisions

- Local CPU debug stays first-class but explicit.
- Local CUDA is the reference implementation for strict GPU behavior and single-process execution.
- RunPod and Beam should be normal adapters, not special cases in app or pipeline code.
- Vast.ai should either be brought behind the same adapter contract or deprecated. A half-supported lifecycle path is worse than no official support because it encourages provider-specific assumptions.

### Open Questions

- Whether Beam or RunPod should be the first remote adapter completed against the new contract.
- Whether artifact storage should standardize on object storage for all remote platforms.
- Whether Vast.ai is still worth maintaining after RunPod/Beam support stabilizes.
- How cancellation and partial-output cleanup should behave across providers.

---

## Migration Strategy

### Phase 0: CI and Confidence Baseline

- Fix existing CI failures.
- Remove stale skipped tests or convert them into explicit hardware/provider skips.
- Implement the planned contract, telemetry, manifest, architecture, and performance-harness smoke tests needed to protect the refactor.
- Establish the first real-video performance baseline and store its manifest/telemetry artifacts.

### Phase 1: Contracts and Telemetry

- Add `docs/architecture/standards.md` with the initial binding rules from this design.
- Add runtime interfaces and result manifest contracts.
- Add structured telemetry abstraction and choose the first metrics implementation.
- Stop adding new stdout parsing dependencies.
- Keep current pipeline behavior otherwise unchanged.

### Phase 2: Strict GPU Boundary

- Introduce `GpuSession` and device-tagged batch types.
- Add strict runtime guard in tests.
- Move GPU hot-path code behind the new session boundary.
- Convert silent CPU fallbacks into explicit backend selection.

### Phase 3: Single-Process Runtime and Metaflow Removal

- Move timing writes out of legacy flow step bodies.
- Replace hidden runtime selection with explicit branch selection.
- Route local execution through one in-process runtime call instead of per-stage Metaflow subprocesses.
- Delete Metaflow flow code, tests, dependencies, and documentation once the replacement runtime/runner path is covered.

### Phase 4: Platform Adapter Cleanup

- Make Beam, RunPod, local, and any remaining Vast.ai runner implement the same runner contract.
- Import results from manifests.
- Remove provider-specific result assumptions from app-facing code.

---

## Acceptance Criteria

- A production GPU run cannot silently use CPU fallback inside strict execution.
- Hidden host transfers and frame rereads fail in strict tests.
- Metaflow is removed from production code and dependency manifests.
- Beam can be completed by implementing a platform adapter, not by editing core pipeline logic.
- Raw SQL is restricted to the data access layer, migration/schema code, and explicit test helpers.
- Pipeline, app service, and platform code use repository/query interfaces for database access.
- `docs/architecture/standards.md` exists and states the binding architecture rules AI agents must follow.
- At least the highest-risk standards have automated enforcement through tests, lint, or runtime guards.
- Package boundaries exist for pipeline contracts, runtime backends, platform adapters, and the data access layer.
- Architecture tests fail on new raw SQL, provider leakage, forbidden GPU imports, or forbidden workflow-framework usage.
- The default CI lane is green before the refactor begins.
- Stale skipped tests are removed or converted into explicit hardware/provider skips.
- CI has a fast CPU-only lane that validates contracts and architecture without provider credentials or GPU hardware.
- CUDA/provider/performance lanes emit manifests, telemetry summaries, and capability reports as artifacts when they run.
- A documented performance harness can run at least one real-video benchmark and produce comparable telemetry artifacts.
- All supported platforms accept the same request contract and return the same manifest shape.
- Provider-specific failures are mapped to stable, provider-neutral categories before app-facing status handling.
- CPU debug remains available as an explicit runtime backend.
- Existing algorithmic outputs are preserved unless a later algorithm-specific design says otherwise.

---

## Notes for Expansion

The main architectural decisions captured here are:

- Local execution is a single in-process runtime call; Metaflow is removed rather than retained as orchestration.
- GPU/CPU selection is a runtime/backend decision, not scattered conditionals.
- Backend-specific duplication is acceptable when it avoids misleading conditionals or shared code bent around one backend.
- Telemetry and metrics are first-class runtime outputs, not incidental logs.
- Performance testing needs a repeatable harness with sample videos, one-command execution, telemetry capture, and comparable reports.
- Repo structure should make pipeline contracts, runtime backends, platform adapters, and data access ownership visible.
- Database access is a repository/query-layer concern, not an incidental implementation detail in any module.
- The final V5.5 output includes an architectural standards document for AI agents and contributors.
- Strict GPU execution is enforced with session capabilities, device-tagged types, guards, and tests.
- Platform differences belong in runner adapters, preflight checks, artifact references, failure mapping, and manifests.

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

Metaflow is removed. The replacement is the lightweight `PipelineRuntime` / `PipelineRunner` surface in Section 1. Runtime selection still exists, but it is plain request data such as `runtime_mode = strict_gpu | cpu_debug`, not a workflow branch.

### What this amendment does NOT change

The rest of V5.5 is orthogonal and still wanted — none of it depends on a workflow graph:

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
- Metaflow imports, flow definitions, dependency declarations, and tests are removed.

### Migration note

This reorders the draft's Phase 3 ahead of where it sits today and makes it concrete: the single-process `PipelineRuntime` is the unit Phase 2's strict-GPU boundary should wrap, Metaflow removal is part of the refactor rather than a future option, and the `fused_refine` CUDA path is the reference implementation to generalize.
