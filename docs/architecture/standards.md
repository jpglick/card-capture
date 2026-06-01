# Card Capture Architecture Standards

> **Status:** Phase-marked. Each rule indicates the phase in which it becomes blocking.
> **Source:** `docs/superpowers/specs/2026-05-24-v5-5-refactoring-design.md`
> **For AI agents:** Read this file before changing any module listed under "Agent Triggers" below. If a requested change violates a rule, stop and surface the conflict.

## Agent Triggers

Read this document before editing any of:
- `pipeline/`
- `src/card_capture/pipeline/`
- `src/card_capture/runtime/`
- `src/card_capture/platforms/`
- `src/card_capture/data/`
- `src/card_capture/storage.py`
- `app/services/`
- `app/api/`
- CI workflow files
- `tests/architecture/`

## Rule Schema

Each rule carries:
- **Phase marker** — `advisory:phase-N` or `blocking:phase-N`.
- **Enforcement marker** — `static` (import lint / AST scan) / `runtime` (guard / assertion) / `review` (human only).

## Rules

### Pipeline Orchestration

- **R-ORCH-1** `blocking:phase-3` (static): No module imports `metaflow`.
- **R-ORCH-2** `blocking:phase-3` (static): No `@step` decorators or `FlowSpec` subclasses in production code.
- **R-ORCH-3** `blocking:phase-3` (runtime): A local run executes all stages in one process; the runtime opens the input video at most once except for explicit, telemetered fallback.
- **R-ORCH-4** `blocking:phase-3` (runtime): Each model is instantiated at most once per `PipelineRuntime.run()` call.

### Runtime Backends

- **R-RT-1** `blocking:phase-2` (static): GPU hot-path modules (per `pyproject.toml [tool.gpu_strict_lint] files`) must not call `cv2.VideoCapture`, `cv2.imread`, `PIL.Image.open`, `torch.Tensor.cpu`, or `torch.Tensor.numpy` except through the approved export helpers. (Enforced as of Phase 2).
- **R-RT-2** `blocking:phase-2` (runtime): `StrictGpuRuntime` raises `ContractViolation` on missing MPS, missing decode backend, or tensor host transfer outside an approved export boundary.
- **R-RT-3** `blocking:phase-2` (review): Production must not silently fall back to CPU. `runtime_mode` is explicit.
- **R-RT-4** `advisory:phase-2` → `blocking:phase-3` (review): Backend duplication is preferred over `if mps` / `if cpu_debug` conditionals in hot paths.

### Data Access

- **R-DATA-1** `blocking:phase-4` (static, Import Linter): No module outside `card_capture.data`, `migrations/`, or `tests/` imports `sqlite3`.
- **R-DATA-2** `blocking:phase-4` (static, raw-SQL scanner): No raw SQL strings outside the allowed roots.
- **R-DATA-3** `blocking:phase-4` (review): Pipeline, app service, and platform code use repository methods, not direct connections.
- **R-DATA-4** `blocking:phase-4` (runtime): SQLite writes are serialized through `card_capture.data.writer`. Both pipeline runtime and FastAPI handlers route writes through it.

### Telemetry

- **R-TEL-1** `blocking:phase-1` (review): Timing data is emitted through `PipelineTelemetry`. No new code parses stdout for timings.
- **R-TEL-2** `advisory:phase-1` (review): The OpenTelemetry Metrics adapter is the default sink. Distributed tracing and span sampling are out of scope for V5.5.
- **R-TEL-3** `blocking:phase-1` (review): Telemetry adapters write SQLite events; pipeline stages must not write events directly.

### Platforms

- **R-PLAT-1** `blocking:phase-5` (static, Import Linter): All execution platforms must live in `card_capture.platforms`.
- **R-PLAT-2** `blocking:phase-5` (review): Every platform returns a `RunManifest` of the same shape.

### Testing

- **R-TEST-1** `blocking:phase-0` (static, skip-audit): Every skipped test names its capability or quarantine reason.
- **R-TEST-2** `blocking:phase-2` (runtime): Strict-guard tests use `monkeypatch.context()` and do not patch outside the context block.
- **R-TEST-3** `blocking:phase-2` (review): `strict_gpu` and `cpu_debug` runs satisfy the equivalence contract on a fixture.

### CI

- **R-CI-1** `blocking:phase-0` (review): Default PR lane is green before refactor work starts.
- **R-CI-2** `blocking:phase-1` (review): PR lane runs Import Linter, GPU-strict AST scanner, and raw-SQL scanner (advisory in Phase 1).

## When Unsure

If a proposed change appears to violate a rule whose phase has been reached, stop and surface the conflict. If the rule is still advisory, document the violation in the commit message but proceed. If a new requirement is incompatible with a rule, propose a spec amendment before code changes.
