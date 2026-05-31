# v5.5.0 Release Tag & Post-Refactor Cleanup — Design

**Date:** 2026-05-31
**Status:** Approved (design)
**Branch:** cleanup work on `cleanup/v5.5-post-merge`; tag on `main`

## Context

The v5.5 refactor has merged to `main` (HEAD `c1e3b37a`, "V55 integration (#66)").
The repository still carries substantial scaffolding from earlier versions and from
a now-abandoned multi-provider cloud strategy (vast.ai, RunPod, Beam, Metaflow remote
orchestration, CUDA deployment). Development is shifting to **Apple Silicon exclusively**
for the foreseeable future. The pluggable provider architecture is acknowledged as weak,
but stale implementations are worse than an imperfect abstraction, so they are being
removed outright rather than left to rot.

This is a cleanup-and-documentation pass, not a feature change. The pipeline's runtime
behavior on the local MPS path must be unchanged when the work is done.

## Decisions (locked)

| Topic | Decision |
|---|---|
| Metaflow | **Remove entirely.** No remote orchestration. |
| CUDA | **Remove all CUDA.** MPS + CPU device paths only. |
| Historical plans/specs | **Preserve as record.** Dated `docs/superpowers/plans/` and `specs/` stay. Git history is the safety net regardless. |
| Authoritative arch docs | **Archive**, don't delete. |
| Cloud-specific docs | **Delete.** |
| `v5.5.0` tag | Annotated, on `main` HEAD, **created and pushed now**. |

## Goals

1. Tag the merged state as `v5.5.0` and push it.
2. Remove all cloud / vast.ai / RunPod / Beam / Metaflow / CUDA code, config, scripts, deps, and tests.
3. Archive authoritative prior-version architecture docs; delete cloud-only docs.
4. Produce `docs/architecture/arch-5.5.md` describing the merged pipeline at the depth of `arch-4.1.md`.
5. Bring operator-facing docs in line with the MPS-only local workflow.

## Non-Goals

- No behavioral change to the local MPS pipeline.
- No refactor of the (weak) runner abstraction beyond collapsing it to the local path.
- No pruning of dated historical plans/specs.
- No new features.

## Sequencing

All cleanup happens on `cleanup/v5.5-post-merge`. The tag is the sole action against
`main`, performed **first** so `v5.5.0` marks the exact merged state before any deletion.

1. **Tag** — annotated `v5.5.0` on `main` HEAD; push to origin.
2. **Working-tree triage** — resolve the pre-existing dirty state (see below) before branching.
3. **Cloud removal** — vast.ai / RunPod / Beam / R2 (one commit).
4. **CUDA removal** — device paths, sampler, detectors, GPU ops, build scaffolding (one commit; the deepest, test-gated).
5. **Metaflow removal** — flow, contract doc, `.metaflow/`, CLI path (one commit).
6. **Docs audit** — archive authoritative arch docs, delete cloud docs (one commit).
7. **New arch doc** — `arch-5.5.md` (one commit).
8. **Operator docs** — OPERATOR/README/QUICK_REFERENCE/CLAUDE/AGENTS/GEMINI, CI, Makefile (one commit).

Each removal commit must leave `pytest -m "not quarantine"` and `lint-imports` green.
If a removal cascades deeper than the surface below, stop and surface it rather than
gutting working pipeline code.

## Working-Tree Triage (pre-existing state)

At start the tree is dirty:
- `M scripts/diag_swap_signals.py` — modified; inspect the diff, keep or revert per intent.
- `?? pytest_output.log` — transient log; delete.
- `?? .gemini/` — inspect; likely keep (tool config) or gitignore.
- `?? docs/superpowers/plans/2026-05-28-v5-5-completion-phase-*.md`, `2024-05-24-dal-v55-migration.md`, `2025-05-15-test-client-lifespan.md` — historical plans; commit them (preserve-as-record) before cleanup so they are not swept up.

## Removal Surface

Paths are the real source tree (excluding `.venv/`, `.worktrees/`, `node_modules/`).
This is the audited starting set; the implementer verifies each file's references
before deletion and follows import cascades.

### Cloud (vast.ai / RunPod / Beam / R2)
- Delete: `app/runpod_handler.py`, `app/beam_handler.py`, `app/worker_core.py`,
  `app/services/runpod_runner.py`, `app/services/beam_runner.py`, `app/services/worker_client.py`.
- Delete: `src/card_capture/platforms/runpod.py`, `platforms/beam.py`, `platforms/manifests.py`,
  `platforms/failures.py` (and collapse `platforms/__init__.py` / `platforms/local.py` to whatever
  the local pipeline still needs, or remove the package if nothing remains).
- Edit `app/api/videos.py`: `_build_runner` collapses to the local `PipelineRunner` only;
  drop the `beam`/`runpod` branches and `pipeline_backend` switching.
- Edit `app/api/config.py`: drop `runpod_*`, `beam_*`, `r2_*` keys.
- Edit `app/services/result_importer.py`: drop RunPod/remote manifest-import assumptions.
- Edit `src/card_capture/pipeline/runner.py`, `pipeline/request.py`: remove remote transport.
- Delete: `docs/runpod-deployment.md`, `docs/vastai-template-setup.md`.
- Delete tests: `tests/platforms/test_runpod_runner.py`, `tests/platforms/test_beam_runner.py`,
  `tests/platforms/test_failures.py`, `tests/app/test_runpod_runner.py`,
  `tests/app/test_runpod_gpu_preflight.py`, `tests/app/test_beam_runner.py`,
  `tests/app/test_worker_client.py`, `tests/app/test_worker_core.py`.
- Edit `pyproject.toml`: drop `runpod` and `beam` optional-dependency groups.
- Edit `.importlinter` if it names removed modules.

### CUDA (deepest commit — test-gated, back off if MPS destabilizes)
- Delete: `src/card_capture/sampler/cuda_sampler.py`.
- Edit `src/card_capture/detectors.py`: remove CUDA/TensorRT backends; keep CoreML/MPS + CPU.
- Edit `src/card_capture/runtime/strict_gpu.py`, `ml/gpu_ops.py`, `gpu_refinement.py`,
  `gpu_utils.py`, `config.py`, `cli.py`: remove CUDA branches; device selection = MPS or CPU.
- Delete build/scripts: `Dockerfile.cuda`, `docker/`,
  `scripts/runpod_pod_debug.sh`, `scripts/runpod_setup.py`, `scripts/setup_cuda_template.py`,
  `scripts/fetch_runpod_perf.py`, `scripts/verify_gpu_image.sh`, `scripts/verify_gpu_native.sh`,
  `scripts/test_docker_local.sh`, `scripts/check_torch_gpu.py`.
- Delete tests: `tests/sampler/test_cuda_sampler.py`, `tests/test_cuda_sampler.py`,
  `tests/test_cuda_sampler_gpu.py`.
- Edit `pyproject.toml`: remove the `cuda` and `provider` pytest markers; drop CUDA-only deps.
  Evaluate the `pipeline_v21` extra (onnxruntime/boxmot/av) and drop if unused by the MPS path.

### Metaflow
- Delete the Metaflow flow under `pipeline/` (keep only `pipeline/contracts.py` / `__init__.py`
  if the local path still imports them; otherwise remove).
- Delete `docs/contracts/metaflow-artifacts.md`.
- Delete `.metaflow/`.
- Edit `src/card_capture/cli.py`: remove the `--pipeline metaflow` path; default and only path is local.

## Docs Audit

- New `docs/archive/` holding authoritative prior-version architecture docs, moved (not copied):
  `docs/architecture/arch-4.1.md`, `docs/architecture/roadmap.md`,
  `docs/architecture/decisions/0004-v4-modular-pipeline.md`,
  `docs/superpowers/pipeline-v3-overview.md`, `docs/V4_CONCERNS.md`, `docs/V4_CONCERNS_PASS2.md`.
  (Final archive membership is confirmed during implementation by reading each doc's scope.)
- Delete cloud-only docs (listed under Cloud removal).
- Leave dated `plans/` and `specs/` in place.
- Refresh `docs/contracts/storage-schema.md` and peers only if v5.5 drifted them
  (the schema-docs validator must still pass).

## New Architecture Document: `docs/architecture/arch-5.5.md`

Modeled on `arch-4.1.md`'s depth. Describes the **merged code as it actually exists**, not
CLAUDE.md's aspirational description. Notably, CLAUDE.md references `src/card_capture/runtime.py`
and `dal.py`, but the merged tree uses the `pipeline/` package with `runtime_local.py`
(LocalPipelineRuntime) and a `data/` repositories layer. The doc reflects the real modules,
read at authoring time. Coverage: stage sequence (sampler → detect → warp/crop → presence
gate → tracking → refine/score → resolve → fuse → dedup/store), threading/GPU boundary,
the data-access layer, the local-only platform path, the MPS device path, and configuration.

## Operator-Facing Updates

Drop all cloud/CUDA/Metaflow commands and reflect the MPS-only local workflow in:
`OPERATOR.md`, `README.md`, `QUICK_REFERENCE.md`, `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`,
`.github/workflows/ci.yml` (remove schema/cloud steps that no longer apply; keep the test lane),
and `Makefile`. CLAUDE.md's "Architecture" and "Key Modules" sections are corrected to match
the real tree and to point at `arch-5.5.md`.

## Testing & Verification

- After each removal commit: `python3 -m pytest tests/ -m "not quarantine" -q` green,
  and `lint-imports` green.
- After CUDA removal specifically: confirm a representative local MPS run still imports and
  initializes (device selection resolves to MPS/CPU with no CUDA references).
- Schema-docs validator (`scripts/validate_schema_docs.py`) passes if it survives the cleanup.

## Risks

- **CUDA removal is the deep one** — it threads through `detectors.py`, the sampler, and GPU
  ops. Isolated commit, test-gated, back off if it destabilizes the MPS path.
- **Runner collapse** — `app/api/videos.py` and `result_importer.py` assume a runner abstraction;
  collapsing to local must not break the web UI's process flow.
- **Hidden imports** — removed modules may be imported transitively; follow each cascade and
  keep the import linter green.
