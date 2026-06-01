# Repository Reorganization — Design Spec

**Date:** 2026-06-01
**Status:** Approved design, pending implementation plan
**Scope:** Whole repository — the `src/card_capture/` package internals **and** repo-root layout.

---

## 1. Motivation

The codebase has accreted structure across many architecture revisions (v3 → v5.5). The
result reads as incoherent in two distinct ways:

**Inside the package (`src/card_capture/`):**

- **24 modules sit directly at the package root** — more than any single subpackage —
  mixing domain algorithms (`scoring`, `cropper`, `fuser`, `selector`, `frame_quality`,
  `occlusion_residual`, `ecc_registration`, `gpu_refinement`, `detectors`,
  `deduplicator`), infrastructure (`gpu_utils`, `workers`, `pipeline_utils`, `storage`),
  cross-cutting types (`config`, `models`, `interfaces`), the CLI, ingestion, and a
  legacy web UI (`review`).
- **Training logic lives in 3+ places**: `train/`, `training/`, `ml/training/`, plus
  `ml/train_fb.py` and `presence/training_data.py`.
- **Fusion is split**: `fusion/` package + root `fuser.py` + `ecc_registration.py`.
- **GPU code is scattered**: `gpu_utils.py`, `gpu_refinement.py`, `ml/gpu_ops.py`,
  `runtime/{gpu_session,strict_gpu}`.
- **Runtime is split**: a `runtime/` package vs. `pipeline/runtime_local.py`.
- Pipeline **stages are thin orchestrators** in `pipeline/stages/` that reach back into
  the root-level algorithm modules — orchestration and implementation are separated
  awkwardly across the package.
- Several **2-file micro-packages** (`identity`, `calibration`, `metrics`, `analysis`,
  `platforms`).

**At the repo root:**

- Scattered scratch/output dirs (`out/`, `card_capture_output/`, `card_capture_uploads/`,
  `sample_run_2026_05_03_*`, `reports/`, `data/`) and **5+ stray SQLite databases**.
- Five overlapping agent/operator docs (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`,
  `OPERATOR.md`, `QUICK_REFERENCE.md`).
- `harness/` as a top-level package, `migrations/` at root, a large
  `docs/superpowers/plans/` pile.

There is already a **layered import-linter contract** (`runtime → pipeline → data`); any
reorganization must respect and strengthen it.

## 2. Goals & non-goals

**Goals**

- Organize the package by **pipeline stage (vertical slices)** that map 1:1 to the
  documented 10-stage architecture, so each stage owns its `run()` orchestrator and its
  algorithm code.
- Eliminate the root-module sprawl and the duplicate training / fusion / GPU / runtime
  groupings.
- Produce a repo root where every top-level entry is *code you edit*, *an asset you
  keep*, or *one ignored scratch dir*.
- Strengthen the automated structural contract (import-linter layers) so the new shape
  is enforced, not just documented.

**Non-goals (deliberately out of scope to keep the cut focused)**

- No behavior changes. This is a pure move + import-rewrite.
- No dead-code removal beyond what is already on the current branch.
- No internal restructuring of `app/` (web) or `harness/` (eval) — only their imports
  into `card_capture` are rewritten.
- No splitting of large files (e.g. `sampler/__init__.py` at 1048 lines). Noted as a
  **follow-up**, not part of this move.
- No removal of the legacy `review/` UI. It is grouped, not deleted.

## 3. Target package structure (`src/card_capture/`)

```
src/card_capture/
  __init__.py
  _warnings.py                  # import-time warning filters (stays at root)
  cli.py                        # console entrypoint (stays at root)

  core/                         # leaf layer — no internal deps; imported by all
    models.py                   # FrameSample, TrackState, QualityScore, Point…
    config.py                   # PipelineConfig
    interfaces.py               # Protocols (CardDetector, FrameSampler)
    device.py                   # pure device probing (split out of gpu_utils)

  stages/                       # ◀ VERTICAL SLICES — one package per arch stage
    __init__.py                 #   registry: exposes each stage's run() callable
    sample/                     # sampler/*, ingestion, adaptive_gap   + run()
    detect/                     # detectors                            + run()
    novelty/                    # presence/* (background novelty)      + run()
    track/                      # tracking/* (botsort, bytetrack, …)   + run()
    refine/                     # cropper, gpu_refinement, occlusion   + run()
    score/                      # scoring, selector, frame_quality,
    │                           #   calibration/per_video_adaptive     + run()
    resolve/                    # front/back side resolution           + run()
    fuse/                       # fuser, fusion/foil, ecc_registration + run()
    dedup/                      # deduplicator, identity/embedding_dist+ run()
    store/                      # storage                              + run()

  shared/                       # cross-stage helpers (used by ≥2 slices)
    pipeline_utils.py
    stage_metrics.py            # moved out of pipeline/ (see §5)

  ml/                           # model zoo + inference (shared assets)
    models/  inference/  embeddings.py  registry.py  scaffolding.py
    errors.py  synthetic_eval.py  gpu_ops.py

  training/                     # ◀ ALL offline training, consolidated
    presence.py                 # ← train/presence.py + presence/training_data.py
    fb.py                       # ← training/fb_trainer.py + ml/train_fb.py
    presence_trainer.py         # ← training/presence_trainer.py
    dedup_calibrate.py          # ← ml/training/dedup_calibrate.py
    hard_case_capture.py        # ← analysis/hard_case_capture.py

  runtime/                      # GPU session + execution infra (top layer)
    gpu_session.py  strict_gpu.py  batches.py  guards.py  cpu_debug.py
    gpu_utils.py                # GPU session/tensor ops (device.py split to core)
    workers.py                  # ← multiprocessing producer/consumer (from root)

  pipeline/                     # orchestration (keeps the name for the layer)
    request.py  runner.py  telemetry.py
    runtime_local.py  runtime_worker.py

  data/                         # DAL — unchanged location
    connection.py  writer.py  sql_queries.py  repositories/*

  review/                       # ◀ legacy Jinja review UI, grouped
    app.py                      # ← review.py
    timeline_data.py
    templates/                  # labeling/review/setup/timeline .html

  platforms/  local.py          # platform abstraction seam (kept)
  metrics/    robustness_pack.py # eval metrics (kept; candidate to move to harness/)
```

### 3.1 Module → home mapping (the full move map)

| From | To |
|---|---|
| `detectors.py` | `stages/detect/detectors.py` |
| `sampler/*`, `ingestion.py`, `adaptive_gap.py` | `stages/sample/` |
| `presence/{background_novelty,…}.py` (not `training_data`) | `stages/novelty/` |
| `tracking/*` | `stages/track/` |
| `cropper.py`, `gpu_refinement.py`, `occlusion_residual.py` | `stages/refine/` |
| `scoring.py`, `selector.py`, `frame_quality.py`, `calibration/per_video_adaptive.py` | `stages/score/` |
| (front/back resolution — thin run() invoking `ml.inference.fb_predict`) | `stages/resolve/` |
| `fuser.py`, `fusion/foil_detection.py`, `ecc_registration.py` | `stages/fuse/` |
| `deduplicator.py`, `identity/embedding_distance.py` | `stages/dedup/` |
| `storage.py` | `stages/store/` |
| `pipeline/stages/<stage>.py` (the `run()` fns) | `stages/<stage>/` — `run` exposed from the slice's `__init__.py`; algorithm modules are siblings |
| `models.py`, `config.py`, `interfaces.py` | `core/` |
| device-probing helpers split from `gpu_utils.py` | `core/device.py` |
| `pipeline_utils.py` | `shared/pipeline_utils.py` |
| `pipeline/stage_metrics.py` | `shared/stage_metrics.py` |
| `workers.py`, remainder of `gpu_utils.py` | `runtime/` |
| `train/presence.py`, `presence/training_data.py` | `training/presence.py` |
| `training/fb_trainer.py`, `ml/train_fb.py` | `training/fb.py` |
| `training/presence_trainer.py` | `training/presence_trainer.py` |
| `ml/training/dedup_calibrate.py` | `training/dedup_calibrate.py` |
| `analysis/hard_case_capture.py` | `training/hard_case_capture.py` |
| `review.py` | `review/app.py` |
| `timeline_data.py`, `templates/*.html` | `review/` |
| `pipeline/{request,runner,telemetry,runtime_local,runtime_worker}.py` | `pipeline/` (unchanged names) |
| `data/*` | `data/` (unchanged) |
| `ml/{models,inference,embeddings,registry,scaffolding,errors,synthetic_eval,gpu_ops}` | `ml/` (unchanged, minus the training files above) |
| `platforms/local.py`, `metrics/robustness_pack.py`, `_warnings.py`, `cli.py`, `__init__.py` | unchanged |

### 3.2 The ML boundary principle

Heavyweight model loading / neural inference stays in **`ml/`** as a shared model zoo
(`DinoEmbedder`, `FBPredictor`, corner refiner, `ml/inference/*`). The **stage slices stay
thin**: pipeline orchestration plus lightweight numpy/cv2 algorithmic code, calling into
`ml/` for neural inference. This avoids duplicating heavy model code across
`detect`/`resolve`/`dedup` and keeps torch imports from leaking into every slice. As a
consequence, `stages/resolve/` is essentially a `run()` that invokes
`ml.inference.fb_predict` — which is an accurate reflection of what that stage does today.

## 4. Repo-root cleanup

### 4.1 Data & databases — one gitignored `var/`

```
var/                    # gitignored; never tracked
  output/               # ← out/, card_capture_output/, sample_run_2026_05_03_*/
  uploads/              # ← card_capture_uploads/
  db/                   # ← data/*.sqlite, data/pipeline.db, root cards.sqlite, *-wal/-shm
  reports/              # ← reports/ (benchmark JSON/MD)
```

- **Preserved as-is, tracked:** `models/` (runtime weights) and `golden_set/`
  (eval fixtures).
- Nothing is deleted. Tracked scratch dirs are `git mv`'d into `var/`; already-untracked
  dirs are moved on disk and covered by a single `var/` gitignore entry. Tracked-vs-
  untracked status is confirmed per dir before touching; **no `rm` without confirmation.**
- Code defaults that currently point at `card_capture_output/cards.sqlite` (in `cli.py`,
  `review.py`) are repointed to `var/...`. The scattered `.gitignore` entries are replaced
  by a single `var/`.

### 4.2 Docs — one home, clear roles

```
README.md               # stays at root (project front door)
CLAUDE.md               # stays at root (canonical agent context)
AGENTS.md, GEMINI.md    # stay at root, become thin pointers to CLAUDE.md
docs/
  OPERATOR.md           # ← moved from root
  QUICK_REFERENCE.md    # ← moved from root
  architecture/arch-5.5.md   # updated to reflect the new module map
```

### 4.3 Tooling & eval — left structurally sound

- `harness/` stays a **top-level** package (dev/eval-only, not shipped runtime). Only its
  `card_capture.*` imports are rewritten. Moving it under `src/` would wrongly make it an
  installed artifact.
- `scripts/` stays the home for diagnostic/calibration/generation scripts
  (incl. `scripts/deadcode/`).
- `migrations/` stays at root (deploy concern; import-linter already special-cases it for
  sqlite access).

## 5. Layering & import-linter changes

Two non-mechanical adjustments make the new dependency graph acyclic:

1. **`stage_metrics` moves `pipeline/` → `shared/`.** Every stage imports
   `stage_metrics`; in the new layout `pipeline` (orchestration) sits *above* `stages` and
   imports them, so a stage importing `pipeline` would be a backward edge. Moving the one
   shared helper down into `shared/` makes the graph clean.

2. **Split GPU primitives from GPU session.** `gpu_utils.py` mixes pure device probing
   (`get_device`, leaf-safe for any stage) with session/tensor machinery. Pure probing →
   `core/device.py`; session orchestration stays in `runtime/`. This keeps stages from
   importing the `runtime` layer (which would violate both the layering and the strict-GPU
   boundary).

**New `.importlinter` layered contract** (top imports lower; replaces the 3-layer one):

```
runtime  →  pipeline  →  stages  →  shared  →  ml  →  data  →  core
```

The existing `forbidden` contracts carry over unchanged:
- sqlite3 only inside `card_capture.data` and `migrations`
- `runtime.strict_gpu` must not import `PIL`/`cv2`
- no provider SDKs (`runpod`, `beam`), no `metaflow` anywhere

The layered contract is the single automated proof that the reorg is structurally sound;
it runs in the verification gate (§7).

## 6. Migration mechanics — LibCST-driven big-bang

A single hard-cut, made safe by treating the **§3.1 move map as the executable source of
truth** and gating on the full suite + import-linter.

**Tooling**

- **LibCST `RenameCommand`** (primary) — deterministic, formatting-preserving CST rename
  of fully-qualified names across the whole codebase, with automatic import fix-up. One
  invocation per move-map row.
- **rope** (fallback) — semantic spot-check for any symbol-level relocation
  (e.g. `stage_metrics`) that name-based rename doesn't catch cleanly.
- **`git mv`** does the physical file moves (preserves history).
- **isort / ruff** post-pass to normalize import ordering.
- Add `libcst` (and `rope`) to the `dev` extra in `pyproject.toml`; add a
  `.libcst.codemod.yaml` (via `python -m libcst.tool initialize .`).

**Execution order (one branch, landed once green):**

1. **Skeleton & moves** — create new package dirs + `__init__.py`s; `git mv` every module
   to its destination, including the `stage_metrics → shared` and
   `gpu_utils → core/device.py` splits.
2. **Materialize the import map** — the deterministic old→new dotted-path table from §3.1.
3. **Rewrite references** — drive LibCST `RenameCommand` from the map (one rename per row)
   across `src/ app/ tests/ harness/ scripts/`, plus the `stages/__init__.py` registry that
   `runtime_local` imports. Grep for *string* module paths / `getattr` references too, not
   just `import` statements. rope as fallback for stragglers.
4. **Contracts & config** — `.importlinter` (new 7-layer order); `pyproject.toml`
   (`packages` / `package-data` for the moved `review/templates/*.html`; entry point
   unchanged); `.gitignore` (`var/`).
5. **Docs** — `CLAUDE.md` Key Modules table + `docs/architecture/arch-5.5.md` module map.
6. **Normalize & verify** — isort/ruff, then the §7 gate.

Example codemod invocation (exact flag spelling confirmed against the installed LibCST
version during implementation):

```bash
python -m libcst.tool initialize .
python -m libcst.tool codemod rename.RenameCommand \
  --old-name=card_capture.detectors \
  --new-name=card_capture.stages.detect.detectors \
  src/ app/ tests/ harness/ scripts/
```

## 7. Verification gate (all must pass before the commit lands)

- `python -m pytest tests/ -m "not quarantine" -q`
- `lint-imports` (import-linter) — proves the layering is acyclic and correct
- `card-capture --help` and `python -c "import card_capture.<subpkg>"` smoke imports of
  each new top-level subpackage
- `card-capture review` import path resolves (Jinja `templates/` found via
  `Path(__file__).parent`, packaged via updated `package-data`)

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Large red window (big-bang) | Deterministic git mv + LibCST rewrite from the fixed map; nothing hand-edited ad hoc |
| `multiprocessing` (`workers.py`) re-imports by dotted path in child procs | Complete rewrite → spawned children import new paths; covered by sampler tests |
| Jinja `templates/` not packaged after move | `pyproject` `package-data` updated; smoke-test `card-capture review` |
| Dynamic / string imports or `getattr` references | Grep for string module paths during step 3, not just `import` statements |
| `app/` (web) imports into `card_capture` | In scope for the rewrite (step 3) |
| import-linter contract itself breaks | It is the gate, not an afterthought; the §5 layering is designed acyclic |
| Test files mirror old module paths | `tests/` included in the rewrite |
| LibCST rename misses a symbol-level move | rope fallback + import-linter + pytest catch it |

## 9. Follow-ups (explicitly deferred)

- Split oversized modules, starting with `sampler/__init__.py` (1048 lines) and
  `workers.py` (724 lines).
- Decide whether the legacy `review/` Jinja UI should be retired in favor of `app/`.
- Consider relocating `metrics/robustness_pack.py` into `harness/` (eval-only).
