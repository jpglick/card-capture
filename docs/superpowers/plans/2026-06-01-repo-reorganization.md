# Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `card_capture` into stage-oriented vertical slices and tidy the repo root, with zero behavior change, using deterministic LibCST codemods.

**Architecture:** A single branch ("big-bang" landing — no compatibility shims), but sequenced so that each task keeps the full test suite + import-linter **green**. That is possible because every move is `git mv` (history-preserving) followed by a LibCST `RenameCommand` that atomically rewrites *all* references in the same task. The existing test suite is the spec: a no-behavior-change refactor is correct iff the suite and the layered import contract stay green.

**Tech Stack:** Python 3.9+, LibCST (codemods), rope (fallback), import-linter, pytest, setuptools.

---

## Conventions (read before starting)

**Design spec:** `docs/superpowers/specs/2026-06-01-repo-reorganization-design.md` — the authoritative target structure and move map. This plan implements it.

**The codemod command.** Each module move is a fully-qualified rename. The invocation pattern (confirm exact flags once in Task 0):

```bash
python -m libcst.tool codemod rename.RenameCommand \
  --old-name=<old.dotted.path> --new-name=<new.dotted.path> \
  src app tests harness scripts
```

`RenameCommand` rewrites `import x`, `from pkg import x`, `from pkg.x import y`, and `x.attr` usages, and fixes/adds/removes the relevant imports. It resolves relative imports (`from .x import y`) via qualified names, so internal package imports are covered too, and it emits **absolute** imports for the new locations.

**Move ordering (why this is safe).** Tasks are ordered leaf-first (`core` → `shared` → stages → orchestration). Two properties keep every task green:

1. Because `RenameCommand` emits *absolute* imports, once a low module (e.g. `core.models`) is moved, every reference to it elsewhere becomes `from card_capture.core.models import …`. Those references are unaffected when their *containing* file is moved later.
2. A module that has a **relative** import to a sibling which is *not* moving in the same task (and isn't being renamed) would break when relocated. Avoid this by moving tightly-coupled leaf clusters together (e.g. Task 1 moves `config`+`workers` together because `config` does `from .workers import …`). If the SVB ever fails with such a case, the straggler sweep's relative-import grep finds it; fix by converting that one import to absolute.

**Standard Verification Block (SVB).** Every move task ends by running these two commands and committing only if both pass:

```bash
python -m pytest tests/ -m "not quarantine" -q          # expect: same pass count as the Task 0 baseline, 0 new failures
lint-imports                                            # expect: "Contracts: N kept, 0 broken."
```

**Straggler sweep (run if the SVB fails).** A failed import usually means a relative import the rename missed, or a string/`getattr` reference. Diagnose with:

```bash
# replace OLDPATH with the dotted path you just moved, e.g. card_capture.detectors
grep -rn "OLDPATH" src app tests harness scripts --include='*.py' | grep -v __pycache__
grep -rn "from \.\+OLDLEAF import\|import OLDLEAF" src --include='*.py'   # relative-import stragglers
```

Fix stragglers by re-running the codemod scoped to the offending dir, or `rope`'s move, or a manual edit. Never leave the SVB red at a commit.

**Package `__init__.py` rule.** When a task creates a new package directory, it must contain an `__init__.py`. New slice packages get an empty placeholder `__init__.py` (just a one-line docstring); the stage `run()` is moved into it later in Task 16.

**Commit message convention.** `refactor(reorg): <area> → <destination>`. End every commit body with the Co-Authored-By trailer.

---

## Task 0: Branch, tooling, and baseline

**Files:**
- Modify: `pyproject.toml` (dev extra)
- Create: `.libcst.codemod.yaml`

- [ ] **Step 1: Create the working branch**

```bash
cd /Users/josh/code/card-capture
git checkout -b refactor/repo-reorganization
```

- [ ] **Step 2: Install the refactor tooling into the dev extra**

Edit `pyproject.toml`, in `[project.optional-dependencies]` change the `dev` list to add `libcst` and `rope`:

```toml
dev = [
  "import-linter>=2.0",
  "pytest>=7.0",
  "vulture>=2.11",
  "libcst>=1.1",
  "rope>=1.11",
]
```

- [ ] **Step 3: Install and initialize LibCST**

```bash
python -m pip install -e '.[dev]'
python -m libcst.tool initialize .
```

Expected: creates `.libcst.codemod.yaml`. Confirm the exact rename flags for this installed version:

```bash
python -m libcst.tool codemod rename.RenameCommand --help
```

Expected: shows `--old-name` and `--new-name`. If the flag spelling differs, use what `--help` reports for every codemod step below.

- [ ] **Step 4: Capture the green baseline**

```bash
python -m pytest tests/ -m "not quarantine" -q | tail -3
lint-imports
```

Record the passed/failed/skipped counts and the "Contracts: N kept" number. **This is the invariant** every later task's SVB must match. If the baseline is not green, stop and report — do not start moving code on a red baseline.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .libcst.codemod.yaml
git commit -m "build(reorg): add libcst + rope to dev extra; init codemod config"
```

---

## Task 1: `core/` — domain types + leaf utilities

Move the whole leaf cluster together: `models`, `config`, `interfaces`, plus the two pure
leaf utilities `workers` and `gpu_utils` (no internal `card_capture` imports; imported
widely from above). They must move in **one** task because `config.py` imports `workers`
(`from .workers import ProcessingOptions`) and `interfaces.py` imports `models` — moving
them together keeps those relative imports valid (both ends land in `core/`).

**Files:**
- Create: `src/card_capture/core/__init__.py`
- Move: `models.py`, `config.py`, `interfaces.py`, `workers.py`, `gpu_utils.py` → `src/card_capture/core/`

- [ ] **Step 1: Create the package and move the modules**

```bash
mkdir -p src/card_capture/core
printf '"""Foundation layer: domain types (models, config, interfaces) and leaf utilities (gpu_utils, workers)."""\n' > src/card_capture/core/__init__.py
git add src/card_capture/core/__init__.py
git mv src/card_capture/models.py     src/card_capture/core/models.py
git mv src/card_capture/config.py     src/card_capture/core/config.py
git mv src/card_capture/interfaces.py src/card_capture/core/interfaces.py
git mv src/card_capture/workers.py    src/card_capture/core/workers.py
git mv src/card_capture/gpu_utils.py  src/card_capture/core/gpu_utils.py
```

- [ ] **Step 2: Rewrite all references**

```bash
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.models     --new-name=card_capture.core.models     src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.config      --new-name=card_capture.core.config      src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.interfaces  --new-name=card_capture.core.interfaces  src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.workers     --new-name=card_capture.core.workers     src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.gpu_utils   --new-name=card_capture.core.gpu_utils   src app tests harness scripts
```

- [ ] **Step 3: Fix the string annotation in `config.py` (if the codemod missed it)**

`config.py` has a string forward-reference annotation `"card_capture.workers.ProcessingOptions"`. Confirm the rename updated it; if not, fix manually:

```bash
grep -n "card_capture.workers" src/card_capture/core/config.py
# if found, change the annotation string to "card_capture.core.workers.ProcessingOptions"
```

- [ ] **Step 4: SVB**

Run the Standard Verification Block. Expect baseline pass count + contracts kept. If red, run the Straggler sweep for `card_capture.models` / `config` / `interfaces` / `workers` / `gpu_utils` (include the relative-import grep — many modules use `from .gpu_utils import`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(reorg): models/config/interfaces/workers/gpu_utils → card_capture.core"
```

---

## Task 2: `shared/` — cross-stage helpers

**Files:**
- Create: `src/card_capture/shared/__init__.py`
- Move: `pipeline_utils.py` → `shared/`, `pipeline/stage_metrics.py` → `shared/`

- [ ] **Step 1: Create the package and move the modules**

```bash
mkdir -p src/card_capture/shared
printf '"""Helpers shared across two or more pipeline stages."""\n' > src/card_capture/shared/__init__.py
git add src/card_capture/shared/__init__.py
git mv src/card_capture/pipeline_utils.py        src/card_capture/shared/pipeline_utils.py
git mv src/card_capture/pipeline/stage_metrics.py src/card_capture/shared/stage_metrics.py
```

- [ ] **Step 2: Rewrite all references**

```bash
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.pipeline_utils        --new-name=card_capture.shared.pipeline_utils  src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.pipeline.stage_metrics --new-name=card_capture.shared.stage_metrics    src app tests harness scripts
```

- [ ] **Step 3: SVB** — expect green. Straggler sweep on `pipeline_utils` / `stage_metrics` if needed.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): pipeline_utils + stage_metrics → card_capture.shared"
```

---

## Task 3: `runtime/` — no moves required

Intentionally empty. `workers.py` and `gpu_utils.py` were originally slated to move into
`runtime/`, but they are leaf utilities with no internal `card_capture` imports and are
imported widely from below the top layer (`config` → `workers`;
`scoring`/`foil_detection`/`sampler`/trainers → `gpu_utils`). Putting them in `runtime` (the
**top** layer) would make every such import a lower→higher violation under the new layered
contract. They therefore moved to `core/` in **Task 1** instead.

`runtime/` keeps only the GPU **session** orchestration it already contains
(`gpu_session`, `strict_gpu`, `batches`, `guards`, `cpu_debug`), which legitimately depends
on `pipeline`. No file moves, no rename, no commit for this task. Proceed to Task 4.

(Task number retained so downstream task references stay stable.)

---

## Tasks 4–13: Stage slices (algorithm modules only)

> Each task creates `stages/<slice>/` with a placeholder `__init__.py`, moves that stage's
> algorithm modules in, and renames references. The stage `run()` stays at
> `pipeline/stages/<slice>.py` for now (its imports get rewritten to the new paths) and is
> relocated in Task 16. Task 4 additionally creates the top-level `stages/__init__.py`.

### Task 4: `stages/detect/`

- [ ] **Step 1: Create `stages/` and the detect slice; move `detectors.py`**

```bash
mkdir -p src/card_capture/stages/detect
printf '"""In-process pipeline stages, organized as vertical slices.\n\nEach slice package exposes a single `run(state, *, telemetry)` callable from its\n__init__ and owns its algorithm modules. Model loading/decode lifecycle belong to\nthe runtime, not the slices.\n"""\n' > src/card_capture/stages/__init__.py
printf '"""Stage 2: YOLO corner detection."""\n' > src/card_capture/stages/detect/__init__.py
git add src/card_capture/stages/__init__.py src/card_capture/stages/detect/__init__.py
git mv src/card_capture/detectors.py src/card_capture/stages/detect/detectors.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.detectors --new-name=card_capture.stages.detect.detectors src app tests harness scripts
```

- [ ] **Step 2: SVB** — straggler sweep on `card_capture.detectors` if red.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(reorg): detectors → card_capture.stages.detect"
```

### Task 5: `stages/sample/`

`sampler/` is a package (keep it as a subpackage); `ingestion.py` and `adaptive_gap.py` are modules.

- [ ] **Step 1: Create the slice and move modules**

```bash
mkdir -p src/card_capture/stages/sample
printf '"""Stage 1: adaptive presence sampling + streaming producer."""\n' > src/card_capture/stages/sample/__init__.py
git add src/card_capture/stages/sample/__init__.py
git mv src/card_capture/sampler       src/card_capture/stages/sample/sampler
git mv src/card_capture/ingestion.py  src/card_capture/stages/sample/ingestion.py
git mv src/card_capture/adaptive_gap.py src/card_capture/stages/sample/adaptive_gap.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.sampler      --new-name=card_capture.stages.sample.sampler      src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.ingestion    --new-name=card_capture.stages.sample.ingestion    src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.adaptive_gap --new-name=card_capture.stages.sample.adaptive_gap src app tests harness scripts
```

- [ ] **Step 2: SVB** — `sampler` is the highest-fan-in package; sweep `card_capture.sampler` (incl. `.frame_producer`, `.valley_splits`, `.valley_detection_per_region`) if red.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(reorg): sampler/ingestion/adaptive_gap → card_capture.stages.sample"
```

### Task 6: `stages/novelty/` (and route `training_data` to training)

`presence/` holds `background_novelty.py`, `classifier.py` (→ novelty) and `training_data.py` (→ training, Task 14). Move the novelty modules now; leave `training_data.py` for Task 14, which deletes the empty `presence/`.

- [ ] **Step 1: Create the slice and move the novelty modules**

```bash
mkdir -p src/card_capture/stages/novelty
printf '"""Stage 3: background novelty gate."""\n' > src/card_capture/stages/novelty/__init__.py
git add src/card_capture/stages/novelty/__init__.py
git mv src/card_capture/presence/background_novelty.py src/card_capture/stages/novelty/background_novelty.py
git mv src/card_capture/presence/classifier.py         src/card_capture/stages/novelty/classifier.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.presence.background_novelty --new-name=card_capture.stages.novelty.background_novelty src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.presence.classifier         --new-name=card_capture.stages.novelty.classifier         src app tests harness scripts
```

- [ ] **Step 2: SVB** — sweep `card_capture.presence.background_novelty` / `.classifier` if red.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(reorg): presence novelty modules → card_capture.stages.novelty"
```

### Task 7: `stages/track/`

Flatten `tracking/` modules into the slice.

- [ ] **Step 1: Create the slice and move modules**

```bash
mkdir -p src/card_capture/stages/track
printf '"""Stage 4: session-aware tracking (BoT-SORT / ByteTrack)."""\n' > src/card_capture/stages/track/__init__.py
git add src/card_capture/stages/track/__init__.py
git mv src/card_capture/tracking/botsort_adapter.py        src/card_capture/stages/track/botsort_adapter.py
git mv src/card_capture/tracking/bytetrack_adapter.py      src/card_capture/stages/track/bytetrack_adapter.py
git mv src/card_capture/tracking/appearance_sessionizer.py src/card_capture/stages/track/appearance_sessionizer.py
git mv src/card_capture/tracking/centroid_jump.py          src/card_capture/stages/track/centroid_jump.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.tracking.botsort_adapter        --new-name=card_capture.stages.track.botsort_adapter        src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.tracking.bytetrack_adapter      --new-name=card_capture.stages.track.bytetrack_adapter      src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.tracking.appearance_sessionizer --new-name=card_capture.stages.track.appearance_sessionizer src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.tracking.centroid_jump          --new-name=card_capture.stages.track.centroid_jump          src app tests harness scripts
```

- [ ] **Step 2: Remove the now-empty `tracking/` package**

```bash
git rm src/card_capture/tracking/__init__.py
rmdir src/card_capture/tracking 2>/dev/null || true
```

- [ ] **Step 3: SVB** — sweep `card_capture.tracking` if red.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): tracking/* → card_capture.stages.track"
```

### Task 8: `stages/refine/`

- [ ] **Step 1: Create the slice and move modules**

```bash
mkdir -p src/card_capture/stages/refine
printf '"""Stage 5: GPU refinement (Kornia warp to 750x1050)."""\n' > src/card_capture/stages/refine/__init__.py
git add src/card_capture/stages/refine/__init__.py
git mv src/card_capture/cropper.py            src/card_capture/stages/refine/cropper.py
git mv src/card_capture/gpu_refinement.py     src/card_capture/stages/refine/gpu_refinement.py
git mv src/card_capture/occlusion_residual.py src/card_capture/stages/refine/occlusion_residual.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.cropper            --new-name=card_capture.stages.refine.cropper            src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.gpu_refinement     --new-name=card_capture.stages.refine.gpu_refinement     src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.occlusion_residual --new-name=card_capture.stages.refine.occlusion_residual src app tests harness scripts
```

- [ ] **Step 2: SVB** — sweep the three names if red.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(reorg): cropper/gpu_refinement/occlusion → card_capture.stages.refine"
```

### Task 9: `stages/score/`

Includes `calibration/per_video_adaptive.py` (flattened).

- [ ] **Step 1: Create the slice and move modules**

```bash
mkdir -p src/card_capture/stages/score
printf '"""Stage 6: quality scoring + adaptive pruning."""\n' > src/card_capture/stages/score/__init__.py
git add src/card_capture/stages/score/__init__.py
git mv src/card_capture/scoring.py       src/card_capture/stages/score/scoring.py
git mv src/card_capture/selector.py      src/card_capture/stages/score/selector.py
git mv src/card_capture/frame_quality.py src/card_capture/stages/score/frame_quality.py
git mv src/card_capture/calibration/per_video_adaptive.py src/card_capture/stages/score/per_video_adaptive.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.scoring       --new-name=card_capture.stages.score.scoring       src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.selector      --new-name=card_capture.stages.score.selector      src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.frame_quality --new-name=card_capture.stages.score.frame_quality src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.calibration.per_video_adaptive --new-name=card_capture.stages.score.per_video_adaptive src app tests harness scripts
```

- [ ] **Step 2: Remove the now-empty `calibration/` package**

```bash
git rm src/card_capture/calibration/__init__.py
rmdir src/card_capture/calibration 2>/dev/null || true
```

- [ ] **Step 3: SVB** — sweep the four names if red.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): scoring/selector/frame_quality/calibration → card_capture.stages.score"
```

### Task 10: `stages/resolve/`

No algorithm modules to move — front/back resolution invokes `ml.inference.fb_predict`. The slice dir is created now; its `run()` arrives in Task 16.

- [ ] **Step 1: Create the slice placeholder**

```bash
mkdir -p src/card_capture/stages/resolve
printf '"""Stage 7: front/back side resolution (invokes ml.inference.fb_predict)."""\n' > src/card_capture/stages/resolve/__init__.py
git add src/card_capture/stages/resolve/__init__.py
```

- [ ] **Step 2: SVB** (no code moved; just confirms still green) and commit

```bash
python -m pytest tests/ -m "not quarantine" -q | tail -3
lint-imports
git commit -m "refactor(reorg): create card_capture.stages.resolve slice"
```

### Task 11: `stages/fuse/`

Flatten `fusion/` modules + `fuser.py` + `ecc_registration.py`.

- [ ] **Step 1: Create the slice and move modules**

```bash
mkdir -p src/card_capture/stages/fuse
printf '"""Stage 8: lighting-diverse median fusion."""\n' > src/card_capture/stages/fuse/__init__.py
git add src/card_capture/stages/fuse/__init__.py
git mv src/card_capture/fuser.py            src/card_capture/stages/fuse/fuser.py
git mv src/card_capture/ecc_registration.py src/card_capture/stages/fuse/ecc_registration.py
git mv src/card_capture/fusion/foil_detection.py src/card_capture/stages/fuse/foil_detection.py
git mv src/card_capture/fusion/median_fusion.py  src/card_capture/stages/fuse/median_fusion.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.fuser                  --new-name=card_capture.stages.fuse.fuser            src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.ecc_registration       --new-name=card_capture.stages.fuse.ecc_registration src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.fusion.foil_detection  --new-name=card_capture.stages.fuse.foil_detection   src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.fusion.median_fusion   --new-name=card_capture.stages.fuse.median_fusion    src app tests harness scripts
```

- [ ] **Step 2: Remove the now-empty `fusion/` package**

```bash
git rm src/card_capture/fusion/__init__.py
rmdir src/card_capture/fusion 2>/dev/null || true
```

- [ ] **Step 3: SVB** — sweep the four names if red.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): fuser/ecc/fusion → card_capture.stages.fuse"
```

### Task 12: `stages/dedup/`

`deduplicator.py` + `identity/embedding_distance.py`.

- [ ] **Step 1: Create the slice and move modules**

```bash
mkdir -p src/card_capture/stages/dedup
printf '"""Stage 9: global dedup (ReID + pHash)."""\n' > src/card_capture/stages/dedup/__init__.py
git add src/card_capture/stages/dedup/__init__.py
git mv src/card_capture/deduplicator.py             src/card_capture/stages/dedup/deduplicator.py
git mv src/card_capture/identity/embedding_distance.py src/card_capture/stages/dedup/embedding_distance.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.deduplicator              --new-name=card_capture.stages.dedup.deduplicator       src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.identity.embedding_distance --new-name=card_capture.stages.dedup.embedding_distance src app tests harness scripts
```

- [ ] **Step 2: Remove the now-empty `identity/` package**

```bash
git rm src/card_capture/identity/__init__.py
rmdir src/card_capture/identity 2>/dev/null || true
```

- [ ] **Step 3: SVB** — sweep `deduplicator` / `identity.embedding_distance` if red.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): deduplicator + identity → card_capture.stages.dedup"
```

### Task 13: `stages/store/`

- [ ] **Step 1: Create the slice and move `storage.py`**

```bash
mkdir -p src/card_capture/stages/store
printf '"""Stage 10: persistence to disk + SQLite."""\n' > src/card_capture/stages/store/__init__.py
git add src/card_capture/stages/store/__init__.py
git mv src/card_capture/storage.py src/card_capture/stages/store/storage.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.storage --new-name=card_capture.stages.store.storage src app tests harness scripts
```

- [ ] **Step 2: SVB** — sweep `card_capture.storage` if red.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(reorg): storage → card_capture.stages.store"
```

---

## Task 14: `training/` — consolidate all offline training

Modules relocated as-is (no content merging). `training/` already exists with `fb_trainer.py`, `presence_trainer.py`.

- [ ] **Step 1: Move the scattered training modules in**

```bash
git mv src/card_capture/train/presence.py          src/card_capture/training/presence.py
git mv src/card_capture/presence/training_data.py  src/card_capture/training/training_data.py
git mv src/card_capture/ml/train_fb.py             src/card_capture/training/train_fb.py
git mv src/card_capture/ml/training/dedup_calibrate.py src/card_capture/training/dedup_calibrate.py
git mv src/card_capture/analysis/hard_case_capture.py  src/card_capture/training/hard_case_capture.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.train.presence            --new-name=card_capture.training.presence          src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.presence.training_data    --new-name=card_capture.training.training_data     src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.ml.train_fb               --new-name=card_capture.training.train_fb          src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.ml.training.dedup_calibrate --new-name=card_capture.training.dedup_calibrate  src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.analysis.hard_case_capture --new-name=card_capture.training.hard_case_capture src app tests harness scripts
```

- [ ] **Step 2: Remove the now-empty packages**

```bash
git rm src/card_capture/train/__init__.py src/card_capture/presence/__init__.py src/card_capture/analysis/__init__.py src/card_capture/ml/training/__init__.py
for d in train presence analysis ml/training; do rmdir src/card_capture/$d 2>/dev/null || true; done
```

- [ ] **Step 3: SVB** — sweep the five old names plus `card_capture.presence` / `card_capture.analysis` / `card_capture.train` / `card_capture.ml.training` if red.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): consolidate all training under card_capture.training"
```

---

## Task 15: `metrics/robustness_pack.py` → `harness/metrics/`

Eval-only; imported only by `tests/regression/*`. Its `tests.regression.pipeline_runner` import stays as-is.

- [ ] **Step 1: Move the module**

```bash
git mv src/card_capture/metrics/robustness_pack.py harness/metrics/robustness_pack.py
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.metrics.robustness_pack --new-name=harness.metrics.robustness_pack src app tests harness scripts
```

- [ ] **Step 2: Remove the now-empty `card_capture/metrics/` package**

```bash
git rm src/card_capture/metrics/__init__.py
rmdir src/card_capture/metrics 2>/dev/null || true
```

- [ ] **Step 3: SVB** — sweep `card_capture.metrics` if red.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): robustness_pack → harness.metrics (eval-only)"
```

---

## Task 16: Relocate stage `run()` modules + the registry

Move each `pipeline/stages/<slice>.py` into its slice as `__init__.py`, then rewrite the
registry import in `runtime_local.py`. The aggregate registry import is broken mid-task and
fixed by the renames before the SVB — so the task is green at its end.

- [ ] **Step 1: Replace each slice placeholder with its run module**

For every slice, drop the placeholder `__init__.py` and move the run module onto it:

```bash
for s in sample detect novelty track refine score resolve fuse dedup store; do
  git rm -q src/card_capture/stages/$s/__init__.py
  git mv src/card_capture/pipeline/stages/$s.py src/card_capture/stages/$s/__init__.py
done
```

- [ ] **Step 2: Rewrite the stage module paths**

```bash
for s in sample detect novelty track refine score resolve fuse dedup store; do
  python -m libcst.tool codemod rename.RenameCommand \
    --old-name=card_capture.pipeline.stages.$s \
    --new-name=card_capture.stages.$s \
    src app tests harness scripts
done
```

This rewrites the `from card_capture.pipeline.stages import (sample as stage_sample, …)` block in `runtime_local.py` to `from card_capture.stages import (…)`.

- [ ] **Step 3: Move the registry docstring module and delete the old stages package**

The top-level `stages/__init__.py` already exists (Task 4). Remove the old empty registry:

```bash
git rm src/card_capture/pipeline/stages/__init__.py
rmdir src/card_capture/pipeline/stages 2>/dev/null || true
```

- [ ] **Step 4: SVB**

Run the SVB. Confirm `runtime_local` imports resolve (`grep -n "from card_capture.stages import" src/card_capture/pipeline/runtime_local.py`). Sweep `card_capture.pipeline.stages` if red.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(reorg): relocate stage run() modules to card_capture.stages.<slice>"
```

---

## Task 17: `review/` — group the legacy Jinja UI

`review.py` → `review/app.py`; `timeline_data.py` and `templates/` move under `review/`.

- [ ] **Step 1: Create the package and move the modules + templates**

```bash
mkdir -p src/card_capture/review
printf '"""Legacy Jinja review UI (card-capture review). Distinct from app/ (SvelteKit)."""\n' > src/card_capture/review/__init__.py
git add src/card_capture/review/__init__.py
git mv src/card_capture/review.py        src/card_capture/review/app.py
git mv src/card_capture/timeline_data.py src/card_capture/review/timeline_data.py
git mv src/card_capture/templates         src/card_capture/review/templates
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.review        --new-name=card_capture.review.app          src app tests harness scripts
python -m libcst.tool codemod rename.RenameCommand --old-name=card_capture.timeline_data --new-name=card_capture.review.timeline_data src app tests harness scripts
```

- [ ] **Step 2: Verify the template path still resolves**

`review/app.py` loads templates via `Path(__file__).parent / "templates"`, which now resolves to `review/templates` automatically — no code change needed. Confirm:

```bash
grep -n 'Path(__file__).parent / "templates"' src/card_capture/review/app.py
```

- [ ] **Step 3: SVB plus a review-app import smoke test**

```bash
python -m pytest tests/ -m "not quarantine" -q | tail -3
lint-imports
python -c "import importlib; importlib.import_module('card_capture.review.app'); print('review import OK')"
```

Sweep `card_capture.review` (note: the new name is a prefix of the old — verify `cli.py`'s `from .review import create_app` became `from .review.app import create_app`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reorg): group legacy review UI under card_capture.review"
```

---

## Task 18: Update `.importlinter`, `pyproject` package-data, `.gitignore`

**Files:**
- Modify: `.importlinter`, `pyproject.toml`, `.gitignore`

- [ ] **Step 1: Replace the layered contract**

In `.importlinter`, replace the `[importlinter:contract:layered]` block with the 7-layer order; update the strict-GPU and sqlite source modules to their new homes if needed (they are unchanged: `card_capture.runtime.strict_gpu` and `card_capture.runtime` still exist):

```ini
[importlinter:contract:layered]
name = layered architecture
type = layers
containers =
    card_capture
layers =
    runtime
    pipeline
    stages
    shared
    ml
    data
    core
```

- [ ] **Step 2: Update `package-data` for the relocated templates**

```bash
grep -n -A3 'tool.setuptools.package-data' pyproject.toml
```

Repoint the templates glob to the new path. The `[tool.setuptools.package-data]` entry should read:

```toml
[tool.setuptools.package-data]
card_capture = ["review/templates/*.html"]
```

(If the existing entry used a different key/glob such as `"templates/*.html"`, replace it with the line above.)

- [ ] **Step 3: Add `var/` to `.gitignore`; remove now-obsolete scratch ignores**

Add a single `var/` line to `.gitignore`. Leave existing model/db ignores that still apply; remove entries that pointed at dirs being consolidated in Task 19 (e.g. `card_capture_output/`, `out/`, `reports/`) once those moves are done — defer the *removal* of those lines to Task 19 so this task stays purely additive.

```bash
printf '\n# consolidated scratch (run outputs, uploads, working DBs, reports)\nvar/\n' >> .gitignore
```

- [ ] **Step 4: SVB (the layered contract is the headline check here)**

```bash
python -m pytest tests/ -m "not quarantine" -q | tail -3
lint-imports
```

Expect: `lint-imports` reports the new 7-layer contract **kept**. The relative order of the
top three (`runtime > pipeline > stages`) and the bottom (`core`) is verified; the middle
order (`shared`, `ml`, `data`) is the most likely to need a small adjustment, since not every
edge among them was pre-checked. If a layer is reported broken, read the offending import
chain `lint-imports` prints and resolve it one of two ways:

- If it's a genuine backward edge (a lower layer importing a higher one — e.g. a stray
  `pipeline`-layer import from a stage; re-check Task 2/16), fix the import or move the helper
  down a layer.
- If it's a legitimate dependency that just contradicts the assumed order among
  `shared`/`ml`/`data` (e.g. `ml` legitimately imports `shared`), **reorder those three
  layers** in the `layers =` list to match reality. Re-run `lint-imports` until all contracts
  are kept.

- [ ] **Step 5: Commit**

```bash
git add .importlinter pyproject.toml .gitignore
git commit -m "refactor(reorg): update import-linter layers, package-data, gitignore"
```

---

## Task 19: Repo-root data/output consolidation

> Reversible-data caution: confirm tracked-vs-untracked before moving; never `rm`. For
> untracked dirs, move on disk; for tracked dirs, `git mv`.

- [ ] **Step 1: Inventory what is tracked**

```bash
for d in out card_capture_output card_capture_uploads data reports sample_run_2026_05_03_fixes sample_run_2026_05_03_fixes_q256; do
  printf '%s: ' "$d"; git ls-files --error-unmatch "$d" >/dev/null 2>&1 && echo TRACKED || echo untracked
done
ls -1 cards.sqlite card_capture_output/cards.sqlite out/cards.sqlite data/*.sqlite data/pipeline.db 2>/dev/null
```

- [ ] **Step 2: Create `var/` and move scratch in**

```bash
mkdir -p var/output var/uploads var/db var/reports
# untracked dirs: plain mv; tracked dirs: git mv (use the inventory from Step 1)
mv out/* var/output/ 2>/dev/null || true
mv card_capture_output/* var/output/ 2>/dev/null || true
mv sample_run_2026_05_03_fixes var/output/ 2>/dev/null || true
mv sample_run_2026_05_03_fixes_q256 var/output/ 2>/dev/null || true
mv card_capture_uploads/* var/uploads/ 2>/dev/null || true
mv data/*.sqlite data/pipeline.db var/db/ 2>/dev/null || true
mv cards.sqlite var/db/ 2>/dev/null || true
# reports/ keep its .gitkeep behavior: move contents
mv reports/* var/reports/ 2>/dev/null || true
```

For any dir reported TRACKED in Step 1, redo its move with `git mv` instead of `mv` so history is preserved, then `git rm -r` the empty original.

- [ ] **Step 3: Repoint default DB/output paths in code**

```bash
grep -rn "card_capture_output\|'out'\|\"out\"" src/card_capture/cli.py src/card_capture/review/app.py
```

Change the defaults found (e.g. `card_capture_output/cards.sqlite` in the `review` subparser and any CLI `--output-dir`/`--db` defaults) to `var/db/cards.sqlite` and `var/output`. Keep `models/` and `golden_set/` references untouched.

- [ ] **Step 4: Finalize `.gitignore`**

Remove the now-obsolete per-dir ignore lines (`out/`, `card_capture_output/`, `card_capture_uploads/`, `reports/`, `data/*.sqlite`, root `cards.sqlite`) that are superseded by `var/`. Leave `models/`-related and `golden_set/` rules.

- [ ] **Step 5: Verify nothing precious was untracked-deleted, then SVB**

```bash
git status --porcelain | head -40
ls models/ golden_set/        # confirm preserved assets untouched
python -m pytest tests/ -m "not quarantine" -q | tail -3
lint-imports
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore(reorg): consolidate scratch data/output/db under var/; repoint defaults"
```

---

## Task 20: Docs consolidation + module-map refresh

**Files:**
- Move: `OPERATOR.md`, `QUICK_REFERENCE.md` → `docs/`
- Modify: `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `docs/architecture/arch-5.5.md`

- [ ] **Step 1: Move operator docs under `docs/`**

```bash
git mv OPERATOR.md docs/OPERATOR.md
git mv QUICK_REFERENCE.md docs/QUICK_REFERENCE.md
```

- [ ] **Step 2: Make `AGENTS.md` / `GEMINI.md` thin pointers**

Replace the contents of each with a one-line pointer (only if they are duplicates of `CLAUDE.md`; if they hold unique content, leave them and skip this step):

```markdown
See [CLAUDE.md](./CLAUDE.md) for the canonical agent context.
```

- [ ] **Step 3: Update `CLAUDE.md` Key Modules table**

Rewrite the `## Key Modules` table to the new paths, e.g. `src/card_capture/pipeline/runtime_local.py` (unchanged), `src/card_capture/stages/<slice>/` (new), `src/card_capture/core/` (models/config/interfaces), `src/card_capture/shared/`, `src/card_capture/runtime/` (workers + gpu). Update the Testing section's path note if any. Update the `OPERATOR.md`/`QUICK_REFERENCE.md` links to `docs/`.

- [ ] **Step 4: Update `docs/architecture/arch-5.5.md` module map**

Update any file-path references in the architecture doc to match the new structure (stages as vertical slices, `core/`, `shared/`, consolidated `training/`, `review/`).

- [ ] **Step 5: SVB + doc-path validator (if present)**

```bash
python -m pytest tests/ -m "not quarantine" -q | tail -3
lint-imports
python scripts/validate_schema_docs.py 2>/dev/null || true
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs(reorg): move operator docs to docs/, refresh module map"
```

---

## Task 21: Final whole-repo verification gate

- [ ] **Step 1: Full suite, contracts, and smoke imports**

```bash
python -m pytest tests/ -m "not quarantine" -q | tail -5
lint-imports
card-capture --help >/dev/null && echo "cli OK"
python - <<'PY'
import importlib
for m in [
    "card_capture.core.models", "card_capture.core.config", "card_capture.core.interfaces",
    "card_capture.core.workers", "card_capture.core.gpu_utils",
    "card_capture.shared.pipeline_utils", "card_capture.shared.stage_metrics",
    "card_capture.stages", "card_capture.stages.detect", "card_capture.stages.sample",
    "card_capture.stages.novelty", "card_capture.stages.track", "card_capture.stages.refine",
    "card_capture.stages.score", "card_capture.stages.resolve", "card_capture.stages.fuse",
    "card_capture.stages.dedup", "card_capture.stages.store",
    "card_capture.training", "card_capture.review.app",
    "card_capture.pipeline.runtime_local",
]:
    importlib.import_module(m)
print("all subpackage imports OK")
PY
```

Expect: pytest pass count == Task 0 baseline; `lint-imports` all contracts kept; cli OK; all imports OK.

- [ ] **Step 2: Confirm no stale references remain**

```bash
# none of these old top-level dotted paths should appear anywhere
for p in card_capture.detectors card_capture.scoring card_capture.cropper card_capture.fuser \
         card_capture.storage card_capture.workers card_capture.gpu_utils card_capture.models \
         card_capture.sampler card_capture.tracking card_capture.presence card_capture.fusion \
         card_capture.deduplicator card_capture.review card_capture.pipeline.stages; do
  hits=$(grep -rln "$p" src app tests harness scripts --include='*.py' | grep -v __pycache__)
  [ -n "$hits" ] && echo "STALE $p:" && echo "$hits"
done
echo "stale scan done"
```

Expect: only "stale scan done" (no `STALE` lines). Any hit is a missed reference — fix with a scoped codemod and re-run the SVB.

- [ ] **Step 3: Build sanity (templates packaged)**

```bash
python -m build --wheel 2>/dev/null && python - <<'PY' || echo "skip build check"
import zipfile, glob
w = sorted(glob.glob("dist/*.whl"))[-1]
names = zipfile.ZipFile(w).namelist()
assert any(n.endswith("review/templates/review.html") for n in names), "templates not packaged!"
print("templates packaged OK")
PY
```

- [ ] **Step 4: Final commit (if Step 2/3 required fixes)**

```bash
git add -A
git commit -m "refactor(reorg): final verification fixes" || echo "nothing to finalize"
```

---

## Done criteria

- `pytest -m "not quarantine"` pass count equals the Task 0 baseline; 0 new failures.
- `lint-imports` reports the 7-layer contract and all `forbidden` contracts **kept**.
- No stale `card_capture.<old-path>` references remain (Task 21 Step 2).
- `card-capture --help` and `card-capture review` import paths resolve; templates are packaged.
- Repo root: only code/assets + a single gitignored `var/`; `models/` and `golden_set/` preserved.
