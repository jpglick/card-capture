# Systematic Dead-Code Elimination — Design Spec

**Date:** 2026-05-31
**Branch base:** `cleanup/v5.5-post-merge`
**Status:** Approved (design), pending implementation plan

---

## 1. Goal

Remove the accumulated dead code from prior pipeline iterations using a
**systematic, evidence-driven process** rather than ad-hoc deletion. Two
distinct cleanup mandates are in scope:

1. **Dead code removal** — code with no live reference path (whole unused
   subpackages and intra-module dead symbols/branches).
2. **Apple-Silicon-only mandate** — v5.5+ supports only Apple Silicon (MPS).
   All non-MPS hardware paths (CUDA, x86 device branches) are removed. CPU
   execution survives **only** behind an explicit opt-in flag.

**Out of scope:** `app/` is explicitly excluded from dead-code analysis and
removal (handled separately later).

**Guiding principle:** Remove eagerly. The robust test suite + video sampling
gate is the arbiter of whether a removal was safe.

---

## 2. Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| D1 | CPU-fallback handling under AS-only mandate | **MPS-only, hard-fail unless `allow_cpu_fallback` flag set.** Remove CUDA outright. |
| D2 | Video pass/fail signal | **Smoke gate, then metric regression.** |
| D3 | Module scope | **Whole modules + intra-module** dead code. |
| D4 | Harness location | `scripts/deadcode/` |
| D5 | AS-only gate mechanism | New `allow_cpu_fallback: bool = False` on `PipelineConfig` |
| D6 | Metric regression tolerance | recall/precision within **±2%**; SSIM ≥ baseline − **0.02** |

---

## 3. Approach

Of three candidate mechanisms for "apply all, then bisect":

- **A — Manifest-driven apply-all + automated subset bisection** *(chosen)*.
  Applies every change at once; on gate failure, binary-searches *subsets* of
  the change list to isolate culprit(s). Handles **multiple independent
  culprits**, the expected failure mode.
- **B — Confidence-tiered sequential gating** *(rejected)*. Less bisecting but
  violates the "all at once" requirement.
- **C — Pure `git bisect` over per-change commits** *(rejected)*. `git bisect`
  assumes a monotonic good→bad transition; independent removals break that when
  2+ changes each fail.

---

## 4. Components

### 4.1 Vulture + whitelist
- Add `vulture` to dev dependencies (`pyproject.toml`).
- **Scan roots:** `src harness scripts tests app` together — consumers count as
  *usage*, preventing false "unused" flags on code they call.
- **Action scope:** only findings located under `src/` are acted on. **Never
  `app/`.**
- `scripts/deadcode/whitelist.py` suppresses known dynamic-usage false
  positives: click commands, pytest fixtures, dataclass/pydantic fields,
  `__all__` exports, console-script entry points.

### 4.2 Findings → confidence-scored manifest
`scripts/deadcode/manifest.json`. Each entry:

```json
{
  "id": "string",
  "kind": "whole_module | symbol | branch | non_mps_path",
  "path": "src/...",
  "symbol": "name or null",
  "vulture_confidence": 90,
  "adjusted_confidence": 95,
  "decision": "remove | keep | investigate",
  "tier": 0,
  "reason": "string"
}
```

`adjusted_confidence` augments vulture's score with a dynamic-reference grep
(getattr / string-name / entry-point / relative-import checks) that vulture
cannot see.

**Tiers:**
- **T0 — Non-Apple-Silicon paths.** CUDA branches removed outright; device
  probing collapsed to MPS-only. Candidate files (from probe): `fuser.py`,
  `models.py`, `scoring.py`, `detectors.py`, `gpu_utils.py`,
  `fusion/foil_detection.py`, plus device-probe sites in `gpu_refinement.py`,
  `frame_quality.py`, `resolve.py`.
- **T1 — High confidence** (vulture ≥90%, no dynamic refs). Whole dead modules +
  dead symbols → `remove`. Candidate whole-module-dead (zero import refs in
  src/app/harness/scripts): `analysis`, `calibration`, `identity`, `metrics`,
  `platforms`, `runtime`, `templates`. (Must be confirmed against relative/
  dynamic imports before removal.)
- **T2 — Medium** (60–90% or ambiguous). Investigate each; set `remove`/`keep`.
  Includes `train`/`training` (referenced only by scripts).
- **T3 — Low / questionable.** Investigate; default `keep` unless proven dead.

### 4.3 CPU-allowed gate (AS-only mandate)
- Remove all CUDA branches.
- Device resolution **hard-fails when MPS is unavailable** unless
  `PipelineConfig.allow_cpu_fallback` is `True`.
- `RuntimeMode == "cpu_debug"` implies `allow_cpu_fallback = True`.
- CPU code paths (e.g. `PrecisionNormalizer` Kornia-failure fallback) remain in
  the tree but are reachable **only** behind the flag.

### 4.4 Validation gate (`scripts/deadcode/gate.py`)
Three stages, fail-fast, reports which stage failed:
- **A. Tests** — `pytest tests/ -m "not quarantine" -q` (591-pass baseline) +
  `tests/architecture/` (import-linter contracts).
- **B. Video smoke** — `card-capture process` on `IMG_5872.MOV` and
  `IMG_5922.MOV`; assert exit 0 and non-zero card rows in SQLite.
- **C. Metric regression** — `card-capture harness run` vs the v5.5 baseline;
  assert recall/precision within ±2% and SSIM ≥ baseline − 0.02 (per D6).

### 4.5 Apply-all + bisect driver (`scripts/deadcode/run.py`)
- Each `remove` entry is one **reversible patch** (a git commit on the cleanup
  branch).
- Apply all → run gate → pass ⇒ done.
- On failure: binary-search subsets (checkout base + subset, re-run gate) to
  isolate culprit(s); flip culprits to `decision = keep`; re-apply remainder;
  re-verify the full gate.
- Emit `scripts/deadcode/report.md`: every change with applied/reverted status
  and reason.

---

## 5. Data flow

```
vulture src harness scripts tests app
   → findings.json
   → enrich (dynamic-ref grep) + score
   → manifest.json
   → run.py: apply all `remove` patches (1 commit each)
   → gate.py (A tests → B video smoke → C metric regression)
        pass → final commit
        fail → subset bisect → demote culprits to keep → re-gate
   → report.md
```

---

## 6. Deliverables

- Cleanup branch with dead code removed and the AS-only runtime enforced.
- Reusable `scripts/deadcode/` harness (`whitelist.py`, `manifest.json`,
  `gate.py`, `run.py`).
- `scripts/deadcode/report.md` decision log.
- `PipelineConfig.allow_cpu_fallback` flag + MPS hard-fail.
- Updated `CLAUDE.md` / `docs/architecture` reflecting removed modules and the
  Apple-Silicon-only runtime.
- `vulture` added to dev dependencies.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Vulture misses dynamic refs (getattr, string dispatch, entry points) | Scan consumers as usage roots; enrich with dynamic-ref grep; whitelist; the video+test gate is the final arbiter. |
| Whole-module "0 imports" is a relative-import false negative | Confirm each candidate against relative/dynamic imports before deletion; gate catches escapes. |
| Multiple independent culprits in one batch | Subset bisection (approach A) isolates each, not just the first. |
| Removing a CPU path needed on dev machines without MPS | `allow_cpu_fallback` flag preserves opt-in CPU execution. |
| Metric baseline trustworthiness | Smoke gate (B) runs first to catch crashes; metric gate (C) uses documented v5.5 baseline with explicit tolerance. |
