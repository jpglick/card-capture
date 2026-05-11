# Pipeline v4.x Robustness — Wave 3/4 Remediation Plan

**Status:** open • **Owner:** unassigned • **Source review date:** 2026-05-11

**Context.** Cross-wave review of the Wave 3 (adaptive thresholds) and Wave 4 (foil-aware fusion) implementations found that unit tests pass but several features either ship dead code, double-count work, or are partially wired. This plan converts the review findings into actionable tasks. Tasks are grouped by priority and ordered so dependent fixes follow their prerequisites.

**Scope.** Touches `src/card_capture/pipeline.py`, `src/card_capture/fusion/{foil_detection,median_fusion}.py`, `src/card_capture/fuser.py`, `src/card_capture/calibration/per_video_adaptive.py`, and the wave 3/4 test files.

**Out of scope.** New algorithmic features; threshold tuning beyond what's needed to validate the fixes.

---

## Group A — Critical: features shipped but not consumed

### A1. Persist the fused canonical, not the raw normalized frame

**Defect.** `pipeline.py:710` computes `fused_canonical = _fuse_canonical_frames_with_foil_awareness(...)` but the value is never read. `pipeline.py:821` still writes `entry["normalized"]` to disk. Net effect: foil detection + ECC + fusion runs on every track, the user pays the cost, the persisted crop is unchanged. This is the user-visible failure of the Wave 4 effort.

**Files.**
- `src/card_capture/pipeline.py` — Stage 9 region around line 700–825.

**Required change.**
- [ ] Decide policy: one fused canonical per track (recommended), or one fused per canonical entry. The Wave 4 plan implies one per track.
- [ ] Replace the lone caller so the rectified-path write at line ~821 persists the fused output for at least the **best canonical** entry. For other canonical entries, decide whether to keep the per-frame normalized crop or also replace them.
- [ ] Update telemetry (`is_canonical`, `sharpness`, `glare_mask`) so it still reflects the source-frame metadata even when the canonical image on disk is a fused composite.
- [ ] Handle the `None` return from `_fuse_canonical_frames_with_foil_awareness` (currently assigned without check — see D2).

**Acceptance.**
- A new integration test (E1) asserts that, for a track with multiple canonical entries, `cv2.imread(rectified_path)` does **not** equal `entry["normalized"]` byte-for-byte and instead matches the fused output. (Equivalent: pixel-mean / L2 distance from any single source frame is nonzero.)
- Existing Wave 4 unit tests still pass.

---

### A2. Pick one fusion entry point — remove the duplicate ECC pass

**Defect.** Two functions both register and fuse:
- `MultiFrameFuser.fuse()` (`fuser.py:60-69`) — Wave 2: `register_frames_via_ecc(selected_frames, reference_index=0)` then `np.median(...)`.
- `_fuse_canonical_frames_with_foil_awareness()` (`pipeline.py:1473-1513`) — Wave 4: same registration, then `glare_rejection_fusion` or `np.median`.

If A1 ships without consolidation, every non-foil card pays for ECC twice (once in `MultiFrameFuser`, once in the Wave 4 wrapper). For non-foils the math is also identical, so the second path adds nothing.

**Files.**
- `src/card_capture/pipeline.py`
- `src/card_capture/fuser.py`

**Required change.**
- [ ] Choose one home. Recommended: extend `MultiFrameFuser.fuse()` to accept a `foil_threshold` (default `None` = always median) and delete `_fuse_canonical_frames_with_foil_awareness`. Stage 9 calls `MultiFrameFuser.fuse(...)` exclusively.
- [ ] Move `select_lighting_diverse_indices()` upstream of foil detection so detection sees the same frame subset that will be fused (avoids drift between detection signal and fusion input).
- [ ] Confirm there is exactly one `register_frames_via_ecc` call per track per pass.

**Acceptance.**
- `grep -rn 'register_frames_via_ecc' src/card_capture/pipeline.py src/card_capture/fuser.py` returns at most one production-path call per track-processing iteration.
- Test suite remains green.

---

### A3. Actually consume the adaptive Hamming threshold

**Defect.** `PipelineContext.get_adaptive_hamming_threshold()` exists; `add_intra_track_distance()` is populated at `pipeline.py:1212`; **but** the same-card check at `pipeline.py:1208` still hard-codes `_SAME_CARD_HAMMING_MAX`. The adaptive value is computed and discarded. Wave 3 Task 5 spec required *"Use adapted thresholds in Stage 4 gate **and Stage 8 resolution**"* — only the Stage 4 novelty gate uses adaptation.

**Files.**
- `src/card_capture/pipeline.py:_resolve_session_tracks` (function around line 1129–1220).

**Required change.**
- [ ] In `_resolve_session_tracks`, after collecting intra-track distances on a first pass (or based on prior tracks in the session), query `context.get_adaptive_hamming_threshold(_SAME_CARD_HAMMING_MAX)` and use the returned value at the `same_card = hamming_distance <= …` comparison.
- [ ] Decide ordering: the spec's intent ("adapt from in-video distribution") requires that distances from earlier tracks influence later ones. Two-pass over `session_tracks` is acceptable if the cost is bounded.
- [ ] Fall back to the global constant when fewer than `MIN_SAMPLE_COUNT` observations are available — the `AdaptiveThresholdComputer` already does this; just make sure the caller passes the right global.

**Acceptance.**
- A new integration test (E2) constructs two tracks whose Hamming distance sits between the global and the adapted threshold, and asserts that the dedup decision flips based on adaptation.
- `grep -n 'get_adaptive_hamming_threshold' src/card_capture/pipeline.py` shows at least one call from a non-`PipelineContext` site.

---

## Group B — Adaptive estimator correctness

### B1. Stop double-counting novelty samples

**Defect.** Two functions both call `context.add_novelty_score(...)`:
- `pipeline.py:1343` — over `ScoredCandidate` list (Stage 4 candidate-level gate).
- `pipeline.py:1405` — over `_PreparedTrack` list (track-level prune).

The same track contributes novelty observations at both granularities, biasing the p50 used by `AdaptiveThresholdComputer.compute_novelty_threshold`. The ±20% clip currently masks the bias but the estimator no longer matches the spec ("p50 of in-video distribution").

**Files.**
- `src/card_capture/pipeline.py` (the two gate functions).

**Required change.**
- [ ] Pick one canonical stage that owns novelty-sample collection. Recommended: the candidate-level gate (line 1343), which sees the full distribution before pruning.
- [ ] Remove `context.add_novelty_score(...)` from the other site; have it just *read* `get_adaptive_novelty_threshold` if needed.
- [ ] Add a comment at the surviving site noting it is the sole collector.

**Acceptance.**
- New unit test in `test_wave3_calibration.py`: simulate a sequence of N novelty observations across both gate calls; assert `len(context.observed_novelty_scores) == N` (not 2N).

---

### B2. Don't duplicate the global Hamming constant in the estimator

**Defect.** `_SAME_CARD_HAMMING_MAX = 22` lives at `pipeline.py:55`. `PipelineContext.get_adaptive_hamming_threshold` at line 286 declares `global_threshold: float = 22`. If anyone tunes the constant, the default drifts silently. Also: pHash distances are integer bits; the `float` signature invites rounding bugs downstream.

**Files.**
- `src/card_capture/pipeline.py:PipelineContext.get_adaptive_hamming_threshold`.

**Required change.**
- [ ] Remove the default value, or make it `_SAME_CARD_HAMMING_MAX` so a single source of truth controls the global.
- [ ] Caller (A3) passes `_SAME_CARD_HAMMING_MAX` explicitly.
- [ ] Optional: change signature to `int` end-to-end if all callers pass ints; otherwise keep float and round at the comparison site.

**Acceptance.** `grep -n '= 22' src/card_capture/pipeline.py` returns one line (the constant definition only).

---

## Group C — Foil detection signal hygiene

### C1. Run foil detection on rectified, **unregistered** frames

**Defect.** `_fuse_canonical_frames_with_foil_awareness` calls `register_frames_via_ecc` first and then `detect_foil_card(working_frames, ...)`. ECC warps frames with bilinear interpolation, which low-pass filters them and attenuates the very Laplacian energy the detector keys off. False-negative bias on the cases the feature exists to catch.

**Files.**
- `src/card_capture/pipeline.py:_fuse_canonical_frames_with_foil_awareness` (or its replacement per A2).

**Required change.**
- [ ] Reorder: `is_foil = detect_foil_card(frames, threshold)` **before** registration. Then `working_frames = register_frames_via_ecc(frames, ...)`. Use `working_frames` for fusion.
- [ ] Rationale comment at the reorder.

**Acceptance.**
- Updated wave 4 test: ECC-registered foil frames still classify as foil; ECC-registered non-foil frames still classify as non-foil. (Validates the reorder doesn't break the unit-test premise.)

---

### C2. Calibrate `foil_threshold`; the current `50.0` is unjustified

**Defect.** `test_foil_detection_threshold_tuning` only asserts `isinstance(result, bool)`. There is no evidence the default `50.0` separates foil from non-foil on real cards. Bright cards / high-ISO video raise Laplacian variance independent of holography.

**Files.**
- New: `scripts/calibrate_foil_threshold.py` (mirror of `scripts/calibrate_wave3.py`).
- `src/card_capture/fusion/foil_detection.py` — optional: subtract a luminance-mean baseline before variance.
- `tests/test_wave4_foil.py` — replace the bool-only assertion with a separation test using held-out fixture frames.

**Required change.**
- [ ] Gather (or synthesize) a small labeled set of foil vs. non-foil canonical frames from the regression corpus.
- [ ] Sweep `threshold` and pick the value at the inflection point; record it as the new default and link the sweep output from the docstring.
- [ ] Optional: normalize variance by per-frame mean Laplacian magnitude so dark/bright cards are comparable.

**Acceptance.**
- New unit test asserts `detect_foil_card(foil_fixture) is True` and `detect_foil_card(non_foil_fixture) is False` at the calibrated default.
- The current `test_foil_detection_threshold_tuning` is replaced or retired.

---

### C3. Use luminance distance in `glare_rejection_fusion`

**Defect.** The "closest to median" pick currently sums |B − Bm| + |G − Gm| + |R − Rm|. Glare is luminance-driven; a frame that is color-shifted but luminance-correct can lose to a frame that is luminance-glaring but color-balanced. Wave 1 already standardized on Lab for novelty — same logic applies here.

**Files.**
- `src/card_capture/fusion/median_fusion.py:glare_rejection_fusion`.

**Required change.**
- [ ] Compute closeness on the L (luminance) channel of Lab, then index BGR pixels by the L-argmin frame.
- [ ] Keep the BGR output unchanged (only the *selection* metric changes).

**Acceptance.**
- Existing `test_glare_rejection_fusion_preserves_luminance` and `test_glare_rejection_fusion_shape` still pass.
- New unit test: three frames where two are color-shifted but luminance-correct and one is luminance-glaring; the L1-BGR version picks the glaring frame for some pixels; the new L-only version does not.

---

## Group D — Hygiene

### D1. Unify test import style

**Defect.** `tests/test_wave4_foil.py` uses `from src.card_capture.fusion.foil_detection import ...`. Every other wave test uses `from card_capture.*`. The `src.` form works today only via PEP 420 namespace-package resolution after pytest adds the repo root to `sys.path`; an editable install breaks it.

**Files.**
- `tests/test_wave4_foil.py` (4 import statements).

**Required change.**
- [ ] Replace `from src.card_capture.X import Y` with `from card_capture.X import Y` (4 sites: lines 3, 52, 71, 86).

**Acceptance.** `pytest tests/test_wave4_foil.py` still passes; `grep -rn 'from src\\.card_capture' tests/` returns no hits.

---

### D2. Null-check or tighten the fusion wrapper return type

**Defect.** `_fuse_canonical_frames_with_foil_awareness` declares `-> Optional[np.ndarray]`, the caller at `pipeline.py:710` ignores the possibility of `None`. Currently mooted by A1 (the value is dropped), but will be a real null-deref the moment A1 lands.

**Files.** Wherever A1's replacement of the call site is.

**Required change.**
- [ ] Either: at the call site, assert `fused_canonical is not None` (it can only be None when `frames` is empty, which Stage 9 must already guard against) and remove `Optional` from the return type; **or** keep `Optional` and add an explicit fallback (e.g., reuse `best_canonical["normalized"]`).
- [ ] Whichever path is chosen, the call site must not propagate `None` into `cv2.imwrite`.

**Acceptance.** A unit test exercising the empty-frames branch documents the chosen behavior.

---

### D3. Expose foil tuning knobs via `ProcessingOptions`

**Defect.** `pipeline.py:712` hard-codes `foil_threshold=50.0` and `use_ecc_registration=True`. The Wave 4 plan called the threshold "tunable" but exposed no surface for tuning.

**Files.**
- `src/card_capture/pipeline.py` — `ProcessingOptions` (find via `grep -n 'class ProcessingOptions' src/card_capture/`).
- `src/card_capture/pipeline.py` — Stage 9 call site.

**Required change.**
- [ ] Add `foil_threshold: float = 50.0` and `enable_foil_aware_fusion: bool = True` to `ProcessingOptions` (or wherever pipeline-level config lives).
- [ ] Thread through to the Stage 9 call.

**Acceptance.** Setting `enable_foil_aware_fusion=False` via the options surface forces every track down the median path, verifiable by a unit test that monkeypatches `detect_foil_card` to always return `True` and confirms median fusion is still used when the flag is off.

---

### D4. Remove the stale `build/lib` artifact

**Defect.** `build/lib/card_capture/pipeline.py` is 31 KB and out of sync with `src/`. Not on the import path today but a footgun for anyone debugging with a custom PYTHONPATH.

**Required change.**
- [ ] `rm -rf build/` and add `build/` to `.gitignore` if not already present.
- [ ] Verify nothing in `pyproject.toml`/CI references `build/lib`.

**Acceptance.** `git status` clean after `rm`; tests still pass.

---

### D5. Make `detect_foil_card`'s contract match its code

**Defect.** Spec said `detect_foil_card` returns `False` for `len(frames) < 2`. Implementation relies on `np.var` of a single-frame stack returning zero, which works by accident and doesn't match the documented contract.

**Files.** `src/card_capture/fusion/foil_detection.py`.

**Required change.**
- [ ] Add explicit `if len(frames) < 2: return False` at the top of `detect_foil_card`.
- [ ] Mirror in `compute_laplacian_variance`: `if len(frames) < 2: return 0.0`.

**Acceptance.** New unit test passes empty / single-frame lists.

---

## Group E — Integration tests (do not skip)

These are the tests that would have caught A1/A2/A3 if they had existed. Land them with the corresponding fix.

### E1. Stage 9 end-to-end: persisted crop reflects the fusion path

- [ ] In `tests/test_wave4_foil.py`, drive `VideoProcessor.process` (or its narrowest exposed wrapper that includes Stage 9) on a synthetic 2-frame track.
- [ ] Assert `cv2.imread(rectified_path)` is not byte-equal to any single source canonical frame for a multi-frame track.
- [ ] Assert that for a single-frame track, the persisted image is byte-equal to the source frame.

### E2. Adaptive Hamming threshold actually flips a dedup decision

- [ ] Construct two `_PreparedTrack`s with controlled pHashes whose Hamming distance is `H` such that `H > _SAME_CARD_HAMMING_MAX` but `H <= adaptive_threshold` when 10+ smaller intra-track distances have been recorded on `PipelineContext`.
- [ ] Assert dedup before adaptation classifies them as different cards; after adaptation, as same.

### E3. Foil-detection separation on labeled fixtures

- [ ] Add a small fixture set under `tests/fixtures/foil/{foil,non_foil}/*.png` (3–5 each).
- [ ] Test asserts `detect_foil_card(frames, threshold=DEFAULT)` returns the labeled class for each.

---

## Suggested order

1. **A1 + E1** together (fix the dead code with the test that ensures it stays fixed).
2. **A2** (collapse the two fusion entry points; depends on A1 to know what the consumer expects).
3. **C1** (correctness of detection signal before any calibration work).
4. **A3 + E2** together (wire adaptive Hamming with its integration test).
5. **B1, B2** (estimator hygiene).
6. **D1, D2, D5** (small, can be batched).
7. **C2 + E3** (calibration & fixtures; needs A2/C1 to be stable first).
8. **C3** (Lab-distance glare rejection).
9. **D3, D4** (operational polish).

## Out-of-scope follow-ups (record for later)

- Whether to fuse all canonical entries or only the best one.
- Whether foil-detection should be tile-wise (some cards have foil panels only).
- Tying threshold auto-sweep (`scripts/calibrate_wave3.py`) to a CI gate so the regression metrics from Wave 1 actually block merges.
