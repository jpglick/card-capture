# Card Capture Pipeline V4: Hardening & Architectural Evolution

**Date:** 2026-05-10
**Status:** Draft (awaiting user review)
**Supersedes:** none — augments V3 architecture documented in `docs/superpowers/pipeline-v3-overview.md`

---

## 1. Goals & Non-Goals

### Goals

1. **Reduce missing/phantom cards on a labeled corpus by ≥ 50% relative to V3.** This is the user's stated top pain.
2. **Make every future quality change measurable** via a regression harness backed by a hand-labeled golden corpus.
3. **Replace fragile heuristics with learned or robust components**, specifically: track-length F/B assignment, pHash dedup, pixel-statistic presence detection, hand-rolled `HysteresisTracker`.
4. **Keep the pipeline runnable end-to-end on a single Apple Silicon Mac.** No CUDA-only models, no cloud dependencies for inference.

### Non-Goals

- Real-time / live capture. Pipeline remains batch processing of recorded videos.
- Cloud or multi-tenant deployment.
- OCR-based card identification (set numbers, names). Future work; out of scope here.
- Replacing the YOLO-OBB detector itself. We augment it with corner refinement and (in gated Phase E) SAM 2 masks; we do not retrain or swap the OBB model.

---

## 2. Constraints

- **Hardware:** Apple Silicon (M-series), MPS backend available. Assumes models that can run via `torch` MPS or Core ML conversion.
  - SAM 2: only Hiera-T or Hiera-S variants in scope; Hiera-L excluded.
  - Embedding model: DINOv2 ViT-S/14 (~22M params) preferred over ViT-B/14.
  - Tracker: ByteTrack (CPU, no re-id net) preferred over BoT-SORT (re-id net per detection) unless ByteTrack measurably under-performs in Phase A.
- **Training:** Tiny classifiers (Front/Back head, presence head) train on Mac in minutes from a few hundred labeled crops. No on-device training of foundation models.
- **Storage:** FAISS index lives on disk alongside the existing SQLite DB; embeddings stored per Card Instance in `storage.py`. SQLite remains the source of truth; FAISS is rebuildable.
- **Determinism:** Pipeline must produce stable output on identical input + identical model weights. The harness depends on this to attribute deltas to changes rather than noise.
- **Corpus size:** User reports < 5 videos available. Spec recommends recording 3 additional videos targeting fast-flip, high-glare/foil, and occlusion-heavy scenarios before the harness can be considered representative.

---

## 3. Bootstrap Phase (Spec 0): Labeled Corpus + Regression Harness

This phase ships first. Nothing downstream is measurable without it.

### 3.1. Labeled corpus

- **Location:** `tests/fixtures/golden_corpus/<video_id>/<video_id>.mp4` plus `<video_id>.truth.json`.
- **Selection criteria:** mix of difficulty modes — easy single-card, fast-flip, slow-flip, high-glare/foil, finger-occlusion, multi-card-in-frame.
- **Initial size:** all currently available videos (< 5) plus 3 newly recorded videos targeting the gaps above.

### 3.2. Ground-truth schema

Per-video file `<video_id>.truth.json`:

```json
{
  "video_id": "abc123",
  "video_path": "tests/fixtures/golden_corpus/abc123/abc123.mp4",
  "labeled_at": "2026-05-10",
  "labeled_by": "josh",
  "expected_cards": [
    {
      "card_id": "card_001",
      "physical_card_key": "topps_chrome_2024_42",
      "front_present": true,
      "back_present": true,
      "approx_front_window_ms": [12500, 15800],
      "approx_back_window_ms": [16100, 18900],
      "notes": "foil card, heavy glare on front"
    }
  ]
}
```

- `card_id` is local to the video.
- `physical_card_key` is the cross-video dedup ground truth. Two cards in different videos with the same `physical_card_key` are duplicates of one physical card.
- `approx_*_window_ms` is approximate to the second; harness tolerates ±500 ms when matching pipeline output to expected windows.

### 3.3. Labeling mode in Review UI

- New route in `review.py`: `GET/POST /label/<video_id>`.
- Reuses existing UI components but inverts the workflow: user defines truth instead of accepting pipeline output.
- Workflow:
  1. Load video → run current pipeline → display Card Instances on a timeline.
  2. User marks each instance as: `real_front`, `real_back`, `phantom`, `duplicate_of:<card_id>`.
  3. User clicks on the timeline at any timestamp where a card was missed and adds a `missing_card` entry.
  4. User assigns optional `physical_card_key` to mark cross-video duplicates.
  5. Save writes `<video_id>.truth.json`. Idempotent: re-opening the page loads existing truth.

### 3.4. Regression harness

- **Location:** `tests/regression/harness.py`.
- **Entry point:** `python -m card_capture.harness run --corpus tests/fixtures/golden_corpus --out reports/<git-sha>.json`.
- **Per video, computes:**
  - **Card recall:** matched_cards / expected_cards.
  - **Phantom rate:** unmatched_pipeline_outputs / total_pipeline_outputs.
  - **F/B accuracy:** correct_angle_assignments / total_angle_assignments (only on cards where both sides exist in truth).
  - **Cross-video dedup F1:** standard precision/recall on `physical_card_key` matches across all videos in the corpus.
  - **ID switches:** count of track ID changes within a session for the same physical card (uses pipeline's per-track logging).
  - **Visual quality proxy:** average sharpness score on canonical crops (already computed by `scoring.py`).
  - **Wall-clock duration** and **peak resident memory** (via `resource.getrusage`).
- **Output artifacts:**
  - `reports/<git-sha>.json` — machine-readable per-video and aggregate metrics.
  - `reports/<git-sha>.md` — human-readable summary, with deltas vs. the previous report if one exists in `reports/`.
- **CI hook:** `make harness` target. No GitHub Actions integration in this spec; local-only.

### 3.5. Spec 0 deliverables

1. `tests/fixtures/golden_corpus/` populated and committed.
2. Labeling mode in `review.py`.
3. `tests/regression/harness.py` + `python -m card_capture.harness` entry point.
4. `reports/baseline_v3.json` and `reports/baseline_v3.md` captured before any other change.

### 3.6. Spec 0 effort

~2–3 days (most is user labeling time).

---

## 4. Phase A: Detection & Tracking Robustness

Directly attacks the user's top pain (missing/phantom cards).

### A1. Stage 1 presence: tiny visual classifier

**Current state.** `AdaptivePresenceSampler` (`sampler.py:273`) z-score-normalizes a composite of sharpness, edge density, variance, motion, and empty-pixel ratio, then Otsu-thresholds. Sophisticated for pixel statistics but has no concept of "card-shaped object" — fires on hands, packaging, shadows.

**Change.** Train a **MobileNetV3-Small binary classifier** ("card present" vs "no card") at 224 px input. Replace the composite-score thresholding step with classifier output; keep the existing batched scan loop and motion estimation as inputs to candidate-frame selection within a presence window.

**Why MobileNetV3-Small.** ~3–5 ms per frame on M-series MPS. Cheaper than YOLO-Nano @ 320 px (~10 ms) and an order of magnitude cheaper than full YOLO-OBB. Trains in minutes from ~few thousand labeled crops.

**Bootstrap training data.** Free from existing pipeline output:
- Positives: rectified crops from current high-confidence YOLO detections (confidence ≥ 0.7).
- Negatives: random patches sampled from frames in the same videos that produced no detections at all.
- Augmentation: standard (random crop, flip, color jitter, brightness) plus glare simulation (blend a soft white blob).

**Where it slots in.** Replaces the call site of `_otsu_threshold` inside `AdaptivePresenceSampler._build_windows`. The composite score becomes a tiebreaker for sharpness selection within a window, not a presence gate.

### A2. Tracker: replace HysteresisTracker with ByteTrack

**Current state.** `HysteresisTracker` (`selector.py:32`) uses confidence-thresholded centroid distance with a hand-rolled flip-detection rule that breaks tracks on >30% area change.

**Change.** Replace with **ByteTrack**. Use the `supervision` library's wrapper or the `ultralytics` built-in integration since YOLO is already a dependency.

**Why ByteTrack first, not BoT-SORT.** ByteTrack's defining feature is two-stage association: first match high-confidence detections to tracks; then run a second pass associating *low-confidence* detections to any unmatched tracks. This directly addresses the user's pain — current code drops low-confidence detections during occlusion, fragmenting tracks. BoT-SORT adds an appearance-embedding (re-id) step per detection. Re-id helps with ID persistence through visual changes (like flips), but in this pipeline the F/B classifier shipping in Phase B is a stronger appearance signal than a generic re-id net. Land ByteTrack; if Phase B harness shows ≥ N ID switches through flips, upgrade to BoT-SORT.

### A3. Adaptive session gap

**Current state.** Fixed `null_patience_frames: 6` in `card_capture_config.json`. Globally tuned, ignores per-video pacing.

**Change.** During Pass 1 of each video, build a histogram of inter-presence-window gaps. Set the session boundary at `max(P95_of_gaps, 0.5s)`. Cap at 3 s max to prevent run-on sessions if the user takes a long break.

### A4. Remove area-drop hard rule

**Current state.** `HysteresisTracker.detect_flip` (`selector.py:146`) breaks the track on >30% area drop or >50% increase.

**Change.** Delete. ByteTrack's Kalman filter + IoU association handles area change correctly: a flipping card briefly appears smaller, but IoU at the same screen position stays high and the track persists. The hand-rolled rule was fighting the tracker.

This is a *deletion*, not a rewrite. The simplification is part of the win.

### Phase A decision gate (soft)

- Card recall on golden corpus ≥ baseline + 20 percentage points.
- Phantom rate ≤ baseline / 2, OR ≤ 2% absolute (whichever is looser — a low baseline trivializes the relative gate).
- Wall-clock not regressed by > 30%.
- ID switches measured (no target yet — instrumentation only; target set in Phase B).

### Phase A deliverables

- `models/presence_classifier.pt` (MobileNetV3-Small weights) + training script `src/card_capture/train/presence.py`.
- `AdaptivePresenceSampler._build_windows` updated to call presence classifier on each scan-resolution proxy frame (scan frame upscaled to 224 px before inference).
- `selector.py`: `HysteresisTracker` deleted; ByteTrack adapter shipping in new `selector_bytetrack.py`.
- `pipeline.py`: per-video adaptive session gap computed in Pass 1; `null_patience_frames` config retained as max-bound override.
- Harness report comparing Phase A vs. baseline.

### Phase A effort

- A1 classifier: 1 day train + 1 day integrate.
- A2 ByteTrack: 1 day.
- A3 adaptive gap: 0.5 day.
- A4 deletion + cleanup: 0.5 day.
- **Total: ~4 days.**

---

## 5. Phase B: Identity & Angle Correctness

### B1. Front/Back binary classifier

**Current state.** Track-length heuristic: longest track in a session = Front. Inverts whenever the user inspects the back longer than the front.

**Change.** **MobileNetV3-Small binary classifier** trained on rectified crops, with a confidence-threshold fallback to track-length.

- **Inference point:** after Stage 5 rectification, on the canonical crop.
- **Output:** `{angle: "F"|"B", confidence: float, source: "classifier"|"track_length_fallback"}` written to Card Instance metadata.
- **Fallback policy:** if classifier confidence < 0.8 on both candidate sides in a session, fall back to track-length. Source field logs which path was taken.
- **Training data:** rectified crops from labeled corpus (~200–500 per class with augmentation). Active-learning loop in Phase D adds more from Review UI corrections over time.
- **Backbone sharing:** Phase A1 and Phase B1 are both MobileNetV3-Small with different heads. Train independently for now; consider shared backbone in a later optimization pass.

### B2. Embedding-based dedup (DINOv2 + FAISS)

**Current state.** pHash on rectified crop in `deduplicator.py`. Brittle to glare differences, slight rotation residual, color cast across videos.

**Change.** Two-tier dedup.

- **Tier 1 (fast filter):** keep pHash + add dHash. Combined Hamming distance < 4 over 64 bits → declare duplicate, short-circuit. Catches the easy near-identical cases for ~free.
- **Tier 2 (real matcher):** **DINOv2 ViT-S/14** embedding (384-dim) computed on the rectified crop. Indexed in **FAISS flat L2** (small dataset, no IVF/HNSW needed yet). Cosine similarity threshold **0.92** for "same physical card" — calibrated via Phase D threshold-sweep harness.

**Both-sides-match requirement.** To declare a cross-video duplicate, both Front and Back embeddings must match (each above the 0.92 threshold). If one side is absent in either record, the present side must clear a higher threshold (0.95). This robustness check protects against B1 errors: a wrong F/B label breaks the both-sides check and surfaces as a low-confidence dedup, which the Phase D review queue prioritizes.

**Storage changes.**
- Add `embedding_front BLOB` and `embedding_back BLOB` columns to the Card Instance table in `storage.py`.
- New file `card_embeddings.faiss` adjacent to the SQLite DB. Rebuilt on demand from the table; SQLite is the source of truth.
- Migration: new columns nullable; backfill on first run by embedding existing crops.

**Why DINOv2 ViT-S/14 on Apple Silicon.** ~22M params, ~50 ms per image on MPS (one-time per Card Instance, not per frame). Self-supervised; designed for general visual identity. Robust to filters, compression, and crops — exactly the variability that breaks pHash across videos.

### Phase B decision gate (soft)

- F/B accuracy on golden corpus ≥ 99% (on cards where both sides are present in truth).
- Cross-video dedup F1 ≥ 0.95.
- No regression in Phase A metrics (recall, phantom rate).

### Phase B deliverables

- `models/fb_classifier.pt` + training script `src/card_capture/train/fb.py`.
- `deduplicator.py` rewritten with two-tier (pHash short-circuit + DINOv2 + FAISS) logic.
- `storage.py` migration: `embedding_front`, `embedding_back` columns added; lazy backfill on first read.
- `card_embeddings.faiss` rebuilder script.
- Harness report comparing Phase B vs. Phase A.

### Phase B effort

- B1 classifier: 1.5 days (data prep, train, integrate).
- B2 dedup: 2 days (storage migration, FAISS plumbing, threshold calibration on harness).
- **Total: ~3.5 days.**

---

## 6. Phase C: Visual Quality

### C1. Multi-frame median + glare-aware fusion

**Current state.** Single canonical frame is rectified and shipped. Top-K sharp frames per session are computed but discarded.

**Change.** For each track, rectify the **top-K = 5 sharpest frames** into the same target grid, then per-pixel median across them.

**Glare-aware mask.** Per frame, mark pixels where V > 240 in HSV (saturated). Exclude marked pixels from the median for that frame. If all K frames are saturated at a location, fall back to the mean across all K.

**Why this works.** The K rectifications use slightly different homographies (sub-pixel diversity from different viewing angles). Glare moves across frames as the card tilts; median across K frames suppresses glare automatically; saturation mask cleans up the residual.

**Implementation.**
- `fuser.py` already exists as a stub — promote it to host the fusion logic.
- `cropper.py` exposes a multi-frame entry point: `rectify_and_fuse(frames, corners_per_frame, target_size, k=5)`.
- `gpu_refinement.py` Kornia path runs K warps; CPU fallback runs sequential warp + numpy median.

**Cost.** K extra warps per Card Instance on GPU. Negligible.

### C2. Corner refinement

**Current state.** YOLO-OBB's 4 corners are accepted as-is. ~5 px error on 4K becomes visible skew in the rectified crop.

**Change.** Classical corner refinement on canonical frames only:

1. For each predicted corner, extract a 96×96 ROI in the high-res frame.
2. Canny edge detection in the ROI.
3. RANSAC line fit on the dominant edges in the two corner directions.
4. Intersect the two refined edge lines → refined corner coordinate.

If RANSAC fails (no dominant line), fall back to the YOLO-OBB corner.

**Cost.** ~10–20 ms per detection on CPU. Run only on canonical frames (one per track), not on every detection.

**Why classical, not learned.** No labeled corner data. Classical RANSAC is a 1-day implementation and works well on clean cards. Learned corner refinement is deferred to Phase E (SAM 2 mask → quad fitting gives this for free).

### C3. Geometric filter on rectified geometry

**Current state.** AABB aspect ratio ∈ [0.50, 0.95], area ∈ [10%, 80%]. Rejects high-tilt cards.

**Change.** Two-stage filter.

1. **Cheap pre-check:** convex hull of 4 corners (no near-collinear triples), area ∈ [10%, 90%]. Drops obvious garbage cheaply.
2. **Geometric check on rectified aspect:** fit homography, project corners to rectified grid, measure rectified width/height. Require rectified aspect ∈ [0.65, 0.78] (real card = 2.5/3.5 ≈ 0.714).

The existing `_aspect_ratio` helper in `selector.py:377` already computes max-edge-based aspect — we replace its use with the rectified version.

### Phase C decision gate (soft)

- Visual sharpness score on canonical crops ≥ baseline + 15%.
- No regression in card recall or phantom rate.
- Manual visual spot-check on labeled corpus confirms reduction in skewed crops.

### Phase C deliverables

- `fuser.py` promoted: implements `rectify_and_fuse(frames, corners_per_frame, target_size, k=5)` with glare-aware median.
- `cropper.py`: multi-frame entry point added; existing single-frame path retained for non-canonical use.
- `detectors.py`: corner-refinement pass on canonical frames; classical RANSAC implementation in new `corner_refine.py`.
- `selector.py`: AABB aspect-ratio filter replaced by convexity + rectified-aspect check.
- Harness report comparing Phase C vs. Phase B.

### Phase C effort

- C1 fusion: 1 day.
- C2 corner refinement: 1 day.
- C3 geometric filter: 0.5 day.
- **Total: ~2.5 days.**

---

## 7. Phase D: Confidence & Closing the Loop

### D1. Composite per-instance confidence

For every Card Instance, compute a single 0–1 score:

```
conf_total = w1 * detection_conf       # max YOLO conf across the track
           + w2 * tracker_stability    # track_length / session_length
           + w3 * sharpness_norm       # existing scoring.py output, normalized
           + w4 * fb_classifier_conf   # from B1
           + w5 * dedup_consistency    # 1.0 if F/B both matched a known dup, else 0.5/1.0
```

Default weights `(0.25, 0.20, 0.20, 0.20, 0.15)`. Tunable via `card_capture_config.json` under a new `confidence_weights` key. Stored in Card Instance metadata.

**UX.** Review UI sorts the queue by lowest `conf_total` first. User attention goes where it matters.

### D2. Active-learning hooks

Every Review UI action that contradicts a model prediction generates a labeled training example written to `tests/fixtures/active_learning/<date>.jsonl`:

- "User flipped angle from F to B" → `{kind: "fb", crop_path, label: "B"}` for the F/B classifier.
- "User marked phantom" → `{kind: "presence", crop_path, label: "no_card"}` for the presence classifier.
- "User merged two cards as duplicates" → `{kind: "dedup_positive", embed_a, embed_b}`; "user split a duplicate" → `{kind: "dedup_negative", embed_a, embed_b}`.

Manual retrain script:

```
python -m card_capture.train fb_classifier --include-active-learning
python -m card_capture.train presence_classifier --include-active-learning
```

No automatic retraining. Trigger when corrections accumulate (~50+).

### D3. A/B harness expansion

Builds on Spec 0 minimal harness:

- **Per-version dashboard:** markdown report comparing two report files side-by-side, with red/green deltas per metric.
- **Per-card diff view:** for any video, show which Card Instances changed status between two pipeline versions. Useful for "did this change break video X?".
- **Threshold sweep mode:** runs the harness across a grid of one parameter (e.g., dedup threshold ∈ [0.88, 0.90, 0.92, 0.94, 0.96]) and reports metric curves. Eliminates threshold-guessing.

### Phase D decision gate (soft)

- Review UI confidently sorts; manual spot-check confirms low-confidence items genuinely need attention.
- ≥ 50 active-learning examples accumulated across all classifiers (indicates the hooks fire).

### Phase D deliverables

- `scoring.py`: `compute_instance_confidence(...)` function; `confidence_weights` config key with defaults.
- `review.py`: queue sorted by `conf_total` ascending; correction handlers write to `tests/fixtures/active_learning/<date>.jsonl`.
- `tests/regression/harness.py` extended: per-version dashboard, per-card diff, threshold-sweep mode.
- Retrain CLI: `python -m card_capture.train {fb_classifier,presence_classifier} --include-active-learning`.

### Phase D effort

- D1: 1 day.
- D2: 1.5 days.
- D3: 1.5 days.
- **Total: ~4 days.**

---

## 8. Phase E (gated): SAM 2 mask-driven pipeline

**Status.** Documented per user request; explicitly **gated** on Phase A–C metrics.

### Gate criteria

- After Phase C, manual visual review confirms median visible corner error remains a meaningful problem (no automated metric without ground-truth corners).
- AND: SAM 2 Hiera-T inference fits in the Mac performance budget (target: ≤ 200 ms per canonical frame).

If either fails, Phase E is skipped. The pipeline stays on the Phase A–D foundation.

### Architecture if we proceed

1. Stage 1 (presence) unchanged — still MobileNet from A1.
2. On a candidate window, run YOLO-OBB on the sharpest frame to get a seed box.
3. Prompt SAM 2 with the seed box → precise mask.
4. **SAM 2 video propagation:** propagate the mask backward and forward across the window using SAM 2's memory module. The masklet defines the session naturally — when SAM 2 loses the object, the session ends.
5. Replace ByteTrack within-window: mask propagation IS the tracking.
6. Fit min-area quad to the mask → corner coordinates. Replaces both YOLO-OBB-corners-as-truth and Phase C's RANSAC refinement.

### What gets deleted if Phase E ships

- ByteTrack integration from A2 (replaced by SAM 2 propagation within a session).
- Phase C2 classical corner refinement (replaced by mask → quad).
- Most hand-rolled session-boundary logic (the masklet ends when SAM 2 says it does).

### What stays

- A1 presence classifier (still cheaper than SAM 2 for the initial scan).
- A3 adaptive session gap (used as a sanity-check max-bound for masklet duration).
- Phases B, C1 (multi-frame fusion), D unchanged.

### Why gated

- ~30–80 ms SAM 2 inference per frame on M-series. Expensive if run broadly. Has to be canonical-windows-only.
- Benefits only justifiable if Phase A–C still leaves visible corner-localization problems.

### Phase E effort

~1–2 weeks if we proceed: SAM 2 setup, model weights management, propagation harness, fallback path when SAM 2 loses the object, deletion of replaced code.

---

## 9. Phase F (gated): Embedding-first identity

**Status.** Documented per user request; **gated** on Phase B + harness showing remaining temporal-identity errors.

### Gate criteria

- After Phase B, harness shows ≥ Z% of identity errors trace to temporal logic (lost track, split session, merged session) rather than embedding mismatches. Z to be set from baseline data; loose target ≥ 30%.

If most remaining errors are embedding-based or quality-based, Phase F is skipped — the rewrite would not pay off.

### Architecture if we proceed

Invert the data flow. Today: `time → session → track → instance → dedup`. Becomes:

1. Extract every "card-like" rectified crop above a quality bar (no temporal grouping yet).
2. Embed all crops with DINOv2 (already in B2).
3. Cluster in embedding space (HDBSCAN or graph-cut). **Each cluster IS a card.**
4. Temporal information becomes a *property* of each cluster, not the source of identity.
5. Front/back pairing: cards in the same temporal window with embedding distance high but co-occurrence high → likely two sides of the same physical card. Solve as a graph problem on (visual similarity, temporal co-occurrence).

### What gets deleted if Phase F ships

- Session boundary logic entirely (sessions become a downstream view, not an upstream constraint).
- Track-based F/B assignment (B1 classifier still needed; track-length tiebreaker goes away).
- Most of `pipeline.py`'s session-resolution code (~hundreds of lines).

### What stays

- A1 presence classifier.
- A2 ByteTrack OR Phase E SAM 2 (whichever shipped) — used to *help* clustering but not as the primary identity signal.
- B1 F/B classifier, B2 embeddings.
- Phases C, D unchanged.

### Why gated

- Largest rewrite in the spec.
- Pays off only if temporal logic is the dominant remaining failure mode after Phase B. The harness is the only honest way to know.

### Phase F effort

~2–3 weeks if we proceed: new clustering pipeline, new identity DB schema, migration of existing data, rewrite of `pipeline.py` orchestration.

---

## 10. Decision Gates Between Phases

| Gate | Phase | Criteria |
|---|---|---|
| **Spec 0 → A** | Bootstrap | Labeled corpus committed; baseline harness report exists. |
| **A → B** | Detection/Tracking | Card recall ≥ baseline + 20 pp; phantom rate ≤ baseline / 2; wall-clock not regressed > 30%. |
| **B → C** | Identity | F/B accuracy ≥ 99%; cross-video dedup F1 ≥ 0.95; no regression in A metrics. |
| **C → D** | Visual Quality | Sharpness score ≥ baseline + 15%; no regression in A or B metrics. |
| **D → E (gated)** | Confidence/Loop | ≥ 50 active-learning examples; review UI sorting confirmed useful. AND: manual review shows corner error remains a meaningful problem AND SAM 2 fits perf budget. |
| **D → F (gated)** | Confidence/Loop | ≥ 30% of remaining identity errors trace to temporal logic. |

Soft gates per user choice: phases run continuously, but the harness pauses for review at each gate so a phase can be pulled back if it regresses.

---

## 11. Risks & Open Questions

### Risks

- **Corpus too small.** < 5 videos limits statistical confidence in harness deltas. Mitigation: record 3 additional videos in priority modes (fast-flip, glare, occlusion) before declaring Spec 0 complete.
- **Training data scarcity.** F/B and presence classifiers need a few hundred labeled examples each. Bootstrap from existing pipeline output works but propagates current biases. Mitigation: seed with hand-curated examples from the labeled corpus.
- **Apple Silicon MPS quirks.** Some PyTorch ops fall back to CPU on MPS. Risk to performance budgets, especially for SAM 2 in Phase E. Mitigation: profile early in Phase A1 (MobileNet) to validate the MPS path; profile SAM 2 before committing to Phase E.
- **Model weight management.** New model weights (MobileNet × 2, DINOv2, possibly SAM 2) need a download/cache strategy. Mitigation: add a `setup` CLI subcommand that downloads required weights idempotently.
- **Storage migration.** Phase B2 adds nullable columns; backfill embeddings for existing data. Risk: long backfill on a large existing DB. Mitigation: backfill is lazy (compute on first read of an instance with NULL embedding).
- **Regression scope creep.** "While we're in here" temptation across 6 phases. Mitigation: each phase has its own implementation plan with frozen scope; refactoring lives in its own follow-up if warranted.

### Open questions

- **K for fusion (Phase C1):** default 5; revisit if quality plateaus.
- **Confidence weights (Phase D1):** defaults are guesses; calibrate via threshold-sweep harness once data exists.
- **BoT-SORT vs. ByteTrack final choice (Phase A2):** decided by Phase B harness data on ID switches through flips.
- **SAM 2 variant (Phase E):** Hiera-T vs. Hiera-S; decide after MPS profiling.

---

## 12. Effort & Sequencing Estimate

| Phase | Description | Effort |
|---|---|---|
| Spec 0 | Labeled corpus + harness | 2–3 days |
| A | Detection & tracking robustness | ~4 days |
| B | Identity & angle correctness | ~3.5 days |
| C | Visual quality | ~2.5 days |
| D | Confidence & closing the loop | ~4 days |
| **Subtotal Phases 0–D** | **Continuous work** | **~16 days (~3 weeks)** |
| E (gated) | SAM 2 mask-driven pipeline | +1–2 weeks |
| F (gated) | Embedding-first identity | +2–3 weeks |
| **Total if every gate opens** | | **~7 weeks** |

---

## Appendix: Mapping recommendations → phases

| Recommendation | Phase | Notes |
|---|---|---|
| Train binary F/B classifier | B1 | |
| Multi-frame median fusion | C1 | |
| Replace pHash with DINOv2 + FAISS | B2 | pHash kept as Tier-1 short-circuit |
| Replace HysteresisTracker with ByteTrack | A2 | BoT-SORT deferred to Phase B re-evaluation |
| Remove 30%-area-drop break rule | A4 | |
| Adaptive session gap | A3 | |
| Corner-refinement pass (RANSAC) | C2 | SAM 2 alternative in gated Phase E |
| Geometric filter on rectified aspect + convexity | C3 | |
| Replace pixel-stat presence with tiny visual classifier | A1 | MobileNet preferred over YOLO-Nano on Mac |
| Composite confidence per Card Instance | D1 | |
| Active-learning hooks | D2 | |
| A/B-able pipeline configs / regression set | Spec 0 + D3 | Minimal in Spec 0; full in D3 |
| SAM 2 architectural shift | E (gated) | |
| Embedding-first identity architectural shift | F (gated) | |
