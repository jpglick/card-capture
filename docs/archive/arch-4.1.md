# Card Capture — Pipeline Architecture v4.1

> Scope: extract clean, deduplicated 750×1050 stills of trading cards from a hand-held workspace video (cards held / placed in front of a roughly fixed camera). This document describes the v4.1 implementation as it lives in `src/card_capture/` and is exhaustive enough to be critiqued by a computer-vision practitioner. It is descriptive, not aspirational — every algorithm, threshold, and short-circuit named here exists in code.

## 0. Pipeline at a Glance

```
                ┌───────────────────────────────┐
   video.mov ──►│ Stage 1 — Adaptive Presence   │ presence windows,
                │ Sampler (two-pass, subprocess)│ valley splits,
                └──────────────┬────────────────┘ background proxies
                               │ FrameSample(full-res, t, idx)
                ┌──────────────▼────────────────┐
                │ Stage 2 — Frame Triage Filter │ drop empty frames
                │ (per-frame, rolling)          │
                └──────────────┬────────────────┘
                               │ FramePacket
                ┌──────────────▼────────────────┐
                │ Stage 3 — Corner Detector     │ 4-corner OBBs
                │ YOLOv8-OBB, batched, GPU      │ + confidence
                └──────────────┬────────────────┘
                               │ DetectionPacket (per detection)
                ┌──────────────▼────────────────┐
                │ Stage 4 — Quad-Novelty Gate   │ drop quads that
                │ vs. BackgroundModel           │ match the empty
                └──────────────┬────────────────┘ workspace
                               │ ScoredCandidate
                ┌──────────────▼────────────────┐
                │ Stage 5 — Session-Aware       │ BoT-SORT (default)
                │ Tracking (BoT-SORT/ByteTrack) │ + 4 reset signals
                └──────────────┬────────────────┘
                               │ TrackState (per card instance)
                ┌──────────────▼────────────────┐
                │ Stage 6 — Lazy GPU Refinement │ Kornia perspective
                │ (Kornia warp_perspective)     │ warp, 750×1050
                └──────────────┬────────────────┘
                               │ normalized BGR
                ┌──────────────▼────────────────┐
                │ Stage 7 — Per-Crop Quality    │ 7-component
                │ Scoring + Track Pruning       │ weighted score
                └──────────────┬────────────────┘
                               │ canonical_entries per track
                ┌──────────────▼────────────────┐
                │ Stage 8 — Front/Back Resolve  │ longest=Front,
                │ (per session)                 │ pHash-gated Back
                └──────────────┬────────────────┘
                               │ _PreparedTrack
                ┌──────────────▼────────────────┐
                │ Stage 9 — Lighting-Diverse    │ 3–4 views with
                │ Selection + Median Fusion     │ different glare
                └──────────────┬────────────────┘ → Fused view
                               │
                ┌──────────────▼────────────────┐
                │ Stage 10 — Global Dedup +     │ pHash inter-video
                │ Storage (SQLite)              │ + ReID cosine
                └───────────────────────────────┘
```

All stages are orchestrated by `VideoProcessor.process()` in `pipeline.py`. Stages 1–3 run in a dedicated producer subprocess (multiprocessing) with a bounded `frame_queue` and `detection_queue`; Stages 4+ run in the main process. This decouples ffmpeg decode and MPS/CUDA model loading from the orchestration loop.

---

## 1. Stage 1 — Adaptive Presence Sampler

**Module:** `sampler/__init__.py :: AdaptivePresenceSampler`.

The goal of Stage 1 is to convert a multi-minute video into a sparse list of frame indices that are "likely to contain a card", and to bound the downstream cost of full-resolution decode + YOLO inference. v4.1 replaces v3's brittle fixed-threshold triage (Laplacian variance, edge density z-scores) with a learned binary classifier as the primary signal.

### 1.1 Pass 1 — Fast scan

- Decode the entire video at `fast_scan_fps = 15.0` (default) with the configured reader backend (decord preferred; OpenCV+VideoToolbox fallback). The decoder is opened with `cv2.CAP_PROP_HW_ACCELERATION = VIDEO_ACCELERATION_ANY` on macOS and softly degrades to software decode (emits a `RuntimeWarning`).
- Each kept frame is resized to a long-edge of `scan_width = 192 px` (preserving aspect). All Pass-1 work happens on this 192-px proxy.
- For each proxy frame, GPU-batched (`gpu_utils.compute_presence_metrics_batched`):
  - Sobel-magnitude score (mean |∇I|).
  - "Empty-pixel" ratio (fraction of pixels ≤ 8/255).
  - Edge-density and variance.
- Also computed CPU-side, frame-to-frame:
  - `motion`: mean |Iₜ − Iₜ₋₁| of the grayscale proxy.
  - `delta_score`: mean absolute pixel diff of the BGR proxy vs. the previous frame.

Output: a list of `_AdaptiveScanFrame(frame_index, timestamp_ms, image, metrics, motion, delta_score)`.

### 1.2 Pass 2 — Per-frame presence scoring

Two paths exist:

1. **Learned classifier (default if `models/presence_classifier.pt` exists).**
   `presence/classifier.py :: PresenceClassifier` is a **MobileNetV3-Small** trained as a 2-class softmax head (`{empty, card_present}`). Input transform: `ToPILImage → Resize(224) → CenterCrop(224) → ToTensor → Normalize(ImageNet)`. Scoring is batched at `chunk_size = 32` proxy frames per forward pass. A frame is *active* iff `P(card_present) ≥ presence_threshold` (default `0.5`).

2. **Unsupervised fallback.** If no weights are found, the sampler combines the Pass-1 metrics into a composite z-score and applies an **Otsu threshold** on the score distribution. An additional edge-density activation (`edge_density > median + 2.5·MAD·1.4826`) is OR'ed in so that highly textured frames are never missed.

### 1.3 Presence-window assembly

`_build_windows` walks the active-flag array and emits `PresenceWindow(start_frame, end_frame, detection_methods)` whenever a contiguous run satisfies `min_presence_frames = 2`. Windows separated by ≤ `window_merge_gap = 3` *scan steps* of inactivity are merged.

A second mechanism — **forced splits** — short-circuits merging at *valley* frames inside an otherwise-active run, ensuring a fast hand-swap does not collapse two cards into one window.

### 1.4 Valley-split detection

`sampler/valley_splits.py :: find_valley_splits` produces a set of frame indices interpreted as "the card on screen probably changed here". Two orthogonal detectors run on the per-scan-frame `sobel_score` and `delta_score` arrays:

1. **Sobel valley.** Walk forward, tracking the running peak `P`. A frame enters a "valley" when its score drops below `P · (1 − valley_drop_ratio)`, default `0.40`. The valley must persist `≥ valley_min_width_frames = 3` scan frames before recovery to count. The split is placed at `argmin` inside the valley.
2. **Delta spike clustering.** Frames with `delta_score ≥ delta_spike_ratio · max(delta_score)` (default `0.50`) are clustered using a merge window of `max(5, 2·valley_min_width_frames)` scan indices. One split per cluster, placed at the cluster's peak.

The delta-spike branch is what catches the hand-blur frames as a card is physically swapped: a real swap produces a short burst of high pixel-difference frames; only one of them is emitted as a split.

### 1.5 Per-window selection (decoded at full res)

For each window, `_score_sharpness_in_window` re-ranks frames by `presence_score` and keeps the top-N (target ~24 frames per window, clamped to `[min_candidates_per_window, max_candidates_per_window] = [3, 48]`), *always* including the boundary frames so tracking continuity is preserved. Selected frame indices are then decoded at full resolution and yielded as `FrameSample(frame_index, timestamp_ms, image, w, h)`.

### 1.6 Background proxies

While walking the scan frames, the sampler maintains a heap of the **lowest-presence-score frames** (where `presence_score < _bg_safety_threshold`) up to `_max_bg_proxies` (default ≈ 30). These are sent downstream as `background_proxies` for the `BackgroundModel` (see §4).

### 1.7 Inter-window gap telemetry

The sampler stores `last_inter_window_gaps_frames` — for each pair of adjacent presence windows, the *frame-index* gap between them. This list is later used to compute an adaptive session-split threshold (§5.4) instead of using a fixed seconds-based value.

### 1.8 Where this can be critiqued

- Presence is decided on a 192-px center-cropped patch using ImageNet normalization. Cards near the frame edge or held nearly off-screen can be cut by the CenterCrop. There is no detection-conditioned ROI for the classifier.
- The classifier provides only a *frame-level* label; there is no localization signal. A bystander card peeking into a corner activates a window the same way a centered card does.
- Sobel-valley detection treats the entire frame's edge mass as a single scalar; a card placed against a textured surface (logo, hand) can suppress valleys.
- `valley_drop_ratio = 0.40` is global; for high-contrast cards the Sobel signal swings far more than 40%, so two cards in a row with no hand interlude can be missed.
- Window selection always includes the first/last frame of a window even if those are motion-blurred; this is intentional for tracking but biases the downstream candidate pool.

---

## 2. Stage 2 — Frame Triage Filter

**Module:** `ingestion.py :: FrameTriageFilter`.

A lightweight per-frame screen that runs in the producer subprocess between the sampler and the detector. v4.1 inherits v3's "wide funnel" stance — it does **not** gate on Laplacian variance or absolute blur. The only rejection criterion is:

```
empty_ratio = mean(gray ≤ empty_pixel_threshold)   # default 8 / 255
accept iff empty_ratio ≤ empty_ratio_threshold     # default 0.98
```

Per-frame metrics (`blur`, `variance`, `empty_ratio`) are attached to the `FramePacket` for telemetry only — they do not steer the pipeline. A `RollingWindowTriage` exists for backwards compatibility but is unused in the v4.1 default config.

**Critique surface:** the wide funnel relies on the corner detector and the quad-novelty gate to discard weak frames; if either is overconfident, garbage propagates.

---

## 3. Stage 3 — Corner Detection

**Module:** `detectors.py :: CardcaptorUltralyticsDetector` (config label: `"docaligner"`, but the implementation is an Ultralytics YOLO model — the original `docaligner` label has been carried through configs unchanged).

### 3.1 Model

- HuggingFace repo: `AlecKarfonta/cardcaptor-v3`, weight file `weights/cardcaptor_v3_best.pt`.
- Architecture: YOLOv8-OBB (Oriented Bounding Box). The model outputs `result.obb` containing `xyxyxyxy` (4-point polygons in arbitrary orientation), per-detection `conf`, and `cls`.
- Inference width: `detection_width = 640` (frames resized to this longest-edge for the model; corners scaled back to original resolution by `(scale_x, scale_y)`).
- Device resolution: `auto → mps → cuda → cpu` via `probe_torch_device_status`. The CLI prompts the user to confirm CPU fallback when neither MPS nor CUDA is available.

### 3.2 Batching

Producer/consumer multiprocess pattern in `_run_pipeline_workers`:

- Producer subprocess: opens video, runs `AdaptivePresenceSampler.sample()`, wraps each `FrameSample` in a `FramePacket`, enqueues into a bounded `frame_queue` (`queue_size = 256`).
- Consumer subprocess: pulls from `frame_queue`, accumulates a `batch` of size `inference_batch_size = 16`, calls `detector.detect_batch(batch, conf=corner_confidence_threshold)`, and emits `DetectionPacket` instances on `detection_queue`.

### 3.3 Postprocessing

`detect_batch` filters `obb.conf < corner_confidence_threshold` (default `0.5`) and discards polygons whose length ≠ 4 or whose area is degenerate. Surviving polygons are scaled to source-frame coordinates and attached to a `CornerDetection(corners, confidence, metadata)`.

### 3.4 Where this can be critiqued

- A single confidence threshold is global. Cards photographed near corners of the frame or partially occluded often score `0.3–0.5` and are dropped here, never reaching the novelty gate that would have validated them.
- YOLOv8-OBB minimizes a Smooth-L1 over angle but does not enforce trapezoidal projective consistency. Out-of-plane tilt produces non-rectangular corner sets that downstream `cv2.getPerspectiveTransform` will warp.
- No multi-instance NMS is applied beyond YOLO's internal NMS; overlapping cards in the same frame are handled by the spatial clusterer downstream (`CandidateSelector`), not here.

---

## 4. Stage 4 — Background Model & Per-Quad Novelty Gate

**Modules:** `presence/background_novelty.py`, called from `pipeline.py :: _filter_candidates_by_novelty` and `_prune_empty_workspace_tracks`.

The premise is workspace-agnostic but not model-agnostic: a *real card* must replace a rectangular patch of the empty workspace with something visually different. The acrylic stand, hinges, stickers on the table are by definition *not* novel relative to baseline.

### 4.1 BackgroundModel

```
gray_bg = mean over N "empty" frames of grayscale(frame)
```

Source of "empty" frames, in order of preference:
1. `background_proxies` collected by the sampler (lowest-presence frames).
2. `from_source_frame_paths(paths, n=30)` — the chronologically first 30 detection source frames after Stage 3, loaded as grayscale. Used by the main process when proxies aren't available across the multiprocessing boundary.

The model carries a single `float32` `(H, W)` array. If the deployed frame differs in resolution, the bg is resized to match.

### 4.2 Per-quad novelty score

```
mask  = polygon_mask((H, W), corners)           # binary fill of the quad
diff  = |gray(frame) − gray_bg|                  # float32, [0, 255]
novelty = clip(mean(diff[mask == 1]) / 255, 0, 1)
```

### 4.3 Gating

Two gating points use the same threshold (`0.08`, ≈ 20 grayscale levels of average difference):

- **Candidate-level prune (`_filter_candidates_by_novelty`)** — between corner detection and tracking. Drops candidates whose interior matches baseline. Source frames are loaded at most once per unique `image_path`.
- **Track-level prune (`_prune_empty_workspace_tracks`)** — after tracking and refinement. For each track, computes the *median* per-quad novelty across all the track's candidates; drops tracks below threshold. This catches tracks built entirely from a static prop (an empty hinge that happened to confuse YOLO).

### 4.4 Critique surface

- The background model is a *single mean*, not a per-pixel variance/Gaussian. A workspace with rotating lighting, breathing camera exposure, or autofocus pumping inflates baseline differences uniformly and softens the gate's selectivity.
- The threshold (`0.08`) is global; a card with a dominant-color back identical to the table will be borderline.
- The mask is rasterized via `cv2.fillPoly`, which uses 8-connected boundaries — near-degenerate quads (extreme oblique angles) get small masks and noisy means.

---

## 5. Stage 5 — Session-Aware Tracking

**Modules:** `tracking/bytetrack_adapter.py`, `tracking/botsort_adapter.py`, `tracking/centroid_jump.py`; orchestrated in `pipeline.py`.

The notion of a *session* is core to v4.1: one session = one card on the workspace from put-down to lift. Many cards per video; each session is later resolved into 1–2 tracks (Front, optional Back).

### 5.1 Tracker backends

- **BoT-SORT (default).** boxmot ≥ 0.17 (`BotSort`) or ≤ 0.16 (`BoTSORT`) via shimmed import. Configured with:
  - `track_high_thresh = tracker_t_high` (adaptive — see §5.3).
  - `track_buffer = lost_track_buffer = 30`.
  - `match_thresh = tracker_t_low`.
  - `cmc_method = None` — camera-motion compensation is disabled because the adapter feeds boxmot a dummy frame.
  - ReID weights: `osnet_x0_25_msmt17.pt` (OSNet small, trained on MSMT17). The adapter also exposes `pending_splits` — frame indices where a ReID identity shift was detected and a session reset is required.
- **ByteTrack** (fallback, via `supervision.ByteTrack`). No ReID, no `pending_splits`, no embedding-based dedup.

### 5.2 Inputs to the tracker

The tracker consumes `ScoredCandidate` objects (one per detection). For each frame, the adapter:
1. Converts each 4-corner polygon to an axis-aligned `xyxy` via `_xyxy_from_corners` (min/max envelope).
2. Builds a `supervision.Detections(xyxy, confidence, class_id=0)` batch.
3. Calls `tracker.update_with_detections(...)` (ByteTrack) or `tracker.update(det, dummy_img)` (BoT-SORT).
4. Maps each returned `tracker_id` to a stable `instance_id` (UUID-4 generated on first sight) stored in a `TrackState`.

A `TrackState` keeps `instance_id, candidates, last_centroid, last_frame_index, missed_frames, active, angle, reid_embedding`. Tracks shorter than `min_track_length` (adaptive — §5.3) are dropped at `finalize()`.

### 5.3 Adaptive tracker thresholds

After Stage 4, the pipeline computes:

```
tracker_t_high = clip(percentile(candidate.score.total, 65), 0.40, 0.75)
tracker_t_low  = max(0.20, tracker_t_high − 0.20)
adaptive_min_track_length = max(3, min(min_track_length, max(3, len(detection_rows) // 3)))
```

The `score.total` used here is the *detector confidence* (v4.1 wires the corner-detector confidence directly into `ScoredCandidate.score`; the multi-factor `QualityScore` enters only after refinement). The intent is to pull tracker thresholds toward the actual confidence distribution of *this video* rather than a fixed global value.

### 5.4 Session-reset signals

Within the per-frame tracking loop, four orthogonal signals can finalize the current session, reset the tracker, and bump `current_session_id`:

1. **Sampler frame-index gap.** Computed via `adaptive_gap.compute_session_gap_frames(inter_window_gaps)`:
   ```
   p50, p95 = percentiles of inter-window gap distribution
   recommended = clip(p95 + safety_pad_frames, floor=0.5s, cap=3.0s) in frames
   ```
   A reset fires when the gap between consecutive *sampler-emitted* frames exceeds `recommended`. This adapts to the user's actual cadence.

2. **Valley split.** A reset fires when any `valley_split_frame` (Stage 1.4) falls inside the current gap, even if the gap itself is short. This catches the rapidly-handed-off card.

3. **Centroid jump.** `tracking/centroid_jump.py :: CentroidJumpDetector` keeps a `deque[maxlen=jump_within_frames=3]` of recent bbox centroids. If `|cx − any_recent_cx| > jump_ratio · frame_width` (default `0.30 · width`), reset. Operates on the **highest-scoring** candidate per frame.

4. **BoT-SORT ReID shift.** When the adapter publishes a `pending_split` for the current frame (an existing spatial region was assigned a new `tracker_id` because the appearance changed), reset.

Each reset emits a `pipeline_event` to SQLite with `reason ∈ {sampled_frame_gap, valley_split, centroid_jump, reid_shift}`.

### 5.5 Critique surface

- BoT-SORT runs with `cmc_method=None` and is fed a *dummy* image for visual features when the adapter updates (`np.zeros((480, 640, 3))`). The ReID backbone runs on this dummy, so the `pending_splits` signal in practice degrades into a bbox-only matcher decorated with appearance noise. Genuine OSNet embeddings are only useful if the adapter is changed to pass real frames.
- The axis-aligned `xyxy` envelope discards orientation — two cards in the same frame at 90° to each other appear roughly co-spatial to ByteTrack.
- Centroid jump is computed on bbox-min/max centers, not on the rectified card center. A card rotated in-place produces a moving envelope center that can spuriously trigger.
- The adaptive `min_track_length` formula (`len(detection_rows) // 3`) is a stand-in for "expect on the order of N/3 detections per real card" and will collapse on videos where one card dominates a long stretch (the threshold becomes harder than necessary).

---

## 6. Stage 6 — Lazy GPU Refinement

**Module:** `gpu_refinement.py :: KorniaNormalizer` (CPU fallback: `cropper.py :: PrecisionNormalizer`).

Rather than warp every candidate, v4.1 defers warping until tracks are finalized. The pipeline collects the union of `(frame_index, candidate)` tuples needed for canonical view selection per track, decodes only those frames at full resolution, then runs a batched perspective warp.

### 6.1 Algorithm

For each `(image, corners)`:
1. Order corners clockwise (`cropper.order_points_clockwise` — uses sum/diff of x±y to find TL/BR/TR/BL).
2. Re-orient for the target canvas (`_orient_for_target_canvas`): roll the polygon so the longest edge aligns vertically; the destination is *always* portrait `750×1050`.
3. Compute the 3×3 perspective matrix `M = cv2.getPerspectiveTransform(src=oriented, dst=[0,0,W,0,W,H,0,H])`.
4. Stack inputs and matrices into a batched `torch.Tensor`, push to the resolved device, call `kornia.geometry.transform.warp_perspective(batch, batch_M, (H, W))`.
5. Convert back to BGR `uint8`; optionally `cv2.rotate(_, ROTATE_180)` if `rotate_180 = True` (default).

### 6.2 Critique surface

- The destination size is fixed at `750×1050`. Standard card aspect is 63.5 × 88.9 mm ≈ 0.714; the canvas is `0.7143…` — close, but the rectified crop is always interpolated to a fixed grid rather than the source resolution, which Kornia's default bilinear filter mildly softens. There is no Lanczos option in the GPU path (the CPU path uses `INTER_LANCZOS4`).
- The 180° rotation is a global flag based on prior user setup (camera mounted upside-down). There is no automatic up-orientation from card-content cues.
- The "safety margin" feature (`safety_margin = 0.015`) of the CPU `PrecisionNormalizer` crops 1.5% on each side and resizes back; the Kornia path has no such margin.

---

## 7. Stage 7 — Quality Scoring + Per-Track Pruning

**Module:** `scoring.py :: QualityScorer`.

Each rectified crop is scored:

```
sharpness    = clamp(var(Laplacian(gray)) / 1000)
glare        = penalty against fraction-of-pixels > 240 (function in fuser.py)
aspect_ratio = clamp(1 − |actual − 63.5/88.9| / 0.25)
size         = clamp((H·W) / target_pixels)         target = 600·900
complexity   = clamp(std(gray) / 80)
border_purity = _border_purity_score(gray)
confidence   = clamp(detection_confidence)

total = 0.25·sharpness + 0.15·glare + 0.15·aspect_ratio + 0.10·size
      + 0.10·complexity + 0.20·border_purity + 0.05·confidence
```

### 7.1 Border purity (new in v4.1)

`_border_purity_score` computes the std-dev of pixel values in the outer 3% ring of the rectified crop and compares it to the interior std-dev:

```
ring_std     = std(gray[outer 3% ring])
interior_std = std(gray[interior])
purity = clamp((interior_std − ring_std) / interior_std)
```

A clean trading-card border (white or black uniform band) has low ring variance. A finger/hand intruding into the rectified crop spikes the ring's variance. The 20% weight is deliberately large because border occlusion is the failure mode most reviewers flag.

### 7.2 Per-track pruning

After scoring, `_prune_empty_workspace_tracks(prepared_tracks, bg, threshold=0.08)` recomputes median quad-novelty per track (using rectified-source frames) and removes the track if `< 0.08`. This is the second quad-novelty gate; the first (Stage 4) was at the candidate level.

### 7.3 Critique surface

- Glare is computed as a fraction of saturated pixels with no spatial structure — a small but bright specular highlight on glossy cards is penalized identically to a large blown-out region.
- The aspect-ratio tolerance is `±0.25` of the ideal — that is a 25% deviation, very loose. A YOLO false-positive on a billfold or game card frequently passes.
- All weights are hand-tuned; there is no calibration set tying these to human-reviewer preferences.
- `target_pixels = 600·900 = 540000`; rectified crops are larger than that by design so the `size` term saturates at 1.0 immediately for any in-distribution capture, contributing no information.

---

## 8. Stage 8 — Front/Back Resolution Per Session

**Module:** `pipeline.py :: _resolve_session_tracks`.

After Stage 7 each track has a `_PreparedTrack` carrying its session id, candidates, an `appearance_vector` (mean-pooled, contrast-normalized inner crop), a `side_score` (the `complexity` term — high for textured fronts, low for plain backs), a `primary_hash` (DCT pHash of best canonical crop), and `candidate_hashes` (pHash plus its 180°-rotated companion for each candidate).

For each session id:
1. Sort the session's prepared tracks by `len(track.candidates)` descending.
2. The longest track is unconditionally `angle = "Front"`.
3. Each remaining track is promoted to `"Back"` *only if* `min_hash_distance(other_hashes, front_hashes) ≤ _SAME_CARD_HAMMING_MAX = 22` (on a 64-bit pHash, 22 bits ≈ 34% disagreement — empirically wider than the inter-card threshold).
4. Tracks that fail the pHash gate keep `angle = "Front"` and are treated as **different cards** in the same session — i.e. the session split signals missed a swap; we recover the two-card outcome here.

### 8.1 Critique surface

- pHash is rotation-sensitive; the system mitigates this by hashing both `normalized` and `rotate_180(normalized)` and taking the minimum distance. There is *not* a flip-axis variant (front-vs-back of the same card has different content but identical contour), which is the very case this stage is supposed to detect — so the rule "if it looks similar to the front, call it back" is dimensionally wrong. In practice the looseness of the threshold (22/64) is what makes it work, not the metric.
- The `side_score` (texture complexity) is computed but not used to *prefer* one assignment over another — it is stored only.
- Track ordering by length is a heuristic; a short, sharp front-view loses to a long, blurry back-view.

---

## 9. Stage 9 — Lighting-Diverse Selection + Median Fusion

**Module:** `fuser.py :: MultiFrameFuser`, invoked per track in `pipeline._select_canonical_entries`.

### 9.1 Selection

`select_lighting_diverse_indices`:
1. For each rectified candidate compute the centroid of the bright-pixel mask (`gray > 200` threshold).
2. Bin candidates into 4 quadrants of the 750×1050 canvas by glare-centroid position.
3. Per quadrant, pick the **sharpest** candidate (Laplacian variance).
4. If fewer than 3 quadrants are populated, top up with the sharpest unselected candidates until either 4 are picked or the track is exhausted.

`_CANONICAL_TARGET_FRAMES = 3`, `_CANONICAL_MAX_FRAMES = 4`.

### 9.2 Fusion

Selected frames share the same 750×1050 grid (a precondition the Kornia warp guarantees). The fused image is:

```
fused = uint8(median(stack(selected_frames), axis=0))
```

Median is preferred over min/mean: glare is an additive outlier in only 1–2 of N frames, so the per-pixel median rejects it without darkening the underlying card. With only 2 frames the median degenerates to mean.

### 9.3 Critique surface

- `gray > 200` is a hard threshold; on holographic / refractor cards the entire surface lights up and the centroid is meaningless. There is no foil-aware fallback.
- The quadrant binning is on the *centroid of all bright pixels*, not on individual specular blobs — a card with two highlights produces a misleading centroid in the middle.
- Median fusion preserves stationary glare across all selected frames. If lighting is fixed and the card never moves between selected views, the glare survives the median.
- The fusion presumes pixel-perfect alignment from Kornia. In practice the corner detector wobbles by 1–3 px frame-to-frame; the median introduces ~1 px ghosting at high-contrast edges.

---

## 10. Stage 10 — Global Deduplication + Storage

**Modules:** `deduplicator.py :: VisualDeduplicator`, `storage.py :: Storage`.

### 10.1 Perceptual hash

`compute_phash`:
1. Crop the inner 60% (margin = 20% each side) of the rectified crop.
2. Grayscale.
3. Resize to 32×32 with `INTER_AREA`.
4. Compute 32×32 DCT, keep the top-left 8×8 low-frequency block.
5. Threshold each coefficient against the **median** of the block.
6. Pack into a 64-bit unsigned integer rendered as 16-hex-char string.

Hamming distance is the popcount of XOR. The default within-canonical-set duplicate threshold is `≤ 6 / 64` (≈ 9%).

### 10.2 ReID embedding (BoT-SORT only)

`is_reid_duplicate(emb_a, emb_b, threshold=0.15)`: cosine distance over the OSNet embedding; ≤ 0.15 ⇒ same card. Used as a secondary dedup when the same physical card appears in two sessions of the same video that the gap-split rule didn't merge.

### 10.3 Storage

SQLite `cards.sqlite` schema (`storage.py`) is the source of truth. Per session/track persisted:

- The rectified canonical entries (each with its pHash, glare mask compressed via zlib, Laplacian map).
- The `Fused_Canonical_View` (derived; never overwrites raw evidence).
- `pipeline_events` (every session reset with reason).
- Telemetry: `run_telemetry.json` (timings, stage counts) and `tracker_association_events.json` (every `new_track` / `assigned_existing` action).

### 10.4 Critique surface

- The pHash is computed on the inner 60% — robust to border variations but throws away the *most discriminative* feature on many trading cards: the player photo crop and team logo near the top/bottom.
- `INTER_AREA` is suboptimal for DCT-based hashing because it doesn't anti-alias the same way as low-pass filtering. The 32→8 reduction via DCT is doing that work anyway, but the input downsample can introduce moiré.
- Threshold-against-median produces hashes with exactly 32 set bits; this is fine for Hamming but makes the hash less "perceptual" than the classical mean-threshold variant near the extremes.
- ReID dedup operates on a *dummy*-image-driven embedding (see §5.5), so its discriminative power is well below what OSNet on real crops would give. The 0.15 threshold was chosen empirically against the current degraded embeddings.

---

## 11. Data Model Reference

| Type | Fields | Lifetime |
|---|---|---|
| `FrameSample` | `frame_index, timestamp_ms, image, w, h` | sampler → producer |
| `FramePacket` | `+ triage_metrics, telemetry` | producer → consumer |
| `CornerDetection` | `corners(4·Point), confidence, metadata` | inside `DetectionPacket` |
| `DetectionPacket` | `frame_index, timestamp_ms, w, h, corner_detection, telemetry` | consumer → main |
| `ScoredCandidate` | `detection_id, timestamp_ms, image_path, score:QualityScore, corners, frame_index` | candidate building → tracker |
| `TrackState` | `instance_id, candidates[], last_centroid, last_frame_index, missed_frames, active, angle, reid_embedding` | tracker → resolver |
| `_PreparedTrack` | `track, session_id, first_frame_index, angle, frame_entries[], canonical_entries[], candidate_hashes[], primary_hash, side_score, appearance_vector, canonical_detection_ids` | resolver → storage |
| `BackgroundModel` | `gray: float32 (H, W)` | per-video |
| `PresenceWindow` | `start_frame, end_frame, frame_candidates[(idx, score)], detection_methods` | sampler |
| `GapDistribution` | `p50, p95, recommended_gap_frames` | once per video |
| `QualityScore` | `total, components{sharpness, glare, aspect_ratio, size, complexity, border_purity, confidence}` | per crop |

---

## 12. Configuration Surface (defaults)

From `config.py :: PipelineConfig`:

```
detector                    = "docaligner"   # → CardcaptorUltralyticsDetector (YOLOv8-OBB)
reader_backend              = "auto"         # decord if present, else OpenCV/VideoToolbox
queue_size                  = 256
inference_batch_size        = 16
corner_confidence           = 0.5
blur_threshold              = 30.0           # telemetry only
variance_threshold          = 20.0           # telemetry only
empty_pixel_threshold       = 0.98
detection_width             = 640
device                      = "auto"         # auto → mps → cuda → cpu
group_gap_ms                = 300            # selector temporal cluster gap
spatial_variance_threshold  = 150.0          # selector centroid split
min_track_length            = 12             # pre-adaptive ceiling
telemetry_scope             = "canonical"
triage_keep_percentile      = 0.05
background_frames           = 30
null_patience_frames        = 6
background_threshold        = 15.0           # NullStateDetector (legacy)
rotate_180                  = False
tracker_backend             = "botsort"
fast_scan_fps               = 15.0
confirm_scan_fps            = 5.0
valley_drop_ratio           = 0.40
valley_min_width_frames     = 3
delta_spike_ratio           = 0.50
centroid_jump_ratio         = 0.30
centroid_jump_frames        = 3
reid_distance_threshold     = 0.6
```

Hard-coded thresholds in `pipeline.py` that materially affect behavior:

```
_CANONICAL_TARGET_FRAMES         = 3
_CANONICAL_MAX_FRAMES            = 4
_SAME_APPEARANCE_HAMMING_MAX     = 8     # ReID-derived near-dup
_SESSION_DUPLICATE_HAMMING_MAX   = 6     # within-session pHash dedup
_SESSION_TEXTINESS_MARGIN        = 0.03
_SESSION_APPEARANCE_SIMILARITY_MIN = 0.995
_SESSION_MERGE_SIMILARITY_MIN    = 0.99
_SAME_CARD_HAMMING_MAX           = 22    # Front/Back gate, 64-bit pHash
NoveltyGate threshold            = 0.08
```

---

## 13. Telemetry & Diagnostics

Output directory layout per run:

```
<output_dir>/
  frames/             persisted source frames (every detection's frame, once)
  crops/              per-card canonical and fused crops
  run_telemetry.json  stage timings, counts, sampler stats
  tracker_association_events.json   every new_track / assigned_existing event
  cards.sqlite        all CardInstances, pipeline_events, video_status
```

Sub-commands for offline inspection:

- `card-capture sampler sessions <video>` — runs only Stage 1, prints predicted sessions and gap statistics; ~35 s vs. 2+ min for the full pipeline.
- `card-capture harness run` — golden-corpus regression report with deltas vs. a baseline JSON.

---

## 14. End-to-End Critique Summary

Practitioners reviewing v4.1 should consider:

1. **Coupling between session-split signals and tracker thresholds is asymmetric.** Adaptive `tracker_t_high` lifts the bar for new-track activation but never lowers it; on low-confidence videos the tracker can drop legitimate cards before any session signal fires.
2. **The pHash-based Front/Back gate uses the wrong invariant.** A front and a back of the same card share a contour but have entirely different content; the system tolerates 34% bit disagreement to compensate, which simultaneously admits genuinely different cards.
3. **BoT-SORT ReID is effectively disabled** because the adapter feeds a zeroed image to the appearance branch. The `reid_distance_threshold = 0.6` and the `pending_splits` signal both run on degraded embeddings.
4. **The quality score is a hand-weighted sum** with no calibration or learned ranker; the `size` term is saturating in-distribution and contributes ~no signal, and the `glare` term ignores spatial structure.
5. **The background model is a single mean.** A per-pixel variance / running average would catch lighting drift; a single mean conflates "moved" with "differently lit".
6. **The sampler's classifier is frame-level, not detection-conditioned.** A localization-aware presence head (e.g. an objectness map crop) would reduce both phantoms and the false-negative cases where a card peeks in from an edge.
7. **Multi-process boundaries hide failures.** The producer subprocess loads MPS tensors lazily; if it crashes silently, the consumer drains and the run reports `no_detections` with no surfaced cause. Both queues have retry/backoff but no structured error code.
8. **Fixed 750×1050 canvas + GPU bilinear** loses sub-pixel content; for grading-style downstream tasks a higher-resolution canvas with Lanczos would be defensible.
