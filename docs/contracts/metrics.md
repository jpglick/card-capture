# v4 Metric Definitions

**Status:** Frozen (Wave 1 sign-off)
**Owned by:** Surface D
**Consumed by:** Surface B (regression tab), Surface C (algorithmic gate)

All metrics are pure functions over `(cards.sqlite, truth.json)`. No pipeline
re-run is required to recompute them. Each metric is independently computable;
the harness runner aggregates them into a single report.

---

## Overview

| Metric | Function | Output range | Gate threshold |
|---|---|---|---|
| `card_recall` | Detected GT cards / total GT cards | [0, 1] or null | ≥ 0.90 |
| `card_precision` | Real detections / total detections | [0, 1] or null | ≥ 0.85 |
| `side_accuracy` | Correct F/B assignments / matched instances | [0, 1] or null | ≥ 0.80 |
| `dedup_accuracy` | ARI of predicted vs. GT dedup clusters | [-1, 1] or null | ARI ≥ 0.70 |
| `image_quality` | Mean SSIM of fused canonical vs. reference | [0, 1] | reported only |

Gate thresholds are initial values, recalibrated after 15 labeled videos are
in the golden set.

---

## card_recall

**Definition:** Matched ground-truth cards / total ground-truth cards.

```
card_recall = |{GT cards matched to a detection}| / |{GT cards}|
```

**Inputs:**
- `cards.sqlite` — `card_instances` and `card_views` tables from a pipeline run.
- `truth.json` — `expected_cards[]` array.
- `video_id` — used to scope the sqlite query to one source video.

**Matching rule:** A detection matches a GT card iff:
1. If `approx_front_window_ms` / `approx_back_window_ms` is present in the GT
   entry: the detection's temporal extent (its `start_ms`–`end_ms` in
   `card_views`) overlaps the GT window AND its assigned `side` matches
   (`front_present=true` → Front, `back_present=true` → Back).
2. If no time window is provided: detections are matched to GT cards by
   detection order within the video (first detection → first GT card, etc.).

Matching is 1-to-1 (Hungarian algorithm or greedy-by-overlap-area) to avoid
double-counting.

**Output range:** [0.0, 1.0]

**Edge cases:**
- 0 GT cards → recall is **undefined**; report as `null`, not `1.0`.
- 0 detections + GT cards present → recall = `0.0`.

**Worked example:**
- GT: 3 cards (card_01, card_02, card_03).
- Detections: card_01 matched, card_02 matched, card_03 missed.
- `card_recall = 2 / 3 ≈ 0.667`.

**Noise floor (initial):** ± 0.01 absolute.

---

## card_precision

**Definition:** Real detections / total detections (phantom rate complement).

```
card_precision = |{detections matched to a GT card}| / |{all detections}|
```

A detection is "real" if it matches a GT card. A detection is "phantom" if it
does not match any GT card (either a false positive or an artefact).

**Inputs:** Same as `card_recall`.

**Matching rule:** Same matching algorithm as `card_recall`; the matched set
is shared between the two metrics.

**Output range:** [0.0, 1.0]

**Edge cases:**
- 0 detections → precision is **undefined**; report as `null`.
- 0 GT cards + detections present → precision = `0.0` (all detections are
  phantoms).

**Worked example:**
- GT: 2 cards.
- Detections: 3 total (2 matched, 1 phantom).
- `card_precision = 2 / 3 ≈ 0.667`.

**Noise floor (initial):** ± 0.01 absolute.

---

## side_accuracy

**Definition:** Correct front/back assignments / total assigned instances
(computed only over detections matched to a GT card).

```
side_accuracy = |{matched detections with correct side}| / |{matched detections}|
```

A GT card with `front_present=true` and `back_present=true` contributes two
matchable instances (one Front, one Back). A detection of that card is correct
if its assigned `side` aligns with the GT.

**Inputs:** Same as `card_recall`.

**Output range:** [0.0, 1.0]

**Edge cases:**
- 0 matched detections → **undefined**; report as `null`.

**Worked example:**
- 4 matched instances: Front/Front/Back/Back in GT.
- Pipeline assigned: Front/Back/Back/Front (2 correct, 2 swapped).
- `side_accuracy = 2 / 4 = 0.50`.

**Noise floor (initial):** ± 0.02 absolute.

---

## dedup_accuracy

**Definition:** Adjusted Rand Index (ARI) of predicted dedup clusters vs.
ground-truth clusters.

```
dedup_accuracy = ARI(predicted_labels, gt_labels)
```

Ground-truth clusters are defined by `physical_card_key`: all detections
matched to GT cards sharing the same `physical_card_key` belong to the same
cluster. Predicted clusters come from the pipeline's dedup output (currently
`dedup_groups` table; v4 successor TBD).

The metric is computed per-video by default; with `--cross-video` it uses GT
cards across all labeled videos.

**Secondary metric:** Pair F1 (F1 over all pairs: whether two detections are
in the same cluster). Reported alongside ARI for diagnostic purposes. The gate
uses ARI only.

**Inputs:**
- `cards.sqlite` — dedup cluster assignments.
- `truth.json` — `physical_card_key` per expected card.
- `video_id` (or all videos if `--cross-video`).

**Output range:** ARI ∈ [-1, 1]; pair F1 ∈ [0, 1]. Both reported as `null`
if fewer than 2 matched detections exist (ARI undefined on trivial clusters).

**Edge cases:**
- Fewer than 2 matched detections → **undefined**; report as `null`.
- All GT cards have distinct `physical_card_key` (no true duplicates): ARI
  should be 1.0 if the pipeline produces singleton clusters.

**Worked example:**
- GT: card_A, card_A (same card, two videos) + card_B.
- GT clusters: {card_A_v1, card_A_v2}, {card_B_v1}.
- Predicted clusters: {card_A_v1}, {card_A_v2, card_B_v1} (wrong grouping).
- ARI < 0 (worse than random); pair F1 = 0.0.

**Noise floor (initial):** ± 0.02 absolute (ARI).

---

## image_quality

**Definition:** Mean SSIM of the pipeline's fused canonical image vs. a
hand-picked reference frame per GT card.

```
image_quality = mean(SSIM(fused_canonical_i, reference_frame_i))
               for all matched cards i with a reference frame
```

PSNR is also reported for sanity but is not a gate metric.

**Inputs:**
- Fused canonical images from `cards.sqlite` (path stored in `card_views`).
- Reference frames stored under
  `golden_set/videos/<video_id>/reference_frames/<card_id>.png`.
- `truth.json` — to join GT cards to matched detections.

**Output range:** Mean SSIM ∈ [0, 1]. Also reports `coverage_pct` (% of GT
cards that have a reference frame available).

**Edge cases:**
- No reference frames exist → **undefined** SSIM; report as `null` with
  `coverage_pct = 0`.
- Mismatched image dimensions: resize reference to match fused canonical
  before SSIM computation (log a warning).

**Noise floor (initial):** ± 0.01 absolute (SSIM).

---

## Noise floor (initial; recalibrated after 15-video golden set)

| Metric | Initial noise floor |
|---|---|
| card_recall | ± 0.01 absolute |
| card_precision | ± 0.01 absolute |
| side_accuracy | ± 0.02 absolute |
| dedup_accuracy (ARI) | ± 0.02 absolute |
| image_quality (SSIM) | ± 0.01 absolute |

A metric delta smaller than the noise floor is not considered a regression or
improvement. The harness displays these bounds in the Regression tab.

---

## Regression gate policy

A pipeline change may merge only if the harness reports, on the full golden
set:

1. No metric is more than 1× the noise floor *below* the current baseline
   (`baseline_v4.1` or the most recent promoted baseline).
2. OR the change explicitly targets a metric, in which case that metric must
   improve by ≥ 1× the noise floor and no other metric regresses by more than
   1× its noise floor.

Exceptions (e.g. deliberate trade-off accepted by Surface D owner) must be
recorded in the PR as a `regression-exception` label with justification.

---

## Implementation notes

- All metric functions live in `harness/metrics/`.
- The shared matching logic (`harness.match.match_detections_to_truth`) is
  called by `card_recall`, `card_precision`, and `side_accuracy` to ensure
  consistent pair assignment.
- `dedup_accuracy` uses `sklearn.metrics.adjusted_rand_score`.
- `image_quality` uses `skimage.metrics.structural_similarity`.
- All functions accept `(db_path: Path, truth_path: Path, video_id: str)`
  keyword arguments and return `float | None`.
