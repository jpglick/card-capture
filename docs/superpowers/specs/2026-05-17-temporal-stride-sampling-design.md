# Duration-Proportional Temporal Stride Sampling

**Date:** 2026-05-17  
**Goal:** Reduce frames sent to YOLO by 3-4× with no quality regression, by replacing presence-score-ranked top-N selection with duration-aware temporal bucketing.

---

## Problem

The `AdaptivePresenceSampler._score_sharpness_in_window` currently selects candidates by sorting all window records by `presence_score` and taking the top-N (capped at `max_candidates_per_window = 48`). Every presence window in practice hits this cap, resulting in dense clusters of near-identical frames rather than temporally distributed coverage.

**Last observed baseline (IMG_5872_real, ~37s video, 60fps source):**
- 555 scan frames → 289 selected → 335 YOLO detections → 7 saved cards
- ~41 YOLO frames per saved card (need ~10-15)
- All 6 presence windows hit the 48-frame cap
- Track lengths: [48, 20, 38, 8, 48, 12, 5, 34, 13, 40, 18, 43] — clustered, not spread

The tracker and fusion stage only need enough frames to establish identity and pick 4 quality crops. Sending 41 redundant near-identical frames per card wastes YOLO time and slows the entire detect step.

---

## Design

### Algorithm Change

**File:** `src/card_capture/sampler/__init__.py`  
**Method:** `AdaptivePresenceSampler._score_sharpness_in_window`

**Current behavior:**
```python
sorted_records = sorted(records, key=lambda r: r.presence_score, reverse=True)
window.frame_candidates = [(r.frame_index, r.presence_score) for r in sorted_records[:target]]
```

**New behavior:**
1. Compute target frame count from window duration:
   ```python
   duration_s = (window.end_frame - window.start_frame) / self.last_source_fps
   target = math.ceil(duration_s * self.target_yolo_fps)
   target = max(self.min_candidates_per_window, min(self.max_candidates_per_window, target))
   ```
2. Divide window records (sorted by frame_index) into `target` equal temporal buckets.
3. From each bucket, pick the record with the highest `presence_score`.
4. Return selected frames in frame_index order.

**Result at 3fps:** a 2s hold → 6 frames, a 6s hold → 18 frames, all capped at 24.

### Edge Cases

| Situation | Handling |
|---|---|
| Window has fewer records than target buckets | One frame per record; no duplication |
| Very short card flash (< 1s) | `min_candidates_per_window = 3` floor applies |
| `last_source_fps` unset (tests/edge) | Fall back to current top-N-by-score behavior |
| Single-frame bucket | That frame is selected (no choice needed) |

### Configuration Changes

**File:** `src/card_capture/config.py`

| Field | Old default | New default | Notes |
|---|---|---|---|
| `target_yolo_fps` | *(new)* | `3.0` | Frames per second of card presence to send to YOLO |
| `max_candidates_per_window` | `48` | `24` | Hard cap; temporal stride makes this generous |

`target_yolo_fps` is the primary speed/quality knob: lower for speed, higher if fast card flips need more coverage. `max_candidates_per_window` remains a safety cap for very long holds.

### Wiring

**`pipeline/steps/start.py`:** Add `target_yolo_fps` to `RunContext` dataclass and populate it from `cfg.target_yolo_fps`.

**`src/card_capture/sampler/__init__.py` `__init__`:** Accept `target_yolo_fps: float = 3.0` parameter, store as `self.target_yolo_fps`.

**`pipeline/steps/detect.py` `_build_sampler_detector`:** Pass `target_yolo_fps` from `ctx` to `AdaptivePresenceSampler`.

### Telemetry

Add `target_yolo_fps` to the `sampler_telemetry` dict so it appears in `run_telemetry.json` alongside `last_selected_frame_count`. This makes it easy to correlate the knob value against frame counts across runs.

---

## Files Touched

| File | Change |
|---|---|
| `src/card_capture/sampler/__init__.py` | `_score_sharpness_in_window` algorithm + `__init__` signature |
| `src/card_capture/config.py` | Add `target_yolo_fps: float = 3.0`, lower `max_candidates_per_window` default to 24 |
| `pipeline/steps/start.py` | Add `target_yolo_fps` to `RunContext`, populate from config |
| `pipeline/steps/detect.py` | Pass `target_yolo_fps` to `AdaptivePresenceSampler` constructor |

---

## Expected Impact

| Metric | Before | After (est.) |
|---|---|---|
| YOLO frames (37s video, 7 cards) | ~289 | ~70-100 |
| Frames per saved card | ~41 | ~10-14 |
| Detect step wall time | baseline | ~3-4× faster |
| Temporal track coverage | clustered | spread across hold duration |

Quality unchanged: `min_track_length = 6` still enforced by tracker; fusion still selects best 4 from whatever YOLO found. Temporal spread may actually improve track stability on fast flips by giving the tracker better temporal signal.

---

## Testing

- Existing sampler unit tests pass unchanged (output shape is the same).
- New unit test for `_score_sharpness_in_window`: given 100 records spanning 10 source seconds at `target_yolo_fps=3`, assert output has ~10 frames with `frame_index` values spread across the full range (not clustered).
- Regression: run on `IMG_5872_real` and confirm `last_selected_frame_count` drops to ~70-100 and `saved_instances` is unchanged at 7.
