# Frame Quality Improvements Design

**Date:** 2026-05-17
**Goal:** Two complementary improvements to how frames are selected for detection and output — ensuring the opening seconds of video are always scanned (catching cards resting on stands before presentation begins), and replacing temporal-stride frame selection with a cheap in-track Laplacian quality scan (giving the warp/fusion step the sharpest available frame rather than a random sparse one).

---

## Feature A: Opening Scan — First Frames Always to YOLO

### Problem

Cards resting on an acrylic stand at the *start* of a video are silently missed. The fast scan (Pass 1) scores each 192px frame for presence; a static card lying motionless on a stand scores below the presence threshold because it looks like stable background. No presence window is formed, and YOLO never sees those frames.

### Solution

In `AdaptivePresenceSampler.sample()`, after computing `deduped_frame_indices` from presence windows, union in a sparse sample of the **first `opening_scan_s` seconds** of source video. These frames bypass the presence gate entirely.

### Mechanics

```python
# Compute opening frames: sparse sample of the first opening_scan_s seconds,
# at the same density as target_yolo_fps.
opening_count = int(source_fps * opening_scan_s)
stride = max(1, int(source_fps / target_yolo_fps))
opening_indices = list(range(0, opening_count, stride))

# Union into existing set — dedup ensures no frame is decoded twice.
deduped_frame_indices = sorted(set(deduped_frame_indices) | set(opening_indices))
```

For a 60fps source at `target_yolo_fps=3` over `opening_scan_s=2.0` this adds frames [0, 20, 40, 60, 80, 100] — 6 frames. These go through triage → YOLO → tracking unchanged. If the card is visible, a track is formed. If the stand is empty, YOLO returns nothing and cost is negligible.

**Code comment requirement:** The block that computes `opening_indices` must include a multi-line comment explaining the WHY clearly:

> "Cards resting on a stand at the start of the video may not register as 'card present' in the fast scan because they are static and score below the presence threshold — the scanner treats them as stable background. We unconditionally include sparse frames from the opening window so that a card placed before filming started is never silently skipped. This is distinct from the presence-gated windows: these frames are always sent to YOLO regardless of presence score."

### New config

| Field | Default | Location |
|---|---|---|
| `opening_scan_s: float` | `2.0` | `PipelineConfig`, `RunContext` |

### Files changed

| File | Change |
|---|---|
| `src/card_capture/config.py` | Add `opening_scan_s: float = 2.0` |
| `pipeline/steps/start.py` | Add to `RunContext`; wire `opening_scan_s=cfg.opening_scan_s` in `init_run` |
| `src/card_capture/sampler/__init__.py` | Add opening-index union in `AdaptivePresenceSampler.sample()` |

---

## Feature B: In-Track Laplacian Quality Scan

### Problem

Temporal-stride selection picks frames **before YOLO runs** at 3fps, giving 4–8 frames per track. The refine step then picks the best from those 4–8 — but if none of them happened to be the sharpest moment of the hold (hand motion, blink of blur between strides), the output crop is softer than necessary. The video contains sharper frames that were never decoded.

### Solution

After tracking confirms a card's time range, do a **cheap Laplacian sharpness scan** of the video within that range at a dense stride. Replace the temporal-stride candidates with the sharpest frames found. Corners for non-YOLO frames are borrowed from the nearest detected frame.

### Mechanics

**Single video pass across all tracks:**

For performance, all tracks are scanned in one forward pass:

1. Collect `(first_frame, last_frame)` for every confirmed track, sorted by `first_frame`
2. Open video capture **once**
3. Scan forward; for each frame within any track's range at `laplacian_scan_stride`, compute Laplacian variance on a **640px-wide downscaled gray frame** (~1ms each)
4. Store per-track sharpness results
5. Close capture — the existing canonical decode loop follows immediately in the same direction

Cost estimate: 10 tracks × 30 frames (2s @ 15fps scan) × 1ms = ~300ms total. No extra seeks.

**Frame selection:**

```python
def _laplacian_select_frames(
    video_path: Path,
    track_ranges: list[tuple[int, int, str, list[tuple[int, list]]]],
    # (first_frame, last_frame, instance_id, [(det_frame_idx, corners), ...])
    scan_stride: int,
    top_k: int,
    max_corner_gap: int,
) -> dict[str, list[tuple[int, list]]]:
    # Returns {instance_id: [(frame_index, corners), ...]} for top_k sharp frames
```

For each selected frame:
- If it's a YOLO-detected frame: use its own corners directly
- Otherwise: use corners from the nearest detected frame, provided `abs(selected - nearest) <= max_corner_gap`. If the nearest detection is further than `max_corner_gap`, fall back to the nearest YOLO-detected frame itself (safe fallback — always has corners)

**Fallback:** if the Laplacian scan finds no frames (video seek failure, corrupt range, empty track), silently use the original temporal-stride frames unchanged.

### New config

| Field | Default | Notes |
|---|---|---|
| `laplacian_scan_stride: int` | `4` | Every 4th source frame ≈ 15fps for 60fps source |
| `max_corner_gap_frames: int` | `15` | ≈ 0.25s at 60fps; extrapolation safety limit |

`top_k` is not a new config field — it is `max(1, ctx.fusion_target_frames)`, already configurable.

### Files changed

| File | Change |
|---|---|
| `src/card_capture/config.py` | Add `laplacian_scan_stride: int = 4`, `max_corner_gap_frames: int = 15` |
| `pipeline/steps/start.py` | Add both fields to `RunContext`; wire in `init_run` |
| `src/card_capture/pipeline_utils.py` | Add `_laplacian_select_frames()` |
| `pipeline/steps/refine.py` | Call `_laplacian_select_frames()` before canonical decode; replace canonical indices with Laplacian-selected ones |

---

## Shared: Config wiring

Both features add fields to `PipelineConfig` and `RunContext`. Wiring follows the same pattern as `target_yolo_fps` and `novelty_floor`.

---

## Performance Summary

| Step | Before | After |
|---|---|---|
| Opening scan (Feature A) | 0 extra frames | +6 frames to YOLO at start (sub-second) |
| In-track scan (Feature B) | 0 extra video reads | +~300ms single-pass Laplacian scan for 10 tracks |
| YOLO inference | Unchanged | Unchanged (no new YOLO calls from Feature B) |

---

## Testing

**Feature A:**
- Unit test: given `opening_scan_s=1.0`, `source_fps=60.0`, `target_yolo_fps=3.0`, assert `opening_indices == [0, 20, 40]` (range(0, 60, 20))
- Unit test: assert opening indices are deduped against existing presence-window indices (no double-decode)
- Unit test: `opening_scan_s=0.0` adds no frames

**Feature B:**
- Unit test for `_laplacian_select_frames`: synthetic 3-frame track — blurry, sharp, blurry — assert the sharp frame is selected
- Unit test: nearest-corner fallback — assert frame at gap > `max_corner_gap_frames` falls back to nearest detection
- Unit test: empty track input returns empty output without raising
