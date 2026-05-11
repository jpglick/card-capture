# Vision-First Segmentation Design

**Date:** 2026-05-10  
**Status:** Approved  
**Context:** Pipeline V4 Phase B

---

## Problem

The current pipeline relies on *temporal silence* to segment sessions between unique cards. It assumes a card swap is always marked by enough empty frames to exceed the session gap threshold. This "Silence Assumption" breaks in two ways:

1. **Nyquist blindness (Level 1):** The sampler scans at 5fps (1 frame every 200ms). Cards flashed for ≤200ms are physically invisible. Card 4 in `IMG_5872.MOV` (0.17s) is never seen.

2. **Merged sessions (Level 2):** Cards 1, 2, and 3 (shown in rapid succession with 0.7s gaps) fall below the merge threshold (0.6s at 5fps) and are bundled into a single 12s presence window. ByteTrack is handed one continuous timeline and tries to track three distinct cards as one object.

The result: 5 predicted sessions for a 6-card video, with one session containing 3 cards that ByteTrack cannot disambiguate.

---

## Approach: Two-Pass Scan + Three-Signal Session Splitter

### Level 1 Fix — Fast Scan Pass

Introduce a dedicated fast scan pass (default 15fps, configurable via `fast_scan_fps`) that uses only cheap Sobel edge magnitude as its presence metric. This makes sub-200ms card appearances visible. The heavy metrics (z-scores, variance, PresenceClassifier) are deferred to Pass 2 which runs only inside confirmed presence windows.

**Backward compatibility:** `scan_fps` is aliased to `confirm_scan_fps` so existing configs continue to work.

### Level 2 Fix — Three Independent Session Splitters

Three signals are evaluated after Pass 1 and during tracking. Any signal independently triggers a session reset:

1. **Valley Split** — score gradient on Pass 1 Sobel scores
2. **Geography Split** — centroid bounding-box jump detector  
3. **ReID Split** — BoT-SORT visual appearance embeddings replacing ByteTrack

---

## Component Design

### 1. Fast Scan Pass (`sampler.py`)

**New parameter:** `fast_scan_fps: float = 15.0` (configurable, alias `scan_fps` → `confirm_scan_fps` for backward compat)

**Pass 1 output:** `list[_ScanFrame]` — lightweight struct:
```python
@dataclass
class _ScanFrame:
    frame_index: int
    timestamp_ms: float
    sobel_score: float   # cheap Sobel edge magnitude mean; no image retained
```

**Pass 1 metric:** Only Sobel edge magnitude mean on a 160×120 downscale. No z-scores, no batching complexity, no classifier. Target: ≤2s scan time for a 3-minute video.

**Pass 2** (unchanged in structure): Runs the full metrics + PresenceClassifier only on frames inside confirmed windows. Produces `list[_AdaptiveScanFrame]` as today.

### 2. Valley Split (`sampler/valley_splits.py`)

Pure function, no state:

```python
def find_valley_splits(
    scores: list[float],
    frame_indices: list[int],
    valley_drop_ratio: float = 0.40,
    valley_min_width_frames: int = 3,
) -> list[int]:
    """Return sorted list of frame indices at which the score gradient indicates
    a card swap (valley between two peaks). A valley qualifies if it drops
    ≥ valley_drop_ratio from the preceding peak and persists for ≥
    valley_min_width_frames fast-scan frames before recovering."""
```

**Integration:** Called after Pass 1, before Pass 2. Valley split frame indices are treated as forced boundaries in `_build_windows()` — no window spans a valley split point.

**New parameters on `AdaptivePresenceSampler`:**
- `valley_drop_ratio: float = 0.40`
- `valley_min_width_frames: int = 3`

**New CLI flags:** `--valley-drop-ratio`, `--valley-min-width`

### 3. Centroid Jump Detector (`tracking/centroid_jump.py`)

Stateful per-session class:

```python
class CentroidJumpDetector:
    def __init__(
        self,
        jump_ratio: float = 0.30,      # fraction of frame width
        jump_within_frames: int = 3,
    ): ...

    def update(self, bbox_xyxy: Optional[np.ndarray], frame_width: int) -> bool:
        """Returns True if a centroid jump exceeding threshold is detected.
        Pass None if no detection in this frame (treated as no-op)."""

    def reset(self) -> None:
        """Call on session reset to clear history."""
```

**Algorithm:** Maintains a rolling 5-frame centroid history. Computes max centroid displacement over the last `jump_within_frames` frames. If displacement > `jump_ratio × frame_width`, returns `True`.

**Primary detection rule:** If multiple candidates in a frame, use highest-scoring candidate's centroid. No-detection frames leave centroid unchanged.

**Integration in `pipeline.py` tracking loop:**
```python
if centroid_detector.update(bbox, frame_width):
    tracker.finalize()
    tracker.reset()
    centroid_detector.reset()
    # log session_reset event with reason="centroid_jump"
```

**New parameters on `ProcessingOptions`:**
- `centroid_jump_ratio: float = 0.30`
- `centroid_jump_frames: int = 3`

### 4. BoT-SORT Adapter (`tracking/botsort_adapter.py`)

Mirrors the `ByteTrackAdapter` interface exactly:

```python
class BoTSORTAdapter:
    def __init__(
        self,
        min_track_length: int = 3,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        reid_distance_threshold: float = 0.6,
    ): ...

    def process(self, candidates: list[ScoredCandidate]) -> list[_AdaptedDetection]: ...
    def finalize(self) -> list[TrackState]: ...
    def reset(self) -> None: ...
    def finalized_tracks(self) -> list[TrackState]: ...
```

**ReID split:** After each `update_with_detections()`, if BoT-SORT assigns a new track ID to a candidate that was previously tracked (embedding distance > `reid_distance_threshold`), emit a session split signal. The adapter exposes `pending_splits: list[int]` (frame indices) consumed by the pipeline loop.

**Dependency:** `supervision >= 0.28` (BoT-SORT ships in this version). Hard fail with `ImportError` and a clear message if unavailable. The `FutureWarning` about ByteTrack deprecation disappears.

**Fallback:** `ByteTrackAdapter` is retained and can be selected via `tracker_backend="bytetrack"` for testing/debugging. Default is `"botsort"`.

**Integration:** `VideoProcessor.__init__` instantiates tracker based on `tracker_backend` param. `CentroidJumpDetector` runs alongside BoT-SORT as an additional guard (both can trigger resets).

### 5. `sampler sessions` Diagnostic (existing, enhanced)

The existing `card-capture sampler sessions <video> --expected N` command already shows predicted session count. It will be updated to:
- Report fast-scan vs. confirm-scan frame counts separately
- Show valley split points with their score drop percentages
- Still complete in ≤45s (no ML inference, no frame decode)

---

## Data Flow

```
Video ──(single read at 15fps)──► _ScanFrame list
        [images stored in memory]        │
                              ┌──────────┴──────────┐
                    find_valley_splits()      Pass 2 full metrics
                    (in-memory, O(n))         (on stored images,
                              │               no video re-read)
                    split boundaries                  │
                              └──────────┬──────────┘
                                   PresenceWindows
                                         │
                              _decode_selected_frames()
                              (second video read, selected frames only)
                                         │
                              ML inference (producer process)
                                         │
                         Pipeline tracking loop (frame by frame)
                              │                    │
                   BoTSORTAdapter             CentroidJumpDetector
               (ReID split signal +           (geography split signal)
                OSNet embeddings cached)
                              │                    │
                              └────────┬───────────┘
                                   session reset
                                       │
                                 TrackState list
                                 (with cached embeddings)
                                       │
                     Refinement: third video read (high-res, sequential)
                                       │
                     VisualDeduplicator: cosine distance on cached
                     OSNet embeddings (no pHash recompute)
                                       │
                                    Save
```

**Video read count: 3 total** (scan, ML decode, refinement) — same as today. The 15fps scan does not add a fourth read because Pass 2 operates on stored images.

---

## Parameters Reference

| Parameter | Default | Where | CLI Flag |
|---|---|---|---|
| `fast_scan_fps` | 15.0 | `AdaptivePresenceSampler` | `--fast-scan-fps` |
| `confirm_scan_fps` | 5.0 | `AdaptivePresenceSampler` | `--confirm-scan-fps` |
| `valley_drop_ratio` | 0.40 | `AdaptivePresenceSampler` | `--valley-drop-ratio` |
| `valley_min_width_frames` | 3 | `AdaptivePresenceSampler` | `--valley-min-width` |
| `centroid_jump_ratio` | 0.30 | `ProcessingOptions` | `--centroid-jump-ratio` |
| `centroid_jump_frames` | 3 | `ProcessingOptions` | `--centroid-jump-frames` |
| `reid_distance_threshold` | 0.6 | `BoTSORTAdapter` | `--reid-distance-threshold` |
| `tracker_backend` | `"botsort"` | `ProcessingOptions` | `--tracker-backend` |

---

## Testing

### New test files

- **`tests/test_valley_splits.py`** — pure function unit tests:
  - Flat signal → no splits
  - Single valley meeting threshold → one split at valley minimum
  - Shallow valley below `valley_drop_ratio` → no split
  - Multi-valley signal → multiple splits
  - Edge case: valley at start/end of signal

- **`tests/test_centroid_jump.py`** — stateful class tests:
  - Jump above threshold triggers `True`
  - Gradual drift over many frames does not trigger
  - `None` (no-detection) frames don't trigger
  - `reset()` clears history; no spurious split after reset
  - Multiple candidates → highest-scoring centroid used

- **`tests/test_botsort_adapter.py`** — interface contract mirrors `test_bytetrack_adapter.py`:
  - Assigns consistent track ID for overlapping boxes across frames
  - First-frame `tracker_id` guard (same regression as bytetrack)
  - `finalize()` only returns tracks ≥ `min_track_length`
  - `reset()` moves active tracks to finalized pool

- **`tests/test_sampler_fast_scan.py`**:
  - 15fps scan produces more `_ScanFrame` entries than 5fps for same video
  - `scan_fps` alias correctly maps to `confirm_scan_fps`
  - `_scan_frames_fast` retained separately from `_scan_frames` (confirm pass)

### Existing tests
All 138 currently passing tests must continue to pass. The 3 pre-existing `test_pipeline.py` failures are out of scope.

---

## Success Criterion

**Primary gate:** `card-capture sampler sessions IMG_5872.MOV --expected 6` reports 5–7 sessions.

**Secondary gate (regression harness):** recall ≥ 0.95, phantom_rate ≤ 0.222 (unchanged from Phase A spec).

---

## Out of Scope

- Front/back classification within a session (separate future phase)
- Training a dedicated ReID model (BoT-SORT's bundled OSNet is sufficient)
- Any changes to the refinement, deduplication, or storage layers
- UI changes
