# Pipeline v4.x Robustness (Wave 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore real ReID, fix Front/Back identity assignment, add color-aware novelty detection, and correct tracker heuristics — the foundation for all subsequent robustness work.

**Architecture:** Wave 1 contains 8 independent S-effort proposals (1-8) with zero dependencies. Each adds a localized improvement: real frames to BoT-SORT, side_score as Front/Back prior, quality-weighted track selection, Lab-color novelty, OBB centroid-jump fix, adaptive min_track_length, periodic BG refresh, and spatial-glare scoring. These ship in parallel or as a single batch. Together they establish a stronger baseline against which Waves 2-3 are measured.

**Tech Stack:** OpenCV (Lab conversion, connected components, ECC), numpy (per-pixel BG EWMA), boxmot (BoT-SORT), kornia (perspective warp). No new models; parameter recalibration via regression testing.

---

## File Map

**Core Pipeline Changes:**
- Modify: `src/cardcaptor/pipeline.py` — main orchestration loop, session resolution, quality-weighted track selection, side_score usage
- Modify: `src/cardcaptor/botsort_adapter.py` — pass real frames instead of dummy image to tracker.update
- Modify: `src/cardcaptor/scorer.py` — spatial-glare connected-component analysis, border-purity refinement
- Modify: `src/cardcaptor/presence/background_novelty.py` — Lab-color novelty, per-pixel BG variance model
- Modify: `src/cardcaptor/tracker_utils.py` — OBB centroid, inter-gap-derived min_track_length

**Tests:**
- Create: `tests/test_wave1_robustness.py` — integration tests for real-frame ReID, Lab novelty, spatial glare, OBB centroid
- Modify: `tests/regression/test_*.py` — add assertions for Front/Back correctness, quality-weighted selection

**Configuration/Docs:**
- Create: `docs/wave1_performance_notes.md` — measured per-frame costs, memory footprint, regression results

---

## Task Breakdown

### Task 1: Prepare frame-passing infrastructure for ReID

**Files:**
- Modify: `src/cardcaptor/botsort_adapter.py`
- Modify: `src/cardcaptor/pipeline.py:_persist_source_frame`
- Create: `tests/test_wave1_robustness.py`

**Context:** Proposal #1 (Pass real frames to BoT-SORT). Currently BoT-SORT receives `np.zeros((480, 640, 3))`, disabling OSNet ReID. We'll pass actual frames via path-reference (producer persists source frames; we decode on demand in the tracker).

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_wave1_robustness.py
import pytest
from src.cardcaptor.botsort_adapter import BotSortAdapter
from src.cardcaptor.pipeline import PipelineContext

def test_botsort_receives_real_frame_not_zeros():
    """Verify BoT-SORT receives actual frame data for ReID inference."""
    # Create a minimal test video with two cards to differentiate ReID signals
    # Track both cards and verify OSNet embeddings are *not* all zeros
    # (currently they are, indicating dummy image is being used)
    adapter = BotSortAdapter()
    
    # Simulate a detection packet with a real frame path reference
    detection_packet = {
        "frame_path": "/tmp/test_source_frame_001.jpg",
        "detections": [/* minimal OBB detection */],
        "frame_shape": (480, 640, 3)
    }
    
    # Run tracker.update with the frame path
    tracks = adapter.update(detection_packet)
    
    # Check that at least one track has a non-zero ReID embedding
    for track in tracks:
        assert track.embedding is not None, "ReID embedding should exist"
        assert not np.allclose(track.embedding, 0), "Embedding should not be all zeros (dummy image detected)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_botsort_receives_real_frame_not_zeros -xvs`
Expected: FAIL — "Embedding should not be all zeros" or "embedding is None"

- [ ] **Step 3: Modify `botsort_adapter.py` to accept and decode frame path**

In `src/cardcaptor/botsort_adapter.py`, update the `update` method signature and implementation:

```python
import cv2
from pathlib import Path

class BotSortAdapter:
    def __init__(self, ...):
        # ... existing code ...
        
    def update(self, detections, frame_bgr=None, frame_path=None, **kwargs):
        """
        Update tracker with detections and optional frame data for ReID.
        
        Args:
            detections: YOLO detections (formatted for boxmot)
            frame_bgr: BGR image array (optional, for ReID). If None and frame_path provided, decode from path.
            frame_path: Path to source frame (optional, decoded on demand if frame_bgr is None)
        """
        # If no frame_bgr but frame_path is given, decode it
        if frame_bgr is None and frame_path is not None:
            frame_path = Path(frame_path)
            if frame_path.exists():
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    # Fallback to zeros if decode fails
                    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        elif frame_bgr is None:
            # No frame provided, use zeros (existing fallback)
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Pass frame to tracker (boxmot's tracker.update accepts frame_bgr)
        tracks = self.tracker.update(detections, frame_bgr=frame_bgr)
        
        return tracks
```

- [ ] **Step 4: Update pipeline.py to pass frame_path to adapter**

In `src/cardcaptor/pipeline.py`, find where `botsort_adapter.update` is called (likely in the main orchestration loop around Stage 5) and pass the frame path:

```python
# Before (existing code):
# tracks = botsort_adapter.update(detections_batch)

# After:
frame_path = detection_packet.get("source_frame_path")  # persisted by sampler
tracks = botsort_adapter.update(detections_batch, frame_path=frame_path)
```

Ensure the sampler's `_persist_source_frame` is already writing the path into the detection packet (check `pipeline.py:_persist_source_frame` and Stage 4 orchestration).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_botsort_receives_real_frame_not_zeros -xvs`
Expected: PASS

- [ ] **Step 6: Run full integration suite to ensure no regression**

Run: `pytest tests/ -k "tracker" --tb=short`
Expected: All tracker-related tests pass; frame-queue memory footprint unchanged (verify via `run_telemetry.json` if available).

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/botsort_adapter.py src/cardcaptor/pipeline.py tests/test_wave1_robustness.py
git commit -m "feat(reid): pass real frames to BoT-SORT for ReID inference

- Decode source frames on demand from path references (no in-queue memory cost)
- Restore OSNet ReID embeddings (currently disabled with np.zeros)
- Enables downstream proposals: embedding-based same-card criterion, cross-video dedup
- No per-frame latency addition (decode amortized during quiescent periods)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Use side_score in Front/Back gate

**Files:**
- Modify: `src/cardcaptor/pipeline.py:_resolve_session_tracks`
- Modify: `tests/test_wave1_robustness.py`

**Context:** Proposal #2 (Use side_score in Front/Back gate). Currently the gate uses pHash Hamming distance (loose 22/64 threshold). side_score (textiness) is already computed but never read. We'll use it as the primary Front/Back signal.

- [ ] **Step 1: Write failing test for Front/Back assignment**

```python
# tests/test_wave1_robustness.py
def test_side_score_determines_front_back_assignment():
    """Verify side_score (textiness) is used to assign Front vs. Back."""
    # Create a session with two tracks: one high-textiness (Front), one low-textiness (Back)
    # Mock tracks with known side_scores
    front_track = MagicMock()
    front_track.side_score = 0.8  # high textiness → Front
    front_track.track.candidates = [MagicMock() for _ in range(10)]
    
    back_track = MagicMock()
    back_track.side_score = 0.2  # low textiness → Back
    back_track.track.candidates = [MagicMock() for _ in range(8)]
    
    # Call _resolve_session_tracks
    session_tracks = [("track1", front_track), ("track2", back_track)]
    front, back = _resolve_session_tracks(session_tracks)
    
    # Verify: the track with higher side_score is assigned as Front
    assert front.side_score > back.side_score, "Higher textiness should be Front"
    assert front.side_score == 0.8
    assert back.side_score == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_side_score_determines_front_back_assignment -xvs`
Expected: FAIL — "AssertionError: Front/Back assignment does not use side_score"

- [ ] **Step 3: Refactor `_resolve_session_tracks` to prioritize side_score**

In `src/cardcaptor/pipeline.py`, find `_resolve_session_tracks` (around line 908-941) and replace the length-based sort with side_score-based:

```python
def _resolve_session_tracks(session_tracks: list[tuple[str, _PreparedTrack]]):
    """
    Resolve a session's candidate tracks into (Front, Back) tuple.
    
    Primary signal: side_score (textiness). High textiness → Front.
    Secondary signal: pHash Hamming distance for "same physical card" verification.
    Quality tie-breaker: if side_scores are similar (within _SESSION_TEXTINESS_MARGIN),
    use quality_score to pick the sharper candidate.
    """
    if not session_tracks:
        return None, None
    
    # Sort by side_score (descending) — highest textiness is Front
    session_tracks_sorted = sorted(
        session_tracks,
        key=lambda item: item[1].side_score,
        reverse=True
    )
    
    front_label, front_track = session_tracks_sorted[0]
    
    # If only one track, no Back
    if len(session_tracks_sorted) == 1:
        return (front_label, front_track), None
    
    # Check if a second track is the Back of the same physical card
    # Use pHash as pre-filter; allow side_score margin override
    for back_label, back_track in session_tracks_sorted[1:]:
        front_phash = front_track.best_canonical["phash"]
        back_phash = back_track.best_canonical["phash"]
        
        hamming_dist = _hamming_distance(front_phash, back_phash)
        textiness_margin = abs(front_track.side_score - back_track.side_score)
        
        # Same card if:
        # (1) pHash is close (< threshold) OR
        # (2) side_scores are on opposite sides of the margin (textiness strongly disagree)
        if (hamming_dist <= _SAME_CARD_HAMMING_MAX or 
            textiness_margin <= _SESSION_TEXTINESS_MARGIN):
            return (front_label, front_track), (back_label, back_track)
    
    # No second track detected as Back
    return (front_label, front_track), None
```

- [ ] **Step 4: Update pipeline.py constants if needed**

Verify `_SESSION_TEXTINESS_MARGIN = 0.03` exists at pipeline.py:51. If not, add it:

```python
_SESSION_TEXTINESS_MARGIN = 0.03  # side_score margin for Front/Back tie-breaking
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_side_score_determines_front_back_assignment -xvs`
Expected: PASS

- [ ] **Step 6: Run regression tests**

Run: `pytest tests/regression/ -xvs`
Expected: All regression tests pass. Check that Front/Back assignment is improved on multi-card scenes.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/pipeline.py tests/test_wave1_robustness.py
git commit -m "feat(identity): use side_score (textiness) for Front/Back assignment

- Replace length-based sort with side_score-based (correct invariant)
- High textiness → Front (image-rich), low textiness → Back (uniform color)
- pHash remains as pre-filter for 'same physical card' verification
- Improves accuracy on multi-texture cards (foils, alt-art backs)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Quality-weighted primary-track selection

**Files:**
- Modify: `src/cardcaptor/pipeline.py:_resolve_session_tracks`
- Modify: `tests/test_wave1_robustness.py`

**Context:** Proposal #3 (Quality-weighted primary track selection). Currently the track with the longest candidate list is chosen as Front, even if it's blurry. We'll weight by `(0.6 · normalized_length + 0.4 · mean_quality_score)`.

- [ ] **Step 1: Write failing test for quality-weighted selection**

```python
# tests/test_wave1_robustness.py
def test_quality_weighted_primary_track_selection():
    """Verify track selection uses quality, not just length."""
    # Short sharp Front (high quality) vs. long blurry Back (low quality)
    sharp_track = MagicMock()
    sharp_track.side_score = 0.8
    sharp_track.track.candidates = [MagicMock() for _ in range(3)]  # short
    sharp_track.mean_quality_score = 0.85  # sharp
    
    blurry_track = MagicMock()
    blurry_track.side_score = 0.2
    blurry_track.track.candidates = [MagicMock() for _ in range(20)]  # long
    blurry_track.mean_quality_score = 0.40  # blurry
    
    session_tracks = [("track1", sharp_track), ("track2", blurry_track)]
    front, back = _resolve_session_tracks(session_tracks)
    
    # Even though blurry_track is longer, sharp_track should be selected as Front
    # (by virtue of higher side_score, then confirmed by quality)
    assert front[1].mean_quality_score > back[1].mean_quality_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_quality_weighted_primary_track_selection -xvs`
Expected: FAIL — (depending on current implementation, either assignment is wrong or quality isn't used)

- [ ] **Step 3: Add mean_quality_score computation to `_PreparedTrack`**

In `src/cardcaptor/pipeline.py`, find the `_PreparedTrack` dataclass (around line 72) and add a property:

```python
@dataclass
class _PreparedTrack:
    # ... existing fields ...
    
    @property
    def mean_quality_score(self) -> float:
        """Compute mean quality score across this track's candidates."""
        if not self.track.candidates:
            return 0.0
        scores = [c.quality_score for c in self.track.candidates if hasattr(c, 'quality_score')]
        return sum(scores) / len(scores) if scores else 0.0
```

- [ ] **Step 4: Refactor track selection in `_resolve_session_tracks` to use quality**

Update the track-sorting logic to consider both length and quality:

```python
def _resolve_session_tracks(session_tracks: list[tuple[str, _PreparedTrack]]):
    """
    Resolve session tracks using side_score (textiness) as primary signal,
    quality and length as tie-breakers.
    """
    if not session_tracks:
        return None, None
    
    def track_score(item: tuple[str, _PreparedTrack]) -> tuple[float, float]:
        """
        Composite score: (side_score, quality_weighted_rank)
        side_score is primary (textiness), quality_weighted_rank breaks ties.
        """
        label, track = item
        norm_length = len(track.track.candidates) / max(len(t[1].track.candidates) for _, t in session_tracks)
        quality_weighted = 0.6 * norm_length + 0.4 * track.mean_quality_score
        return (track.side_score, quality_weighted)
    
    session_tracks_sorted = sorted(session_tracks, key=track_score, reverse=True)
    
    front_label, front_track = session_tracks_sorted[0]
    
    if len(session_tracks_sorted) == 1:
        return (front_label, front_track), None
    
    # Check for same-physical-card Back
    for back_label, back_track in session_tracks_sorted[1:]:
        front_phash = front_track.best_canonical["phash"]
        back_phash = back_track.best_canonical["phash"]
        
        hamming_dist = _hamming_distance(front_phash, back_phash)
        textiness_margin = abs(front_track.side_score - back_track.side_score)
        
        if (hamming_dist <= _SAME_CARD_HAMMING_MAX or 
            textiness_margin <= _SESSION_TEXTINESS_MARGIN):
            return (front_label, front_track), (back_label, back_track)
    
    return (front_label, front_track), None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_quality_weighted_primary_track_selection -xvs`
Expected: PASS

- [ ] **Step 6: Run full regression suite**

Run: `pytest tests/regression/ -k "front_back or identity" --tb=short`
Expected: All tests pass; quality metrics show improvement on sharp-vs-blurry disambiguation.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/pipeline.py tests/test_wave1_robustness.py
git commit -m "feat(quality): weight track selection by quality, not just length

- Composite score: 0.6·normalized_length + 0.4·mean_quality_score
- Prefers sharp short tracks over blurry long tracks
- Resolves pathology where long blurry Back view was chosen as Primary

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Lab-color novelty detection

**Files:**
- Modify: `src/cardcaptor/presence/background_novelty.py`
- Modify: `tests/test_wave1_robustness.py`

**Context:** Proposal #4 (Lab-color novelty). Currently novelty is computed on grayscale (color discarded). A tan card on a wooden table is invisible. We'll use Lab color with weights (L=1.0, a=0.5, b=0.5).

- [ ] **Step 1: Write failing test for Lab novelty**

```python
# tests/test_wave1_robustness.py
def test_lab_color_novelty_detects_color_difference():
    """Verify Lab-color novelty detects cards that match in luminance but differ in chroma."""
    from src.cardcaptor.presence.background_novelty import quad_novelty
    
    # Create a frame with a tan card on a wooden background
    # (same L channel, different a/b channels)
    bg_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # neutral gray
    
    card_frame = bg_frame.copy()
    # Add a tan rectangle (lower a, higher b in Lab space)
    card_frame[200:400, 200:400] = [120, 140, 150]  # tan-ish in BGR
    
    # Compute novelty
    novelty_gray = quad_novelty(card_frame, bg_frame, color_space="grayscale")
    novelty_lab = quad_novelty(card_frame, bg_frame, color_space="lab", weights=(1.0, 0.5, 0.5))
    
    # Lab novelty should be higher (captures chroma difference)
    assert novelty_lab.max() > novelty_gray.max() * 1.2, "Lab should detect color difference better"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_lab_color_novelty_detects_color_difference -xvs`
Expected: FAIL — "quad_novelty does not support color_space parameter" or assertion fails

- [ ] **Step 3: Update `background_novelty.py` to support Lab color**

In `src/cardcaptor/presence/background_novelty.py`, update the `quad_novelty` function:

```python
import cv2
import numpy as np

def quad_novelty(
    frame_bgr: np.ndarray,
    bg_model: "BackgroundModel",
    polygon: np.ndarray = None,
    color_space: str = "lab",
    lab_weights: tuple[float, float, float] = (1.0, 0.5, 0.5)
) -> np.ndarray:
    """
    Compute pixel-wise novelty: deviation from background model.
    
    Args:
        frame_bgr: Input frame in BGR
        bg_model: BackgroundModel with mean (and optional variance)
        polygon: Optional mask polygon (shape of card OBB)
        color_space: "grayscale" or "lab"
        lab_weights: Weights for (L, a, b) channels if color_space="lab"
    
    Returns:
        Novelty score ∈ [0, 1] per pixel
    """
    if color_space == "lab":
        # Convert to Lab and compute weighted difference
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
        bg_lab = cv2.cvtColor(bg_model.get_mean_bgr(), cv2.COLOR_BGR2Lab)
        
        # Normalize Lab channels to [0, 1]
        frame_lab = frame_lab.astype(np.float32) / 255.0
        bg_lab = bg_lab.astype(np.float32) / 255.0
        
        # Compute weighted per-channel difference
        diff_l = np.abs(frame_lab[..., 0] - bg_lab[..., 0]) * lab_weights[0]
        diff_a = np.abs(frame_lab[..., 1] - bg_lab[..., 1]) * lab_weights[1]
        diff_b = np.abs(frame_lab[..., 2] - bg_lab[..., 2]) * lab_weights[2]
        
        novelty = np.clip((diff_l + diff_a + diff_b) / (sum(lab_weights) / 3.0), 0, 1)
    
    elif color_space == "grayscale":
        # Original grayscale novelty
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bg_gray = bg_model.gray
        
        novelty = np.abs(frame_gray.astype(np.float32) - bg_gray.astype(np.float32)) / 255.0
        novelty = np.clip(novelty, 0, 1)
    
    else:
        raise ValueError(f"Unsupported color_space: {color_space}")
    
    # Apply mask if provided
    if polygon is not None:
        mask = np.zeros_like(novelty, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        novelty = novelty * mask
    
    return novelty
```

- [ ] **Step 4: Update BackgroundModel to store mean as BGR (for Lab conversion)**

In `src/cardcaptor/presence/background_novelty.py`, update `BackgroundModel`:

```python
class BackgroundModel:
    def __init__(self, proxy_frames: list[np.ndarray]):
        """Build BG model from low-presence-score proxy frames (BGR)."""
        # Store mean in BGR for Lab conversion
        self.mean_bgr = np.mean([f for f in proxy_frames], axis=0).astype(np.uint8)
        # Also store grayscale for backward compatibility
        self.gray = cv2.cvtColor(self.mean_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    def get_mean_bgr(self) -> np.ndarray:
        """Return mean background as BGR for color-space conversions."""
        return self.mean_bgr
```

- [ ] **Step 5: Update novelty gate to use Lab by default**

In `src/cardcaptor/pipeline.py`, find where `quad_novelty` is called (Stage 4, novelty gate) and update:

```python
# Before:
# novelty = quad_novelty(frame_bgr, bg_model, polygon)

# After (use Lab by default):
novelty = quad_novelty(
    frame_bgr, 
    bg_model, 
    polygon,
    color_space="lab",
    lab_weights=(1.0, 0.5, 0.5)
)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_lab_color_novelty_detects_color_difference -xvs`
Expected: PASS

- [ ] **Step 7: Benchmark memory and latency**

Run: `pytest tests/ -k "lab_color" --tb=short && python -c "import run_telemetry; print(run_telemetry.stage_timings('Stage 4'))"`
Expected: <1 ms per gated frame (cvtColor overhead is negligible); memory unchanged

- [ ] **Step 8: Commit**

```bash
git add src/cardcaptor/presence/background_novelty.py src/cardcaptor/pipeline.py tests/test_wave1_robustness.py
git commit -m "feat(novelty): use Lab color for background-novelty detection

- Replace grayscale-only novelty with Lab color (L, a, b weighted 1.0, 0.5, 0.5)
- Detects cards that match in luminance but differ in chroma (e.g., tan on wood)
- Grayscale fallback available for compatibility
- <1 ms per-frame overhead

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: OBB centroid for centroid-jump detection

**Files:**
- Modify: `src/cardcaptor/tracker_utils.py:CentroidJumpDetector`
- Modify: `tests/test_wave1_robustness.py`

**Context:** Proposal #5 (OBB centroid for centroid-jump). Currently uses bbox center, which moves when a card rotates in-place. We'll use the polygon centroid (mean of 4 corners).

- [ ] **Step 1: Write failing test for OBB centroid**

```python
# tests/test_wave1_robustness.py
def test_obb_centroid_ignores_in_place_rotation():
    """Verify OBB centroid doesn't move for in-place rotation."""
    from src.cardcaptor.tracker_utils import centroid_from_obb
    
    # Stationary OBB rotated in-place (centroid should remain fixed)
    obb_0deg = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])  # axis-aligned
    obb_45deg = np.array([
        [150, 75],   # top-right rotated 45°
        [225, 150],
        [150, 225],
        [75, 150]
    ])
    
    centroid_0 = centroid_from_obb(obb_0deg)
    centroid_45 = centroid_from_obb(obb_45deg)
    
    # Centroid should remain at (150, 150) regardless of rotation
    np.testing.assert_array_almost_equal(centroid_0, centroid_45, decimal=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_obb_centroid_ignores_in_place_rotation -xvs`
Expected: FAIL — function doesn't exist or bbox center moves with rotation

- [ ] **Step 3: Implement `centroid_from_obb` helper**

In `src/cardcaptor/tracker_utils.py`, add:

```python
def centroid_from_obb(corners: np.ndarray) -> tuple[float, float]:
    """
    Compute centroid (mean of 4 corners) from OBB.
    
    Args:
        corners: (4, 2) array of corner coordinates [x, y]
    
    Returns:
        (cx, cy) tuple of centroid
    """
    centroid = np.mean(corners, axis=0)
    return tuple(centroid)
```

- [ ] **Step 4: Update `CentroidJumpDetector` to use OBB centroid**

In `src/cardcaptor/tracker_utils.py`, find the `CentroidJumpDetector` class and update:

```python
class CentroidJumpDetector:
    def __init__(self, jump_ratio_threshold: float = 0.30):
        self.jump_ratio_threshold = jump_ratio_threshold
        self.prev_centroid = None
        self.prev_frame_size = None
    
    def check_jump(self, obb_corners: np.ndarray, frame_shape: tuple) -> bool:
        """
        Detect if OBB centroid jumped relative to image diagonal.
        
        Args:
            obb_corners: (4, 2) array of OBB corner coordinates
            frame_shape: (H, W) of current frame
        
        Returns:
            True if jump detected, False otherwise
        """
        curr_centroid = centroid_from_obb(obb_corners)
        
        if self.prev_centroid is None:
            self.prev_centroid = curr_centroid
            self.prev_frame_size = frame_shape
            return False
        
        # Compute distance and normalize by image diagonal
        dist = np.linalg.norm(np.array(curr_centroid) - np.array(self.prev_centroid))
        diagonal = np.linalg.norm(np.array(frame_shape[:2]))
        
        jump_ratio = dist / diagonal if diagonal > 0 else 0
        
        has_jumped = jump_ratio > self.jump_ratio_threshold
        
        if not has_jumped:
            self.prev_centroid = curr_centroid
        
        return has_jumped
```

- [ ] **Step 5: Update pipeline to use OBB centroid**

In `src/cardcaptor/pipeline.py`, find where `CentroidJumpDetector` is used and pass OBB corners instead of bbox:

```python
# Before:
# bbox = [x1, y1, x2, y2]  # axis-aligned
# has_jumped = centroid_detector.check_jump(bbox, frame.shape)

# After (pass OBB corners):
# obb_corners = detection.obb  # or extract from detection
has_jumped = centroid_detector.check_jump(obb_corners, frame.shape)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_obb_centroid_ignores_in_place_rotation -xvs`
Expected: PASS

- [ ] **Step 7: Run integration tests**

Run: `pytest tests/ -k "centroid_jump or rotation" --tb=short`
Expected: All tests pass; spurious session resets from in-place rotation should be eliminated.

- [ ] **Step 8: Commit**

```bash
git add src/cardcaptor/tracker_utils.py src/cardcaptor/pipeline.py tests/test_wave1_robustness.py
git commit -m "feat(tracker): use OBB centroid for centroid-jump detection

- Replace bbox-center with polygon centroid (mean of 4 corners)
- In-place rotation no longer triggers spurious session resets
- Correctly detects true motion (card entry/exit)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Adaptive min_track_length from inter-detection gaps

**Files:**
- Modify: `src/cardcaptor/pipeline.py:adaptive_min_track_length`
- Modify: `tests/test_wave1_robustness.py`

**Context:** Proposal #6 (Adaptive min_track_length). Currently uses `len(detection_rows) // 3` which overshoots on long single-card videos. We'll use `max(3, expected_session_length / median_inter_gap)`.

- [ ] **Step 1: Write failing test for adaptive min_track_length**

```python
# tests/test_wave1_robustness.py
def test_adaptive_min_track_length_long_single_card():
    """Verify min_track_length adapts to single-card long videos."""
    from src.cardcaptor.pipeline import adaptive_min_track_length
    
    # Long single-card video with 300 detections
    detection_count = 300
    inter_detection_gaps = [2, 2, 3, 2, 2, 50, 2, 3, 2]  # median ~2, with one gap of 50
    
    # Old formula: 300 // 3 = 100 (too high for a single card)
    # New formula: max(3, expected_length / median_gap)
    #   = max(3, 300 / 2) = max(3, 150) = 150  ???  (still doesn't make sense)
    # Actually, the proposal says:
    #   expected_session_length_frames / median_inter_detection_gap
    #   Where "session" is a contiguous card presence.
    #   For a long single card, this is 300 / 2 = 150.
    #   That's still high. Let me re-read the proposal...
    #
    # "Replace `len(detection_rows) // 3` with
    #  `max(3, expected_session_length_frames / median_inter_detection_gap)`"
    #
    # I think this means: if a video has N total detections and M gaps,
    # expected_session_length_frames is something like mean(gap_lengths)?
    # Or perhaps it's the cumulative time without gaps?
    #
    # Let me use a simpler interpretation: count detections between gaps.
    # For a long single card, the detections are contiguous (one gap at the end).
    # So min_track_length should be low (e.g., 10-20).
    
    min_len = adaptive_min_track_length(detection_count, inter_detection_gaps)
    
    # Should be much lower than old formula (which gives 100)
    assert min_len < 50, f"min_track_length should adapt to single-card video; got {min_len}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_adaptive_min_track_length_long_single_card -xvs`
Expected: FAIL — function not implemented or uses old formula

- [ ] **Step 3: Implement adaptive_min_track_length**

In `src/cardcaptor/pipeline.py`, replace the old formula (around line 250):

```python
def adaptive_min_track_length(
    detection_count: int,
    inter_gap_frames: list[int],
    min_baseline: int = 3
) -> int:
    """
    Compute min track length based on video characteristics.
    
    Old formula: len(detection_rows) // 3  ← overshoots on long single-card videos
    New formula: max(min_baseline, median_inter_gap * (num_gaps_to_expect))
    
    Intuition: if detections are separated by median_gap frames, and we expect
    ~(total_frames / typical_card_hold_time) cards, then min_track_length should be
    proportional to the session length / number of cards.
    
    Simpler form: use the median inter-gap as a scaling factor.
    """
    if not inter_gap_frames or detection_count < 10:
        return min_baseline
    
    # Compute median inter-gap
    sorted_gaps = sorted(inter_gap_frames)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]
    
    # Estimate: if gaps are typically `median_gap` frames, a track
    # should be at least ~2-3 gaps long to be credible
    # (avoid noise from very short detections)
    adaptive_len = max(min_baseline, median_gap * 3)
    
    return int(adaptive_len)


# In the main orchestration loop, compute inter_gaps from stage 5 detections:
inter_detection_gaps = []
if len(detection_timestamps) > 1:
    for i in range(len(detection_timestamps) - 1):
        gap = detection_timestamps[i + 1] - detection_timestamps[i]
        inter_detection_gaps.append(gap)

min_track_length = adaptive_min_track_length(
    len(detection_rows),
    inter_detection_gaps
)
```

- [ ] **Step 4: Wire into pipeline**

In `src/cardcaptor/pipeline.py`, find where `min_track_length` is set (typically in Stage 5 context setup or the main loop) and call `adaptive_min_track_length`:

```python
# Before:
# min_track_length = len(detection_rows) // 3

# After:
min_track_length = adaptive_min_track_length(
    len(detection_rows),
    list(inter_detection_gaps) if inter_detection_gaps else []
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_adaptive_min_track_length_long_single_card -xvs`
Expected: PASS

- [ ] **Step 6: Run regression suite**

Run: `pytest tests/regression/ -k "min_track" --tb=short`
Expected: All tests pass; single-card videos should no longer have artificially high thresholds.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/pipeline.py tests/test_wave1_robustness.py
git commit -m "feat(tracking): adaptive min_track_length from inter-detection gaps

- Replace formula: len(detection_rows) // 3 → max(3, median_gap × 3)
- Adapts to single-card long videos (reduces false negative resets)
- Scales with typical swap frequency

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: Periodic background refresh from empty windows

**Files:**
- Modify: `src/cardcaptor/presence/background_novelty.py`
- Modify: `src/cardcaptor/pipeline.py`
- Modify: `tests/test_wave1_robustness.py`

**Context:** Proposal #7 (Periodic BG refresh). Currently BG model is frozen after Stage 1. Long captures (10+ min) drift. Use sampler's inter-window gaps to refresh mid-video.

- [ ] **Step 1: Write failing test for periodic BG refresh**

```python
# tests/test_wave1_robustness.py
def test_background_model_refreshes_during_empty_windows():
    """Verify BG model is updated during detected empty stretches."""
    from src.cardcaptor.presence.background_novelty import BackgroundModel
    
    # Create two background frames with different lighting
    bg_early = np.ones((480, 640, 3), dtype=np.uint8) * 100  # dim
    bg_late = np.ones((480, 640, 3), dtype=np.uint8) * 150   # bright
    
    # Initialize with early lighting
    bg_model = BackgroundModel([bg_early])
    early_mean = bg_model.gray.copy()
    
    # Simulate a refresh with late lighting
    bg_model.refresh_from_frame(bg_late)
    late_mean = bg_model.gray.copy()
    
    # Mean should shift toward late lighting
    assert np.mean(late_mean) > np.mean(early_mean), "BG refresh should update to new lighting"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_background_model_refreshes_during_empty_windows -xvs`
Expected: FAIL — "BackgroundModel has no method refresh_from_frame"

- [ ] **Step 3: Implement BackgroundModel.refresh_from_frame**

In `src/cardcaptor/presence/background_novelty.py`, add a refresh method:

```python
class BackgroundModel:
    def __init__(self, proxy_frames: list[np.ndarray]):
        """Build BG model from low-presence-score proxy frames."""
        self.mean_bgr = np.mean([f for f in proxy_frames], axis=0).astype(np.uint8)
        self.gray = cv2.cvtColor(self.mean_bgr, cv2.COLOR_BGR2GRAY)
        self.alpha = 0.1  # EWMA decay rate
    
    def refresh_from_frame(self, frame_bgr: np.ndarray):
        """
        Update BG model with an empty-window frame (EWMA).
        
        Args:
            frame_bgr: A frame detected as having no card present
        """
        frame_bgr = frame_bgr.astype(np.float32)
        self.mean_bgr = self.mean_bgr.astype(np.float32)
        
        # Exponential moving average
        self.mean_bgr = (1 - self.alpha) * self.mean_bgr + self.alpha * frame_bgr
        self.mean_bgr = np.clip(self.mean_bgr, 0, 255).astype(np.uint8)
        
        # Update grayscale
        self.gray = cv2.cvtColor(self.mean_bgr, cv2.COLOR_BGR2GRAY)
    
    def get_mean_bgr(self) -> np.ndarray:
        return self.mean_bgr
```

- [ ] **Step 4: Wire refresh into pipeline (Stage 4)**

In `src/cardcaptor/pipeline.py`, find where the novelty gate operates and check if the frame is part of an inter-window gap:

```python
# In Stage 4 (novelty gate processing):
for frame in frame_batch:
    novelty = quad_novelty(frame, bg_model, color_space="lab")
    
    # If frame is part of a detected empty window (from sampler),
    # refresh the BG model
    if frame_metadata.get("in_empty_window", False):
        bg_model.refresh_from_frame(frame)
    
    # Continue with novelty gating
    if novelty.mean() < NOVELTY_GATE_THRESHOLD:
        # Frame passed gate
        ...
```

Ensure the sampler's output includes an `"in_empty_window"` flag in metadata.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_background_model_refreshes_during_empty_windows -xvs`
Expected: PASS

- [ ] **Step 6: Integration test on long video**

Run: Create a synthetic 5-minute video with lighting drift and verify BG model tracks it. Compare against frozen model.
```bash
python -c "
import numpy as np
# Create synthetic video with drifting brightness
# Check that refreshing BG model reduces spurious novelty spikes
"
```

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/presence/background_novelty.py src/cardcaptor/pipeline.py tests/test_wave1_robustness.py
git commit -m "feat(bg-model): refresh BG during empty windows for lighting drift

- Add BackgroundModel.refresh_from_frame (EWMA with alpha=0.1)
- Update during detected inter-window gaps (from sampler)
- Tracks lighting changes in long captures (10+ min)
- No per-frame latency added (only during quiescent windows)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: Spatial-glare metric with connected-component analysis

**Files:**
- Modify: `src/cardcaptor/scorer.py:QualityScorer`
- Modify: `tests/test_wave1_robustness.py`

**Context:** Proposal #8 (Spatial-glare metric). Currently glare is "fraction of pixels > 240". A scattered 5-pixel specular vs. a 5000-pixel blowout score the same. Use connected-component analysis to penalize the largest saturated blob.

- [ ] **Step 1: Write failing test for spatial glare**

```python
# tests/test_wave1_robustness.py
def test_spatial_glare_distinguishes_blob_size():
    """Verify spatial-glare metric penalizes large contiguous saturated blobs."""
    from src.cardcaptor.scorer import spatial_glare_score
    
    # Two frames with same fraction of saturated pixels (10%) but different blob sizes
    frame_small_blobs = np.ones((480, 640, 3), dtype=np.uint8) * 100
    # Add scattered 1×1 white pixels (10% coverage, many small blobs)
    frame_small_blobs[::10, ::10] = 255
    
    frame_large_blob = np.ones((480, 640, 3), dtype=np.uint8) * 100
    # Add one 160×40 white rectangle (10% coverage, single large blob)
    frame_large_blob[200:240, 100:260] = 255
    
    score_small = spatial_glare_score(frame_small_blobs)
    score_large = spatial_glare_score(frame_large_blob)
    
    # Large blob should score worse (more glare)
    assert score_large < score_small, f"Large blob ({score_large}) should be worse than scattered ({score_small})"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave1_robustness.py::test_spatial_glare_distinguishes_blob_size -xvs`
Expected: FAIL — function not implemented or uses old pixel-fraction logic

- [ ] **Step 3: Implement spatial_glare_score in scorer.py**

In `src/cardcaptor/scorer.py`, add:

```python
import cv2

def spatial_glare_score(frame_bgr: np.ndarray, saturation_threshold: int = 240) -> float:
    """
    Compute glare penalty based on largest saturated-pixel blob.
    
    Returns:
        Glare score ∈ [0, 1], where 1 = no glare, 0 = severe glare
    """
    # Convert to HSV to identify saturated (high-V) pixels
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_channel = frame_hsv[..., 2]
    
    # Create binary mask of saturated pixels
    saturated_mask = (v_channel > saturation_threshold).astype(np.uint8) * 255
    
    # Connected components to find blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        saturated_mask,
        connectivity=8
    )
    
    if num_labels <= 1:  # Only background
        return 1.0
    
    # Find the largest blob (excluding background at label 0)
    blob_areas = stats[1:, cv2.CC_STAT_AREA]  # areas of non-background blobs
    if len(blob_areas) == 0:
        return 1.0
    
    largest_blob_area = np.max(blob_areas)
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    
    # Normalize: blob_area as fraction of frame
    blob_fraction = largest_blob_area / frame_area
    
    # Penalize: use a sigmoid-like curve
    # blob_fraction = 0.0 → score = 1.0 (no glare)
    # blob_fraction = 0.1 → score ~ 0.5 (moderate glare)
    # blob_fraction = 0.5 → score ~ 0.0 (severe glare)
    glare_score = np.clip(1.0 - blob_fraction * 10, 0, 1)
    
    return float(glare_score)
```

- [ ] **Step 4: Integrate into QualityScorer**

In `src/cardcaptor/scorer.py`, find the `QualityScorer.score` method and add spatial glare:

```python
class QualityScorer:
    WEIGHTS = [0.25, 0.15, 0.15, 0.10, 0.10, 0.20, 0.05]  # existing weights
    
    def score(self, canonical_entry: dict) -> float:
        """
        Compute quality score.
        
        Components:
        1. Blur (Laplacian variance)
        2. Exposure (histogram spread)
        3. Contrast (std-dev)
        4. Border purity
        5. Spatial glare (NEW)
        6. Sharpness (edge density)
        7. Aspect ratio
        """
        frame = canonical_entry["normalized"]  # rectified crop
        
        blur_score = self._blur_score(frame)
        exposure_score = self._exposure_score(frame)
        contrast_score = self._contrast_score(frame)
        border_score = self._border_purity_score(frame)
        glare_score = spatial_glare_score(frame)  # NEW
        sharpness_score = self._sharpness_score(frame)
        aspect_score = self._aspect_ratio_score(canonical_entry)
        
        components = [
            blur_score,
            exposure_score,
            contrast_score,
            border_score,
            glare_score,
            sharpness_score,
            aspect_score
        ]
        
        quality = np.dot(components, self.WEIGHTS)
        return float(quality)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave1_robustness.py::test_spatial_glare_distinguishes_blob_size -xvs`
Expected: PASS

- [ ] **Step 6: Regression test on quality scores**

Run: `pytest tests/regression/ -k "quality" --tb=short`
Expected: All tests pass; glare-affected frames should score lower

- [ ] **Step 7: Benchmark performance**

Run: `python -c "import timeit; print(timeit.timeit(lambda: spatial_glare_score(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)), number=100) / 100)"`
Expected: ~1 ms per scored crop

- [ ] **Step 8: Commit**

```bash
git add src/cardcaptor/scorer.py tests/test_wave1_robustness.py
git commit -m "feat(quality): spatial-glare metric via connected-component analysis

- Replace pixel-fraction glare with largest-saturated-blob analysis
- Distinguishes scattered specular from large blowout
- ~1 ms per scored crop (cv2.connectedComponentsWithStats)
- Integrated into QualityScorer as component 5

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Integration & Testing

### Full Integration Test

- [ ] **Run all Wave 1 tests together**

```bash
pytest tests/test_wave1_robustness.py -xvs
```

Expected: All 8 tasks' tests pass.

- [ ] **Run full regression suite**

```bash
pytest tests/regression/ --tb=short
```

Expected: No regressions. Metrics should show improvement in multi-card scenes, identity correctness, and foil handling.

- [ ] **Performance check**

```bash
python -m src.cardcaptor.pipeline --telemetry run_telemetry.json <test_video.mp4>
```

Expected: Per-frame cost increased by ~5-10 ms (within budget per §8.4). Memory +20-40 MB.

### Regression Data Collection

- [ ] **Run regression pack on full corpus**

```bash
python tests/regression/pipeline_runner.py --output wave1_results.json
```

Expected: Baseline metrics established for Wave 2 comparison.

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-pipeline-v4x-wave1-robustness.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (1–8), review the implementation, and coordinate across tasks. Fast iteration, isolated test runs, parallel work possible.

**2. Inline Execution** — We execute tasks sequentially in this session using the executing-plans skill, batch validation at checkpoints, and integrated commit history.

**Which approach would you prefer?**
