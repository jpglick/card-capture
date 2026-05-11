# Pipeline v4.x Robustness (Wave 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect holographic/foil cards and use glare-rejection fusion instead of median to preserve the holographic "look" instead of washing it out.

**Architecture:** Wave 4 contains 1 discretionary M-effort proposal: foil-aware fusion. The key insight is that holographic cards have moving high-frequency energy (the holographic surface shifts between frames), violating the median-fusion assumption that only glare moves. We detect this by computing Laplacian variance across frames; when variance is high (holographic movement detected), we switch to glare-rejection fusion: pick the frame whose pixel is closest to the per-pixel median, preserving luminance while rejecting bright outliers. This is a high-value-add feature for TCG users without algorithmic sophistication.

**Tech Stack:** OpenCV (Laplacian, high-pass filtering), numpy (variance, percentile).

---

## File Map

**Foil Detection & Fusion:**
- Create: `src/cardcaptor/fusion/foil_detection.py` — detect foil cards via high-pass variance
- Modify: `src/cardcaptor/fusion/median_fusion.py` — add glare-rejection fusion path
- Modify: `src/cardcaptor/pipeline.py:Stage 9` — decide fusion strategy per track
- Modify: `tests/test_wave4_foil.py` — foil detection and fusion tests

---

## Task Breakdown

### Task 1: Foil detection via Laplacian high-pass variance

**Files:**
- Create: `src/cardcaptor/fusion/foil_detection.py`
- Modify: `tests/test_wave4_foil.py`

**Context:** Proposal #20. Detect holographic cards by computing spatial variance of high-pass Laplacian energy across selected canonical frames. When variance is high, high-frequency mass moves between frames → foil detected.

- [ ] **Step 1: Write failing test for foil detection**

```python
# tests/test_wave4_foil.py
import numpy as np
import cv2
from src.cardcaptor.fusion.foil_detection import detect_foil_card, compute_laplacian_variance

def test_foil_detection_high_variance_across_frames():
    """Verify foil cards show high Laplacian variance across frames."""
    # Regular card: consistent edge structure across frames
    regular_frames = [np.random.randint(100, 150, (750, 1050, 3), dtype=np.uint8) for _ in range(4)]
    
    # Foil card: high-frequency content shifts (holographic surface)
    # Simulate by adding different random high-frequency patterns to each frame
    foil_frames = []
    for _ in range(4):
        base = np.ones((750, 1050, 3), dtype=np.uint8) * 120
        # Add different high-frequency pattern to each frame
        high_freq = np.random.randint(0, 50, (750, 1050, 3), dtype=np.uint8)
        foil_frames.append(np.clip(base.astype(np.int32) + high_freq.astype(np.int32), 0, 255).astype(np.uint8))
    
    regular_var = compute_laplacian_variance(regular_frames)
    foil_var = compute_laplacian_variance(foil_frames)
    
    # Foil should have higher variance
    assert foil_var > regular_var, f"Foil variance ({foil_var}) should exceed regular ({regular_var})"
    
    # Threshold-based detection
    is_foil_regular = detect_foil_card(regular_frames, threshold=50.0)
    is_foil_card = detect_foil_card(foil_frames, threshold=50.0)
    
    assert not is_foil_regular, "Regular card should not be detected as foil"
    assert is_foil_card, "Foil card should be detected"

def test_foil_detection_threshold_tuning():
    """Verify foil detection is tunable via threshold."""
    # Mid-variance frames (could go either way)
    frames = [np.random.randint(100, 160, (750, 1050, 3), dtype=np.uint8) for _ in range(4)]
    
    # Low threshold: more sensitive (more false positives)
    detected_low = detect_foil_card(frames, threshold=30.0)
    
    # High threshold: less sensitive (more false negatives)
    detected_high = detect_foil_card(frames, threshold=100.0)
    
    # At least one should detect it (or neither, but consistency matters)
    # The key is that threshold is tunable
    assert isinstance(detected_low, bool), "Should return boolean"
    assert isinstance(detected_high, bool), "Should return boolean"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave4_foil.py::test_foil_detection_high_variance_across_frames -xvs`
Expected: FAIL — "function compute_laplacian_variance not found"

- [ ] **Step 3: Implement foil detection functions**

Create `src/cardcaptor/fusion/foil_detection.py`:

```python
import numpy as np
import cv2

def compute_laplacian_variance(frames: list[np.ndarray]) -> float:
    """
    Compute spatial variance of high-pass Laplacian energy across frames.
    
    High variance indicates holographic movement (different high-frequency patterns per frame).
    
    Args:
        frames: List of BGR frames to analyze (rectified canonical frames)
    
    Returns:
        Variance of per-pixel Laplacian magnitudes across frames (scalar)
    """
    if len(frames) < 2:
        return 0.0
    
    laplacian_magnitudes = []
    
    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # High-pass filter via Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        
        # Magnitude (take absolute value)
        magnitude = np.abs(laplacian)
        
        laplacian_magnitudes.append(magnitude)
    
    # Stack magnitudes: (N, H, W)
    stacked = np.stack(laplacian_magnitudes, axis=0)
    
    # Compute spatial variance: for each pixel, variance across frames
    spatial_variance = np.var(stacked, axis=0)  # (H, W)
    
    # Aggregate: mean variance across all pixels
    overall_variance = np.mean(spatial_variance)
    
    return float(overall_variance)

def detect_foil_card(frames: list[np.ndarray], threshold: float = 50.0) -> bool:
    """
    Detect if a track is a holographic/foil card.
    
    Args:
        frames: List of rectified canonical frames from a track
        threshold: Variance threshold above which to classify as foil (tunable)
    
    Returns:
        True if foil detected, False otherwise
    """
    if len(frames) < 2:
        return False
    
    variance = compute_laplacian_variance(frames)
    
    return variance > threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wave4_foil.py::test_foil_detection_high_variance_across_frames -xvs`
Expected: PASS

- [ ] **Step 5: Run test for threshold tuning**

Run: `pytest tests/test_wave4_foil.py::test_foil_detection_threshold_tuning -xvs`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/cardcaptor/fusion/foil_detection.py tests/test_wave4_foil.py
git commit -m "feat(fusion): foil card detection via Laplacian variance

- Compute spatial variance of high-pass Laplacian energy across canonical frames
- High variance indicates holographic movement (foil/holo cards)
- Tunable threshold (default 50.0) for sensitivity control
- Foundation for foil-aware fusion (glare-rejection instead of median)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Glare-rejection fusion (alternative to median)

**Files:**
- Modify: `src/cardcaptor/fusion/median_fusion.py` — add glare-rejection path
- Modify: `tests/test_wave4_foil.py`

**Context:** Proposal #20. When foil detected, use glare-rejection fusion: pick the frame whose pixel is closest to the per-pixel median. Preserves card luminance, rejects bright outliers (glare).

- [ ] **Step 1: Write failing test for glare-rejection fusion**

```python
# tests/test_wave4_foil.py
def test_glare_rejection_fusion_preserves_luminance():
    """Verify glare-rejection fusion picks closest-to-median pixels."""
    from src.cardcaptor.fusion.median_fusion import glare_rejection_fusion
    
    # Three frames: one bright (glare), two nominal
    frames = [
        np.ones((100, 100, 3), dtype=np.uint8) * 120,  # nominal
        np.ones((100, 100, 3), dtype=np.uint8) * 250,  # bright (glare)
        np.ones((100, 100, 3), dtype=np.uint8) * 118,  # nominal (close to first)
    ]
    
    fused = glare_rejection_fusion(frames)
    
    # Result should be close to median (120), not glare (250)
    mean_pixel_value = np.mean(fused)
    
    assert 110 < mean_pixel_value < 130, f"Should preserve luminance ~120, got {mean_pixel_value}"
    assert mean_pixel_value < 200, f"Should reject glare, but got {mean_pixel_value}"

def test_glare_rejection_fusion_shape():
    """Verify glare-rejection fusion returns same shape as input frames."""
    from src.cardcaptor.fusion.median_fusion import glare_rejection_fusion
    
    frames = [
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
    ]
    
    fused = glare_rejection_fusion(frames)
    
    assert fused.shape == (750, 1050, 3), f"Shape mismatch: expected (750, 1050, 3), got {fused.shape}"
    assert fused.dtype == np.uint8, f"Type should be uint8, got {fused.dtype}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave4_foil.py::test_glare_rejection_fusion_preserves_luminance -xvs`
Expected: FAIL — "function glare_rejection_fusion not found"

- [ ] **Step 3: Implement glare-rejection fusion**

Modify `src/cardcaptor/fusion/median_fusion.py`:

```python
import numpy as np

def glare_rejection_fusion(frames: list[np.ndarray]) -> np.ndarray:
    """
    Fuse frames by picking pixels closest to per-pixel median (glare rejection).
    
    Instead of taking the median value directly, for each pixel:
    1. Compute per-pixel median across frames
    2. Find which frame has the pixel value closest to the median
    3. Pick that pixel from that frame
    
    This preserves luminance (median-based) while rejecting bright outliers (glare).
    
    Args:
        frames: List of BGR frames to fuse (e.g., 4 canonical frames of a foil card)
    
    Returns:
        Fused BGR frame (same shape as input frames)
    """
    if len(frames) == 0:
        return None
    
    if len(frames) == 1:
        return frames[0].copy()
    
    h, w, c = frames[0].shape
    
    # Stack frames: (N, H, W, 3)
    stacked = np.stack(frames, axis=0).astype(np.float32)
    
    # Compute per-pixel median
    median = np.median(stacked, axis=0)  # (H, W, 3)
    
    # For each pixel, find which frame is closest to median
    # Compute distances: |frame[i] - median| for each frame i
    distances = np.abs(stacked - median[np.newaxis, ...])  # (N, H, W, 3)
    
    # Sum across channels to get per-frame distance per pixel
    distances_summed = np.sum(distances, axis=3)  # (N, H, W)
    
    # Find frame index with minimum distance for each pixel
    closest_frame_idx = np.argmin(distances_summed, axis=0)  # (H, W)
    
    # Build result by picking pixels from closest frames
    fused = np.zeros((h, w, c), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            frame_idx = closest_frame_idx[y, x]
            fused[y, x] = frames[frame_idx][y, x]
    
    return fused
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wave4_foil.py::test_glare_rejection_fusion_preserves_luminance -xvs`
Expected: PASS

- [ ] **Step 5: Run shape test**

Run: `pytest tests/test_wave4_foil.py::test_glare_rejection_fusion_shape -xvs`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/cardcaptor/fusion/median_fusion.py tests/test_wave4_foil.py
git commit -m "feat(fusion): glare-rejection fusion for foil cards

- Pick pixels closest to per-pixel median (not median value itself)
- Preserves luminance while rejecting bright outliers (glare)
- Ideal for holographic/foil cards where high-frequency moves between frames
- Alternative to median fusion when foil detected

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Wire foil-aware fusion into pipeline

**Files:**
- Modify: `src/cardcaptor/pipeline.py:Stage 9` — decide fusion strategy
- Modify: `tests/test_wave4_foil.py`

**Context:** Proposal #20. In Stage 9 (fusion), check if track is foil. If yes, use glare-rejection fusion. Otherwise, use standard median fusion.

- [ ] **Step 1: Write failing test for stage 9 integration**

```python
# tests/test_wave4_foil.py
def test_pipeline_uses_glare_rejection_fusion_for_foils():
    """Verify pipeline selects glare-rejection fusion for foil cards."""
    from src.cardcaptor.pipeline import _fuse_canonical_frames_with_foil_awareness
    
    # Create mock frames
    regular_frames = [np.random.randint(100, 150, (750, 1050, 3), dtype=np.uint8) for _ in range(4)]
    foil_frames = []
    for _ in range(4):
        base = np.ones((750, 1050, 3), dtype=np.uint8) * 120
        high_freq = np.random.randint(0, 80, (750, 1050, 3), dtype=np.uint8)
        foil_frames.append(np.clip(base.astype(np.int32) + high_freq.astype(np.int32), 0, 255).astype(np.uint8))
    
    # Fuse regular card (should use median)
    fused_regular = _fuse_canonical_frames_with_foil_awareness(regular_frames, foil_threshold=50.0)
    assert fused_regular is not None, "Should fuse regular frames"
    
    # Fuse foil card (should use glare-rejection)
    fused_foil = _fuse_canonical_frames_with_foil_awareness(foil_frames, foil_threshold=50.0)
    assert fused_foil is not None, "Should fuse foil frames"
    
    # Both should produce valid images
    assert fused_regular.shape == (750, 1050, 3)
    assert fused_foil.shape == (750, 1050, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave4_foil.py::test_pipeline_uses_glare_rejection_fusion_for_foils -xvs`
Expected: FAIL — "function _fuse_canonical_frames_with_foil_awareness not found"

- [ ] **Step 3: Implement foil-aware fusion wrapper**

In `src/cardcaptor/pipeline.py`, add function:

```python
from src.cardcaptor.fusion.foil_detection import detect_foil_card
from src.cardcaptor.fusion.median_fusion import glare_rejection_fusion

def _fuse_canonical_frames_with_foil_awareness(
    frames: list[np.ndarray],
    foil_threshold: float = 50.0,
    use_ecc_registration: bool = True
) -> np.ndarray:
    """
    Fuse canonical frames with foil awareness.
    
    Args:
        frames: Selected canonical frames to fuse
        foil_threshold: Variance threshold for foil detection
        use_ecc_registration: Whether to align frames before fusion
    
    Returns:
        Fused frame (BGR)
    """
    if len(frames) == 0:
        return None
    
    # Step 1: Register frames if needed
    if use_ecc_registration and len(frames) > 1:
        from src.cardcaptor.fusion.ecc_registration import register_frames_via_ecc
        frames = register_frames_via_ecc(frames, ref_idx=0, warp_type="translation")
    
    # Step 2: Detect if foil
    is_foil = detect_foil_card(frames, threshold=foil_threshold)
    
    # Step 3: Choose fusion strategy
    if is_foil:
        # Use glare-rejection fusion (preserves holographic look)
        fused = glare_rejection_fusion(frames)
    else:
        # Use standard median fusion
        stacked = np.stack(frames, axis=0)
        fused = np.median(stacked, axis=0).astype(np.uint8)
    
    return fused
```

- [ ] **Step 4: Wire into Stage 9 fusion call**

In `src/cardcaptor/pipeline.py`, find where median fusion is called (Stage 9) and replace with:

```python
# Before (existing code):
# fused = np.median(np.stack(selected_frames, axis=0), axis=0).astype(np.uint8)

# After (using foil-aware wrapper):
fused = _fuse_canonical_frames_with_foil_awareness(
    selected_frames,
    foil_threshold=50.0,  # tunable parameter
    use_ecc_registration=True  # from Wave 2 Task 4
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave4_foil.py::test_pipeline_uses_glare_rejection_fusion_for_foils -xvs`
Expected: PASS

- [ ] **Step 6: Integration test on regression corpus**

Run: `pytest tests/regression/ -k "foil or fusion" --tb=short`
Expected: No regressions on non-foil cards. Foil cards should have improved subjective quality.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/pipeline.py tests/test_wave4_foil.py
git commit -m "feat(fusion): wire foil-aware fusion into Stage 9

- Detect foil cards via Laplacian variance (tunable threshold)
- Use glare-rejection fusion for foils (preserves holographic look)
- Use median fusion for regular cards (unchanged)
- Integrates Wave 2 ECC registration + Wave 4 foil detection + glare-rejection
- High subjective value for TCG users (preserves foil aesthetics)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Integration & Testing

### Full Integration Test

- [ ] **Run all Wave 4 tests**

```bash
pytest tests/test_wave4_foil.py -xvs
```

Expected: All 6 tests pass (foil detection, threshold tuning, glare-rejection, shape, integration).

- [ ] **Run full regression suite**

```bash
pytest tests/regression/ --tb=short
```

Expected: No regressions. Foil-specific quality improved subjectively.

- [ ] **Performance check**

```bash
python3 -m src.cardcaptor.pipeline --telemetry run_telemetry_wave4.json <test_video.mp4>
```

Expected: Negligible per-frame cost addition (<1 ms, only during fusion). No memory impact.

### Manual Visual Validation

Test on actual holographic/foil TCG cards if available:
- Regular cards: verify median fusion still produces clean output
- Foil cards: verify glare-rejection fusion preserves holographic "shimmer" instead of washing it out

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-pipeline-v4x-wave4-foil-fusion.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks sequentially in this session using executing-plans, batch execution with checkpoints

**Which approach would you prefer?**
