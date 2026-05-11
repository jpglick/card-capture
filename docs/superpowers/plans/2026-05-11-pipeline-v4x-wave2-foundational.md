# Pipeline v4.x Robustness (Wave 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational robustness metrics and algorithmic improvements (per-pixel BG, oriented-IoU, ECC fusion, per-region valleys, per-tile occlusion) that unblock Wave 3 calibration work.

**Architecture:** Wave 2 contains 6 independent M-effort proposals. Task 1 (robustness metric pack) is foundational—it defines how all subsequent waves are measured. Tasks 2-6 are algorithmic improvements that can ship in parallel once #1 establishes the measurement framework. Together they add ~15-20 ms per-frame cost (mostly ECC and occlusion residual), prepare the pipeline for learned calibration, and establish the regression corpus as the source of truth.

**Tech Stack:** OpenCV (ECC, connected components, Sobel), numpy (per-pixel operations, EWMA), Shapely (oriented-IoU), scipy (tile statistics), pytest (regression suite).

---

## File Map

**Core Robustness & Metrics:**
- Create: `src/cardcaptor/metrics/robustness_pack.py` — card recall, precision, F1, multi-card survival, foil survival
- Modify: `tests/regression/truth.py` — extend schema with scene_type, foil_label
- Modify: `tests/regression/metrics.py` — compute robustness metrics from truth labels
- Create: `tests/regression/test_robustness_metrics.py` — unit tests for metric computation

**Background Model (Per-Pixel):**
- Modify: `src/cardcaptor/presence/background_novelty.py:BackgroundModel` — add variance tracking
- Modify: `src/cardcaptor/pipeline.py` — wire Mahalanobis novelty into Stage 4 gate
- Modify: `tests/test_wave2_robustness.py` — per-pixel BG tests

**Multi-Card Handling:**
- Modify: `src/cardcaptor/tracking/botsort_adapter.py` — implement oriented-IoU metric
- Modify: `src/cardcaptor/pipeline.py` — substitute rotated_iou for axis-aligned IoU in adapter
- Modify: `src/cardcaptor/presence/valley_detection.py` — implement per-region valley detection
- Modify: `tests/test_wave2_robustness.py` — multi-card and valley tests

**Fusion & Quality:**
- Modify: `src/cardcaptor/fusion/median_fusion.py` — add ECC re-registration before median
- Modify: `src/cardcaptor/scoring.py` — add per-tile occlusion residual as quality penalty
- Modify: `tests/test_wave2_robustness.py` — fusion and occlusion tests

**Calibration Harness:**
- Create: `scripts/calibrate_wave2.py` — grid sweep over adaptive thresholds (Wave 3 prep)

---

## Task Breakdown

### Task 1: Robustness metric pack (foundational for Wave 3)

**Files:**
- Create: `src/cardcaptor/metrics/robustness_pack.py`
- Modify: `tests/regression/truth.py` — add scene_type, foil_label columns
- Modify: `tests/regression/metrics.py` — add metric computation
- Create: `tests/regression/test_robustness_metrics.py`

**Context:** Proposal #9. Define five metrics that measure robustness: card recall, card precision, Front/Back F1, multi-card scene survival, foil survival. These become the source of truth for tuning thresholds in Wave 3.

- [ ] **Step 1: Write failing test for robustness metrics**

```python
# tests/regression/test_robustness_metrics.py
import pytest
from src.cardcaptor.metrics.robustness_pack import RobustnessMetrics

def test_card_recall_counts_matched_cards():
    """Verify card recall = matched_cards / total_truth_cards."""
    truth_labels = [
        {"card_id": "A", "side": "Front"},
        {"card_id": "B", "side": "Front"},
        {"card_id": "C", "side": "Front"},
    ]
    predictions = [
        {"canonical_id": 1, "matched_card_id": "A"},
        {"canonical_id": 2, "matched_card_id": "B"},
        # C is missing (false negative)
    ]
    
    metrics = RobustnessMetrics(truth_labels, predictions)
    recall = metrics.card_recall()
    
    assert recall == 2 / 3, f"Expected recall 0.667, got {recall}"

def test_front_back_f1_measures_angle_accuracy():
    """Verify Front/Back F1 measures correctness of angle assignment."""
    truth_labels = [
        {"card_id": "A", "side": "Front"},
        {"card_id": "B", "side": "Back"},
    ]
    predictions = [
        {"canonical_id": 1, "matched_card_id": "A", "side": "Front"},   # correct
        {"canonical_id": 2, "matched_card_id": "B", "side": "Front"},   # wrong (should be Back)
    ]
    
    metrics = RobustnessMetrics(truth_labels, predictions)
    f1 = metrics.front_back_f1()
    
    # 1 correct Front, 0 correct Back; precision=0.5, recall=0.5, F1=0.5
    assert abs(f1 - 0.5) < 0.01, f"Expected F1 ~0.5, got {f1}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_robustness_metrics.py -xvs`
Expected: FAIL — "RobustnessMetrics class not found"

- [ ] **Step 3: Extend truth.py schema with scene metadata**

In `tests/regression/truth.py`, update the CardTruth dataclass:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CardTruth:
    card_id: str              # Unique identifier for physical card
    side: str                 # "Front" or "Back"
    bounding_box: tuple       # (x1, y1, x2, y2) in source frame
    corners: list             # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    # NEW FIELDS for Wave 2 metrics:
    scene_type: str           # "single_card", "multi_card", "rapid_swap"
    foil_label: Optional[str] = None  # "foil", "holo", None (regular)
    is_occluded: bool = False         # True if any occlusion in truth labels
    occlusion_type: Optional[str] = None  # "finger", "sticker", "adjacent_card", "lens_flare", None
```

- [ ] **Step 4: Implement RobustnessMetrics class**

Create `src/cardcaptor/metrics/robustness_pack.py`:

```python
import numpy as np
from typing import List, Dict

class RobustnessMetrics:
    """Compute robustness metrics against ground truth."""
    
    def __init__(self, truth_labels: List[Dict], predictions: List[Dict]):
        """
        Args:
            truth_labels: List of dicts with keys: card_id, side, scene_type, foil_label
            predictions: List of dicts with keys: canonical_id, matched_card_id, side
        """
        self.truth = truth_labels
        self.predictions = predictions
        self._match_predictions_to_truth()
    
    def _match_predictions_to_truth(self):
        """Build mapping from prediction to ground truth."""
        self.matched_pairs = []  # (truth_idx, pred_idx) pairs
        
        for pred in self.predictions:
            if pred.get("matched_card_id") is None:
                continue
            for truth_idx, truth in enumerate(self.truth):
                if truth["card_id"] == pred["matched_card_id"]:
                    self.matched_pairs.append((truth_idx, self.predictions.index(pred)))
    
    def card_recall(self) -> float:
        """Fraction of ground-truth cards that produced ≥1 canonical entry."""
        matched_truth_ids = set(self.truth[t_idx]["card_id"] for t_idx, _ in self.matched_pairs)
        all_truth_ids = set(t["card_id"] for t in self.truth)
        
        if len(all_truth_ids) == 0:
            return 1.0
        return len(matched_truth_ids) / len(all_truth_ids)
    
    def card_precision(self) -> float:
        """Fraction of canonical entries matching a ground-truth card."""
        if len(self.predictions) == 0:
            return 1.0
        matched_count = len(self.matched_pairs)
        return matched_count / len(self.predictions)
    
    def front_back_f1(self) -> float:
        """F1 score for correctness of Front/Back angle assignment."""
        correct_front = 0
        correct_back = 0
        total_front_truth = 0
        total_back_truth = 0
        
        for truth_idx, pred_idx in self.matched_pairs:
            truth_side = self.truth[truth_idx]["side"]
            pred_side = self.predictions[pred_idx].get("side", "Front")
            
            if truth_side == "Front":
                total_front_truth += 1
                if pred_side == "Front":
                    correct_front += 1
            else:  # Back
                total_back_truth += 1
                if pred_side == "Back":
                    correct_back += 1
        
        if (total_front_truth + total_back_truth) == 0:
            return 1.0
        
        recall = (correct_front + correct_back) / (total_front_truth + total_back_truth)
        
        if len(self.predictions) == 0:
            precision = 1.0
        else:
            correct_total = correct_front + correct_back
            precision = correct_total / len(self.matched_pairs) if self.matched_pairs else 0.0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def multi_card_scene_survival(self) -> float:
        """Card recall restricted to multi_card / rapid_swap scenes."""
        multi_card_truth = [t for t in self.truth if t.get("scene_type") in ("multi_card", "rapid_swap")]
        if len(multi_card_truth) == 0:
            return 1.0
        
        matched_count = sum(
            1 for t_idx, _ in self.matched_pairs
            if self.truth[t_idx].get("scene_type") in ("multi_card", "rapid_swap")
        )
        return matched_count / len(multi_card_truth)
    
    def foil_survival(self) -> float:
        """Card recall + fusion-quality on foil/holo cards."""
        foil_truth = [t for t in self.truth if t.get("foil_label") in ("foil", "holo")]
        if len(foil_truth) == 0:
            return 1.0
        
        matched_count = sum(
            1 for t_idx, _ in self.matched_pairs
            if self.truth[t_idx].get("foil_label") in ("foil", "holo")
        )
        return matched_count / len(foil_truth)
    
    def compute_all(self) -> Dict[str, float]:
        """Return all metrics as a dict."""
        return {
            "card_recall": self.card_recall(),
            "card_precision": self.card_precision(),
            "front_back_f1": self.front_back_f1(),
            "multi_card_survival": self.multi_card_scene_survival(),
            "foil_survival": self.foil_survival(),
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/regression/test_robustness_metrics.py -xvs`
Expected: PASS

- [ ] **Step 6: Integrate metrics into regression runner**

Update `tests/regression/metrics.py` to compute and report robustness metrics after each regression run:

```python
# In metrics.py, add function:
def report_robustness_metrics(truth_labels, predictions):
    from src.cardcaptor.metrics.robustness_pack import RobustnessMetrics
    
    metrics = RobustnessMetrics(truth_labels, predictions)
    all_metrics = metrics.compute_all()
    
    print("\n=== ROBUSTNESS METRICS ===")
    for metric_name, value in all_metrics.items():
        print(f"{metric_name}: {value:.3f}")
    
    return all_metrics
```

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/metrics/robustness_pack.py tests/regression/truth.py tests/regression/metrics.py tests/regression/test_robustness_metrics.py
git commit -m "feat(metrics): add robustness metric pack (card recall, precision, F1, multi-card, foil)

- Define 5 core metrics for pipeline robustness assessment
- Extend truth.py schema with scene_type, foil_label, occlusion metadata
- RobustnessMetrics class computes all metrics against ground truth
- Foundation for Wave 3 threshold auto-sweep and adaptive tuning

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Per-pixel background model with variance (Mahalanobis novelty)

**Files:**
- Modify: `src/cardcaptor/presence/background_novelty.py:BackgroundModel`
- Modify: `src/cardcaptor/pipeline.py:_filter_candidates_by_novelty` — call quad_novelty with variance
- Modify: `tests/test_wave2_robustness.py`

**Context:** Proposal #10. Replace single grayscale mean with per-pixel mean + variance. Novelty becomes Mahalanobis-style score: `clip((|frame - mean| - k·sqrt(var)) / 255, 0, 1)`.

- [ ] **Step 1: Write failing test for per-pixel variance BG model**

```python
# tests/test_wave2_robustness.py
import numpy as np
from src.cardcaptor.presence.background_novelty import BackgroundModel, quad_novelty

def test_per_pixel_bg_tracks_variance():
    """Verify BG model stores per-pixel variance."""
    # Create two proxy frames with different variance at different pixels
    frame1 = np.ones((100, 100), dtype=np.uint8) * 128
    frame1[0:50, 0:50] = 150  # High variance region
    
    frame2 = np.ones((100, 100), dtype=np.uint8) * 120
    frame2[0:50, 0:50] = 145  # Same region, different values
    
    bg_model = BackgroundModel([frame1, frame2])
    
    # Check that variance is stored
    assert hasattr(bg_model, 'variance'), "BG model should have variance attribute"
    assert bg_model.variance.shape == (100, 100), "Variance should match frame shape"
    assert bg_model.variance[25, 25] > bg_model.variance[75, 75], "High-variance region should have higher variance"

def test_mahalanobis_novelty_with_variance():
    """Verify Mahalanobis novelty accounts for variance."""
    # BG: mean=128, low variance
    bg_frame = np.ones((100, 100), dtype=np.uint8) * 128
    bg_model = BackgroundModel([bg_frame])
    
    # Card: +20 from mean (noticeable but within noise for high-variance areas)
    card_frame = np.ones((100, 100), dtype=np.uint8) * 148
    
    novelty = quad_novelty(card_frame, bg_model, color_space="lab", use_variance=True, k=2.0)
    
    # Mahalanobis should produce moderate novelty (not extreme)
    assert 0.1 < novelty.mean() < 0.9, f"Expected moderate novelty, got {novelty.mean()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave2_robustness.py::test_per_pixel_bg_tracks_variance -xvs`
Expected: FAIL — "BG model should have variance attribute"

- [ ] **Step 3: Extend BackgroundModel to track variance**

In `src/cardcaptor/presence/background_novelty.py`:

```python
class BackgroundModel:
    def __init__(self, proxy_frames: list[np.ndarray]):
        """Build BG model from low-presence-score proxy frames."""
        # Compute per-pixel mean and variance
        self.mean_bgr = np.mean([f for f in proxy_frames], axis=0).astype(np.uint8)
        self.gray = cv2.cvtColor(self.mean_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        
        # NEW: Per-pixel variance for Mahalanobis novelty
        # Convert frames to float for variance computation
        frames_float = [f.astype(np.float32) for f in proxy_frames]
        frames_stacked = np.stack(frames_float, axis=0)  # (N, H, W, 3)
        
        # Variance per pixel, per channel
        self.variance_bgr = np.var(frames_stacked, axis=0).astype(np.float32)  # (H, W, 3)
        self.variance_gray = np.var([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in proxy_frames], axis=0).astype(np.float32)  # (H, W)
        
        self.alpha = 0.1  # EWMA decay for refresh
    
    def refresh_from_frame(self, frame_bgr: np.ndarray):
        """Update BG model (already implemented in Wave 1)."""
        # Existing implementation
        ...
    
    def get_mean_bgr(self) -> np.ndarray:
        return self.mean_bgr
    
    def get_variance_bgr(self) -> np.ndarray:
        """Return per-pixel variance for Mahalanobis computation."""
        return self.variance_bgr
```

- [ ] **Step 4: Implement Mahalanobis novelty in quad_novelty**

Update `quad_novelty()` function in background_novelty.py:

```python
def quad_novelty(
    frame_bgr: np.ndarray,
    bg_model: BackgroundModel,
    polygon: np.ndarray = None,
    color_space: str = "lab",
    lab_weights: tuple[float, float, float] = (1.0, 0.5, 0.5),
    use_variance: bool = True,
    k: float = 2.0
) -> np.ndarray:
    """
    Compute pixel-wise novelty: deviation from background model.
    
    Args:
        frame_bgr: Input frame in BGR
        bg_model: BackgroundModel with mean and optional variance
        polygon: Optional mask polygon (shape of card OBB)
        color_space: "grayscale" or "lab"
        lab_weights: Weights for (L, a, b) channels if color_space="lab"
        use_variance: If True, use Mahalanobis-style scoring with variance
        k: Mahalanobis k parameter (standard deviations to tolerate)
    
    Returns:
        Novelty score ∈ [0, 1] per pixel
    """
    if color_space == "lab":
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
        bg_lab = cv2.cvtColor(bg_model.get_mean_bgr(), cv2.COLOR_BGR2Lab)
        
        frame_lab = frame_lab.astype(np.float32) / 255.0
        bg_lab = bg_lab.astype(np.float32) / 255.0
        
        # Compute per-channel differences
        diff_l = np.abs(frame_lab[..., 0] - bg_lab[..., 0]) * lab_weights[0]
        diff_a = np.abs(frame_lab[..., 1] - bg_lab[..., 1]) * lab_weights[1]
        diff_b = np.abs(frame_lab[..., 2] - bg_lab[..., 2]) * lab_weights[2]
        
        if use_variance and hasattr(bg_model, 'variance_bgr'):
            # Mahalanobis: (diff - k*std) / scale
            var_bgr = bg_model.get_variance_bgr()
            std_l = np.sqrt(var_bgr[..., 0] / 255.0) * lab_weights[0]
            std_a = np.sqrt(var_bgr[..., 1] / 255.0) * lab_weights[1]
            std_b = np.sqrt(var_bgr[..., 2] / 255.0) * lab_weights[2]
            
            novelty_l = np.clip((diff_l - k * std_l) / 0.3, 0, 1)
            novelty_a = np.clip((diff_a - k * std_a) / 0.3, 0, 1)
            novelty_b = np.clip((diff_b - k * std_b) / 0.3, 0, 1)
            
            novelty = (novelty_l + novelty_a + novelty_b) / 3.0
        else:
            # Simple L1 novelty (no variance weighting)
            novelty = np.clip((diff_l + diff_a + diff_b) / (sum(lab_weights) / 3.0), 0, 1)
    
    elif color_space == "grayscale":
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_gray = bg_model.gray.astype(np.float32)
        
        diff = np.abs(frame_gray - bg_gray) / 255.0
        
        if use_variance and hasattr(bg_model, 'variance_gray'):
            std = np.sqrt(bg_model.variance_gray) / 255.0
            novelty = np.clip((diff - k * std) / 0.3, 0, 1)
        else:
            novelty = np.clip(diff, 0, 1)
    
    else:
        raise ValueError(f"Unsupported color_space: {color_space}")
    
    # Apply mask if provided
    if polygon is not None:
        mask = np.zeros_like(novelty, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        novelty = novelty * mask
    
    return novelty
```

- [ ] **Step 5: Wire into pipeline Stage 4**

In `src/cardcaptor/pipeline.py`, update novelty gate calls:

```python
# In _filter_candidates_by_novelty():
novelty = quad_novelty(
    frame_bgr,
    bg_model,
    polygon,
    color_space="lab",
    lab_weights=(1.0, 0.5, 0.5),
    use_variance=True,  # NEW
    k=2.0  # Tolerate 2 std deviations
)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_wave2_robustness.py::test_per_pixel_bg_tracks_variance -xvs`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/presence/background_novelty.py src/cardcaptor/pipeline.py tests/test_wave2_robustness.py
git commit -m "feat(bg-model): per-pixel BG variance for Mahalanobis novelty

- Store per-pixel mean + variance instead of global mean
- Novelty becomes Mahalanobis-style: (diff - k*std) / scale
- Tolerates natural lighting variation better than fixed threshold
- Fallback to simple L1 if variance unavailable

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Oriented-IoU association for multi-card tracking

**Files:**
- Modify: `src/cardcaptor/tracking/botsort_adapter.py` — add rotated_iou() helper
- Modify: `src/cardcaptor/tracking/botsort_adapter.py:BotSortAdapter` — use rotated_iou in association
- Modify: `tests/test_wave2_robustness.py`

**Context:** Proposal #11. Two cards at 90° to each other produce overlapping axis-aligned bboxes and confuse the tracker. Use rotated IoU (OBB to OBB) instead.

- [ ] **Step 1: Write failing test for oriented-IoU**

```python
# tests/test_wave2_robustness.py
import numpy as np
from src.cardcaptor.tracking.botsort_adapter import rotated_iou

def test_rotated_iou_orthogonal_cards():
    """Verify rotated IoU correctly handles orthogonal cards."""
    # Card A: axis-aligned, corners at (0,0), (100,0), (100,100), (0,100)
    card_a = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    
    # Card B: 90° rotated, same center as A, corners at (40, -40), (140, 40), (100, 140), (0, 100)
    # Centered at (50, 50), rotated 90°
    center = np.array([50, 50])
    card_b = np.array([
        [40, -40],   # top-left rotated
        [140, 40],   # top-right rotated
        [100, 140],  # bottom-right rotated
        [0, 100]     # bottom-left rotated
    ], dtype=np.float32)
    
    iou = rotated_iou(card_a, card_b)
    
    # Orthogonal cards should have low IoU (minimal overlap despite axis-aligned overlap)
    assert iou < 0.3, f"Orthogonal cards should have low IoU, got {iou}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave2_robustness.py::test_rotated_iou_orthogonal_cards -xvs`
Expected: FAIL — "function rotated_iou not found"

- [ ] **Step 3: Implement rotated_iou using Shapely**

In `src/cardcaptor/tracking/botsort_adapter.py`, add:

```python
import numpy as np
from shapely.geometry import Polygon

def rotated_iou(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two rotated bounding boxes.
    
    Args:
        corners_a: (4, 2) array of corner coordinates for OBB A
        corners_b: (4, 2) array of corner coordinates for OBB B
    
    Returns:
        IoU score ∈ [0, 1]
    """
    try:
        # Convert corners to Shapely Polygon
        poly_a = Polygon(corners_a)
        poly_b = Polygon(corners_b)
        
        # Compute intersection and union
        intersection = poly_a.intersection(poly_b).area
        union = poly_a.area + poly_b.area - intersection
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    except Exception:
        # Fallback to axis-aligned IoU if rotation computation fails
        return axis_aligned_iou_from_corners(corners_a, corners_b)

def axis_aligned_iou_from_corners(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Fallback: compute axis-aligned IoU from corner points."""
    x1_a, y1_a = corners_a.min(axis=0)
    x2_a, y2_a = corners_a.max(axis=0)
    
    x1_b, y1_b = corners_b.min(axis=0)
    x2_b, y2_b = corners_b.max(axis=0)
    
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - intersection
    
    return float(intersection / union) if union > 0 else 0.0
```

- [ ] **Step 4: Integrate rotated_iou into BotSortAdapter**

Update `BotSortAdapter.update()` or the association method to use rotated_iou:

```python
# In BotSortAdapter, where association happens:
# Replace axis-aligned IoU calculation with rotated_iou

def compute_iou_matrix(detections, tracks):
    """Compute IoU matrix using rotated IoU."""
    n_detections = len(detections)
    n_tracks = len(tracks)
    iou_matrix = np.zeros((n_detections, n_tracks))
    
    for i, det in enumerate(detections):
        det_corners = np.array(det.corners, dtype=np.float32)  # OBB corners
        
        for j, track in enumerate(tracks):
            track_corners = np.array(track.estimated_bbox, dtype=np.float32)  # OBB corners
            
            iou_matrix[i, j] = rotated_iou(det_corners, track_corners)
    
    return iou_matrix
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave2_robustness.py::test_rotated_iou_orthogonal_cards -xvs`
Expected: PASS

- [ ] **Step 6: Integration test on multi-card scenes**

Run: `pytest tests/ -k "multi_card" --tb=short`
Expected: Multi-card tracking tests pass; no false associations between orthogonal cards.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/tracking/botsort_adapter.py tests/test_wave2_robustness.py
git commit -m "feat(tracking): oriented-IoU association for multi-card scenes

- Replace axis-aligned IoU with rotated IoU (Shapely Polygon intersection)
- Correctly handles cards at 90° to each other (no spurious overlap)
- Fallback to axis-aligned IoU if rotation fails
- Improves tracker association in multi-card and rapid-swap scenarios

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: ECC re-registration before median fusion

**Files:**
- Create: `src/cardcaptor/fusion/ecc_registration.py` — ECC warp computation
- Modify: `src/cardcaptor/fusion/median_fusion.py` — apply ECC before median
- Modify: `tests/test_wave2_robustness.py`

**Context:** Proposal #12. Corner detector wobbles ±1-3 px frame-to-frame. Current median ghosts edges. Run `cv2.findTransformECC` (translation or Euclidean) on each non-reference frame, apply warp, then median.

- [ ] **Step 1: Write failing test for ECC registration**

```python
# tests/test_wave2_robustness.py
import numpy as np
import cv2
from src.cardcaptor.fusion.ecc_registration import register_frames_via_ecc

def test_ecc_registers_shifted_frames():
    """Verify ECC aligns shifted copies of the same frame."""
    # Create reference frame
    ref = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Create shifted copy (translate by 2 pixels)
    warp_matrix = np.array([[1, 0, 2], [0, 1, 2]], dtype=np.float32)
    shifted = cv2.warpAffine(ref, warp_matrix, (100, 100))
    
    # Register shifted frame back to reference
    frames = [ref, shifted]
    aligned_frames = register_frames_via_ecc(frames, ref_idx=0, warp_type="translation")
    
    # Aligned frame should closely match reference
    diff = cv2.absdiff(aligned_frames[1], ref)
    mean_diff = np.mean(diff)
    
    assert mean_diff < 10, f"Aligned frame should closely match reference, got mean diff {mean_diff}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave2_robustness.py::test_ecc_registers_shifted_frames -xvs`
Expected: FAIL — "function register_frames_via_ecc not found"

- [ ] **Step 3: Implement ECC registration**

Create `src/cardcaptor/fusion/ecc_registration.py`:

```python
import numpy as np
import cv2

def compute_ecc_warp(src: np.ndarray, ref: np.ndarray, warp_type: str = "Euclidean") -> np.ndarray:
    """
    Compute ECC (Enhanced Correlation Coefficient) warp matrix from src to ref.
    
    Args:
        src: Source frame (uint8 BGR)
        ref: Reference frame (uint8 BGR)
        warp_type: "translation", "Euclidean", "affine", or "homography"
    
    Returns:
        Warp matrix (2x3 for translation/Euclidean/affine, 3x3 for homography)
    """
    # Convert to grayscale for ECC
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Initialize warp matrix
    if warp_type == "translation":
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    elif warp_type == "Euclidean":
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    elif warp_type == "affine":
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    else:  # homography
        warp_matrix = np.eye(3, dtype=np.float32)
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)
    
    # Compute ECC
    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_gray, src_gray, warp_matrix,
            cv2.MOTION_TRANSLATION if warp_type == "translation" else cv2.MOTION_EUCLIDEAN,
            criteria
        )
    except cv2.error:
        # Convergence failed; return identity
        if warp_type == "homography":
            return np.eye(3, dtype=np.float32)
        else:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    return warp_matrix

def register_frames_via_ecc(frames: list[np.ndarray], ref_idx: int = 0, warp_type: str = "translation") -> list[np.ndarray]:
    """
    Register a list of frames to a reference frame using ECC.
    
    Args:
        frames: List of BGR frames to register
        ref_idx: Index of reference frame
        warp_type: Type of warp (translation, Euclidean, affine, homography)
    
    Returns:
        List of aligned frames
    """
    ref = frames[ref_idx]
    aligned_frames = []
    
    for i, frame in enumerate(frames):
        if i == ref_idx:
            aligned_frames.append(frame.copy())
        else:
            warp_matrix = compute_ecc_warp(frame, ref, warp_type)
            
            if warp_type == "homography":
                aligned = cv2.warpPerspective(frame, warp_matrix, (frame.shape[1], frame.shape[0]))
            else:
                aligned = cv2.warpAffine(frame, warp_matrix, (frame.shape[1], frame.shape[0]))
            
            aligned_frames.append(aligned)
    
    return aligned_frames
```

- [ ] **Step 4: Integrate into median fusion**

Modify `src/cardcaptor/fusion/median_fusion.py`:

```python
import cv2
import numpy as np
from src.cardcaptor.fusion.ecc_registration import register_frames_via_ecc

def fuse_canonical_frames(frames: list[np.ndarray], use_ecc_registration: bool = True) -> np.ndarray:
    """
    Fuse selected frames into a single canonical image.
    
    Args:
        frames: List of BGR frames to fuse
        use_ecc_registration: If True, align frames before fusion
    
    Returns:
        Fused BGR image
    """
    if len(frames) == 0:
        return None
    
    if use_ecc_registration and len(frames) > 1:
        # Register all frames to the first (median or sharpest) frame
        frames = register_frames_via_ecc(frames, ref_idx=0, warp_type="translation")
    
    # Compute median across registered frames
    frames_stacked = np.stack(frames, axis=0)  # (N, H, W, 3)
    fused = np.median(frames_stacked, axis=0).astype(np.uint8)
    
    return fused
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave2_robustness.py::test_ecc_registers_shifted_frames -xvs`
Expected: PASS

- [ ] **Step 6: Integration test on fusion quality**

Run: `pytest tests/ -k "fusion" --tb=short`
Expected: Fused images have less ghosting around edges; sharpness metrics improve.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/fusion/ecc_registration.py src/cardcaptor/fusion/median_fusion.py tests/test_wave2_robustness.py
git commit -m "feat(fusion): ECC re-registration before median fusion

- Align non-reference frames to reference via cv2.findTransformECC
- Eliminates ghosting from corner-detector wobble (±1-3 px jitter)
- Translation-based warp (cheap, most common case)
- Fallback to unregistered median if ECC convergence fails

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Per-region valley detection for multi-card swaps

**Files:**
- Modify: `src/cardcaptor/presence/valley_detection.py` — implement per-region valley detection
- Modify: `src/cardcaptor/pipeline.py:Stage 1` — wire per-region valleys
- Modify: `tests/test_wave2_robustness.py`

**Context:** Proposal #13. Global Sobel valley collapses to scalar; a "card A leaves left, card B enters right" swap with no global valley is missed. Tile proxy frame into 3×3 grid; emit split if any tile's valley persists ≥ valley_min_width_frames.

- [ ] **Step 1: Write failing test for per-region valley detection**

```python
# tests/test_wave2_robustness.py
import numpy as np
from src.cardcaptor.presence.valley_detection import per_region_valley_detection

def test_per_region_valley_detects_left_exit_right_entry():
    """Verify per-region valley detects swap on opposite sides."""
    # Create proxy frame sequence: card A leaves left, card B enters right
    # Region valley should detect this even if global Sobel is flat
    
    frames = []
    for t in range(10):
        frame = np.ones((192, 256), dtype=np.uint8) * 128
        
        if t < 5:
            # Card A on left
            frame[50:150, 20:80] = 200
        
        if t >= 5:
            # Card B on right
            frame[50:150, 170:230] = 200
        
        frames.append(frame)
    
    # Detect valleys per region
    valleys = per_region_valley_detection(frames, grid_size=3)
    
    # Should detect valley in left column (A exiting) and/or right column (B entering)
    left_col_valleys = sum(1 for v in valleys if v[1] == 0)  # column 0
    right_col_valleys = sum(1 for v in valleys if v[1] == 2)  # column 2
    
    assert (left_col_valleys > 0 or right_col_valleys > 0), "Should detect valley in left or right region"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave2_robustness.py::test_per_region_valley_detects_left_exit_right_entry -xvs`
Expected: FAIL — "function per_region_valley_detection not found"

- [ ] **Step 3: Implement per-region valley detection**

In `src/cardcaptor/presence/valley_detection.py`, add:

```python
import numpy as np
import cv2

def per_region_valley_detection(
    frames: list[np.ndarray],
    grid_size: int = 3,
    min_valley_width_frames: int = 3
) -> list[tuple[int, int, int]]:
    """
    Detect valleys per tile of proxy frame grid.
    
    Args:
        frames: List of grayscale proxy frames (H, W)
        grid_size: Divide frame into grid_size × grid_size tiles (default 3×3)
        min_valley_width_frames: Minimum consecutive frames with valley to report
    
    Returns:
        List of (frame_idx, grid_row, grid_col) tuples where valley detected
    """
    if len(frames) < 2:
        return []
    
    h, w = frames[0].shape
    tile_h = h // grid_size
    tile_w = w // grid_size
    
    valleys_detected = []
    
    # For each tile, compute per-frame Sobel values
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            
            # Compute Sobel edge density per frame in this tile
            sobel_values = []
            for frame in frames:
                tile = frame[y1:y2, x1:x2]
                sobel_x = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3)
                sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
                sobel_values.append(sobel_mag.mean())
            
            # Detect valleys (local minima in Sobel curve)
            sobel_values = np.array(sobel_values)
            
            for i in range(1, len(sobel_values) - 1):
                # Simple valley: drop from prior frame
                if sobel_values[i] < 0.3 * sobel_values[i-1]:  # 70% drop
                    # Check if valley persists
                    valley_width = 1
                    for j in range(i + 1, len(sobel_values)):
                        if sobel_values[j] < 0.3 * sobel_values[j-1]:
                            valley_width += 1
                        else:
                            break
                    
                    if valley_width >= min_valley_width_frames:
                        valleys_detected.append((i, row, col))
    
    return valleys_detected

def find_valley_splits_per_region(proxy_frame_sequence: list[np.ndarray], grid_size: int = 3) -> list[int]:
    """
    Find frame indices where a valley split (swap) is detected in any tile.
    
    Returns:
        List of frame indices where split detected
    """
    valleys = per_region_valley_detection(proxy_frame_sequence, grid_size=grid_size)
    
    # Aggregate valleys by frame
    split_frames = set(v[0] for v in valleys)
    return sorted(split_frames)
```

- [ ] **Step 4: Wire into pipeline Stage 1**

In `src/cardcaptor/pipeline.py`, find where valley splits are detected and add per-region check:

```python
# In Stage 1 sampler / valley detection section:
from src.cardcaptor.presence.valley_detection import find_valley_splits_per_region

# Existing global valley detection:
global_splits = find_valley_splits(proxy_frames)

# NEW: Per-region valley detection
regional_splits = find_valley_splits_per_region(proxy_frames, grid_size=3)

# Combine: emit split if either global or regional detects it
all_splits = sorted(set(global_splits) | set(regional_splits))

# Use all_splits for session boundaries
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave2_robustness.py::test_per_region_valley_detects_left_exit_right_entry -xvs`
Expected: PASS

- [ ] **Step 6: Integration test on multi-card captures**

Run: `pytest tests/ -k "multi_card or swap" --tb=short`
Expected: Rapid-swap sequences correctly detected; no false negatives on "A leaves, B enters" patterns.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/presence/valley_detection.py src/cardcaptor/pipeline.py tests/test_wave2_robustness.py
git commit -m "feat(sampler): per-region valley detection for multi-card swaps

- Tile proxy frame into 3×3 grid, compute Sobel valley per tile
- Detect swaps on opposite sides (A leaves left, B enters right)
- Complements global valley detection; combines both signals
- Reduces false negatives in rapid-swap scenarios

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Per-tile occlusion residual detection (agent-agnostic)

**Files:**
- Create: `src/cardcaptor/quality/occlusion_residual.py` — per-tile median and residual computation
- Modify: `src/cardcaptor/scorer.py:QualityScorer` — integrate occlusion residual
- Modify: `tests/test_wave2_robustness.py`

**Context:** Proposal #14. For each tracked card, maintain per-tile (5×7 grid on 750×1050 rectified) running median across recent frames. Compute |current_tile − median_tile|; high-residual contiguous blob = occlusion. Agent-agnostic: fingers, stickers, lens flare, foil shifts all behave the same.

- [ ] **Step 1: Write failing test for per-tile occlusion**

```python
# tests/test_wave2_robustness.py
import numpy as np
from src.cardcaptor.quality.occlusion_residual import compute_occlusion_residual_score

def test_occlusion_residual_detects_fingertip():
    """Verify per-tile occlusion detects interior fingertip."""
    # Create clean card canonical (750×1050)
    clean_card = np.random.randint(100, 200, (750, 1050, 3), dtype=np.uint8)
    
    # Add fingertip (dark blob) in the middle
    clean_card[300:400, 400:500] = 50  # Dark finger
    
    # Running median from prior frames (clean, no finger)
    running_medians = [np.random.randint(100, 200, (750, 1050, 3), dtype=np.uint8) for _ in range(5)]
    
    # Compute residual
    occlusion_score = compute_occlusion_residual_score(clean_card, running_medians, tile_grid=(5, 7))
    
    # Should detect high residual from finger
    assert occlusion_score > 0.5, f"Fingertip should produce high occlusion score, got {occlusion_score}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave2_robustness.py::test_occlusion_residual_detects_fingertip -xvs`
Expected: FAIL — "function compute_occlusion_residual_score not found"

- [ ] **Step 3: Implement per-tile occlusion detection**

Create `src/cardcaptor/quality/occlusion_residual.py`:

```python
import numpy as np
import cv2

def compute_occlusion_residual_score(
    frame: np.ndarray,
    running_median_frames: list[np.ndarray],
    tile_grid: tuple[int, int] = (5, 7),
    residual_threshold: float = 0.3
) -> float:
    """
    Compute occlusion penalty based on per-tile appearance residuals.
    
    Args:
        frame: Current rectified card frame (BGR, e.g., 750×1050)
        running_median_frames: List of prior frames to compute median
        tile_grid: (rows, cols) grid of tiles
        residual_threshold: Threshold for high-residual tile (normalized 0-1)
    
    Returns:
        Occlusion score ∈ [0, 1], where 1 = no occlusion, 0 = severe
    """
    if len(running_median_frames) == 0:
        return 1.0  # No prior data, assume no occlusion
    
    # Compute per-tile median from prior frames
    tile_rows, tile_cols = tile_grid
    h, w = frame.shape[:2]
    tile_h = h // tile_rows
    tile_w = w // tile_cols
    
    # Stack frames for median
    stacked = np.stack(running_median_frames, axis=0)  # (N, H, W, 3)
    tile_medians = np.median(stacked, axis=0).astype(np.uint8)  # (H, W, 3)
    
    # Compute residuals per tile
    high_residual_tiles = []
    
    for row in range(tile_rows):
        for col in range(tile_cols):
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            
            frame_tile = frame[y1:y2, x1:x2].astype(np.float32)
            median_tile = tile_medians[y1:y2, x1:x2].astype(np.float32)
            
            # Per-pixel residual
            residual = np.abs(frame_tile - median_tile) / 255.0
            residual_score = residual.mean()
            
            if residual_score > residual_threshold:
                high_residual_tiles.append((row, col, residual_score))
    
    # Check if high-residual tiles form a contiguous blob (connected component)
    if len(high_residual_tiles) == 0:
        return 1.0  # No occlusion
    
    # Create mask of high-residual tiles
    residual_mask = np.zeros((tile_rows, tile_cols), dtype=np.uint8)
    for row, col, _ in high_residual_tiles:
        residual_mask[row, col] = 1
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(residual_mask, connectivity=8)
    
    if num_labels <= 1:
        return 1.0  # Only background
    
    # Find largest blob
    blob_sizes = np.bincount(labels.flatten())
    largest_blob_size = np.max(blob_sizes[1:])  # Exclude background
    total_tiles = tile_rows * tile_cols
    
    # Blob fraction of grid
    blob_fraction = largest_blob_size / total_tiles
    
    # Penalty: if blob covers >30% of tiles, it's whole-card occlusion (not localized)
    if blob_fraction > 0.3:
        # Likely a Front→Back transition, not a localized occlusion
        return 1.0
    
    # Penalize localized blobs
    # blob_fraction=0.05 (small) → score~0.9
    # blob_fraction=0.15 (moderate) → score~0.5
    # blob_fraction=0.30 (large localized) → score~0.1
    occlusion_score = np.clip(1.0 - blob_fraction * 3, 0, 1)
    
    return float(occlusion_score)
```

- [ ] **Step 4: Integrate into QualityScorer**

Modify `src/cardcaptor/scorer.py`:

```python
from src.cardcaptor.quality.occlusion_residual import compute_occlusion_residual_score

class QualityScorer:
    def score(self, canonical_entry: dict, prior_frames: list[np.ndarray] = None) -> float:
        """
        Compute quality score.
        
        Components:
        1. Blur (Laplacian variance)
        2. Exposure (histogram spread)
        3. Contrast (std-dev)
        4. Border purity
        5. Glare (spatial)
        6. Sharpness (edge density)
        7. Aspect ratio
        8. Occlusion residual (NEW)
        """
        frame = canonical_entry["normalized"]  # rectified crop
        
        blur_score = self._blur_score(frame)
        exposure_score = self._exposure_score(frame)
        contrast_score = self._contrast_score(frame)
        border_score = self._border_purity_score(frame)
        glare_score = self._spatial_glare_score(frame)
        sharpness_score = self._sharpness_score(frame)
        aspect_score = self._aspect_ratio_score(canonical_entry)
        
        # Occlusion residual (NEW)
        if prior_frames and len(prior_frames) > 0:
            occlusion_score = compute_occlusion_residual_score(frame, prior_frames, tile_grid=(5, 7))
        else:
            occlusion_score = 1.0
        
        components = [
            blur_score,
            exposure_score,
            contrast_score,
            border_score,
            glare_score,
            sharpness_score,
            aspect_score,
            occlusion_score
        ]
        
        # Updated weights to include occlusion
        weights = [0.25, 0.12, 0.15, 0.10, 0.03, 0.20, 0.05, 0.10]  # sum to 1.0
        
        quality = np.dot(components, weights)
        return float(quality)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave2_robustness.py::test_occlusion_residual_detects_fingertip -xvs`
Expected: PASS

- [ ] **Step 6: Integration test on occluded cards**

Run: `pytest tests/ -k "occlusion or interior" --tb=short`
Expected: Interior occlusions (fingers, stickers, flare) detected; quality scores drop appropriately.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/quality/occlusion_residual.py src/cardcaptor/scorer.py tests/test_wave2_robustness.py
git commit -m "feat(quality): per-tile occlusion residual detection (agent-agnostic)

- Maintain per-tile running median across recent frames
- Detect localized appearance changes (fingers, stickers, lens flare, foil shifts)
- Connected-component analysis: blob <30% of tiles = localized occlusion
- Integrated as quality component 8 (weight 0.10)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Integration & Testing

### Full Integration Test

- [ ] **Run all Wave 2 tests together**

```bash
pytest tests/test_wave2_robustness.py -xvs
```

Expected: All 13 tasks' tests pass.

- [ ] **Run full regression suite**

```bash
pytest tests/regression/ --tb=short
```

Expected: No regressions. Robustness metrics show improvement in multi-card scenes, occlusion handling, and fusion quality.

- [ ] **Performance check**

```bash
python3 -m src.cardcaptor.pipeline --telemetry run_telemetry_wave2.json <test_video.mp4>
```

Expected: Per-frame cost increased by ~15-20 ms (ECC ~10 ms, occlusion residual ~5 ms). Memory +50-100 MB (tile state, median frames).

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-pipeline-v4x-wave2-foundational.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks sequentially in this session using executing-plans, batch validation at checkpoints

**Which approach would you prefer?**
