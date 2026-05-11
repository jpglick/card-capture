# Pipeline v4.x Robustness (Wave 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable automated threshold tuning and embedding-based identity matching by implementing embedding distance metrics, hard-case capture for active learning, and threshold auto-sweep with per-video adaptation.

**Architecture:** Wave 3 contains 5 S-M effort proposals with two dependency chains. Task 1 (embedding-based same-card) is foundational—it replaces pHash with OSNet embeddings, enabled by Wave 1's real-frame ReID. Task 2 uses Task 1 for cross-video dedup. Task 3 (hard-case capture) runs in parallel, collecting ambiguous sessions. Tasks 4-5 form a second chain: auto-sweep (Task 4) measures robustness metrics from Wave 2, then Task 5 uses those metrics to compute per-video adaptive thresholds. Together they close the loop: Wave 1-2 improve signal quality, Wave 3 tunes the pipeline to find the sweet spot.

**Tech Stack:** scipy (cosine distance), sklearn (percentile computation), numpy (online estimators), pytest (grid search validation).

---

## File Map

**Identity & Deduplication:**
- Create: `src/cardcaptor/identity/embedding_distance.py` — cosine-based same-card verification
- Modify: `src/cardcaptor/pipeline.py:_resolve_session_tracks` — use embeddings instead of pHash
- Modify: `src/cardcaptor/pipeline.py:_is_reid_duplicate` — embedding-based cross-video dedup
- Modify: `tests/test_wave3_calibration.py` — embedding distance tests

**Active Learning:**
- Create: `src/cardcaptor/analysis/hard_case_capture.py` — identify and persist ambiguous sessions
- Modify: `src/cardcaptor/pipeline.py` — wire hard-case detection
- Create: `hard_cases.jsonl` — JSONL file appended each run with hard cases

**Calibration & Adaptation:**
- Create: `scripts/calibrate_wave3.py` — grid sweep over thresholds
- Create: `src/cardcaptor/calibration/per_video_adaptive.py` — online threshold adaptation
- Modify: `src/cardcaptor/pipeline.py` — wire per-video thresholds
- Modify: `tests/test_wave3_calibration.py` — calibration and adaptation tests

---

## Task Breakdown

### Task 1: Embedding-based same-card criterion (replaces pHash)

**Files:**
- Create: `src/cardcaptor/identity/embedding_distance.py`
- Modify: `src/cardcaptor/pipeline.py:_resolve_session_tracks` — replace pHash gate
- Modify: `tests/test_wave3_calibration.py`

**Context:** Proposal #15. OSNet embeddings are now available (Wave 1 real frames). Replace pHash Hamming distance with cosine distance of embeddings. Calibrate threshold against regression set's labeled same-card pairs.

- [ ] **Step 1: Write failing test for embedding distance**

```python
# tests/test_wave3_calibration.py
import numpy as np
from src.cardcaptor.identity.embedding_distance import embedding_same_card_score, compute_embedding_distance

def test_embedding_distance_same_physical_card():
    """Verify embeddings of same card (different views) are similar."""
    # Two embeddings of the same physical card (Front and Back)
    emb_front = np.array([0.1, 0.2, 0.15, 0.25, 0.3] + [0.0]*507, dtype=np.float32)  # 512-dim
    emb_back = np.array([0.12, 0.18, 0.16, 0.24, 0.29] + [0.0]*507, dtype=np.float32)  # similar
    
    distance = compute_embedding_distance(emb_front, emb_back)
    
    # Same card should have low cosine distance (<0.5)
    assert distance < 0.5, f"Same card should have low distance, got {distance}"

def test_embedding_distance_different_cards():
    """Verify embeddings of different cards are dissimilar."""
    emb_card_a = np.random.randn(512).astype(np.float32)
    emb_card_a /= np.linalg.norm(emb_card_a)  # normalize
    
    emb_card_b = np.random.randn(512).astype(np.float32)
    emb_card_b /= np.linalg.norm(emb_card_b)  # normalize
    
    distance = compute_embedding_distance(emb_card_a, emb_card_b)
    
    # Different cards should have high distance (>0.5)
    assert distance > 0.3, f"Different cards should have higher distance, got {distance}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave3_calibration.py::test_embedding_distance_same_physical_card -xvs`
Expected: FAIL — "function compute_embedding_distance not found"

- [ ] **Step 3: Implement embedding distance functions**

Create `src/cardcaptor/identity/embedding_distance.py`:

```python
import numpy as np
from scipy.spatial.distance import cosine

def compute_embedding_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine distance between two embeddings.
    
    Args:
        emb_a: OSNet embedding (512-dim float32)
        emb_b: OSNet embedding (512-dim float32)
    
    Returns:
        Cosine distance ∈ [0, 2], where 0 = identical, 1 = orthogonal, 2 = opposite
    """
    # Normalize embeddings
    emb_a_norm = emb_a / (np.linalg.norm(emb_a) + 1e-8)
    emb_b_norm = emb_b / (np.linalg.norm(emb_b) + 1e-8)
    
    # Cosine distance = 1 - cosine_similarity
    distance = cosine(emb_a_norm, emb_b_norm)
    
    return float(distance)

def embedding_same_card_score(distance: float, threshold: float = 0.5) -> bool:
    """
    Determine if two embeddings represent the same physical card.
    
    Args:
        distance: Cosine distance from compute_embedding_distance
        threshold: Distance threshold for same-card decision (default 0.5)
    
    Returns:
        True if same card (distance < threshold), False otherwise
    """
    return distance < threshold
```

- [ ] **Step 4: Update _resolve_session_tracks to use embeddings**

In `src/cardcaptor/pipeline.py`, find `_resolve_session_tracks` and update:

```python
from src.cardcaptor.identity.embedding_distance import compute_embedding_distance, embedding_same_card_score

def _resolve_session_tracks(session_tracks):
    """
    Resolve session tracks using side_score (primary), embeddings (secondary).
    """
    if not session_tracks:
        return None, None
    
    # Sort by side_score (textiness) — primary signal
    session_tracks_sorted = sorted(
        session_tracks,
        key=lambda item: item[1].side_score,
        reverse=True
    )
    
    front_label, front_track = session_tracks_sorted[0]
    
    if len(session_tracks_sorted) == 1:
        return (front_label, front_track), None
    
    # Check for same-physical-card Back using embeddings
    # Get best representative embedding from front track
    front_embedding = front_track.best_canonical.get("embedding")  # OSNet embedding, 512-dim
    
    for back_label, back_track in session_tracks_sorted[1:]:
        back_embedding = back_track.best_canonical.get("embedding")
        
        if front_embedding is None or back_embedding is None:
            # Fallback to pHash if embeddings unavailable
            return _resolve_with_phash_fallback(front_label, front_track, back_label, back_track)
        
        # Compute embedding distance
        distance = compute_embedding_distance(front_embedding, back_embedding)
        
        # Same card if embedding distance is low
        if embedding_same_card_score(distance, threshold=0.5):
            return (front_label, front_track), (back_label, back_track)
    
    return (front_label, front_track), None

def _resolve_with_phash_fallback(front_label, front_track, back_label, back_track):
    """Fallback to pHash-based same-card detection if embeddings unavailable."""
    front_phash = front_track.best_canonical.get("phash")
    back_phash = back_track.best_canonical.get("phash")
    
    if front_phash is None or back_phash is None:
        return (front_label, front_track), None
    
    hamming_dist = _hamming_distance(front_phash, back_phash)
    if hamming_dist <= 22:  # existing threshold
        return (front_label, front_track), (back_label, back_track)
    
    return (front_label, front_track), None
```

- [ ] **Step 5: Ensure embeddings are stored in canonical entries**

Verify that `best_canonical` dict includes "embedding" key. This should flow from Wave 1's real-frame ReID (BoT-SORT already computes embeddings). Check `_build_candidates()` or `_prepare_track()` to ensure embeddings are persisted.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_wave3_calibration.py::test_embedding_distance_same_physical_card -xvs`
Expected: PASS

- [ ] **Step 7: Run regression tests on identity correctness**

Run: `pytest tests/regression/ -k "front_back or identity" --tb=short`
Expected: Front/Back F1 metric improves (embeddings are better at same-card detection than pHash).

- [ ] **Step 8: Commit**

```bash
git add src/cardcaptor/identity/embedding_distance.py src/cardcaptor/pipeline.py tests/test_wave3_calibration.py
git commit -m "feat(identity): embedding-based same-card criterion (replaces pHash)

- Use OSNet cosine embeddings for same-physical-card detection
- Replace pHash Hamming distance (loose 22/64 threshold) with embeddings
- Fallback to pHash if embeddings unavailable (backward compat)
- Threshold: cosine_distance < 0.5 for same-card decision
- Foundation for cross-video dedup and per-video adaptation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Cross-video dedup via embedding (same-card across videos)

**Files:**
- Modify: `src/cardcaptor/pipeline.py:_is_reid_duplicate` — use embeddings for dedup
- Modify: `tests/test_wave3_calibration.py`

**Context:** Proposal #16. After Task 1, use embeddings for cross-video dedup. pHash as cheap pre-filter (broad candidates), embeddings as deciding metric.

- [ ] **Step 1: Write failing test for cross-video dedup**

```python
# tests/test_wave3_calibration.py
def test_cross_video_dedup_via_embeddings():
    """Verify same card detected across different videos via embeddings."""
    from src.cardcaptor.identity.embedding_distance import embedding_same_card_score, compute_embedding_distance
    
    # Card A from video 1 and video 2 (should be deduped)
    canonical_video1 = {
        "phash": "abc123",  # same content hash (pHash pre-filter)
        "embedding": np.array([0.1, 0.2, 0.15, 0.25, 0.3] + [0.0]*507, dtype=np.float32)
    }
    
    canonical_video2 = {
        "phash": "abc123",  # same pHash
        "embedding": np.array([0.11, 0.19, 0.16, 0.24, 0.31] + [0.0]*507, dtype=np.float32)  # similar embedding
    }
    
    distance = compute_embedding_distance(canonical_video1["embedding"], canonical_video2["embedding"])
    is_same = embedding_same_card_score(distance, threshold=0.5)
    
    assert is_same, f"Same card across videos should be detected, got distance={distance}"

def test_cross_video_dedup_avoids_false_positives():
    """Verify different cards not deduped even with same pHash."""
    # Card A and Card B from different videos (same pHash by chance, but different content)
    canonical_a = {
        "phash": "abc123",
        "embedding": np.random.randn(512).astype(np.float32)
    }
    canonical_a["embedding"] /= np.linalg.norm(canonical_a["embedding"])
    
    canonical_b = {
        "phash": "abc123",  # unfortunate pHash collision
        "embedding": np.random.randn(512).astype(np.float32)  # different visual content
    }
    canonical_b["embedding"] /= np.linalg.norm(canonical_b["embedding"])
    
    distance = compute_embedding_distance(canonical_a["embedding"], canonical_b["embedding"])
    is_same = embedding_same_card_score(distance, threshold=0.5)
    
    assert not is_same, f"Different cards should not be deduped, got distance={distance}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave3_calibration.py::test_cross_video_dedup_via_embeddings -xvs`
Expected: FAIL (either test structure is wrong or dedup logic not yet implemented)

- [ ] **Step 3: Update _is_reid_duplicate to use embeddings**

In `src/cardcaptor/pipeline.py`, find `_is_reid_duplicate` (Stage 10) and update:

```python
from src.cardcaptor.identity.embedding_distance import compute_embedding_distance, embedding_same_card_score

def _is_reid_duplicate(canonical_a, canonical_b, use_embeddings=True):
    """
    Determine if two canonicals represent the same physical card (cross-video dedup).
    
    Args:
        canonical_a: Canonical from video A
        canonical_b: Canonical from video B
        use_embeddings: If True, use embeddings; else fall back to pHash
    
    Returns:
        True if same card (should be deduped), False otherwise
    """
    # Step 1: pHash as cheap pre-filter (broad candidates)
    phash_a = canonical_a.get("phash")
    phash_b = canonical_b.get("phash")
    
    if phash_a is None or phash_b is None:
        return False
    
    hamming_dist = _hamming_distance(phash_a, phash_b)
    
    # If pHashes are very different, definitely not the same card
    if hamming_dist > 22:  # existing threshold
        return False
    
    # Step 2: pHash is close or identical; use embeddings to decide
    if use_embeddings:
        emb_a = canonical_a.get("embedding")
        emb_b = canonical_b.get("embedding")
        
        if emb_a is None or emb_b is None:
            # No embeddings; use pHash result
            return hamming_dist <= 15  # stricter threshold without embeddings
        
        # Compute embedding distance
        distance = compute_embedding_distance(emb_a, emb_b)
        
        # Same card if embedding distance is low
        return embedding_same_card_score(distance, threshold=0.5)
    
    else:
        # Fallback: pHash only
        return hamming_dist <= 15
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wave3_calibration.py::test_cross_video_dedup_via_embeddings -xvs`
Expected: PASS

- [ ] **Step 5: Integration test on cross-video corpus**

Run: `pytest tests/regression/ -k "cross_video or dedup" --tb=short`
Expected: Cross-video dedup metrics improve (fewer false negatives from pHash collisions).

- [ ] **Step 6: Commit**

```bash
git add src/cardcaptor/pipeline.py tests/test_wave3_calibration.py
git commit -m "feat(identity): cross-video dedup via embedding distance

- pHash pre-filter (broad candidates, fast)
- Embeddings as deciding metric (discriminative, view-invariant)
- Replaces pHash-only dedup; avoids false positives from collisions
- Backward compat: fallback to stricter pHash threshold if embeddings unavailable

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Hard-case capture for active learning

**Files:**
- Create: `src/cardcaptor/analysis/hard_case_capture.py`
- Modify: `src/cardcaptor/pipeline.py` — wire capture logic
- Create: `hard_cases.jsonl` (appended each run)
- Modify: `tests/test_wave3_calibration.py`

**Context:** Proposal #17. On every pipeline run, append to `hard_cases.jsonl` any session that:
- Resolves to >2 distinct Fronts (missed swap)
- Has Front-vs-other Hamming margin within 4 of threshold (borderline)
- Keeps detection with confidence ∈ [0.45, 0.55] (just-above-threshold)
- Has best-canonical border_purity < 0.3 (likely occlusion)

- [ ] **Step 1: Write failing test for hard-case detection**

```python
# tests/test_wave3_calibration.py
import json
import os
from src.cardcaptor.analysis.hard_case_capture import is_hard_case, capture_hard_case

def test_hard_case_multiple_fronts():
    """Verify hard-case detection catches >2 Fronts."""
    session = {
        "video_id": "test_video_001",
        "session_id": "session_001",
        "front_tracks": [
            {"track_id": "t1", "side": "Front"},
            {"track_id": "t2", "side": "Front"},
            {"track_id": "t3", "side": "Front"}
        ],
        "back_tracks": []
    }
    
    hard_case_reason = is_hard_case(session)
    
    assert hard_case_reason is not None, f"Should detect >2 Fronts as hard case, got: {hard_case_reason}"
    assert "multiple_fronts" in hard_case_reason

def test_hard_case_borderline_hamming():
    """Verify hard-case detection catches borderline pHash distances."""
    session = {
        "video_id": "test_video_002",
        "session_id": "session_002",
        "front_tracks": [
            {
                "track_id": "t1",
                "side": "Front",
                "phash": "a" * 64,
                "hamming_to_best": 20  # close to threshold 22, within margin of 4
            }
        ]
    }
    
    hard_case_reason = is_hard_case(session)
    
    assert hard_case_reason is not None, "Should detect borderline Hamming as hard case"
    assert "borderline" in hard_case_reason or "hamming" in hard_case_reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave3_calibration.py::test_hard_case_multiple_fronts -xvs`
Expected: FAIL — "function is_hard_case not found"

- [ ] **Step 3: Implement hard-case detection**

Create `src/cardcaptor/analysis/hard_case_capture.py`:

```python
import json
from typing import Optional, Dict, List
from datetime import datetime

def is_hard_case(session: Dict) -> Optional[str]:
    """
    Determine if a session is a hard case (ambiguous, edge case).
    
    Returns:
        Reason string if hard case, None otherwise
    """
    # Check 1: >2 distinct Fronts (missed swap)
    front_count = len(session.get("front_tracks", []))
    if front_count > 2:
        return f"multiple_fronts_{front_count}"
    
    # Check 2: Borderline Hamming distance (within 4 of threshold 22)
    for track in session.get("front_tracks", []):
        hamming = track.get("hamming_to_best", 0)
        if 18 <= hamming <= 26:  # threshold ±4
            return f"borderline_hamming_{hamming}"
    
    # Check 3: Detection confidence just-above-threshold ∈ [0.45, 0.55]
    for det in session.get("detections", []):
        conf = det.get("confidence", 0.0)
        if 0.45 <= conf <= 0.55:
            return f"borderline_confidence_{conf:.2f}"
    
    # Check 4: Low border purity (<0.3)
    best_canonical = session.get("best_canonical", {})
    border_purity = best_canonical.get("border_purity", 1.0)
    if border_purity < 0.3:
        return f"low_border_purity_{border_purity:.2f}"
    
    return None

def capture_hard_case(session: Dict, reason: str, output_file: str = "hard_cases.jsonl"):
    """
    Persist a hard case to JSONL file for active learning.
    
    Args:
        session: Session data
        reason: Hard-case reason (from is_hard_case)
        output_file: Path to JSONL output file
    """
    hard_case_record = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "video_id": session.get("video_id"),
        "session_id": session.get("session_id"),
        "front_count": len(session.get("front_tracks", [])),
        "back_count": len(session.get("back_tracks", [])),
        "detection_count": len(session.get("detections", []))
    }
    
    # Append to JSONL file (create if not exists)
    with open(output_file, "a") as f:
        f.write(json.dumps(hard_case_record) + "\n")

def load_hard_cases(input_file: str = "hard_cases.jsonl") -> List[Dict]:
    """Load all hard cases from JSONL file."""
    hard_cases = []
    
    if not os.path.exists(input_file):
        return hard_cases
    
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                hard_cases.append(json.loads(line))
    
    return hard_cases
```

- [ ] **Step 4: Wire into pipeline (end of session resolution)**

In `src/cardcaptor/pipeline.py`, find where sessions are finalized (after `_resolve_session_tracks`) and add:

```python
from src.cardcaptor.analysis.hard_case_capture import is_hard_case, capture_hard_case

def _finalize_session(session_tracks):
    """Process a complete session."""
    front, back = _resolve_session_tracks(session_tracks)
    
    # ... existing finalization logic ...
    
    # NEW: Capture hard cases
    session_dict = {
        "video_id": self.video_id,
        "session_id": session_id,
        "front_tracks": [t[1] for t in session_tracks],
        "back_tracks": []  # or populate from back if available
    }
    
    hard_case_reason = is_hard_case(session_dict)
    if hard_case_reason:
        capture_hard_case(session_dict, hard_case_reason, "hard_cases.jsonl")
    
    return front, back
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave3_calibration.py::test_hard_case_multiple_fronts -xvs`
Expected: PASS

- [ ] **Step 6: Integration test**

Run: `pytest tests/ -k "hard_case" --tb=short`
Expected: Hard cases captured correctly; JSONL file grows monotonically.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/analysis/hard_case_capture.py src/cardcaptor/pipeline.py tests/test_wave3_calibration.py
git commit -m "feat(learning): hard-case capture for active learning

- Identify ambiguous sessions: >2 Fronts, borderline Hamming, borderline confidence, low border_purity
- Persist to hard_cases.jsonl (JSONL format, one record per line)
- Growths monotonically; provides ground for future learned models
- No impact on throughput (post-hoc classification)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Threshold auto-sweep harness

**Files:**
- Create: `scripts/calibrate_wave3.py` — grid sweep harness
- Modify: `tests/regression/metrics.py` — metrics reporting
- Modify: `tests/test_wave3_calibration.py`

**Context:** Proposal #18. Grid sweep over thresholds using robustness metrics from Wave 2. Find Pareto front of precision/recall.

- [ ] **Step 1: Write failing test for grid sweep**

```python
# tests/test_wave3_calibration.py
from scripts.calibrate_wave3 import grid_sweep_thresholds

def test_grid_sweep_finds_pareto_optimal_points():
    """Verify threshold sweep finds Pareto-optimal operating points."""
    # Run sweep on small synthetic corpus
    results = grid_sweep_thresholds(
        novelty_thresholds=[0.05, 0.08, 0.10],
        hamming_thresholds=[18, 20, 22],
        metric_corpus="tests/regression/truth.py",
        output_file="/tmp/sweep_results.json"
    )
    
    assert len(results) > 0, "Sweep should return results"
    assert "pareto_front" in results, "Should identify Pareto front"
    
    # Each point on Pareto front should be non-dominated
    pareto = results["pareto_front"]
    for point in pareto:
        assert point["metric"] is not None, "Points should have metrics"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave3_calibration.py::test_grid_sweep_finds_pareto_optimal_points -xvs`
Expected: FAIL — "function grid_sweep_thresholds not found"

- [ ] **Step 3: Implement calibration harness**

Create `scripts/calibrate_wave3.py`:

```python
import json
import itertools
from src.cardcaptor.metrics.robustness_pack import RobustnessMetrics
from tests.regression.pipeline_runner import run_regression_pack

def grid_sweep_thresholds(novelty_thresholds, hamming_thresholds, metric_corpus, output_file):
    """
    Grid sweep over thresholds and report Pareto-optimal points.
    
    Args:
        novelty_thresholds: List of novelty gate thresholds to sweep
        hamming_thresholds: List of same-card Hamming thresholds to sweep
        metric_corpus: Path to regression truth corpus
        output_file: JSON output file for results
    
    Returns:
        Dict with 'all_points' (all sweep results) and 'pareto_front' (Pareto-optimal)
    """
    all_points = []
    
    # Grid sweep
    for novelty_thresh in novelty_thresholds:
        for hamming_thresh in hamming_thresholds:
            print(f"Sweeping: novelty={novelty_thresh}, hamming={hamming_thresh}")
            
            # Run pipeline with these thresholds
            predictions = run_regression_pack(
                novelty_threshold=novelty_thresh,
                hamming_threshold=hamming_thresh
            )
            
            # Compute robustness metrics
            truth_labels = load_truth_labels(metric_corpus)
            metrics_obj = RobustnessMetrics(truth_labels, predictions)
            metrics = metrics_obj.compute_all()
            
            # Compute F1 as objective
            recall = metrics["card_recall"]
            precision = metrics["card_precision"]
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            point = {
                "novelty_threshold": novelty_thresh,
                "hamming_threshold": hamming_thresh,
                "card_recall": recall,
                "card_precision": precision,
                "f1": f1,
                "metrics": metrics
            }
            all_points.append(point)
    
    # Compute Pareto front
    pareto_front = compute_pareto_front(all_points, ["card_recall", "card_precision"])
    
    results = {
        "all_points": all_points,
        "pareto_front": pareto_front,
        "grid_size": f"{len(novelty_thresholds)} × {len(hamming_thresholds)}"
    }
    
    # Write results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPareto front ({len(pareto_front)} points):")
    for point in pareto_front:
        print(f"  Recall={point['card_recall']:.3f}, Precision={point['card_precision']:.3f}, "
              f"Novelty={point['novelty_threshold']:.2f}, Hamming={point['hamming_threshold']}")
    
    return results

def compute_pareto_front(points, objectives):
    """
    Compute Pareto front (non-dominated points).
    
    Args:
        points: List of dicts with objective values
        objectives: List of objective names to optimize (higher is better)
    
    Returns:
        List of points on Pareto front
    """
    front = []
    
    for point in points:
        dominated = False
        
        for other in points:
            if point == other:
                continue
            
            # Check if 'other' dominates 'point'
            other_better = all(other[obj] >= point[obj] for obj in objectives)
            other_strictly_better = any(other[obj] > point[obj] for obj in objectives)
            
            if other_better and other_strictly_better:
                dominated = True
                break
        
        if not dominated:
            front.append(point)
    
    return front
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wave3_calibration.py::test_grid_sweep_finds_pareto_optimal_points -xvs`
Expected: PASS

- [ ] **Step 5: Integration test on regression corpus**

Run the actual sweep on the full regression corpus (offline, can take 5-10 minutes):
```bash
python scripts/calibrate_wave3.py \
  --novelty-thresholds 0.05 0.06 0.07 0.08 0.09 0.10 \
  --hamming-thresholds 18 19 20 21 22 23 24 \
  --output calibration_results.json
```

Expected: Pareto front identified with 5-10 points showing precision/recall tradeoff.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_wave3.py tests/test_wave3_calibration.py
git commit -m "feat(calibration): threshold auto-sweep harness for robustness tuning

- Grid sweep over novelty gate + Hamming distance thresholds
- Computes card recall, precision, F1 for each point via regression metrics
- Identifies Pareto-optimal operating points (non-dominated)
- Human picks the operating point (recall/precision tradeoff)
- Foundation for per-video adaptive thresholds

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Per-video adaptive thresholds

**Files:**
- Create: `src/cardcaptor/calibration/per_video_adaptive.py` — online threshold adaptation
- Modify: `src/cardcaptor/pipeline.py:Stage 4 & 8` — wire per-video thresholds
- Modify: `tests/test_wave3_calibration.py`

**Context:** Proposal #19. After Task 4 establishes global thresholds, adapt them per-video using online estimators (novelty distribution percentiles, intra-track Hamming percentiles).

- [ ] **Step 1: Write failing test for adaptive thresholds**

```python
# tests/test_wave3_calibration.py
from src.cardcaptor.calibration.per_video_adaptive import AdaptiveThresholdComputer

def test_adaptive_novelty_threshold_from_distribution():
    """Verify per-video novelty threshold computed from in-video distribution."""
    # Simulate in-video novelty scores
    novelty_scores = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    
    computer = AdaptiveThresholdComputer()
    adaptive_novelty = computer.compute_novelty_threshold(novelty_scores, global_threshold=0.08)
    
    # Should be near median or percentile-based estimate
    assert 0.08 <= adaptive_novelty <= 0.15, f"Adaptive threshold should be in reasonable range, got {adaptive_novelty}"

def test_adaptive_hamming_threshold_from_intra_track_distances():
    """Verify per-video Hamming threshold from intra-track distance percentiles."""
    # Simulate intra-track Hamming distances (same card, different frames)
    intra_track_distances = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]  # same card variations
    
    computer = AdaptiveThresholdComputer()
    adaptive_hamming = computer.compute_hamming_threshold(intra_track_distances, global_threshold=22)
    
    # Should be higher percentile of intra-track (to avoid false resets)
    assert adaptive_hamming >= 10, f"Adaptive Hamming should be ≥10, got {adaptive_hamming}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wave3_calibration.py::test_adaptive_novelty_threshold_from_distribution -xvs`
Expected: FAIL — "class AdaptiveThresholdComputer not found"

- [ ] **Step 3: Implement per-video adaptive threshold computation**

Create `src/cardcaptor/calibration/per_video_adaptive.py`:

```python
import numpy as np
from typing import List, Optional

class AdaptiveThresholdComputer:
    """Compute per-video adaptive thresholds from online distributions."""
    
    def compute_novelty_threshold(self, novelty_scores: List[float], global_threshold: float) -> float:
        """
        Compute per-video novelty threshold from in-video candidate distribution.
        
        Strategy: Use p50 (median) of observed novelty scores, clipped to global ± 20%.
        
        Args:
            novelty_scores: List of novelty scores from all candidates in video
            global_threshold: Global baseline threshold (from calibration)
        
        Returns:
            Adapted novelty threshold for this video
        """
        if len(novelty_scores) < 3:
            return global_threshold
        
        median = np.median(novelty_scores)
        
        # Clip to ±20% of global
        lower_bound = global_threshold * 0.8
        upper_bound = global_threshold * 1.2
        
        adapted = np.clip(median, lower_bound, upper_bound)
        return float(adapted)
    
    def compute_hamming_threshold(self, intra_track_distances: List[float], global_threshold: float) -> float:
        """
        Compute per-video Hamming threshold from intra-track pHash variation.
        
        Strategy: Use p75 (75th percentile) of intra-track distances.
        Same card can vary up to this amount; anything beyond is likely different card.
        
        Args:
            intra_track_distances: List of Hamming distances between frames of same track
            global_threshold: Global baseline threshold (from calibration)
        
        Returns:
            Adapted Hamming threshold for this video
        """
        if len(intra_track_distances) < 3:
            return global_threshold
        
        p75 = np.percentile(intra_track_distances, 75)
        
        # Add safety margin and clip to global ± 10%
        adapted = p75 * 1.2  # 20% safety margin
        lower_bound = global_threshold * 0.9
        upper_bound = global_threshold * 1.1
        
        adapted = np.clip(adapted, lower_bound, upper_bound)
        return float(adapted)
```

- [ ] **Step 4: Wire into pipeline (Stage 4 & 8)**

In `src/cardcaptor/pipeline.py`, update novelty gate and session resolution:

```python
from src.cardcaptor.calibration.per_video_adaptive import AdaptiveThresholdComputer

class PipelineContext:
    def __init__(self, video_id, global_novelty_threshold=0.08, global_hamming_threshold=22):
        self.video_id = video_id
        self.global_novelty_threshold = global_novelty_threshold
        self.global_hamming_threshold = global_hamming_threshold
        
        # Per-video adaptive thresholds (computed on-the-fly)
        self.adaptive_computer = AdaptiveThresholdComputer()
        self.observed_novelty_scores = []
        self.observed_intra_track_distances = []
    
    def get_adaptive_novelty_threshold(self) -> float:
        """Get novelty threshold adapted to this video."""
        if len(self.observed_novelty_scores) < 10:
            return self.global_novelty_threshold
        return self.adaptive_computer.compute_novelty_threshold(
            self.observed_novelty_scores,
            self.global_novelty_threshold
        )
    
    def get_adaptive_hamming_threshold(self) -> float:
        """Get Hamming threshold adapted to this video."""
        if len(self.observed_intra_track_distances) < 10:
            return self.global_hamming_threshold
        return self.adaptive_computer.compute_hamming_threshold(
            self.observed_intra_track_distances,
            self.global_hamming_threshold
        )

# In Stage 4 novelty gate:
novelty_threshold = context.get_adaptive_novelty_threshold()
if novelty_score < novelty_threshold:
    # Pass gate

# In Stage 8 session resolution:
hamming_threshold = context.get_adaptive_hamming_threshold()
if hamming_dist <= hamming_threshold:
    # Same physical card
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wave3_calibration.py::test_adaptive_novelty_threshold_from_distribution -xvs`
Expected: PASS

- [ ] **Step 6: Integration test on video-specific adaptation**

Run: `pytest tests/ -k "adaptive" --tb=short`
Expected: Per-video thresholds computed correctly; gates use adaptive values.

- [ ] **Step 7: Commit**

```bash
git add src/cardcaptor/calibration/per_video_adaptive.py src/cardcaptor/pipeline.py tests/test_wave3_calibration.py
git commit -m "feat(calibration): per-video adaptive thresholds from online estimators

- Novelty threshold: p50 of in-video candidate distribution, clipped ±20%
- Hamming threshold: p75 of intra-track pHash variation, clipped ±10%
- Computed on-the-fly during processing (no pre-computation)
- Graceful fallback to global thresholds if insufficient samples (<10)
- Improves robustness to scene-specific variation without retraining

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Integration & Testing

### Full Integration Test

- [ ] **Run all Wave 3 tests together**

```bash
pytest tests/test_wave3_calibration.py -xvs
```

Expected: All task tests pass (20+ tests covering embeddings, dedup, hard-case, calibration, adaptation).

- [ ] **Run full regression suite**

```bash
pytest tests/regression/ --tb=short
```

Expected: No regressions. Metrics should show improvement from Task 1 (embeddings) and Task 2 (cross-video dedup). Hard cases captured. Calibration harness runs successfully.

- [ ] **Performance check**

```bash
python3 -m src.cardcaptor.pipeline --telemetry run_telemetry_wave3.json <test_video.mp4>
```

Expected: No per-frame latency addition (all adaptive logic is online, post-hoc, or offline). Memory unchanged.

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-pipeline-v4x-wave3-calibration.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks sequentially in this session using executing-plans, batch validation at checkpoints

**Which approach would you prefer?**
