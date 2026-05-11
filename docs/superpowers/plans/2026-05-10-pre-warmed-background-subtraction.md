# Pre-Warmed Background Subtraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate "card stand" phantom detections by pre-warming the `NullStateDetector` with globally-minimal presence frames found during the Fast Scan pass and wiring it into the pipeline's producer loop.

**Architecture:** 
1. `AdaptivePresenceSampler` identifies 5 frames with globally lowest `presence_score` during 15fps scan.
2. `NullStateDetector` is initialized with these frames via a new `warmup_batch` method.
3. The pipeline's `_producer_main` uses the detector to skip "workspace empty" frames before ML inference.

**Tech Stack:** Python 3.9, OpenCV, NumPy, `card_capture` pipeline.

---

### Task 1: Enhance `NullStateDetector` with Batch Warmup

**Files:**
- Modify: `src/card_capture/pipeline.py`
- Test: `tests/test_detectors.py`

- [ ] **Step 1: Write failing unit test for batch warmup**

```python
def test_null_state_detector_warmup_batch():
    from card_capture.pipeline import NullStateDetector
    import numpy as np
    detector = NullStateDetector(frames=5, threshold=10.0)
    # 5 identical frames
    frames = [np.full((100, 100), 128, dtype=np.uint8) for _ in range(5)]
    detector.warmup_batch(frames)
    
    assert detector.background_model is not None
    assert detector.frame_count == 5
    # Testing a frame that is identical to background
    test_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert detector.is_workspace_empty(test_frame) is True
    # Testing a frame that is very different
    test_frame_diff = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert detector.is_workspace_empty(test_frame_diff) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detectors.py::test_null_state_detector_warmup_batch -v`
Expected: FAIL with "AttributeError: 'NullStateDetector' object has no attribute 'warmup_batch'"

- [ ] **Step 3: Implement `warmup_batch` and fix `is_workspace_empty` to handle color input**

```python
class NullStateDetector:
    # ... existing __init__ ...

    def warmup_batch(self, frames: list[np.ndarray]) -> None:
        """Initialize background model from a batch of frames."""
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if self.background_model is None:
                self.background_model = gray.astype(np.float32)
                self.frame_count = 1
            else:
                self.background_model = (
                    (self.background_model * self.frame_count + gray) / (self.frame_count + 1)
                )
                self.frame_count += 1

    def is_workspace_empty(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.background_model is None:
            self.background_model = np.zeros_like(gray, dtype=np.float32)

        if self.frame_count < self.frames:
            self.background_model = (
                (self.background_model * self.frame_count + gray) / (self.frame_count + 1)
            )
            self.frame_count += 1
            return False 
        
        diff = cv2.absdiff(gray, self.background_model.astype(np.uint8))
        return float(np.mean(diff)) < self.threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_detectors.py::test_null_state_detector_warmup_batch -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline.py tests/test_detectors.py
git commit -m "feat: add warmup_batch to NullStateDetector"
```

---

### Task 2: Global Background Discovery in `AdaptivePresenceSampler`

**Files:**
- Modify: `src/card_capture/sampler/__init__.py`
- Test: `tests/test_sampler.py`

- [ ] **Step 1: Write failing test for background proxy collection**

```python
def test_sampler_collects_background_proxies(tmp_path):
    from card_capture.sampler import AdaptivePresenceSampler
    import numpy as np
    # Mock scan video to return some frames with very low presence scores
    sampler = AdaptivePresenceSampler()
    # Need to verify that background_proxies is populated after a scan
    assert hasattr(sampler, 'background_proxies')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sampler.py -k background_proxies -v`
Expected: FAIL

- [ ] **Step 3: Update `AdaptivePresenceSampler` to track minimal presence frames**

```python
class AdaptivePresenceSampler:
    def __init__(self, ...):
        # ... existing ...
        self.background_proxies: list[np.ndarray] = []
        self._max_bg_proxies = 5
        self._bg_safety_threshold = 0.4 # If min score > this, we have no empty stand

    def _scan_video(self, video_path: Path) -> list[_AdaptiveScanFrame]:
        # ... existing setup ...
        records = []
        bg_candidates: list[tuple[float, np.ndarray]] = [] # list of (score, image)

        # In the loop where records are appended:
        # After flush_batch() or inside it where records are created:
        for record in new_records:
            # We use presence_score if classifier was used, or score_records fallback
            score = record.presence_score
            if len(bg_candidates) < self._max_bg_proxies:
                heapq.heappush(bg_candidates, (-score, record.image.copy()))
            elif score < -bg_candidates[0][0]:
                heapq.heapreplace(bg_candidates, (-score, record.image.copy()))
        
        # After loop:
        min_score = min([-c[0] for c in bg_candidates]) if bg_candidates else 1.0
        if min_score < self._bg_safety_threshold:
            self.background_proxies = [c[1] for c in bg_candidates]
        else:
            self.background_proxies = []
            
        return records
```
*Note: Use `import heapq` in `sampler/__init__.py`.*

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sampler.py -k background_proxies -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/sampler/__init__.py
git commit -m "feat: collect global background proxies in AdaptivePresenceSampler"
```

---

### Task 3: Wire Pre-Warmed Detector into Pipeline

**Files:**
- Modify: `src/card_capture/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing integration test for background filtering**

```python
def test_pipeline_filters_background_stand(tmp_path, capsys):
    # Mock sampler to provide background proxies that match a "static" frame
    # Verify that accepted_frame_count excludes the static frames
    pass # implementation depends on test_pipeline fixtures
```

- [ ] **Step 2: Update `_run_pipeline_workers` to pass background proxies**

```python
def _run_pipeline_workers(
    video_path: Path,
    video_id: int,
    frame_dir: Path,
    sampler,
    detector,
    options: ProcessingOptions,
) -> tuple[_ProducerStats, list[_DetectionEnvelope]]:
    # ...
    background_proxies = getattr(sampler, "background_proxies", [])
    # ...
    producer = ctx.Process(
        target=_producer_main,
        args=(
            str(video_path),
            video_id,
            sampler,
            options.blur_threshold,
            options.variance_threshold,
            options.empty_pixel_threshold,
            options.background_frames,
            options.background_threshold,
            options.triage_keep_percentile,
            frame_queue,
            stats_queue,
            error_queue,
            background_proxies, # NEW ARG
        ),
        name="producer",
    )
    # ...
```

- [ ] **Step 3: Update `_producer_main` to pre-warm and filter**

```python
def _producer_main(
    # ...
    error_queue,
    background_proxies: list[np.ndarray], # NEW ARG
) -> None:
    # ...
    null_detector = NullStateDetector(frames=background_frames, threshold=background_threshold)
    if background_proxies:
        null_detector.warmup_batch(background_proxies)
        
    try:
        for frame in sampler.sample(Path(video_path), 0.0):
            # ... triage check ...
            
            # NEW: Background check
            if null_detector.is_workspace_empty(frame.image):
                continue
                
            # ... rest of producer loop ...
```

- [ ] **Step 4: Run pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline.py
git commit -m "feat: wire pre-warmed NullStateDetector into pipeline producer"
```

---

### Task 4: Verification & Final Polish

- [ ] **Step 1: Run full test suite**

Run: `source .venv/bin/activate && pytest`
Expected: 209+ passed.

- [ ] **Step 2: Manual Check (Instructional)**
Instruct user to run:
`.venv/bin/card-capture process ~/IMG_5872.MOV --tracker-backend botsort`
Verify that Instances 1 & 2 (the card stand) are gone.
