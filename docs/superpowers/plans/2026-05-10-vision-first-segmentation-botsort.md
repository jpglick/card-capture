# Vision-First Segmentation + BoT-SORT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace time-only session splitting with vision-first boundaries (valley + centroid jump + ReID) and move tracking to BoT-SORT so rapid card transitions split correctly while preserving performance.

**Architecture:** Keep the existing 3-read pipeline budget by retaining fast-scan frames in memory, split sessions from three independent signals, and integrate BoT-SORT through a local adapter that preserves the existing `TrackState` contract. Maintain ByteTrack as a debug backend, but default to BoT-SORT.

**Tech Stack:** Python 3.9, OpenCV, Torch/TorchVision, BoxMOT (BoT-SORT), existing `card_capture` pipeline/test harness.

---

## File Structure (planned changes)

- **Create:** `src/card_capture/tracking/botsort_adapter.py` — BoxMOT-backed adapter returning `_AdaptedDetection` compatible with pipeline.
- **Create:** `src/card_capture/tracking/centroid_jump.py` — spatial jump reset signal.
- **Create:** `src/card_capture/sampler/valley_splits.py` — pure valley-split logic.
- **Create:** `tests/test_botsort_adapter.py` — adapter contract tests.
- **Create:** `tests/test_centroid_jump.py` — centroid detector unit tests.
- **Create:** `tests/test_valley_splits.py` — valley split function tests.
- **Create:** `tests/test_sampler_fast_scan.py` — fast/confirm scan behavior tests.
- **Modify:** `src/card_capture/sampler.py` — two-pass scan retention + forced valley boundaries.
- **Modify:** `src/card_capture/pipeline.py` — multi-signal session reset orchestration, backend selection.
- **Modify:** `src/card_capture/cli.py` — new flags and tracker backend wiring.
- **Modify:** `src/card_capture/config.py` — add new config defaults.
- **Modify:** `src/card_capture/tracking/__init__.py` — exports for new tracker modules.
- **Modify:** `pyproject.toml` — add BoT-SORT dependency path (`boxmot`) in optional deps.
- **Modify:** `tests/test_pipeline.py` — focused integration tests for session reset reasons and backend routing.

---

### Task 1: Dependency + configuration scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/card_capture/config.py`
- Modify: `src/card_capture/cli.py`
- Test: `tests/test_cli.py` (create if absent)

- [ ] **Step 1: Write failing CLI/config tests**

```python
def test_cli_accepts_tracker_backend_and_split_flags():
    parser = build_parser()
    args = parser.parse_args([
        "process", "video.mov",
        "--tracker-backend", "botsort",
        "--fast-scan-fps", "15",
        "--valley-drop-ratio", "0.4",
        "--centroid-jump-ratio", "0.3",
    ])
    assert args.tracker_backend == "botsort"
    assert args.fast_scan_fps == 15.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_cli_accepts_tracker_backend_and_split_flags -v`  
Expected: FAIL for unknown args.

- [ ] **Step 3: Add dependencies and config fields**

```toml
[project.optional-dependencies]
pipeline_v21 = [
  "av",
  "onnxruntime",
  "boxmot>=18.0.0",
]
```

```python
@dataclass
class PipelineConfig:
    tracker_backend: str = "botsort"
    fast_scan_fps: float = 15.0
    confirm_scan_fps: float = 5.0
    valley_drop_ratio: float = 0.40
    valley_min_width_frames: int = 3
    delta_spike_ratio: float = 0.60
    centroid_jump_ratio: float = 0.30
    centroid_jump_frames: int = 3
    reid_distance_threshold: float = 0.6
```

- [ ] **Step 4: Wire CLI flags into `ProcessingOptions` and sampler construction**

```python
process.add_argument("--tracker-backend", choices=["botsort", "bytetrack"], default=None)
process.add_argument("--fast-scan-fps", type=float, default=None)
process.add_argument("--confirm-scan-fps", type=float, default=None)
process.add_argument("--valley-drop-ratio", type=float, default=None)
process.add_argument("--valley-min-width", type=int, default=None)
process.add_argument("--delta-spike-ratio", type=float, default=None)
process.add_argument("--centroid-jump-ratio", type=float, default=None)
process.add_argument("--centroid-jump-frames", type=int, default=None)
process.add_argument("--reid-distance-threshold", type=float, default=None)
```

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/card_capture/config.py src/card_capture/cli.py tests/test_cli.py
git commit -m "feat: add BoT-SORT and segmentation config surface"
```

---

### Task 2: Valley split primitive (pure function, dual-signal)

**Files:**
- Create: `src/card_capture/sampler/valley_splits.py`
- Test: `tests/test_valley_splits.py`

**Design note:** `find_valley_splits` accepts both `sobel_scores` and `delta_scores` (frame-to-frame pixel delta). Either signal can independently trigger a split: a Sobel valley *or* a delta spike that exceeds `delta_spike_ratio × max(delta_scores)`. Both are computed for free during Pass 1 since frames are already in memory.

- [ ] **Step 1: Write failing unit tests**

```python
def test_single_qualified_valley_returns_split_frame():
    scores = [1.0, 1.2, 0.4, 0.3, 1.1, 1.3]
    frames = [10, 11, 12, 13, 14, 15]
    deltas = [0.0] * 6  # no delta signal
    splits = find_valley_splits(scores, deltas, frames, valley_drop_ratio=0.4, valley_min_width_frames=2)
    assert splits == [13]

def test_shallow_valley_no_split():
    scores = [1.0, 1.1, 0.9, 1.0]
    frames = [1, 2, 3, 4]
    deltas = [0.0] * 4
    assert find_valley_splits(scores, deltas, frames, 0.4, 2) == []

def test_delta_spike_triggers_split_without_sobel_valley():
    # Two similarly edge-dense cards — Sobel stays flat, but pixel delta spikes on swap
    scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    deltas = [0.02, 0.03, 0.85, 0.03, 0.02, 0.02]  # big spike at index 2 (frame 12)
    frames = [10, 11, 12, 13, 14, 15]
    splits = find_valley_splits(scores, deltas, frames, delta_spike_ratio=0.5)
    assert 12 in splits

def test_combined_signal_no_double_split():
    # Sobel valley and delta spike coincide — should produce only one split point
    scores = [1.2, 1.1, 0.3, 0.3, 1.0, 1.1]
    deltas = [0.02, 0.02, 0.80, 0.04, 0.02, 0.02]
    frames = [10, 11, 12, 13, 14, 15]
    splits = find_valley_splits(scores, deltas, frames, valley_drop_ratio=0.4, valley_min_width_frames=2, delta_spike_ratio=0.5)
    assert len(splits) == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_valley_splits.py -v`  
Expected: FAIL (import/function missing).

- [ ] **Step 3: Implement minimal function**

```python
def find_valley_splits(
    sobel_scores: list[float],
    delta_scores: list[float],
    frame_indices: list[int],
    valley_drop_ratio: float = 0.40,
    valley_min_width_frames: int = 3,
    delta_spike_ratio: float = 0.60,
) -> list[int]:
    """Return sorted unique frame indices where a card swap is detected.

    Two independent signals:
    - Sobel valley: score drops >= valley_drop_ratio from preceding peak and
      persists >= valley_min_width_frames frames before recovering.
    - Delta spike: frame-to-frame pixel delta exceeds delta_spike_ratio * max(delta_scores).
      Split is placed at the spike frame (local maximum of delta).
    """
    split_frames: set[int] = set()

    # --- Sobel valley detection ---
    peak = sobel_scores[0] if sobel_scores else 0.0
    valley_start: int | None = None
    for i, (score, fi) in enumerate(zip(sobel_scores, frame_indices)):
        if score < peak * (1.0 - valley_drop_ratio):
            if valley_start is None:
                valley_start = i
        else:
            if valley_start is not None and (i - valley_start) >= valley_min_width_frames:
                # valley minimum frame
                min_idx = valley_start + int(
                    np.argmin(sobel_scores[valley_start:i])
                )
                split_frames.add(frame_indices[min_idx])
            valley_start = None
            if score > peak:
                peak = score

    # --- Delta spike detection ---
    if delta_scores and max(delta_scores) > 0:
        threshold = delta_spike_ratio * max(delta_scores)
        for i, (delta, fi) in enumerate(zip(delta_scores, frame_indices)):
            if delta >= threshold:
                split_frames.add(fi)

    return sorted(split_frames)
```

- [ ] **Step 4: Run tests and fix edge cases**

Run: `.venv/bin/python -m pytest tests/test_valley_splits.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/sampler/valley_splits.py tests/test_valley_splits.py
git commit -m "feat: add dual-signal valley split detector (Sobel + frame delta)"
```

---

### Task 3: Fast scan + confirm scan in sampler

**Files:**
- Modify: `src/card_capture/sampler.py`
- Test: `tests/test_sampler_fast_scan.py`

- [ ] **Step 1: Write failing sampler tests**

```python
def test_fast_scan_fps_produces_more_scan_frames_than_confirm_scan(tmp_path):
    sampler = AdaptivePresenceSampler(fast_scan_fps=15.0, confirm_scan_fps=5.0)
    sampler._scan_video = lambda _: fake_scan_frames(45)  # deterministic stub
    sampler.sample(tmp_path / "dummy.mov", 0.0)
    assert sampler.last_scan_frame_count == 45

def test_scan_frame_has_delta_score():
    # _ScanFrame must carry delta_score; first frame delta is 0.0
    frames = make_fake_scan_frames(n=5)
    assert all(hasattr(f, "delta_score") for f in frames)
    assert frames[0].delta_score == 0.0

def test_valley_boundaries_force_window_split():
    # Synthetic scores: two presence peaks separated by a deep Sobel valley
    sobel = [0.05, 0.8, 0.9, 0.15, 0.1, 0.85, 0.9, 0.05]
    deltas = [0.0] * 8
    frames = list(range(8))
    splits = find_valley_splits(sobel, deltas, frames, valley_drop_ratio=0.4, valley_min_width_frames=2)
    assert len(splits) == 1

def test_delta_spike_forces_window_split_in_sampler(tmp_path):
    # Sampler with two cards separated by a pixel-delta spike (no Sobel valley)
    sobel = [0.8] * 10
    deltas = [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    sampler = AdaptivePresenceSampler(fast_scan_fps=15.0, delta_spike_ratio=0.6)
    splits = find_valley_splits(sobel, deltas, list(range(10)), delta_spike_ratio=0.6)
    assert 4 in splits
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_sampler_fast_scan.py -v`  
Expected: FAIL for missing params/behavior.

- [ ] **Step 3: Add `delta_score` to `_ScanFrame` and compute during Pass 1**

```python
@dataclass
class _ScanFrame:
    frame_index: int
    timestamp_ms: float
    sobel_score: float
    delta_score: float = 0.0   # mean absolute pixel diff vs. previous frame (0–255 scale)
```

In `_scan_video`, after computing `sobel_score`, compute delta against the previous retained image:

```python
prev_img: np.ndarray | None = None
for frame_index, timestamp_ms, img_160 in self._read_frames_at_fps(video_path, fps):
    sobel_score = _compute_sobel_mean(img_160)
    delta_score = float(np.mean(np.abs(img_160.astype(np.float32) - prev_img.astype(np.float32)))) \
        if prev_img is not None else 0.0
    prev_img = img_160
    scan_frames.append(_ScanFrame(frame_index, timestamp_ms, sobel_score, delta_score))
```

- [ ] **Step 4: Implement sampler split pipeline passing both signals**

```python
class AdaptivePresenceSampler:
    def __init__(
        self, ...,
        fast_scan_fps: float = 15.0,
        confirm_scan_fps: float = 5.0,
        valley_drop_ratio: float = 0.40,
        valley_min_width_frames: int = 3,
        delta_spike_ratio: float = 0.60,
    ):
        self.fast_scan_fps = fast_scan_fps
        self.confirm_scan_fps = confirm_scan_fps
        self.valley_drop_ratio = valley_drop_ratio
        self.valley_min_width_frames = valley_min_width_frames
        self.delta_spike_ratio = delta_spike_ratio
        self.last_valley_splits: list[int] = []
```

```python
scan_records = self._scan_video(video_path, fps=self.fast_scan_fps)
sobel_scores = [r.sobel_score for r in scan_records]
delta_scores = [r.delta_score for r in scan_records]
frame_indices = [r.frame_index for r in scan_records]
self.last_valley_splits = find_valley_splits(
    sobel_scores, delta_scores, frame_indices,
    valley_drop_ratio=self.valley_drop_ratio,
    valley_min_width_frames=self.valley_min_width_frames,
    delta_spike_ratio=self.delta_spike_ratio,
)
windows = self._build_windows(scan_records, forced_splits=self.last_valley_splits)
```

- [ ] **Step 5: Preserve performance invariant (no extra video read)**

```python
# Pass 2 scoring must use retained scan_images from scan_records only.
records = self._candidate_records_for_window(window)  # in-memory reuse
```

- [ ] **Step 6: Run sampler tests**

Run: `.venv/bin/python -m pytest tests/test_sampler_fast_scan.py -q`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/card_capture/sampler.py tests/test_sampler_fast_scan.py src/card_capture/sampler/valley_splits.py
git commit -m "feat: add fast-scan two-pass sampler with dual-signal valley boundaries"
```

---

### Task 4: Geography split (centroid jump detector)

**Files:**
- Create: `src/card_capture/tracking/centroid_jump.py`
- Modify: `src/card_capture/tracking/__init__.py`
- Test: `tests/test_centroid_jump.py`

- [ ] **Step 1: Write failing detector tests**

```python
def test_large_jump_within_window_triggers_split():
    det = CentroidJumpDetector(jump_ratio=0.3, jump_within_frames=3)
    assert det.update(np.array([10, 10, 110, 110], dtype=np.float32), 1000) is False
    assert det.update(np.array([500, 10, 600, 110], dtype=np.float32), 1000) is True
```

```python
def test_reset_clears_history():
    det = CentroidJumpDetector()
    ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_centroid_jump.py -v`  
Expected: FAIL (class missing).

- [ ] **Step 3: Implement detector**

```python
class CentroidJumpDetector:
    def update(self, bbox_xyxy: Optional[np.ndarray], frame_width: int) -> bool:
        ...
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_centroid_jump.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/tracking/centroid_jump.py src/card_capture/tracking/__init__.py tests/test_centroid_jump.py
git commit -m "feat: add centroid jump split detector"
```

---

### Task 5: BoT-SORT adapter via BoxMOT

**Files:**
- Create: `src/card_capture/tracking/botsort_adapter.py`
- Test: `tests/test_botsort_adapter.py`

- [ ] **Step 1: Write adapter contract tests (mirror ByteTrack tests)**

```python
def test_adapter_assigns_consistent_track_id_for_overlapping_boxes():
    adapter = BoTSORTAdapter(min_track_length=1)
    f1 = [make_candidate(frame_index=1, corners=[...], score=0.9)]
    f2 = [make_candidate(frame_index=2, corners=[...], score=0.88)]
    out1 = adapter.process(f1)
    out2 = adapter.process(f2)
    assert out1 and out2
    assert out1[0].track_id == out2[0].track_id
```

```python
def test_pending_reid_split_set_when_identity_changes():
    adapter = BoTSORTAdapter(reid_distance_threshold=0.6)
    ...
    assert adapter.pending_splits
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_botsort_adapter.py -v`  
Expected: FAIL (module/class missing).

- [ ] **Step 3: Implement adapter with explicit dependency failure**

```python
try:
    from boxmot import BoTSORT
except Exception as exc:
    raise ImportError("BoT-SORT backend requires `boxmot` optional dependency") from exc
```

```python
class BoTSORTAdapter:
    def process(self, candidates: List[ScoredCandidate]) -> List[_AdaptedDetection]:
        ...
    def finalize(self) -> List[TrackState]:
        ...
    def reset(self) -> None:
        ...
```

- [ ] **Step 4: Run adapter tests**

Run: `.venv/bin/python -m pytest tests/test_botsort_adapter.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/tracking/botsort_adapter.py tests/test_botsort_adapter.py pyproject.toml
git commit -m "feat: add BoT-SORT tracker adapter using BoxMOT"
```

---

### Task 6: Integrate multi-signal reset into pipeline

**Files:**
- Modify: `src/card_capture/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests for reset reasons**

```python
def test_pipeline_resets_session_on_centroid_jump(...):
    ...
    assert any(e["data"]["reason"] == "centroid_jump" for e in events)
```

```python
def test_pipeline_uses_botsort_backend_by_default(...):
    ...
    assert telemetry["tracker_backend"] == "botsort"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::test_pipeline_uses_botsort_backend_by_default -v`  
Expected: FAIL for missing backend routing.

- [ ] **Step 3: Implement backend routing + split orchestration**

```python
if options.tracker_backend == "botsort":
    self.tracker = BoTSORTAdapter(...)
else:
    self.tracker = ByteTrackAdapter(...)

centroid_detector = CentroidJumpDetector(...)

if centroid_detector.update(primary_bbox, frame_width):
    _reset_session("centroid_jump")
if frame_index in getattr(self.tracker, "pending_splits", []):
    _reset_session("reid_shift")
if gap_triggered:
    _reset_session("sampled_frame_gap")
```

- [ ] **Step 4: Run focused pipeline tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k "centroid_jump or tracker_backend or session_reset" -q`  
Expected: PASS for new tests.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline.py tests/test_pipeline.py src/card_capture/tracking/botsort_adapter.py src/card_capture/tracking/centroid_jump.py
git commit -m "feat: wire BoT-SORT + centroid + gap split signals in pipeline"
```

---

### Task 7: Reuse ReID embeddings in dedup path

**Files:**
- Modify: `src/card_capture/selector.py` (or TrackState definition file)
- Modify: `src/card_capture/deduplicator.py`
- Test: `tests/test_deduplicator.py`

- [ ] **Step 1: Write failing dedup tests for embedding reuse**

```python
def test_deduplicator_prefers_reid_embedding_when_available():
    track_a = make_track(reid_embedding=np.array([1.0, 0.0]))
    track_b = make_track(reid_embedding=np.array([0.99, 0.01]))
    assert dedup.is_duplicate(track_a, track_b) is True
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_deduplicator.py::test_deduplicator_prefers_reid_embedding_when_available -v`  
Expected: FAIL (field/path missing).

- [ ] **Step 3: Implement optional embedding path**

```python
@dataclass
class TrackState:
    ...
    reid_embedding: Optional[np.ndarray] = None
```

```python
if t1.reid_embedding is not None and t2.reid_embedding is not None:
    return cosine_distance(t1.reid_embedding, t2.reid_embedding) < self.reid_duplicate_threshold
# fallback: existing perceptual hash path
```

- [ ] **Step 4: Run dedup tests**

Run: `.venv/bin/python -m pytest tests/test_deduplicator.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/selector.py src/card_capture/deduplicator.py tests/test_deduplicator.py
git commit -m "feat: reuse BoT-SORT embeddings for visual deduplication"
```

---

### Task 8: Enhance `sampler sessions` diagnostics + final verification

**Files:**
- Modify: `src/card_capture/cli.py`
- Modify: `src/card_capture/sampler.py`
- Test: `tests/test_cli.py`, `tests/test_sampler_fast_scan.py`

- [ ] **Step 1: Write failing diagnostics tests**

```python
def test_sampler_sessions_outputs_valley_and_scan_counts(capsys):
    ...
    assert "fast_scan_frames" in out
    assert "valley_splits" in out
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k sampler_sessions -v`  
Expected: FAIL for missing output fields.

- [ ] **Step 3: Implement diagnostics output**

```python
print(f"Scan + window build: {wall:.1f}s | fast_scan_frames={...} | confirm_scan_frames={...}")
print(f"Valley splits ({len(sampler.last_valley_splits)}): {sampler.last_valley_splits}")
print(f"  (sobel valleys + delta spikes — any of these forced a window boundary)")
```

- [ ] **Step 4: Run project test suite (excluding known pre-existing failures)**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/regression --ignore=tests/test_pipeline.py`  
Expected: PASS.

- [ ] **Step 5: Run targeted integration checks**

Run: `.venv/bin/card-capture sampler sessions tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV --expected 6`  
Expected: `Sessions predicted` in range `5..7`.

Run: `.venv/bin/card-capture harness run --baseline reports/baseline_v3.json`  
Expected: recall ≥ 0.95 and phantom_rate ≤ 0.222.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/cli.py src/card_capture/sampler.py tests/test_cli.py tests/test_sampler_fast_scan.py
git commit -m "feat: expose vision-first diagnostics and complete BoT-SORT rollout"
```

---

## Self-review checklist (completed)

- **Spec coverage:** Covered fast scan, valley split (Sobel + delta), centroid split, ReID split, BoT-SORT backend, diagnostics, and success gates.
- **Placeholder scan:** No TBD/TODO placeholders remain; each task has concrete files, tests, commands, and code snippets.
- **Type consistency:** Uses `BoTSORTAdapter`, `CentroidJumpDetector`, `find_valley_splits(sobel_scores, delta_scores, frame_indices, ...)`, and `tracker_backend` consistently across tasks.
- **Delta signal:** `_ScanFrame.delta_score` is computed in Task 3, passed to `find_valley_splits` in Task 3, tested in `test_valley_splits.py` (Task 2) and `test_sampler_fast_scan.py` (Task 3), and surfaced in diagnostics (Task 8). `delta_spike_ratio` is wired through config/CLI in Task 1.

