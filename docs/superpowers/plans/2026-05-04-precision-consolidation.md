# Session Consolidation & Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement strict session-based consolidation (merging all tracks in a session into one card), add aspect-ratio gating, enforce 180-degree rotation for output, and provide runtime profiling diagnostics.

**Architecture:**
- **Session Consolidation:** Bypass perceptual hashing for deduplication. Any tracks occurring within the same `session_id` are grouped. The longest is the "Front", the second longest is the "Back".
- **Precision Gating:** Add an aspect ratio filter (0.60 to 0.85) in `detectors.py` to kill lightbox stand false positives.
- **Rotation:** Add `rotate_180` config flag and apply `cv2.ROTATE_180` in `KorniaNormalizer` and `PrecisionNormalizer`.
- **Diagnostics:** Wrap pipeline stages in `time.time()` blocks and print a summary. Disable HF Symlinks warning.

**Tech Stack:** Python, OpenCV, SQLite.

---

### Task 1: Precision Filters & HF Warning (`src/card_capture/cli.py`, `src/card_capture/detectors.py`)
- Silence HF warning.
- Add aspect ratio filter to `detectors.py`.

### Task 2: 180-Degree Rotation (`src/card_capture/config.py`, `src/card_capture/pipeline.py`, `src/card_capture/gpu_refinement.py`)
- Add `rotate_180` to Config.
- Apply rotation after perspective warp.

### Task 3: Session Consolidation (`src/card_capture/pipeline.py`)
- Remove `VisualDeduplicator` hashing logic.
- Implement logic in `_resolve_session_tracks` to automatically group all tracks in a session into a single `CardInstance`.

### Task 4: Runtime Profiling Diagnostics (`src/card_capture/pipeline.py`)
- Add `time.time()` tracking for the 5 stages and print a summary string at the end of the `process` loop.

---

### Task 1 Details: Precision Filters

- [ ] **Step 1: Silence HF Warning**
Modify `cli.py` to include the environment variable at the top.
```python
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

- [ ] **Step 2: Aspect Ratio Gating**
Modify `detectors.py` in `CardcaptorUltralyticsDetector.detect`:
```python
                if len(polygon) != 4:
                    continue

                poly_arr = np.array(polygon, dtype=np.float32)
                poly_area = cv2.contourArea(poly_arr)
                frame_area = frame.width * frame.height
                if not (0.1 * frame_area <= poly_area <= 0.8 * frame_area):
                    continue
                
                # Check aspect ratio
                width_top = np.linalg.norm(poly_arr[1] - poly_arr[0])
                width_bottom = np.linalg.norm(poly_arr[2] - poly_arr[3])
                height_right = np.linalg.norm(poly_arr[2] - poly_arr[1])
                height_left = np.linalg.norm(poly_arr[3] - poly_arr[0])
                w = max(width_top, width_bottom)
                h = max(height_right, height_left)
                ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                if not (0.60 <= ratio <= 0.85):
                    continue
```

- [ ] **Step 3: Commit**
```bash
git add src/card_capture/cli.py src/card_capture/detectors.py
git commit -m "feat: add aspect ratio gating and silence HF warning"
```

### Task 2 Details: 180-Degree Rotation

- [ ] **Step 1: Config Update**
Modify `src/card_capture/config.py`:
```python
    triage_keep_percentile: float = 0.05
    rotate_180: bool = True
```

- [ ] **Step 2: Pass to Processor**
Modify `cli.py` and `pipeline.py` (`ProcessingOptions`) to pass `rotate_180`.

- [ ] **Step 3: Apply Rotation in Kornia**
Modify `gpu_refinement.py` in `warp_canonical_batch`:
```python
        images: List[np.ndarray] = []
        for w in warped.cpu():
            rgb = kornia.tensor_to_image(w * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = cv2.rotate(bgr, cv2.ROTATE_180) # Force rotation
            images.append(bgr)
```

- [ ] **Step 4: Commit**
```bash
git add src/card_capture/config.py src/card_capture/cli.py src/card_capture/pipeline.py src/card_capture/gpu_refinement.py
git commit -m "feat: enforce 180-degree rotation on output crops"
```

### Task 3 Details: Session Consolidation

- [ ] **Step 1: Rewrite `_resolve_session_tracks`**
Modify `src/card_capture/pipeline.py` to ignore hashing and merge by session.
```python
def _resolve_session_tracks(
    prepared_tracks: list[_PreparedTrack],
    deduplicator: VisualDeduplicator,
) -> None:
    by_session: dict[int, list[_PreparedTrack]] = {}
    for prepared in prepared_tracks:
        by_session.setdefault(prepared.session_id, []).append(prepared)

    for session_id, tracks in by_session.items():
        if not tracks: continue
        # Sort by track length (descending)
        tracks.sort(key=lambda t: len(t.track.candidates), reverse=True)
        
        # Longest track is front
        representative = tracks[0]
        representative.duplicate_track_index = None
        representative.angle = "Front"
        
        # Second longest is back
        if len(tracks) > 1:
            back_track = tracks[1]
            back_track.duplicate_track_index = prepared_tracks.index(representative)
            back_track.angle = "Back"
            
        # Any remaining are fragments, merge them to the front
        for frag in tracks[2:]:
            frag.duplicate_track_index = prepared_tracks.index(representative)
            frag.angle = "Front"
```

- [ ] **Step 2: Commit**
```bash
git add src/card_capture/pipeline.py
git commit -m "feat: implement explicit session consolidation"
```

### Task 4 Details: Runtime Profiling

- [ ] **Step 1: Timing wrappers**
Modify `VideoProcessor.process` to time stages:
```python
        import time
        t_start = time.time()
        # ... ingestion logic ...
        t_ingest = time.time() - t_start
        
        # ... tracker finalization ...
        t_track = time.time() - t_start - t_ingest
        
        # ... normalized batch ...
        t_refine = time.time() - t_start - t_ingest - t_track
        
        # ... database commit ...
        t_storage = time.time() - t_start - t_ingest - t_track - t_refine
        
        print(f"\n--- Performance Summary ---")
        print(f"Ingestion: {t_ingest:.2f}s")
        print(f"Tracking:  {t_track:.2f}s")
        print(f"Refining:  {t_refine:.2f}s")
        print(f"Storage:   {t_storage:.2f}s")
        print(f"Total:     {time.time() - t_start:.2f}s")
        print(f"---------------------------\n")
```

- [ ] **Step 2: Commit**
```bash
git add src/card_capture/pipeline.py
git commit -m "feat: add performance profiling output"
```
