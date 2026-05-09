# Session-Anchored Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Presence-First initialization, Session-Aware lifecycle management, and Lazy Batch Warping for performance.

**Architecture:**
- **Presence-First:** Pipeline scans first frames to calibrate background/workspace model.
- **Session-Aware Lifecycle:** Pipeline tracks active vs. null state; `SessionManager` handles track flushing and database grouping.
- **Lazy Batch Warping:** Normalization/Homography/Hashing delayed until session-finalization and performed in batch via Kornia on the GPU.

**Tech Stack:** Python, PyTorch, Kornia, SQLite3, OpenCV.

---

### Task 1: Session Registry & Initialization (`src/card_capture/pipeline.py`)
- Create `SessionManager` class to manage `active_session_id`.
- Refactor `VideoProcessor.process` to integrate the "Presence-First" scan (first 30 frames check).

### Task 2: Lazy Batch Warping (`src/card_capture/gpu_refinement.py` & `src/card_capture/pipeline.py`)
- Refactor `KorniaNormalizer` to handle batch warping.
- Modify `VideoProcessor` to collect raw frame paths/corners during tracking, warping only the top canonical frames per track after session end.

### Task 3: Diagnostic Telemetry (`src/card_capture/pipeline.py`)
- Augment the processing loop with stage-specific timing `print` statements.
- Integrate event-based logging for session resets, flip triggers, and hashing distances into `track_telemetry`.

---

### Task 1 Details: Session Registry

- [ ] **Step 1: Define `SessionManager`**
```python
class SessionManager:
    def __init__(self):
        self.active_session_id: Optional[str] = None
        
    def start_session(self, timestamp: int):
        self.active_session_id = str(timestamp)
```

- [ ] **Step 2: Update `VideoProcessor`**
Integrate `SessionManager` and `PresenceFirstSampler` logic (or equivalent detection scanning).

- [ ] **Step 3: Commit**
```bash
git add src/card_capture/pipeline.py
git commit -m "feat: add session manager"
```

### Task 2 Details: Lazy Batch Warping

- [ ] **Step 1: Refine `warp_canonical_batch`**
Update `KorniaNormalizer` to efficiently warp a list of paths and corners.

- [ ] **Step 2: Integrate in pipeline**
Modify `VideoProcessor.process` to replace per-frame normalization with per-session batching.

- [ ] **Step 3: Commit**
```bash
git add src/card_capture/gpu_refinement.py src/card_capture/pipeline.py
git commit -m "feat: implement lazy batch warping"
```

### Task 3 Details: Diagnostic Feedback

- [ ] **Step 1: Add diagnostic logging to pipeline loop**
Add `print(f"[Stage: Detection] | {elapsed_time}ms | Session: {id}")` at key stages.

- [ ] **Step 2: Update storage methods**
Add specific telemetry logging for `flip_event` and `dedup_match` events.

- [ ] **Step 3: Commit**
```bash
git add src/card_capture/pipeline.py src/card_capture/storage.py
git commit -m "feat: add runtime diagnostic telemetry"
```
