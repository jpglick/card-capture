# Session-Aware Pipeline Implementation Plan

**Goal:** Transform the pipeline from a continuous stream processor to a session-aware system that resets, scopes, and cleans track fragments based on workspace activity.

**Architecture:**
- **Lifecycle Management:** A new `SessionManager` in `VideoProcessor` tracks the transition between "Active" and "Null" states.
- **Hard-Flush:** On transition to Null State, the pipeline will perform a "Hard-Flush" of the tracker, discarding tracks shorter than the minimum length and clearing all active state.
- **Deduplication Scoping:** Hashing comparisons will be restricted to the current session ID to prevent historical false-positive duplicates.

---

### Task 1: Session Lifecycle Management

- [ ] **Modify `src/card_capture/pipeline.py`:** Add `SessionManager` class to track `active_session_id`.
- [ ] **Update `VideoProcessor`:**
    - Initialize `current_session_id`.
    - In the processing loop, detect Null State transition (Empty -> Active).
    - Trigger `flush()` on `tracker` when Active -> Empty transition occurs.
    - Discard tracks in `tracker.finalize()` that are shorter than `min_track_length`.

### Task 2: Geometric & Confidence Gating

- [ ] **Refine `src/card_capture/detectors.py`:**
    - Implement Aspect Ratio filter (0.5 - 1.5).
    - Implement Area Ratio filter (10% - 80%).
- [ ] **Refine `src/card_capture/selector.py`:**
    - Pass detector `confidence` into `ScoredCandidate`.
    - Update `HysteresisTracker` to weight track scores by `confidence`.

### Task 3: Scoped Deduplication

- [ ] **Modify `src/card_capture/storage.py`:** Update `find_canonical_for_hashes` to include `session_id` in the `WHERE` clause.
- [ ] **Update `src/card_capture/pipeline.py`:** Pass `current_session_id` to deduplication logic.

---

### Validation Plan
1. **Workspace Reset:** Verify that 100% of "Null State" frames result in zero ML inference calls.
2. **Track Count:** Confirm the pipeline total track count drops from 53 to ~14 (matching the number of unique physical cards).
3. **Session Integrity:** Verify with `sqlite3` that `card_instances` are now grouped strictly within a `session_id` and that no duplicate cards from previous sessions are incorrectly identified.
