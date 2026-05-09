# Design Spec: Session-Anchored, Presence-First Pipeline

## 1. Overview
This design refactors the Card Capture pipeline into a robust, session-aware system. It transitions from stream-based processing to a session-based lifecycle, anchored by explicit workspace calibration.

## 2. Presence-First Initialization
The pipeline no longer assumes the first frames are empty.
- **Stage 1 (Calibration):** The first 30 frames are scanned for card presence.
    - If a card is detected, the pipeline halts background model training and waits for a "Null State" (workspace empty).
    - If no card is detected, the background model is trained on these frames.
- **Session Reset:** Every "Null State" (empty workspace) triggers a session boundary, grouping all card presentations and flips into a coherent `session_id`.

## 3. Performance & Diagnostics (Runtime Telemetry)
The pipeline will now emit real-time performance metrics to `stdout` to assist in monitoring and debugging.
- **Stage-Specific Metrics:** As each stage (Ingestion, Inference, Refinement) completes, the pipeline prints the elapsed time and current session status.
- **Diagnostic Logging:** 
    - **Flip Triggers:** Logs `max_area` and `drop_ratio` for all flip events.
    - **Session Transitions:** Logs `workspace_reset` events with specific trigger reasons (`init_scan`, `null_state_timeout`, etc.).
    - **Deduplication Decisions:** Logs Hamming distances for all candidate merges, allowing verification of the 0.995 similarity threshold.

## 4. Pipeline Refinement
- **Lazy Batch Warping:** Homography, normalization, and hashing are moved out of the track processing loop and are only executed once per finalized session for the top-scoring candidate frames.
- **Derivative Flip Tracking:** Flip detection is now based on a continuous area drop rate within a 5-frame window, providing robustness against detection dropouts.
- **Center-Crop Hashing:** Deduplication will exclusively process the center 60% of card crops, neutralizing edge-glare artifacts that plague full-rectified hashes.

## 5. Implementation Roadmap
- **Task 1: Session & Init:** Implement `PresenceFirstSampler` and `SessionRegistry`.
- **Task 2: Lazy Warping:** Refactor `VideoProcessor` to collect raw candidates and warp in a single batch post-session.
- **Task 3: Diagnostic Logging:** Augment existing storage methods to persist pipeline events for live feedback.

---
*Note: Upon implementation, the runtime output will look like: `[Stage: Detection] | 14ms | Session: 1 | Card: Front | Confidence: 0.85`.*
