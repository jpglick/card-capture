# Card Capture Pipeline: Technical Overview & Current State

## 1. Current Pipeline Architecture
The pipeline operates as a discrete-event system with five primary stages, designed to transform raw video frames into rectified, deduplicated card instances.

1.  **Ingestion & Triaging (`sampler.py`):**
    - The sampler reads the video file and generates `FrameSamples`.
    - `RollingWindowTriage` filters out frames with high motion blur, ensuring only "sharp" frames are passed to inference.

2.  **Detection & Null-State Filtering (`detectors.py`, `pipeline.py`):**
    - **Null-State Detector:** The pipeline first checks if the workspace is "empty" using an `absdiff` against a moving background model. If the workspace is empty (Null State), the detector inference is short-circuited to save compute, and a "Global Reset" of all active tracks is triggered.
    - **Detector Inference:** When active, the `CardcaptorUltralyticsDetector` processes frames to identify 4-corner polygons.
    - **Geometric Gate:** Detections are immediately filtered by aspect ratio (0.5–1.5) and area (10%–80% of frame) to reject lightbox/background noise.

3.  **Tracking & State Management (`selector.py`):**
    - The `HysteresisTracker` maintains `TrackState`.
    - It uses spatial distance and detector confidence to bridge frames.
    - **Flip Logic:** A "Null-Squeeze" flip detector triggers a track split and a state toggle ("Front" -> "Back") if the detected polygon area drops below 20% of the median track area, indicating an edge-on card orientation.

4.  **Rectification & Normalization (`cropper.py`):**
    - `CardCropper` identifies the top and right edges of the polygon.
    - If `width > height`, it mathematically rolls the corner array to force a portrait (2.5x3.5) aspect ratio before applying `cv2.getPerspectiveTransform` to rectify the card.

5.  **Canonicalization & Storage (`storage.py`, `pipeline.py`):**
    - **Visual Deduplication:** After tracks are finalized, the pipeline generates visual hashes for all views.
    - **Invariant Hashing:** It computes hashes for both the original view and a 180-degree rotation, storing the minimum Hamming distance to the existing database canonical hashes.
    - **Storage:** Only finalized tracks are persisted.

---

## 2. The Operational Flow (Step-by-Step)

1.  **Frame Ingestion:** `producer_main` reads and triage-filters frames.
2.  **Workspace Check:** `VideoProcessor` checks if the workspace is empty.
    - If **Active**: Run detector inference.
    - If **Empty**: Trigger session reset (flush tracks, clear state).
3.  **Inference:** `consumer_main` runs detection on batch frames.
4.  **Candidate Building:** `_build_candidates` converts raw detections into `ScoredCandidate` objects.
5.  **Tracking Loop:** `tracker.process(candidate)` builds `TrackState`.
6.  **Finalization:** `tracker.finalize()` terminates tracks that fail length requirements (min 3 frames).
7.  **Deduplication:** The `VisualDeduplicator` groups tracks that are visually identical within the session and picks the canonical view.
8.  **Telemetry Logging:** Telemetry (aspect, area, etc.) and debug frames are logged directly to the SQLite `track_telemetry` table.
9.  **Database Commit:** `storage.add_card_instance` and `storage.add_card_view` record the result of the session.

## 3. Pipeline Diagnostic Report (Last Run)
- **Video:** IMG_5596.MOV
- **Processing Results:** 2037 frames, 678 detections, 51 instances saved.
- **Track Health Metrics:**
    - **Total Telemetry Points:** 1190
    - **Frame Coverage:** Detected frames range from index 167 to 1956.
    - **Average Track Length:** 11.67 frames.
    - **Fragmentation:** A significant portion of tracks are 3 frames long, confirming micro-fragmentation.
- **Deduplication:** Only 2 instances flagged as duplicates; target of ~14 unique cards not met, suggesting further refinement of deduplication or session scoping is required.
- **Null State Usage:** The `NullStateDetector` is active but has not successfully consolidated fragmented segments into single, cohesive card sessions.
