# Card Capture Pipeline: V3 Technical Overview

## 1. The Problem We Are Solving
The goal of this application is to autonomously extract high-quality, normalized images of trading cards and sports cards from unstructured videos. 

A user presents cards to a camera (often hand-held), flips them to show the back, and then removes them to show the next card. The pipeline must ingest this video, intelligently locate the sharpest frames where the card is fully visible and not moving, detect the four corners of the card regardless of orientation, rectify the perspective to a perfect flat 2.5x3.5 ratio, group the "Front" and "Back" views of the same physical card together, and deduplicate identical cards shown across different videos.

## 2. Sample Video Format
- **Resolution & Aspect Ratio:** Typically high-resolution 4K (e.g., 2160x3840), shot vertically (portrait mode).
- **Content:** A continuous video capturing a single "workspace" (like a desk or stand).
- **Action:** A person places cards into the frame, holds them or rests them on a stand, flips them to show the back, and removes them. 
- **Challenges:** The workspace frequently transitions to an "empty stand" state between cards. Cards experience motion blur, glare, partial finger occlusions, and severe perspective skew when flipped.

---

## 3. Current V3 Adaptive Pipeline Architecture

The pipeline has been overhauled from a cadence-based scanner into a highly optimized, temporal **Adaptive Pipeline** designed to minimize ML inference overhead and eliminate micro-fragmentation.

### Stage 1: Adaptive Presence Sampling (`sampler.py`)
Instead of running heavy YOLO inference on every Nth frame, we do a fast 2-pass temporal scan:
- **Pass 1 (Low-Res Scan):** Reads the video at a low resolution and low framerate. It uses lightweight algorithms (Sobel edge detection, empty pixel ratios, variance) to find "Presence Windows"—continuous chunks of time where a card is physically in the frame.
- **Pass 2 (Sharpness Extraction):** Seeks into the identified Presence Windows at full resolution, computes a Laplacian variance (sharpness score), and yields only the Top K sharpest frames per window. 
- *Benefit:* We only run heavy ML inference on ~5-10 frames per card appearance, and they are guaranteed to be the sharpest available.

### Stage 2: ML Inference & Geometric Filtering (`detectors.py`)
- **Downscaled Inference:** The sharp frames are resized to a 640px proxy to run fast inference via `CardcaptorUltralyticsDetector` (YOLO OBB).
- **Upscaled Projection:** The detected 4-corner bounding boxes are scaled back precisely to the native 4K resolution of the source frame.
- **Geometric Filtering:** The scaled polygons are strictly evaluated. They must cover between 10% and 80% of the total frame area and possess an aspect ratio between 0.50 and 0.95. This immediately rejects background noise and false positives.

### Stage 3: Temporal Sessions & Spatial Tracking (`selector.py`, `pipeline.py`)
Because the sampler only yields frames when a card is present, time itself becomes our primary grouping mechanic.
- **Session Boundaries:** If the pipeline detects a temporal gap > 6 frames (~0.5 seconds) with no valid candidates, a `Session Reset` is triggered. A "Session" rigidly encapsulates a single physical card swap.
- **Spatial Tracking (`HysteresisTracker`):** Within a Session, detections are grouped into Tracks based on centroid proximity. 
- **Shape-Change Detection:** If the tracking logic sees a massive drop in area (>30%) or massive spike (>50%), it breaks the track. This successfully isolates edge-on card flips or the empty card stand, ensuring they don't corrupt the canonical track.

### Stage 4: Session Resolution & Deduplication (`pipeline.py`)
Once a Session ends:
- The system gathers all tracks in that Session and sorts them by length.
- **Angle Assignment:** The longest track is assumed to be the `"Front"`. The second longest is the `"Back"`. Any smaller fragmented tracks are folded into the Front as duplicates.
- **Global Deduplication:** The pipeline computes a perceptual hash (`phash`) of the canonical views. If this hash perfectly matches a card seen in a previous video, the new instance is marked as a duplicate, preventing database bloat.

### Stage 5: Lazy GPU Rectification (`gpu_refinement.py`, `cropper.py`)
- Rather than rectifying every frame during tracking, the pipeline waits until the Session is fully resolved. 
- It sequentially decodes *only* the specific high-res frames chosen as canonical candidates.
- `KorniaNormalizer` uses hardware acceleration to warp the 4-corner polygon into a flat, portrait image.

### Stage 6: The Diagnostic & Review UI (`review.py`, `timeline.html`)
- **Review UI:** A clean interface allowing users to Accept, Reject, and annotate notes on successfully captured cards, complete with a filmstrip of alternative canonical candidate crops.
- **Timeline UI:** A highly visual diagnostic view showing a chronological timeline of `Session Resets`, tracking events, and Card Instances. It visualizes how the pipeline grouped the temporal data, complete with thumbnails, duration, angles, and duplicate status.