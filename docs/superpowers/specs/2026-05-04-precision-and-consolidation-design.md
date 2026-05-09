# Design Spec: Pipeline Precision & Session Consolidation

## 1. Overview
The pipeline is currently experiencing tracking fragmentation during flips, missing initial cards due to overzealous sampling thresholds, and upside-down orientations. To resolve this, we are abandoning track-level flip detection in favor of strict **Session-Based Consolidation**, adjusting sampling sensitivity, and introducing strict aspect-ratio gating to kill false positives.

## 2. Session-Based Consolidation (Fixing Duplicates & Flips)
**The Problem:** When a card flips, it goes edge-on, causing the ML model to lose the bounding box. The tracker sees a massive gap and creates a *new* track when the back of the card appears. Because it's a new track, it's labeled "Front" and gets evaluated as a separate card instance.
**The Solution:**
- **One Session = One Card.** The `AdaptivePresenceSampler` already defines "sessions" (continuous presence windows separated by empty gaps). We will assume that *all* tracks within a single session belong to the *same* physical card.
- **Front/Back Assignment:** The longest track in a session is the "Front". The second longest track (if it exists and is separated by a temporal gap) is the "Back".
- **Deduplication:** We completely bypass visual hashing for intra-session tracks. They are automatically merged into a single `CardInstance` with multiple `CardView`s.

## 3. Resolving Missing First Cards & Upside-Down Images
- **Missing First Card:** The `AdaptivePresenceSampler`'s Otsu threshold is discarding frames with low motion (like when a card is held still at the very beginning). We will implement a strict floor for the Otsu threshold so it never discards frames with very high `edge_density` and `sharpness` regardless of motion.
- **Upside-Down Images:** All images are consistently captured upside down. We will add a `rotate_180: bool = True` option to `PipelineConfig` and apply `cv2.ROTATE_180` to the final normalized image arrays.

## 4. Eradicating Lightbox Stand False Positives
- **Geometric Gating:** The current `0.1` to `0.8` area filter is not enough. The lightbox stand is being detected because it forms a polygon. Trading cards have a strict aspect ratio of 2.5/3.5 (~0.714).
- **The Fix:** Before accepting any detection in `detectors.py`, we will calculate the length of the edges and reject any polygon whose aspect ratio (short_edge / long_edge) falls outside the range of `0.60` to `0.85`.

## 5. Performance Diagnostics & HF Warnings
- **HF Hub Warning:** Inject `os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"` at the very top of `cli.py`.
- **Runtime Diagnostics:** We will add explicit `time.time()` wrappers around the 5 major pipeline stages (Sampling, Detection, Tracking, Normalization, Storage) and print a consolidated performance breakdown at the end of the run so a human can instantly see where the 6-minute bottleneck is occurring.

---
*Please review this design. Upon approval, I will generate the implementation plan.*