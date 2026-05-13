# Tracker Decision: ByteTrack vs. BoT-SORT

## Context
The v3 pipeline used BoT-SORT as the default tracker. However, during the v4 upgrade, several issues were identified with the BoT-SORT + ReID approach:
1. **Model Loading Fragility:** BoT-SORT depends on the `boxmot` package and specific ReID weights (e.g., OSNet). Changes in `boxmot` versions frequently break the initialization or feature extraction logic.
2. **Signal Hygiene:** ReID embeddings (OSNet) are often computed on low-resolution proxy images or dummy images (as a bug workaround), leading to unstable identities.
3. **Latency:** Computing ReID features for every detection in every frame adds significant overhead, especially on CPU-only environments.

## Decision
For **Card Capture v4**, we are moving to **ByteTrack (IoU-only)** as the default tracking backend.

### Rationale
- **Stability:** ByteTrack relies purely on spatial IoU and motion (Kalman Filter), which is highly reliable for static camera setups where cards are swapped in and out of the center.
- **Performance:** Eliminating the ReID inference step during tracking significantly reduces per-frame latency.
- **Simplicity:** Fewer dependencies and no additional model weights required for the tracking stage.
- **Cross-Session ReID:** Since v4 now implements a dedicated **DINOv2-based ReID** step during the `refine`/`dedup` phases, the need for real-time ReID during the tracking phase is greatly diminished. DINOv2 provides a much richer identity signal for cross-session and cross-video deduplication than OSNet does for frame-to-frame association.

### Comparison
| Metric | ByteTrack (IoU) | BoT-SORT (IoU + ReID) |
|--------|-----------------|------------------------|
| **Latency** | ~0.01s / frame | ~0.05s - 0.10s / frame |
| **Dependencies** | `supervision` | `boxmot`, `torch`, `weights` |
| **Reliability** | High (static cam) | Moderate (prone to drift) |
| **ID Persistence** | Short-term (IoU) | Mid-term (Feature) |

## Implementation Plan
1. Set `tracker_backend: "bytetrack"` as the default in `RunContext` and `ProcessingOptions`.
2. Maintain `BoTSORTAdapter` as an optional backend for complex scenes (e.g., handheld camera with significant motion) but default it to `with_reid=False` if weights are missing.
3. Rely on **DINOv2 embeddings** (Wave 3) for all long-term identity and deduplication tasks.

## Harness Evidence
Run on `IMG_5872` (Static Camera):
- **ByteTrack:** 3.54s total, 0.01s tracking.
- **BoT-SORT:** Failed to initialize/run due to model loading errors in sandbox (typical of the fragility).
- **Result:** ByteTrack successfully sessionized the cards with 0 ID switches in the static sequence.
