# Design Spec: Payload Optimization & UI Precision

## 1. Overview
The pipeline is technically running in parallel (Multiprocessing), but it is **Payload-Bound**. We are currently sending 24MB (4K) frames across process boundaries, resulting in massive serialization overhead. This design introduces "Detection Proxies" and UI-based session diagnostics.

## 2. Performance: The "Small-Image-Proxy" Strategy
**The Problem:** 105 seconds for 280 frames is ~0.37s/frame. This slowness is caused by `pickle` serializing 7GB of 4K frame data across the `multiprocessing` queue.
**The Solution:**
- **Lightweight Producer:** The `_producer_main` will resize the frame to `detection_width` (640px) *before* putting it on the `frame_queue`.
- **Inference Speedup:** The YOLO `Consumer` will process these small images 10x faster.
- **Lazy High-Res Decode:** The `Refinement` stage (Stage 4) already receives the `source_frame_path`. It will decode the high-resolution 4K frame from disk *only* for the final 8 canonical images, keeping the process boundaries clean.

## 3. Precision & Orientation
- **Dynamic Rotation:** Add a config toggle `"auto_rotate_180": false`. Apply 180-degree rotation *only* if explicitly enabled, and ensure it's applied consistently to both GPU and CPU paths.
- **Aspect Ratio Padding:** Relax the detector gate further to **0.40 - 1.0** to ensure no cards are missed due to steep handheld angles.

## 4. UI Observability
- **Session IDs:** Display the `Session #` badge on each card in the review UI.
- **Null-State Timeline:** Add a "Session Resets" section to the bottom of the review UI, listing every time the workspace was marked empty and for how long.

## 5. Environment
- **HF Symlinks:** Move `os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"` to the very top of `src/card_capture/__init__.py` to ensure it loads before any other imports.

---
*Please approve this performance and precision design. I will then generate the implementation tasks.*