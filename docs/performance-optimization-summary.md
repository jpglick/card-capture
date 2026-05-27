# Performance Optimization Summary: 4K Zero-Download Pipeline

This document chronicles the rounds of performance investigation, diagnosis, and fixes applied to the Card Capture pipeline to optimize it for high-resolution 4K processing on Apple Silicon (Mac Mini M4).

## Initial State & Constraints
* **Issue:** 4K runs were extremely slow, crashing with Out-Of-Memory (OOM) errors, and filling up the disk. 
* **Hardware:** Mac Mini M4. The pipeline needed to saturate the Media Engine (for decoding), the GPU/MPS (for tensor ops/warping), and the Apple Neural Engine (ANE via CoreML for YOLO inference).

## Round 1: MPS Fast Path & Best-Frame Selection
* **Problem:** The existing pipeline eagerly warped every detection using Kornia, which at 4K resolution caused massive VRAM usage and OOM crashes.
* **Fix:** Transitioned from a "warp everything" to a "select then warp" strategy.
* **Implementation:**
  * Implemented cheap, warp-free `flatness_score` and `clarity_score_gpu` (Laplacian variance on the ROI).
  * Routed MPS runs through the non-fused path, allowing the pipeline to shortlist the top 5 candidates per track based on cheap scores before performing the expensive high-res Kornia warp.
  * Added a strict `require_device("mps")` guard to prevent silent performance regressions to CPU.

## Round 2: The Coordinate Scaling Bug (Poor Detections)
* **Problem:** Detections were not actually cards. The pipeline was generating false positives by warping arbitrary background regions.
* **Diagnosis:** A coordinate scaling mismatch. The pipeline resized 4K frames to letterboxed 640px thumbnails for YOLO, but YOLO's output coordinates were being applied back to the 4K frames without properly accounting for the padding and scale factors.
* **Fix:** Implemented the **Square Squeeze**. The pipeline now resizes 4K frames to exactly 640x640 (ignoring aspect ratio). This eliminated letterboxing ambiguity, allowing for perfect, simple mathematical scaling of the bounding box coordinates back to the original 4K space.
* **Novelty Restoration:** Re-enabled the Novelty Gate in the MPS path using background proxies. Without background subtraction, YOLO was tracking static rectangular shadows and desk features as cards.

## Round 3: The 20fps Bottleneck & Zero-Download Pipeline
* **Problem:** The pipeline hit a wall at ~20fps. The GPU was underutilized.
* **Diagnosis:** "Sync Stalls" caused by CPU-to-GPU data transfers. Even though images were resized on the GPU, the pipeline was downloading the massive 25MB 4K frames back to the CPU to compute novelty scores and color conversions.
* **Fix: Zero-Download Architecture.**
  * Moved all 4K NV12-to-BGR color conversions to the GPU.
  * Moved the Novelty Gate to the GPU (`gpu_roi_mean_abs_diff`), doing background subtraction against a GPU-resident background model in VRAM.
  * Rewrote the DINOv2 ReID embedding pass to accept GPU tensors directly (`embed_tensors_batch`), entirely bypassing the Disk-to-CPU-to-GPU bounce where crops were previously saved to disk and re-read via PIL.
  * Shrunk the batch size to 4 to bound VRAM usage and prevent swap thrashing.

## Round 4: The 10fps Wall & Deep Tensor Stalls
* **Problem:** Despite the Zero-Download pipeline, throughput inexplicably dropped to ~10fps.
* **Diagnosis & Fixes (Three hidden bottlenecks):**
  1. **CoreML Slicing Stall:** Slicing an asynchronous PyTorch MPS tensor (e.g., `batch[i:i+1]`) to feed into the CoreML model forced a catastrophic 150ms synchronous stall per frame as the driver struggled to resolve memory pointers.
     * *Fix:* Injected a hard `torch.mps.synchronize()` immediately before the CoreML inference loop, boosting inference to >50fps.
  2. **Eager Warp Memory Churn:** When Kornia warped the cards, PyTorch's broadcasting mechanism created a massive `float32` copy of the entire 4K canvas *for every single detection*, causing ~2.5GB of VRAM churn per batch and taking 100ms.
     * *Fix:* Added a BBox Pre-Crop step. The 4K tensor is sliced down to the card's bounding box region *before* passing it to Kornia, dropping warp time from 100ms down to 2ms.
  3. **The PyAV Decoder Bug:** The PyAV (FFmpeg) decoder couldn't properly map reference frames (B-frames) in Apple's 4K HEVC iPhone footage, causing it to crash into a slow software fallback (6fps).
     * *Fix:* Switched the macOS decoding backend to OpenCV `CAP_AVFOUNDATION`. This natively hooks into Apple's Media Engine. Implemented a `grab()` and `retrieve()` loop to fast-forward the hardware decoder without ever allocating 25MB BGR arrays into RAM for skipped frames.

## Round 5: Concurrency, Threading, & UI Visibility
* **Problem:** The Svelte UI progress bar moved in lockstep, making it impossible to see which hardware component was the bottleneck. Occasional tracking errors occurred.
* **Fixes:**
  * **Tri-Buffer Pipeline UI:** Expanded the progress reporting to show three concurrent bars (`detect.decoder`, `detect.gpu`, and `detect.main`), complete with real-time FPS metrics.
  * **Queue Expansion:** Increased inter-thread queue sizes from 2 to 16, allowing the Media Engine decoder to run far ahead of the Neural Engine, maximizing overlap.
  * **Thread Memory Corruption:** OpenCV's `retrieve()` reuses internal C++ memory buffers. Because the decoder thread ran ahead, it overwrote frames sitting in the queue, feeding garbage to the GPU worker. Fixed by explicitly calling `.copy()` on the sampled frames.
  * **Tracker Hardening:** Fixed a `KeyError` on missing `source_frame_path` metadata, and patched a `card_swap` hash instability that caused false-positive session resets when multiple cards were on screen.

## Final Result
The pipeline is now a highly-optimized, concurrent, **Zero-Download** architecture that correctly saturates the M4's Media Engine, GPU, and Neural Engine simultaneously, consistently processing 4K HEVC footage at the hardware's physical limits (30-50+ FPS).