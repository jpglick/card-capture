# Card Capture v2.1 Architecture & Implementation Specification

**Project:** Sports Trading Card Image Extraction from Video  
**Target Audience:** Implementation LLM / Coding Agent  
**Status:** Approved Architecture Specification  

## 1. Architectural Mandate & Paradigm Shifts

This specification replaces the v1 and proposed v2 pipelines. The goal is to maximize accuracy, eliminate CPU-bound I/O bottlenecks, and structure the data for future ML grading.

You are instructed to implement the following core architectural shifts:
1. **I/O Shift:** Replace `cv2.VideoCapture` with `decord` for high-speed, batch-oriented video ingestion.
2. **Detection Shift:** Deprecate YOLO bounding-box detection + contour/edge refinement. Implement zero-shot Document Corner Detection (via `docaligner-docsaid` or equivalent ONNX model) to predict 4 keypoints directly.
3. **Grouping Shift:** Deprecate perceptual hashing (pHash) for deduplication. Implement Object Tracking (SORT/Centroid-based) across sequential frames to group detections into a single `CardInstance`.
4. **Processing Paradigm:** Transition from synchronous single-thread processing to an asynchronous or Multi-Processing Queue (Producer-Consumer) architecture.

---

## 2. Updated Data Model

The core abstraction shifts from "Best Frame" to a "Multi-View Instance".

* **`CardInstance`**: Represents a single physical trading card tracked through the video. Assigned a unique tracking ID (e.g., `card_1`).
* **`CardView`**: A specific perspective-corrected extraction of a `CardInstance` from a single frame.
* **`Evidence`**: The raw, uncropped frame data associated with a `CardView`, preserved for future diagnostic or grading models.

---

## 3. The v2.1 Pipeline Stages

### Stage 1: Ingestion & Fast Triage (Producer Thread)
* **Tool:** `decord.VideoReader`
* **Action:** Read video frames sequentially into NumPy arrays/PyTorch tensors.
* **Filter:** Compute a lightweight blur/variance metric on the raw frame. Drop frames that are entirely blurred (camera motion) or completely empty.
* **Output:** Push valid `(frame_index, frame_data)` tuples to an Inference Queue.

### Stage 2: Zero-Shot Corner Detection (Consumer Thread - GPU)
* **Tool:** `docaligner` (ONNX) or equivalent 4-corner regression model.
* **Action:** Pull frames from the Inference Queue. Run inference to detect the 4 exact corners `[top-left, top-right, bottom-right, bottom-left]` of the trading card.
* **Output:** Skip frames with low confidence. Pass successful detections to the Tracking logic.

### Stage 3: Object Tracking & Instance Grouping
* **Tool:** Simple Online and Realtime Tracking (SORT) or custom Centroid-Distance tracker.
* **Action:** Calculate the bounding rect or centroid of the 4 detected corners. Compare to active tracks in previous frames using Intersection over Union (IoU) or Euclidean distance.
* **Outcome:** * If matched: Append frame data to existing `CardInstance`.
    * If unmatched: Spawn a new `CardInstance` ID.

### Stage 4: Perspective Rectification (Homography)
* **Tool:** `cv2.getPerspectiveTransform` and `cv2.warpPerspective`
* **Action:** Using the 4 predicted corners from Stage 2, compute the homography matrix to warp the card region into a standardized, perfectly vertical, fixed-ratio rectangle (e.g., 2.5 x 3.5 aspect ratio).

### Stage 5: View Quality Scoring
* **Action:** For every `CardView` within a `CardInstance`, compute quality metrics on the *rectified* crop:
    1.  **Sharpness:** Laplacian variance.
    2.  **Glare/Reflection:** Saturated pixel percentage thresholding.
* **Selection:** Identify the `CardView` with the highest combined score as the `Canonical View`. Keep 2-3 alternative views (e.g., highest sharpness with different glare profiles) as supplementary evidence.

### Stage 6: Persistence & Storage
* **Action:** Write the `CardInstance` and its associated `CardViews` to the SQLite database.
* **Artifacts:** Save the rectified crops (JPEG/WebP) and the raw evidence frames.

---

## 4. Implementation Directives for LLM Agent

When executing this specification, adhere strictly to these rules:

1.  **Isolate Work:** Do not attempt to refactor the entire pipeline in one pass. Issue changes in modular, testable steps (e.g., Step 1: Swap OpenCV for Decord. Step 2: Implement DocAligner. Step 3: Implement Tracking).
2.  **Type Hinting:** Use strict Python type hints (`typing` module, `dataclasses`, or `pydantic`) for all data structures passing between pipeline stages.
3.  **ONNX Preference:** When integrating the corner detector, prefer `onnxruntime` implementations over heavy PyTorch dependencies to keep the deployment footprint lightweight, unless GPU acceleration requirements dictate otherwise.
4.  **Graceful Degradation:** If `decord` fails to build/install on a specific architecture, provide a fallback to `PyAV` (`av` module). Do not fall back to `cv2.VideoCapture` unless absolutely necessary.
5.  **No Hand-Coded Contours:** Do not write any code utilizing `cv2.Canny`, `cv2.findContours`, or Hough transforms for card boundary detection. Rely entirely on the neural network (Stage 2) for geometric keypoints.