# Remaining Work — Wave 5 and Beyond

**Date:** 2026-05-16  
**Theme:** Accuracy is good. Focus shifts to performance, code health, and model integration.

This document captures all unimplemented roadmap items from the v4 audit, organized into
waves. Wave 5 is the immediate target. Later waves are prioritized but not scheduled.

---

## Wave 5 Status — **COMPLETE** (2026-05-16)

| Item | Status | Notes |
|---|---|---|
| 5.1 Field-drop assertion | ✅ Done | `PipelineConfig.to_options()` now raises immediately on uncovered field |
| 5.2 Retire `pipeline.py` | ✅ Done | Deleted (~2500 lines). Worker code lives in `src/card_capture/workers.py`. CLI monolith path removed. |
| 5.3 CoreML fast path | ✅ Investigated — MPS wins | Thorough benchmark showed MPS (94ms/frame) is 3× faster than ONNX+CoreML EP (325ms/frame). ANE does not outperform GPU for YOLO11m at usable batch sizes. Also discovered: model is already YOLO11m-OBB. ONNX export kept as verified fallback. |
| 5.4 VideoToolbox decode | ✅ Done | `_sample_with_pyav()` tries `hwaccel=videotoolbox` on macOS first; logs decoder at startup. `prefer_hw_decode: bool = True` in `PipelineConfig`. |

---

## CoreML Investigation (2026-05-16)

**Verdict: ultralytics CoreML OBB export is structurally broken — not a configuration issue.**

Tested with a real card frame against both backends:

| | MPS (working) | CoreML task=obb (broken) |
|---|---|---|
| `result.obb.conf` | `[0.67, 0.58, 0.44]` — proper probabilities | `[39.4, 9.2, 45.8, 18.6]` — raw logits |
| Coordinates (x) | `184, 641, 410, -46` — pixel space | `-238, -185, -211, -264` — all negative |
| Polygon area | Card-sized (~100k px²) | 0.4–0.8% of frame (garbage) |

**Root cause:** The ultralytics CoreML export includes only the raw neural network weights. The YOLO post-processing — sigmoid on confidence, anchor → pixel coordinate decode, OBB angle extraction — runs in Python inside `ultralytics.engine.predictor` for the MPS path, but is **not baked into the CoreML model**. When ultralytics loads the `.mlpackage`, it skips that Python post-processing and passes raw network outputs directly to the results parser, producing logits instead of probabilities and pre-decoded offsets instead of pixel coordinates.

This affects all OBB export formats that don't include post-processing. ONNX with `opset≥17` includes the decode; CoreML does not. `task='obb'` is a metadata hint that doesn't fix the missing computation.

**Full investigation results (2026-05-16):**

| Backend | ms/frame (batch=16) | Notes |
|---|---|---|
| MPS PyTorch | **94ms** | Winner |
| ONNX+CoreML EP, batch=1 loop | 325ms | ANE active (419/441 ops) but loop overhead dominates |
| ONNX CPU, batch=16 | 296ms | CoreML EP rejects static b>1, falls back to CPU |

ANE is optimized for lightweight mobile architectures (MobileNet, EfficientNet). YOLO11m with 134 layers and 20M parameters saturates the M4 GPU more efficiently than the ANE.

**Key discovery:** `cardcaptor-v3` is already YOLO11m-OBB — the planned model upgrade was already done by the model author. No retraining needed.

**ONNX export** (`models/cardcaptor_v3.onnx`, 80 MB) is kept as a verified-correct fallback for non-MPS environments (Linux + CUDA, CI without Metal).

**Current status:** MPS is primary backend. ONNX/CoreML paths disabled. `.mlpackage` kept for reference.

---

## Wave 5 — Performance & Code Health (immediate)

### 5.1 Fix config field-drop bug

**Current state:** `PipelineConfig.to_options()` auto-maps fields by name into
`ProcessingOptions`. Any `ProcessingOptions` field with no matching `PipelineConfig`
field silently falls through to its dataclass default — meaning a config value set in
`card_capture_config.json` can be invisibly ignored at runtime. The comment in the code
even acknowledges this: `# else: ProcessingOptions-only field; use its default`.

**What to do:**
- After `to_options()` builds `kwargs`, assert that every non-default-only field in
  `ProcessingOptions` is covered either by a rename entry or a name match in
  `PipelineConfig`. Raise `ValueError` with the uncovered field name.
- Alternatively, delete `ProcessingOptions` entirely and make
  `_run_pipeline_workers` accept a `PipelineConfig` directly (preferred — removes the
  dual-dataclass problem that caused this bug in the first place).

**Impact:** Silent correctness bugs. Config changes that appear to stick but don't.

---

### 5.2 Retire `pipeline.py` (monolith)

**Current state:** `src/card_capture/pipeline.py` is deprecated and prints a
`DeprecationWarning` on import. It is still alive because `pipeline/steps/detect.py`
imports `_run_pipeline_workers`, `_producer_main`, `_consumer_main`, and
`ProcessingOptions` from it. The `VideoProcessor` class is unused by the Metaflow path.

**What to do:**
1. Move `_run_pipeline_workers`, `_producer_main`, `_consumer_main`, `_consume_batch`,
   `_ConsumerStats`, `_ProducerStats`, `ProcessingOptions`, and all helper functions they
   depend on into a new module: `src/card_capture/workers.py` (or
   `pipeline/steps/_workers.py`).
2. Update `pipeline/steps/detect.py` import to point at the new location.
3. Delete `pipeline.py`. The `VideoProcessor` class (legacy single-process path) goes
   with it.
4. Remove the `DeprecationWarning` machinery.

**Impact:** Removes ~2500 lines of dead code. Eliminates the dual-maintenance surface.
Future changes to the producer/consumer only happen in one place.

---

### 5.3 CoreML fast path for YOLO

**Current state:** `CardcaptorUltralyticsDetector` loads `cardcaptor-v3` as a PyTorch
model and runs inference via the MPS backend. A CoreML wrapper stub exists at
`src/card_capture/ml/models/coreml_detector.py` but `coremltools` is not installed and
the export has never been run.

**What to do:**
1. Add `coremltools` to `pyproject.toml` optional deps (`[model]` extras group).
2. Export the existing weights once:
   ```python
   from ultralytics import YOLO
   model = YOLO("~/.cache/huggingface/hub/.../cardcaptor_v3_best.pt")
   model.export(format="coreml", imgsz=640, nms=True)
   # produces cardcaptor_v3_best.mlpackage alongside the .pt
   ```
3. Cache the `.mlpackage` in `models/` so it ships with the repo (it's ~20 MB).
4. In `CardcaptorUltralyticsDetector._load_model()`, on `platform.system() == "Darwin"`
   and `platform.machine() == "arm64"`, prefer the `.mlpackage` over the `.pt`. Ultralytics
   can load and run CoreML models with the same `YOLO(path)(images)` API.
5. Add `device="coreml"` to `probe_torch_device_status` or handle it at the detector level.

**Expected impact:** CoreML models run on the Apple Neural Engine (ANE) rather than the
GPU for the compute-heavy layers, freeing the GPU for Kornia refinement. Typical
throughput improvement over MPS: 1.5–2.5× on OBB inference at 640px.

**Note on "YOLO26":** The roadmap used this as a placeholder for a future model
generation. No such version exists in the current ultralytics package. The CoreML export
of the existing YOLOv8-OBB weights is the realistic implementation of Phase 4 item 1.
Model architecture upgrades (v9, v11) can happen later if accuracy warrants retraining.

---

### 5.4 VideoToolbox hardware decode for frame sampling

**Current state:** `_open_capture()` in `ingestion.py` already tries
`cv2.CAP_AVFOUNDATION` (AVFoundation, which uses VideoToolbox internally) as its first
choice for OpenCV-backed reads — so the presence scan pass (`_scan_video`) already gets
hardware decode. The gap is `_sample_with_pyav()` and `_sample_with_decord()`, which are
used for the high-fps frame sampling pass and don't request hardware acceleration.

**What to do:**
1. In `_sample_with_pyav()`, open the container with VideoToolbox codec options:
   ```python
   container = av.open(str(video_path), options={
       "allowed_extensions": "ALL",
       "hwaccel": "videotoolbox",     # FFmpeg VideoToolbox hwaccel
   })
   ```
   Wrap in try/except and fall back to the current software path if VideoToolbox isn't
   available (non-macOS, or unsupported codec).
2. In `_sample_with_decord()`, check if decord was built with VideoToolbox support
   (`decord.VideoReader` accepts `ctx=decord.gpu(0)` on CUDA; for VideoToolbox there is
   no direct decord API — if unavailable, prefer pyav with VideoToolbox over decord on
   macOS).
3. Expose a `prefer_hw_decode: bool = True` flag in `PipelineConfig` for diagnostics.
4. Log the resolved decoder at sampler startup (one line: `[sampler] decoder=pyav/videotoolbox`).

**Expected impact:** Hardware decode on H.264/HEVC reduces CPU load during the 15fps
presence scan and the full-resolution frame sampling pass. On Apple Silicon, VideoToolbox
decode uses the dedicated video decoder engine (not CPU or GPU), so it runs
concurrently with YOLO inference on the GPU.

---

## Wave 6 — ML Model Integration

Items where the code infrastructure exists but the models are untrained or not wired into
the inference path.

### 6.1 Train and wire the F/B classifier

**Current state:** `src/card_capture/ml/models/fb_classifier.py` (MobileNetV3-Small)
exists. `models/fb_classifier.pt` exists but contains random weights — it has never been
trained. The inference path in `pipeline/steps/store.py` calls `fb_predict.py` which
guards with `is_available()` and falls back to the longest-track heuristic when the model
isn't trained.

**What to do:**
1. Wire the existing `POST /training/retrain/fb_classifier` endpoint to actually call
   `PresenceTrainer` (or the FB equivalent) — the endpoint exists but the handler doesn't
   trigger training.
2. Collect labels via the `/label/fb` UI until the dataset is large enough (~500+
   labeled crops per class recommended).
3. Run retrain, verify eval metrics, promote checkpoint to `models/fb_classifier.pt`.
4. Remove the heuristic fallback once the model achieves >90% accuracy on the eval set.

**Impact:** F/B side detection is currently heuristic-based. A trained classifier
directly improves accuracy on videos where the same card appears front and back in the
same session.

---

### 6.2 Wire DINOv2 + FAISS dedup into the inference path

**Current state:** `src/card_capture/ml/inference/dino_dedup.py` (FAISS dedup) and
`src/card_capture/ml/models/dino_embedder.py` (ViT-S/14 embedder) both exist and are
tested. `pipeline/steps/dedup.py` still calls the pHash deduplicator
(`src/card_capture/deduplicator.py`). The DINOv2 path is never invoked.

**What to do:**
1. In `pipeline/steps/dedup.py`, add a feature flag `use_dino_dedup: bool` to
   `PipelineConfig` (default `False` while validating).
2. When `True`, call `dino_dedup` instead of `deduplicator`. The output contract is the
   same (dedup groups).
3. Run regression harness against golden set with both paths and compare card recall/precision.
4. Flip the default to `True` once DINOv2 outperforms pHash on the golden set.

**Note:** ViT-S/14 adds ~100ms per run for embedding (runs once on canonical crops, not
per frame). FAISS query is sub-millisecond at current card volumes.

---

### 6.3 Presence classifier retrain endpoint

**Current state:** `src/card_capture/training/presence_trainer.py` exists and is tested.
`POST /training/retrain/presence_classifier` exists in the API but the handler doesn't
call the trainer — it returns a stub job.

**What to do:** Wire the handler to `PresenceTrainer.train()` and surface training
progress via SSE. The labeled data already flows through the `/training/presence` labeling
UI.

**Impact:** Better presence classification reduces the number of frames that enter YOLO
(the main bottleneck) by more aggressively gating empty-workspace frames.

---

## Wave 7 — Detection Intelligence

Items that require new algorithmic work, not just wiring.

### 7.1 Detection-conditioned sampler (objectness signal)

The sampler currently selects frames by sharpness percentile (triage) with no input from
the detector. A cheap objectness signal — a tiny classification head or the first YOLO
layer alone — could gate frame selection before full inference. This collapses the
produce/consume gap: instead of the consumer being the bottleneck, the producer becomes
smarter and sends fewer frames.

**Depends on:** Wave 5.4 (hardware decode reduces producer CPU load, making objectness
feasible in the producer process).

### 7.2 Per-pixel background variance model

`per_video_adaptive.py` maintains a mean background model for null-state detection.
Extending it to track per-pixel variance would allow the null-state detector to suppress
false positives on regions that are always noisy (e.g., studio lights, reflective
surfaces) without raising the global threshold.

### 7.3 Higher-resolution rectified canvas

Canonical crops are currently normalized to a fixed size. Increasing to 1000×1400 with
Lanczos resampling preserves more card detail for dedup and F/B classification. Should
be gated behind a config flag since it increases storage and Kornia warp time.

### 7.4 vImage perspective warp

The Kornia warp in `pipeline/steps/refine.py` runs on MPS. Apple's `vImage` framework
(CPU, but SIMD-optimized) can do perspective warps at comparable speed to MPS for the
sizes involved, and doesn't compete with YOLO/CoreML for GPU time. Worth benchmarking
once CoreML (Wave 5.3) is live — if YOLO moves to ANE, MPS may be idle during refine
and Kornia is already optimal.

---

## Wave 8 — Infrastructure & Quality

Lower urgency, but accumulating technical debt.

### 8.1 Structured exit codes and error protocol

Child process failures (producer/consumer) currently propagate through a timeout + error
queue mechanism that can silently hang for 60 seconds before surfacing. Replacing with
structured exit codes and immediate parent notification would reduce failure diagnosis
time significantly.

### 8.2 Real ReID embeddings for BoT-SORT

`pipeline/steps/store.py` has a stub `reid_embedding` column. ByteTrack (the current
default) doesn't use ReID. If BoT-SORT is re-enabled, populating the embedding from
DINOv2 (available after Wave 6.2) would improve re-identification across track gaps.

### 8.3 Learned quality ranker

The current `QualityScorer` in `src/card_capture/scoring.py` uses a weighted heuristic
(sharpness, glare, border purity, confidence). A learned ranker trained on user verdicts
from the labeling UI would directly optimize for human-judged quality. Low priority while
the heuristic correlation with verdicts remains high.

### 8.4 A/B regression comparison view

`app/api/regression.py` has the route stub. The compare endpoint returns data but the
frontend `routes/regression/compare/` page is not functional. Blocked on having enough
baseline runs to make comparison meaningful.

---

## Deferred (not planned)

| Item | Reason |
|---|---|
| VNDetectRectanglesRequest | Classical CV fails on glare/foil/occlusion |
| Single-frame neural glare removal | Ill-posed on holographic surfaces; median fusion is principled |
| Cloud GPU training | Local MPS sufficient at current dataset sizes |
| OpenTelemetry / Prometheus | SQLite telemetry is sufficient |
| Qdrant vector DB | FAISS in-process is sub-ms for ≤100K cards |
| YOLOv9/v11 architecture swap | Would require retraining cardcaptor from scratch; no labeled detection dataset in repo; defer until model quality is the limiting factor |

---

## Open Questions

1. **DINOv2 variant:** ViT-S/14 (speed) vs ViT-B/14 (accuracy) for dedup. Benchmark
   on labeled dedup groups before committing.
2. **BoT-SORT vs ByteTrack:** ByteTrack currently active. BoT-SORT with real ReID needs
   Wave 6.2 (DINOv2) and Wave 8.2 before it's competitive. Revisit after Wave 6.
3. **CoreML batch size:** CoreML models have a fixed batch dimension at export time.
   Export at `batch=1` for latency, or `batch=N` matching `inference_batch_size`. Profile
   both before committing.
