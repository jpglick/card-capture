# Detect Stage — YOLO Bottleneck Diagnostic

**Purpose:** Reference for a targeted session diagnosing why YOLO inference dominates detect time.
**Date recorded:** 2026-05-16

---

## Run Timing Data (all runs with stage instrumentation)

Only the last two runs have per-stage timing (instrumentation added 2026-05-16).

| run_id | status | wall time | detect | novelty | track | refine | score | resolve | dedup | store |
|---|---|---|---|---|---|---|---|---|---|---|
| run_f858eb1d | **failed** | 8m 48s | **6m 12s** | 7.3s | 2.1s | 2m 19s | 0.02s | 2.4s | — | — |
| run_68660d45 | **completed** | 8m 58s | **6m 9s** | 7.6s | 2.1s | 2m 21s | 0.02s | 2.7s | 1.4s | 0.4s |

**Detect is 68–69% of total wall time. Refine (GPU warp) is 26%. Everything else is noise.**

Earlier runs (no stage instrumentation, wall time only):

| run_id | status | wall time | notes |
|---|---|---|---|
| run_f47734ba | completed | 4m 35s | earlier video, fewer tracks |
| run_ca433690 | completed | 3m 46s | shorter video |
| run_413a4df4 | completed | 37s | fake/test detector |
| run_978c9659 | completed | 37s | fake/test detector |

---

## Detect Sub-Stage Breakdown (from log timestamps)

The detect timer is a single block. Internal breakdown inferred from log timestamps:

```
run_f858eb1d (failed, 6m 13s detect):
  14:30:50  detect/2 task starts
  14:30:51  subprocess spawns (sampler + YOLO consumer)
  14:31:17  AdaptivePresenceSampler._scan_video took 23.79 seconds  ← presence pass done
  14:37:02  detect/2 task finished

  Presence scan:   ~24 seconds   (6%)
  YOLO inference:  ~347 seconds  (94%)   ← THE BOTTLENECK

run_68660d45 (completed, 6m 9s detect):
  14:43:53  detect/2 task starts
  14:44:18  AdaptivePresenceSampler._scan_video took 21.30 seconds  ← presence pass done
  14:50:03  detect/2 task finished

  Presence scan:   ~22 seconds   (6%)
  YOLO inference:  ~347 seconds  (94%)
```

Both runs: **presence scan is fast and not the issue. YOLO is consuming ~5m 45s.**

---

## Architecture: Producer / Consumer

Detect runs two subprocesses concurrently via `multiprocessing` (spawn context):

```
video file
    │
    ▼
[Producer process]   src/card_capture/pipeline.py:_producer_main
  AdaptivePresenceSampler._scan_video()   → 192px @ 15fps fast pass
  ↓ yielded FrameSamples
  Frame Triage (blur, variance, empty_pixel, background)
  ↓ accepted frames (FramePackets at original resolution)
  frame_queue (maxsize=256)
         │
         ▼
[Consumer process]   src/card_capture/pipeline.py:_consumer_main
  CardcaptorUltralyticsDetector.detect_batch()
    - resizes frames to 640px wide
    - runs YOLOv8-OBB inference in batches of 16
    - filters by confidence ≥ 0.4
  detection_queue
         │
         ▼
[Main process]
  _drain_detection_queue()
  → DetectOutput (detection_rows list)
```

Key: producer and consumer run **concurrently**. If YOLO is slower than the sampler
(almost certainly true), the frame_queue fills up and the producer blocks. YOLO is
the pace-setter.

---

## Relevant Code

### Entry point: `pipeline/steps/detect.py`

```python
def run(ctx: RunContext) -> DetectOutput:
    sampler, detector = _build_sampler_detector(ctx)
    options = _ctx_to_options(ctx, output_dir)

    stats, raw_rows = _run_pipeline_workers(   # ← entire detect time is here
        video_path=video_path,
        video_id=ctx.video_id,
        frame_dir=frame_dir,
        sampler=sampler,
        detector=detector,
        options=options,
    )
```

Detector construction (hardcoded 640px):
```python
detector = CardcaptorUltralyticsDetector(
    confidence_threshold=ctx.corner_confidence,   # default 0.4
    detection_width=640,                           # HARDCODED — not from config
    device="auto",
)
```

### YOLO detector: `src/card_capture/detectors.py`

```python
class CardcaptorUltralyticsDetector:
    model_name = "AlecKarfonta/cardcaptor-v3"   # HuggingFace model

    def detect_batch(self, frames: list[FramePacket], confidence_threshold):
        model = self._load_model()
        detect_images = []
        scale_factors = []
        for frame in frames:
            # Resize to detection_width (640) maintaining aspect ratio
            if packet_w > self.detection_width:
                scaled_w = self.detection_width
                scaled_h = max(1, int(round(packet_h * self.detection_width / packet_w)))
                detect_image = cv2.resize(frame.image, (scaled_w, scaled_h))
            else:
                detect_image = frame.image
            detect_images.append(detect_image)

        # Single YOLO call for the whole batch
        results = model(detect_images, verbose=False)   # ← ultralytics batched inference
        # ... unpack OBB results, apply confidence filter, rescale corners back
```

Model loading (lazy, cached after first call):
```python
def _load_model(self):
    if self._model is not None:
        return self._model
    model_path = _resolve_model_path(self.repo_id, self.filename)
    self._model = YOLO(model_path)
    self._model.to(self._resolve_device())   # "auto" → probes MPS/CUDA/CPU
    return self._model
```

### Worker orchestration: `src/card_capture/pipeline.py`

```python
def _run_pipeline_workers(...):
    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=options.queue_size)       # default 256
    detection_queue = ctx.Queue(maxsize=options.queue_size)   # default 256

    producer = ctx.Process(target=_producer_main, ...)   # sampler + triage
    consumer = ctx.Process(target=_consumer_main, ...)   # YOLO batching

    producer.start()
    consumer.start()
    detections = _drain_detection_queue(detection_queue, ...)
    ...
```

Consumer batch size comes from `options.inference_batch_size` (default 16).

---

## Config Values (defaults, `src/card_capture/config.py`)

| param | default | what it controls |
|---|---|---|
| `inference_batch_size` | 16 | frames per YOLO call |
| `detection_width` | 640 | resize target before YOLO (also hardcoded in detect.py) |
| `corner_confidence` | 0.4 | YOLO confidence gate |
| `fast_scan_fps` | 15.0 | presence scan frame rate |
| `triage_keep_percentile` | 0.05 | fraction of frames triage keeps |
| `queue_size` | 256 | producer→consumer queue depth |
| `presence_threshold` | 0.4 | presence classifier gate (0.0 = disabled) |

**Note:** `detection_width=640` is hardcoded in `_build_sampler_detector` in `detect.py`
and ignores `ctx.detection_width` / `config.detection_width`. Both exist in config but the
hardcoded value wins at runtime.

---

## Key Unknowns to Measure

These numbers are not currently logged and need instrumentation to answer the question
"how many frames is YOLO actually seeing?"

1. **`accepted_frame_count`** — how many frames survived triage and entered the YOLO consumer.
   Available in `DetectOutput.accepted_frame_count` and `stats.accepted_frame_count` but
   not currently logged. This is the single most important number.

2. **Per-batch YOLO latency** — time per `model(batch, verbose=False)` call. Can be
   instrumented in `CardcaptorUltralyticsDetector.detect_batch()`.

3. **Device actually used** — `device="auto"` probes MPS/CUDA/CPU. The resolved device
   is in `TorchDeviceStatus.resolved` but not logged at startup. If YOLO is running on
   CPU instead of MPS the cost is 10–20× higher.

4. **Triage pass rate** — what fraction of presence-window frames survive Stage 2 triage.
   `triage_keep_percentile=0.05` sounds aggressive (keep top 5%) but the exact semantics
   need verification in `_producer_main`.

5. **Presence window coverage** — what fraction of the video is inside presence windows vs.
   skipped by the sampler entirely. Available in `sampler_telemetry` but not surfaced.

---

## Files to Focus On

```
src/card_capture/detectors.py              YOLO wrapper — detect_batch(), _load_model()
src/card_capture/pipeline.py               _run_pipeline_workers(), _producer_main(), _consumer_main()
src/card_capture/sampler/__init__.py       AdaptivePresenceSampler._scan_video(), _find_presence_windows()
pipeline/steps/detect.py                   Entry point, _build_sampler_detector(), _ctx_to_options()
src/card_capture/config.py                 All tunable knobs
```

---

## Hypotheses (prioritised)

1. **Too many frames passing to YOLO** — triage is not filtering aggressively enough, or
   the presence classifier (currently ~50% accuracy, barely trained) is marking too many
   frames as "present". More training data + retrain should reduce YOLO load directly.

2. **YOLO running on CPU** — if `device="auto"` resolved to CPU (MPS not available in the
   subprocess, or MPS returned an error), each batch takes ~10–20× longer. This is
   silent — only a log line at model load time would reveal it.

3. **batch_size=16 is suboptimal for MPS** — MPS throughput often improves with larger
   batches (32–64) up to VRAM limit. If throughput is the limit, increasing batch size
   reduces kernel overhead per frame.

4. **640px is overkill for long-range frames** — frames where the card occupies <20% of
   the frame don't benefit from 640px resolution. A dynamic resize (smaller for zoomed-out
   frames) could cut inference time on large-card-count videos.
