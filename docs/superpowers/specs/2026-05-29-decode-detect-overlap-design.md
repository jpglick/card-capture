# Decode↔Detect Overlap (Sampler Slow-Path Fix) — Design

**Date:** 2026-05-29
**Status:** Approved (Approach A)
**Branch context:** `v55-integration`

## Problem

The web/process path runs `LocalPipelineRuntime` (`src/card_capture/pipeline/runtime_local.py`),
which executes the ten stages strictly sequentially. The `sample` stage
(`src/card_capture/pipeline/stages/sample.py`) calls `list(sampler.sample())`,
fully materializing every sampled frame before `detect` starts. For the
reference run (`run_f8c85c4f`, `IMG_5922.MOV`) this stage took ~36 s with the
GPU/ANE idle the entire time.

### Measured evidence (this machine, the actual run video)

Video: **3840×2160, 10-bit HEVC, ~59.94 fps, 145.5 s, 969 MB** → ~8722 source
frames; `target_yolo_fps = 3.0` keeps ~436.

- The "hardware decode" request is a **no-op**. `sampler/__init__.py:229` passes
  `av.open(options={"hwaccel": "videotoolbox"})`, but `options` are *demuxer*
  (AVFormatContext) options — `hwaccel` is an ffmpeg-CLI concept, not a libav
  option key. It is silently ignored, `av.open` never raises, so `hw=True` is
  returned and the log prints a false `decoder=videotoolbox` (`:238`). Probe:
  with vs. without the option → identical format (`yuv420p10le`) and speed
  (254.9 vs 257.5 fps).
- Real hardware decode does **not** help wall time. Using PyAV's correct
  `HWAccel` API engages VideoToolbox (format flips to `p010le`) but is *slower*:
  199.5 fps vs 242.0 fps software. Apple's multithreaded software HEVC decoder
  beats single-stream HW decode for offline throughput.
- The 36 s is **decode volume**, not a per-frame slow path: decoding all ~8722
  frames at ~240 fps ≈ 36 s, plus `to_ndarray(format="bgr24")` on ~436 kept
  frames at 12.3 ms each ≈ 5 s. Decode-everything dominates (~85–90%).
- The GPU is idle during `sample` because the staged runtime serializes the
  stages: `sample` fully completes before `detect` begins. There is **no
  `UnifiedRuntime`** in the codebase (CLAUDE.md's "Unified Runtime" section is
  stale/aspirational); the `runtime/` package holds only GPU-boundary *guards*.

## Goals

- Keep the GPU/ANE busy during decode by overlapping `sample` (CPU/ffmpeg
  decode) with `detect` (CoreML inference).
- Reduce total pipeline wall time by ~`min(decode, detect)` (~25–30 s) — total
  goes from `decode + detect` to ≈ `max(decode, detect)`.
- Stay minimal and low-risk: no new orchestrator, no changes to downstream
  stages, no telemetry/sparkline structural changes.
- Independently, remove the misleading no-op `hwaccel` line.

## Non-Goals

- Building the producer/worker + eager crop-cache `UnifiedRuntime`.
- Reducing the ~10 GB frame-retention peak. Full-res frames in
  `state["sampled_frames"]` are required downstream by `novelty.py:38`,
  `track.py:33`, and `refine.py` (which warps from them and never re-decodes),
  so frames must persist `sample`→`refine` regardless. Bounding that peak is a
  separate crop-cache effort, explicitly deferred.
- Hardware video decode (measured slower here).

## Chosen Approach — A: Producer thread spanning `sample`→`detect`

`sample` starts a background decode thread that pushes frames into a bounded
queue and returns immediately; `detect` drains the queue, batches, and runs
CoreML while later frames are still decoding. `sample` and `detect` remain
distinct stages, so `_STAGES`, the telemetry `_STAGES` tuples, and the
sparkline `ALL_STAGES` are unchanged. `sample`'s timing shrinks to ~decode
startup; `detect`'s timing grows to the overlapped window — which honestly
reflects the new behavior.

Rejected alternative (B): merge into one `ingest` stage. Cleaner thread
lifecycle and a single timing, but ripples into `_STAGES`,
`app/services/pipeline_telemetry.py`, `app/worker_core.py`, and
`PipelineSparkline.svelte` for the same runtime win. Not worth the churn.

## Components

### `FrameProducer` (new, in `src/card_capture/sampler/`)

A small wrapper owning the decode thread.

- Constructed with a configured `StrideSampler` (and resolved video path).
- `start()`: launches a daemon thread that iterates `sampler.sample()`,
  `put`s each `FrameSample` onto `queue.Queue(maxsize=32)`, and on completion
  `put`s a sentinel. Any exception is captured in an error holder, then the
  sentinel is enqueued so the consumer always unblocks.
- Exposes the `queue`, a `join()`, and an `error` accessor.
- Owns and closes its PyAV container in a `finally` inside the thread.
- The producer performs **no** telemetry calls (thread safety).

### `sample.py` (modified)

- Drop `frames = list(sampler.sample())`.
- Build the `StrideSampler` and populate up-front attributes downstream stages
  read (e.g. `last_source_fps`, `last_inter_window_gaps_frames`) — these are
  computed from a cheap cv2 FPS probe before any decoding.
- Create and `start()` a `FrameProducer`.
- Set `state`: `frame_queue`, `frame_producer`, `sampler`, `video_path`, and
  `sampled_frames = []` (filled by the consumer).
- Block until the first frame is available, then return. This keeps a small,
  meaningful `sample` bar in the sparkline (decode startup) and costs only one
  un-overlapped frame of decode (~negligible). All remaining frames overlap
  `detect`.

### `detect.py` (modified)

- Load the YOLO/CoreML detector once (unchanged).
- Replace `frames = state["sampled_frames"]` iteration with a queue-drain loop:
  pull frames until the sentinel, accumulate batches of 16, call
  `detect_batch`, extend `detections`, **append each consumed frame to
  `state["sampled_frames"]`**, and emit the existing
  `telemetry.progress("detect", …)`.
- On sentinel: `producer.join()`; if `producer.error` is set, re-raise it.
- Set `state["detections"]` as today.

## Data Flow

```
sample stage (main thread)
  build StrideSampler  ──>  FrameProducer.start()
                              │  (daemon thread)
                              │   for f in sampler.sample():
                              │       queue.put(f)        # bounded(32) → backpressure
                              │   queue.put(SENTINEL)
  state += {frame_queue, frame_producer, sampler, sampled_frames=[]}
  return

detect stage (main thread)            ── overlaps decode thread ──
  while True:
    f = queue.get()
    if f is SENTINEL: break
    batch.append(f); sampled_frames.append(f)
    if len(batch)==16: detections += detect_batch(batch); progress(...)
  flush final batch
  producer.join(); raise producer.error if any
  state["detections"] = rows
```

## Concurrency, Memory, Performance

- Overlap: producer (CPU decode) runs concurrently with consumer (CoreML/ANE).
  Total ≈ `max(decode, detect)` instead of the sum.
- `maxsize=32` bounds the producer's lead and caps in-flight frames; it does not
  reduce the retained `sampled_frames` peak (intentional — downstream needs it).
- Expected: ~25–30 s off the run; GPU/ANE busy during decode.

## Telemetry & Thread Safety

- Only the consumer (main thread) calls `telemetry`. The producer never does.
- `sample` timing → ~decode startup; `detect` timing → overlapped window.
- No changes to `_STAGES`, the telemetry layer, or the sparkline.

## Error Handling

- Producer exception → captured in holder, sentinel enqueued so the consumer
  unblocks. `detect` joins and re-raises, so the runtime's existing
  `stage_failed:detect` path (`runtime_local.py:121`) records it.
- Producer always closes its PyAV container (`finally`).
- If `detect` itself raises mid-drain, it must still `join()` the producer
  (drain or signal-stop) so the daemon thread does not linger; handled in a
  `finally`.

## Independent Fix — drop the no-op hwaccel line

In `src/card_capture/sampler/__init__.py`:
- Remove the `options={"hwaccel": "videotoolbox"}` branch in
  `_open_pyav_container` (`:223–233`) so we just `av.open(str(video_path))`.
- Update the `_sample_with_pyav` log (`:238`) to stop claiming
  `decoder=videotoolbox`; report the real software decode.

This is correctness/honesty only — software decode is already the faster path
here, so there is no performance regression.

## Testing

- **Unit — `FrameProducer`:** frames yielded match `StrideSampler.sample()` in
  count and order; sentinel terminates the consumer; a sampler that raises
  propagates via the error holder; queue bound is respected.
- **Integration — `sample`+`detect` parity:** over a short fixture video, the
  overlapped path produces identical `state["detections"]` and
  `state["sampled_frames"]` (count/order/content) vs. the current sequential
  path. Reuse existing pipeline tests where possible.
- **Regression:** full suite `python3 -m pytest tests/ -m "not quarantine" -q`.
- **hwaccel:** assert no `videotoolbox`/`hwaccel` option is passed to `av.open`
  and the log no longer claims hardware decode.

## Risks

- Thread lifecycle: a decode thread outliving the `sample` stage. Mitigated by
  daemon thread + explicit `join()` in `detect`'s `finally` and the error
  holder. Bounded queue prevents runaway memory.
- Downstream sampler attributes: ensure `state["sampler"]` still carries the
  fields later stages read (populate before starting the producer). Verify
  against `track.py` and any sampler-attribute consumers during implementation.
