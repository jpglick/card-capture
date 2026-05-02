# Quick Reference: Card Capture v2.1

## v2.1 Problem Statement
Extract high-quality trading card stills from local videos with predictable throughput and a bounded-memory multiprocessing flow.

## High-Level v2.1 Flow
```
Video -> Stage 1 producer (sample + triage + persist frame) -> frame_queue
      -> Stage 2 consumer (batched detector inference + confidence filter) -> detection_queue
      -> Parent orchestration (storage writes + candidate selection + best export)
```

This is a producer/consumer pipeline implemented in `src/card_capture/pipeline.py`.

## v2.1 CLI Flags (process)

```bash
card-capture process <video> \
  --output-dir <dir> \
  --db <db.sqlite> \
  --detector {docaligner,fake} \
  --reader-backend {auto,decord,pyav} \
  --queue-size 64 \
  --inference-batch-size 16 \
  --corner-confidence 0.5 \
  --blur-threshold 30.0 \
  --variance-threshold 20.0 \
  --empty-pixel-threshold 0.98 \
  --detection-width 640 \
  --device {auto,cpu,mps,cuda}
```

## v2.1 Runtime Entities

- `ProcessingOptions`: runtime knobs passed from CLI to pipeline.
- `_FrameEnvelope`: stage1 payload (`FramePacket` + saved source frame path).
- `_DetectionEnvelope`: stage2 output payload (`DetectionPacket` + triage metrics + source path).
- `_ProducerStats`: stage1 counters (`frame_count`, `accepted_frame_count`).
- `ProcessingResult`: top-level result returned to CLI with frame/detection/save totals.

## Stage Responsibilities

1. Stage 1 Producer
   - Reads sampled frames.
   - Applies triage (`blur`, `variance`, `empty_pixel` thresholds).
   - Saves accepted frame JPEGs.
   - Pushes `_FrameEnvelope` rows to `frame_queue`.
2. Stage 2 Consumer
   - Collects frames into batches (`--inference-batch-size`).
   - Runs detector inference (`detect_batch` if available, otherwise per-frame fallback).
   - Filters by `--corner-confidence`.
   - Pushes `_DetectionEnvelope` rows to `detection_queue`.
3. Parent Process
   - Drains detections until sentinel.
   - Persists videos, instances, views, and evidence rows.
   - Runs selector and copies canonical images into `output_dir/best`.

## Queue and Backpressure Knobs

- `--queue-size` controls both `frame_queue` and `detection_queue` capacity.
- `--inference-batch-size` trades latency vs detector throughput.
- Queue puts use bounded retry/backoff; hard timeouts raise runtime errors instead of hanging.

## Smoke Command (v2.1)

```bash
card-capture process <temp-video> \
  --output-dir <temp-out> \
  --db <temp-db> \
  --detector fake \
  --reader-backend auto \
  --queue-size 8 \
  --inference-batch-size 4 \
  --corner-confidence 0.5
```
