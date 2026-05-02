# Card Capture

Extract high-quality sports card stills from local video files.

## Installation

```bash
pip install -e .
```

To use the ML-based card detection features, also install the optional model dependencies:

```bash
pip install -e ".[model]"
```

## Quick Start

### Process a Video

Process a local video with the v2.1 stage1/stage2 pipeline:

```bash
card-capture process ~/path/to/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --detector docaligner \
  --reader-backend auto \
  --queue-size 64 \
  --inference-batch-size 16 \
  --corner-confidence 0.5
```

The command writes extracted frames under `output-dir/frames`, selected best images under `output-dir/best`, and metadata into the SQLite database.

### v2.1 Pipeline Architecture

Card Capture v2.1 uses a multiprocessing producer/consumer pipeline:

```
Stage 1 Producer (sample + triage + persist frame JPEG) -> frame_queue
Stage 2 Consumer (batched detection + confidence filtering) -> detection_queue
Parent Process (storage writes + candidate selection + best image export)
```

### v2.1 Flags

Core throughput and filtering flags:
- `--reader-backend {auto,decord,pyav}`: frame ingestion backend (`auto` picks the best available backend).
- `--queue-size N`: max queue size shared by stage1 and stage2 workers.
- `--inference-batch-size N`: consumer batch size for detector inference.
- `--corner-confidence X`: minimum accepted corner confidence (`0.0` to `1.0`).
- `--blur-threshold X`, `--variance-threshold X`, `--empty-pixel-threshold X`: stage1 triage thresholds.
- `--detection-width X`, `--device {auto,cpu,mps,cuda}`: detector inference sizing/device controls.

## Advanced Usage

### Smoke Test with Fake Detector

Use the synthetic sampler + fake detector path to validate wiring and storage quickly:

```bash
card-capture process ~/path/to/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --detector fake \
  --reader-backend auto \
  --queue-size 8 \
  --inference-batch-size 4 \
  --corner-confidence 0.5
```

### Review Extracted Frames

Start the local web UI:

```bash
card-capture review --db card_capture_output/cards.sqlite --port 8000
```

Then open `http://localhost:8000`.

## Development

Run tests:

```bash
pip install -e ".[test]"
pytest tests/
```

## License

See LICENSE for details.
