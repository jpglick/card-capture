# Card Capture

Extract high-quality sports card stills from local video files.

## Installation

```bash
pip install -e .
```

Install the v2.1 pipeline runtime dependencies:

```bash
pip install -e ".[pipeline_v21]"
```

To use the ML-based card detection features, also install the optional model dependencies:

```bash
pip install -e ".[model]"
```

To use the review UI:

```bash
pip install -e ".[review]"
```

### Decord Backend

`--reader-backend auto` now prefers `decord` when it is importable and falls back to `pyav` otherwise.

On Apple Silicon, if PyTorch reports that MPS is unavailable at runtime, the CLI now stops and asks whether to continue on CPU. That keeps GPU fallback explicit instead of silently degrading performance.

`decord` is installed separately from `.[pipeline_v21]` because PyPI does not publish wheels for Apple Silicon macOS, and its macOS PyPI wheels are limited to older Intel CPython builds.

Install `decord` with one of these paths:

- Linux x86_64 / Windows amd64 with a supported Python:

```bash
pip install decord
```

- Apple Silicon macOS:

```bash
mkdir -p .tools
cd .tools
curl -L https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj
cd ..
.tools/bin/micromamba create -y -p "$PWD/.decord-env" -c conda-forge python=3.11 decord ffmpeg pip
.tools/bin/micromamba run -p "$PWD/.decord-env" pip install -e ".[pipeline_v21,model,review,test]"
```

Then run the app through that environment:

```bash
.tools/bin/micromamba run -p "$PWD/.decord-env" card-capture process ~/path/to/video.mov --reader-backend decord
```

## Local Execution & GPU Acceleration

> [!IMPORTANT]
> **GPU/MPS acceleration is NOT available within restricted CLI environments (like Gemini CLI).** 
> 
> High-resolution video processing and performance testing **MUST** be run manually in a local terminal to utilize Apple Silicon (MPS) or NVIDIA (CUDA) hardware. Running inside the AI agent environment will force a CPU fallback, resulting in significantly slower processing times (~10x slower).

To run locally:
```bash
.venv/bin/python -m card_capture.cli process <video_path> --output-dir card_capture_output --db card_capture_output/cards.sqlite --config card_capture_config.json
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
- `--reader-backend {auto,decord,pyav}`: frame ingestion backend (`auto` prefers `decord`, otherwise falls back to `pyav`).
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
