# Card Capture

Extract high-quality sports card stills from local video files. Optimized for Apple Silicon.

## Installation

```bash
pip install -e .
```

Install the tracking (BoT-SORT) and PyAV fallback dependencies:

```bash
pip install -e ".[legacy_tracking]"
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

`decord` is installed separately from `.[legacy_tracking]` because PyPI does not publish wheels for Apple Silicon macOS, and its macOS PyPI wheels are limited to older Intel CPython builds.

Install `decord` with one of these paths:

- Linux x86_64 / Windows amd64 with a supported Python:

```bash
pip install decord
```

- Apple Silicon macOS (Recommended):

```bash
mkdir -p .tools
cd .tools
curl -L https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj
cd ..
.tools/bin/micromamba create -y -p "$PWD/.decord-env" -c conda-forge python=3.11 decord ffmpeg pip
.tools/bin/micromamba run -p "$PWD/.decord-env" pip install -e ".[legacy_tracking,model,review,test]"
```

Then run the app through that environment:

```bash
.tools/bin/micromamba run -p "$PWD/.decord-env" card-capture process ~/path/to/video.mov --reader-backend decord
```

## Local Execution & GPU Acceleration

> [!IMPORTANT]
> **GPU/MPS acceleration is NOT available within restricted CLI environments (like Gemini CLI).** 
> 
> High-resolution video processing and performance testing **MUST** be run manually in a local terminal to utilize Apple Silicon (MPS) hardware. Running inside the AI agent environment will force a CPU fallback, resulting in significantly slower processing times (~10x slower).

To run locally:
```bash
card-capture process <video_path> --output-dir out --db out/cards.sqlite
```

## Quick Start

### Process a Video

Process a local video with the v5.5 pipeline:

```bash
card-capture process ~/path/to/video.mov \
  --output-dir out \
  --db out/cards.sqlite \
  --detector docaligner \
  --reader-backend auto \
  --corner-confidence 0.5
```

The command writes extracted 750×1050 rectified stills under `out/crops` and metadata into the SQLite database.

### v5.5 Pipeline Architecture

Card Capture v5.5 uses a high-performance in-process pipeline:

```
Stage 1: Adaptive Presence Sampler
Stage 2: YOLO Corner Detection
Stage 3: Background Novelty Gate
Stage 4: Session-Aware Tracking
Stage 5: GPU Refinement (Kornia)
Stage 6: Quality Scoring + Pruning
Stage 7: Front/Back Resolution
Stage 8: Lighting-Diverse Fusion
Stage 9: Global Deduplication
Stage 10: Storage (Single-Writer DAL)
```

### v5.5 Flags

Core throughput and filtering flags:
- `--reader-backend {auto,decord,pyav}`: frame ingestion backend.
- `--corner-confidence X`: minimum accepted corner confidence (`0.0` to `1.0`).
- `--detection-width X`, `--device {auto,cpu,mps}`: detector inference sizing/device controls.

## Advanced Usage

### Review Extracted Frames

Start the local web UI:

```bash
card-capture review --db out/cards.sqlite --port 8000
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
