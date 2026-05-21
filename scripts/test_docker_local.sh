#!/usr/bin/env bash
# Run a quick sanity check against the local card-capture-cuda image.
# Tests that decord, torch, CudaSampler and decode_frames_gpu all import correctly.
# Uses CC_CUDA_ALLOW_CPU_FALLBACK=1 so it works on machines without an NVIDIA GPU.
#
# Usage:
#   ./scripts/test_docker_local.sh
#   ./scripts/test_docker_local.sh /path/to/video.mov   # also runs the full pipeline

set -e

IMAGE="ghcr.io/jpglick/card-capture-cuda:latest"
VIDEO="${1:-}"

echo "=== Import check ==="
docker run --platform linux/amd64 --rm \
  -e CC_CUDA_ALLOW_CPU_FALLBACK=1 \
  --entrypoint python3 \
  "$IMAGE" \
  -c "
import decord
print('decord:', decord.__version__)

import torch
print('torch CUDA available:', torch.cuda.is_available())

from card_capture.sampler.cuda_sampler import CudaSampler
print('CudaSampler OK')

from card_capture.pipeline_utils import decode_frames_gpu
print('decode_frames_gpu OK')

from card_capture.detectors import CardcaptorUltralyticsDetector
print('detector OK')

print()
print('All imports OK')
"

if [ -z "$VIDEO" ]; then
  echo ""
  echo "Tip: pass a video path to also run the full pipeline:"
  echo "  $0 /path/to/video.mov"
  exit 0
fi

echo ""
echo "=== Full pipeline run (CPU fallback, CUDA code paths) ==="
echo "Video: $VIDEO"

OUTPUT_DIR="/tmp/cc-local-test"
rm -rf "$OUTPUT_DIR"

docker run --platform linux/amd64 --rm \
  -e CC_CUDA_ALLOW_CPU_FALLBACK=1 \
  -e METAFLOW_USER=localtest \
  --entrypoint bash \
  -v "$VIDEO:/tmp/test_video.mov:ro" \
  -v "$OUTPUT_DIR:/tmp/cc-out" \
  "$IMAGE" \
  -c "cd /workspace/card-capture && git pull -q 2>/dev/null || true && \
      python3 -m pipeline.card_capture_flow --no-pylint run \
        --video /tmp/test_video.mov \
        --output-dir /tmp/cc-out \
        --db /tmp/cc-out/cards.sqlite \
        --config-preset balanced"

echo ""
echo "Results in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || true
