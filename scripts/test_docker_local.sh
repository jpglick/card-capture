#!/usr/bin/env bash
# Sanity-check the card-capture-cuda Docker image locally (no NVIDIA GPU needed).
#
# On non-NVIDIA machines, decord and GPU-only modules are skipped with a clear
# message. Everything else (torch, card_capture logic, Metaflow pipeline) is tested.
# The image rebuild with libstdcxx-ng will enable the full decord test.
#
# Usage:
#   ./scripts/test_docker_local.sh                    # import check only
#   ./scripts/test_docker_local.sh /path/to/video.mov # + full pipeline run

set -e

IMAGE="ghcr.io/jpglick/card-capture-cuda:latest"
VIDEO="${1:-}"
INNER_PY="/tmp/cc_docker_test_$$.py"
PIPELINE_PY="/tmp/cc_docker_pipeline_$$.py"
trap 'rm -f "$INNER_PY" "$PIPELINE_PY"' EXIT

python3 - > "$INNER_PY" << 'WRITE_SCRIPT'
script = r'''
import sys

def try_import(name, from_module=None, as_name=None):
    label = as_name or (from_module + "." + name if from_module else name)
    try:
        if from_module:
            mod = __import__(from_module, fromlist=[name])
            getattr(mod, name)
        else:
            __import__(name)
        print(f"{label}: OK")
        return True
    except (OSError, ImportError, Exception) as e:
        msg = str(e)
        if any(k in msg for k in ("libcuda", "libnvidia", "GLIBCXX", "No CUDA")):
            print(f"{label}: SKIPPED (GPU library unavailable on non-NVIDIA host)")
        else:
            print(f"{label}: FAILED — {e}")
        return False

print("--- GPU libraries (may be skipped on non-NVIDIA host) ---")
try_import("decord")
try_import("CudaSampler", "card_capture.sampler.cuda_sampler")

print()
print("--- Core imports ---")
try_import("torch")
try_import("decode_frames_gpu", "card_capture.pipeline_utils")
try_import("CardcaptorUltralyticsDetector", "card_capture.detectors")
try_import("KorniaNormalizer", "card_capture.gpu_refinement")
try_import("ResultImporter", "app.services.result_importer")

import torch
if not torch.cuda.is_available():
    print(f"\ntorch.cuda.is_available(): False (expected on non-NVIDIA host)")

print("\nDone.")
'''
print(script)
WRITE_SCRIPT

echo "=== Import check ==="
docker run --platform linux/amd64 --rm \
  -e CC_CUDA_ALLOW_CPU_FALLBACK=1 \
  -e PYTHONPATH=/workspace/card-capture \
  -w /workspace/card-capture \
  --entrypoint python3 \
  -v "$INNER_PY:/tmp/inner.py:ro" \
  "$IMAGE" /tmp/inner.py

if [ -z "$VIDEO" ]; then
  echo ""; echo "Pass a video to also run the full pipeline:"; echo "  $0 /path/to/video.mov"
  exit 0
fi

echo ""; echo "=== Full pipeline (CPU fallback, CUDA code paths) ==="; echo "Video: $VIDEO"

OUTPUT_DIR="/tmp/cc-local-test"
rm -rf "$OUTPUT_DIR"; mkdir -p "$OUTPUT_DIR"

python3 - > "$PIPELINE_PY" << 'WRITE_PIPELINE'
script = r'''
import subprocess, sys, os
os.chdir("/workspace/card-capture")
subprocess.run(["git","pull","-q"], capture_output=True)
ret = subprocess.run([sys.executable,"-m","pipeline.card_capture_flow","--no-pylint","run",
    "--video","/tmp/test_video.mov","--output-dir","/tmp/cc-out",
    "--db","/tmp/cc-out/cards.sqlite","--config-preset","balanced"])
sys.exit(ret.returncode)
'''
print(script)
WRITE_PIPELINE

docker run --platform linux/amd64 --rm \
  -e CC_CUDA_ALLOW_CPU_FALLBACK=1 \
  -e METAFLOW_USER=localtest \
  --entrypoint python3 \
  -v "$PIPELINE_PY:/tmp/pipeline.py:ro" \
  -v "$VIDEO:/tmp/test_video.mov:ro" \
  -v "$OUTPUT_DIR:/tmp/cc-out" \
  "$IMAGE" /tmp/pipeline.py

echo ""; echo "Results in: $OUTPUT_DIR"; ls -lh "$OUTPUT_DIR/" 2>/dev/null || true
