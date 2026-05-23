#!/usr/bin/env bash
# Build the card-capture CUDA image and verify GPU acceleration end-to-end.
# Run on a machine with an NVIDIA GPU + Docker + nvidia-container-toolkit.
# Tested target: RTX 4090 (compute capability 8.9).
#
# Usage:
#   ./scripts/verify_gpu_image.sh                 # build + verify
#   ./scripts/verify_gpu_image.sh --skip-build    # verify existing :verify tag
#   ./scripts/verify_gpu_image.sh --tag foo:bar   # custom local tag
#
# Exits 0 iff every check passes. Silent CPU fallback is treated as failure
# (throughput thresholds catch it).

set -euo pipefail

TAG="card-capture-cuda:verify"
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "PREFLIGHT: docker can see the GPU"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 \
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { echo "FAIL: docker can't see GPU. Install nvidia-container-toolkit."; exit 1; }

if [ "$SKIP_BUILD" = "0" ]; then
  hr "BUILD: $TAG  (decord compiles from source — expect ~10-15 min on first run)"
  docker build -f Dockerfile.cuda -t "$TAG" .
fi

hr "VERIFY: running GPU acceleration checks inside the image"
# 'PY' is single-quoted so bash inside the container does NOT expand $ in the
# Python source. Exit code from python propagates through bash to docker.
docker run --rm --gpus all --entrypoint /bin/bash "$TAG" -c '
set -e
python3 - <<'\''PY'\''
import os, sys, time, subprocess
fails = []

def section(name):
    print(f"\n--- {name} ---", flush=True)

# 1) torch CUDA visibility
section("1. torch CUDA visibility")
import torch
print(f"  torch: {torch.__version__}   cuda: {torch.version.cuda}")
print(f"  cuda_available: {torch.cuda.is_available()}   devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    fails.append("torch.cuda not available")
else:
    print(f"  device: {torch.cuda.get_device_name(0)}")

# 2) decord NVDEC import + GPU context
section("2. decord NVDEC import + gpu(0)")
import decord
print(f"  decord: {decord.__version__}")
ctx = decord.gpu(0)
print(f"  decord.gpu(0): {ctx}")

# 3) generate a real 1080p h264 video so NVDEC throughput is measurable
section("3. ffmpeg generate 5s 1080p test video")
vid = "/tmp/cc_verify_5s_1080p.mp4"
subprocess.check_call([
    "ffmpeg", "-y", "-f", "lavfi",
    "-i", "testsrc=duration=5:size=1920x1080:rate=30",
    "-c:v", "h264", "-pix_fmt", "yuv420p", "-loglevel", "error", vid
])
print(f"  wrote {vid} ({os.path.getsize(vid)/1_048_576:.1f} MB)")

# 4) decord NVDEC throughput — silent CPU fallback shows up as low fps
section("4. decord NVDEC throughput (4090: expect >1000 fps on 1080p)")
vr = decord.VideoReader(vid, ctx=ctx)
n = len(vr)
print(f"  frames in video: {n}   avg_fps: {vr.get_avg_fps()}")
t0 = time.time()
batch = vr.get_batch(list(range(n)))
elapsed_ms = (time.time() - t0) * 1000
fps = n / (elapsed_ms / 1000) if elapsed_ms else 0.0
print(f"  decoded {n} frames in {elapsed_ms:.1f} ms  ({fps:.0f} fps)")
print(f"  output shape: {batch.asnumpy().shape}")
# CPU h264 of 1080p is ~300-500 fps on a modern core; NVDEC on Ada is ~1500+.
# 600 fps threshold leaves headroom for noisy boxes but catches CPU fallback.
if fps < 600:
    fails.append(f"decord throughput only {fps:.0f} fps — likely silent CPU fallback")

# 5) kornia perspective warp on CUDA (Stage 6 of the pipeline)
section("5. kornia perspective warp on CUDA")
import kornia.geometry.transform as KT
img = torch.rand(8, 3, 1050, 750, device="cuda")
src = torch.tensor([[[0,0],[749,0],[749,1049],[0,1049]]]*8, dtype=torch.float32, device="cuda")
dst = torch.tensor([[[20,20],[729,20],[729,1029],[20,1029]]]*8, dtype=torch.float32, device="cuda")
M = KT.get_perspective_transform(src, dst)
torch.cuda.synchronize(); t0 = time.time()
warped = KT.warp_perspective(img, M, (1050, 750))
torch.cuda.synchronize(); t_ms = (time.time() - t0) * 1000
print(f"  warped shape={tuple(warped.shape)}   device={warped.device}   {t_ms:.1f} ms (batch of 8)")
if warped.device.type != "cuda":
    fails.append("kornia warp not on cuda")

# 6) ultralytics YOLO inference on CUDA (Stage 3)
section("6. ultralytics YOLO inference on CUDA")
from ultralytics import YOLO
from huggingface_hub import try_to_load_from_cache
import numpy as np
weights = try_to_load_from_cache("AlecKarfonta/cardcaptor-v3", "weights/cardcaptor_v3_best.pt")
print(f"  weights cached at: {weights}")
if not weights:
    fails.append("YOLO weights not pre-baked in image cache")
else:
    model = YOLO(weights)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device="cuda", imgsz=640, verbose=False)  # warmup
    torch.cuda.synchronize(); t0 = time.time()
    res = model.predict(dummy, device="cuda", imgsz=640, verbose=False)
    torch.cuda.synchronize(); t_ms = (time.time() - t0) * 1000
    dev = str(res[0].boxes.data.device) if (res and res[0].boxes is not None) else "<no boxes>"
    print(f"  YOLO result device: {dev}   inference: {t_ms:.1f} ms")
    if "cuda" not in dev:
        fails.append(f"YOLO not on cuda (got {dev})")

# 7) Our actual code path — pipeline_utils.decode_frames_gpu round-trip.
#    This is what the refine step calls. If this fails the whole pipeline fails.
section("7. pipeline_utils.decode_frames_gpu round-trip")
from card_capture.pipeline_utils import decode_frames_gpu
out = decode_frames_gpu(vid, [0, 50, 100, n - 1])
print(f"  decoded indices: {sorted(out.keys())}")
print(f"  shapes: {[out[k].shape for k in sorted(out)]}")

# 8) GPU was actually used (smoke check)
section("8. nvidia-smi snapshot")
print("  " + subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader"
]).decode().strip().replace("\n", "\n  "))

if fails:
    print("\nFAIL:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("\nALL GPU ACCELERATION CHECKS PASSED")
PY
'

echo
echo "VERIFIED: $TAG is GPU-accelerated end-to-end on this machine."
