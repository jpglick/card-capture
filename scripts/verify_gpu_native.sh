#!/usr/bin/env bash
# Verify GPU acceleration directly on the current host — no docker required.
# Use this when you're already inside a GPU-capable container (e.g. a RunPod
# Pod started from our prod image or runpod/pytorch:devel).
#
# Usage:
#   ./scripts/verify_gpu_native.sh                  # run all checks
#   ./scripts/verify_gpu_native.sh --no-install     # don't pip install anything
#
# Exits 0 iff every check passes. Silent CPU fallback fails the throughput
# check, so this is a real verification — not just an import smoke test.

set -euo pipefail

NO_INSTALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install) NO_INSTALL=1; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "PREFLIGHT: nvidia-smi on host"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { echo "FAIL: nvidia-smi failed. GPU not accessible from this pod."; exit 1; }

if [ "$NO_INSTALL" = "0" ]; then
  hr "DEPS: ensure ffmpeg + python packages"
  command -v ffmpeg >/dev/null \
    || apt-get install -y -q --no-install-recommends ffmpeg
  # If decord is missing here, we don't auto-build (takes 10 min). Reported
  # below as a failure with a hint.
  pip install -q --no-cache-dir kornia ultralytics huggingface-hub av >/dev/null 2>&1 || true
  # Install card-capture so pipeline_utils is importable (no-op if already done)
  pip install -q -e ".[model,app]" --no-deps >/dev/null 2>&1 || true
fi

hr "VERIFY: GPU acceleration checks"
python3 - <<'PY'
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
decord = None
ctx = None
try:
    import decord as _d
    decord = _d
    print(f"  decord: {decord.__version__}")
    ctx = decord.gpu(0)
    print(f"  decord.gpu(0): {ctx}")
except Exception as e:
    fails.append(f"decord import/gpu(0) failed: {type(e).__name__}: {e}")
    print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"  Hint: if libavformat.so.58 missing, run:")
    print(f"    apt-get install -y libavformat-dev libavcodec-dev libavfilter-dev libavutil-dev libswscale-dev libavdevice-dev")

# 3) generate a real 1080p h264 test video so NVDEC throughput is measurable
section("3. ffmpeg generate 5s 1080p test video")
vid = "/tmp/cc_verify_5s_1080p.mp4"
subprocess.check_call([
    "ffmpeg", "-y", "-f", "lavfi",
    "-i", "testsrc=duration=5:size=1920x1080:rate=30",
    "-c:v", "h264", "-pix_fmt", "yuv420p", "-loglevel", "error", vid
])
print(f"  wrote {vid} ({os.path.getsize(vid)/1_048_576:.1f} MB)")

# 4) decord NVDEC throughput — catches silent CPU fallback that import alone won't
if decord is not None and ctx is not None:
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
    if fps < 600:
        fails.append(f"decord throughput only {fps:.0f} fps — likely silent CPU fallback")
else:
    print("\n--- 4. SKIPPED — decord not usable ---")

# 5) kornia perspective warp on CUDA (Stage 6 of the pipeline)
section("5. kornia perspective warp on CUDA")
try:
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
except Exception as e:
    fails.append(f"kornia: {type(e).__name__}: {e}")

# 6) ultralytics YOLO inference on CUDA (Stage 3)
section("6. ultralytics YOLO inference on CUDA")
try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    import numpy as np
    weights = try_to_load_from_cache("AlecKarfonta/cardcaptor-v3", "weights/cardcaptor_v3_best.pt")
    if not weights:
        print("  weights not in cache; downloading (~160 MB)…")
        weights = hf_hub_download("AlecKarfonta/cardcaptor-v3", "weights/cardcaptor_v3_best.pt")
    print(f"  weights: {weights}")
    model = YOLO(weights)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device="cuda", imgsz=640, verbose=False)  # warmup
    torch.cuda.synchronize(); t0 = time.time()
    res = model.predict(dummy, device="cuda", imgsz=640, verbose=False)
    torch.cuda.synchronize(); t_ms = (time.time() - t0) * 1000
    # Blank dummy image may produce zero detections (boxes=None), so check
    # the model's weight device directly — that's what determined where
    # inference ran.
    weights_dev = str(next(model.model.parameters()).device)
    print(f"  YOLO weights device: {weights_dev}   inference: {t_ms:.1f} ms")
    if "cuda" not in weights_dev:
        fails.append(f"YOLO model not on cuda (got {weights_dev})")
except Exception as e:
    fails.append(f"YOLO: {type(e).__name__}: {e}")

# 7) Our actual code path — pipeline_utils.decode_frames_gpu round-trip.
if decord is not None and ctx is not None:
    section("7. pipeline_utils.decode_frames_gpu round-trip")
    try:
        from card_capture.pipeline_utils import decode_frames_gpu
        n_local = len(decord.VideoReader(vid, ctx=ctx))
        out = decode_frames_gpu(vid, [0, 50, 100, n_local - 1])
        print(f"  decoded indices: {sorted(out.keys())}")
        print(f"  shapes: {[out[k].shape for k in sorted(out)]}")
    except Exception as e:
        fails.append(f"decode_frames_gpu: {type(e).__name__}: {e}")

# 8) GPU was actually busy
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

echo
echo "VERIFIED: GPU acceleration works on this pod."
