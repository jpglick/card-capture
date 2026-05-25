#!/bin/bash
# Container entrypoint — handles both vast.ai and RunPod.
# RunPod injects RUNPOD_POD_ID; vast.ai does not.

set -e

# ALWAYS sync first, regardless of CMD args. The serverless template sets
# docker_start_cmd="python3 -m app.runpod_handler" which arrives as args to
# this script; if we exec them before syncing, the worker silently runs the
# baked image's code forever (every Python fix invisible to serverless until
# the next image rebuild). Sync must precede any args branching.
echo "[start.sh] Syncing /workspace/card-capture to origin/main…"
cd /workspace/card-capture
if git fetch origin main --depth=1 -q; then
    git reset --hard origin/main -q
    echo "[start.sh] On commit: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
else
    echo "[start.sh] ERROR: git fetch failed — running baked code at $(git rev-parse --short HEAD)" >&2
fi

# Re-link the editable install so Python picks up git-pulled source changes.
# --no-deps avoids reinstalling torch/torchvision. Errors are shown, not hidden.
echo "[start.sh] Re-linking editable install…"
pip install -e '.[app]' --no-deps -q || echo "[start.sh] WARNING: editable install failed"

# CUDA warmup — pay JIT compile + cuDNN autotune costs ONCE at container start
# so the per-step metaflow subprocesses don't each take an 80s+ first-inference
# hit. PTX kernels go to $CUDA_CACHE_PATH (persists across processes within
# this container); cuDNN heuristics still re-autotune per process but with
# the kernels already compiled the cost drops from ~80s to ~5-10s per step.
#
# CRITICAL: this MUST be time-bounded and MUST NOT abort start.sh on failure.
# Earlier symptom: workers stuck "running" with jobs in queue forever because
# a hung op (ffmpeg/decord/YOLO model load) deadlocked the warmup and start.sh
# never reached `exec runpod_handler`. timeout 120 + `|| true` guarantee that
# a hang/failure costs at most 2 minutes of cold-start, never the whole worker.
if [ -n "$RUNPOD_POD_ID" ] || nvidia-smi >/dev/null 2>&1; then
  echo "[start.sh] CUDA warmup (one-time JIT + cuDNN autotune, time-bounded)…"
  mkdir -p "${CUDA_CACHE_PATH:-/root/.nv/ComputeCache}"
  cat > /tmp/cc_warmup.py <<'PY'
import time, os, subprocess
t_all = time.time()

t = time.time(); import torch
_ = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
print(f"torch cuda init: {(time.time()-t)*1000:.0f}ms", flush=True)

t = time.time()
from ultralytics import YOLO
from huggingface_hub import try_to_load_from_cache
import numpy as np
weights = try_to_load_from_cache("AlecKarfonta/cardcaptor-v3", "weights/cardcaptor_v3_best.pt")
engine = os.path.splitext(weights)[0] + ".engine"
try:
    if not os.path.exists(engine):
        YOLO(weights).export(format="engine", half=True, dynamic=True, imgsz=640, device=0, verbose=False)
    m = YOLO(engine)
    print(f"trt engine ready: {(time.time()-t)*1000:.0f}ms", flush=True)
except Exception as e:
    print(f"trt export failed ({e}); warming .pt fp16", flush=True)
    m = YOLO(weights); m.half()
m.predict(np.zeros((640,640,3), dtype=np.uint8), device="cuda", imgsz=640, half=True, verbose=False)
print(f"yolo warmup: {(time.time()-t)*1000:.0f}ms", flush=True)

t = time.time()
import decord
decord.bridge.set_bridge("torch")
vid = "/tmp/cc_warmup.mp4"
subprocess.check_call([
    "ffmpeg", "-y", "-f", "lavfi",
    "-i", "testsrc=duration=2:size=640x360:rate=30",
    "-c:v", "h264", "-pix_fmt", "yuv420p", "-loglevel", "error", vid],
    timeout=30)
vr = decord.VideoReader(vid, ctx=decord.gpu(0))
_ = vr.get_batch([0, 10, 20]).cpu().numpy()
os.remove(vid)
print(f"decord nvdec warmup: {(time.time()-t)*1000:.0f}ms", flush=True)

t = time.time()
import kornia.geometry.transform as KT
img = torch.rand(1, 3, 1050, 750, device="cuda")
src = torch.tensor([[[0,0],[749,0],[749,1049],[0,1049]]], dtype=torch.float32, device="cuda")
dst = torch.tensor([[[10,10],[739,10],[739,1039],[10,1039]]], dtype=torch.float32, device="cuda")
M = KT.get_perspective_transform(src, dst)
_ = KT.warp_perspective(img, M, (1050, 750))
torch.cuda.synchronize()
print(f"kornia warmup: {(time.time()-t)*1000:.0f}ms", flush=True)

print(f"TOTAL: {(time.time()-t_all)*1000:.0f}ms", flush=True)
PY
  # Subshell + `|| true` keeps start.sh going even if warmup times out / errors.
  # timeout returns 124 on hard timeout; any non-zero just means "no warmup benefit
  # for this worker, but it can still serve jobs".
  ( timeout 120 python3 /tmp/cc_warmup.py 2>&1 | sed 's/^/[warmup] /' ) \
    || echo "[start.sh] WARNING: CUDA warmup did not complete in 120s — continuing without it"
  rm -f /tmp/cc_warmup.py
fi

# CMD passthrough goes AFTER the sync. Dev Pods created with
# "Container Start Command: sleep infinity" (or `docker run … <image> bash`)
# still work — they just get a synced workspace first. Serverless workers
# (CMD = "python3 -m app.runpod_handler" from runpod_setup.py) also hit this
# branch, since the serverless logic below would do the same exec anyway.
if [ "$#" -gt 0 ]; then
    echo "[start.sh] CMD args present — exec'ing: $*"
    exec "$@"
fi

if [ -n "$RUNPOD_POD_ID" ]; then
    # ── RunPod serverless ──────────────────────────────────────────────────
    echo "[start.sh] RunPod detected (pod $RUNPOD_POD_ID) — ensuring runpod SDK…"
    pip install "runpod>=1.7.0" -q 2>&1 | tail -2
    echo "[start.sh] Starting runpod_handler…"
    exec python3 -m app.runpod_handler
else
    # ── vast.ai ────────────────────────────────────────────────────────────
    echo "[start.sh] Starting vastai_worker on port 8765…"
    nohup uvicorn app.vastai_worker:app \
        --host 0.0.0.0 \
        --port 8765 \
        --log-level info \
        > /tmp/worker.log 2>&1 &

    WORKER_PID=$!
    echo "[start.sh] Worker started (pid $WORKER_PID) — logs at /tmp/worker.log"

    if command -v sshd &>/dev/null; then
        echo "[start.sh] Starting sshd…"
        exec /usr/sbin/sshd -D
    else
        echo "[start.sh] sshd not found — sleeping to keep container alive"
        exec sleep infinity
    fi
fi
