#!/bin/bash
# Container entrypoint — handles both vast.ai and RunPod.
# RunPod injects RUNPOD_POD_ID; vast.ai does not.

set -e

echo "[start.sh] Pulling latest code…"
cd /workspace/card-capture
git pull origin main -q || echo "[start.sh] WARNING: git pull failed, using baked code"

echo "[start.sh] Installing app layer…"
pip install -e '.[app]' -q 2>&1 | tail -3

if [ -n "$RUNPOD_POD_ID" ]; then
    # ── RunPod serverless ──────────────────────────────────────────────────
    # Don't re-run pip install — torch/torchvision are pinned to the base image
    # and reinstalling would pull incompatible CPU versions. Only ensure runpod SDK is present.
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
