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
