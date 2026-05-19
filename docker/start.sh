#!/bin/bash
# Container entrypoint for vast.ai GPU instances.
#
# vast.ai uses SSH as the container's main process and executes the
# user-provided "onstart" script via SSH. If no SSH key is registered
# in the vast.ai account, onstart silently fails and the worker never
# starts. This entrypoint starts the worker unconditionally, before
# handing off to sshd, so it works with or without SSH keys.

set -e

echo "[start.sh] Pulling latest code…"
cd /workspace/card-capture
git pull origin main -q || echo "[start.sh] WARNING: git pull failed, using baked code"

echo "[start.sh] Installing app layer…"
pip install -e '.[app]' -q 2>&1 | tail -3

echo "[start.sh] Starting vastai_worker on port 8765…"
nohup uvicorn app.vastai_worker:app \
    --host 0.0.0.0 \
    --port 8765 \
    --log-level info \
    > /tmp/worker.log 2>&1 &

WORKER_PID=$!
echo "[start.sh] Worker started (pid $WORKER_PID) — logs at /tmp/worker.log"

# Hand off to sshd so vast.ai can SSH in for management.
# If sshd isn't installed (shouldn't happen since base image includes it),
# fall back to sleeping indefinitely to keep the container alive.
if command -v sshd &>/dev/null; then
    echo "[start.sh] Starting sshd…"
    exec /usr/sbin/sshd -D
else
    echo "[start.sh] sshd not found — sleeping to keep container alive"
    exec sleep infinity
fi
