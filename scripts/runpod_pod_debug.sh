#!/usr/bin/env bash
# Run inside a RunPod Pod that was started FROM our prod image
# (ghcr.io/jpglick/card-capture-cuda:latest). Answers: "is the cuvidDestroyDecoder
# symbol resolvable on this RunPod environment?" If yes, the fix is a one-line
# stub deletion in Dockerfile.cuda. If no, we must drop decord and use PyAV.
#
# Paste this whole script into an SSH session on the pod (no edits needed).

set +e  # diagnostics; do not exit on first failure

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "PROD IMAGE? (we expect /workspace/card-capture to exist)"
ls -la /workspace 2>/dev/null | head -5
python3 -c "import sys; print('python:', sys.executable, sys.version)"

hr "NVIDIA DRIVER MOUNT — does /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 exist?"
ls -la /usr/lib/x86_64-linux-gnu/libnvcuvid* 2>/dev/null || echo "  NOT FOUND — RunPod did not mount video codec lib"

hr "CUDA STUB — does the build-time stub still ship?"
ls -la /usr/local/cuda/lib64/stubs/libnvcuvid* 2>/dev/null || echo "  no stub present"

hr "REAL DRIVER LIB SYMBOL CHECK — does cuvidDestroyDecoder exist in the mounted lib?"
if [ -f /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 ]; then
  nm -D /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 2>&1 | grep -c cuvidDestroyDecoder \
    | xargs -I{} echo "  cuvidDestroyDecoder symbol count in driver lib: {}"
else
  echo "  cannot check — driver lib not present"
fi

hr "STUB SYMBOL CHECK — does the stub define cuvidDestroyDecoder? (expect: 0)"
if [ -f /usr/local/cuda/lib64/stubs/libnvcuvid.so ]; then
  nm -D /usr/local/cuda/lib64/stubs/libnvcuvid.so 2>&1 | grep -c cuvidDestroyDecoder \
    | xargs -I{} echo "  cuvidDestroyDecoder in stub: {} (any value > 0 is a bug)"
fi

hr "LDD libdecord.so — which libnvcuvid is resolved?"
LIBDECORD=/opt/conda/lib/python3.10/site-packages/decord/libdecord.so
ldd "$LIBDECORD" 2>&1 | grep -iE 'nvcuvid|not found|stub'

hr "BASELINE: import decord (current state — expect the prod error)"
python3 -c "import decord; print('decord OK, version:', decord.__version__)" 2>&1 | head -3

hr "FIX ATTEMPT 1: remove CUDA stub so the real driver lib wins ld.so resolution"
rm -fv /usr/local/cuda/lib64/stubs/libnvcuvid.so /usr/local/cuda/lib64/stubs/libnvcuvid.so.1 2>/dev/null
ldconfig
python3 -c "import decord; print('decord OK after stub removal, version:', decord.__version__)" 2>&1 | head -3

hr "FIX ATTEMPT 2 (only if 1 still failed): force LD_LIBRARY_PATH to driver mount first"
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  python3 -c "import decord; print('decord OK with LD_LIBRARY_PATH override, version:', decord.__version__)" 2>&1 | head -3

hr "VERDICT"
echo "  - If 'decord OK after stub removal' printed: the fix is one line in Dockerfile.cuda (rm the stub in Stage 2)."
echo "  - If only 'with LD_LIBRARY_PATH override' worked: add LD_LIBRARY_PATH to start.sh."
echo "  - If both failed AND 'cuvidDestroyDecoder symbol count in driver lib' was 0: RunPod's driver mount is too old or doesn't include nvcuvid — we must replace decord (PyAV or torchcodec)."
echo "  - If driver lib was 'NOT FOUND' at all: ditto — replace decord."
