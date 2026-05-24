# RunPod Cloud Deployment

How card-capture runs on RunPod, what was hard to get right, current performance
measurements, and the queue of optimizations worth measuring next.

This is a living document. Add to it after every meaningful test or change.

---

## 1. Architecture overview

```
User uploads .MOV
    │
    ▼
Cloudflare R2 (object store)             ← runs/<run_id>/input.mov
    │
    ▼
RunPod Serverless endpoint               ← gpu_ids=ADA_24, workers_min=0, workers_max=3
    │
    ▼ container start
ghcr.io/jpglick/card-capture-cuda:latest ← multi-stage Dockerfile.cuda
    │
    ▼ ENTRYPOINT
docker/start.sh                          ← git pull, re-link editable install,
                                           detect RUNPOD_POD_ID, exec runpod_handler
    │
    ▼
app/runpod_handler.py                    ← downloads video from R2, runs pipeline,
                                           collects DB diagnostics, packages results,
                                           uploads tarball back to R2
    │
    ▼
app/worker_core.py                       ← spawns Metaflow subprocess for the
                                           CardCaptureFlow pipeline
    │
    ▼
pipeline/card_capture_flow.py            ← 10-stage GPU pipeline (see CLAUDE.md)
```

**Provider:** RunPod Serverless.
**GPU pool:** `ADA_24` (RTX 4090, 24 GB VRAM).
**Workers:** 0–3 (idle scale-to-zero after 30s).
**Storage:** Cloudflare R2 (S3-compatible) for both inputs and outputs.
**Image registry:** `ghcr.io/jpglick/card-capture-cuda:latest`, pinned to digest
in the RunPod template via `scripts/runpod_setup.py --create`.

---

## 2. The image: rules learned the hard way

The current `Dockerfile.cuda` is the survivor of a long debugging chain. Each
rule below is non-obvious; breaking any one of them reintroduces a specific
failure mode. Don't change these without re-running `scripts/verify_gpu_native.sh`.

### 2.1 Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

- **Why runpod, not pytorch/pytorch:** RunPod's base uses **system Python**
  (no conda). conda Python puts `libstdc++.so.6` at `/opt/conda/lib`, which
  becomes `libdecord.so`'s `RPATH` — and that libstdc++ predictably lacked
  `GLIBCXX_3.4.30`. Every fix attempt on `pytorch/pytorch` (cp, COPY, LD_PRELOAD,
  conda install libstdcxx-ng) eventually failed for one reason or another.
  System Python sidesteps the entire problem.
- **Why devel (not runtime):** RunPod publishes only the devel variant for
  modern PyTorch — there is no runtime variant to switch to (checked: 1
  runtime tag across 437 tags, and it's old).
- **Why CUDA 12.4 / Python 3.11:** Matches the base. Our pyproject.toml's
  declared deps all support both.

### 2.2 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`

Without `video`, nvidia-container-toolkit doesn't mount `libnvcuvid.so.1`,
`/dev/nvidia-modeset`, or `/dev/dri/*`. decord's NVDEC layer then fails with
`CUDA error 100: no CUDA-capable device is detected` — even though
`torch.cuda.is_available()` returns True (PyTorch only needs `compute`).

### 2.3 Build-time libnvcuvid stub, deleted after build

The base ships no `libnvcuvid` stub for cmake to find. We create a minimal
one (`SONAME=libnvcuvid.so.1`) at `/usr/local/cuda/lib64/stubs/libnvcuvid.so`,
build decord against it, then **delete it in the same RUN**. If the stub
survived to runtime it would shadow RunPod's driver-mounted real lib in
ld.so resolution.

### 2.4 `patchelf --add-needed libnvcuvid.so.1` on `libdecord.so`

Our stub exports a single symbol (`nvcuvid_stub`) that decord doesn't call.
Modern gcc/ld default to `--as-needed`, which means **the linker dropped
the `libnvcuvid` dependency entirely** because libdecord wasn't using any of
its exports. The compiled `libdecord.so` had no `DT_NEEDED` entry for
`libnvcuvid.so.1` — so at import time, ld.so never tries to load the real
driver-mounted lib, and `cuvidDestroyDecoder` references stay undefined.

`patchelf --add-needed libnvcuvid.so.1` rewrites the DT_NEEDED list on the
installed `libdecord.so` to force ld.so to resolve it.

### 2.5 `start.sh` passes CMD args through

```bash
if [ "$#" -gt 0 ]; then exec "$@"; fi
```

Without this, our ENTRYPOINT unconditionally starts the worker, so creating
a RunPod Pod with `sleep infinity` as the start command would thrash —
worker exits ("no test_input.json"), container restarts, repeat. The
pass-through enables interactive Pods for debugging.

---

## 3. Verification

### 3.1 `scripts/verify_gpu_native.sh` — runs in any GPU container

Use when iterating in a RunPod Pod (no docker required). Runs 8 checks:
torch CUDA visibility, decord NVDEC import + gpu(0), NVDEC throughput on a
generated 1080p video, kornia perspective warp on a CUDA tensor, ultralytics
YOLO inference (model weight device check), `pipeline_utils.decode_frames_gpu`
round-trip, and nvidia-smi snapshot.

**The throughput check is the important one.** Import-only smoke tests don't
catch silent CPU fallback; the 600 fps threshold does.

### 3.2 `scripts/verify_gpu_image.sh` — runs on bare-metal with Docker

Same checks, but builds the actual `Dockerfile.cuda` first. Use on a bare-metal
GPU host with docker + nvidia-container-toolkit to verify the Dockerfile
itself, not just the host environment.

### 3.3 `scripts/runpod_pod_debug.sh` — pod-side decord/libnvcuvid debug

One-paste recipe that dumps every relevant fact about libnvcuvid resolution:
which files exist, which one ld.so picks, which have the cuvidDestroyDecoder
symbol. Use when decord fails with an undefined symbol error.

---

## 4. Performance measurements

### 4.1 NVDEC decode (decord.gpu(0), warm-only, single batch)

Measured on an RTX 4090 (24 GB) RunPod Pod with the current image:

| Resolution | Source | Throughput | Note |
|---|---|---:|---|
| 1080p (1920×1080) | ffmpeg `testsrc`, 30 fps, 30s | **434 fps** warm | decoder util 98% during decode, 0% during host copy |
| 4K landscape (3840×2160) | ffmpeg `testsrc`, 30 fps, 15s | **114 fps** warm | |
| 4K portrait (2160×3840) | ffmpeg `testsrc`, 30 fps, 15s | **110 fps** warm | matches production input orientation |

GPU/CPU ratio (decord on GPU vs decord on CPU same file, same script):
**3.13×** at 1080p. NVDEC is genuinely engaged.

**Bottleneck:** roughly half of wall time is `.asnumpy()` — copying decoded
frame buffers from GPU memory to host across PCIe. nvidia-smi during the
decode shows two phases:

```
gpu=12% decoder=98%   ← NVDEC decode phase (~1 GB/s of decoded video)
gpu=61% decoder=0%    ← post-decode copy + colorspace conversion
```

For a typical 2-minute 4K card capture video (~3600 frames at 30 fps), end-to-end
decode time at the measured ~110 fps would be ~33 seconds. Sparse-index decode
(typical for the refine step) adds NVDEC seek overhead and is slower per frame.

### 4.2 Stage timings (from one failed serverless run, decode_frames_gpu broke after)

```
detect    82.1s   ← Stages 1-3 (sampler + triage + YOLO-OBB)
novelty   27.1s   ← Stage 4 (background novelty gate)
track      2.4s   ← Stage 5
start      0.6s
```

These need re-measurement on a successful run. With the patchelf fix
landed, the next end-to-end serverless run will print
`[diag] detect_telemetry: {...}` (added in commit 77c7717a) with:
`yolo_device`, `yolo_frames`, `yolo_batches`, `yolo_elapsed_s`, `triage_pass_rate`.

Add the actual numbers to this section when available.

### 4.3 Cold-start measurements

| Phase | Observed | Source |
|---|---:|---|
| Job queue → worker assigned | 12.6s | `delayTime` from RunPod API on one successful container start |
| Container fitness checks | 1.8s | RunPod's pre-handler probes |
| R2 download (218 MB video) | 2.7s | `[diag] R2 download` log line |
| GPU compute benchmark (matmul) | 19 ms | RunPod's CUDA fitness check |

Image transfer + extract time was the user's main reported pain point.
The runpod/pytorch base alone is **7.4 GB compressed** (decompressed ~15 GB);
our added layers are ~3 GB. RunPod likely caches the runpod base on workers
since it's their own publication, so primarily our added layers transfer
on a cold start. No direct measurement yet — capture cold-start times once
the pipeline runs cleanly.

---

## 5. Future optimizations (ranked by expected impact)

Each entry: what + why + estimated impact + estimated effort + status.

### 5.1 decord torch bridge — eliminate GPU→host roundtrip

**What:** `decord.bridge.set_bridge('torch')` makes `vr.get_batch(...)`
return torch CUDA tensors directly. Pass straight into kornia's
`warp_perspective` with no host copy.

**Why:** Current `decode_frames_gpu` calls `.asnumpy()` on every batch. Frames
go GPU → host → GPU again (because Stage 6 does the warp on CUDA anyway).
nvidia-smi confirms ~50% of decode wall time is post-decode copy.

**Expected impact:** ~2× throughput on decode-bound stages (refine). At 4K,
expect 110 fps → ~200 fps.

**Effort:** ~half a day. Touches `pipeline_utils.decode_frames_gpu` (return
type), `pipeline/steps/refine.py` (consume torch tensors), tests.

**Status:** not started.

### 5.2 YOLO warmup at container start

**What:** Run one dummy `model.predict()` during `start.sh` so the first
job's detect step doesn't pay JIT + cuDNN autotune cost.

**Why:** Ultralytics + cuDNN heuristics take 5–15s on first inference per
worker. The previous 82s detect was likely cold-start dominated.

**Expected impact:** Cuts first-job-per-worker detect time by ~10s. No effect
on subsequent jobs on the same warm worker.

**Effort:** ~1 hour. Add to `start.sh` or as a preamble in `runpod_handler.handler()`.

**Status:** not started.

### 5.3 Move novelty step to GPU

**What:** Rewrite `src/card_capture/presence/background_novelty.py` to do
the per-detection comparison against the workspace baseline on the GPU
(torch tensors instead of numpy).

**Why:** 27s for novelty on a typical video is consistent with per-detection
numpy absdiff at 50ms × 500 detections. A vectorized GPU implementation
should drop this 10×.

**Expected impact:** Stage 4 drops from ~27s to ~3s.

**Effort:** 1 day. Replace numpy with torch; verify no semantic drift.

**Status:** hypothesis only — need `detect_telemetry` to confirm CPU-boundness.

### 5.4 Slim the image

**What:** Reduce image size to speed cold-start image extraction. Options:
- **Network-mount the pre-baked models** (HuggingFace cache + torch.hub cache)
  to a RunPod network volume instead of baking into the image (~500 MB off).
- **Pre-build the decord wheel** to a GitHub release artifact, install via
  URL instead of compiling in the image. Removes `cmake`, `build-essential`,
  `libav*-dev` from the image entirely (~1 GB off).

**Why:** Image pull + extract is on every cold start. 1.5 GB savings = 5–15s
faster cold start at typical RunPod network speeds.

**Expected impact:** ~15% smaller image, ~5–15s faster cold start.

**Effort:** Network volume — 2 hours. Pre-built wheel — half a day + CI setup.

**Status:** tried apt-purge inline (broke `libavformat.so.58`); reverted.

### 5.5 Batch & stride tuning

**What:** `cuda_batch_size` and `cuda_stride` knobs are currently default
(`32` and `2`). Once telemetry is available, try larger batches for YOLO
and finer/coarser strides to find the throughput sweet spot.

**Why:** Larger YOLO batches improve GPU utilization (current detect timings
suggest the model isn't saturating the 4090). Stride changes affect how
many frames hit detect/novelty.

**Expected impact:** Unknown without telemetry. Likely 1.2–2× on detect.

**Effort:** 1 hour to instrument and sweep.

**Status:** waiting on `detect_telemetry` from a successful run.

### 5.6 Persistent warm workers (vs scale-to-zero)

**What:** Set `workers_min=1` on the RunPod endpoint instead of `0`. Keeps
one worker hot at all times — no cold start for the first job in an idle
window.

**Why:** Cold start is currently ~15s (queue + container + fitness +
warmup). For latency-sensitive workloads this matters.

**Trade-off:** Pay for 1 idle worker continuously vs. cold-start penalty
per first-after-idle job. At RunPod's RTX 4090 prices (~$0.40/hr) this is
~$300/month for one always-on worker.

**Expected impact:** Eliminates 12–15s cold start on first job after idle.

**Effort:** One line in `scripts/runpod_setup.py`.

**Status:** not started; depends on workload cadence.

---

## 6. Open questions (need data to answer)

- **Why is the detect step slow on real video?** 82s in the last failed run
  was much longer than the ~5-10s a 4090 should need for YOLOv8-OBB on
  ~500-1000 frames. Suspect cold-start kernel compile (one-time per worker)
  OR YOLO running at batch=1. `detect_telemetry.yolo_batches` /
  `detect_telemetry.yolo_frames` ratio answers this.
- **Is the novelty step GPU- or CPU-bound?** Suspected CPU per §5.3.
  Add timing inside `background_novelty.py` to confirm.
- **What's the actual cold-start image transfer time?** Captured from
  RunPod's internal logs (not yet exposed in our handler). Could time
  ourselves via `[diag]` at the very top of `runpod_handler.handler()`.
- **Real-world NVDEC fps on actual card capture footage?** `testsrc` is
  unusually compressible. Real 4K phone-camera h264 at higher bitrate may
  decode slower.

---

## 7. Operational runbook

### Build + push image

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile.cuda \
  -t ghcr.io/jpglick/card-capture-cuda:latest \
  --push .
```

### Update RunPod serverless template to new image digest

```bash
docker pull ghcr.io/jpglick/card-capture-cuda:latest
python3 scripts/runpod_setup.py --create
```

`runpod_setup.py` resolves the local image's digest with `docker inspect`
and pins the RunPod template at `<image>@sha256:…` so RunPod doesn't keep
pulling a stale `:latest`.

### Spin up an interactive dev Pod

In RunPod console → Pods → Deploy:
- **Image:** `ghcr.io/jpglick/card-capture-cuda:latest`
- **GPU:** RTX 4090 (match the serverless pool)
- **Container Start Command:** `sleep infinity` (works because of `start.sh`
  passthrough, §2.5)
- **Volume:** 20 GB

Then web terminal → `cd /workspace/card-capture` → `./scripts/verify_gpu_native.sh`.

**Terminate it when done.** Pods bill hourly even when idle.

### Fetch a failed serverless job's full error

Programmatic (uses creds from `card_capture_config.json`):

```python
import json, httpx
from pathlib import Path
cfg = json.loads(Path('card_capture_config.json').read_text())
hdr = {'Authorization': f'Bearer {cfg["runpod_api_key"]}'}
ep = cfg['runpod_endpoint_id']
# list recent
r = httpx.get(f'https://api.runpod.ai/v2/{ep}/requests', headers=hdr).json()
job_id = r['requests'][0]['id']
# pull error
r = httpx.get(f'https://api.runpod.ai/v2/{ep}/status/{job_id}', headers=hdr).json()
err = json.loads(r['error'])['error_message']
print(err.encode().decode('unicode_escape'))
```

The runpod_handler also writes structured diagnostics on failure (commit
77c7717a): `_collect_db_diagnostics` runs even when the pipeline raises,
so `pipeline_runs.detect_telemetry_json` reaches the error response.

---

## 8. Performance log

A running ledger of measurements as we iterate. Append rows with date,
image commit, and observations.

| Date | Image commit | What was measured | Result | Note |
|---|---|---|---|---|
| 2026-05-23 | (pre-saga) | detect/novelty/track on failed serverless run | 82.1s / 27.1s / 2.4s | Pipeline failed at refine; numbers may be cold-start inflated |
| 2026-05-23 | c7a73c8a | NVDEC 1080p throughput (warm) | 434 fps | Pod from current image; ratio 3.13× vs CPU; decoder util peaked 98% |
| 2026-05-23 | c7a73c8a | NVDEC 4K landscape throughput (warm) | 114 fps | |
| 2026-05-23 | c7a73c8a | NVDEC 4K portrait throughput (warm) | 110 fps | Matches production input |
| 2026-05-24 | 69166f68 | First successful end-to-end serverless run | pipeline 114.4s, 18 cards | detect 31.6s, novelty ~0.5s, track 2.2s, refine 67.3s |
| 2026-05-24 | 69166f68 | Cold start (delay before execution) | 18.7s | Image already warm on worker |
| 2026-05-24 | 69166f68 | Refine op breakdown | kornia_warp 2.0s (↓10×), laplacian 34.7s | laplacian timer included dead _compress_array work — see fix below |
| 2026-05-24 | 69166f68 | GPU utilization | mean 15.6%, peak 100% | Up from 8.5%; still starved during refine |
| 2026-05-24 | 69166f68 | VRAM peak | 22.2 GB | Stage tagging didn't fire (all "unknown"); still need per-stage breakdown |

---

## 9. Reference: commits that built this setup

The bottom of the saga, for archaeology:

| Commit | Change |
|---|---|
| `8dfcdee8` | Switch base to `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (kills the entire libstdc++/conda RPATH class of bugs) |
| `ae395ac0` | `ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` (mounts NVDEC lib + DRI nodes) |
| `474494a3` | Build-time `libnvcuvid` stub, deleted at end of RUN (cmake satisfied; no runtime shadowing) |
| `cc14f33f` | `patchelf --add-needed libnvcuvid.so.1` on libdecord.so (forces ld.so to load real driver lib) |
| `c7a73c8a` | `decode_frames_gpu` NDArray subscripting fix (decord 0.6 API) |
| `0f28ed13` | `start.sh` passes CMD args through (enables `sleep infinity` Pods) |
| `77c7717a` | Surface `detect_telemetry` even on failure (so the next failure is actionable) |
| `39bf224b` / `14e2b102` | `verify_gpu_image.sh` / `verify_gpu_native.sh` (catch silent CPU fallback) |
| `d71b0243` | `runpod_pod_debug.sh` (one-paste libnvcuvid resolution debug) |
