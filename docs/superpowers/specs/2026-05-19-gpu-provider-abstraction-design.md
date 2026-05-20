# GPU Provider Abstraction Design

**Date:** 2026-05-19  
**Status:** Approved  
**Scope:** Add Beam and RunPod as switchable GPU backends alongside vast.ai

---

## Problem

vast.ai has unreliable SSH proxy infrastructure (10+ minute wait times, random connection failures). The pipeline execution logic is provider-agnostic but is tightly coupled to vast.ai's instance lifecycle and SSH tunnel model. Adding alternative providers requires untangling this coupling.

---

## Goals

- Swap GPU provider via a single config field (`pipeline_backend`)
- Reuse pipeline execution logic across all providers
- Support: vast.ai (existing), Beam (new), RunPod (new)
- No external storage dependencies — use provider-native file transfer

---

## Architecture Overview

```
app/api/videos.py
    └── selects runner via pipeline_backend config
            ├── "mps"     → PipelineRunner   (local, existing)
            ├── "vastai"  → VastAIRunner      (existing, unchanged)
            ├── "beam"    → BeamRunner         (new)
            └── "runpod"  → RunPodRunner       (new)

app/services/gpu_runner.py       GPURunner Protocol (3 methods)
app/services/vast_runner.py      VastAIRunner (unchanged)
app/services/beam_runner.py      BeamRunner (new)
app/services/runpod_runner.py    RunPodRunner (new)

app/worker_core.py               Shared pipeline execution logic (extracted)
app/vastai_worker.py             Thin wrapper around worker_core (thinned)
app/beam_handler.py              Beam endpoint handler (deployed to Beam)
app/runpod_handler.py            RunPod serverless handler (deployed to RunPod)
```

---

## Section 1: GPURunner Protocol

**File:** `app/services/gpu_runner.py`

```python
from typing import Protocol

class GPURunner(Protocol):
    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **kw,
    ) -> None: ...

    async def run_batch_async(self, jobs: list[dict]) -> None: ...

    async def destroy_instance(self) -> None: ...
```

Using `Protocol` (structural subtyping) means `VastAIRunner` already satisfies the interface with no changes to its class declaration.

**Runner selection in `app/api/videos.py`:**

```python
backend = cfg.get("pipeline_backend", "mps")
if backend == "vastai":
    runner = VastAIRunner(...)
elif backend == "beam":
    runner = BeamRunner(...)
elif backend == "runpod":
    runner = RunPodRunner(...)
else:
    runner = PipelineRunner(...)
```

---

## Section 2: Worker Core Extraction

**File:** `app/worker_core.py` (new)

Extracted from `app/vastai_worker.py`:

| Symbol | Purpose |
|---|---|
| `CUDA_CONFIG_OVERRIDES` | Dict of CUDA-specific config keys |
| `apply_cuda_config() -> dict` | Writes CUDA overrides; returns originals |
| `restore_config(original: dict)` | Restores overwritten config keys |
| `run_pipeline(job_id, video_path, config_preset, output_dir) -> Path` | Runs Metaflow pipeline subprocess; returns db_path |
| `package_results(job_id, output_dir, db_path) -> bytes` | Builds tarball; **returns bytes** (not written to disk) |

`package_results` returns bytes so each provider handler can do what it needs (upload to storage, return inline, write to disk). No behavior change for vast.ai — it just writes the bytes to disk as before.

**`app/vastai_worker.py`** is thinned: `_run_pipeline` and `_package_results` are replaced with calls to `worker_core`. The FastAPI app, endpoint definitions, job queue, and shutdown logic are unchanged.

---

## Section 3: File Transfer

Each provider uses its own native storage. No external S3/R2 dependency.

### Beam — Beam Volumes

- **Input:** Runner uploads video to a Beam Volume via Beam's upload API. Volume path is passed to the endpoint as a string argument.
- **Output:** Handler writes tarball bytes to a results path on the Volume. Runner downloads via Beam's download API.
- **Cleanup:** Runner deletes both paths from the Volume after import.

### RunPod — RunPod S3 API

RunPod provides built-in S3-compatible object storage (`docs.runpod.io/storage/s3-api`). Standard boto3 with RunPod's endpoint URL and bucket credentials.

- **Input:** Runner uploads video to `{bucket}/runs/{run_id}/input.mov` using boto3.
- **Output:** Handler uploads tarball to `{bucket}/runs/{run_id}/results.tar.gz`. Runner downloads and deletes.
- **Worker credentials:** RunPod injects S3 credentials as environment variables inside the container automatically.
- **Runner credentials:** Configured explicitly via config fields (see Section 5).

---

## Section 4: Beam Handler + BeamRunner

### `app/beam_handler.py` (deployed to Beam)

```python
import beam
from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results

@beam.endpoint(cpu=4, memory="16Gi", gpu="A10G", volumes=[beam.Volume(...)])
def process_video(run_id: str, video_volume_path: str, results_volume_path: str, config_preset: str = "balanced"):
    original = apply_cuda_config()
    try:
        output_dir = Path(f"/tmp/cc_output/{run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        db_path = run_pipeline(run_id, video_volume_path, config_preset, output_dir)
        tarball_bytes = package_results(run_id, output_dir, db_path)
        Path(results_volume_path).write_bytes(tarball_bytes)
    finally:
        restore_config(original)
    return {"status": "complete", "results_path": results_volume_path}
```

### `app/services/beam_runner.py`

- `__init__`: takes `api_key`, `volume_id`, `endpoint_id`, plus shared `bus`, `db_path`, `output_base`
- `run_async`: uploads video → invokes endpoint → polls task status → downloads tarball → `ResultImporter.import_tarball()` → cleanup
- `run_batch_async`: sequential `run_async` calls (Beam scales horizontally automatically)
- `destroy_instance`: no-op

Run record/fail/finish methods are the same pattern as `VastAIRunner` (copy, do not share — they're 3 lines each).

---

## Section 5: RunPod Handler + RunPodRunner

### `app/runpod_handler.py` (deployed to RunPod)

```python
import runpod
import boto3
from pathlib import Path
from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results

def handler(job):
    inp = job["input"]
    run_id = inp["run_id"]
    s3_input_key = inp["video_s3_key"]
    s3_results_key = inp["results_s3_key"]
    bucket = inp["bucket"]

    s3 = boto3.client("s3")  # credentials injected by RunPod
    video_path = Path(f"/tmp/{run_id}_input.mov")
    s3.download_file(bucket, s3_input_key, str(video_path))

    output_dir = Path(f"/tmp/cc_output/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = apply_cuda_config()
    try:
        db_path = run_pipeline(run_id, str(video_path), inp.get("config_preset", "balanced"), output_dir)
        tarball_bytes = package_results(run_id, output_dir, db_path)
    finally:
        restore_config(original)

    s3.put_object(Bucket=bucket, Key=s3_results_key, Body=tarball_bytes)
    return {"status": "complete", "results_key": s3_results_key}

runpod.serverless.start({"handler": handler})
```

### `app/services/runpod_runner.py`

- `__init__`: takes `api_key`, `endpoint_id`, `s3_bucket`, `s3_access_key_id`, `s3_secret_access_key`, plus shared `bus`, `db_path`, `output_base`
- `run_async`:
  1. Upload video to RunPod S3 bucket (`runs/{run_id}/input.mov`)
  2. Submit job via RunPod API (`POST /v2/{endpoint_id}/run`)
  3. Poll status (`GET /v2/{endpoint_id}/status/{job_id}`) every 3s
  4. Download results tarball from S3
  5. `ResultImporter.import_tarball()`
  6. Delete both S3 objects
- `run_batch_async`: sequential `run_async` calls
- `destroy_instance`: no-op

RunPod S3 endpoint: confirm from `docs.runpod.io/storage/s3-api` before implementation — likely `https://storage.runpod.io` but verify.

---

## Section 6: Config Changes

`app/api/config.py` — add to `_COMPUTE_FIELDS` and `_COMPUTE_DEFAULTS`:

```python
# Beam
"beam_api_key": str,           default: ""
"beam_volume_id": str,         default: ""
"beam_endpoint_id": str,       default: ""

# RunPod
"runpod_api_key": str,         default: ""
"runpod_endpoint_id": str,     default: ""
"runpod_s3_bucket": str,       default: ""
"runpod_s3_access_key_id": str, default: ""
"runpod_s3_secret_access_key": str, default: ""
```

`pipeline_backend` already exists; valid values expand to include `"beam"` and `"runpod"`.

---

## Files Changed

| File | Change |
|---|---|
| `app/services/gpu_runner.py` | **New** — Protocol definition |
| `app/services/beam_runner.py` | **New** — BeamRunner |
| `app/services/runpod_runner.py` | **New** — RunPodRunner |
| `app/beam_handler.py` | **New** — Beam endpoint (deployed to Beam) |
| `app/runpod_handler.py` | **New** — RunPod handler (deployed to RunPod) |
| `app/worker_core.py` | **New** — extracted pipeline logic |
| `app/vastai_worker.py` | **Modified** — thinned to call worker_core |
| `app/api/videos.py` | **Modified** — runner selection logic |
| `app/api/config.py` | **Modified** — new config fields |
| `app/services/vast_runner.py` | **Unchanged** |

---

## Key Decisions

- **Protocol not ABC**: no changes to `VastAIRunner` class declaration
- **`package_results` returns bytes**: callers decide what to do with them (write to disk, upload, return inline)
- **Provider-native storage only**: Beam Volumes for Beam, RunPod S3 for RunPod — no external bucket
- **Run record methods not shared**: 3-line helpers copied per runner; not worth a base class for the coupling it would create
- **`destroy_instance` is a no-op for serverless runners**: Beam and RunPod manage container lifecycle
