# Vast.ai Integration — Sub-project A: Integration Layer Design

**Date:** 2026-05-17
**Tag:** `mps-v5` marks the state before any vast.ai work begins.

---

## Overview

The Mac Mini continues to run the web app (FastAPI + Svelte). The heavy pipeline work is offloaded to an ephemeral vast.ai GPU instance. The instance is provisioned on demand, processes one job or a batch of jobs, and is destroyed immediately after results are downloaded. The Mac Mini is the sole orchestrator — the instance is stateless between runs.

This spec covers **sub-project A only**: the integration layer (provisioning, lifecycle, communication, results import, settings UI, batch mode). The CUDA pipeline that runs on the instance is sub-project B.

---

## Key Decisions

| Decision | Choice |
|---|---|
| Communication | HTTP API on instance |
| Code deployment | Baked vast.ai disk template + `git pull` on boot |
| Results transfer | Tarball download at job completion |
| Instance auth | None (ephemeral window, low sensitivity) |
| App architecture | New `VastAIRunner` alongside `PipelineRunner` |

---

## Section 1: Instance Lifecycle

`VastAIRunner` owns the full lifecycle for every job and batch.

### Sequence

1. **Provision** — Call vast.ai SDK with the configured GPU type and pre-baked template ID. Receive `instance_id` and public IP.
2. **Boot** — The instance's userscript executes automatically:
   - `git pull` the repo (configured branch, default `main`)
   - `pip install -e .` (app layer only — heavy deps pre-installed on template)
   - Start `uvicorn app.vastai_worker:app --port 8765`
   - Target: 60–90s total boot time
3. **Ready check** — Poll `GET http://{ip}:8765/health` every 5s with a 2-minute timeout. Failure = abort + destroy instance.
4. **Job execution** — Upload video, submit job, poll, download tarball (see Sections 2–3).
5. **Spin down** — After the final job's tarball is downloaded and confirmed (`DELETE /jobs/{id}`), the worker exits → vast.ai billing stops. `VastAIRunner` also calls `vastai destroy {instance_id}` as a belt-and-suspenders step.
6. **Idle guard** — If the worker process has received no new job within `cuda_idle_timeout_s` (default 300s), it exits. Guards against Mac crash mid-batch leaving an orphaned instance.

### Orphan protection

`VastAIRunner` stores the active `instance_id` in `card_capture_config.json` under `"active_vast_instance"` at provision time and clears it on destroy. On app startup, if this key is set, the app checks whether the instance is still running and destroys it if so. Prevents billing leaks across app restarts.

---

## Section 2: `VastAIRunner` Service

**File:** `app/services/vast_runner.py`

Implements the same async interface as `PipelineRunner` so the existing runs API needs only a backend selector.

```python
class VastAIRunner:
    async def run_async(self, run_id: str, *, video: str, output_dir: str,
                        db: str, gpu_type: str, config_preset: str) -> None
    async def run_batch_async(self, jobs: list[JobSpec]) -> None
    async def destroy_instance(self) -> None
```

`VastAIRunner` holds instance state (`instance_id`, `instance_ip`) and delegates to three collaborators:

### Collaborators

**`VastAIClient`** — `app/services/vast_client.py`
Thin wrapper around the vast.ai Python SDK. Methods: `provision(gpu_type, template_id)`, `destroy(instance_id)`, `list_offers(gpu_type)`. Isolated for testability.

**`InstanceWorkerClient`** — `app/services/worker_client.py`
HTTP client for the instance's FastAPI worker. Methods: `health_check()`, `upload_video(path)`, `submit_job(params)`, `poll_status(job_id)`, `download_results(job_id, dest)`, `confirm_downloaded(job_id)`.

**`ResultImporter`** — `app/services/result_importer.py`
Unpacks the downloaded tarball, copies crops to the local output directory, imports card rows from `export.json` into the local `cards.sqlite`. Emits the same `run_completed` SSE event as the local pipeline.

### Backend selection

The runs API reads `pipeline_backend` from `card_capture_config.json`:

```json
{ "pipeline_backend": "cuda" }
```

When `"cuda"`, instantiate `VastAIRunner`; otherwise use the existing `PipelineRunner`. No other changes to the runs route.

`VastAIRunner.run_async` emits the same SSE events (`run_started`, `run_completed`, `run_failed`) at the same lifecycle points so the existing runs page, SSE stream, and pipeline_runs table all work unchanged.

---

## Section 3: Instance-Side Worker

**File:** `app/vastai_worker.py`

A standalone minimal FastAPI app started on the instance. Does not import the full web UI stack.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Returns 200 when worker is ready to accept jobs |
| `POST` | `/jobs` | Upload video + params, enqueue job, return `job_id` |
| `GET` | `/jobs/{id}` | Status: `pending \| running \| complete \| failed` + `progress_pct` |
| `GET` | `/jobs/{id}/results` | Stream results tarball (only when `complete`) |
| `DELETE` | `/jobs/{id}` | Confirm tarball received; triggers idle-shutdown if queue empty |

### Job execution

Jobs run from an async in-memory queue with a single worker coroutine. Sequential execution is intentional — one CUDA pipeline saturates the 4090; parallel jobs would thrash GPU memory without throughput gain.

The worker calls the CUDA pipeline (sub-project B) directly — no Metaflow on the instance. Metaflow's local datastore remains on the Mac Mini.

### Results tarball

Contents:
- `crops/` — all fused JPEG files (`instance_{id}_fused.jpg`)
- `export.json` — all card rows as JSON (instance_id, track_id, confidence, fused_image_path, quality scores)
- `frames/` — source frames (included only if `include_frames: true` in job params)

### Idle shutdown

After `DELETE /jobs/{id}` empties the queue, the worker calls `sys.exit(0)`. A systemd unit or supervisord config on the template restarts on crash but not on clean exit — so a successful completion triggers billing stop. The Mac Mini also calls `vastai destroy` as belt-and-suspenders (see Section 1).

---

## Section 4: Config, Settings UI, and GPU Selection

### Config fields

Added to `card_capture_config.json`:

```json
{
  "pipeline_backend": "mps",
  "cuda_gpu_type": "RTX 4090",
  "vast_template_id": "",
  "cuda_idle_timeout_s": 300,
  "active_vast_instance": null
}
```

`vast_api_key` is read from the `VAST_API_KEY` environment variable. It is never written to the config file. The settings UI masks the key on display and reads it from env at validation time.

### GPU types

| Label | SDK query |
|---|---|
| `RTX 4090` | `gpu_name=RTX 4090` |
| `Flagship` | Sort available offers by TFLOPS descending, take cheapest at top tier |
| `RTX 5060 Ti` | `gpu_name=RTX 5060 Ti` |
| `Custom` | Free-text offer search string (power user escape hatch) |

### Settings page — new Compute section

Added to `/settings` below existing sections:

- **Pipeline** — toggle: `Local (MPS)` / `Cloud GPU (vast.ai)`. Switching to Cloud reveals the remaining fields.
- **GPU type** — dropdown with the four options above.
- **API key** — masked text input. On blur, tests the key with a lightweight vast.ai API call and shows ✓ or ✗.
- **Idle timeout** — number input (seconds), default 300.
- **Vast template ID** — text input for the pre-baked disk template ID.

### Runs page

When a run executes remotely, the run card shows a small "☁ Cloud" badge. The SSE progress stream and run detail page are otherwise unchanged.

---

## Section 5: Batch Mode

### API

```
POST /api/v1/runs/batch    — body: { video_ids: [1, 2, 3] } → { batch_id }
GET  /api/v1/runs/batch/{batch_id} — overall status + per-video progress
```

`VastAIRunner.run_batch_async` provisions one instance, then processes jobs sequentially — waiting for each tarball download and local import before submitting the next. This keeps instance disk usage low and lets the Mac import results incrementally.

On error: a failed video is marked `failed` and the batch continues. The instance is destroyed only after the last video (successful or failed). Individual failed jobs can be retried from the runs page.

### Batch UI

Located at `/batch`. Displays the existing videos list with a checkbox on each row. A **Process Batch** button at the top (enabled when ≥1 video is checked) submits the selected video IDs and redirects to the batch status view.

The batch status view shows a list of the submitted videos with individual progress bars and status labels. Each completed video links to its run detail page. No drag-and-drop.

Batch runs also appear individually on the main runs page as they complete, so the existing cards/runs UI remains the primary result view.

---

## New Files

| File | Purpose |
|---|---|
| `app/services/vast_runner.py` | `VastAIRunner` — lifecycle orchestration |
| `app/services/vast_client.py` | `VastAIClient` — vast.ai SDK wrapper |
| `app/services/worker_client.py` | `InstanceWorkerClient` — HTTP client for instance API |
| `app/services/result_importer.py` | `ResultImporter` — tarball unpack + SQLite import |
| `app/vastai_worker.py` | Instance-side FastAPI worker |
| `app/api/batch.py` | Batch API routes |
| `app/web/src/routes/batch/+page.svelte` | Batch UI |

## Modified Files

| File | Change |
|---|---|
| `app/api/runs.py` | Backend selector (MPS vs CUDA) |
| `app/api/config.py` | Expose new vast.ai config fields |
| `app/web/src/routes/settings/+page.svelte` | Compute section |
| `app/web/src/lib/api/types.ts` | Batch types |
| `app/web/src/lib/api/client.ts` | Batch API methods |
| `card_capture_config.json` | New fields (pipeline_backend, cuda_gpu_type, etc.) |
