# Card Capture — Pipeline Architecture v5.5

> Scope: extract clean, deduplicated 750×1050 stills of trading cards from a 4K portrait workspace video. This document describes the v5.5 implementation, which uses an in-process runtime optimized for Apple Silicon (MPS). It replaces the Metaflow-based v4.1 architecture.

## Overview

The Card Capture pipeline is a computer-vision system designed to process high-resolution (4K) portrait videos of trading cards. The goal is to produce high-quality 750×1050 rectified stills and persist card metadata and telemetry to a SQLite database.

**Input:** 4K portrait `.MOV` or `.MP4` video.
**Output:** 750×1050 rectified JPEG stills + `cards.sqlite` database.

## Runtime Model: LocalPipelineRuntime

Version 5.5 introduces `LocalPipelineRuntime` (in `src/card_capture/pipeline/runtime_local.py`), an in-process runtime that replaces the Metaflow-based orchestration. 

### Key Characteristics:
- **In-Process Execution:** All stages run within a single process to eliminate IPC overhead and redundant I/O.
- **Synchronous Surface:** The `PipelineRunner` (in `src/card_capture/pipeline/runner.py`) provides a uniform `submit`/`wait`/`cancel` interface.
- **Runtime Modes:**
  - `strict_gpu`: Enforces Apple Silicon MPS-strict execution.
  - `cpu_debug`: Forces CPU execution for debugging.
  - `mixed_compat`: Allows mixed CPU/GPU execution.

## Stage Sequence

The pipeline executes a sequence of 10 stages, defined in `src/card_capture/pipeline/stages/`:

1.  **sample** (`sample.py`): Adaptive Presence Sampler. Starts a streaming decode producer.
2.  **detect** (`detect.py`): YOLO Corner Detection. Drains the producer and batches frames for inference.
3.  **novelty** (`novelty.py`): Background Novelty Gate. Discards detections that match the empty workspace.
4.  **track** (`track.py`): Session-Aware Tracking. Assigns detections to stable card instances using BoT-SORT or ByteTrack.
5.  **refine** (`refine.py`): GPU Refinement. Performs Kornia perspective warps to 750×1050 and computes quality scores.
6.  **score** (`score.py`): Quality Scoring + Pruning. Applies adaptive thresholds to prune low-quality or non-card tracks.
7.  **resolve** (`resolve.py`): Front/Back Resolution. Determines card side (Front/Back) and resolves session identities.
8.  **fuse** (`fuse.py`): Lighting-Diverse Fusion. Selects best views and performs median fusion to reject glare.
9.  **dedup** (`dedup.py`): Global Deduplication. Identifies duplicate cards within and across runs using ReID embeddings and pHash.
10. **store** (`store.py`): Storage via Repositories. Persists images to disk and metadata to SQLite.

## Threading & GPU Boundary

To maintain high performance and avoid blocking the GPU:
- **Producer Thread:** Video decoding and frame preprocessing (Stage 1) happen on a dedicated producer thread.
- **Worker Thread:** All PyTorch, Kornia, and model inference (Stages 2-9) are confined to the worker context.
- **Main Thread:** Orchestration, telemetry, and final data persistence.

MPS (Metal Performance Shaders) is the primary acceleration backend. CUDA is explicitly unsupported in this build.

## Data-Access Layer (DAL)

All database operations are abstracted through the `card_capture.data` repositories:
- **Repositories:** `RunsRepository`, `EventsRepository`, `CardsRepository`.
- **Single-Writer Discipline:** The `Writer` (in `src/card_capture/data/writer.py`) ensures thread-safe, sequential writes to SQLite.
- **Connection Management:** `open_connection` and `read_connection` provide controlled access to the database.
- **Enforcement:** Import-linter contracts prevent stages from making direct `sqlite3` calls.

## Device Path

- **MPS (Default):** Optimized for Apple Silicon.
- **CoreML:** Used by the detector for the fast path when available.
- **CPU Fallback:** Available for `cpu_debug` mode or when hardware acceleration is missing.

## Configuration: PipelineConfig

Key knobs in `src/card_capture/pipeline/request.py`:
- `corner_confidence`: YOLO detection threshold (default: `0.5`).
- `background_novelty_threshold`: Minimum difference from workspace (default: `0.08`).
- `fast_scan_fps`: Sampler scan rate (default: `15.0`).
- `valley_drop_ratio`: Sensitivity for hand-swap detection (default: `0.40`).
- `tracker_backend`: `"botsort"` or `"bytetrack"`.

## Serialized Contracts

- **PipelineRunRequest:** JSON-serializable request containing `run_id`, `input_video`, `output_root`, and `config`.
- **RunManifest:** Final result containing stage timings, card records, and artifact paths.

---
For prior versions see [`docs/archive/`](../archive/).
