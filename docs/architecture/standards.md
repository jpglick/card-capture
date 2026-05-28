# V5.5 Architectural Standards

This document defines the binding rules for the V5.5 refactoring. Violations
are caught by the `tests/architecture/` AST scanners and Import Linter.

## 1. Package Boundaries

- **`card_capture.pipeline`**: Pure domain logic and protocols.
  - MUST NOT import `sqlite3`.
  - MUST NOT import provider SDKs (`runpod`, `beam`, `vastai`).
  - MUST NOT import `app.*`.
- **`card_capture.runtime`**: In-process execution and GPU resource management.
  - `runtime.strict_gpu` MUST NOT import `cv2.imread`, `cv2.imwrite`, or `PIL`.
  - MUST NOT reach into `app.*`.
- **`card_capture.data`**: The ONLY place for raw SQL and `sqlite3` calls.
- **`card_capture.platforms`**: The ONLY place for provider-specific SDKs.

## 2. GPU Resident Operations (Strict GPU Boundary)

To achieve the 100ms/frame target, the GPU must not be blocked by CPU I/O.

- **Forbidden in GPU code**:
  - File I/O (`cv2.imread`, `imwrite`, `open()`).
  - Video decoding (`cv2.VideoCapture`).
  - Synchronous CPU transfers (`tensor.cpu()`, `tensor.numpy()`).
- **Exceptions**:
  - `card_capture.runtime.batches` export helpers are the ONLY allowed path for `tensor.cpu()`.

## 3. Data Integrity (Single Writer)

- SQLite MUST be operated in WAL mode.
- All pipeline writes MUST go through the `card_capture.data.writer` queue to
  prevent "database is locked" errors in the unified runtime.

## 4. Telemetry

- Every stage MUST report `elapsed_ms` to the `PipelineTelemetry` sink.
- All resource-intensive stages MUST report `resource_sample` (e.g., peak VRAM).
