# Project Instructions: Card Capture

## Foundational Mandates

### 1. Source Control Hygiene
- **NEVER** commit the `.venv` directory, large binaries, or large media files (e.g., `.MOV`, `.MP4`, `.sqlite`).
- Always verify your `git status` before committing to ensure no generated artifacts or virtual environments are staged.
- Use the provided `.gitignore` to maintain a clean repository.

### 2. Execution & Testing Environment
- **GPU/MPS Acceleration:** Hardware acceleration (Apple Silicon MPS or CUDA) is **NOT** available within the Gemini CLI restricted execution environment.
- **Manual Processing:** All performance-sensitive commands and final processing runs **MUST** be executed manually in a local terminal to utilize the GPU and verify real-world timings.
- Use the Gemini CLI for research, strategy, and code modification, but perform the "Act & Validate" phase of high-resolution video processing in your native shell.

## Architecture & Conventions

### 1. Unified In-Process Runtime
- The pipeline MUST run within a single process using `UnifiedRuntime` to minimize IPC overhead and redundant I/O.
- Metaflow is relegated to remote orchestration and historical baseline tracking; local development and high-performance production runs use the in-process loop.

### 2. Strict GPU Boundary
- All PyTorch, Kornia, and model inference MUST happen within the guarded `_worker` thread context of the runtime.
- CPU tasks (video decoding, metadata storage) MUST remain on the producer/main threads to prevent blocking the GPU.
- Use `PrecisionNormalizer` for all homography and warping to ensure cross-platform consistency.

### 3. Data Access Layer (DAL)
- ALL database writes MUST go through the `SingleWriterDAL` to prevent SQLite concurrency locks.
- No direct `sqlite3` calls allowed in pipeline stages; use the DAL protocols.

### 4. Session-Anchored Logic
- The pipeline relies on discrete temporal sessions to group card presentations.
- Tracking and deduplication use these sessions as logical boundaries.
