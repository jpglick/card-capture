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
- **Session-Anchored Logic:** The pipeline relies on discrete temporal sessions to group card presentations.
- **Kornia GPU Refinement:** Normalization and homography should be batch-processed using Kornia tensors on the MPS device whenever possible.
