# AI Agent Mandates — Card Capture

This file defines foundational rules and architectural standards for all AI agents
(Gemini, Claude, GPT, etc.) working on the Card Capture project.

---

## 1. Security & Hygiene

- **NO SECRETS:** Never log, print, or commit API keys, `.env` files, or credentials.
- **NO LARGE BINARIES:** Never commit `.venv`, `.sqlite`, `.MOV`, `.MP4`, or large `.pt` models.
- **SOURCE CONTROL:** Do not stage or commit changes unless explicitly requested. Verify `git status` before any commit to ensure no artifacts are accidentally included.

## 2. Execution Environment

- **GPU LIMITATION:** Hardware acceleration (Apple Silicon MPS or NVIDIA CUDA) is **NOT** available within the restricted execution environment of most AI agents.
- **MANUAL VERIFICATION:** All performance-sensitive commands and final processing runs **MUST** be executed manually in a local terminal to verify real-world timings and GPU behavior.
- **SIMULATION:** When hardware is required for tests, use mocks or the `@pytest.mark.quarantine` marker if the test cannot be reasonably mocked.

## 3. Architecture & Conventions (v5.5+)

- **UNIFIED RUNTIME:** The pipeline MUST run within a single process using `UnifiedRuntime`. Metaflow is relegated to remote orchestration and historical tracking.
- **STRICT GPU BOUNDARY:** All PyTorch/Kornia operations MUST happen within the guarded `_worker` thread context of the runtime. CPU tasks (decoding, metadata) MUST remain on producer/main threads.
- **CENTRALIZED MODELS:** Use `card_capture.models` for all domain objects. Do not define redundant dataclasses in local modules.
- **DATA ACCESS LAYER (DAL):** ALL database writes MUST go through the `SingleWriterDAL` in `card_capture.dal` to prevent SQLite concurrency locks. No direct `sqlite3` calls in pipeline logic.
- **SESSION-ANCHORED LOGIC:** The pipeline relies on discrete temporal sessions to group card presentations. Tracking and deduplication use these as logical boundaries.

## 4. Engineering Standards

- **IDEMPOTENCY:** Migrations and setup scripts must be idempotent. Use `IF NOT EXISTS` in SQL and `mkdir(exist_ok=True)` in Python.
- **TEST-DRIVEN:** Always add or update tests for any logic change. Ensure the suite remains green (`pytest -m "not quarantine"`).
- **CONVENTIONS:** Adhere to existing naming (snake_case), formatting (Black/PEP8), and type hinting (strict) conventions.
