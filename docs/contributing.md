# Contributing to Card Capture v5.5

## Surface Ownership
Card Capture is developed with ownership split across four primary surfaces:
- **Surface A (Pipeline/Orchestration):** In-process pipeline, ingestion, core computer vision steps, storage integration.
- **Surface B (Frontend):** SvelteKit dashboard, labeling UX, regression visualization.
- **Surface C (Machine Learning):** F/B classifier, presence detection, DINOv2 embeddings, model training.
- **Surface D (Harness/API):** API endpoints (Contract 2), Regression testing, truth schema (Contract 4), and Golden Set.

## V4_CONCERNS (Historical)
The v4 development cycle tracked architectural flaws and technical debt in `docs/archive/V4_CONCERNS.md`. For the current v5.5 cycle, refer to the project roadmap and issues.

## Contract Changes
The boundaries between surfaces are defined in `docs/contracts/`.
- `v1-api.md`: FastAPI schema shapes.
- `storage-schema.md`: Database structure (`cards.sqlite`).
- `truth-schema.md`: Labeling payload structures.

**Policy:** Any structural change to these contracts requires explicit sign-off from all four surface owners. Do not alter schemas without updating the contract docs and ensuring all surfaces are aware.
