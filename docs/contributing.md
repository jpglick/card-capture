# Contributing to Card Capture v4

## Wave Development Cycle
Card Capture v4 is developed in "Waves", with ownership split across four primary surfaces:
- **Surface A (Pipeline/Orchestration):** Metaflow backbone, ingestion, core computer vision steps, storage integration.
- **Surface B (Frontend):** SvelteKit dashboard, labeling UX, regression visualization.
- **Surface C (Machine Learning):** F/B classifier, presence detection, DINOv2 embeddings, model training.
- **Surface D (Harness/API):** API endpoints (Contract 2), Regression testing, truth schema (Contract 4), and Golden Set.

Each wave progresses through planning, implementation, and review.

## V4_CONCERNS
Every architectural flaw, bug, missing integration, or technical debt identified during a wave MUST be logged in `V4_CONCERNS.md`.
- No wave ships (is marked resolved) until its blocking "High" concerns are mitigated or formally deferred.
- Concerns are never deleted; they are moved to the "Resolved" section with a reference to the fixing PR or commit.

## Contract Changes
The boundaries between surfaces are defined in `docs/contracts/`.
- `v1-api.md`: FastAPI schema shapes.
- `storage-schema.md`: Database structure (`cards.sqlite`).
- `truth-schema.md`: Labeling payload structures.

**Policy:** Any structural change to these contracts requires explicit sign-off from all four surface owners. Do not alter schemas without updating the contract docs and ensuring all surfaces are aware.
