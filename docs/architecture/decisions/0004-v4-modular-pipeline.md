# 4. v4 Modular Pipeline using Metaflow

Date: 2026-05-13

## Status
Accepted

## Context
The v2.1/v3 monolithic pipeline (`src/card_capture/pipeline.py`) became impossible to test and scale. Adding active learning and new ML models caused OOMs and unmaintainable branching logic. We needed a workflow orchestrator.

## Decision
Adopt Metaflow for local execution. Break the pipeline into discrete steps: start, detect, novelty, track, refine, score, resolve, fuse, dedup, store. Pass artifacts via `RunContext`. Retain the SQLite database for the API layer. Use SvelteKit + FastAPI for the labeling/dashboard UI.

## Consequences
**Pros:** 
- Resume capability
- Isolated step testing
- Clear artifact lineage
- Bounded memory per step

**Cons:** 
- Added Metaflow dependency
- Requires translating legacy tests to use the new flow
- Requires a dual-write (Metaflow + SQLite) for UI compatibility
