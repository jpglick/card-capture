# Sports Card Video Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python MVP that processes a video, detects trading cards through a swappable detector interface, saves high-quality card stills as image files, stores metadata in SQLite, and exposes a local review UI.

**Architecture:** The app is split into small modules: domain models, storage, sampling, detection, cropping, scoring, selection, orchestration, CLI, and review UI. The detector interface normalizes card detections so the initial `cardcaptor-v3` Ultralytics adapter can later be replaced by ONNX Runtime without touching the rest of the pipeline.

**Tech Stack:** Python 3.9+, OpenCV, NumPy, SQLite, argparse, optional Ultralytics/Hugging Face Hub for model inference, optional FastAPI/Jinja/Uvicorn for review UI.

---

## File Structure

- Create `pyproject.toml`: package metadata, dependencies, console script.
- Create `src/card_capture/__init__.py`: package marker.
- Create `src/card_capture/models.py`: dataclasses for frames, detections, crops, scores, saved cards.
- Create `src/card_capture/storage.py`: SQLite schema and metadata operations.
- Create `src/card_capture/sampler.py`: OpenCV video frame sampler.
- Create `src/card_capture/detectors.py`: detector protocol, fake detector, Ultralytics cardcaptor adapter.
- Create `src/card_capture/cropper.py`: oriented polygon ordering and perspective crop.
- Create `src/card_capture/scoring.py`: sharpness, glare, size, and combined quality score.
- Create `src/card_capture/selector.py`: grouping and best-candidate selection.
- Create `src/card_capture/pipeline.py`: end-to-end processing orchestration.
- Create `src/card_capture/cli.py`: `process` and `review` commands.
- Create `src/card_capture/review.py`: local FastAPI review app factory.
- Create `src/card_capture/templates/review.html`: simple review gallery.
- Create `tests/test_storage.py`: SQLite behavior.
- Create `tests/test_cropper.py`: crop geometry behavior.
- Create `tests/test_scoring_selector.py`: score composition and candidate selection.
- Create `tests/test_cli.py`: CLI argument/error behavior.

## Tasks

### Task 1: Project Skeleton And Models

**Files:**
- Create: `pyproject.toml`
- Create: `src/card_capture/__init__.py`
- Create: `src/card_capture/models.py`
- Test: import smoke through later tests

- [ ] Write package metadata and dataclasses for normalized pipeline objects.
- [ ] Verify the package imports with `PYTHONPATH=src python3 -c "import card_capture; print(card_capture.__version__)"`.

### Task 2: SQLite Storage

**Files:**
- Create: `src/card_capture/storage.py`
- Test: `tests/test_storage.py`

- [ ] Write failing tests for schema creation, video insertion, detection insertion, saved-card insertion, and review decisions.
- [ ] Run `PYTHONPATH=src python3 -m pytest tests/test_storage.py -v` and verify failure.
- [ ] Implement `Storage` with SQLite migrations and typed row helpers.
- [ ] Run the same test and verify pass.

### Task 3: Crop Geometry

**Files:**
- Create: `src/card_capture/cropper.py`
- Test: `tests/test_cropper.py`

- [ ] Write failing tests for stable polygon ordering and perspective crop dimensions.
- [ ] Run `PYTHONPATH=src python3 -m pytest tests/test_cropper.py -v` and verify failure.
- [ ] Implement `order_points_clockwise` and `CardCropper.crop`.
- [ ] Run the same test and verify pass.

### Task 4: Quality Scoring And Selection

**Files:**
- Create: `src/card_capture/scoring.py`
- Create: `src/card_capture/selector.py`
- Test: `tests/test_scoring_selector.py`

- [ ] Write failing tests for sharpness ranking, glare penalty, score component output, timestamp grouping, and max candidate selection.
- [ ] Run `PYTHONPATH=src python3 -m pytest tests/test_scoring_selector.py -v` and verify failure.
- [ ] Implement `QualityScorer` and `CandidateSelector`.
- [ ] Run the same test and verify pass.

### Task 5: Sampling, Detection, And Pipeline

**Files:**
- Create: `src/card_capture/sampler.py`
- Create: `src/card_capture/detectors.py`
- Create: `src/card_capture/pipeline.py`
- Test: extend `tests/test_scoring_selector.py` or add `tests/test_pipeline.py`

- [ ] Write failing tests using a fake sampler and fake detector to prove the pipeline saves the highest-quality crop and records metadata.
- [ ] Run `PYTHONPATH=src python3 -m pytest tests/test_pipeline.py -v` and verify failure.
- [ ] Implement the sampler, detector interface, lazy Ultralytics adapter, and process orchestrator.
- [ ] Run the same test and verify pass.

### Task 6: CLI And Review UI

**Files:**
- Create: `src/card_capture/cli.py`
- Create: `src/card_capture/review.py`
- Create: `src/card_capture/templates/review.html`
- Test: `tests/test_cli.py`

- [ ] Write failing CLI tests for missing video path and process invocation with fake detector mode.
- [ ] Run `PYTHONPATH=src python3 -m pytest tests/test_cli.py -v` and verify failure.
- [ ] Implement argparse commands and a small FastAPI review app factory.
- [ ] Run the same test and verify pass.

### Task 7: Full Verification

**Files:**
- Modify only files needed to fix verification failures.

- [ ] Run `PYTHONPATH=src python3 -m pytest -v`.
- [ ] Run `PYTHONPATH=src python3 -m card_capture.cli --help`.
- [ ] Run `PYTHONPATH=src python3 -m card_capture.cli process --help`.
- [ ] Run `PYTHONPATH=src python3 -m card_capture.cli review --help`.

## Self-Review

- Spec coverage: local video processing, pretrained detector adapter, crop/deskew, scoring, candidate selection, files plus SQLite, CLI, and review UI are covered.
- Placeholder scan: no task depends on undefined behavior; optional real model dependencies are intentionally lazy because the local environment may not have network access.
- Type consistency: normalized dataclasses flow through storage, cropper, scorer, selector, and pipeline.

## Note On Git

This directory is not a git repository, so implementation will use verification checkpoints instead of commit checkpoints.
