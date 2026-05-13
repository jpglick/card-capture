# v4 Surface D — Harness / Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the regression harness, metric definitions, golden-set management, labeling-endpoint backend, and hard-case capture wire-up — the substrate that lets every Surface-C algorithmic change land with regression evidence.

**Architecture:** Pure functions over `(cards.sqlite, truth.json)` for all metrics — no pipeline coupling, no Metaflow dependency, no model loads. Surface D's deliverables are read-only against the pipeline (it observes; it does not drive). The harness CLI is a thin Click wrapper; the regression API is FastAPI route handlers; the label endpoints are CRUD over Surface A's new tables.

**Tech Stack:** Python ≥3.11, Click (existing CLI framework), SQLite (existing), Pydantic v2, NumPy, scikit-image (SSIM), scikit-learn (Adjusted Rand Index), FastAPI (consuming Surface A's app factory).

**Spec reference:** `docs/superpowers/specs/2026-05-12-v4-architecture-design.md`. This plan implements Surface D across Waves 1, 2, and 3.

**Critical safety gate.** Surface D is the safety gate. Until Phase D2 (metrics + harness CLI) ships, no algorithmic refactor in Surface C may merge. Phase D1 (truth schema) blocks Phase D2 and parts of Surface B Wave 2.

---

## File Structure

**New files (this plan creates):**

- `harness/__init__.py`
- `harness/schema.py` — Pydantic models for `truth.json`.
- `harness/validator.py` — JSON Schema validation + Pydantic round-trip.
- `harness/metrics/__init__.py`
- `harness/metrics/card_recall.py`
- `harness/metrics/card_precision.py`
- `harness/metrics/side_accuracy.py`
- `harness/metrics/dedup_accuracy.py` — ARI + pair F1.
- `harness/metrics/image_quality.py` — SSIM + PSNR.
- `harness/match.py` — ground-truth ↔ detection matcher (used by recall/precision/side metrics).
- `harness/runner.py` — orchestrates a run: load truth, load detections, compute metrics, persist `regression_runs` row.
- `harness/baseline.py` — read/write `regression_baselines`, including the `baseline_v4.1` freeze.
- `harness/hard_cases.py` — wraps existing `analysis/hard_case_capture.py` writes into `hard_cases` table.
- `harness/cli.py` — Click commands `card-capture harness {run,baselines,compare}`.
- `app/services/labeling_service.py` — backend for `/api/v1/label/*` endpoints.
- `app/services/regression_service.py` — backend for `/api/v1/regression/*` endpoints.
- `app/services/golden_set.py` — manage labeled-video registry + status.
- `tests/harness/test_schema.py`
- `tests/harness/test_validator.py`
- `tests/harness/test_metrics_card_recall.py`
- `tests/harness/test_metrics_card_precision.py`
- `tests/harness/test_metrics_side_accuracy.py`
- `tests/harness/test_metrics_dedup_accuracy.py`
- `tests/harness/test_metrics_image_quality.py`
- `tests/harness/test_match.py`
- `tests/harness/test_runner.py`
- `tests/harness/test_cli.py`
- `tests/harness/fixtures/truth_examples/` — hand-authored truth.json files for unit tests.
- `tests/harness/fixtures/runs/` — hand-authored `cards.sqlite` snapshots for metric tests.
- `tests/app/test_label_endpoints.py`
- `tests/app/test_regression_endpoints.py`
- `docs/contracts/truth-schema.md` — frozen Contract 4.
- `docs/contracts/metrics.md` — frozen metric definitions.
- `golden_set/README.md` — the labeled-video registry index.
- `golden_set/videos/` — per-video labeling output directory (gitignored beyond README).

**Modified files (this plan touches):**

- `src/card_capture/cli.py` — `harness` subcommand registered (delegates to `harness.cli`).
- `src/card_capture/analysis/hard_case_capture.py` — emits to `hard_cases` table (in addition to existing JSON dumps).
- `app/api/regression.py` — handlers wired to `regression_service`.
- `app/api/label.py` — handlers wired to `labeling_service`.

---

## Phase D0 — Contract 4: Truth Schema + Metric Definitions (Wave 1, Day 1)

### Task D0.1: Draft Contract 4 — truth schema

**Files:**
- Create: `docs/contracts/truth-schema.md`

- [ ] **Step 1: Write the schema doc**

```markdown
# Contract 4 — truth.json Schema

Per-video ground-truth file consumed by the harness, written by the labeling
UX (Surface B), validated by `harness.schema.TruthFile`.

## Top-level shape

```json
{
  "video_id": "practice_session_03",
  "schema_version": 1,
  "expected_cards": [ { ... } ]
}
```

## `expected_cards[]` element

Required fields:
- `card_id` (string): stable identifier within this truth file.
- `front_present` (bool).
- `back_present` (bool).
- `physical_card_key` (string): cross-video identity; cards with the same key
  in different videos are duplicates of the same physical card.
- `is_foil` (bool).

Optional fields:
- `approx_front_window_ms` ([int, int] | null): inclusive timestamp range.
- `approx_back_window_ms` ([int, int] | null).
- `notes` (string).

## Backward compatibility

`schema_version: 0` (the current `templates/labeling.html` output) is read by
adding a compat adapter in `harness.schema.from_legacy()`. New labeling UX
writes `schema_version: 1` only.
```

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/truth-schema.md
git commit -m "docs(contracts): freeze Contract 4 truth.json schema"
```

### Task D0.2: Draft metric definitions

**Files:**
- Create: `docs/contracts/metrics.md`

- [ ] **Step 1: Write the metric-definitions doc**

Each metric: name, formula, inputs, output range, edge cases (no detections, no truth, divide-by-zero), worked example.

```markdown
# v4 Metric Definitions

All metrics are pure functions over `(cards.sqlite, truth.json)`. No pipeline
re-run is required to recompute them.

## card_recall
Matched ground-truth cards / total ground-truth cards.
- Range: [0, 1].
- Matching: detected card matches GT card iff its temporal extent overlaps
  `approx_*_window_ms` AND its assigned `side` matches the GT's
  `front_present` / `back_present`. If no time window provided, match by
  detection order within the video.
- Edge case: 0 GT cards → recall is undefined; report as `null` not `1.0`.

## card_precision
Real detections / total detections.
- Real = matched to a GT card. Phantom = unmatched (real card not in GT, OR
  no card at all).
- Edge case: 0 detections → undefined; report as `null`.

## side_accuracy
Correct front/back assignments / total assigned instances.
- Only computed over detections that matched a GT card.
- A GT card with `front_present=true` and `back_present=true` allows two
  correct assignments.

## dedup_accuracy (Adjusted Rand Index OR pair F1)
- Ground truth: clusters defined by `physical_card_key` across all detections
  matched to GT in this video (and across videos if `--cross-video` flag set).
- Predicted: pipeline's `dedup_groups` table or its v4 successor.
- Report both ARI and pair F1; default gate uses ARI.

## image_quality (SSIM)
- Reference: a hand-picked best frame per GT card (stored under
  `golden_set/videos/<id>/reference_frames/`).
- Test: fused canonical of the detection matched to the GT card.
- Report mean SSIM over all matched cards; also PSNR for sanity.
- Edge case: no reference frame → skip card; report coverage % alongside.

## Noise floor (initial; recalibrated after 3 runs)
- recall: ±0.01 absolute
- precision: ±0.01 absolute
- side accuracy: ±0.02 absolute
- dedup ARI: ±0.02 absolute
- SSIM: ±0.01 absolute
```

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/metrics.md
git commit -m "docs(contracts): freeze v4 metric definitions"
```

### Task D0.3: Ack from B/C and Contract 4 freeze

Same shape as Surface A's Task A0.4. After ack:

```bash
git tag -a v4-contract4-frozen -m "Contract 4 ack'd by B, C, D"
```

---

## Phase D1 — Truth Schema + Validator (Wave 1)

### Task D1.1: Write the failing schema tests

**Files:**
- Create: `tests/harness/test_schema.py`
- Create: `tests/harness/fixtures/truth_examples/minimal_valid.json`
- Create: `tests/harness/fixtures/truth_examples/with_optional_fields.json`
- Create: `tests/harness/fixtures/truth_examples/missing_required.json`
- Create: `tests/harness/fixtures/truth_examples/legacy_v0.json`

- [ ] **Step 1: Write fixtures**

`minimal_valid.json`:
```json
{
  "video_id": "v1",
  "schema_version": 1,
  "expected_cards": [
    {
      "card_id": "c1",
      "front_present": true,
      "back_present": false,
      "physical_card_key": "k1",
      "is_foil": false
    }
  ]
}
```

`with_optional_fields.json`:
```json
{
  "video_id": "v2",
  "schema_version": 1,
  "expected_cards": [
    {
      "card_id": "c1",
      "front_present": true,
      "back_present": true,
      "physical_card_key": "k1",
      "is_foil": true,
      "approx_front_window_ms": [1000, 2500],
      "approx_back_window_ms": [3000, 4200],
      "notes": "occluded by hand at 1.7s"
    }
  ]
}
```

`missing_required.json` omits `physical_card_key`.

`legacy_v0.json` is whatever the current `templates/labeling.html` emits — the surface owner reads that file to produce this fixture.

- [ ] **Step 2: Write tests**

```python
# tests/harness/test_schema.py
import json
from pathlib import Path
from pydantic import ValidationError
import pytest

from harness.schema import TruthFile, from_legacy

FIX = Path("tests/harness/fixtures/truth_examples")

def test_minimal_valid_parses():
    payload = json.loads((FIX / "minimal_valid.json").read_text())
    tf = TruthFile.model_validate(payload)
    assert tf.video_id == "v1"
    assert len(tf.expected_cards) == 1

def test_with_optional_fields_parses():
    payload = json.loads((FIX / "with_optional_fields.json").read_text())
    tf = TruthFile.model_validate(payload)
    assert tf.expected_cards[0].approx_front_window_ms == (1000, 2500)
    assert tf.expected_cards[0].is_foil is True

def test_missing_required_field_rejected():
    payload = json.loads((FIX / "missing_required.json").read_text())
    with pytest.raises(ValidationError):
        TruthFile.model_validate(payload)

def test_legacy_v0_converts():
    payload = json.loads((FIX / "legacy_v0.json").read_text())
    tf = from_legacy(payload)
    assert tf.schema_version == 1
    # exact assertions depend on legacy content; surface owner fills in.

def test_window_validation_rejects_inverted_range():
    payload = {
        "video_id": "v3", "schema_version": 1,
        "expected_cards": [{
            "card_id": "c1", "front_present": True, "back_present": False,
            "physical_card_key": "k1", "is_foil": False,
            "approx_front_window_ms": [5000, 2000],
        }],
    }
    with pytest.raises(ValidationError):
        TruthFile.model_validate(payload)
```

- [ ] **Step 3: Run — fails**

```
pytest tests/harness/test_schema.py -v
```

### Task D1.2: Implement schema

**Files:**
- Create: `harness/__init__.py`
- Create: `harness/schema.py`

- [ ] **Step 1: Implement**

```python
# harness/schema.py
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExpectedCard(BaseModel):
    model_config = ConfigDict(frozen=True)
    card_id: str
    front_present: bool
    back_present: bool
    physical_card_key: str
    is_foil: bool
    approx_front_window_ms: tuple[int, int] | None = None
    approx_back_window_ms: tuple[int, int] | None = None
    notes: str | None = None

    @field_validator("approx_front_window_ms", "approx_back_window_ms")
    @classmethod
    def _ordered_window(cls, v):
        if v is None:
            return v
        start, end = v
        if start > end:
            raise ValueError(f"window must be (start, end) with start <= end; got {v!r}")
        return v


class TruthFile(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    schema_version: int = Field(ge=1)
    expected_cards: list[ExpectedCard]


def from_legacy(payload: dict[str, Any]) -> TruthFile:
    """Adapter: read schema_version==0 (current labeling.html output) and
    return a TruthFile with schema_version==1.

    The exact legacy shape is read by the surface owner from
    templates/labeling.html and tests/harness/fixtures/truth_examples/legacy_v0.json.
    Map legacy fields one-to-one into schema_version=1.
    """
    if payload.get("schema_version", 0) == 1:
        return TruthFile.model_validate(payload)
    # ... legacy mapping (surface owner fills in based on legacy_v0.json).
    raise NotImplementedError("legacy adapter not yet implemented; see tests/harness/fixtures/truth_examples/legacy_v0.json")
```

- [ ] **Step 2: Run tests — passes (except the legacy test until adapter is filled in)**

```
pytest tests/harness/test_schema.py -v
```

- [ ] **Step 3: Fill in legacy adapter**

The surface owner reads `templates/labeling.html` (referenced in CLAUDE.md §3.9 / §11) to determine the legacy field names, then implements the mapping in `from_legacy`.

- [ ] **Step 4: Run all tests — pass**

- [ ] **Step 5: Commit**

```bash
git add harness/ tests/harness/test_schema.py tests/harness/fixtures/
git commit -m "feat(harness): truth.json Pydantic schema with legacy adapter"
```

### Task D1.3: JSON Schema export + validator CLI

**Files:**
- Create: `harness/validator.py`
- Create: `tests/harness/test_validator.py`

- [ ] **Step 1: Failing test**

```python
# tests/harness/test_validator.py
import subprocess
from pathlib import Path

FIX = Path("tests/harness/fixtures/truth_examples")

def test_validator_passes_on_valid():
    result = subprocess.run(
        ["python", "-m", "harness.validator", str(FIX / "minimal_valid.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

def test_validator_fails_on_invalid():
    result = subprocess.run(
        ["python", "-m", "harness.validator", str(FIX / "missing_required.json")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "physical_card_key" in result.stderr
```

- [ ] **Step 2: Implement validator**

```python
# harness/validator.py
"""CLI: python -m harness.validator path/to/truth.json"""
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from harness.schema import TruthFile


def validate_file(path: Path) -> int:
    try:
        TruthFile.model_validate_json(path.read_text())
    except ValidationError as exc:
        print(f"{path}: invalid\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    exit_code = 0
    for p in paths:
        exit_code |= validate_file(p)
    sys.exit(exit_code)
```

- [ ] **Step 3: Run test — passes**

- [ ] **Step 4: Commit**

```bash
git add harness/validator.py tests/harness/test_validator.py
git commit -m "feat(harness): truth.json validator CLI"
```

---

## Phase D2 — Metrics (Wave 1; parallelizable by metric)

Each metric is its own task with its own test. Tasks D2.1–D2.5 can run in parallel within Surface D since they share no code.

### Task D2.1: card_recall + match.py

**Files:**
- Create: `harness/match.py`
- Create: `harness/metrics/__init__.py`
- Create: `harness/metrics/card_recall.py`
- Create: `tests/harness/test_match.py`
- Create: `tests/harness/test_metrics_card_recall.py`
- Create: `tests/harness/fixtures/runs/all_matched/cards.sqlite` (hand-authored)

- [ ] **Step 1: Hand-author a tiny `cards.sqlite` fixture**

A 3-card run: 2 detected and matched to GT, 1 missed. Use `sqlite3` CLI or a Python script committed under `tests/harness/fixtures/runs/_seed_all_matched.py` to (re)create. The fixture is committed as a `.sqlite` file under 50 KB.

- [ ] **Step 2: Failing tests**

```python
# tests/harness/test_match.py
from pathlib import Path
from harness.match import match_detections_to_truth
from harness.schema import TruthFile
import json

def test_all_matched_returns_correct_pairs():
    truth = TruthFile.model_validate_json(
        Path("tests/harness/fixtures/runs/all_matched/truth.json").read_text())
    pairs = match_detections_to_truth(
        db_path=Path("tests/harness/fixtures/runs/all_matched/cards.sqlite"),
        truth=truth,
        video_id="all_matched",
    )
    # 2 matches, 1 unmatched GT (the missed card)
    assert sum(1 for p in pairs if p.gt_card_id and p.detection_id) == 2
    assert sum(1 for p in pairs if p.gt_card_id and not p.detection_id) == 1
```

```python
# tests/harness/test_metrics_card_recall.py
from harness.metrics.card_recall import card_recall
from pathlib import Path

def test_recall_2_of_3():
    score = card_recall(
        db_path=Path("tests/harness/fixtures/runs/all_matched/cards.sqlite"),
        truth_path=Path("tests/harness/fixtures/runs/all_matched/truth.json"),
        video_id="all_matched",
    )
    assert score == 2 / 3

def test_recall_undefined_when_no_truth(tmp_path):
    truth_path = tmp_path / "empty.json"
    truth_path.write_text('{"video_id":"x","schema_version":1,"expected_cards":[]}')
    score = card_recall(
        db_path=Path("tests/harness/fixtures/runs/all_matched/cards.sqlite"),
        truth_path=truth_path, video_id="x",
    )
    assert score is None
```

- [ ] **Step 3: Implement match.py**

```python
# harness/match.py
"""Match detected cards (from cards.sqlite) to ground-truth cards (from
truth.json) for a given video. Used by recall, precision, and side metrics.

Matching rule:
- If GT card has approx_*_window_ms, detection's timestamp_ms must fall in
  that window AND its assigned side must equal the GT's front_present/
  back_present flag for that side.
- If no window, match in detection order within the video (first-detected =
  GT[0], second = GT[1], ...). This is the legacy fallback.
"""
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from harness.schema import TruthFile


@dataclass(frozen=True)
class MatchPair:
    gt_card_id: str | None
    detection_id: str | None
    side_match: bool


def match_detections_to_truth(
    db_path: Path, truth: TruthFile, video_id: str,
) -> list[MatchPair]:
    detections = _load_detections(db_path, video_id)
    return _match(detections, truth.expected_cards)


def _load_detections(db_path: Path, video_id: str) -> list[dict]:
    # Surface owner reads src/card_capture/storage.py to identify the table(s)
    # that hold detections for a given video; this query is a placeholder of
    # shape. Replace `card_views` / column names with the actual schema.
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT instance_id, side, timestamp_ms "
            "FROM card_views WHERE video_id = ? ORDER BY timestamp_ms",
            (video_id,),
        ).fetchall()
    return [{"instance_id": r[0], "side": r[1], "timestamp_ms": r[2]} for r in rows]


def _match(detections, expected_cards) -> list[MatchPair]:
    # Implementation: greedy assignment by window overlap, then ordering
    # fallback. Surface owner writes this and gets the test green.
    ...  # write the implementation; the test enforces correctness
```

- [ ] **Step 4: Implement card_recall**

```python
# harness/metrics/card_recall.py
import json
from pathlib import Path

from harness.match import match_detections_to_truth
from harness.schema import TruthFile


def card_recall(*, db_path: Path, truth_path: Path, video_id: str) -> float | None:
    truth = TruthFile.model_validate_json(truth_path.read_text())
    if not truth.expected_cards:
        return None
    pairs = match_detections_to_truth(db_path, truth, video_id)
    matched = sum(1 for p in pairs if p.gt_card_id and p.detection_id)
    return matched / len(truth.expected_cards)
```

- [ ] **Step 5: Run tests — pass**

- [ ] **Step 6: Commit**

```bash
git add harness/match.py harness/metrics/__init__.py harness/metrics/card_recall.py tests/harness/
git commit -m "feat(harness): card_recall metric with detection↔truth matcher"
```

### Tasks D2.2 – D2.5: remaining metrics

Each follows the same TDD pattern as D2.1. For each metric:

1. Hand-author a fixture `cards.sqlite` + `truth.json` exercising the metric's edge cases.
2. Write the test with concrete expected values.
3. Run — fails.
4. Implement under `harness/metrics/<name>.py` as a pure function.
5. Run — passes.
6. Commit.

**Per-metric specifics:**

- **D2.2 `card_precision`**: similar to recall; fixture includes 1 phantom detection unmatched to any GT card.
- **D2.3 `side_accuracy`**: fixture includes one correct side assignment and one incorrect.
- **D2.4 `dedup_accuracy`**: fixture is a multi-video setup with `physical_card_key` ground truth and a predicted `dedup_groups` table. Use `sklearn.metrics.adjusted_rand_score` for ARI; implement pair F1 from scratch. Two metrics, returned as a `DedupAccuracy(ari: float, pair_f1: float)` dataclass.
- **D2.5 `image_quality`**: fixture includes 2 `final_cards` rows pointing to `tests/harness/fixtures/runs/.../crops/*.png` + a `reference_frames/*.png` pair. Use `skimage.metrics.structural_similarity` for SSIM and `peak_signal_noise_ratio` for PSNR. Return `ImageQuality(mean_ssim: float, mean_psnr: float, coverage: float)`.

Each task ends with a TDD-style commit. Five tasks total; can be done by five different agents in parallel within Surface D.

### Task D2.6: Aggregate runner

**Files:**
- Create: `harness/runner.py`
- Create: `tests/harness/test_runner.py`

- [ ] **Step 1: Failing test**

```python
# tests/harness/test_runner.py
from pathlib import Path
from harness.runner import run_metrics

def test_runner_returns_all_metrics():
    report = run_metrics(
        db_path=Path("tests/harness/fixtures/runs/all_matched/cards.sqlite"),
        truth_dir=Path("tests/harness/fixtures/runs/all_matched"),
        videos=["all_matched"],
    )
    assert "card_recall" in report.metrics
    assert "card_precision" in report.metrics
    assert "side_accuracy" in report.metrics
    assert "dedup_accuracy" in report.metrics
    assert "image_quality" in report.metrics
    assert report.per_video[0].video_id == "all_matched"
```

- [ ] **Step 2: Implement runner**

```python
# harness/runner.py
from dataclasses import dataclass, field
from pathlib import Path

from harness.metrics.card_recall import card_recall
from harness.metrics.card_precision import card_precision
from harness.metrics.side_accuracy import side_accuracy
from harness.metrics.dedup_accuracy import dedup_accuracy
from harness.metrics.image_quality import image_quality


@dataclass
class PerVideoReport:
    video_id: str
    metrics: dict[str, float | None | dict]


@dataclass
class Report:
    metrics: dict[str, float | None | dict]  # aggregate
    per_video: list[PerVideoReport] = field(default_factory=list)


def run_metrics(*, db_path: Path, truth_dir: Path, videos: list[str]) -> Report:
    per_video: list[PerVideoReport] = []
    for video_id in videos:
        truth_path = truth_dir / f"{video_id}.truth.json"
        # Some fixtures store truth.json alongside cards.sqlite; harness owner
        # picks one convention and writes the search logic.
        if not truth_path.exists():
            truth_path = truth_dir / "truth.json"
        m = {
            "card_recall": card_recall(db_path=db_path, truth_path=truth_path, video_id=video_id),
            "card_precision": card_precision(db_path=db_path, truth_path=truth_path, video_id=video_id),
            "side_accuracy": side_accuracy(db_path=db_path, truth_path=truth_path, video_id=video_id),
            "dedup_accuracy": dedup_accuracy(db_path=db_path, truth_path=truth_path, video_id=video_id),
            "image_quality": image_quality(db_path=db_path, truth_path=truth_path, video_id=video_id),
        }
        per_video.append(PerVideoReport(video_id=video_id, metrics=m))

    aggregate = _aggregate(per_video)
    return Report(metrics=aggregate, per_video=per_video)


def _aggregate(per_video: list[PerVideoReport]) -> dict:
    out = {}
    for key in ("card_recall", "card_precision", "side_accuracy"):
        vals = [pv.metrics[key] for pv in per_video if pv.metrics[key] is not None]
        out[key] = (sum(vals) / len(vals)) if vals else None
    # dedup + image_quality are dataclasses; aggregate by averaging fields.
    ...  # surface owner finishes the aggregation logic
    return out
```

- [ ] **Step 3: Run — passes**

- [ ] **Step 4: Commit**

```bash
git add harness/runner.py tests/harness/test_runner.py
git commit -m "feat(harness): aggregate metric runner returning per-video + summary"
```

---

## Phase D3 — Baseline + Regression Run Persistence (Wave 1)

### Task D3.1: Baseline read/write

**Files:**
- Create: `harness/baseline.py`
- Create: `tests/harness/test_baseline.py`

- [ ] **Step 1: Failing test**

```python
# tests/harness/test_baseline.py
import sqlite3
from pathlib import Path
import pytest

from harness.baseline import freeze_baseline, get_baseline, list_baselines
from migrations.run_migrations import apply_migrations

def test_freeze_and_get(tmp_path: Path):
    db = tmp_path / "cards.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    freeze_baseline(
        db_path=db,
        name="baseline_v4.1",
        code_sha="abc123",
        config={"detector": "docaligner"},
        metrics={"card_recall": 0.92},
        per_video=[{"video_id": "v1", "card_recall": 0.92}],
    )
    b = get_baseline(db_path=db, name="baseline_v4.1")
    assert b.metrics["card_recall"] == 0.92
    assert b.code_sha == "abc123"

def test_baseline_name_unique(tmp_path: Path):
    db = tmp_path / "cards.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    freeze_baseline(db_path=db, name="x", code_sha="a", config={}, metrics={}, per_video=[])
    with pytest.raises(Exception):
        freeze_baseline(db_path=db, name="x", code_sha="b", config={}, metrics={}, per_video=[])
```

- [ ] **Step 2: Implement**

```python
# harness/baseline.py
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Baseline:
    name: str
    code_sha: str
    config: dict
    metrics: dict
    per_video: list[dict]


def freeze_baseline(*, db_path: Path, name: str, code_sha: str,
                   config: dict, metrics: dict, per_video: list[dict]) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO regression_baselines(name, code_sha, config_json) VALUES (?, ?, ?)",
            (name, code_sha, json.dumps(config)),
        )
        baseline_id = cur.lastrowid
        conn.execute(
            "INSERT INTO regression_runs(baseline_id, code_sha, config_json, metrics_json, per_video_json) VALUES (?,?,?,?,?)",
            (baseline_id, code_sha, json.dumps(config), json.dumps(metrics), json.dumps(per_video)),
        )
        conn.commit()
        return baseline_id


def get_baseline(*, db_path: Path, name: str) -> Baseline:
    with sqlite3.connect(db_path) as conn:
        bid, code_sha, config_json = conn.execute(
            "SELECT baseline_id, code_sha, config_json FROM regression_baselines WHERE name = ?",
            (name,),
        ).fetchone()
        metrics_json, per_video_json = conn.execute(
            "SELECT metrics_json, per_video_json FROM regression_runs "
            "WHERE baseline_id = ? ORDER BY created_at ASC LIMIT 1",
            (bid,),
        ).fetchone()
    return Baseline(
        name=name, code_sha=code_sha,
        config=json.loads(config_json),
        metrics=json.loads(metrics_json),
        per_video=json.loads(per_video_json),
    )


def list_baselines(*, db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM regression_baselines ORDER BY name").fetchall()]
```

- [ ] **Step 3: Tests pass**

- [ ] **Step 4: Commit**

```bash
git add harness/baseline.py tests/harness/test_baseline.py
git commit -m "feat(harness): regression baseline read/write"
```

### Task D3.2: Regression run persistence

**Files:**
- Modify: `harness/baseline.py` (add `persist_run`)
- Create: `tests/harness/test_persist_run.py`

Same TDD pattern. `persist_run(db_path, baseline_name, metrics, per_video, code_sha, config) -> int` writes a `regression_runs` row. Return the run id.

---

## Phase D4 — Harness CLI (Wave 1)

### Task D4.1: `card-capture harness run`

**Files:**
- Create: `harness/cli.py`
- Modify: `src/card_capture/cli.py` (register `harness` subcommand group)
- Create: `tests/harness/test_cli.py`

- [ ] **Step 1: Failing test**

```python
# tests/harness/test_cli.py
import json
import subprocess
from pathlib import Path

def test_harness_run_against_baseline(tmp_path: Path):
    db = tmp_path / "cards.sqlite"
    # seed db + truth — surface owner picks fixture
    ...
    result = subprocess.run(
        ["card-capture", "harness", "run", "--baseline", "baseline_v4.1",
         "--db", str(db), "--truth-dir", "tests/harness/fixtures/runs/all_matched",
         "--videos", "all_matched", "--out", str(tmp_path / "report.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert "card_recall" in report["metrics"]
    assert "per_video" in report
```

- [ ] **Step 2: Implement CLI**

```python
# harness/cli.py
import json
import subprocess
from pathlib import Path
import click

from harness.baseline import get_baseline, persist_run
from harness.runner import run_metrics


@click.group()
def harness():
    """v4 regression harness."""


@harness.command()
@click.option("--baseline", required=True, help="baseline name to compare against")
@click.option("--db", required=True, type=click.Path(path_type=Path))
@click.option("--truth-dir", required=True, type=click.Path(path_type=Path))
@click.option("--videos", required=True, help="comma-separated video ids")
@click.option("--out", type=click.Path(path_type=Path), default=None)
def run(baseline, db, truth_dir, videos, out):
    """Run regression harness and compare to baseline."""
    video_ids = [v.strip() for v in videos.split(",") if v.strip()]
    report = run_metrics(db_path=db, truth_dir=truth_dir, videos=video_ids)
    b = get_baseline(db_path=db, name=baseline)
    deltas = _compute_deltas(report.metrics, b.metrics)
    code_sha = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    run_id = persist_run(db_path=db, baseline_name=baseline,
                        metrics=report.metrics,
                        per_video=[pv.__dict__ for pv in report.per_video],
                        code_sha=code_sha, config={})
    payload = {
        "run_id": run_id,
        "baseline": baseline,
        "metrics": report.metrics,
        "per_video": [pv.__dict__ for pv in report.per_video],
        "deltas": deltas,
    }
    if out:
        out.write_text(json.dumps(payload, indent=2, default=str))
    click.echo(json.dumps(deltas, indent=2, default=str))


def _compute_deltas(current: dict, baseline: dict) -> dict:
    out = {}
    for k, v in current.items():
        bv = baseline.get(k)
        if isinstance(v, (int, float)) and isinstance(bv, (int, float)):
            out[k] = v - bv
        else:
            out[k] = None
    return out
```

- [ ] **Step 3: Register in main CLI**

```python
# src/card_capture/cli.py — add near other subcommand registration
from harness.cli import harness as harness_group
cli.add_command(harness_group)
```

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add harness/cli.py src/card_capture/cli.py tests/harness/test_cli.py
git commit -m "feat(harness): card-capture harness run CLI"
```

### Task D4.2: Determinism check — same-input stability across 3 runs

**Files:**
- Create: `tests/harness/test_determinism.py`

- [ ] **Step 1: Failing test (smoke)**

```python
# tests/harness/test_determinism.py
import subprocess
import json
from pathlib import Path

def test_three_consecutive_runs_within_noise(tmp_path):
    out1 = tmp_path / "r1.json"; out2 = tmp_path / "r2.json"; out3 = tmp_path / "r3.json"
    for o in (out1, out2, out3):
        subprocess.run([
            "card-capture", "harness", "run",
            "--baseline", "baseline_v4.1",
            "--db", "tests/harness/fixtures/runs/all_matched/cards.sqlite",
            "--truth-dir", "tests/harness/fixtures/runs/all_matched",
            "--videos", "all_matched",
            "--out", str(o),
        ], check=True)
    a = json.loads(out1.read_text())["metrics"]
    b = json.loads(out2.read_text())["metrics"]
    c = json.loads(out3.read_text())["metrics"]
    for k in ("card_recall", "card_precision", "side_accuracy"):
        if a[k] is None: continue
        assert abs(a[k] - b[k]) < 0.01
        assert abs(b[k] - c[k]) < 0.01
```

- [ ] **Step 2: Implement until stable**

If the test fails (metrics jitter beyond noise floor), the harness owner identifies the source of non-determinism in the pipeline (likely thread-scheduling in Stages 1–3, or random seeds in scoring). File issues against Surface A; do NOT band-aid metrics with rounding. The Wave 1 acceptance criterion is real determinism.

- [ ] **Step 3: Commit**

```bash
git add tests/harness/test_determinism.py
git commit -m "test(harness): three-run determinism gate"
```

### Task D4.3: Freeze `baseline_v4.1`

**Files:** none new — invokes `harness.baseline.freeze_baseline` via the CLI.

- [ ] **Step 1: Add `card-capture harness baseline freeze` subcommand**

```python
# harness/cli.py — add subcommand
@harness.group()
def baseline():
    """Manage regression baselines."""

@baseline.command("freeze")
@click.option("--name", required=True)
@click.option("--db", required=True, type=click.Path(path_type=Path))
@click.option("--truth-dir", required=True, type=click.Path(path_type=Path))
@click.option("--videos", required=True)
def freeze(name, db, truth_dir, videos):
    """Run metrics on the CURRENT pipeline and freeze the result as a baseline."""
    video_ids = [v.strip() for v in videos.split(",") if v.strip()]
    report = run_metrics(db_path=db, truth_dir=truth_dir, videos=video_ids)
    code_sha = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    freeze_baseline(
        db_path=db, name=name, code_sha=code_sha, config={},
        metrics=report.metrics,
        per_video=[pv.__dict__ for pv in report.per_video],
    )
    click.echo(f"frozen: {name} @ {code_sha}")
```

- [ ] **Step 2: Freeze baseline against the pre-Surface-A-refactor pipeline**

```
card-capture harness baseline freeze \
  --name baseline_v4.1 \
  --db cards.sqlite \
  --truth-dir golden_set/videos \
  --videos $(cat golden_set/videos/_index.txt | paste -sd,)
```

- [ ] **Step 3: Commit baseline freeze (in db + tag)**

```bash
git tag -a v4-baseline-frozen -m "baseline_v4.1 frozen against pre-Surface-A pipeline"
```

---

## Phase D5 — Hard-Case Capture Wire-Up (Wave 1)

### Task D5.1: Hard-case writer

**Files:**
- Create: `harness/hard_cases.py`
- Create: `tests/harness/test_hard_cases.py`
- Modify: `src/card_capture/analysis/hard_case_capture.py` (additional DB write)

- [ ] **Step 1: Failing test**

```python
# tests/harness/test_hard_cases.py
import sqlite3
from pathlib import Path

from harness.hard_cases import record_hard_case
from migrations.run_migrations import apply_migrations

def test_record_hard_case_persists(tmp_path):
    db = tmp_path / "cards.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    case_id = record_hard_case(
        db_path=db, run_id="r1", frame_index=42, stage_id="score",
        reason="border_purity<0.2", thumbnail_path="t.png", source_frame_path="s.png",
    )
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT reason FROM hard_cases WHERE case_id = ?", (case_id,)).fetchone()
    assert row[0] == "border_purity<0.2"
```

- [ ] **Step 2: Implement**

```python
# harness/hard_cases.py
import sqlite3
from pathlib import Path


def record_hard_case(*, db_path: Path, run_id: str, frame_index: int,
                    stage_id: str, reason: str, thumbnail_path: str,
                    source_frame_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO hard_cases(run_id, frame_index, stage_id, reason, thumbnail_path, source_frame_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, frame_index, stage_id, reason, thumbnail_path, source_frame_path),
        )
        conn.commit()
        return cur.lastrowid
```

- [ ] **Step 3: Wire existing hard_case_capture.py**

The surface owner reads `src/card_capture/analysis/hard_case_capture.py` and adds a call to `harness.hard_cases.record_hard_case` at the same point where it currently writes the JSON file (so JSON + DB both populated). The existing behavior is preserved; the DB write is additive.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add harness/hard_cases.py src/card_capture/analysis/hard_case_capture.py tests/harness/test_hard_cases.py
git commit -m "feat(harness): hard cases persisted to DB in addition to JSON"
```

---

## Phase D6 — Regression API (Wave 1; depends on Surface A app shell)

### Task D6.1: Regression service

**Files:**
- Create: `app/services/regression_service.py`
- Create: `tests/app/test_regression_endpoints.py`

- [ ] **Step 1: Failing endpoint test**

```python
# tests/app/test_regression_endpoints.py
from fastapi.testclient import TestClient
from app.main import create_app

def test_get_baselines():
    client = TestClient(create_app())
    r = client.get("/api/v1/regression/baselines")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_compare_two_runs():
    client = TestClient(create_app())
    r = client.get("/api/v1/regression/compare?a=baseline_v4.1&b=baseline_v4.1")
    # comparing a baseline to itself: deltas all zero
    assert r.status_code == 200
    body = r.json()
    for k, v in body.get("deltas", {}).items():
        if isinstance(v, (int, float)):
            assert v == 0.0
```

- [ ] **Step 2: Implement service**

```python
# app/services/regression_service.py
from pathlib import Path
from harness.baseline import get_baseline, list_baselines


class RegressionService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def list_baselines(self) -> list[dict]:
        return [{"name": n} for n in list_baselines(db_path=self.db_path)]

    def compare(self, a: str, b: str) -> dict:
        ba = get_baseline(db_path=self.db_path, name=a)
        bb = get_baseline(db_path=self.db_path, name=b)
        deltas = {}
        for k in set(ba.metrics) | set(bb.metrics):
            va, vb = ba.metrics.get(k), bb.metrics.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                deltas[k] = vb - va
            else:
                deltas[k] = None
        return {"a": a, "b": b, "deltas": deltas}
```

- [ ] **Step 3: Wire route handlers**

```python
# app/api/regression.py — replace stubs
from fastapi import APIRouter, Request

router = APIRouter()


def _service(request: Request):
    return request.app.state.regression_service


@router.get("/baselines")
def list_baselines(request: Request):
    return _service(request).list_baselines()


@router.get("/compare")
def compare(a: str, b: str, request: Request):
    return _service(request).compare(a, b)
```

In `app/main.py`, add `app.state.regression_service = RegressionService(db_path=Path("cards.sqlite"))` (path resolved from settings later).

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add app/services/regression_service.py app/api/regression.py app/main.py tests/app/test_regression_endpoints.py
git commit -m "feat(app): regression service + endpoints"
```

---

## Phase D7 — Label Endpoints (Wave 1 backend; Wave 2 consumed by Surface B)

### Task D7.1: Labeling service

**Files:**
- Create: `app/services/labeling_service.py`
- Create: `tests/app/test_label_endpoints.py`
- Modify: `app/api/label.py`

- [ ] **Step 1: Failing endpoint tests**

```python
# tests/app/test_label_endpoints.py
from fastapi.testclient import TestClient
from app.main import create_app

def test_get_put_truth_roundtrip():
    client = TestClient(create_app())
    payload = {
        "video_id": "v_abc",
        "schema_version": 1,
        "expected_cards": [{
            "card_id": "c1", "front_present": True, "back_present": False,
            "physical_card_key": "k1", "is_foil": False,
        }],
    }
    put = client.put("/api/v1/label/truth/v_abc", json=payload)
    assert put.status_code in (200, 204)
    got = client.get("/api/v1/label/truth/v_abc")
    assert got.status_code == 200
    assert got.json()["video_id"] == "v_abc"

def test_post_fb_label_persists():
    client = TestClient(create_app())
    r = client.post("/api/v1/label/fb", json={
        "instance_id": "inst_1", "frame_index": 100, "side": "front",
    })
    assert r.status_code in (200, 201)

def test_get_clusters_returns_list():
    client = TestClient(create_app())
    r = client.get("/api/v1/label/clusters")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
```

- [ ] **Step 2: Implement service**

```python
# app/services/labeling_service.py
import json
import sqlite3
from pathlib import Path

from harness.schema import TruthFile


class LabelingService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def get_truth(self, video_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM truth_files WHERE video_id = ?", (video_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_truth(self, video_id: str, payload: dict) -> None:
        # validate before writing
        tf = TruthFile.model_validate(payload)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO truth_files(video_id, schema_version, payload_json) VALUES (?, ?, ?) "
                "ON CONFLICT(video_id) DO UPDATE SET schema_version=excluded.schema_version, "
                "payload_json=excluded.payload_json, updated_at=datetime('now')",
                (video_id, tf.schema_version, tf.model_dump_json()),
            )
            conn.commit()

    def post_fb_label(self, instance_id: str, frame_index: int, side: str,
                     labeler: str | None = None, source_run_id: int | None = None) -> int:
        if side not in ("front", "back", "uncertain"):
            raise ValueError(f"invalid side: {side}")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO fb_labels(source_run_id, instance_id, frame_index, side, labeler) "
                "VALUES (?, ?, ?, ?, ?)",
                (source_run_id, instance_id, frame_index, side, labeler),
            )
            conn.commit()
            return cur.lastrowid

    def next_fb_candidate(self) -> dict | None:
        """Return one unlabeled high-confidence detection for the trainer UI."""
        # Surface owner reads storage.py to identify the detection table and
        # join against fb_labels to filter out already-labeled instances. The
        # query selects the highest-confidence un-labeled detection.
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT cv.instance_id, cv.frame_index, cv.image_path "
                "FROM card_views cv "
                "LEFT JOIN fb_labels fl ON fl.instance_id = cv.instance_id "
                "WHERE fl.label_id IS NULL "
                "ORDER BY cv.confidence DESC LIMIT 1",
            ).fetchone()
        if not row:
            return None
        return {"instance_id": row[0], "frame_index": row[1], "image_path": row[2]}

    def list_clusters(self, status: str | None = None) -> list[dict]:
        sql = "SELECT cluster_id, predicted_member_ids_json, confirmed_member_ids_json, status, updated_at FROM dedup_clusters"
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"; params = (status,)
        sql += " ORDER BY cluster_id"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {"cluster_id": r[0], "predicted": json.loads(r[1]),
             "confirmed": json.loads(r[2]) if r[2] else None,
             "status": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def patch_cluster(self, cluster_id: int, *, status: str | None = None,
                     confirmed: list[str] | None = None) -> None:
        sets, params = [], []
        if status is not None:
            sets.append("status = ?"); params.append(status)
        if confirmed is not None:
            sets.append("confirmed_member_ids_json = ?"); params.append(json.dumps(confirmed))
        sets.append("updated_at = datetime('now')")
        params.append(cluster_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE dedup_clusters SET {', '.join(sets)} WHERE cluster_id = ?", params)
            conn.commit()
```

- [ ] **Step 3: Wire routes**

```python
# app/api/label.py — replace stubs
from fastapi import APIRouter, Request, HTTPException
from app.schemas.v1 import LabelTruth, LabelFB, DedupCluster

router = APIRouter()


def _svc(request: Request):
    return request.app.state.labeling_service


@router.get("/truth/{video_id}", response_model=LabelTruth | None)
def get_truth(video_id: str, request: Request):
    payload = _svc(request).get_truth(video_id)
    return payload


@router.put("/truth/{video_id}", status_code=204)
def put_truth(video_id: str, body: dict, request: Request):
    if body.get("video_id") and body["video_id"] != video_id:
        raise HTTPException(status_code=400, detail="video_id mismatch")
    body["video_id"] = video_id
    _svc(request).put_truth(video_id, body)


@router.get("/fb/next")
def get_next_fb(request: Request):
    return _svc(request).next_fb_candidate() or {}


@router.post("/fb", status_code=201)
def post_fb(body: LabelFB, request: Request):
    label_id = _svc(request).post_fb_label(
        instance_id=body.instance_id, frame_index=body.frame_index, side=body.side,
        labeler=getattr(body, "labeler", None),
    )
    return {"label_id": label_id}


@router.get("/clusters")
def list_clusters(request: Request, status: str | None = None):
    return _svc(request).list_clusters(status=status)


@router.patch("/clusters/{cluster_id}", status_code=204)
def patch_cluster(cluster_id: int, body: dict, request: Request):
    _svc(request).patch_cluster(
        cluster_id, status=body.get("status"), confirmed=body.get("confirmed"),
    )
```

In `app/main.py`, add `app.state.labeling_service = LabelingService(db_path=Path("cards.sqlite"))`.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add app/services/labeling_service.py app/api/label.py app/main.py tests/app/test_label_endpoints.py
git commit -m "feat(app): label endpoints backed by LabelingService"
```

---

## Phase D8 — Golden Set Bootstrap (Wave 1)

### Task D8.1: Golden-set directory + 5 bootstrap videos

**Files:**
- Create: `golden_set/README.md`
- Create: `golden_set/videos/_index.txt`
- Create: `golden_set/videos/<id>/truth.json` for 5 bootstrap videos (manually labeled by the user).

- [ ] **Step 1: README**

```markdown
# Golden Set

Labeled videos used by the regression harness. Each subdirectory under
`videos/` is one labeled video:

```
videos/<video_id>/
    truth.json           # Contract 4 schema
    source.mov.symlink   # symlink or note pointing to source video
    reference_frames/    # hand-picked reference frames for SSIM
```

`_index.txt` lists all labeled video ids, one per line. The harness CLI
reads this file when `--videos` is omitted.

Coverage targets:
- 15 total videos by Wave 1 gate.
- Must include: clean run, glare, foil, hand occlusion, fast swaps,
  edge-on flips, dark workspace, bright workspace, mixed orientations,
  partial visibility, multi-card-in-frame.
```

- [ ] **Step 2: Label 5 bootstrap videos**

Surface D's owner labels 5 videos using the current `templates/labeling.html`. Convert each via `harness.schema.from_legacy` to schema_version=1 and commit under `golden_set/videos/<id>/`.

This step is human work; the agent prepares the directory structure and conversion script.

- [ ] **Step 3: Commit**

```bash
git add golden_set/
git commit -m "feat(harness): bootstrap 5 labeled videos for regression baseline"
```

---

## Phase D9 — Wave 2 Plan Stub

**Status:** Outline only. Re-plan when Wave 1 gate is green AND Surface B's label UX is ready.

Wave 2 Surface-D tasks:

- **D9.1** Grow golden set from 5 → 15. Coverage matrix per D8.1 README.
- **D9.2** Recalibrate noise floor after 3 baseline runs.
- **D9.3** Training-set export endpoints: `POST /training/datasets/from_truth`, `POST /training/datasets/from_fb_labels`.
- **D9.4** Cross-video dedup ground-truth join (via `physical_card_key`) for full-system dedup metric.

## Phase D10 — Wave 3 Plan Stub

- **D10.1** Promote-to-baseline endpoint (`POST /regression/baselines/promote`) gated on no-regression.
- **D10.2** A/B comparison API for B's playground (`POST /regression/run/playground`).
- **D10.3** Hard-case → training-set one-click endpoint.

---

## Self-Review (post-write)

- **Spec coverage:** Contract 4 (§2.2 C4) → Phase D0; truth schema → Phase D1; metrics → Phase D2; regression baselines/runs → Phase D3; harness CLI → Phase D4; hard cases → Phase D5; regression API → Phase D6; label endpoints → Phase D7; golden set → Phase D8.
- **Placeholders:** every step either contains code or names a specific file the surface owner reads to fill in details (e.g. `templates/labeling.html` for legacy adapter). No "TBD" or "add appropriate handling."
- **Type consistency:** `TruthFile`, `ExpectedCard`, `Baseline`, `Report`, `PerVideoReport`, `MatchPair`, `RegressionService`, `LabelingService` are referenced consistently.

---

## Execution Handoff

Plan complete at `docs/superpowers/plans/2026-05-12-v4-surface-d-harness.md`.

Surface D's contract-drafting tasks (D0.*) ack-block Surface A's Wave 1 freeze; metric implementation (D2.*) is parallelizable across five sub-agents within Surface D. Phase D6 (regression API) depends on Surface A's app factory existing (Task A3.3).

Wave 2 and Wave 3 phases (D9, D10) re-plan via this same skill when those waves begin.
