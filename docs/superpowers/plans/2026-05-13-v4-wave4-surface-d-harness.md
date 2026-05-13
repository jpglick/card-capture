# Wave 4 — Surface D (Harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the harness so a baseline frozen by `harness/cli.py` is reproducible (config snapshotted, metrics homogeneous, truth-file naming canonical, bootstrap data tracked or regenerable).

**Architecture:** Single agent, ~4 PRs. Surface D owns harness/, golden_set/, tests/harness/, scripts/generate_reference_frames.py. D1 blocked-by A2 (single canonical config dataclass). D2/D3/D4 only blocked-by E1.

**Tech Stack:** Pydantic v2, pytest, click (CLI), Python 3.11.

**Spec:** `docs/superpowers/specs/2026-05-13-v4-wave4-hardening-design.md` §7.

**Files owned by Surface D:** `harness/**`, `golden_set/**`, `tests/harness/**`, `scripts/generate_reference_frames.py`. Single doc touch on `docs/contracts/truth-schema.md` for D3.

---

## Pre-flight

- [ ] **P1: Confirm E1 is merged; check A2 status for D1**

```bash
git fetch origin main
ls .github/workflows/test.yml
grep -n "class Options" src/card_capture/pipeline.py src/card_capture/config.py
```

Expected: workflow exists. For D1 specifically: `src/card_capture/pipeline.py` no longer defines `Options` (only `config.py` does). If A2 isn't merged, do D2/D3/D4 first and come back to D1.

- [ ] **P2: Create the worktree**

```bash
git worktree add ../card-capture-wave4-d -b wave4/d-harness origin/main
cd ../card-capture-wave4-d
pip install -e ".[harness,test]"
python -m pytest tests/ -q
```

Expected: tests pass.

---

## Task 1: D2 — Unify metric return types

(Done before D1 because D1 reads from the metric Report.)

**Files:**
- Create: `harness/metrics/types.py`
- Modify: `harness/metrics/card_recall.py`, `card_precision.py`, `side_accuracy.py`, `dedup_accuracy.py`, `image_quality.py`
- Modify: `harness/runner.py`
- Modify: `harness/cli.py` (compute_deltas)
- Create: `tests/harness/test_runner_roundtrip.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/harness/test_runner_roundtrip.py`:

```python
"""The harness Report must JSON-serialise cleanly.

Closes V4_CONCERNS §1.11.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.runner import run_metrics
from harness.metrics.types import MetricResult

FIXTURE_DIR = Path("tests/harness/fixtures/runs/all_matched")
TRUTH_DIR = Path("tests/harness/fixtures/truth_examples")


def test_metric_result_is_serialisable():
    m = MetricResult(name="card_recall", value=0.92, extras={"matched": 23})
    s = json.dumps(m.model_dump())
    back = MetricResult.model_validate_json(s)
    assert back == m


def test_metric_result_handles_none_value():
    m = MetricResult(name="image_quality", value=None, extras={"reason": "no_reference"})
    s = json.dumps(m.model_dump())
    back = MetricResult.model_validate_json(s)
    assert back.value is None


def test_run_metrics_report_roundtrips_through_json(tmp_path):
    """A full Report from run_metrics serialises to JSON and back."""
    db_path = FIXTURE_DIR / "cards.sqlite"
    if not db_path.exists():
        pytest.skip("fixture DB not present")

    report = run_metrics(
        db_path=db_path,
        truth_dir=TRUTH_DIR,
        videos=["minimal_valid"],
    )

    # Serialise the whole report. Every metric must be a MetricResult or
    # a plain dict.
    payload = {
        "metrics": {k: v.model_dump() if isinstance(v, MetricResult) else v
                    for k, v in report.metrics.items()},
        "per_video": [
            {
                "video_id": pv.video_id,
                "metrics": {
                    k: v.model_dump() if isinstance(v, MetricResult) else v
                    for k, v in pv.metrics.items()
                },
            }
            for pv in report.per_video
        ],
    }
    s = json.dumps(payload)
    back = json.loads(s)
    assert "metrics" in back
    assert "per_video" in back
```

- [ ] **Step 1.2: Run the tests — expect FAIL (module missing)**

```bash
pytest tests/harness/test_runner_roundtrip.py -v
```

Expected: ImportError on `harness.metrics.types`.

- [ ] **Step 1.3: Define `MetricResult`**

Create `harness/metrics/types.py`:

```python
"""Shared metric result type.

Every harness metric function returns a `MetricResult`; aggregate
across videos by averaging `value` and merging `extras`.

Closes V4_CONCERNS §1.11.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class MetricResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: float | None
    extras: dict[str, Any] = {}
```

- [ ] **Step 1.4: Migrate each metric function**

For each of `card_recall.py`, `card_precision.py`, `side_accuracy.py`:

```python
# BEFORE: def card_recall(...) -> float | None: return 0.9
# AFTER:
from .types import MetricResult

def card_recall(...) -> MetricResult:
    value = ...  # existing computation
    return MetricResult(name="card_recall", value=value, extras={...})
```

For `dedup_accuracy.py` and `image_quality.py` — currently they return
their own dataclasses (`DedupAccuracy`, `ImageQuality`). Convert:

```python
# BEFORE: returns DedupAccuracy(ari=0.85, ...)
# AFTER:
return MetricResult(
    name="dedup_accuracy",
    value=ari,
    extras={"n_clusters_predicted": k_pred, "n_clusters_truth": k_true},
)
```

Delete the `DedupAccuracy` and `ImageQuality` dataclasses (or keep as
internal helpers if other code still imports them — but the public
return type is `MetricResult`).

- [ ] **Step 1.5: Update `harness/runner.py`**

Replace any `float | None | DedupAccuracy | ImageQuality` type hints
with `MetricResult`. Aggregate by averaging `MetricResult.value` (skip
None values). Stash per-video extras in a flat dict.

- [ ] **Step 1.6: Update `harness/cli.py:_compute_deltas`**

```python
def _compute_deltas(
    current: dict[str, MetricResult],
    baseline: dict[str, MetricResult],
) -> dict[str, float | None]:
    deltas = {}
    for name, cur in current.items():
        base = baseline.get(name)
        if cur.value is None or base is None or base.value is None:
            deltas[name] = None
            continue
        deltas[name] = cur.value - base.value
    return deltas
```

- [ ] **Step 1.7: Run the tests — expect PASS**

```bash
pytest tests/harness/test_runner_roundtrip.py -v
python -m pytest tests/ -q
```

Expected: green. The pre-existing metric unit tests in
`tests/harness/test_metrics_*.py` will need their `.value`/`.extras`
expectations updated — that's part of this PR.

- [ ] **Step 1.8: Commit and open PR**

```bash
git add harness/ tests/harness/
git commit -m "refactor(harness): unify metric return types as MetricResult

Replaces the mixed float | None | DedupAccuracy | ImageQuality return
types with a single Pydantic MetricResult model. runner.py and
cli.py:_compute_deltas updated. Round-trip test confirms the Report
serialises cleanly to JSON.

Closes V4_CONCERNS §1.11.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push -u origin wave4/d-harness
gh pr create --title "[Wave 4 — Surface D] Unify metric return types (D2)" --body "$(cat <<'EOF'
## Summary
- New harness.metrics.types.MetricResult — every metric returns this.
- runner.py + cli.py handle the uniform type; persistence is JSON-safe.
- Round-trip test guards against future regressions.

Closes V4_CONCERNS §1.11.

## Test plan
- [x] new test: tests/harness/test_runner_roundtrip.py
- [x] existing tests/harness/test_metrics_*.py updated to assert on MetricResult shape
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 2: D3 — Canonical truth-file naming

**Files:**
- Modify: `harness/runner.py`
- Modify: `docs/contracts/truth-schema.md`
- Modify: `tests/contracts/test_drift.py` (extend drift gate)
- Rename: `golden_set/videos/IMG_5872/truth.json` → `golden_set/videos/IMG_5872.truth.json`
- Modify: `golden_set/videos/_index.txt`

- [ ] **Step 2.1: Rebase**

```bash
git fetch origin main && git rebase origin/main
```

- [ ] **Step 2.2: Write the failing test**

Append to `tests/harness/test_validator.py` (or create a new
`tests/harness/test_runner_truth_resolution.py`):

```python
"""Canonical truth-file naming: <truth_dir>/<video_id>.truth.json.

Closes V4_CONCERNS §1.13.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from harness.runner import _find_truth


def test_canonical_naming_resolves_without_warning(tmp_path: Path):
    canonical = tmp_path / "video_42.truth.json"
    canonical.write_text("{}")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        path = _find_truth(tmp_path, "video_42")

    assert path == canonical


def test_legacy_directory_naming_emits_deprecation_warning(tmp_path: Path):
    legacy = tmp_path / "video_42" / "truth.json"
    legacy.parent.mkdir()
    legacy.write_text("{}")

    with pytest.warns(DeprecationWarning, match="video_42.truth.json"):
        path = _find_truth(tmp_path, "video_42")

    assert path == legacy


def test_single_truth_json_emits_deprecation_warning(tmp_path: Path):
    single = tmp_path / "truth.json"
    single.write_text("{}")

    with pytest.warns(DeprecationWarning):
        path = _find_truth(tmp_path, "video_42")

    assert path == single
```

- [ ] **Step 2.3: Run the tests — expect FAIL**

```bash
pytest tests/harness/test_runner_truth_resolution.py -v
```

Expected: fail; current `_find_truth` doesn't emit warnings.

- [ ] **Step 2.4: Update `harness/runner.py:_find_truth`**

```python
import warnings

def _find_truth(truth_dir: Path, video_id: str) -> Path:
    """Resolve the truth file for a video.

    Canonical naming: <truth_dir>/<video_id>.truth.json (Wave 4+).

    Legacy fallbacks (emit DeprecationWarning; removed in Wave 5):
    - <truth_dir>/<video_id>/truth.json
    - <truth_dir>/truth.json
    """
    canonical = truth_dir / f"{video_id}.truth.json"
    if canonical.exists():
        return canonical

    legacy_dir = truth_dir / video_id / "truth.json"
    if legacy_dir.exists():
        warnings.warn(
            f"Legacy truth file at {legacy_dir}; rename to "
            f"{canonical} (removed in Wave 5).",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy_dir

    single = truth_dir / "truth.json"
    if single.exists():
        warnings.warn(
            f"Legacy single-file truth at {single}; rename to "
            f"{canonical} (removed in Wave 5).",
            DeprecationWarning,
            stacklevel=2,
        )
        return single

    raise FileNotFoundError(f"No truth file found for {video_id} in {truth_dir}")
```

- [ ] **Step 2.5: Rename the IMG_5872 truth file**

```bash
git mv golden_set/videos/IMG_5872/truth.json \
       golden_set/videos/IMG_5872.truth.json
```

If `golden_set/videos/IMG_5872/` has other content (reference frames),
leave the directory; only the `truth.json` moves out.

Update `golden_set/videos/_index.txt` if it references the old path.

- [ ] **Step 2.6: Document the convention in Contract 4**

Edit `docs/contracts/truth-schema.md`. Add a new section (place after
"Top-level shape"):

```markdown
## File naming

Canonical: `<truth_dir>/<video_id>.truth.json` (flat, sortable).

The runner accepts two legacy forms with a `DeprecationWarning`, to be
removed in Wave 5:

- `<truth_dir>/<video_id>/truth.json`
- `<truth_dir>/truth.json` (single-video fixtures only)

New truth files MUST use the canonical form.
```

- [ ] **Step 2.7: Extend the drift gate (E3 follow-up)**

Edit `tests/contracts/test_drift.py`. Append:

```python
def test_truth_schema_documents_canonical_naming():
    """Contract 4 must document the canonical .truth.json naming."""
    contract = _read(CONTRACTS_DIR / "truth-schema.md")
    assert "<video_id>.truth.json" in contract or ".truth.json" in contract, (
        "truth-schema.md must document the canonical .truth.json naming"
    )
```

- [ ] **Step 2.8: Run the tests — expect PASS**

```bash
pytest tests/harness/ tests/contracts/ -v
python -m pytest tests/ -q
```

Expected: green.

- [ ] **Step 2.9: Commit and open PR**

```bash
git add harness/runner.py \
        docs/contracts/truth-schema.md \
        tests/harness/test_runner_truth_resolution.py \
        tests/contracts/test_drift.py \
        golden_set/videos/IMG_5872.truth.json \
        golden_set/videos/_index.txt
git rm golden_set/videos/IMG_5872/truth.json 2>/dev/null || true
git commit -m "refactor(harness): canonical truth-file naming

Canonical: <truth_dir>/<video_id>.truth.json. Legacy directory and
single-file forms emit DeprecationWarning; removed in Wave 5.

IMG_5872 truth file renamed to follow the convention. Contract 4
documents the convention; drift gate covers the doc.

Closes V4_CONCERNS §1.13.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface D] Canonical truth-file naming (D3)" --body "$(cat <<'EOF'
## Summary
- Canonical: <truth_dir>/<video_id>.truth.json.
- Two legacy forms now emit DeprecationWarning.
- IMG_5872 truth file renamed.
- Contract 4 documents the convention; drift gate covers it.

Closes V4_CONCERNS §1.13.

## Test plan
- [x] new tests: test_runner_truth_resolution (3 tests), drift gate updated
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 3: D4 — Golden-set + `harness_config.json` policy

**Files:**
- Move: `harness_config.json` (repo root) → `harness/config.example.json`
- Create: `harness/config.py`
- Modify: `.gitignore`
- Modify: `golden_set/README.md`
- Track: `scripts/generate_reference_frames.py` (was untracked)
- Modify: `tests/harness/test_config.py` (new)

- [ ] **Step 3.1: Rebase**

```bash
git fetch origin main && git rebase origin/main
```

- [ ] **Step 3.2: Read existing `harness_config.json` content**

```bash
cat harness_config.json
```

Determine the schema. Typical fields: `baseline_name`, `truth_dir`,
`db_path`, `videos`. Use this to write the Pydantic model below.

- [ ] **Step 3.3: Write the failing test**

Create `tests/harness/test_config.py`:

```python
"""Tests for harness.config.HarnessConfig.

Closes V4_CONCERNS §1.4.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config import HarnessConfig, load_config


def test_config_loads_from_example(tmp_path: Path):
    """The committed example file is valid HarnessConfig."""
    example = Path("harness/config.example.json")
    if not example.exists():
        pytest.skip("example not yet committed")
    HarnessConfig.model_validate_json(example.read_text())


def test_config_load_real_file(tmp_path: Path):
    payload = {
        "baseline_name": "baseline_v4",
        "truth_dir": "golden_set/videos",
        "db_path": "cards.sqlite",
        "videos": ["IMG_5872"],
    }
    cfg_path = tmp_path / "harness_config.json"
    cfg_path.write_text(json.dumps(payload))
    cfg = load_config(cfg_path)
    assert cfg.baseline_name == "baseline_v4"
    assert cfg.videos == ["IMG_5872"]


def test_config_rejects_unknown_field(tmp_path: Path):
    payload = {"baseline_name": "b", "truth_dir": "t", "db_path": "d", "vidos": []}
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps(payload))
    with pytest.raises(Exception):
        load_config(cfg_path)
```

- [ ] **Step 3.4: Run the tests — expect FAIL**

```bash
pytest tests/harness/test_config.py -v
```

Expected: ImportError on `harness.config`.

- [ ] **Step 3.5: Create `harness/config.py`**

```python
"""Harness runtime config.

Real config files live next to the SQLite DB and are gitignored. The
committed `harness/config.example.json` documents the schema.

Closes V4_CONCERNS §1.4.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields

    baseline_name: str
    truth_dir: str
    db_path: str
    videos: list[str] = []


def load_config(path: Path) -> HarnessConfig:
    """Load and validate a harness config JSON file."""
    return HarnessConfig.model_validate_json(Path(path).read_text())
```

- [ ] **Step 3.6: Move the current file**

```bash
git mv harness_config.json harness/config.example.json
```

Verify the moved file still validates:

```bash
python -c "from harness.config import HarnessConfig; HarnessConfig.model_validate_json(open('harness/config.example.json').read())"
```

If validation fails, either (a) edit `config.example.json` to match the
schema, or (b) widen the schema. Prefer (a) — the example should reflect
the intended shape.

- [ ] **Step 3.7: Update `.gitignore`**

Append to the existing `.gitignore`:

```
# Harness runtime config — example lives at harness/config.example.json.
/harness_config.json

# Golden-set reference frames are regenerable via
# scripts/generate_reference_frames.py.
golden_set/**/reference_frames/
```

- [ ] **Step 3.8: Commit `scripts/generate_reference_frames.py`**

```bash
git add scripts/generate_reference_frames.py
```

Read the file first. If it uses a deterministic seed, leave as-is. If
it uses a random seed each run, add `seed=42` (or a CLI flag with that
default) so reference frames are reproducible.

- [ ] **Step 3.9: Document in `golden_set/README.md`**

Replace or expand the existing README:

```markdown
# Golden Set

Curated videos + truth files used by the regression harness.

## Layout

```
golden_set/
├── README.md
├── videos/
│   ├── _index.txt              # list of video_id's, one per line
│   ├── <video_id>.truth.json   # truth file per video (Contract 4)
│   └── <video_id>/
│       └── reference_frames/   # gitignored; regenerable
└── ...
```

## Truth files

Canonical naming: `<video_id>.truth.json`. See
`docs/contracts/truth-schema.md` for shape and validation rules.

## Reference frames

Reference frames are inputs to the image-quality (SSIM) metric. They
are *regenerable* via `scripts/generate_reference_frames.py` and
therefore gitignored. Re-generate with:

```bash
python scripts/generate_reference_frames.py \
    --video <path-to-video.mov> \
    --video-id <video_id> \
    --out golden_set/videos/<video_id>/reference_frames/
```

The script uses a fixed random seed so frames are reproducible across
machines.

## Harness config

Runtime config (paths, baseline name, video list) lives in a file next
to the SQLite DB — **not** in this directory. See
`harness/config.example.json`.
```

- [ ] **Step 3.10: Run the tests — expect PASS**

```bash
pytest tests/harness/test_config.py -v
python -m pytest tests/ -q
```

Expected: green.

- [ ] **Step 3.11: Commit and open PR**

```bash
git add harness/config.py harness/config.example.json \
        tests/harness/test_config.py \
        .gitignore golden_set/README.md \
        scripts/generate_reference_frames.py
git commit -m "feat(harness): config schema, golden_set policy, regenerable frames

- Moves harness_config.json from repo root to harness/config.example.json.
- New harness.config.HarnessConfig (Pydantic) validates real config files.
- Reference frames gitignored + regenerable via scripts/generate_reference_frames.py.
- README documents the layout.

Closes V4_CONCERNS §1.3, §1.4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface D] Harness config + golden-set policy (D4)" --body "$(cat <<'EOF'
## Summary
- harness_config.json moved out of repo root into harness/config.example.json + Pydantic schema.
- Reference frames gitignored, regenerable via scripts/generate_reference_frames.py.
- golden_set/README.md documents the layout and regen command.

Closes V4_CONCERNS §1.3, §1.4.

## Test plan
- [x] new test: tests/harness/test_config.py (3 tests)
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 4: D1 — `harness/cli.py` config-loading

(Depends on A2 — single canonical Options dataclass. Run last.)

**Files:**
- Modify: `harness/config.py` (add `load_pipeline_config`)
- Modify: `harness/cli.py` (replace `config={}` placeholders)
- Create: `tests/harness/test_cli_config.py`

- [ ] **Step 4.1: Confirm A2 merged**

```bash
git fetch origin main && git rebase origin/main
python -c "from card_capture.config import Options; print(Options)"
grep "class Options" src/card_capture/pipeline.py || echo "✓ no duplicate Options in pipeline.py"
```

Expected: import succeeds; no duplicate dataclass.

- [ ] **Step 4.2: Write the failing test**

Create `tests/harness/test_cli_config.py`:

```python
"""harness.cli must persist a non-empty config snapshot with every baseline.

Closes V4_CONCERNS §1.12.
"""
from __future__ import annotations

import json

from harness.config import load_pipeline_config


def test_load_pipeline_config_returns_non_empty_dict():
    cfg = load_pipeline_config(preset_name="balanced")
    assert isinstance(cfg, dict)
    assert cfg  # non-empty
    # Spot-check a known field from card_capture.config.Options.
    assert "tracker_backend" in cfg


def test_load_pipeline_config_roundtrips_through_json():
    cfg = load_pipeline_config(preset_name="balanced")
    s = json.dumps(cfg)
    back = json.loads(s)
    assert back == cfg


def test_load_pipeline_config_unknown_preset_raises():
    import pytest
    with pytest.raises((ValueError, KeyError)):
        load_pipeline_config(preset_name="does_not_exist")
```

- [ ] **Step 4.3: Run the test — expect FAIL**

```bash
pytest tests/harness/test_cli_config.py -v
```

Expected: ImportError on `harness.config.load_pipeline_config`.

- [ ] **Step 4.4: Add the loader**

Append to `harness/config.py`:

```python
from dataclasses import asdict

from card_capture.config import Options


# Built-in presets mirror app/api/config.py:_BUILTIN_PRESETS.
# Keep in sync — see V4_CONCERNS §1.9 (drift gate covers this in Wave 5).
_PRESETS: dict[str, dict] = {
    "fast": {
        "corner_confidence": 0.40,
        "background_novelty_threshold": 0.06,
        "centroid_jump_ratio": 0.35,
        "valley_drop_ratio": 0.35,
        "foil_threshold": 50.0,
    },
    "balanced": {
        "corner_confidence": 0.50,
        "background_novelty_threshold": 0.08,
        "centroid_jump_ratio": 0.30,
        "valley_drop_ratio": 0.40,
        "foil_threshold": 50.0,
    },
    "quality": {
        "corner_confidence": 0.60,
        "background_novelty_threshold": 0.10,
        "centroid_jump_ratio": 0.25,
        "valley_drop_ratio": 0.45,
        "foil_threshold": 50.0,
    },
}


def load_pipeline_config(preset_name: str) -> dict:
    """Return the full pipeline config dict for *preset_name*.

    Starts from the default `Options` dataclass (the canonical source
    post-A2), then overlays the preset's overrides. Result is
    JSON-serialisable.
    """
    if preset_name not in _PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; "
            f"expected one of {list(_PRESETS)}"
        )

    config = asdict(Options())
    config.update(_PRESETS[preset_name])
    return config
```

- [ ] **Step 4.5: Replace the TODO placeholders in `harness/cli.py`**

Read `harness/cli.py` around lines 90 and 158 to confirm the placeholder
locations. Both calls currently look like:

```python
persist_run(
    ...,
    config={},  # TODO: load current pipeline config
    ...
)
```

Replace each `config={},  # TODO: load current pipeline config` with:

```python
config=load_pipeline_config(preset_name=preset),
```

…and add `from harness.config import load_pipeline_config` to the
imports.

The `preset` variable should be available from the CLI's `--preset`
option. If the CLI doesn't take a `--preset` flag yet, add one with
default `"balanced"`:

```python
@click.option(
    "--preset",
    default="balanced",
    type=click.Choice(["fast", "balanced", "quality"]),
    help="Pipeline config preset to record alongside the baseline.",
)
```

- [ ] **Step 4.6: Run the tests — expect PASS**

```bash
pytest tests/harness/test_cli_config.py -v
python -m pytest tests/ -q
```

Expected: green.

- [ ] **Step 4.7: Commit and open PR**

```bash
git add harness/config.py harness/cli.py tests/harness/test_cli_config.py
git commit -m "feat(harness): persist real pipeline config with baseline + run

Replaces the two 'config={}, # TODO' placeholders in harness/cli.py
(lines 90 and 158) with a real config snapshot loaded from the
canonical card_capture.config.Options + preset overrides. Adds
--preset flag to the CLI.

Closes V4_CONCERNS §1.12.
Depends on A2 (single canonical Options).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface D] Harness CLI config persistence (D1)" --body "$(cat <<'EOF'
## Summary
- harness/config.load_pipeline_config() builds a real pipeline config dict from the canonical Options dataclass + preset overrides.
- Replaces the two 'config={}, # TODO' placeholders in harness/cli.py.
- Adds --preset flag.

Closes V4_CONCERNS §1.12.
Blocked-by: A2 (#<N>).

## Test plan
- [x] new tests: tests/harness/test_cli_config.py (3 tests)
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 5: Update V4_CONCERNS.md and final verification

- [ ] **Step 5.1: Move §1.3, §1.4, §1.11, §1.12, §1.13 to §2**

Edit `V4_CONCERNS.md`:

- §1.3 → §2.18 (D4 PR number)
- §1.4 → §2.19 (D4 PR number)
- §1.11 → §2.20 (D2 PR number)
- §1.12 → §2.21 (D1 PR number)
- §1.13 → §2.22 (D3 PR number)

Commit and push.

- [ ] **Step 5.2: Report completion**

Surface D is done.

---

## Self-Review Checklist

- [ ] D1, D2, D3, D4 merged.
- [ ] `V4_CONCERNS.md` §1.3, §1.4, §1.11, §1.12, §1.13 moved to §2.
- [ ] CI green on `main`.
- [ ] IMG_5872 truth file at canonical path; legacy fallbacks warn.
- [ ] `harness_config.json` no longer at repo root.
- [ ] Reference frames gitignored; regen command documented.
- [ ] Round-trip test green: full Report JSON-serialises.
- [ ] `harness/cli.py` no longer has TODO placeholders.
