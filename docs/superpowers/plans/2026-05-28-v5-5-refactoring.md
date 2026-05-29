# V5.5 Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor card-capture from a Metaflow-driven multi-process pipeline into a single-process pipeline runtime with statically-enforced architectural boundaries, eliminating the ~52s re-decode and ~4-6min fuse-fanout overhead while preventing future architectural drift.

**Architecture:** Six sequential phases.

- **Phase 0** establishes a green CI baseline (prerequisite).
- **Phase 1** adds contracts (`PipelineRuntime` / `PipelineRunner` / `PipelineTelemetry` protocols, `RunManifest`) and advisory static enforcement (Import Linter, AST scanners, OTel Metrics).
- **Phase 2** introduces `GpuSession`, device-tagged batch types, and named CPU-export boundaries; advisory checks tighten to blocking.
- **Phase 3** collapses the per-stage Metaflow subprocess graph into one in-process runtime call and removes Metaflow.
- **Phase 4** introduces `card_capture.data` with single-writer SQLite covering pipeline, FastAPI, and harness writers.
- **Phase 5** unifies platform adapters (Beam, RunPod, Vast.ai) under the `PipelineRunner` contract.

Each phase produces working software on its own. Within each phase, tasks follow TDD: write failing test → run it → implement → run it → commit.

**Tech Stack:** Python 3.9+, PyTorch (CUDA/MPS), Kornia, OpenCV, PyAV, SQLite WAL, OpenTelemetry Metrics SDK, Import Linter, pytest, FastAPI.

**Spec:** `docs/superpowers/specs/2026-05-24-v5-5-refactoring-design.md` is the source of truth. When this plan and the spec disagree, the spec wins; raise the conflict before proceeding.

---

## File Structure

New packages and files created across the plan:

```text
docs/architecture/
    standards.md                              Phase 1; phase-marked binding rules
.importlinter                                 Phase 1; package boundary contracts

pyproject.toml                                Phase 1; add lint deps + gpu_strict_lint config

src/card_capture/
    pipeline/                                 Phase 1; replaces Metaflow flow code
        __init__.py
        request.py                            PipelineRunRequest / Result / RunManifest
        telemetry.py                          PipelineTelemetry protocol + OTel adapter
        runtime.py                            PipelineRuntime protocol
        runner.py                             PipelineRunner protocol
        stages/                               Phase 3; in-process stage facades
            __init__.py
            sample.py
            detect.py
            novelty.py
            track.py
            refine.py
            score.py
            resolve.py
            fuse.py
            dedup.py
            store.py
    runtime/                                  Phase 2
        __init__.py
        gpu_session.py                        GpuSession + capability check
        batches.py                            Device-tagged batch types + export helpers
        guards.py                             Forbidden-op runtime guard
        strict_gpu.py                         StrictGpuRuntime
        cpu_debug.py                          CpuDebugRuntime
    platforms/                                Phase 5
        __init__.py
        local.py
        runpod.py
        beam.py
        vastai.py
        manifests.py
    data/                                     Phase 4
        __init__.py
        connection.py
        schema.py
        writer.py                             Single-writer queue
        repositories/
            __init__.py
            runs.py
            events.py
            cards.py
            videos.py
            labeling.py
            telemetry.py

tests/
    architecture/                             Phase 1
        __init__.py
        test_import_linter.py
        test_gpu_strict_calls.py
        test_raw_sql_outside_data.py
        test_metaflow_absent.py               Phase 3 (added when Phase 3 lands)
        test_skip_audit.py                    Phase 0
    runtime/                                  Phase 2
        __init__.py
        test_gpu_session.py
        test_batches.py
        test_strict_guard_monkeypatch.py
        test_cpu_debug_strict_gpu_equivalence.py
    pipeline/                                 Phase 1+3
        test_manifest_roundtrip.py
        test_telemetry_protocol.py
        test_runtime_smoke.py
    data/                                     Phase 4
        __init__.py
        test_writer_serializes.py
        test_runs_repository.py
        test_events_repository.py
        test_cards_repository.py
    performance/                              Phase 0
        __init__.py
        test_perf_harness_smoke.py
        test_no_redecode.py                   Phase 3

harness/performance/                          Phase 0
    __init__.py
    runner.py
    report.py

pipeline/                                     Removed in Phase 3
    card_capture_flow.py                      DELETED
    steps/*.py                                DELETED (after migration)
```

---

# Phase 0: CI and Confidence Baseline

**Goal:** Make the default CI lane green and add the perf harness scaffold before any architecture changes.

**Acceptance:** `pytest tests/ -q` exits 0 with no unexplained skips; perf harness produces a JSON report on a synthetic fixture.

---

### Task 0.1: Inventory current CI failures and skipped tests

**Files:**
- Create: `docs/superpowers/plans/v5-5/ci-baseline-inventory.md`

- [ ] **Step 1: Run the full test suite and capture output**

Run:
```bash
python3 -m pytest tests/ -q --tb=no 2>&1 | tee /tmp/v55-ci-inventory.log
```

Expected: a list of failing/skipped tests. Per CLAUDE.md the known pre-existing failures are at least `tests/migrations/test_schema.py::test_migrations_are_idempotent`, several in `test_wave1/2_robustness.py`, and `test_path_equivalence.py`.

- [ ] **Step 2: Categorize each failure/skip into one of: real-bug, stale-fixture, missing-hardware, missing-credentials, plan-changed-behavior**

Write the categorization into `docs/superpowers/plans/v5-5/ci-baseline-inventory.md`. One row per test:

```markdown
# V5.5 CI Baseline Inventory

| Test ID | Status | Category | Disposition (fix / quarantine / hardware-skip / delete) |
|---|---|---|---|
| tests/migrations/test_schema.py::test_migrations_are_idempotent | FAIL | real-bug | fix |
| tests/test_wave1_robustness.py::test_X | FAIL | plan-changed-behavior | delete |
...
```

- [ ] **Step 3: Commit the inventory**

```bash
git add docs/superpowers/plans/v5-5/ci-baseline-inventory.md
git commit -m "docs(v5.5): inventory CI failures and skipped tests for phase 0"
```

---

### Task 0.2: Fix or quarantine each failing test per inventory

**Files:**
- Modify: each test identified in Task 0.1 with disposition `fix` or `quarantine`
- Modify: `pyproject.toml` (add quarantine marker)

- [ ] **Step 1: Register a `quarantine` pytest marker**

Edit `pyproject.toml`, add to `[tool.pytest.ini_options]` (create the section if missing):

```toml
[tool.pytest.ini_options]
markers = [
  "quarantine: known-broken test, must reference an issue and disposition; excluded from default CI lane",
  "cuda: requires NVIDIA CUDA hardware",
  "mps: requires Apple Silicon MPS",
  "provider: requires external provider credentials",
  "benchmark: real-video performance benchmark",
  "slow: takes >5s; excluded from fast PR lane",
]
addopts = "-m 'not quarantine and not benchmark'"
```

- [ ] **Step 2: Apply markers to each test per the inventory**

For tests categorized `missing-hardware`, change `@pytest.mark.skip(...)` to `@pytest.mark.cuda` / `@pytest.mark.mps` etc. with `pytest.importorskip` or capability gate.

For tests categorized `real-bug`, fix them. For each fix:

```bash
git add tests/path/test.py src/...
git commit -m "test(v55-phase0): fix <test> per CI baseline inventory"
```

For tests categorized `plan-changed-behavior`, delete them with a commit message referencing this plan.

For tests categorized `stale-fixture` with no clear ownership, mark `@pytest.mark.quarantine(reason="<issue-link or one-line>", disposition="<delete-after-phase-N | rewrite>")`.

- [ ] **Step 3: Run the full default lane**

Run:
```bash
python3 -m pytest tests/ -q --tb=short
```

Expected: PASS with 0 unexplained skips and 0 failures. Any quarantined tests are deselected by `addopts`.

- [ ] **Step 4: Commit final marker pass**

```bash
git add pyproject.toml tests/
git commit -m "test(v55-phase0): register markers and clear default CI lane"
```

---

### Task 0.3: Add skip-audit test

**Files:**
- Create: `tests/architecture/__init__.py`
- Create: `tests/architecture/test_skip_audit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/architecture/__init__.py` (empty file).

Create `tests/architecture/test_skip_audit.py`:

```python
"""Audit that every @pytest.mark.skip / skipif / quarantine has a reason."""
from __future__ import annotations
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"

ALLOWED_MARKERS = {"cuda", "mps", "provider", "benchmark", "slow", "quarantine"}


def _iter_test_files():
    for p in TEST_ROOT.rglob("test_*.py"):
        yield p


def _decorator_name(dec: ast.expr) -> str:
    # @pytest.mark.skip / @pytest.mark.skipif / @pytest.mark.foo / @pytest.mark.foo(...)
    node = dec.func if isinstance(dec, ast.Call) else dec
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_no_unexplained_skips():
    violations: list[str] = []
    for path in _iter_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                name = _decorator_name(dec)
                if name in {"pytest.mark.skip", "pytest.mark.skipif"}:
                    has_reason = isinstance(dec, ast.Call) and any(
                        kw.arg == "reason" for kw in dec.keywords
                    )
                    if not has_reason:
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}::{node.name} has skip without reason="
                        )
                if name.startswith("pytest.mark."):
                    marker = name.removeprefix("pytest.mark.")
                    # skip/skipif handled above; allow registered project markers.
                    if marker in {"skip", "skipif"}:
                        continue
                    if marker not in ALLOWED_MARKERS:
                        # Allowed if registered in pyproject markers list; we trust pytest's
                        # PytestUnknownMarkWarning to catch unregistered markers separately.
                        pass
    assert not violations, "\n".join(violations)
```

- [ ] **Step 2: Run the test and verify it passes (no unexplained skips remain after Task 0.2)**

Run:
```bash
python3 -m pytest tests/architecture/test_skip_audit.py -v
```

Expected: PASS. If FAIL, return to Task 0.2 for the offending tests.

- [ ] **Step 3: Commit**

```bash
git add tests/architecture/
git commit -m "test(v55-phase0): add skip-audit to lock in clean test inventory"
```

---

### Task 0.4: Create performance harness scaffold

**Files:**
- Create: `harness/performance/__init__.py`
- Create: `harness/performance/runner.py`
- Create: `harness/performance/report.py`

- [ ] **Step 1: Create `harness/performance/__init__.py`** (empty)

- [ ] **Step 2: Write `harness/performance/runner.py`**

```python
"""Lightweight performance harness for V5.5.

One command produces a JSON report comparable across branches/machines.
This phase ships the scaffold + synthetic fixture path; real-video paths
are added in later phases.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


@dataclasses.dataclass
class PerfReport:
    run_id: str
    profile: str
    video: str
    git_sha: str
    machine: Mapping[str, Any]
    timings_ms: Mapping[str, float]
    counters: Mapping[str, int]
    cards_extracted: int
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _machine_info() -> Mapping[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


def run(profile: str, video: str, out_dir: Path) -> PerfReport:
    """Run a perf profile. Phase 0 supports the `synthetic_smoke` profile only."""
    run_id = uuid.uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    counters: dict[str, int] = {}
    timings: dict[str, float] = {}
    cards = 0
    error: str | None = None
    try:
        if profile == "synthetic_smoke":
            # Phase 0: pretend stage timings; this proves the harness shape works.
            for stage in ("sample", "detect", "track", "refine", "score", "fuse", "store"):
                stage_start = time.perf_counter()
                time.sleep(0.01)
                timings[stage] = (time.perf_counter() - stage_start) * 1000.0
            counters["frames_decoded"] = 0
            counters["model_loads"] = 0
            counters["video_reopens"] = 0
            cards = 0
        else:
            raise ValueError(f"unknown perf profile in phase 0: {profile!r}")
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    timings["__total__"] = (time.perf_counter() - start) * 1000.0

    report = PerfReport(
        run_id=run_id,
        profile=profile,
        video=video,
        git_sha=_git_sha(),
        machine=_machine_info(),
        timings_ms=timings,
        counters=counters,
        cards_extracted=cards,
        error=error,
    )
    (out_dir / "perf_report.json").write_text(report.to_json())
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.performance")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--profile", required=True)
    run_p.add_argument("--video", required=True)
    run_p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        report = run(args.profile, args.video, args.out)
        print(report.to_json())
        return 0 if report.error is None else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 3: Write `harness/performance/report.py`** (placeholder for future compare/diff helpers)

```python
"""Comparison and aggregation helpers for perf reports. Filled out in later phases."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from harness.performance.runner import PerfReport  # noqa: F401  (re-export)


def load_reports(paths: Iterable[Path]) -> list[dict]:
    import json
    return [json.loads(p.read_text()) for p in paths]
```

- [ ] **Step 4: Verify the CLI runs**

Run:
```bash
python3 -m harness.performance run --profile synthetic_smoke --video none --out /tmp/v55-perf
cat /tmp/v55-perf/perf_report.json
```

Expected: JSON output with `run_id`, `git_sha`, per-stage `timings_ms`, and `cards_extracted: 0`.

- [ ] **Step 5: Commit**

```bash
git add harness/performance/
git commit -m "feat(v55-phase0): perf harness scaffold with synthetic_smoke profile"
```

---

### Task 0.5: Perf harness smoke test

**Files:**
- Create: `tests/performance/__init__.py`
- Create: `tests/performance/test_perf_harness_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/performance/__init__.py` (empty).

`tests/performance/test_perf_harness_smoke.py`:

```python
"""Smoke test: the perf harness produces a well-formed JSON report."""
from __future__ import annotations

import json
from pathlib import Path

from harness.performance.runner import run


def test_synthetic_smoke_writes_report(tmp_path: Path):
    report = run(profile="synthetic_smoke", video="none", out_dir=tmp_path)
    out = tmp_path / "perf_report.json"
    assert out.exists(), "perf_report.json was not written"
    data = json.loads(out.read_text())

    # Required keys (shape-only assertion; values vary)
    for key in ("run_id", "profile", "video", "git_sha", "machine", "timings_ms", "counters", "cards_extracted"):
        assert key in data, f"missing key {key!r} in report"
    assert "__total__" in data["timings_ms"]
    assert report.error is None
```

- [ ] **Step 2: Run and confirm PASS**

Run:
```bash
python3 -m pytest tests/performance/test_perf_harness_smoke.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/performance/
git commit -m "test(v55-phase0): smoke test for perf harness shape"
```

---

### Task 0.6: Add CI lane definition (or document the manual command)

**Files:**
- Modify: `.github/workflows/ci.yml` (if exists) OR create `docs/superpowers/plans/v5-5/ci-lane-commands.md`

- [ ] **Step 1: Check for existing CI**

Run:
```bash
ls .github/workflows/ 2>/dev/null
```

If a workflow exists, add a step `python3 -m pytest tests/ -q` and a step `python3 -m pytest tests/architecture/ -q`.

If no workflow exists, create `docs/superpowers/plans/v5-5/ci-lane-commands.md` documenting the canonical commands the project owner will run locally or via a future workflow:

```markdown
# V5.5 CI Lane Commands

## Fast PR lane (default; no GPU, no credentials, no real videos)

```bash
python3 -m pytest tests/ -q
python3 -m pytest tests/architecture/ -q
python3 -m pytest tests/performance/test_perf_harness_smoke.py -q
```

Optional hardware lane (CUDA):

```bash
python3 -m pytest tests/ -q -m cuda
```

Optional hardware lane (MPS):

```bash
python3 -m pytest tests/ -q -m mps
```
```

- [ ] **Step 2: Commit**

```bash
git add .github/ docs/superpowers/plans/v5-5/ 2>/dev/null
git commit -m "docs(v55-phase0): document fast PR CI lane commands"
```

**Phase 0 complete.** The default test suite is green, skips are explicit, the perf harness produces JSON. Tag this state:

```bash
git tag v55-phase0-complete
```

---

# Phase 1: Contracts, Telemetry, and Static Enforcement (Advisory)

**Goal:** Introduce the protocols (`PipelineRuntime`, `PipelineRunner`, `PipelineTelemetry`) and `RunManifest` model. Add Import Linter and AST scanners as advisory PR-lane checks (they report but do not fail the build yet). Add the OTel Metrics adapter.

**Acceptance:** New `card_capture.pipeline` package exists with protocols and manifest. Import Linter, GPU-strict AST scanner, and raw-SQL scanner all execute on the PR lane in advisory mode. `docs/architecture/standards.md` exists.

**Behavior change:** None. Phase 1 is additive.

---

### Task 1.1: Create `card_capture.pipeline` package skeleton

**Files:**
- Create: `src/card_capture/pipeline/__init__.py`

- [ ] **Step 1: Verify directory does not collide with existing `pipeline/` top-level**

Run:
```bash
ls src/card_capture/pipeline 2>/dev/null && echo "exists" || echo "ok-to-create"
```

Expected: `ok-to-create`. (The top-level `pipeline/` directory is different — that's Metaflow code, removed in Phase 3.)

- [ ] **Step 2: Create the package init**

`src/card_capture/pipeline/__init__.py`:

```python
"""V5.5 pipeline package: runtime/runner/telemetry contracts and stage facades.

This package replaces the top-level `pipeline/` Metaflow code in Phase 3.
Imports inside this package must not reach into app/, platforms/, or sqlite3.
"""
```

- [ ] **Step 3: Commit**

```bash
git add src/card_capture/pipeline/__init__.py
git commit -m "feat(v55-phase1): create card_capture.pipeline package skeleton"
```

---

### Task 1.2: Define `PipelineRunRequest`, `PipelineRunResult`, `RunManifest`

**Files:**
- Create: `src/card_capture/pipeline/request.py`
- Create: `tests/pipeline/test_manifest_roundtrip.py`

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_manifest_roundtrip.py`:

```python
"""RunManifest must round-trip through JSON without provider-specific assumptions."""
from __future__ import annotations

import json
from pathlib import Path

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
    ContractViolation,
)


def test_manifest_roundtrip_minimal():
    manifest = RunManifest(
        run_id="abc123",
        runtime_mode="cpu_debug",
        input_video="artifact://local/in.MOV",
        output_artifacts=["artifact://local/abc123/cards/"],
        cards=[],
        stage_timings=[StageTiming(stage="sample", elapsed_ms=12.5)],
        contract_violations=[],
        version="0.5.5+phase1",
    )
    blob = manifest.to_json()
    again = RunManifest.from_json(blob)
    assert again == manifest


def test_request_serializable_only_references():
    req = PipelineRunRequest(
        run_id="abc123",
        input_video="artifact://local/in.MOV",
        output_root="artifact://local/abc123/",
        runtime_mode="cpu_debug",
        config={"corner_confidence": 0.5},
    )
    blob = json.dumps(req.to_dict())
    again = PipelineRunRequest.from_dict(json.loads(blob))
    assert again == req


def test_contract_violation_has_stable_code():
    v = ContractViolation(code="cpu_read_in_strict", metadata={"call_site": "foo:42"})
    assert v.code == "cpu_read_in_strict"
    assert v.metadata["call_site"] == "foo:42"
```

- [ ] **Step 2: Run and confirm it fails for `ImportError`**

Run:
```bash
python3 -m pytest tests/pipeline/test_manifest_roundtrip.py -v
```

Expected: FAIL with `ModuleNotFoundError: card_capture.pipeline.request`.

- [ ] **Step 3: Implement `request.py`**

`src/card_capture/pipeline/request.py`:

```python
"""Serializable contracts passed between runtime, runner, app, and harness.

Values passed across this boundary must remain JSON-serializable. They must
not include CUDA tensors, model objects, open video handles, or process-local
resources.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Literal, Mapping


RuntimeMode = Literal["strict_gpu", "cpu_debug", "mixed_compat"]


@dataclasses.dataclass(frozen=True)
class PipelineRunRequest:
    run_id: str
    input_video: str            # artifact:// reference
    output_root: str            # artifact:// reference
    runtime_mode: RuntimeMode
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["config"] = dict(self.config)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineRunRequest":
        return cls(
            run_id=data["run_id"],
            input_video=data["input_video"],
            output_root=data["output_root"],
            runtime_mode=data["runtime_mode"],
            config=dict(data.get("config", {})),
        )


@dataclasses.dataclass(frozen=True)
class StageTiming:
    stage: str
    elapsed_ms: float
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ContractViolation:
    code: str                   # stable, machine-readable category
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class CardRecord:
    """Provider-neutral card output. Refined in Phase 3 to match storage."""
    card_instance_id: str
    front_crop: str             # artifact:// reference
    back_crop: str | None = None
    quality: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RunManifest:
    run_id: str
    runtime_mode: RuntimeMode
    input_video: str
    output_artifacts: list[str]
    cards: list[CardRecord]
    stage_timings: list[StageTiming]
    contract_violations: list[ContractViolation]
    version: str
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "RunManifest":
        data = json.loads(blob)
        return cls(
            run_id=data["run_id"],
            runtime_mode=data["runtime_mode"],
            input_video=data["input_video"],
            output_artifacts=list(data.get("output_artifacts", [])),
            cards=[CardRecord(**c) for c in data.get("cards", [])],
            stage_timings=[StageTiming(**s) for s in data.get("stage_timings", [])],
            contract_violations=[ContractViolation(**v) for v in data.get("contract_violations", [])],
            version=data["version"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclasses.dataclass(frozen=True)
class PipelineRunResult:
    manifest: RunManifest
    manifest_path: str | None = None
```

- [ ] **Step 4: Run tests; confirm PASS**

Run:
```bash
python3 -m pytest tests/pipeline/test_manifest_roundtrip.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/request.py tests/pipeline/test_manifest_roundtrip.py
git commit -m "feat(v55-phase1): PipelineRunRequest / RunManifest / CardRecord contracts"
```

---

### Task 1.3: Define `PipelineTelemetry` protocol + tests

**Files:**
- Create: `src/card_capture/pipeline/telemetry.py`
- Create: `tests/pipeline/test_telemetry_protocol.py`

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_telemetry_protocol.py`:

```python
"""PipelineTelemetry callers can swap implementations without code changes."""
from __future__ import annotations

from card_capture.pipeline.telemetry import (
    PipelineTelemetry,
    NoopTelemetry,
    InMemoryTelemetry,
)


def test_noop_satisfies_protocol():
    sink: PipelineTelemetry = NoopTelemetry()
    sink.stage_started("detect", {})
    sink.stage_finished("detect", 1234, {"frames": 100})
    sink.resource_sample({"vram_mb": 4096})
    sink.contract_violation("cpu_read_in_strict", {"call_site": "f.py:1"})


def test_inmemory_records_events_in_order():
    sink = InMemoryTelemetry()
    sink.stage_started("detect", {})
    sink.stage_finished("detect", 1234, {"frames": 100})
    sink.resource_sample({"vram_mb": 4096})
    sink.contract_violation("cpu_read_in_strict", {})
    kinds = [e.kind for e in sink.events]
    assert kinds == ["stage_started", "stage_finished", "resource_sample", "contract_violation"]
```

- [ ] **Step 2: Run and confirm FAIL (`ModuleNotFoundError`)**

Run:
```bash
python3 -m pytest tests/pipeline/test_telemetry_protocol.py -v
```

- [ ] **Step 3: Implement `telemetry.py`**

`src/card_capture/pipeline/telemetry.py`:

```python
"""Application-facing telemetry contract.

Implementations include a no-op for tests, an in-memory recorder for tests/
debugging, and (added in Task 1.4) an OpenTelemetry Metrics adapter.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping, Protocol


@dataclasses.dataclass(frozen=True)
class TelemetryEvent:
    kind: str
    payload: Mapping[str, object]


class PipelineTelemetry(Protocol):
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
    def resource_sample(self, sample: Mapping[str, object]) -> None: ...
    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None: ...


class NoopTelemetry:
    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None: ...
    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None: ...
    def resource_sample(self, sample: Mapping[str, object]) -> None: ...
    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None: ...


class InMemoryTelemetry:
    """For tests. Not thread-safe; use one instance per run."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def stage_started(self, stage: str, metadata: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("stage_started", {"stage": stage, **metadata}))

    def stage_finished(self, stage: str, elapsed_ms: int, metadata: Mapping[str, object]) -> None:
        self.events.append(
            TelemetryEvent("stage_finished", {"stage": stage, "elapsed_ms": elapsed_ms, **metadata})
        )

    def resource_sample(self, sample: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("resource_sample", dict(sample)))

    def contract_violation(self, code: str, metadata: Mapping[str, object]) -> None:
        self.events.append(TelemetryEvent("contract_violation", {"code": code, **metadata}))
```

- [ ] **Step 4: Run tests; PASS**

Run:
```bash
python3 -m pytest tests/pipeline/test_telemetry_protocol.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/telemetry.py tests/pipeline/test_telemetry_protocol.py
git commit -m "feat(v55-phase1): PipelineTelemetry protocol + Noop/InMemory impls"
```

---

### Task 1.4: OpenTelemetry Metrics adapter for `PipelineTelemetry`

**Files:**
- Modify: `pyproject.toml` (add `opentelemetry-sdk`, `opentelemetry-api` deps)
- Modify: `src/card_capture/pipeline/telemetry.py` (add `OtelMetricsTelemetry`)
- Create: `tests/pipeline/test_telemetry_otel.py`

- [ ] **Step 1: Add OTel deps to pyproject**

Edit `pyproject.toml` under `[project] dependencies`:

```toml
dependencies = [
  "numpy",
  "opencv-python",
  "Pillow>=10.4.0",
  "supervision>=0.21,<0.28",
  "vastai>=0.5.0",
  "httpx>=0.27.0",
  "python-dotenv>=1.0.0",
  "boto3>=1.34.0",
  "opentelemetry-api>=1.25.0",
  "opentelemetry-sdk>=1.25.0",
]
```

Install:
```bash
pip install -e .
```

- [ ] **Step 2: Write failing test**

`tests/pipeline/test_telemetry_otel.py`:

```python
"""OtelMetricsTelemetry records measurements via an InMemoryMetricReader."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.metrics")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from card_capture.pipeline.telemetry import OtelMetricsTelemetry


def test_stage_finished_emits_histogram():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    sink = OtelMetricsTelemetry(meter=provider.get_meter("card_capture.pipeline"))
    sink.stage_started("detect", {})
    sink.stage_finished("detect", elapsed_ms=42, metadata={"frames": 100})

    metrics = reader.get_metrics_data()
    names = []
    for rm in metrics.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                names.append(m.name)
    assert "card_capture.pipeline.stage.duration_ms" in names
```

- [ ] **Step 3: Run and confirm FAIL**

Run:
```bash
python3 -m pytest tests/pipeline/test_telemetry_otel.py -v
```

- [ ] **Step 4: Implement `OtelMetricsTelemetry`**

Append to `src/card_capture/pipeline/telemetry.py`:

```python
from opentelemetry.metrics import Meter


class OtelMetricsTelemetry:
    """Publishes stage timings and counters via OpenTelemetry Metrics.

    Traces and span sampling are intentionally out of scope (see spec
    Non-Goals). This adapter records histograms/counters only.
    """

    def __init__(self, meter: Meter) -> None:
        self._meter = meter
        self._stage_duration = meter.create_histogram(
            name="card_capture.pipeline.stage.duration_ms",
            description="Per-stage elapsed wall time",
            unit="ms",
        )
        self._violation_counter = meter.create_counter(
            name="card_capture.pipeline.contract_violations",
            description="Strict-contract violations recorded by the runtime",
        )
        self._resource_sample = meter.create_histogram(
            name="card_capture.pipeline.resource_sample",
            description="Generic resource sample (free-form payload via attributes)",
        )

    def stage_started(self, stage: str, metadata):
        # Stage start is metadata only; durations are recorded on finish.
        pass

    def stage_finished(self, stage: str, elapsed_ms: int, metadata):
        attrs = {"stage": stage, **{k: str(v) for k, v in metadata.items()}}
        self._stage_duration.record(elapsed_ms, attributes=attrs)

    def resource_sample(self, sample):
        # Record any single numeric field if present; otherwise a count of 1.
        numeric = next((v for v in sample.values() if isinstance(v, (int, float))), 1)
        attrs = {k: str(v) for k, v in sample.items()}
        self._resource_sample.record(numeric, attributes=attrs)

    def contract_violation(self, code: str, metadata):
        attrs = {"code": code, **{k: str(v) for k, v in metadata.items()}}
        self._violation_counter.add(1, attributes=attrs)
```

- [ ] **Step 5: Run; PASS**

```bash
python3 -m pytest tests/pipeline/test_telemetry_otel.py -v
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/card_capture/pipeline/telemetry.py tests/pipeline/test_telemetry_otel.py
git commit -m "feat(v55-phase1): OpenTelemetry Metrics adapter for PipelineTelemetry"
```

---

### Task 1.5: Define `PipelineRuntime` and `PipelineRunner` protocols

**Files:**
- Create: `src/card_capture/pipeline/runtime.py`
- Create: `src/card_capture/pipeline/runner.py`

- [ ] **Step 1: Create `runtime.py`**

```python
"""PipelineRuntime: executes one run in one process."""
from __future__ import annotations

from typing import Protocol

from .request import PipelineRunRequest, PipelineRunResult


class PipelineRuntime(Protocol):
    def run(self, request: PipelineRunRequest) -> PipelineRunResult: ...
```

- [ ] **Step 2: Create `runner.py`**

```python
"""PipelineRunner: handles local or remote submission."""
from __future__ import annotations

import dataclasses
from typing import Protocol

from .request import PipelineRunRequest, PipelineRunResult


@dataclasses.dataclass(frozen=True)
class PipelineRunHandle:
    run_id: str
    backend: str                # "local", "runpod", "beam", "vastai"
    opaque: str = ""            # provider-specific job id, opaque to callers


@dataclasses.dataclass(frozen=True)
class PipelineRunStatus:
    state: str                  # "pending", "running", "succeeded", "failed", "cancelled"
    progress: float = 0.0       # 0.0..1.0
    detail: str = ""


class PipelineRunner(Protocol):
    """Synchronous interface. Async wrapper added in Phase 5 for remote adapters."""

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle: ...
    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult: ...
    def cancel(self, handle: PipelineRunHandle) -> None: ...
```

- [ ] **Step 3: Smoke-import**

Run:
```bash
python3 -c "from card_capture.pipeline.runtime import PipelineRuntime; from card_capture.pipeline.runner import PipelineRunner, PipelineRunHandle; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/pipeline/runtime.py src/card_capture/pipeline/runner.py
git commit -m "feat(v55-phase1): PipelineRuntime and PipelineRunner protocols"
```

---

### Task 1.6: Add `.importlinter` config (advisory)

**Files:**
- Create: `.importlinter`
- Modify: `pyproject.toml` (add `import-linter` dev dep)
- Create: `tests/architecture/test_import_linter.py`

- [ ] **Step 1: Install import-linter**

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
dev = [
  "import-linter>=2.0",
  "pytest>=7.0",
]
```

Install:
```bash
pip install -e ".[dev]"
```

- [ ] **Step 2: Write `.importlinter`**

```ini
[importlinter]
root_packages =
    card_capture
    app
    pipeline

[importlinter:contract:no-sqlite3-outside-data]
name = sqlite3 only inside card_capture.data and migrations
type = forbidden
source_modules =
    card_capture.pipeline
    card_capture.runtime
    card_capture.platforms
    app
forbidden_modules =
    sqlite3

[importlinter:contract:no-provider-sdk-outside-platforms]
name = provider SDKs only inside card_capture.platforms
type = forbidden
source_modules =
    card_capture.pipeline
    card_capture.runtime
    app.api
    app.services.cards_service
    app.services.runs_service
forbidden_modules =
    runpod
    beam
    vastai

[importlinter:contract:strict-gpu-no-image-io]
name = strict GPU code must not import OpenCV/PIL file IO
type = forbidden
source_modules =
    card_capture.runtime.strict_gpu
forbidden_modules =
    PIL
    PIL.Image
    cv2.imgcodecs

[importlinter:contract:layered]
name = layered architecture
type = layers
containers =
    card_capture
layers =
    pipeline
    runtime
    data
```

Note: contracts reference modules that don't fully exist yet — that is intentional. Import Linter only checks contracts whose source modules exist; missing sources are no-ops until later phases populate them.

- [ ] **Step 3: Write test that runs Import Linter and reports (advisory mode in Phase 1)**

`tests/architecture/test_import_linter.py`:

```python
"""Import Linter contracts. Advisory in Phase 1, blocking in Phase 2."""
from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.skipif(
    os.environ.get("V55_IMPORT_LINT_BLOCKING") != "1",
    reason="Phase 1 advisory mode: set V55_IMPORT_LINT_BLOCKING=1 to fail on violations",
)
def test_import_contracts_blocking():
    result = subprocess.run(
        ["lint-imports"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"Import Linter violations:\n{result.stdout}\n{result.stderr}")


def test_import_contracts_advisory():
    """Run Import Linter and print results without failing (Phase 1)."""
    result = subprocess.run(
        ["lint-imports"], capture_output=True, text=True, check=False
    )
    print("=== Import Linter (advisory) ===")
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    # No assertion: advisory only in Phase 1.
```

- [ ] **Step 4: Run; verify both tests pass (advisory does nothing, blocking is skipped)**

```bash
python3 -m pytest tests/architecture/test_import_linter.py -v
```

Expected: 2 PASS (one skipped via skipif, one always-pass advisory).

- [ ] **Step 5: Commit**

```bash
git add .importlinter pyproject.toml tests/architecture/test_import_linter.py
git commit -m "feat(v55-phase1): Import Linter contracts (advisory)"
```

---

### Task 1.7: GPU-strict AST scanner (advisory, empty glob)

**Files:**
- Modify: `pyproject.toml` (add `[tool.gpu_strict_lint]` section)
- Create: `tests/architecture/test_gpu_strict_calls.py`

- [ ] **Step 1: Add config section to `pyproject.toml`**

```toml
[tool.gpu_strict_lint]
# Files in scope of the GPU-strict AST scanner. Populated in Phase 2.
files = []
forbidden_calls = [
  "cv2.VideoCapture",
  "cv2.imread",
  "cv2.imwrite",            # exception: card_capture.runtime.batches.to_cpu_for_export
  "PIL.Image.open",
  "torch.Tensor.cpu",
  "torch.Tensor.numpy",
]
allowed_export_helpers = [
  "card_capture.runtime.batches.to_cpu_for_score",
  "card_capture.runtime.batches.to_cpu_for_phash",
  "card_capture.runtime.batches.to_cpu_for_export",
  "card_capture.runtime.batches.to_cpu_for_fuse",
  "card_capture.runtime.batches.to_cpu_for_dedup",
]
```

- [ ] **Step 2: Write test (also implements the scanner)**

`tests/architecture/test_gpu_strict_calls.py`:

```python
"""Static AST scan: forbidden CPU/IO calls inside files tagged GPU-resident.

Phase 1: advisory (no files in scope, always passes).
Phase 2: populates `pyproject.toml [tool.gpu_strict_lint] files` and tightens.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh).get("tool", {}).get("gpu_strict_lint", {})


def _resolve_attr_chain(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _scan_file(path: Path, forbidden: set[str]) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _resolve_attr_chain(node.func)
            for f in forbidden:
                # Match exact qualified name (cv2.imread) or method name suffix (.cpu / .numpy)
                if name == f or name.endswith("." + f.split(".")[-1]) and f.split(".")[-1] in {"cpu", "numpy"}:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")
                    break
    return violations


@pytest.mark.skipif(
    os.environ.get("V55_GPU_STRICT_BLOCKING") != "1",
    reason="Phase 1 advisory mode: set V55_GPU_STRICT_BLOCKING=1 to fail on violations",
)
def test_no_forbidden_calls_in_gpu_files_blocking():
    cfg = _load_config()
    forbidden = set(cfg.get("forbidden_calls", []))
    files = []
    for glob in cfg.get("files", []):
        files.extend(REPO_ROOT.glob(glob))
    violations: list[str] = []
    for p in files:
        violations.extend(_scan_file(p, forbidden))
    assert not violations, "\n".join(violations)


def test_gpu_strict_calls_advisory():
    cfg = _load_config()
    forbidden = set(cfg.get("forbidden_calls", []))
    files = []
    for glob in cfg.get("files", []):
        files.extend(REPO_ROOT.glob(glob))
    print(f"=== GPU-strict AST scan (advisory): {len(files)} files in scope ===")
    for p in files:
        for v in _scan_file(p, forbidden):
            print(v)
```

- [ ] **Step 3: Run; both tests pass (no files in scope)**

```bash
python3 -m pytest tests/architecture/test_gpu_strict_calls.py -v
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/architecture/test_gpu_strict_calls.py
git commit -m "feat(v55-phase1): GPU-strict AST scanner skeleton (advisory)"
```

---

### Task 1.8: Raw-SQL AST scanner (advisory)

**Files:**
- Create: `tests/architecture/test_raw_sql_outside_data.py`

- [ ] **Step 1: Write the scanner test**

```python
"""Static AST scan: raw SQL string literals outside card_capture.data and migrations.

Phase 1: advisory.
Phase 4: blocking (after data layer migration).
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOTS = (
    "src/card_capture/data/",
    "migrations/",
    "tests/",            # tests may contain raw SQL fixtures
    "harness/schema.py",
)

# Heuristic: a string literal that begins with SELECT/INSERT/UPDATE/DELETE/CREATE/PRAGMA/ALTER/DROP
SQL_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|PRAGMA|ALTER|DROP|WITH)\b", re.IGNORECASE
)


def _iter_python_files():
    for root in ("src", "app", "pipeline", "harness"):
        for p in (REPO_ROOT / root).rglob("*.py"):
            rel = str(p.relative_to(REPO_ROOT))
            if any(rel.startswith(a) for a in ALLOWED_ROOTS):
                continue
            yield p


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if SQL_RE.match(node.value):
                out.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: raw SQL literal")
    return out


@pytest.mark.skipif(
    os.environ.get("V55_RAW_SQL_BLOCKING") != "1",
    reason="Phase 1 advisory: set V55_RAW_SQL_BLOCKING=1 to fail on violations",
)
def test_no_raw_sql_outside_data_blocking():
    violations: list[str] = []
    for p in _iter_python_files():
        violations.extend(_scan(p))
    assert not violations, "\n".join(violations)


def test_raw_sql_advisory():
    print("=== Raw-SQL scan (advisory) ===")
    for p in _iter_python_files():
        for v in _scan(p):
            print(v)
```

- [ ] **Step 2: Run; verify advisory PASS, blocking skipped**

```bash
python3 -m pytest tests/architecture/test_raw_sql_outside_data.py -v -s
```

The advisory test will print current raw-SQL call sites. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/architecture/test_raw_sql_outside_data.py
git commit -m "feat(v55-phase1): raw-SQL AST scanner (advisory)"
```

---

### Task 1.9: Create `docs/architecture/standards.md` with phase-marked rules

**Files:**
- Create: `docs/architecture/standards.md`

- [ ] **Step 1: Write the standards document**

`docs/architecture/standards.md`:

```markdown
# Card Capture Architecture Standards

> **Status:** Phase-marked. Each rule indicates the phase in which it becomes blocking.
> **Source:** `docs/superpowers/specs/2026-05-24-v5-5-refactoring-design.md`
> **For AI agents:** Read this file before changing any module listed under "Agent Triggers" below. If a requested change violates a rule, stop and surface the conflict.

## Agent Triggers

Read this document before editing any of:
- `pipeline/`
- `src/card_capture/pipeline/`
- `src/card_capture/runtime/`
- `src/card_capture/platforms/`
- `src/card_capture/data/`
- `src/card_capture/storage.py`
- `app/services/`
- `app/api/`
- CI workflow files
- `tests/architecture/`

## Rule Schema

Each rule carries:
- **Phase marker** — `advisory:phase-N` or `blocking:phase-N`.
- **Enforcement marker** — `static` (import lint / AST scan) / `runtime` (guard / assertion) / `review` (human only).

## Rules

### Pipeline Orchestration

- **R-ORCH-1** `blocking:phase-3` (static): No module imports `metaflow`.
- **R-ORCH-2** `blocking:phase-3` (static): No `@step` decorators or `FlowSpec` subclasses in production code.
- **R-ORCH-3** `blocking:phase-3` (runtime): A local run executes all stages in one process; the runtime opens the input video at most once except for explicit, telemetered fallback.
- **R-ORCH-4** `blocking:phase-3` (runtime): Each model is instantiated at most once per `PipelineRuntime.run()` call.

### Runtime Backends

- **R-RT-1** `blocking:phase-2` (static): GPU hot-path modules (per `pyproject.toml [tool.gpu_strict_lint] files`) must not call `cv2.VideoCapture`, `cv2.imread`, `PIL.Image.open`, `torch.Tensor.cpu`, or `torch.Tensor.numpy` except through the approved export helpers.
- **R-RT-2** `blocking:phase-2` (runtime): `StrictGpuRuntime` raises `ContractViolation` on missing CUDA/MPS, missing decode backend, or tensor host transfer outside an approved export boundary.
- **R-RT-3** `blocking:phase-2` (review): Production must not silently fall back to CPU. `runtime_mode` is explicit.
- **R-RT-4** `advisory:phase-2` → `blocking:phase-3` (review): Backend duplication is preferred over `if cuda` / `if mps` / `if cpu_debug` conditionals in hot paths.

### Data Access

- **R-DATA-1** `blocking:phase-4` (static, Import Linter): No module outside `card_capture.data`, `migrations/`, or `tests/` imports `sqlite3`.
- **R-DATA-2** `blocking:phase-4` (static, raw-SQL scanner): No raw SQL strings outside the allowed roots.
- **R-DATA-3** `blocking:phase-4` (review): Pipeline, app service, and platform code use repository methods, not direct connections.
- **R-DATA-4** `blocking:phase-4` (runtime): SQLite writes are serialized through `card_capture.data.writer`. Both pipeline runtime and FastAPI handlers route writes through it.

### Telemetry

- **R-TEL-1** `blocking:phase-1` (review): Timing data is emitted through `PipelineTelemetry`. No new code parses stdout for timings.
- **R-TEL-2** `advisory:phase-1` (review): The OpenTelemetry Metrics adapter is the default sink. Distributed tracing and span sampling are out of scope for V5.5.
- **R-TEL-3** `blocking:phase-1` (review): Telemetry adapters write SQLite events; pipeline stages must not write events directly.

### Platforms

- **R-PLAT-1** `blocking:phase-5` (static, Import Linter): Provider SDKs (`runpod`, `beam`, vast.ai clients) are imported only inside `card_capture.platforms`.
- **R-PLAT-2** `blocking:phase-5` (review): Every platform returns a `RunManifest` of the same shape.
- **R-PLAT-3** `blocking:phase-5` (review): Provider failures are mapped to stable categories (`preflight_failed`, `submission_failed`, ...) before app-facing status.

### Testing

- **R-TEST-1** `blocking:phase-0` (static, skip-audit): Every skipped test names its capability or quarantine reason.
- **R-TEST-2** `blocking:phase-2` (runtime): Strict-guard tests use `monkeypatch.context()` and do not patch outside the context block.
- **R-TEST-3** `blocking:phase-2` (review): `strict_gpu` and `cpu_debug` runs satisfy the equivalence contract on a fixture.

### CI

- **R-CI-1** `blocking:phase-0` (review): Default PR lane is green before refactor work starts.
- **R-CI-2** `blocking:phase-1` (review): PR lane runs Import Linter, GPU-strict AST scanner, and raw-SQL scanner (advisory in Phase 1).

## When Unsure

If a proposed change appears to violate a rule whose phase has been reached, stop and surface the conflict. If the rule is still advisory, document the violation in the commit message but proceed. If a new requirement is incompatible with a rule, propose a spec amendment before code changes.
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture/standards.md
git commit -m "docs(v55-phase1): architecture standards with phase-marked rules"
```

---

### Task 1.10: Wire architecture tests into the CI lane

**Files:**
- Modify: `docs/superpowers/plans/v5-5/ci-lane-commands.md` (from Task 0.6)
- (If GitHub workflow exists, also modify it.)

- [ ] **Step 1: Add architecture lane to CI commands**

Edit `docs/superpowers/plans/v5-5/ci-lane-commands.md`, append:

```markdown
## Architecture lane (advisory in Phase 1; blocking in later phases)

```bash
python3 -m pytest tests/architecture/ -q
```

To run in blocking mode (will fail on violations):

```bash
V55_IMPORT_LINT_BLOCKING=1 V55_GPU_STRICT_BLOCKING=1 V55_RAW_SQL_BLOCKING=1 \
  python3 -m pytest tests/architecture/ -q
```
```

- [ ] **Step 2: Run the architecture lane**

```bash
python3 -m pytest tests/architecture/ -q
```

Expected: PASS (advisory mode in Phase 1).

- [ ] **Step 3: Commit and tag Phase 1 complete**

```bash
git add docs/superpowers/plans/v5-5/ci-lane-commands.md
git commit -m "docs(v55-phase1): document architecture CI lane commands"
git tag v55-phase1-complete
```

**Phase 1 complete.** All contracts exist as protocols; static enforcement runs advisory; standards document is in place.

---

# Phase 2: Strict GPU Boundary

**Goal:** Introduce `GpuSession`, device-tagged batch types, the approved CPU-export helpers, and the `StrictGpuRuntime` + `CpuDebugRuntime` wrappers. Migrate the GPU hot path to require `GpuSession`. Tighten Phase-1 advisory checks to blocking.

**Acceptance:** Strict-guard tests pass via `monkeypatch.context()`. Equivalence contract test passes on a fixture. Import Linter, GPU-strict, and raw-SQL scanners all run blocking on the PR lane (raw-SQL stays advisory until Phase 4).

---

### Task 2.1: Create `card_capture.runtime` package

**Files:**
- Create: `src/card_capture/runtime/__init__.py`

- [ ] **Step 1: Create the package init**

```python
"""V5.5 runtime backends.

Submodules:
- gpu_session: GpuSession capability object
- batches: Device-tagged batch types and approved CPU-export helpers
- guards: Forbidden-op runtime guard (used by StrictGpuRuntime)
- strict_gpu: StrictGpuRuntime wrapper
- cpu_debug: CpuDebugRuntime
"""
```

- [ ] **Step 2: Commit**

```bash
git add src/card_capture/runtime/__init__.py
git commit -m "feat(v55-phase2): create card_capture.runtime package"
```

---

### Task 2.2: `GpuSession`

**Files:**
- Create: `src/card_capture/runtime/gpu_session.py`
- Create: `tests/runtime/__init__.py`
- Create: `tests/runtime/test_gpu_session.py`

- [ ] **Step 1: Write the failing test**

`tests/runtime/__init__.py` (empty).

`tests/runtime/test_gpu_session.py`:

```python
from __future__ import annotations

import pytest
import torch

from card_capture.pipeline.telemetry import NoopTelemetry
from card_capture.runtime.gpu_session import GpuSession, MissingGpuError


def test_session_requires_device():
    with pytest.raises(TypeError):
        GpuSession()  # type: ignore[call-arg]


def test_session_records_capability():
    sess = GpuSession(device=torch.device("cpu"), strict=False, telemetry=NoopTelemetry())
    assert sess.device.type == "cpu"
    assert sess.strict is False


def test_strict_session_rejects_cpu_device():
    with pytest.raises(MissingGpuError):
        GpuSession(device=torch.device("cpu"), strict=True, telemetry=NoopTelemetry())
```

- [ ] **Step 2: Run and confirm FAIL**

```bash
python3 -m pytest tests/runtime/test_gpu_session.py -v
```

- [ ] **Step 3: Implement `gpu_session.py`**

```python
"""GpuSession: capability object required to enter GPU hot-path code."""
from __future__ import annotations

import dataclasses

import torch

from card_capture.pipeline.telemetry import PipelineTelemetry


class MissingGpuError(RuntimeError):
    """Raised when StrictGpuRuntime is constructed without a GPU device."""


@dataclasses.dataclass(frozen=True)
class GpuSession:
    device: torch.device
    strict: bool
    telemetry: PipelineTelemetry

    def __post_init__(self) -> None:
        if self.strict and self.device.type == "cpu":
            raise MissingGpuError(
                f"strict GpuSession requires a GPU device, got {self.device}"
            )
```

- [ ] **Step 4: Run; PASS**

```bash
python3 -m pytest tests/runtime/test_gpu_session.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/runtime/gpu_session.py tests/runtime/
git commit -m "feat(v55-phase2): GpuSession capability object"
```

---

### Task 2.3: Device-tagged batch types

**Files:**
- Create: `src/card_capture/runtime/batches.py`
- Create: `tests/runtime/test_batches.py`

- [ ] **Step 1: Write the failing test**

`tests/runtime/test_batches.py`:

```python
from __future__ import annotations

import pytest
import torch

from card_capture.runtime.batches import (
    GpuFrameBatch,
    GpuCropBatch,
    GpuEmbeddingBatch,
    WrongDeviceError,
)


def _gpu_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("no GPU available")


def test_frame_batch_accepts_gpu_tensor():
    dev = _gpu_device()
    t = torch.zeros((2, 3, 1080, 1920), device=dev, dtype=torch.float32)
    batch = GpuFrameBatch(tensor=t)
    assert batch.tensor.device.type in {"cuda", "mps"}


def test_frame_batch_rejects_cpu_tensor():
    t = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with pytest.raises(WrongDeviceError):
        GpuFrameBatch(tensor=t)


def test_crop_batch_enforces_layout():
    dev = _gpu_device()
    bad = torch.zeros((4, 750, 1050, 3), device=dev)  # NHWC instead of NCHW
    with pytest.raises(ValueError):
        GpuCropBatch(tensor=bad)


def test_embedding_batch_2d_only():
    dev = _gpu_device()
    e = torch.zeros((8, 384), device=dev)
    GpuEmbeddingBatch(tensor=e)  # ok
    with pytest.raises(ValueError):
        GpuEmbeddingBatch(tensor=torch.zeros((8, 384, 1), device=dev))
```

- [ ] **Step 2: Run and FAIL**

```bash
python3 -m pytest tests/runtime/test_batches.py -v
```

- [ ] **Step 3: Implement `batches.py` (batch types only; export helpers in Task 2.4)**

```python
"""Device-tagged batch types and approved CPU-export helpers.

Wrapper types enforce device residency, layout, and dtype at construction.
Approved export helpers are the only legal `.cpu()` / `.numpy()` call sites
inside files tagged GPU-resident (see pyproject.toml [tool.gpu_strict_lint]).
"""
from __future__ import annotations

import dataclasses

import torch


class WrongDeviceError(TypeError):
    """Raised when a device-tagged batch is constructed with a CPU tensor."""


def _require_gpu(t: torch.Tensor, name: str) -> None:
    if t.device.type not in ("cuda", "mps"):
        raise WrongDeviceError(f"{name} requires a GPU tensor; got device={t.device}")


@dataclasses.dataclass(frozen=True)
class GpuFrameBatch:
    """NCHW float frames on GPU."""
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _require_gpu(self.tensor, "GpuFrameBatch")
        if self.tensor.dim() != 4:
            raise ValueError(f"GpuFrameBatch expects 4D NCHW, got shape {tuple(self.tensor.shape)}")


@dataclasses.dataclass(frozen=True)
class GpuCropBatch:
    """NCHW float crops on GPU, 750x1050 canonical."""
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _require_gpu(self.tensor, "GpuCropBatch")
        if self.tensor.dim() != 4:
            raise ValueError(f"GpuCropBatch expects 4D NCHW, got shape {tuple(self.tensor.shape)}")
        # Layout assertion: C must be small (3 or 4); H/W must be larger than C.
        _, c, h, w = self.tensor.shape
        if c > 8 or h < c or w < c:
            raise ValueError(
                f"GpuCropBatch shape {tuple(self.tensor.shape)} not NCHW; "
                "expected N, C, H, W with C <= 8"
            )


@dataclasses.dataclass(frozen=True)
class GpuEmbeddingBatch:
    """2D embedding tensor on GPU."""
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _require_gpu(self.tensor, "GpuEmbeddingBatch")
        if self.tensor.dim() != 2:
            raise ValueError(f"GpuEmbeddingBatch expects 2D, got shape {tuple(self.tensor.shape)}")
```

- [ ] **Step 4: Run; PASS**

```bash
python3 -m pytest tests/runtime/test_batches.py -v
```

(Tests are skipped on machines without GPU.)

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/runtime/batches.py tests/runtime/test_batches.py
git commit -m "feat(v55-phase2): device-tagged batch types"
```

---

### Task 2.4: Approved CPU-export boundary helpers

**Files:**
- Modify: `src/card_capture/runtime/batches.py` (append helpers)
- Create: `tests/runtime/test_export_helpers.py`

- [ ] **Step 1: Write the failing test**

`tests/runtime/test_export_helpers.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import torch

from card_capture.runtime.batches import (
    GpuCropBatch,
    GpuEmbeddingBatch,
    to_cpu_for_score,
    to_cpu_for_phash,
    to_cpu_for_dedup,
    to_cpu_for_fuse,
    to_cpu_for_export,
)


def _gpu_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("no GPU available")


def test_to_cpu_for_score_returns_dict():
    dev = _gpu_device()
    crops = GpuCropBatch(tensor=torch.rand((2, 3, 1050, 750), device=dev))
    scores = to_cpu_for_score(crops)
    assert isinstance(scores, list)
    assert all(isinstance(s, dict) for s in scores)
    for s in scores:
        # Required keys per spec Section 2 table
        for k in ("sharpness", "glare", "aspect_ratio", "size", "complexity", "border_purity", "confidence"):
            assert k in s


def test_to_cpu_for_phash_returns_hex():
    dev = _gpu_device()
    crops = GpuCropBatch(tensor=torch.rand((2, 3, 1050, 750), device=dev))
    hashes = to_cpu_for_phash(crops)
    assert len(hashes) == 2
    assert all(isinstance(h, str) for h in hashes)
    assert all(int(h, 16) >= 0 for h in hashes)


def test_to_cpu_for_dedup_returns_float32_arrays():
    dev = _gpu_device()
    emb = GpuEmbeddingBatch(tensor=torch.rand((3, 384), device=dev))
    arrs = to_cpu_for_dedup(emb)
    assert isinstance(arrs, np.ndarray)
    assert arrs.dtype == np.float32
    assert arrs.shape == (3, 384)


def test_to_cpu_for_export_writes_png(tmp_path):
    dev = _gpu_device()
    crops = GpuCropBatch(tensor=torch.rand((1, 3, 1050, 750), device=dev))
    paths = to_cpu_for_export(crops, out_dir=tmp_path, basenames=["card_0"])
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".png"
```

- [ ] **Step 2: Run; FAIL**

```bash
python3 -m pytest tests/runtime/test_export_helpers.py -v
```

- [ ] **Step 3: Implement export helpers in `batches.py`**

Append to `src/card_capture/runtime/batches.py`:

```python
# --- Approved CPU-export boundary helpers ---------------------------------
#
# Each helper is the SOLE legal `.cpu()` / `.numpy()` / cv2.imwrite call site
# inside files tagged GPU-resident. The GPU-strict AST scanner (Phase 2) will
# fail if a tagged file calls `tensor.cpu()` outside these helpers.

import cv2  # noqa: E402  (intentional: this module owns the boundary)
import imagehash  # type: ignore[import-not-found]  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pathlib import Path  # noqa: E402


def _to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """Internal: NCHW float [0,1] GPU -> HWC uint8 CPU."""
    x = t.detach()
    x = (x.clamp(0, 1) * 255).to(torch.uint8)
    x = x.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
    return x.cpu().numpy()  # boundary


def to_cpu_for_score(crops: GpuCropBatch) -> list[dict[str, float]]:
    """Reduce a crop batch to per-crop scalar quality scores.

    Stage 7 boundary. Returns one dict per crop matching QualityScore fields.
    """
    arr = _to_uint8_hwc(crops.tensor)  # (N, H, W, C)
    out: list[dict[str, float]] = []
    for img in arr:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        glare = float((gray > 240).mean())
        h, w = gray.shape
        aspect = float(h / w) if w else 0.0
        size = float(h * w)
        complexity = float(cv2.Canny(gray, 100, 200).mean())
        # border purity: variance of outer 8px ring
        ring = np.concatenate([gray[:8, :].ravel(), gray[-8:, :].ravel(),
                                gray[:, :8].ravel(), gray[:, -8:].ravel()])
        border_purity = float(1.0 / (ring.std() + 1.0))
        out.append({
            "sharpness": sharpness,
            "glare": glare,
            "aspect_ratio": aspect,
            "size": size,
            "complexity": complexity,
            "border_purity": border_purity,
            "confidence": 1.0,        # filled by detector
        })
    return out


def to_cpu_for_phash(crops: GpuCropBatch) -> list[str]:
    """Stage 8/10 boundary: perceptual hashes as hex strings."""
    arr = _to_uint8_hwc(crops.tensor)
    return [str(imagehash.phash(Image.fromarray(img))) for img in arr]


def to_cpu_for_fuse(crops: GpuCropBatch) -> np.ndarray:
    """Stage 9 boundary: NHWC uint8 array for fusion candidate selection."""
    return _to_uint8_hwc(crops.tensor)


def to_cpu_for_dedup(embeddings: GpuEmbeddingBatch) -> np.ndarray:
    """Stage 10 boundary: float32 numpy embeddings for cosine comparison."""
    return embeddings.tensor.detach().to(torch.float32).cpu().numpy()


def to_cpu_for_export(
    crops: GpuCropBatch, out_dir: Path, basenames: list[str]
) -> list[Path]:
    """Final export boundary: writes 750x1050 PNGs to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = _to_uint8_hwc(crops.tensor)
    if len(arr) != len(basenames):
        raise ValueError(f"basenames length {len(basenames)} != batch size {len(arr)}")
    paths: list[Path] = []
    for img, base in zip(arr, basenames):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        p = out_dir / f"{base}.png"
        cv2.imwrite(str(p), bgr)
        paths.append(p)
    return paths
```

Note: `imagehash` may need adding to `pyproject.toml` — check first:

```bash
python3 -c "import imagehash" 2>&1 | head -2
```

If missing, add `"imagehash>=4.3"` to `[project.dependencies]` in `pyproject.toml` and `pip install -e .`.

- [ ] **Step 4: Run; PASS**

```bash
python3 -m pytest tests/runtime/test_export_helpers.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/runtime/batches.py tests/runtime/test_export_helpers.py pyproject.toml
git commit -m "feat(v55-phase2): approved CPU-export boundary helpers"
```

---

### Task 2.5: `monkeypatch.context()` strict-guard tests

**Files:**
- Create: `src/card_capture/runtime/guards.py`
- Create: `tests/runtime/test_strict_guard_monkeypatch.py`

- [ ] **Step 1: Write the failing test (defines the guard surface)**

`tests/runtime/test_strict_guard_monkeypatch.py`:

```python
"""Strict guard uses monkeypatch.context() to verify forbidden ops fail.

The guard does NOT globally patch torch in production; that would break
third-party libraries. In production, StrictGpuRuntime exposes only safe
APIs. In tests, monkeypatch.context() proves the behavioral contract.
"""
from __future__ import annotations

import pytest
import torch

from card_capture.pipeline.telemetry import InMemoryTelemetry
from card_capture.runtime.guards import (
    StrictGuardActive,
    raise_forbidden_call,
    strict_section,
)


def test_strict_section_traps_tensor_cpu(monkeypatch):
    telemetry = InMemoryTelemetry()
    original_cpu = torch.Tensor.cpu
    with strict_section(telemetry=telemetry):
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor, "cpu", raise_forbidden_call("torch.Tensor.cpu"))
            t = torch.zeros((1,))
            with pytest.raises(StrictGuardActive):
                t.cpu()
    # Outside the context, torch.Tensor.cpu is restored.
    assert torch.Tensor.cpu is original_cpu
    # Violation recorded.
    assert any(e.kind == "contract_violation" for e in telemetry.events)


def test_strict_section_traps_cv2_imread(monkeypatch):
    import cv2
    telemetry = InMemoryTelemetry()
    original = cv2.imread
    with strict_section(telemetry=telemetry):
        with monkeypatch.context() as m:
            m.setattr(cv2, "imread", raise_forbidden_call("cv2.imread"))
            with pytest.raises(StrictGuardActive):
                cv2.imread("does_not_matter.png")
    assert cv2.imread is original


def test_violation_carries_stable_code():
    telemetry = InMemoryTelemetry()
    with strict_section(telemetry=telemetry):
        try:
            raise_forbidden_call("torch.Tensor.numpy")()
        except StrictGuardActive:
            pass
    violations = [e for e in telemetry.events if e.kind == "contract_violation"]
    assert violations
    assert violations[0].payload["code"] == "forbidden_call:torch.Tensor.numpy"
```

- [ ] **Step 2: Run; FAIL**

```bash
python3 -m pytest tests/runtime/test_strict_guard_monkeypatch.py -v
```

- [ ] **Step 3: Implement `guards.py`**

```python
"""Strict-mode runtime guard.

The guard is **not** a global monkeypatch in production. It records contract
violations into telemetry when invoked. Tests use `monkeypatch.context()` to
prove that, when a forbidden call IS made, the guard fires and records the
violation with a stable code.

Inside a `strict_section`, callers may install patched versions of forbidden
operations using `raise_forbidden_call(name)` as the replacement; the patched
callable raises `StrictGuardActive` and records the violation.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Callable, Iterator

from card_capture.pipeline.telemetry import PipelineTelemetry


_local = threading.local()


class StrictGuardActive(RuntimeError):
    """Raised when a forbidden call fires inside a strict section."""


@contextlib.contextmanager
def strict_section(telemetry: PipelineTelemetry) -> Iterator[None]:
    """Mark the current thread as inside a strict GPU section."""
    prev = getattr(_local, "telemetry", None)
    _local.telemetry = telemetry
    try:
        yield
    finally:
        _local.telemetry = prev


def _current_telemetry() -> PipelineTelemetry | None:
    return getattr(_local, "telemetry", None)


def raise_forbidden_call(name: str) -> Callable[..., object]:
    """Return a callable that records a violation and raises StrictGuardActive.

    Intended for use with `monkeypatch.context()` in tests:

        with monkeypatch.context() as m:
            m.setattr(torch.Tensor, "cpu", raise_forbidden_call("torch.Tensor.cpu"))
            ...
    """
    def _trap(*args, **kwargs):
        sink = _current_telemetry()
        if sink is not None:
            sink.contract_violation(f"forbidden_call:{name}", {"name": name})
        raise StrictGuardActive(f"Forbidden call inside strict_section: {name}")

    return _trap
```

- [ ] **Step 4: Run; PASS**

```bash
python3 -m pytest tests/runtime/test_strict_guard_monkeypatch.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/runtime/guards.py tests/runtime/test_strict_guard_monkeypatch.py
git commit -m "feat(v55-phase2): strict_section guard with monkeypatch.context() tests"
```

---

### Task 2.6: `CpuDebugRuntime` (skeleton)

**Files:**
- Create: `src/card_capture/runtime/cpu_debug.py`

- [ ] **Step 1: Write the runtime**

```python
"""CPU debug runtime.

A minimal, intentionally slow implementation of the PipelineRuntime contract
that runs on CPU only. Used for deterministic local debugging and CI.

Stage execution is wired in Phase 3 when stage facades exist. For now,
the runtime returns an empty manifest so the contract is satisfied.
"""
from __future__ import annotations

import time

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
)
from card_capture.pipeline.telemetry import PipelineTelemetry, NoopTelemetry


class CpuDebugRuntime:
    def __init__(self, telemetry: PipelineTelemetry | None = None) -> None:
        self._telemetry = telemetry or NoopTelemetry()

    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        if request.runtime_mode != "cpu_debug":
            raise ValueError(
                f"CpuDebugRuntime requires runtime_mode='cpu_debug', got {request.runtime_mode!r}"
            )
        start = time.perf_counter()
        # Stage wiring lands in Phase 3. Phase 2: return a well-formed manifest.
        timings: list[StageTiming] = []
        manifest = RunManifest(
            run_id=request.run_id,
            runtime_mode="cpu_debug",
            input_video=request.input_video,
            output_artifacts=[],
            cards=[],
            stage_timings=timings,
            contract_violations=[],
            version="0.5.5+phase2",
            metadata={"phase": "phase2-skeleton"},
        )
        self._telemetry.stage_finished(
            "__total__", int((time.perf_counter() - start) * 1000), {}
        )
        return PipelineRunResult(manifest=manifest)
```

- [ ] **Step 2: Smoke-test**

```bash
python3 -c "
from card_capture.pipeline.request import PipelineRunRequest
from card_capture.runtime.cpu_debug import CpuDebugRuntime
r = CpuDebugRuntime()
result = r.run(PipelineRunRequest(run_id='x', input_video='artifact://local/x.MOV', output_root='artifact://local/x/', runtime_mode='cpu_debug'))
print(result.manifest.runtime_mode, result.manifest.version)
"
```

Expected: `cpu_debug 0.5.5+phase2`.

- [ ] **Step 3: Commit**

```bash
git add src/card_capture/runtime/cpu_debug.py
git commit -m "feat(v55-phase2): CpuDebugRuntime skeleton"
```

---

### Task 2.7: `StrictGpuRuntime` (skeleton)

**Files:**
- Create: `src/card_capture/runtime/strict_gpu.py`

- [ ] **Step 1: Write the runtime**

```python
"""Strict GPU runtime wrapper.

In production this runtime does NOT globally monkeypatch torch. Instead, it
exposes only the safe device-tagged batch APIs to stage code. Forbidden
imports inside strict stage modules are caught statically by the GPU-strict
AST scanner (Phase 2 blocking).

Set CC_GPU_STRICT=1 to enable additional runtime assertion checks (device
tags, batch invariants) — this does NOT enable global monkeypatching.
"""
from __future__ import annotations

import os
import time

import torch

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
)
from card_capture.pipeline.telemetry import PipelineTelemetry, NoopTelemetry
from card_capture.runtime.gpu_session import GpuSession, MissingGpuError


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise MissingGpuError("No CUDA or MPS device available for StrictGpuRuntime")


class StrictGpuRuntime:
    def __init__(self, telemetry: PipelineTelemetry | None = None) -> None:
        self._telemetry = telemetry or NoopTelemetry()
        self._assert_mode = os.environ.get("CC_GPU_STRICT") == "1"

    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        if request.runtime_mode != "strict_gpu":
            raise ValueError(
                f"StrictGpuRuntime requires runtime_mode='strict_gpu', got {request.runtime_mode!r}"
            )
        device = _select_device()
        session = GpuSession(device=device, strict=True, telemetry=self._telemetry)

        start = time.perf_counter()
        timings: list[StageTiming] = []
        # Stage wiring lands in Phase 3.
        _ = session  # silence unused while skeleton

        manifest = RunManifest(
            run_id=request.run_id,
            runtime_mode="strict_gpu",
            input_video=request.input_video,
            output_artifacts=[],
            cards=[],
            stage_timings=timings,
            contract_violations=[],
            version="0.5.5+phase2",
            metadata={"phase": "phase2-skeleton", "device": str(device)},
        )
        self._telemetry.stage_finished(
            "__total__", int((time.perf_counter() - start) * 1000), {"device": str(device)}
        )
        return PipelineRunResult(manifest=manifest)
```

- [ ] **Step 2: Commit**

```bash
git add src/card_capture/runtime/strict_gpu.py
git commit -m "feat(v55-phase2): StrictGpuRuntime skeleton"
```

---

### Task 2.8: Equivalence contract test (CPU debug ↔ strict GPU)

**Files:**
- Create: `tests/runtime/test_cpu_debug_strict_gpu_equivalence.py`

- [ ] **Step 1: Write the test**

```python
"""Equivalence contract per spec Section 6.

Phase 2 ships the assertion shape with skeleton runtimes. Phase 3 wires
real stage execution; the test will then exercise actual card outputs.
"""
from __future__ import annotations

import pytest
import torch

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.runtime.cpu_debug import CpuDebugRuntime
from card_capture.runtime.strict_gpu import StrictGpuRuntime
from card_capture.runtime.gpu_session import MissingGpuError


def test_manifest_shape_matches_across_runtimes():
    req_cpu = PipelineRunRequest(
        run_id="abc",
        input_video="artifact://local/x.MOV",
        output_root="artifact://local/abc/",
        runtime_mode="cpu_debug",
    )
    cpu = CpuDebugRuntime().run(req_cpu).manifest

    try:
        req_gpu = PipelineRunRequest(
            run_id="abc",
            input_video="artifact://local/x.MOV",
            output_root="artifact://local/abc/",
            runtime_mode="strict_gpu",
        )
        gpu = StrictGpuRuntime().run(req_gpu).manifest
    except MissingGpuError:
        pytest.skip("no GPU available")

    # Identity / schema: exact match per equivalence contract table.
    assert cpu.run_id == gpu.run_id
    assert cpu.input_video == gpu.input_video
    assert sorted(vars(cpu).keys()) == sorted(vars(gpu).keys())

    # Counts: exact match.
    assert len(cpu.cards) == len(gpu.cards)

    # Phase 3 will expand this test with: geometric tolerance (corners ± 2px),
    # quality score tolerance (± 5%), pHash within 4 bits, embedding cosine ≥ 0.95.
```

- [ ] **Step 2: Run; PASS (or skip if no GPU)**

```bash
python3 -m pytest tests/runtime/test_cpu_debug_strict_gpu_equivalence.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/runtime/test_cpu_debug_strict_gpu_equivalence.py
git commit -m "test(v55-phase2): equivalence contract shape assertion"
```

---

### Task 2.9: Tighten Phase-1 advisory checks to blocking

**Files:**
- Modify: `pyproject.toml [tool.gpu_strict_lint] files = [...]` (populate)
- Modify: `tests/architecture/test_import_linter.py` (remove advisory-only path)
- Modify: `tests/architecture/test_gpu_strict_calls.py` (remove env gate)

- [ ] **Step 1: Populate the gpu_strict_lint file list**

Edit `pyproject.toml`:

```toml
[tool.gpu_strict_lint]
files = [
  "src/card_capture/runtime/strict_gpu.py",
  "src/card_capture/runtime/gpu_session.py",
  "src/card_capture/runtime/guards.py",
  # batches.py is the EXPORT BOUNDARY itself; intentionally NOT in scope.
]
```

- [ ] **Step 2: Make Import Linter blocking by default**

Edit `tests/architecture/test_import_linter.py`. Replace the `test_import_contracts_blocking` body to remove the env gate:

```python
def test_import_contracts():
    result = subprocess.run(
        ["lint-imports"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"Import Linter violations:\n{result.stdout}\n{result.stderr}")
```

Delete `test_import_contracts_advisory`.

- [ ] **Step 3: Make GPU-strict scanner blocking by default**

Edit `tests/architecture/test_gpu_strict_calls.py`. Rename `test_no_forbidden_calls_in_gpu_files_blocking` → `test_no_forbidden_calls_in_gpu_files` and remove the `@pytest.mark.skipif`. Delete `test_gpu_strict_calls_advisory`.

Keep raw-SQL scanner advisory (it tightens in Phase 4).

- [ ] **Step 4: Run; all PASS**

```bash
python3 -m pytest tests/architecture/ -v
```

If violations surface, fix them before committing.

- [ ] **Step 5: Update standards.md**

In `docs/architecture/standards.md`, change R-RT-1 marker from `blocking:phase-2` (already correct) and add a note: "Enforced as of Phase 2."

- [ ] **Step 6: Commit and tag**

```bash
git add pyproject.toml tests/architecture/ docs/architecture/standards.md
git commit -m "feat(v55-phase2): tighten Import Linter + GPU-strict scanner to blocking"
git tag v55-phase2-complete
```

**Phase 2 complete.** GPU runtime contract exists, strict guard is testable, equivalence shape is asserted. Two of three static checks are blocking.

---

# Phase 3: Single-Process Runtime and Metaflow Removal

**Goal:** Replace Metaflow per-stage subprocesses with a single in-process `PipelineRuntime.run()` call. Each stage runs as a direct function call; decoded frames and loaded models are reused across stages. Delete Metaflow.

**Acceptance:** A 4K reference video processes end-to-end in one process; `refine` reuses frames from `detect` (no re-decode); models load once; `pipeline/card_capture_flow.py` and `pipeline/steps/*.py` are deleted; `import metaflow` is forbidden by Import Linter.

**Reference:** The existing CUDA `fused_refine` path (`pipeline/steps/fused_refine.py`) is the *pattern* (single process, frames in memory). Each backend implements its own version; CUDA-specific machinery does not generalize.

---

### Task 3.1: Stage facade package

**Files:**
- Create: `src/card_capture/pipeline/stages/__init__.py`
- Create one file per stage (sample, detect, novelty, track, refine, score, resolve, fuse, dedup, store)

- [ ] **Step 1: Create the package init**

`src/card_capture/pipeline/stages/__init__.py`:

```python
"""In-process pipeline stages.

Each module exposes a single `run()` function that takes a stage-specific
input and `GpuSession | None` and returns a stage-specific output. Stages
do not own model loading or decode lifecycle — those belong to the runtime.

Stages map 1:1 onto pipeline/steps/*.py in V4; this is the deliberate
re-homing: same algorithmic work, no subprocess boundary, no datastore
pickling between stages.
"""
```

- [ ] **Step 2: Create stub stage modules**

For each stage in `(sample, detect, novelty, track, refine, score, resolve, fuse, dedup, store)`, create `src/card_capture/pipeline/stages/<name>.py` with:

```python
"""<Stage> facade. Wraps the existing V4 implementation; will be inlined further as Phase 3 progresses."""
from __future__ import annotations
# Real wiring happens in Tasks 3.3-3.8.
```

- [ ] **Step 3: Commit**

```bash
git add src/card_capture/pipeline/stages/
git commit -m "feat(v55-phase3): pipeline.stages package skeleton"
```

---

### Task 3.2: In-process `LocalPipelineRuntime` (loop body)

**Files:**
- Create: `src/card_capture/pipeline/runtime_local.py`
- Create: `tests/pipeline/test_runtime_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_runtime_smoke.py`:

```python
"""LocalPipelineRuntime runs all stages in one process and produces a manifest.

Phase 3 smoke test: uses a tiny synthetic video fixture. Real-video tests
go in tests/performance/.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


@pytest.mark.skipif(not FIXTURE.exists(), reason="tiny_synthetic.MOV fixture not present")
def test_local_runtime_single_process(tmp_path):
    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id="smoke",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    result = runtime.run(req)

    # Models load at most once per run.
    model_loads = [e for e in telemetry.events if e.payload.get("event") == "model_load"]
    assert len(model_loads) <= 4  # YOLO + DINOv2 + tracker + (any future) — one each, not per stage

    # Video opens at most once.
    decode_opens = [e for e in telemetry.events if e.payload.get("event") == "decode_open"]
    assert len(decode_opens) == 1, f"expected exactly one decode_open, saw {len(decode_opens)}"

    # All known stages emitted.
    finished = {e.payload["stage"] for e in telemetry.events if e.kind == "stage_finished"}
    for stage in ("sample", "detect", "novelty", "track", "refine", "score", "resolve", "fuse", "dedup", "store"):
        assert stage in finished, f"stage {stage} missing"

    assert result.manifest.runtime_mode == "cpu_debug"
```

- [ ] **Step 2: Run and confirm FAIL (import error or no fixture)**

```bash
python3 -m pytest tests/pipeline/test_runtime_smoke.py -v
```

- [ ] **Step 3: Create the synthetic fixture**

```bash
mkdir -p tests/fixtures
# Generate a 2-second 1920x1080 black video using ffmpeg if available, else skip.
ffmpeg -y -f lavfi -i color=black:s=1920x1080:d=2 -c:v libx264 -pix_fmt yuv420p tests/fixtures/tiny_synthetic.MOV 2>/dev/null \
  || echo "ffmpeg not available; smoke test will skip"
```

If ffmpeg unavailable, commit a tiny pre-made fixture via git LFS or document the manual creation step.

- [ ] **Step 4: Implement `LocalPipelineRuntime`**

`src/card_capture/pipeline/runtime_local.py`:

```python
"""LocalPipelineRuntime: executes all stages in one process.

This is the V5.5 replacement for the Metaflow flow. Stages run as direct
function calls; loaded models, decoded frames, and GPU-resident tensors
are passed between stages as in-memory objects.

The runtime selects an execution backend (StrictGpu / CpuDebug) based on
`request.runtime_mode`. Backend-specific decode and model loading live in
the backend modules; this orchestrator owns sequencing, telemetry, and
manifest construction.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
)
from card_capture.pipeline.telemetry import PipelineTelemetry, NoopTelemetry
from card_capture.pipeline.stages import (
    sample as stage_sample,
    detect as stage_detect,
    novelty as stage_novelty,
    track as stage_track,
    refine as stage_refine,
    score as stage_score,
    resolve as stage_resolve,
    fuse as stage_fuse,
    dedup as stage_dedup,
    store as stage_store,
)


_STAGES = (
    ("sample", stage_sample),
    ("detect", stage_detect),
    ("novelty", stage_novelty),
    ("track", stage_track),
    ("refine", stage_refine),
    ("score", stage_score),
    ("resolve", stage_resolve),
    ("fuse", stage_fuse),
    ("dedup", stage_dedup),
    ("store", stage_store),
)


class LocalPipelineRuntime:
    def __init__(self, telemetry: PipelineTelemetry | None = None) -> None:
        self._telemetry = telemetry or NoopTelemetry()

    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        timings: list[StageTiming] = []
        violations: list = []
        run_id = request.run_id or uuid.uuid4().hex[:12]

        # State carried across stages — frames, detections, tracks, crops, scores, etc.
        # The actual shape grows as Tasks 3.3-3.8 wire stages.
        state: dict = {"request": request}

        for name, module in _STAGES:
            self._telemetry.stage_started(name, {})
            start = time.perf_counter()
            try:
                module.run(state, telemetry=self._telemetry)
            except Exception as exc:
                violations.append({"code": f"stage_failed:{name}", "metadata": {"error": repr(exc)}})
                self._telemetry.contract_violation(
                    f"stage_failed:{name}", {"error": repr(exc)}
                )
                raise
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            timings.append(StageTiming(stage=name, elapsed_ms=elapsed_ms))
            self._telemetry.stage_finished(name, elapsed_ms, {})

        manifest = RunManifest(
            run_id=run_id,
            runtime_mode=request.runtime_mode,
            input_video=request.input_video,
            output_artifacts=state.get("output_artifacts", []),
            cards=state.get("cards", []),
            stage_timings=timings,
            contract_violations=violations,
            version="0.5.5+phase3",
        )
        return PipelineRunResult(manifest=manifest)
```

- [ ] **Step 5: Add no-op `run(state, telemetry)` to each stage stub**

For each `src/card_capture/pipeline/stages/<name>.py`, append:

```python
def run(state: dict, *, telemetry) -> None:
    """Placeholder — wired to real V4 implementation in Tasks 3.3-3.8."""
    return None
```

- [ ] **Step 6: Run the smoke test; PASS (or skip if no fixture)**

```bash
python3 -m pytest tests/pipeline/test_runtime_smoke.py -v
```

Expected: PASS at smoke-shape level, even though stages are no-ops. The assertions on `model_loads` and `decode_opens` will pass because zero ≤ N.

- [ ] **Step 7: Commit**

```bash
git add src/card_capture/pipeline/runtime_local.py src/card_capture/pipeline/stages/ tests/pipeline/test_runtime_smoke.py tests/fixtures/
git commit -m "feat(v55-phase3): LocalPipelineRuntime loop body + stage stubs"
```

---

### Task 3.3: Wire `sample`, `detect` stages with shared decode

**Files:**
- Modify: `src/card_capture/pipeline/stages/sample.py`
- Modify: `src/card_capture/pipeline/stages/detect.py`

- [ ] **Step 1: Implement `sample.py` by delegating to existing sampler**

`src/card_capture/pipeline/stages/sample.py`:

```python
"""Stage 1: Adaptive Presence Sampler.

Wraps src/card_capture/sampler/__init__.py. The key V5.5 change: we open
the input video ONCE and stash the open handle in state so detect/refine
can reuse it.
"""
from __future__ import annotations

from card_capture.sampler import run_sampler  # existing API; verify import path


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    video_path = request.input_video.removeprefix("artifact://local/")
    telemetry.resource_sample({"event": "decode_open", "path": video_path})
    sampler_result = run_sampler(video_path, config=request.config)
    state["sampler"] = sampler_result
    state["sampled_frames"] = sampler_result.frames  # GpuFrameBatch in strict, np array in cpu_debug
    state["video_path"] = video_path
```

Note: confirm `run_sampler` actually exists; if the signature differs, adapt. The point is: emit a single `decode_open` telemetry event and stash the result.

- [ ] **Step 2: Implement `detect.py` (reuses frames, loads model once)**

`src/card_capture/pipeline/stages/detect.py`:

```python
"""Stage 3: YOLO Corner Detection.

Reuses `state["sampled_frames"]` produced by the sample stage. Loads the
YOLO model once on first call and stashes it in state for any later stage
that needs it (none currently; refine uses its own model).
"""
from __future__ import annotations

from card_capture.detectors import load_yolo_model, detect_corners_batch


def run(state: dict, *, telemetry) -> None:
    if "yolo_model" not in state:
        telemetry.resource_sample({"event": "model_load", "model": "yolo_obb"})
        state["yolo_model"] = load_yolo_model(state["request"].config)
    frames = state["sampled_frames"]
    detections = detect_corners_batch(state["yolo_model"], frames, config=state["request"].config)
    state["detections"] = detections
```

Adapt API names to match `src/card_capture/detectors.py`. If `load_yolo_model` doesn't exist with this name, search and use the real one.

- [ ] **Step 3: Run smoke test; verify it still passes**

```bash
python3 -m pytest tests/pipeline/test_runtime_smoke.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/pipeline/stages/sample.py src/card_capture/pipeline/stages/detect.py
git commit -m "feat(v55-phase3): wire sample+detect stages with shared decode handle"
```

---

### Task 3.4: Wire `novelty`, `track`, `refine` (refine reuses frames — no re-decode)

**Files:**
- Modify: `src/card_capture/pipeline/stages/novelty.py`
- Modify: `src/card_capture/pipeline/stages/track.py`
- Modify: `src/card_capture/pipeline/stages/refine.py`
- Create: `tests/performance/test_no_redecode.py`

- [ ] **Step 1: Implement `novelty.py`**

```python
"""Stage 4: Background Novelty Gate."""
from __future__ import annotations

from card_capture.presence.background_novelty import gate_detections


def run(state: dict, *, telemetry) -> None:
    state["detections"] = gate_detections(
        state["detections"],
        state.get("sampled_frames"),
        config=state["request"].config,
    )
```

- [ ] **Step 2: Implement `track.py`**

```python
"""Stage 5: Session-Aware Tracking.

Tracker backend (BoT-SORT or ByteTrack) is selected from request.config.
"""
from __future__ import annotations

from card_capture.tracking.botsort_adapter import BotSortTracker
from card_capture.tracking.bytetrack_adapter import ByteTrackTracker


def run(state: dict, *, telemetry) -> None:
    cfg = state["request"].config
    backend = cfg.get("tracker_backend", "bytetrack")
    if backend == "botsort":
        tracker = BotSortTracker(cfg)
    else:
        tracker = ByteTrackTracker(cfg)
    state["tracks"] = tracker.assign(state["detections"], state["sampled_frames"])
```

- [ ] **Step 3: Implement `refine.py` — reuses `state["sampled_frames"]`**

```python
"""Stage 6: GPU Refinement (Kornia perspective warp -> 750x1050).

CRITICAL V5.5 CHANGE: reads frames from state["sampled_frames"], NOT from
disk. The V4 code re-decoded the source video here; V5.5 must never do
that. If state["sampled_frames"] is missing, that is a contract violation,
not a fallback path.
"""
from __future__ import annotations

from card_capture.gpu_refinement import refine_tracks_to_crops


def run(state: dict, *, telemetry) -> None:
    frames = state.get("sampled_frames")
    if frames is None:
        telemetry.contract_violation(
            "refine_without_frames",
            {"hint": "sample stage must populate state['sampled_frames']"},
        )
        raise RuntimeError("refine stage reached without sampled_frames in state")
    state["crops"] = refine_tracks_to_crops(
        state["tracks"], frames, config=state["request"].config
    )
```

- [ ] **Step 4: Write no-redecode test**

`tests/performance/test_no_redecode.py`:

```python
"""Assert that refine does not re-open the source video.

Uses InMemoryTelemetry to count `decode_open` events.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_refine_does_not_redecode(tmp_path):
    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id="r",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    runtime.run(req)

    opens = [
        e for e in telemetry.events
        if e.kind == "resource_sample" and e.payload.get("event") == "decode_open"
    ]
    assert len(opens) == 1, f"expected exactly 1 decode_open, saw {len(opens)}: {opens}"
```

- [ ] **Step 5: Run; PASS**

```bash
python3 -m pytest tests/pipeline/test_runtime_smoke.py tests/performance/test_no_redecode.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/pipeline/stages/novelty.py src/card_capture/pipeline/stages/track.py src/card_capture/pipeline/stages/refine.py tests/performance/test_no_redecode.py
git commit -m "feat(v55-phase3): wire novelty+track+refine with shared frames (no re-decode)"
```

---

### Task 3.5: Wire `score`, `resolve`, `fuse` (fuse: in-process loop, no foreach)

**Files:**
- Modify: `src/card_capture/pipeline/stages/score.py`
- Modify: `src/card_capture/pipeline/stages/resolve.py`
- Modify: `src/card_capture/pipeline/stages/fuse.py`

- [ ] **Step 1: Implement `score.py`**

```python
"""Stage 7: Quality Scoring."""
from __future__ import annotations

from card_capture.scoring import score_crops


def run(state: dict, *, telemetry) -> None:
    state["scored"] = score_crops(state["crops"], state["tracks"], config=state["request"].config)
```

- [ ] **Step 2: Implement `resolve.py`**

```python
"""Stage 8: Front/Back Resolution."""
from __future__ import annotations

from card_capture.identity import resolve_front_back


def run(state: dict, *, telemetry) -> None:
    state["resolved"] = resolve_front_back(state["scored"], config=state["request"].config)
```

- [ ] **Step 3: Implement `fuse.py` (CRITICAL: in-process loop, no subprocess foreach)**

```python
"""Stage 9: Lighting-Diverse Fusion.

V5.5 change: this stage was a Metaflow `foreach` that spawned one subprocess
per track (~4-6 minutes overhead on the reference video). V5.5 runs the
fusion loop in-process via a plain `for` loop. The fusion algorithm is
unchanged — see src/card_capture/fuser.py.
"""
from __future__ import annotations

from card_capture.fuser import fuse_track


def run(state: dict, *, telemetry) -> None:
    fused = []
    for track_id, candidates in state["resolved"].items():
        fused.append(fuse_track(track_id, candidates, config=state["request"].config))
        # Per-track telemetry so dashboards still see the same shape.
        telemetry.stage_finished("fuse_track", 0, {"track_id": track_id})
    state["fused"] = fused
```

- [ ] **Step 4: Run smoke; PASS**

```bash
python3 -m pytest tests/pipeline/test_runtime_smoke.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/
git commit -m "feat(v55-phase3): wire score+resolve+fuse (in-process fuse loop)"
```

---

### Task 3.6: Wire `dedup`, `store`

**Files:**
- Modify: `src/card_capture/pipeline/stages/dedup.py`
- Modify: `src/card_capture/pipeline/stages/store.py`

- [ ] **Step 1: Implement `dedup.py`**

```python
"""Stage 10: Global Dedup."""
from __future__ import annotations

from card_capture.deduplicator import dedupe_fused


def run(state: dict, *, telemetry) -> None:
    state["final_cards"] = dedupe_fused(state["fused"], config=state["request"].config)
```

- [ ] **Step 2: Implement `store.py`**

```python
"""Stage 10b: Storage.

In Phase 3 this stage still writes via card_capture.storage. Phase 4
will replace direct SQL with card_capture.data repositories.
"""
from __future__ import annotations

from card_capture.storage import store_cards


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    out_paths = store_cards(state["final_cards"], request.output_root, config=request.config)
    state["cards"] = state["final_cards"]
    state["output_artifacts"] = out_paths
```

- [ ] **Step 3: Run smoke; PASS**

```bash
python3 -m pytest tests/pipeline/test_runtime_smoke.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/pipeline/stages/dedup.py src/card_capture/pipeline/stages/store.py
git commit -m "feat(v55-phase3): wire dedup+store stages"
```

---

### Task 3.7: Replace `card-capture process` CLI to use `LocalPipelineRuntime`

**Files:**
- Modify: `src/card_capture/cli.py`

- [ ] **Step 1: Find the existing `process` subcommand**

Run:
```bash
grep -n "def.*process\|add_parser.*process" src/card_capture/cli.py | head -10
```

- [ ] **Step 2: Replace its body to call `LocalPipelineRuntime.run(...)` instead of invoking Metaflow**

Edit `src/card_capture/cli.py`. The new `process` handler:

```python
def _cmd_process(args) -> int:
    from card_capture.pipeline.request import PipelineRunRequest
    from card_capture.pipeline.runtime_local import LocalPipelineRuntime
    from card_capture.pipeline.telemetry import InMemoryTelemetry

    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id=args.run_id or uuid.uuid4().hex[:12],
        input_video=f"artifact://local/{args.video}",
        output_root=f"artifact://local/{args.output_dir}/",
        runtime_mode=args.runtime_mode,
        config={"corner_confidence": args.corner_confidence} if args.corner_confidence else {},
    )
    result = runtime.run(req)
    print(result.manifest.to_json())
    return 0
```

Update the subparser to expose `--runtime-mode {strict_gpu, cpu_debug}`.

- [ ] **Step 3: Verify CLI works**

```bash
card-capture process tests/fixtures/tiny_synthetic.MOV --output-dir /tmp/v55-cli --runtime-mode cpu_debug
```

Expected: JSON manifest printed to stdout.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/cli.py
git commit -m "feat(v55-phase3): card-capture process CLI uses LocalPipelineRuntime"
```

---

### Task 3.8: Real-video performance baseline

**Files:**
- Modify: `harness/performance/runner.py` (add `local_v55` profile)
- Create: `tests/performance/test_local_baseline.py`

- [ ] **Step 1: Add a `local_v55` profile**

Edit `harness/performance/runner.py`, replace the `synthetic_smoke`-only branch with:

```python
if profile == "synthetic_smoke":
    # ... existing code unchanged
elif profile == "local_v55":
    from card_capture.pipeline.request import PipelineRunRequest
    from card_capture.pipeline.runtime_local import LocalPipelineRuntime
    from card_capture.pipeline.telemetry import InMemoryTelemetry

    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id=run_id,
        input_video=f"artifact://local/{video}",
        output_root=f"artifact://local/{out_dir / run_id}/",
        runtime_mode="cpu_debug",  # use strict_gpu when running on GPU
    )
    pipeline_start = time.perf_counter()
    result = runtime.run(req)
    timings["__pipeline__"] = (time.perf_counter() - pipeline_start) * 1000.0

    for st in result.manifest.stage_timings:
        timings[st.stage] = float(st.elapsed_ms)

    counters["frames_decoded"] = sum(
        1 for e in telemetry.events if e.payload.get("event") == "frame_decoded"
    )
    counters["model_loads"] = sum(
        1 for e in telemetry.events if e.payload.get("event") == "model_load"
    )
    counters["video_reopens"] = sum(
        1 for e in telemetry.events if e.payload.get("event") == "decode_open"
    ) - 1  # one is expected
    cards = len(result.manifest.cards)
else:
    raise ValueError(f"unknown perf profile: {profile!r}")
```

- [ ] **Step 2: Write a baseline test (manual, gated by benchmark marker)**

`tests/performance/test_local_baseline.py`:

```python
"""Real-video baseline. Run manually:
    python3 -m pytest tests/performance/test_local_baseline.py -m benchmark
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.performance.runner import run


BASELINE_VIDEO = Path(os.environ.get("V55_BASELINE_VIDEO", "/nonexistent.MOV"))


@pytest.mark.benchmark
@pytest.mark.skipif(not BASELINE_VIDEO.exists(), reason="V55_BASELINE_VIDEO not set or missing")
def test_local_v55_runs_in_one_process(tmp_path):
    report = run(profile="local_v55", video=str(BASELINE_VIDEO), out_dir=tmp_path)
    assert report.error is None
    assert report.counters["video_reopens"] == 0, "refine must not re-open the video"
    assert report.counters["model_loads"] <= 4, "models must load once per run"
```

- [ ] **Step 3: Run baseline manually with the reference video**

```bash
V55_BASELINE_VIDEO=/path/to/IMG_5922.MOV python3 -m pytest tests/performance/test_local_baseline.py -m benchmark -v -s
```

Document the timing in `docs/superpowers/plans/v5-5/baseline-results.md`.

- [ ] **Step 4: Commit**

```bash
git add harness/performance/runner.py tests/performance/test_local_baseline.py docs/superpowers/plans/v5-5/
git commit -m "perf(v55-phase3): local_v55 perf profile + baseline test"
```

---

### Task 3.9: Expand equivalence contract test with numerical tolerances

**Files:**
- Modify: `tests/runtime/test_cpu_debug_strict_gpu_equivalence.py`

Phase 2 shipped this test with shape-only assertions because stages were stubs. Now that stages execute real algorithms, expand the test to enforce the per-field-class tolerance table from spec Section 6.

- [ ] **Step 1: Add field-class assertions**

Replace the body of the existing equivalence test with:

```python
"""Equivalence contract per spec Section 6 table.

Field classes:
- Identity / schema: exact
- Counts: exact
- Geometric (corners): ± 2 px
- Quality scores: ± 5% (relative) / ± 0.02 (absolute for `total`)
- pHash: Hamming distance ≤ 4 bits
- ReID embeddings: cosine similarity ≥ 0.95
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.runtime.cpu_debug import CpuDebugRuntime
from card_capture.runtime.strict_gpu import StrictGpuRuntime
from card_capture.runtime.gpu_session import MissingGpuError


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _cosine(a: list[float], b: list[float]) -> float:
    A = np.asarray(a, dtype=np.float32); B = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(A) * np.linalg.norm(B)) or 1.0
    return float(np.dot(A, B) / denom)


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_cpu_debug_strict_gpu_equivalence(tmp_path):
    try:
        gpu_runtime = StrictGpuRuntime()
    except MissingGpuError:
        pytest.skip("no GPU available")

    base = dict(input_video=f"artifact://local/{FIXTURE}",
                output_root=f"artifact://local/{tmp_path}/",
                config={})
    cpu = CpuDebugRuntime().run(PipelineRunRequest(
        run_id="eq-cpu", runtime_mode="cpu_debug", **base
    )).manifest
    gpu = gpu_runtime.run(PipelineRunRequest(
        run_id="eq-gpu", runtime_mode="strict_gpu", **base
    )).manifest

    # 1. Identity / schema: exact.
    assert sorted(vars(cpu).keys()) == sorted(vars(gpu).keys())

    # 2. Counts: exact.
    assert len(cpu.cards) == len(gpu.cards), \
        f"card count differs: cpu={len(cpu.cards)} gpu={len(gpu.cards)}"

    # Pair cards by card_instance_id (same input video -> same identity field).
    cpu_by_id = {c.card_instance_id: c for c in cpu.cards}
    for g in gpu.cards:
        c = cpu_by_id.get(g.card_instance_id)
        assert c is not None, f"GPU card {g.card_instance_id} missing from CPU run"

        # 3. Quality scores: ±5% relative, ±0.02 absolute on `total`.
        for metric in ("sharpness", "glare", "aspect_ratio", "size", "complexity", "border_purity"):
            if metric in c.quality and metric in g.quality:
                cv, gv = c.quality[metric], g.quality[metric]
                if cv == 0:
                    assert abs(gv) < 0.02, f"{metric}: cpu=0, gpu={gv}"
                else:
                    rel = abs(gv - cv) / abs(cv)
                    assert rel <= 0.05, f"{metric}: cpu={cv} gpu={gv} rel={rel:.3f} > 0.05"
        if "total" in c.quality and "total" in g.quality:
            assert abs(c.quality["total"] - g.quality["total"]) <= 0.02

        # 4. pHash: Hamming distance ≤ 4 bits (if both runtimes computed it).
        cpu_phash = c.quality.get("phash_hex")
        gpu_phash = g.quality.get("phash_hex")
        if isinstance(cpu_phash, str) and isinstance(gpu_phash, str):
            assert _hamming_hex(cpu_phash, gpu_phash) <= 4

        # 5. ReID embeddings: cosine ≥ 0.95 (if both runtimes computed it).
        cpu_emb = c.quality.get("reid_embedding")
        gpu_emb = g.quality.get("reid_embedding")
        if isinstance(cpu_emb, list) and isinstance(gpu_emb, list):
            assert _cosine(cpu_emb, gpu_emb) >= 0.95
```

- [ ] **Step 2: Run; PASS (or skip on machines without GPU)**

```bash
python3 -m pytest tests/runtime/test_cpu_debug_strict_gpu_equivalence.py -v
```

If the test fails because per-card quality fields differ more than tolerance, the equivalence contract is being violated by stage code — investigate before continuing. Do not weaken the tolerances to make the test pass.

- [ ] **Step 3: Commit**

```bash
git add tests/runtime/test_cpu_debug_strict_gpu_equivalence.py
git commit -m "test(v55-phase3): expand equivalence contract with numerical tolerances"
```

---

### Task 3.10: Delete Metaflow

**Files:**
- Delete: `pipeline/card_capture_flow.py`
- Delete: `pipeline/steps/*.py`
- Modify: `pipeline/__init__.py` (clear contents)
- Modify: `pipeline/contracts.py` (re-export from `card_capture.pipeline.request` for transition)
- Modify: `pyproject.toml` (remove metaflow dep)
- Create: `tests/architecture/test_metaflow_absent.py`

- [ ] **Step 1: Verify nothing imports from `pipeline.card_capture_flow` or `pipeline.steps`**

Run:
```bash
grep -rn "from pipeline\.\(card_capture_flow\|steps\)" src/ app/ tests/ harness/ 2>/dev/null
grep -rn "import pipeline\.\(card_capture_flow\|steps\)" src/ app/ tests/ harness/ 2>/dev/null
```

Expected: no matches. If anything imports from these paths, repoint to `card_capture.pipeline.stages` first.

- [ ] **Step 2: Delete the Metaflow code**

```bash
rm pipeline/card_capture_flow.py
rm -r pipeline/steps/
```

- [ ] **Step 3: Convert `pipeline/__init__.py` to a deprecation shim**

```python
"""DEPRECATED: V4 Metaflow pipeline package.

V5.5 replaces this with `card_capture.pipeline`. This module exists only to
keep `pipeline.contracts` imports working through Phase 3-Phase 4 transition.
"""
```

- [ ] **Step 4: Update `pipeline/contracts.py` to re-export from new location**

```python
"""Transitional shim. New code should import from card_capture.pipeline.request."""
from card_capture.pipeline.request import (  # noqa: F401
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
    ContractViolation,
    CardRecord,
)
```

- [ ] **Step 5: Remove metaflow from pyproject**

Edit `pyproject.toml` — remove `metaflow` from any `[project.dependencies]` or optional-dependencies block. Search:

```bash
grep -n metaflow pyproject.toml
```

Remove every match.

- [ ] **Step 6: Add Import Linter contract forbidding metaflow**

Edit `.importlinter`, add:

```ini
[importlinter:contract:no-metaflow]
name = metaflow must not be imported anywhere
type = forbidden
source_modules =
    card_capture
    app
    pipeline
    harness
forbidden_modules =
    metaflow
```

- [ ] **Step 7: Add archtest asserting metaflow is absent**

`tests/architecture/test_metaflow_absent.py`:

```python
"""Verify Metaflow has been removed (V5.5 Phase 3)."""
from __future__ import annotations

import subprocess


def test_no_metaflow_imports():
    result = subprocess.run(
        ["grep", "-rn", "import metaflow\\|from metaflow", "src/", "app/", "pipeline/", "harness/", "tests/"],
        capture_output=True, text=True, check=False,
    )
    # grep returns 1 when no matches; that's success.
    assert result.returncode == 1, f"metaflow imports remain:\n{result.stdout}"


def test_metaflow_not_in_dependencies():
    import pathlib, sys
    pyproject = pathlib.Path("pyproject.toml").read_text()
    assert "metaflow" not in pyproject.lower(), "metaflow still in pyproject.toml"
```

- [ ] **Step 8: Run all architecture tests; PASS**

```bash
python3 -m pytest tests/architecture/ -v
```

- [ ] **Step 9: Commit and tag**

```bash
git add -A pipeline/ .importlinter pyproject.toml tests/architecture/test_metaflow_absent.py
git commit -m "feat(v55-phase3): remove Metaflow; pipeline runs in one process"
git tag v55-phase3-complete
```

**Phase 3 complete.** Local runs are single-process. Metaflow is gone. The re-decode is eliminated. Models load once per run.

---

# Phase 4: Data Access Layer + Single-Writer

**Goal:** Centralize SQLite access behind `card_capture.data` repositories and a single-writer queue covering pipeline, FastAPI, and harness writers. Tighten raw-SQL scanner to blocking.

**Acceptance:** No raw SQL or `sqlite3.connect` outside `card_capture.data`, migrations, and allowlisted test helpers. Lock-contention regression test passes. Standards rule R-DATA-1..4 blocking.

---

### Task 4.1: Create `card_capture.data` package + connection helper

**Files:**
- Create: `src/card_capture/data/__init__.py`
- Create: `src/card_capture/data/connection.py`

- [ ] **Step 1: Package init**

```python
"""V5.5 data access layer. All SQLite reads/writes route through this package."""
```

- [ ] **Step 2: Connection helper**

`src/card_capture/data/connection.py`:

```python
"""SQLite connection management.

Connections use WAL mode and a 5-second busy_timeout. Writers must route
through card_capture.data.writer (Task 4.2) to ensure serialization;
direct write usage is allowed only inside this package for the writer's
worker thread.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def open_connection(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode={'ro' if read_only else 'rwc'}"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def read_connection(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = open_connection(db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 3: Commit**

```bash
git add src/card_capture/data/__init__.py src/card_capture/data/connection.py
git commit -m "feat(v55-phase4): card_capture.data package + connection helper"
```

---

### Task 4.2: Single-writer queue

**Files:**
- Create: `src/card_capture/data/writer.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_writer_serializes.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_writer_serializes.py`:

```python
"""Writer queue serializes concurrent writes from multiple submitters."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer, Write


def _init_db(path: Path) -> None:
    conn = open_connection(path)
    conn.execute("CREATE TABLE IF NOT EXISTS counts (k TEXT PRIMARY KEY, n INTEGER)")
    conn.execute("INSERT OR REPLACE INTO counts(k, n) VALUES ('total', 0)")
    conn.close()


def test_writer_serializes_concurrent_increments(tmp_path):
    db = tmp_path / "wtest.db"
    _init_db(db)

    writer = Writer(db_path=db)
    writer.start()
    try:
        def submit_increment():
            writer.submit(Write(
                sql="UPDATE counts SET n = n + 1 WHERE k = 'total'",
                params=(),
            ))

        threads = [threading.Thread(target=submit_increment) for _ in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        writer.flush()
    finally:
        writer.stop()

    conn = open_connection(db, read_only=True)
    n = conn.execute("SELECT n FROM counts WHERE k='total'").fetchone()[0]
    assert n == 50, f"expected 50 increments, got {n} (write was not serialized)"


def test_writer_processes_in_submit_order(tmp_path):
    db = tmp_path / "order.db"
    conn = open_connection(db)
    conn.execute("CREATE TABLE events (i INTEGER PRIMARY KEY AUTOINCREMENT, v INTEGER)")
    conn.close()

    writer = Writer(db_path=db)
    writer.start()
    try:
        for v in range(20):
            writer.submit(Write(sql="INSERT INTO events(v) VALUES (?)", params=(v,)))
        writer.flush()
    finally:
        writer.stop()

    conn = open_connection(db, read_only=True)
    rows = conn.execute("SELECT v FROM events ORDER BY i").fetchall()
    assert [r[0] for r in rows] == list(range(20))
```

- [ ] **Step 2: Run; FAIL**

```bash
python3 -m pytest tests/data/test_writer_serializes.py -v
```

- [ ] **Step 3: Implement `writer.py`**

`src/card_capture/data/writer.py`:

```python
"""Single-writer queue for SQLite.

Spec Section 5: SQLite WAL allows concurrent readers but one writer at a
time. V5.5 routes all writes — from pipeline runtime, FastAPI handlers,
and harness — through this writer. An in-process queue serializes writes
on a dedicated worker thread; the worker holds the only write connection.

For cross-process callers (e.g., the FastAPI app in a separate uvicorn
worker), the same `Writer` API can be backed by an IPC queue in a later
iteration; the public submit/flush/stop surface is process-agnostic.
"""
from __future__ import annotations

import dataclasses
import queue
import sqlite3
import threading
from pathlib import Path

from .connection import open_connection


@dataclasses.dataclass(frozen=True)
class Write:
    sql: str
    params: tuple = ()


_SENTINEL = object()


class Writer:
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name="card-capture-writer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._thread is None:
                return
            self._q.put(_SENTINEL)
            self._thread.join()
            self._thread = None
            if self._error is not None:
                err, self._error = self._error, None
                raise err

    def submit(self, write: Write) -> None:
        if self._thread is None:
            raise RuntimeError("Writer.start() before submit()")
        self._q.put(write)

    def flush(self) -> None:
        """Block until the queue is empty (best-effort)."""
        self._q.join()

    def _loop(self) -> None:
        conn = open_connection(self._db_path)
        try:
            while True:
                item = self._q.get()
                try:
                    if item is _SENTINEL:
                        return
                    write: Write = item
                    conn.execute(write.sql, write.params)
                except BaseException as exc:  # noqa: BLE001
                    self._error = exc
                    return
                finally:
                    self._q.task_done()
        finally:
            conn.close()
```

- [ ] **Step 4: Run; PASS**

```bash
python3 -m pytest tests/data/test_writer_serializes.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/data/writer.py tests/data/
git commit -m "feat(v55-phase4): single-writer queue serializes SQLite writes"
```

---

### Task 4.3: `runs` repository

**Files:**
- Create: `src/card_capture/data/repositories/__init__.py`
- Create: `src/card_capture/data/repositories/runs.py`
- Create: `tests/data/test_runs_repository.py`

- [ ] **Step 1: Empty `__init__.py`**

```bash
touch src/card_capture/data/repositories/__init__.py
```

- [ ] **Step 2: Write failing test**

`tests/data/test_runs_repository.py`:

```python
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.runs import RunsRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE pipeline_runs(
            run_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            state TEXT NOT NULL,
            started_at_ms INTEGER,
            completed_at_ms INTEGER,
            cards_extracted INTEGER,
            error TEXT
        )
    """)
    conn.close()


def test_mark_started_then_completed(tmp_path):
    db = tmp_path / "r.db"
    _init_schema(db)
    writer = Writer(db)
    writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=db)
        repo.mark_started(run_id="r1", video_id="v1")
        repo.mark_completed(run_id="r1", cards_extracted=12)
        writer.flush()
        row = repo.get("r1")
    finally:
        writer.stop()
    assert row["state"] == "completed"
    assert row["cards_extracted"] == 12


def test_mark_failed_records_error(tmp_path):
    db = tmp_path / "rf.db"
    _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=db)
        repo.mark_started("r2", "v2")
        repo.mark_failed("r2", error="boom")
        writer.flush()
        row = repo.get("r2")
    finally:
        writer.stop()
    assert row["state"] == "failed"
    assert row["error"] == "boom"
```

- [ ] **Step 3: Run; FAIL**

```bash
python3 -m pytest tests/data/test_runs_repository.py -v
```

- [ ] **Step 4: Implement `runs.py`**

```python
"""pipeline_runs repository."""
from __future__ import annotations

import time
from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class RunsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def mark_started(self, run_id: str, video_id: str) -> None:
        now = int(time.time() * 1000)
        self._writer.submit(Write(
            sql="""
                INSERT OR REPLACE INTO pipeline_runs(run_id, video_id, state, started_at_ms)
                VALUES (?, ?, 'started', ?)
            """,
            params=(run_id, video_id, now),
        ))

    def mark_completed(self, run_id: str, cards_extracted: int) -> None:
        now = int(time.time() * 1000)
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs
                SET state='completed', completed_at_ms=?, cards_extracted=?
                WHERE run_id=?
            """,
            params=(now, cards_extracted, run_id),
        ))

    def mark_failed(self, run_id: str, error: str) -> None:
        now = int(time.time() * 1000)
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs SET state='failed', completed_at_ms=?, error=? WHERE run_id=?
            """,
            params=(now, error, run_id),
        ))

    def get(self, run_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT run_id, video_id, state, started_at_ms, completed_at_ms, cards_extracted, error "
                "FROM pipeline_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            keys = ("run_id", "video_id", "state", "started_at_ms", "completed_at_ms", "cards_extracted", "error")
            return dict(zip(keys, row))
```

- [ ] **Step 5: Run; PASS**

```bash
python3 -m pytest tests/data/test_runs_repository.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/data/repositories/
git commit -m "feat(v55-phase4): RunsRepository"
```

---

### Task 4.4: `events` repository

**Files:**
- Create: `src/card_capture/data/repositories/events.py`
- Create: `tests/data/test_events_repository.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path
import json

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.events import EventsRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE pipeline_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            video_id TEXT,
            stage TEXT NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            metadata TEXT
        )
    """)
    conn.close()


def test_record_stage_finished_and_list(tmp_path):
    db = tmp_path / "e.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = EventsRepository(writer=writer, db_path=db)
        repo.record_stage_finished("r1", "v1", "detect", 1234, {"frames": 100})
        repo.record_stage_finished("r1", "v1", "refine", 5678, {})
        writer.flush()
        rows = repo.list_for_run("r1")
    finally:
        writer.stop()
    assert [r["stage"] for r in rows] == ["detect", "refine"]
    assert json.loads(rows[0]["metadata"])["frames"] == 100
```

- [ ] **Step 2: Run + FAIL**

```bash
python3 -m pytest tests/data/test_events_repository.py -v
```

- [ ] **Step 3: Implement**

```python
"""pipeline_events repository."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class EventsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def record_stage_finished(
        self, run_id: str, video_id: str | None, stage: str,
        elapsed_ms: int, metadata: Mapping[str, object],
    ) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT INTO pipeline_events(run_id, video_id, stage, elapsed_ms, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
            params=(run_id, video_id, stage, elapsed_ms, json.dumps(dict(metadata))),
        ))

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, video_id, stage, elapsed_ms, metadata FROM pipeline_events "
                "WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        keys = ("run_id", "video_id", "stage", "elapsed_ms", "metadata")
        return [dict(zip(keys, r)) for r in rows]
```

- [ ] **Step 4: Run + PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(v55-phase4): EventsRepository"`

---

### Task 4.5: `cards` repository

**Files:**
- Create: `src/card_capture/data/repositories/cards.py`
- Create: `tests/data/test_cards_repository.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.cards import CardsRepository
from card_capture.pipeline.request import CardRecord


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE card_instances(
            card_instance_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            front_crop TEXT NOT NULL,
            back_crop TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE card_views(
            card_instance_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (card_instance_id, metric)
        )
    """)
    conn.close()


def test_store_and_get(tmp_path):
    db = tmp_path / "c.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = CardsRepository(writer=writer, db_path=db)
        repo.store_final_cards("r1", [
            CardRecord(
                card_instance_id="card_0",
                front_crop="artifact://local/r1/crops/card_0.png",
                back_crop=None,
                quality={"sharpness": 12.3, "glare": 0.05},
            ),
        ])
        writer.flush()
        cards = repo.list_for_run("r1")
    finally:
        writer.stop()
    assert len(cards) == 1
    assert cards[0]["card_instance_id"] == "card_0"
    assert cards[0]["quality"]["sharpness"] == 12.3
```

- [ ] **Step 2: Run + FAIL**
- [ ] **Step 3: Implement**

```python
"""card_instances + card_views repository."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write
from card_capture.pipeline.request import CardRecord


class CardsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def store_final_cards(self, run_id: str, cards: Iterable[CardRecord]) -> None:
        for c in cards:
            self._writer.submit(Write(
                sql="""
                    INSERT OR REPLACE INTO card_instances(card_instance_id, run_id, front_crop, back_crop)
                    VALUES (?, ?, ?, ?)
                """,
                params=(c.card_instance_id, run_id, c.front_crop, c.back_crop),
            ))
            for metric, value in c.quality.items():
                self._writer.submit(Write(
                    sql="""
                        INSERT OR REPLACE INTO card_views(card_instance_id, metric, value)
                        VALUES (?, ?, ?)
                    """,
                    params=(c.card_instance_id, metric, float(value)),
                ))

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT card_instance_id, front_crop, back_crop FROM card_instances WHERE run_id=?",
                (run_id,),
            ).fetchall()
            out = []
            for cid, front, back in rows:
                quality = dict(conn.execute(
                    "SELECT metric, value FROM card_views WHERE card_instance_id=?", (cid,)
                ).fetchall())
                out.append({
                    "card_instance_id": cid,
                    "front_crop": front,
                    "back_crop": back,
                    "quality": quality,
                })
            return out

    def get(self, card_instance_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT card_instance_id, run_id, front_crop, back_crop FROM card_instances WHERE card_instance_id=?",
                (card_instance_id,),
            ).fetchone()
            if row is None:
                return None
            quality = dict(conn.execute(
                "SELECT metric, value FROM card_views WHERE card_instance_id=?", (card_instance_id,)
            ).fetchall())
            return {"card_instance_id": row[0], "run_id": row[1],
                    "front_crop": row[2], "back_crop": row[3], "quality": quality}
```

- [ ] **Step 4: Run + PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(v55-phase4): CardsRepository"`

---

### Task 4.6: `videos` repository

**Files:**
- Create: `src/card_capture/data/repositories/videos.py`
- Create: `tests/data/test_videos_repository.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.videos import VideosRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE videos(
            video_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            registered_at_ms INTEGER NOT NULL,
            metadata TEXT
        )
    """)
    conn.close()


def test_register_and_get(tmp_path):
    db = tmp_path / "v.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = VideosRepository(writer=writer, db_path=db)
        repo.register("v1", "/path/to/v1.MOV", {"duration_s": 60})
        writer.flush()
        row = repo.get("v1")
    finally:
        writer.stop()
    assert row["video_id"] == "v1"
    assert row["metadata"]["duration_s"] == 60
```

- [ ] **Step 2: Run + FAIL**
- [ ] **Step 3: Implement**

```python
"""videos repository."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class VideosRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def register(self, video_id: str, path: str, metadata: Mapping[str, object]) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT OR REPLACE INTO videos(video_id, path, registered_at_ms, metadata)
                VALUES (?, ?, ?, ?)
            """,
            params=(video_id, path, int(time.time() * 1000), json.dumps(dict(metadata))),
        ))

    def get(self, video_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT video_id, path, registered_at_ms, metadata FROM videos WHERE video_id=?",
                (video_id,),
            ).fetchone()
            if row is None:
                return None
            return {"video_id": row[0], "path": row[1],
                    "registered_at_ms": row[2], "metadata": json.loads(row[3] or "{}")}

    def list_recent(self, limit: int = 50) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT video_id, path, registered_at_ms, metadata FROM videos "
                "ORDER BY registered_at_ms DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"video_id": r[0], "path": r[1], "registered_at_ms": r[2],
                 "metadata": json.loads(r[3] or "{}")} for r in rows]
```

- [ ] **Step 4: Run + PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(v55-phase4): VideosRepository"`

---

### Task 4.7: `labeling` repository

**Files:**
- Create: `src/card_capture/data/repositories/labeling.py`
- Create: `tests/data/test_labeling_repository.py`

The existing schema for truth/labeling lives in migrations `0006_fb_labels_no_card.sql`, `0007_card_instances_hidden.sql`. Inspect them first to match column names exactly.

- [ ] **Step 1: Inspect existing schema**

```bash
cat migrations/0006_fb_labels_no_card.sql
cat migrations/0007_card_instances_hidden.sql
```

- [ ] **Step 2: Write failing test** — use the actual fb_labels / truth_files column names from the schema.

```python
# Skeleton; adapt column names to migrations 0006-0007.
from pathlib import Path
from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.labeling import LabelingRepository


def test_store_and_query_fb_label(tmp_path):
    db = tmp_path / "lab.db"
    # Apply migrations 0006, 0007 against tmp_path/lab.db (use existing migrations.run_migrations or inline CREATE TABLE).
    writer = Writer(db); writer.start()
    try:
        repo = LabelingRepository(writer=writer, db_path=db)
        repo.store_fb_label(video_id="v1", frame_index=42, label="front")
        writer.flush()
        labels = repo.list_for_video("v1")
    finally:
        writer.stop()
    assert any(l["label"] == "front" and l["frame_index"] == 42 for l in labels)
```

- [ ] **Step 3: Implement `LabelingRepository`** with methods: `store_fb_label(video_id, frame_index, label, ...)`, `list_for_video(video_id)`, `get_truth_payload(video_id)`, `list_unlabeled(limit)`. Use the schema's exact column names.

- [ ] **Step 4: Run + PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(v55-phase4): LabelingRepository"`

---

### Task 4.8: `telemetry` repository

**Files:**
- Create: `src/card_capture/data/repositories/telemetry.py`
- Create: `tests/data/test_telemetry_repository.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path
import json

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.telemetry import TelemetryRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE telemetry_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            at_ms INTEGER NOT NULL
        )
    """)
    conn.close()


def test_record_and_list(tmp_path):
    db = tmp_path / "t.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = TelemetryRepository(writer=writer, db_path=db)
        repo.record_event(run_id="r1", kind="resource_sample", payload={"vram_mb": 4096})
        writer.flush()
        events = repo.list_for_run("r1")
    finally:
        writer.stop()
    assert events[0]["kind"] == "resource_sample"
    assert json.loads(events[0]["payload"])["vram_mb"] == 4096
```

- [ ] **Step 2: Run + FAIL**
- [ ] **Step 3: Implement**

```python
"""telemetry_events repository (durable mirror of in-memory telemetry)."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class TelemetryRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def record_event(self, run_id: str | None, kind: str, payload: Mapping[str, object]) -> None:
        self._writer.submit(Write(
            sql="INSERT INTO telemetry_events(run_id, kind, payload, at_ms) VALUES (?, ?, ?, ?)",
            params=(run_id, kind, json.dumps(dict(payload)), int(time.time() * 1000)),
        ))

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, kind, payload, at_ms FROM telemetry_events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [{"run_id": r[0], "kind": r[1], "payload": r[2], "at_ms": r[3]} for r in rows]
```

- [ ] **Step 4: Run + PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(v55-phase4): TelemetryRepository"`

When in doubt about real table column names anywhere in Tasks 4.4–4.8, inspect the canonical schema in `migrations/0001_v4_schema.sql` through `migrations/0012_*.sql`.

---

### Task 4.9: Migrate pipeline `store` stage to repositories

**Files:**
- Modify: `src/card_capture/pipeline/stages/store.py`
- Modify: `src/card_capture/pipeline/runtime_local.py` (instantiate Writer + repositories)

- [ ] **Step 1: Update `LocalPipelineRuntime` to instantiate Writer + repos and pass them via state**

Edit `src/card_capture/pipeline/runtime_local.py`. In `__init__`, accept an optional `db_path`. In `run()`, instantiate `Writer(db_path).start()`, build `RunsRepository`, `EventsRepository`, `CardsRepository`, stash into state, and call `writer.stop()` in a finally block.

- [ ] **Step 2: Rewrite `store.py` to use `CardsRepository`**

```python
"""Stage 10b: Storage via repository."""
from __future__ import annotations


def run(state: dict, *, telemetry) -> None:
    cards_repo = state["repos"]["cards"]
    runs_repo = state["repos"]["runs"]

    cards_repo.store_final_cards(state["request"].run_id, state["final_cards"])
    runs_repo.mark_completed(state["request"].run_id, cards_extracted=len(state["final_cards"]))
    state["cards"] = state["final_cards"]
    state["output_artifacts"] = []  # populated by export-boundary helpers, not store
```

- [ ] **Step 3: Run smoke test; PASS**

```bash
python3 -m pytest tests/pipeline/test_runtime_smoke.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/pipeline/stages/store.py src/card_capture/pipeline/runtime_local.py
git commit -m "feat(v55-phase4): pipeline store stage uses repositories"
```

---

### Task 4.10: Migrate FastAPI writers to repositories + writer

**Files:**
- Modify: each file under `app/api/` and `app/services/` that opens a SQLite connection
- Modify: `app/main.py` (instantiate a process-wide Writer)

- [ ] **Step 1: Inventory app-side writes**

Run:
```bash
grep -rn "sqlite3.connect\|conn.execute.*INSERT\|conn.execute.*UPDATE\|conn.execute.*DELETE" app/ | grep -v "^app/web/"
```

For each match, identify which repository owns that write.

- [ ] **Step 2: Wire the process-wide Writer into FastAPI's lifespan**

Edit `app/main.py`:

```python
from contextlib import asynccontextmanager
from card_capture.data.writer import Writer

@asynccontextmanager
async def lifespan(app):
    writer = Writer(db_path=app.state.db_path)
    writer.start()
    app.state.writer = writer
    try:
        yield
    finally:
        writer.stop()

# Wire lifespan into FastAPI(...)
```

- [ ] **Step 3: Replace each direct write call with a repository call using `request.app.state.writer`**

This is mechanical — one file at a time. For each:

```python
# Before
conn = sqlite3.connect(db_path)
conn.execute("UPDATE pipeline_runs SET state='cancelled' WHERE run_id=?", (run_id,))

# After
repo = RunsRepository(writer=request.app.state.writer, db_path=db_path)
repo.mark_cancelled(run_id)
```

Add `mark_cancelled` to RunsRepository if missing.

- [ ] **Step 4: Run app + repository tests**

```bash
python3 -m pytest tests/data/ tests/app/ -v
```

- [ ] **Step 5: Commit per logical group**

```bash
git commit -m "refactor(v55-phase4): app/services/<X> uses RunsRepository"
```

---

### Task 4.11: Migrate harness writers

**Files:**
- Modify: `harness/baseline.py`, `harness/runner.py`, `harness/match.py`, `harness/schema.py` (and any other `harness/*.py` file that opens a SQLite connection or executes SQL)

The harness owns regression baseline writes. It must route through the same `Writer` surface as the pipeline and app, otherwise it competes for the SQLite write lock.

- [ ] **Step 1: Inventory harness writes**

```bash
grep -rn "sqlite3.connect\|conn.execute.*INSERT\|conn.execute.*UPDATE\|conn.execute.*DELETE\|conn.execute.*CREATE\|conn.execute.*ALTER" harness/
```

For each match, identify the table and replace with the corresponding repository call. `harness/schema.py` is allowed to keep raw SQL for migration-style operations (it falls under the schema-helper allowlist).

- [ ] **Step 2: Wire a `Writer` into the harness CLI entry point**

Edit `harness/cli.py`. At the top of each subcommand that writes:

```python
from card_capture.data.writer import Writer

writer = Writer(db_path=args.db)
writer.start()
try:
    # ... existing logic, using repositories ...
finally:
    writer.stop()
```

- [ ] **Step 3: Replace each direct-write call site with a repository method call**

Mechanical edits. For each match from Step 1: identify table, swap `conn.execute("INSERT INTO X ...")` for `<XRepo>(writer=writer, db_path=db).method(...)`. If no existing repository method covers the call, add one to the appropriate repository file (Tasks 4.3–4.8) before swapping the call site.

- [ ] **Step 4: Run harness tests + raw-SQL scanner against `harness/`**

```bash
python3 -m pytest tests/harness/ -v
python3 -m pytest tests/architecture/test_raw_sql_outside_data.py -v
```

Expected: PASS. The raw-SQL scanner advisory output (`-s` flag) should no longer flag any file outside `harness/schema.py`.

- [ ] **Step 5: Commit**

```bash
git add harness/ src/card_capture/data/repositories/
git commit -m "refactor(v55-phase4): harness writes via repositories"
```

---

### Task 4.12: Tighten raw-SQL scanner to blocking

**Files:**
- Modify: `tests/architecture/test_raw_sql_outside_data.py`

- [ ] **Step 1: Remove the env gate**

Edit the test file: rename `test_no_raw_sql_outside_data_blocking` → `test_no_raw_sql_outside_data` and delete `@pytest.mark.skipif`. Delete `test_raw_sql_advisory`.

- [ ] **Step 2: Add the `card_capture.data` Import Linter contract for `sqlite3`**

Verify `.importlinter` already forbids `sqlite3` outside `card_capture.data`. If not, add (Task 1.6 should have included this).

- [ ] **Step 3: Run; PASS**

```bash
python3 -m pytest tests/architecture/ -v
```

- [ ] **Step 4: Commit and tag**

```bash
git add tests/architecture/
git commit -m "feat(v55-phase4): raw-SQL scanner blocking"
git tag v55-phase4-complete
```

**Phase 4 complete.** All SQLite writes route through one Writer per process. Repositories are the only place direct SQL lives.

---

# Phase 5: Platform Adapter Cleanup

**Goal:** Unify Beam, RunPod, Vast.ai under one `PipelineRunner` contract. App imports results from `RunManifest` only — no provider-specific filesystem assumptions.

**Acceptance:** All adapters implement the same `submit/wait/cancel` surface. Failure mapping is uniform. Import Linter forbids provider SDKs outside `card_capture.platforms`.

---

### Task 5.1: Create `card_capture.platforms` package + `LocalRunner`

**Files:**
- Create: `src/card_capture/platforms/__init__.py`
- Create: `src/card_capture/platforms/local.py`
- Create: `src/card_capture/platforms/manifests.py`
- Create: `tests/platforms/__init__.py`
- Create: `tests/platforms/test_local_runner.py`

- [ ] **Step 1: Empty `__init__.py`**

- [ ] **Step 2: Write failing test**

`tests/platforms/test_local_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.platforms.local import LocalRunner


def test_local_runner_executes_synchronously(tmp_path):
    runner = LocalRunner()
    req = PipelineRunRequest(
        run_id="l",
        input_video=f"artifact://local/{Path('tests/fixtures/tiny_synthetic.MOV').absolute()}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    handle = runner.submit(req)
    assert handle.backend == "local"
    result = runner.wait(handle)
    assert result.manifest.run_id == "l"
```

- [ ] **Step 3: Implement `local.py`**

```python
"""LocalRunner: runs the pipeline in-process via LocalPipelineRuntime."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from card_capture.pipeline.request import PipelineRunRequest, PipelineRunResult
from card_capture.pipeline.runner import PipelineRunHandle
from card_capture.pipeline.runtime_local import LocalPipelineRuntime


@dataclass
class _LocalJob:
    handle: PipelineRunHandle
    result: PipelineRunResult


class LocalRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, _LocalJob] = {}

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        handle = PipelineRunHandle(
            run_id=request.run_id or uuid.uuid4().hex[:12],
            backend="local",
            opaque="",
        )
        result = LocalPipelineRuntime().run(request)
        self._jobs[handle.run_id] = _LocalJob(handle=handle, result=result)
        return handle

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        return self._jobs[handle.run_id].result

    def cancel(self, handle: PipelineRunHandle) -> None:
        self._jobs.pop(handle.run_id, None)
```

- [ ] **Step 4: Run; PASS**

```bash
python3 -m pytest tests/platforms/test_local_runner.py -v
```

- [ ] **Step 5: Implement `manifests.py` (artifact reference resolver)**

```python
"""Artifact reference resolution.

artifact://local/<path>          -> local filesystem
artifact://s3/<bucket>/<key>     -> S3 (resolved by app)
artifact://beam/<volume>/<path>  -> Beam volume (resolved by adapter)
artifact://runpod/<job>/<path>   -> RunPod object storage (resolved by adapter)
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def resolve_local(reference: str) -> Path:
    parsed = urlparse(reference)
    if parsed.scheme != "artifact" or parsed.netloc != "local":
        raise ValueError(f"not a local artifact reference: {reference!r}")
    return Path(parsed.path.lstrip("/"))


def is_local(reference: str) -> bool:
    return reference.startswith("artifact://local/")
```

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/platforms/ tests/platforms/
git commit -m "feat(v55-phase5): card_capture.platforms package + LocalRunner"
```

---

### Task 5.2: RunPod adapter

**Files:**
- Move: `app/services/runpod_runner.py` → `src/card_capture/platforms/runpod.py`
- Modify: imports in `app/api/`, `app/services/__init__.py` to use the new path
- Create: `tests/platforms/test_runpod_runner.py`

The existing `runpod_runner.py` already encapsulates RunPod-specific upload/invoke/poll logic. The refactor: rehome to `card_capture.platforms` and adapt the public surface to `PipelineRunner` (submit / wait / cancel + manifest return).

- [ ] **Step 1: Move the file and rename the class**

```bash
git mv app/services/runpod_runner.py src/card_capture/platforms/runpod.py
```

Rename the class inside to `RunPodRunner`. Update its imports.

- [ ] **Step 2: Conform to `PipelineRunner`**

The class must implement:

```python
class RunPodRunner:
    def __init__(self, *, transport=None, credentials=None, ...): ...
    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle: ...
    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult: ...
    def cancel(self, handle: PipelineRunHandle) -> None: ...
```

Inside:
- `submit()` uploads `request.input_video` to RunPod object storage, kicks off the serverless job with `request.runtime_mode` passed as a job parameter, returns a `PipelineRunHandle(run_id=request.run_id, backend="runpod", opaque=<job_id>)`.
- `wait()` polls until the job terminates, downloads `run_manifest.json` from the result location, parses with `RunManifest.from_json`, returns `PipelineRunResult(manifest=..., manifest_path=<local cached path>)`.
- `cancel()` calls RunPod's cancel API for the opaque job id.
- All raised exceptions map to one of the categories in `card_capture.platforms.failures` (Task 5.6).

- [ ] **Step 3: Update imports in app code**

```bash
grep -rn "from app.services.runpod_runner\|import runpod_runner" app/
```

Replace each with `from card_capture.platforms.runpod import RunPodRunner`.

- [ ] **Step 4: Write fake-transport test**

`tests/platforms/test_runpod_runner.py`:

```python
"""RunPod adapter with a fake transport — exercises the PipelineRunner shape without credentials."""
from __future__ import annotations

from card_capture.pipeline.request import (
    PipelineRunRequest, RunManifest, PipelineRunResult,
)
from card_capture.platforms.runpod import RunPodRunner


class FakeRunPodTransport:
    """Records calls; returns canned responses."""
    def __init__(self, manifest: RunManifest) -> None:
        self._manifest = manifest
        self.calls: list[tuple] = []

    def upload(self, src, dst): self.calls.append(("upload", src, dst))
    def start_job(self, payload): self.calls.append(("start", payload)); return "job-1"
    def poll(self, job_id): self.calls.append(("poll", job_id)); return "succeeded"
    def download_manifest(self, job_id): return self._manifest.to_json()
    def cancel(self, job_id): self.calls.append(("cancel", job_id))


def test_runpod_runner_returns_manifest(tmp_path):
    manifest = RunManifest(
        run_id="r", runtime_mode="strict_gpu", input_video="artifact://runpod/job-1/in.MOV",
        output_artifacts=[], cards=[], stage_timings=[], contract_violations=[],
        version="0.5.5+phase5",
    )
    runner = RunPodRunner(transport=FakeRunPodTransport(manifest))
    handle = runner.submit(PipelineRunRequest(
        run_id="r", input_video="artifact://local/in.MOV",
        output_root="artifact://runpod/job-1/", runtime_mode="strict_gpu",
    ))
    assert handle.backend == "runpod"
    result = runner.wait(handle)
    assert isinstance(result, PipelineRunResult)
    assert result.manifest.run_id == "r"


def test_runpod_runner_cancel_calls_transport():
    manifest = RunManifest(
        run_id="x", runtime_mode="strict_gpu", input_video="",
        output_artifacts=[], cards=[], stage_timings=[], contract_violations=[],
        version="0.5.5+phase5",
    )
    transport = FakeRunPodTransport(manifest)
    runner = RunPodRunner(transport=transport)
    handle = runner.submit(PipelineRunRequest(
        run_id="x", input_video="artifact://local/in.MOV",
        output_root="artifact://runpod/job-1/", runtime_mode="strict_gpu",
    ))
    runner.cancel(handle)
    assert any(c[0] == "cancel" for c in transport.calls)
```

- [ ] **Step 5: Run + PASS**

```bash
python3 -m pytest tests/platforms/test_runpod_runner.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/platforms/runpod.py app/ tests/platforms/test_runpod_runner.py
git commit -m "refactor(v55-phase5): RunPod adapter implements PipelineRunner"
```

---

### Task 5.3: Beam adapter

**Files:**
- Move: `app/beam_handler.py` + `app/services/beam_runner.py` → `src/card_capture/platforms/beam.py`
- Modify: app imports
- Create: `tests/platforms/test_beam_runner.py`

The Beam adapter is partial in V4 — Phase 5 finishes it against the `PipelineRunner` contract. The acceptance criterion (spec): "Beam can be completed by implementing a platform adapter, not by editing core pipeline logic."

- [ ] **Step 1: Move and consolidate Beam code**

```bash
git mv app/services/beam_runner.py src/card_capture/platforms/beam.py
# beam_handler.py is the in-Beam worker entrypoint; keep it under app/ for now.
```

Inside `card_capture/platforms/beam.py`, consolidate any client-side Beam logic (volume reference resolution, endpoint invocation, polling, download). The Beam-side worker (`app/beam_handler.py`) stays where it is for now; it deserializes a `PipelineRunRequest`, calls `LocalPipelineRuntime`, writes `RunManifest` to the Beam volume.

- [ ] **Step 2: Conform to `PipelineRunner`** (same shape as Task 5.2 RunPodRunner — class name `BeamRunner`).

- [ ] **Step 3: Update app imports**

- [ ] **Step 4: Write fake-transport test** matching Task 5.2's structure, with `FakeBeamTransport`.

```python
class FakeBeamTransport:
    def __init__(self, manifest): self._m = manifest; self.calls = []
    def push_volume(self, src, dst): self.calls.append(("push", src, dst))
    def invoke_endpoint(self, payload): self.calls.append(("invoke", payload)); return "beam-job-1"
    def poll(self, job_id): return "succeeded"
    def pull_manifest(self, job_id): return self._m.to_json()
    def cancel(self, job_id): self.calls.append(("cancel", job_id))


def test_beam_runner_returns_manifest():
    # ... same shape as RunPod test, with BeamRunner ...
```

- [ ] **Step 5: Run + PASS**
- [ ] **Step 6: Commit**

```bash
git add src/card_capture/platforms/beam.py app/ tests/platforms/test_beam_runner.py
git commit -m "feat(v55-phase5): Beam adapter implements PipelineRunner"
```

---

### Task 5.4: Vast.ai adapter (or deprecate)

**Files:**
- Decision point: keep or deprecate per spec Section 8.

The spec ("Vast.ai should either be brought behind the same adapter contract or deprecated") leaves this open. Confirm intent before doing work:

- [ ] **Step 1: Confirm Vast.ai status with project owner**

If Vast.ai is being kept:

- [ ] **Step 2a: Move `app/services/vast_runner.py`, `app/services/vast_client.py`, `app/vastai_worker.py` → `src/card_capture/platforms/vastai.py`** (consolidated) and the Vast worker entrypoint stays under app/ for now.
- [ ] **Step 3a: Conform to `PipelineRunner`** (class `VastaiRunner`).
- [ ] **Step 4a: Fake-transport test** (same structure as RunPod / Beam, `FakeVastaiTransport`).
- [ ] **Step 5a: Run + PASS**
- [ ] **Step 6a: Commit** `git commit -m "feat(v55-phase5): Vast.ai adapter implements PipelineRunner"`

If Vast.ai is being deprecated:

- [ ] **Step 2b: Delete provider files**

```bash
git rm app/services/vast_runner.py app/services/vast_client.py app/vastai_worker.py
```

- [ ] **Step 3b: Remove any `vastai` references from `app/api/`, `app/services/__init__.py`, and `pyproject.toml`**

```bash
grep -rn "vast\|Vast" app/ src/ pyproject.toml | grep -v "/web/"
```

Remove the `vastai>=0.5.0` dependency from `pyproject.toml`.

- [ ] **Step 4b: Update standards.md** — note R-PLAT-1 only forbids `runpod` and `beam` SDKs outside `card_capture.platforms`; Vast.ai is no longer a supported runtime.

- [ ] **Step 5b: Run architecture tests; PASS**

```bash
python3 -m pytest tests/architecture/ -v
```

- [ ] **Step 6b: Commit** `git commit -m "refactor(v55-phase5): deprecate Vast.ai runtime"`

---

### Task 5.5: Manifest-based result import

**Files:**
- Modify: `app/services/result_importer.py` (read from manifest, not provider-specific paths)

- [ ] **Step 1: Identify each provider-path assumption in `result_importer.py`**

```bash
grep -n "runpod\|beam\|vast" app/services/result_importer.py
```

- [ ] **Step 2: Replace with manifest-driven import**

The new import flow:

```python
def import_run(manifest: RunManifest, repos: Repositories) -> None:
    for card in manifest.cards:
        repos.cards.store_final_cards(manifest.run_id, [card])
    for timing in manifest.stage_timings:
        repos.events.record_stage_finished(manifest.run_id, ..., timing.stage, timing.elapsed_ms, timing.metadata)
    repos.runs.mark_completed(manifest.run_id, cards_extracted=len(manifest.cards))
```

- [ ] **Step 3: Test against a synthetic manifest**

`tests/app/test_result_importer_manifest.py`:

```python
def test_import_uses_manifest_only(tmp_path):
    manifest = RunManifest(...minimal...)
    import_run(manifest, repos=FakeRepos())
    # Assert: no provider-SDK calls, no filesystem walks under provider-specific roots.
```

- [ ] **Step 4: Run; PASS**
- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(v55-phase5): result importer reads from manifest only"
```

---

### Task 5.6: Failure category mapping helper

**Files:**
- Create: `src/card_capture/platforms/failures.py`

- [ ] **Step 1: Define categories**

```python
"""Provider failure categories per spec Section 8."""
from __future__ import annotations

PREFLIGHT_FAILED = "preflight_failed"
SUBMISSION_FAILED = "submission_failed"
INPUT_TRANSFER_FAILED = "input_transfer_failed"
RUNTIME_CONTRACT_FAILED = "runtime_contract_failed"
RUNTIME_EXECUTION_FAILED = "runtime_execution_failed"
OUTPUT_TRANSFER_FAILED = "output_transfer_failed"
RESULT_IMPORT_FAILED = "result_import_failed"
CANCELLED = "cancelled"
TIMEOUT = "timeout"

ALL = (
    PREFLIGHT_FAILED, SUBMISSION_FAILED, INPUT_TRANSFER_FAILED,
    RUNTIME_CONTRACT_FAILED, RUNTIME_EXECUTION_FAILED, OUTPUT_TRANSFER_FAILED,
    RESULT_IMPORT_FAILED, CANCELLED, TIMEOUT,
)
```

- [ ] **Step 2: Test that each adapter maps to one of these**

`tests/platforms/test_failure_categories.py`:

```python
import pytest

from card_capture.platforms import failures
from card_capture.platforms.runpod import RunPodRunner
# ... beam, vastai


@pytest.mark.parametrize("Runner", [RunPodRunner])  # add others
def test_provider_errors_map_to_categories(Runner):
    # Force a known failure via fake transport.
    runner = Runner(transport=FakeTransport(force_error="auth_fail"))
    handle = runner.submit(...)
    status = runner.poll(handle) if hasattr(runner, "poll") else None
    # The status detail must be one of the documented categories.
    # Adapt assertion to the adapter shape.
```

- [ ] **Step 3: Run + PASS**

- [ ] **Step 4: Commit and tag**

```bash
git commit -m "feat(v55-phase5): provider failure category mapping"
git tag v55-phase5-complete
```

**Phase 5 complete.** All platforms share `PipelineRunner`. Manifest is the only inter-system contract.

---

# Wrap-up

After Phase 5 lands:

- [ ] **Update `docs/architecture/standards.md`**: change every `advisory:phase-N` marker that has reached its phase to `blocking:phase-N`. Remove any stale "Phase X will" prose.
- [ ] **Run the full acceptance criteria checklist** from the spec (`docs/superpowers/specs/2026-05-24-v5-5-refactoring-design.md`, "Acceptance Criteria" section) and confirm each item.
- [ ] **Delete `pipeline/__init__.py` and `pipeline/contracts.py` transitional shims** once no imports remain.
- [ ] **Tag** `v55-complete`.

```bash
git tag v55-complete
```

---

# Notes for executors

- **Always run the architecture lane before committing**: `python3 -m pytest tests/architecture/ -q`.
- **Existing code paths referenced** (`run_sampler`, `load_yolo_model`, `detect_corners_batch`, `gate_detections`, `refine_tracks_to_crops`, `score_crops`, `resolve_front_back`, `fuse_track`, `dedupe_fused`, `store_cards`) — these are conceptual; the actual function names in the codebase may differ. Verify with `grep -rn "def <name>" src/` before adapting. If a function does not exist with the named shape, the plan task that wraps it is the place to adapt — do not invent new algorithm code, only re-shape the call sites.
- **When in doubt**, the spec (`docs/superpowers/specs/2026-05-24-v5-5-refactoring-design.md`) overrides this plan. If they disagree, surface the conflict before proceeding.
- **Phase boundaries are checkpoints.** Stop and confirm acceptance criteria before moving to the next phase. Do not start Phase 3 with Phase 2 advisory checks still un-tightened, etc.
