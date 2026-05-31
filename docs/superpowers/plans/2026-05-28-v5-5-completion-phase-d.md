# V5.5 Completion — Phase D: Platform Adapters and Vast.ai Deprecation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 5 with a uniform `PipelineRunner` surface across `LocalRunner`, `RunpodRunner`, `BeamRunner`. Delete Vast.ai end-to-end. Add `manifests.py` and `failures.py`. Populate `tests/platforms/`.

**Architecture:** Nine tasks. The work is mostly net-new (only the old `RunpodRuntime` against the dead `RemoteRuntime` Protocol exists today).

**Tech Stack:** RunPod SDK, Beam SDK, pytest, Python Protocols.

**Parent plan:** `docs/superpowers/plans/2026-05-28-v5-5-completion.md` (Phase D section). When this plan and the parent disagree, the parent wins.

**Acceptance:**
1. `LocalRunner`, `RunpodRunner`, `BeamRunner` all implement `card_capture.pipeline.runner.PipelineRunner` (`submit`/`wait`/`cancel`).
2. `from card_capture.platforms.failures import map_provider_failure` exists and returns one of the stable categories.
3. `from card_capture.platforms.manifests import import_manifest, export_manifest` round-trips a `RunManifest` to/from a JSON file.
4. `app/services/vast_runner.py`, `app/services/vast_client.py`, `tests/app/test_vast*.py` are deleted; `vastai` is removed from `pyproject.toml` and from `.importlinter`'s forbidden list.
5. `tests/platforms/` contains at least: `__init__.py`, `test_local_runner.py`, `test_runpod_runner.py`, `test_beam_runner.py`, `test_failures.py`, `test_manifests.py`.
6. `PYTHONPATH=src:. lint-imports` exits 0.

**Pre-flight context (verified 2026-05-28):**
- `src/card_capture/platforms/` contains only `__init__.py` (empty) and `runpod.py` (still implementing the dead `RemoteRuntime` Protocol).
- `src/card_capture/pipeline/remote.py` still exists (499 B).
- `tests/platforms/` does not exist.
- `app/services/vast_runner.py` (13459 B) and `vast_client.py` (5287 B) still exist.
- `tests/app/test_vast_client.py`, `test_vast_runner.py`, `test_vastai_worker.py` still exist.
- `pyproject.toml` line 15 still has `vastai>=0.5.0`.
- `.importlinter` `[importlinter:contract:no-provider-sdk-outside-platforms]` still forbids `vastai`.

**Phase D is independent of Phase C's caller migration** in the sense that it doesn't touch raw-sqlite3 callsites — but Phase D should NOT be started until Phase C lands, because the test suite must be green so regressions introduced by Phase D are visible.

---

## File Structure

```text
src/card_capture/pipeline/remote.py                        DELETED
src/card_capture/pipeline/runner.py                        Modified (docstring tightening)
src/card_capture/platforms/__init__.py                     Replaced (currently empty)
src/card_capture/platforms/failures.py                     Created
src/card_capture/platforms/manifests.py                    Created
src/card_capture/platforms/local.py                        Created
src/card_capture/platforms/runpod.py                       Rewritten
src/card_capture/platforms/beam.py                         Created
app/services/vast_runner.py                                DELETED
app/services/vast_client.py                                DELETED
tests/app/test_vast_client.py                              DELETED
tests/app/test_vast_runner.py                              DELETED
tests/app/test_vastai_worker.py                            DELETED
pyproject.toml                                             Modified (drop vastai)
.importlinter                                              Modified (drop vastai)
tests/platforms/__init__.py                                Created
tests/platforms/test_failures.py                           Created
tests/platforms/test_manifests.py                          Created
tests/platforms/test_local_runner.py                       Created
tests/platforms/test_runpod_runner.py                      Created
tests/platforms/test_beam_runner.py                        Created
```

---

### Task D.1: Collapse `pipeline/remote.py` into `pipeline/runner.py`

**Files:**
- Delete: `src/card_capture/pipeline/remote.py`
- Modify: `src/card_capture/pipeline/runner.py` (docstring only)
- Modify: any importer of `RemoteRuntime`

- [ ] **Step 1: Audit `RemoteRuntime` and `pipeline.remote` imports**

```bash
grep -rn 'RemoteRuntime\|from card_capture.pipeline.remote\|from .remote import\|pipeline\.remote' --include='*.py' src/ app/ tests/
```

Note every hit. Expected: `src/card_capture/platforms/runpod.py` (extends `RemoteRuntime`), possibly `pipeline/contracts.py` shim, no tests.

- [ ] **Step 2: Delete the file**

```bash
git rm src/card_capture/pipeline/remote.py
```

- [ ] **Step 3: Tighten `pipeline/runner.py` docstring**

Read `src/card_capture/pipeline/runner.py`. If the top docstring is short, replace it with:

```python
"""PipelineRunner: uniform submit/wait/cancel surface for local and remote backends.

Concrete implementations live in `card_capture.platforms.*`. The handle is
opaque to callers; each backend stores its provider-specific job_id in
`PipelineRunHandle.opaque`. This module replaces the earlier `RemoteRuntime`
Protocol (collapsed in Phase D); legacy callers should migrate to
`PipelineRunner`.
"""
```

Leave the rest of the file (the `PipelineRunHandle`, `PipelineRunStatus`, `PipelineRunner` Protocol) unchanged.

- [ ] **Step 4: Smoke-import**

```bash
python3 -c "from card_capture.pipeline.runner import PipelineRunner, PipelineRunHandle, PipelineRunStatus; print('ok')"
```

Expected: `ok`. Imports of `card_capture.pipeline.remote` should NOT exist after Task D.5 (which rewrites runpod.py); for now, anything that imports `RemoteRuntime` will break. That's OK — D.5 fixes it.

- [ ] **Step 5: Commit**

```bash
git add -A src/card_capture/pipeline/
git commit -m "refactor(v55-phaseD): drop RemoteRuntime; PipelineRunner is the only protocol"
```

---

### Task D.2: Create `platforms/failures.py` + `tests/platforms/test_failures.py`

**Files:**
- Create: `src/card_capture/platforms/failures.py`
- Create: `tests/platforms/__init__.py`
- Create: `tests/platforms/test_failures.py`

- [ ] **Step 1: Write the failing test**

`tests/platforms/__init__.py` (empty file).

`tests/platforms/test_failures.py`:

```python
"""failures.py exposes stable categories and a mapping helper."""
from __future__ import annotations

from card_capture.platforms.failures import (
    PROVIDER_FAILURE_CATEGORIES,
    ProviderFailure,
    map_provider_failure,
)


def test_categories_are_stable_strings():
    for c in ("preflight_failed", "submission_failed", "execution_failed",
              "result_invalid", "cancelled", "unknown"):
        assert c in PROVIDER_FAILURE_CATEGORIES


def test_map_unknown_provider_returns_unknown_category():
    failure = map_provider_failure(provider="runpod", raw="<garbled blob>")
    assert failure.category == "unknown"
    assert failure.provider == "runpod"
    assert failure.raw == "<garbled blob>"


def test_map_well_known_runpod_phrases():
    assert map_provider_failure(provider="runpod", raw="endpoint not found").category == "preflight_failed"
    assert map_provider_failure(provider="runpod", raw="JOB FAILED: out of memory").category == "execution_failed"
    assert map_provider_failure(provider="runpod", raw="cancelled by user").category == "cancelled"


def test_map_well_known_beam_phrases():
    assert map_provider_failure(provider="beam", raw="deployment not found").category == "preflight_failed"
    assert map_provider_failure(provider="beam", raw="task failed during exec").category == "execution_failed"


def test_provider_failure_validates_category():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        ProviderFailure(provider="runpod", category="bogus", raw="x")
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_failures.py -v
```

Expected: FAIL with `ModuleNotFoundError: card_capture.platforms.failures`.

- [ ] **Step 3: Implement**

`src/card_capture/platforms/failures.py`:

```python
"""Stable failure categories for provider runs.

All adapters surface failures through `map_provider_failure(provider, raw)`,
returning a `ProviderFailure` with a category drawn from
`PROVIDER_FAILURE_CATEGORIES`. App-facing status code paths consume the
category, not the raw provider string.
"""
from __future__ import annotations

import dataclasses
import re
from typing import FrozenSet


PROVIDER_FAILURE_CATEGORIES: FrozenSet[str] = frozenset({
    "preflight_failed",     # provider-side rejection before job started
    "submission_failed",    # network/auth error during submit
    "execution_failed",     # provider says the job ran and failed
    "result_invalid",       # provider returned, but manifest could not be parsed
    "cancelled",            # caller-initiated cancellation
    "unknown",              # everything else
})


@dataclasses.dataclass(frozen=True)
class ProviderFailure:
    provider: str
    category: str
    raw: str

    def __post_init__(self) -> None:
        if self.category not in PROVIDER_FAILURE_CATEGORIES:
            raise ValueError(
                f"unknown category {self.category!r}; "
                f"must be one of {sorted(PROVIDER_FAILURE_CATEGORIES)}"
            )


_RUNPOD_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"endpoint\s+not\s+found", re.I), "preflight_failed"),
    (re.compile(r"unauthorized|invalid\s+api\s+key", re.I), "preflight_failed"),
    (re.compile(r"timeout\s+during\s+submit", re.I), "submission_failed"),
    (re.compile(r"job\s+failed", re.I), "execution_failed"),
    (re.compile(r"out\s+of\s+memory|oom", re.I), "execution_failed"),
    (re.compile(r"cancell?ed", re.I), "cancelled"),
)

_BEAM_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"deployment\s+not\s+found", re.I), "preflight_failed"),
    (re.compile(r"app\s+id\s+missing|missing\s+credentials", re.I), "preflight_failed"),
    (re.compile(r"timeout", re.I), "submission_failed"),
    (re.compile(r"task\s+failed|error\s+in\s+task", re.I), "execution_failed"),
    (re.compile(r"cancell?ed", re.I), "cancelled"),
)


def map_provider_failure(*, provider: str, raw: str) -> ProviderFailure:
    """Best-effort categorization of a provider failure string."""
    patterns = {
        "runpod": _RUNPOD_PATTERNS,
        "beam": _BEAM_PATTERNS,
        "local": (),
    }.get(provider, ())
    for pattern, category in patterns:
        if pattern.search(raw):
            return ProviderFailure(provider=provider, category=category, raw=raw)
    return ProviderFailure(provider=provider, category="unknown", raw=raw)
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python3 -m pytest tests/platforms/test_failures.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/failures.py tests/platforms/__init__.py tests/platforms/test_failures.py
git commit -m "feat(v55-phaseD): platforms.failures with stable categories + provider phrase mapping"
```

---

### Task D.3: Create `platforms/manifests.py` + `tests/platforms/test_manifests.py`

**Files:**
- Create: `src/card_capture/platforms/manifests.py`
- Create: `tests/platforms/test_manifests.py`

- [ ] **Step 1: Failing test**

`tests/platforms/test_manifests.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import RunManifest
from card_capture.platforms.manifests import export_manifest, import_manifest


def _sample_manifest() -> RunManifest:
    return RunManifest(
        run_id="r1",
        runtime_mode="cpu_debug",
        input_video="artifact://local/x.MOV",
        output_artifacts=["artifact://local/r1/cards/"],
        cards=[],
        stage_timings=[],
        contract_violations=[],
        version="0.5.5+phaseD",
    )


def test_export_then_import_roundtrip(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    path = export_manifest(manifest, tmp_path / "manifest.json")
    assert path.exists()
    loaded = import_manifest(path)
    assert loaded == manifest


def test_import_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_manifest(tmp_path / "nope.json")


def test_export_creates_parent_directories(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    path = export_manifest(manifest, tmp_path / "nested" / "subdir" / "m.json")
    assert path.exists()
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_manifests.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement**

`src/card_capture/platforms/manifests.py`:

```python
"""Manifest import/export helpers shared by platform adapters."""
from __future__ import annotations

from pathlib import Path

from card_capture.pipeline.request import RunManifest


def export_manifest(manifest: RunManifest, path: Path | str) -> Path:
    """Write a manifest to disk as JSON. Returns the resolved path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.to_json())
    return p


def import_manifest(path: Path | str) -> RunManifest:
    """Load a manifest from disk; raises FileNotFoundError if the path is empty."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    return RunManifest.from_json(p.read_text())
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python3 -m pytest tests/platforms/test_manifests.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/manifests.py tests/platforms/test_manifests.py
git commit -m "feat(v55-phaseD): platforms.manifests import/export helpers"
```

---

### Task D.4: Implement `LocalRunner`

**Files:**
- Create: `src/card_capture/platforms/local.py`
- Create: `tests/platforms/test_local_runner.py`

- [ ] **Step 1: Failing test**

`tests/platforms/test_local_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runner import PipelineRunHandle
from card_capture.platforms.local import LocalRunner


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


@pytest.mark.skipif(not FIXTURE.exists(), reason="tiny_synthetic.MOV fixture not present")
def test_submit_then_wait_returns_manifest(tmp_path: Path) -> None:
    runner = LocalRunner()
    req = PipelineRunRequest(
        run_id="local-1",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    handle = runner.submit(req)
    assert handle.backend == "local"
    assert handle.run_id == "local-1"
    result = runner.wait(handle)
    assert result.manifest.run_id == "local-1"


def test_cancel_is_idempotent_noop() -> None:
    runner = LocalRunner()
    runner.cancel(PipelineRunHandle(run_id="x", backend="local"))
    runner.cancel(PipelineRunHandle(run_id="x", backend="local"))


def test_wait_on_unknown_handle_raises() -> None:
    runner = LocalRunner()
    import pytest as _pytest
    with _pytest.raises(KeyError):
        runner.wait(PipelineRunHandle(run_id="nope", backend="local"))
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_local_runner.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement**

`src/card_capture/platforms/local.py`:

```python
"""Local synchronous runner.

`submit` runs the pipeline inline and stashes the result keyed by run_id;
`wait` returns the stashed result. This implementation is intentionally
synchronous — the PipelineRunner protocol shape is preserved for parity
with remote backends, not for actual parallelism.
"""
from __future__ import annotations

import threading
from typing import Dict

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
)
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.pipeline.runtime_local import LocalPipelineRuntime


class LocalRunner(PipelineRunner):
    def __init__(self, runtime: LocalPipelineRuntime | None = None) -> None:
        self._runtime = runtime or LocalPipelineRuntime()
        self._results: Dict[str, PipelineRunResult] = {}
        self._lock = threading.Lock()

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        result = self._runtime.run(request)
        with self._lock:
            self._results[request.run_id] = result
        return PipelineRunHandle(run_id=request.run_id, backend="local")

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        with self._lock:
            result = self._results.get(handle.run_id)
        if result is None:
            raise KeyError(f"unknown run_id {handle.run_id!r}")
        return result

    def cancel(self, handle: PipelineRunHandle) -> None:
        # Synchronous local runs cannot be cancelled mid-flight; this is a
        # no-op so callers can treat all backends uniformly.
        return
```

- [ ] **Step 4: Run, PASS**

```bash
python3 -m pytest tests/platforms/test_local_runner.py -v
```

Expected: 2 PASS + 1 SKIP (or 3 PASS if the fixture is present).

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/local.py tests/platforms/test_local_runner.py
git commit -m "feat(v55-phaseD): LocalRunner implements PipelineRunner"
```

---

### Task D.5: Rewrite `RunpodRunner` to `PipelineRunner`

**Files:**
- Modify: `src/card_capture/platforms/runpod.py` (rewrite)
- Create: `tests/platforms/test_runpod_runner.py`

The current `RunpodRuntime` extends `RemoteRuntime` (deleted in D.1). Replace with `RunpodRunner` implementing `PipelineRunner`.

- [ ] **Step 1: Write the failing test (using a stub runpod client)**

`tests/platforms/test_runpod_runner.py`:

```python
from __future__ import annotations

from card_capture.pipeline.request import PipelineRunRequest, RunManifest
from card_capture.platforms.runpod import RunpodRunner, RunpodRunnerError


class _StubEndpoint:
    def __init__(self) -> None:
        self._jobs: dict[str, "_StubJob"] = {}

    def run(self, payload: dict) -> "_StubJob":
        job = _StubJob(run_id=payload["run_id"])
        self._jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> "_StubJob":
        return self._jobs[job_id]


class _StubJob:
    def __init__(self, run_id: str, force_status: str | None = None,
                 error_msg: str = "") -> None:
        self.id = f"rp-{run_id}"
        self._run_id = run_id
        self._status = force_status or "IN_PROGRESS"
        self._error = error_msg

    def status(self) -> str:
        if self._status == "IN_PROGRESS":
            self._status = "COMPLETED"
        return self._status

    def output(self) -> dict:
        manifest = RunManifest(
            run_id=self._run_id, runtime_mode="strict_gpu",
            input_video="artifact://s3/x.MOV", output_artifacts=[],
            cards=[], stage_timings=[], contract_violations=[],
            version="0.5.5+test",
        )
        return {"manifest_json": manifest.to_json()}

    def cancel(self) -> None:
        self._status = "CANCELLED"

    def error(self) -> str:
        return self._error


def _make_runner(endpoint) -> RunpodRunner:
    """Bypass __init__ to avoid importing the real runpod SDK."""
    r = RunpodRunner.__new__(RunpodRunner)
    r._endpoint = endpoint
    r._poll_interval = 0.0
    return r


def test_submit_wait_roundtrip() -> None:
    endpoint = _StubEndpoint()
    runner = _make_runner(endpoint)
    req = PipelineRunRequest(
        run_id="rp1", input_video="artifact://s3/x.MOV",
        output_root="artifact://s3/rp1/", runtime_mode="strict_gpu",
    )
    handle = runner.submit(req)
    assert handle.backend == "runpod"
    assert handle.opaque == "rp-rp1"
    result = runner.wait(handle)
    assert result.manifest.run_id == "rp1"


def test_failed_job_raises_categorized_error() -> None:
    endpoint = _StubEndpoint()
    # Pre-stage a job in FAILED state.
    failed = _StubJob(run_id="rp2", force_status="FAILED",
                      error_msg="JOB FAILED: out of memory")
    endpoint._jobs[failed.id] = failed
    runner = _make_runner(endpoint)
    import pytest as _pytest
    with _pytest.raises(RunpodRunnerError) as exc_info:
        runner.wait(type("H", (), {"opaque": failed.id, "run_id": "rp2", "backend": "runpod"})())
    assert exc_info.value.failure.category == "execution_failed"
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_runpod_runner.py -v
```

Expected: FAIL on attribute / signature mismatch.

- [ ] **Step 3: Rewrite the runner**

Replace `src/card_capture/platforms/runpod.py` in full:

```python
"""RunPod serverless backend implementing PipelineRunner."""
from __future__ import annotations

import time

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
)
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.platforms.failures import (
    ProviderFailure,
    map_provider_failure,
)


class RunpodRunnerError(RuntimeError):
    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(f"{failure.category}: {failure.raw}")
        self.failure = failure


class RunpodRunner(PipelineRunner):
    def __init__(self, *, api_key: str, endpoint_id: str,
                 poll_interval: float = 1.0) -> None:
        import runpod  # local import keeps the SDK out of fast-path callers
        runpod.api_key = api_key
        self._endpoint = runpod.Endpoint(endpoint_id)
        self._poll_interval = poll_interval

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        try:
            job = self._endpoint.run(request.to_dict())
        except Exception as exc:  # noqa: BLE001
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod", raw=repr(exc))
            ) from exc
        return PipelineRunHandle(
            run_id=request.run_id, backend="runpod", opaque=job.id,
        )

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        job = self._endpoint.get_job(handle.opaque)
        while job.status() not in ("COMPLETED", "FAILED", "CANCELLED"):
            time.sleep(self._poll_interval)
        status = job.status()
        if status == "CANCELLED":
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod", raw="cancelled")
            )
        if status == "FAILED":
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod",
                                     raw=str(job.error() or "JOB FAILED"))
            )
        output = job.output()
        try:
            manifest = RunManifest.from_json(output["manifest_json"])
        except Exception as exc:  # noqa: BLE001
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod",
                                     raw=f"result_invalid: {exc!r}")
            ) from exc
        return PipelineRunResult(manifest=manifest)

    def cancel(self, handle: PipelineRunHandle) -> None:
        try:
            self._endpoint.get_job(handle.opaque).cancel()
        except Exception as exc:  # noqa: BLE001
            raise RunpodRunnerError(
                map_provider_failure(provider="runpod", raw=repr(exc))
            ) from exc
```

- [ ] **Step 4: Update any caller of the old `RunpodRuntime` name**

```bash
grep -rn 'RunpodRuntime' --include='*.py' src/ app/ tests/
```

Update each to `RunpodRunner`, including the constructor — new signature requires keyword args `api_key` and `endpoint_id`.

- [ ] **Step 5: Run, PASS**

```bash
python3 -m pytest tests/platforms/test_runpod_runner.py -v
```

Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/platforms/runpod.py tests/platforms/test_runpod_runner.py app/ src/
git commit -m "feat(v55-phaseD): RunpodRunner implements PipelineRunner, maps failures uniformly"
```

---

### Task D.6: Implement `BeamRunner`

**Files:**
- Create: `src/card_capture/platforms/beam.py`
- Create: `tests/platforms/test_beam_runner.py`

- [ ] **Step 1: Failing test with a stub Beam client**

`tests/platforms/test_beam_runner.py`:

```python
from __future__ import annotations

from card_capture.pipeline.request import PipelineRunRequest, RunManifest
from card_capture.platforms.beam import BeamRunner, BeamRunnerError


class _StubBeamApp:
    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}

    def run(self, payload: dict) -> dict:
        task_id = f"beam-{payload['run_id']}"
        self._tasks[task_id] = {"payload": payload, "status": "RUNNING"}
        return {"task_id": task_id}

    def status(self, task_id: str) -> str:
        s = self._tasks[task_id]["status"]
        if s == "RUNNING":
            self._tasks[task_id]["status"] = "SUCCEEDED"
            return "SUCCEEDED"
        return s

    def result(self, task_id: str) -> dict:
        payload = self._tasks[task_id]["payload"]
        manifest = RunManifest(
            run_id=payload["run_id"], runtime_mode="strict_gpu",
            input_video=payload["input_video"], output_artifacts=[],
            cards=[], stage_timings=[], contract_violations=[],
            version="0.5.5+test",
        )
        return {"manifest_json": manifest.to_json()}

    def cancel(self, task_id: str) -> None:
        self._tasks[task_id]["status"] = "CANCELLED"


def _make_runner(app) -> BeamRunner:
    r = BeamRunner.__new__(BeamRunner)
    r._app = app
    r._poll_interval = 0.0
    return r


def test_submit_wait_roundtrip() -> None:
    app = _StubBeamApp()
    runner = _make_runner(app)
    req = PipelineRunRequest(
        run_id="bm1", input_video="artifact://s3/x.MOV",
        output_root="artifact://s3/bm1/", runtime_mode="strict_gpu",
    )
    handle = runner.submit(req)
    assert handle.backend == "beam"
    result = runner.wait(handle)
    assert result.manifest.run_id == "bm1"
```

- [ ] **Step 2: Confirm fail**

```bash
python3 -m pytest tests/platforms/test_beam_runner.py -v
```

- [ ] **Step 3: Implement**

`src/card_capture/platforms/beam.py`:

```python
"""Beam serverless backend implementing PipelineRunner."""
from __future__ import annotations

import time

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
)
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.platforms.failures import (
    ProviderFailure,
    map_provider_failure,
)


class BeamRunnerError(RuntimeError):
    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(f"{failure.category}: {failure.raw}")
        self.failure = failure


class BeamRunner(PipelineRunner):
    def __init__(self, *, deployment_id: str, api_key: str,
                 poll_interval: float = 2.0) -> None:
        import beam  # local import; SDK kept out of fast-path callers
        self._app = beam.deployment(deployment_id, token=api_key)
        self._poll_interval = poll_interval

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        try:
            response = self._app.run(request.to_dict())
        except Exception as exc:  # noqa: BLE001
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw=repr(exc))
            ) from exc
        return PipelineRunHandle(
            run_id=request.run_id, backend="beam", opaque=response["task_id"],
        )

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        while True:
            status = self._app.status(handle.opaque)
            if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(self._poll_interval)
        if status == "CANCELLED":
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw="cancelled")
            )
        if status == "FAILED":
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw="task failed")
            )
        result = self._app.result(handle.opaque)
        try:
            manifest = RunManifest.from_json(result["manifest_json"])
        except Exception as exc:  # noqa: BLE001
            raise BeamRunnerError(
                map_provider_failure(provider="beam",
                                     raw=f"result_invalid: {exc!r}")
            ) from exc
        return PipelineRunResult(manifest=manifest)

    def cancel(self, handle: PipelineRunHandle) -> None:
        try:
            self._app.cancel(handle.opaque)
        except Exception as exc:  # noqa: BLE001
            raise BeamRunnerError(
                map_provider_failure(provider="beam", raw=repr(exc))
            ) from exc
```

- [ ] **Step 4: Run, PASS**

```bash
python3 -m pytest tests/platforms/test_beam_runner.py -v
```

Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/platforms/beam.py tests/platforms/test_beam_runner.py
git commit -m "feat(v55-phaseD): BeamRunner implements PipelineRunner"
```

---

### Task D.7: Deprecate Vast.ai end-to-end

**Files:**
- Delete: `app/services/vast_runner.py`
- Delete: `app/services/vast_client.py`
- Delete: `tests/app/test_vast_client.py`
- Delete: `tests/app/test_vast_runner.py`
- Delete: `tests/app/test_vastai_worker.py`
- Modify: `pyproject.toml`
- Modify: `.importlinter`
- Modify: any caller of `vast_runner` / `vast_client`

- [ ] **Step 1: Audit callers**

```bash
grep -rn 'vast_runner\|vast_client\|vastai\|VastRunner\|VastClient' --include='*.py' src/ app/ tests/ harness/
```

For each non-test caller:
- If the call was platform-agnostic (e.g., "submit a job to whichever backend the user picked"), redirect to `RunpodRunner`.
- If the call was Vast-specific UX (e.g., a "Run on Vast.ai" API route), remove the route and any UI artifact that referenced it.

- [ ] **Step 2: Delete files**

```bash
git rm app/services/vast_runner.py app/services/vast_client.py
git rm tests/app/test_vast_client.py tests/app/test_vast_runner.py tests/app/test_vastai_worker.py
```

- [ ] **Step 3: Remove `vastai` from `pyproject.toml` `[project] dependencies`**

```bash
grep -n vastai pyproject.toml
```

Delete the matching line (e.g., `  "vastai>=0.5.0",`). Verify the dependencies block still parses:

```bash
python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))" 2>/dev/null || \
python3 -c "import tomli; tomli.load(open('pyproject.toml','rb'))"
```

Expected: no error.

- [ ] **Step 4: Remove `vastai` from `.importlinter`**

Find the `[importlinter:contract:no-provider-sdk-outside-platforms]` block and drop `vastai` from `forbidden_modules`. The contract still forbids `runpod` and `beam`.

- [ ] **Step 5: Run the suite**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q --tb=line | tail -10
```

Expected: PASS. Any `ImportError` raised by glue that referenced the deleted modules must be fixed in this commit.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(v55-phaseD): deprecate and remove Vast.ai integration"
```

---

### Task D.8: Populate `platforms/__init__.py`

**Files:**
- Modify: `src/card_capture/platforms/__init__.py` (currently empty)

- [ ] **Step 1: Write the package init**

```python
"""Platform adapters: uniform PipelineRunner surface over local + remote backends."""
from card_capture.platforms.failures import (
    PROVIDER_FAILURE_CATEGORIES,
    ProviderFailure,
    map_provider_failure,
)
from card_capture.platforms.local import LocalRunner
from card_capture.platforms.manifests import export_manifest, import_manifest
from card_capture.platforms.runpod import RunpodRunner, RunpodRunnerError
from card_capture.platforms.beam import BeamRunner, BeamRunnerError

__all__ = [
    "PROVIDER_FAILURE_CATEGORIES",
    "ProviderFailure",
    "map_provider_failure",
    "LocalRunner",
    "RunpodRunner",
    "RunpodRunnerError",
    "BeamRunner",
    "BeamRunnerError",
    "export_manifest",
    "import_manifest",
]
```

- [ ] **Step 2: Smoke-import**

```bash
python3 -c "from card_capture.platforms import LocalRunner, RunpodRunner, BeamRunner, map_provider_failure, export_manifest; print('ok')"
```

Expected: `ok`. (RunpodRunner/BeamRunner imports of `runpod`/`beam` are lazy inside `__init__`, so the import succeeds even without the SDKs present.)

- [ ] **Step 3: Commit**

```bash
git add src/card_capture/platforms/__init__.py
git commit -m "feat(v55-phaseD): platforms package re-exports unified runner surface"
```

---

### Task D.9: Lint-imports clean; full suite green

**Files:** none — verification only.

- [ ] **Step 1: Lint**

```bash
PYTHONPATH=src:. lint-imports
```

Expected: all contracts PASS.

- [ ] **Step 2: Full suite**

```bash
python3 -m pytest tests/ -m 'not quarantine' -q --tb=line | tail -5
```

Expected: PASS, no errors. Pass count >= Phase-C baseline + the new platform tests (~10 new tests).

- [ ] **Step 3: Tag the milestone**

```bash
git tag v55-phaseD-complete
```

**Phase D complete.** Platform adapters share a uniform surface; Vast.ai is gone; tests/platforms is populated.
