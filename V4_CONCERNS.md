# v4 Concerns — Living Document

**Owner:** Josh (jpglick)
**Started:** 2026-05-12
**Last updated:** 2026-05-13 (Wave 2 sweep)
**Status:** Append-only as Wave 1/2 implementation lands. Resolve items by linking the PR or commit that closes them; do not silently delete.

This is the running list of concerns about the v4 implementation against the
contracts and plan in `CLAUDE.md` Appendix A. Concerns are tagged by severity:

- **Blocker** — must resolve before the next phase can ship; will cause
  silent data loss, contract drift, or rework.
- **High** — must resolve before merging to `main` or before Phase 0 can be
  declared done; will cause user-visible bugs or contract violations.
- **Medium** — should resolve within the current phase; will cause friction
  but is not load-bearing.
- **Low** — note and revisit; design tension or future tech debt.

When adding an item:
1. Pick the lowest severity that's still honest.
2. Cite a file path and line number when applicable.
3. Propose a fix or a decision the team needs to make. "Just a concern" is
   not enough — say what would close it.

---

## 1. Open concerns

### 1.1 — Migration runner silently no-ops then marks itself applied — **High** *(partial mitigation 2026-05-13)*

**Where:** `migrations/run_migrations.py` (`apply_migrations`).

**Status update:** `app/main.py` now calls `Storage(db).initialize()` *before*
`apply_migrations(db)`, so the FastAPI boot path no longer hits this bug.
But `migrations/run_migrations.py` itself is unchanged — it still swallows
"no such table" on `ALTER TABLE pipeline_events` and `CREATE INDEX … ON
pipeline_events`, then records the file as applied. Any other caller that
runs migrations before `storage.initialize` — the harness CLI, a Metaflow
step boot path, headless `card-capture process`, anyone running migrations
manually on a fresh DB — gets the same silent failure.

**Why the tests don't catch it:** `tests/migrations/test_schema.py`
either seeds `pipeline_events` first or never inspects it at all.

**Fix:** Don't `INSERT INTO _migrations` if any statement was skipped due
to "no such table"; let the next boot retry. Or — better — drop the skip
branch entirely and document that storage must be initialized first.

---

### 1.2 — `Optional` referenced but not imported in training router — **Blocker**

**Where:** `app/api/training.py:51, 60`.

The two new `/hard_cases` routes use `stage_id: Optional[str] = None` and
return `request.app.state.mining_service.list_hard_cases(stage_id=stage_id)`
but the file's imports are only:

```python
from fastapi import APIRouter, HTTPException, Request
from app.schemas.v1 import DatasetSummary, RetrainRequest, TrainingJobSummary, TrainingJobDetail
from app.services.training_service import TrainingService
```

No `from typing import Optional`, no `from __future__ import annotations`.
Python evaluates the annotation at function-def time, so this raises
`NameError: name 'Optional' is not defined` on import — meaning
`app/main.py` never loads, uvicorn never starts, and every contract
conformance test in `tests/app/test_api_contract.py` is dead-import-failing.

**Why no test caught it:** if the tests are run, they must currently be
failing on collection. If they're passing, CI must be skipping the file —
which is a separate problem (see §1.6). The bug ships either way.

**Fix:** Add `from typing import Optional` (and verify `tests/app/test_api_contract.py`
actually runs in CI before claiming this is closed).

---

### 1.3 — `fb_classifier.py` has a fatal import typo — **Blocker**

**Where:** `src/card_capture/ml/fb_classifier.py:7`.

```python
import torch.nn as annotations
```

Should be `import torch.nn as nn`. The file then refers to `torch.nn.Linear`
and `torch.nn.Module` via the unaliased `torch.nn` namespace, so the file
still *imports* (the bogus `annotations` alias is unused), but anyone who
later adds `nn.Module` / `nn.Sequential` will hit `NameError: name 'nn' is
not defined`. More importantly, the typo is a smoke signal that the F/B
classifier — the headline Wave 2 deliverable (Phase 3 #2 in the plan) —
was committed without being run.

**Fix:** Repair the alias, write a test that instantiates `FBClassifier()`
and runs a forward pass on a `1×3×224×224` tensor, and gate the next
Wave 2 PR on it.

---

### 1.4 — Tracker default disagrees between two Options dataclasses — **High**

**Where:** `src/card_capture/config.py:31` says `tracker_backend: str = "bytetrack"`;
`src/card_capture/pipeline.py:205` says `tracker_backend: str = "botsort"`.

Both are dataclasses, both expose `tracker_backend`, and the live pipeline
code at `src/card_capture/pipeline.py:467` keys behaviour off this string
(`if options.tracker_backend == "botsort": …`). Which one wins at runtime
depends on which Options instance the caller constructed. Phase 3 #4 of
the plan was specifically "swap default tracker to ByteTrack to avoid the
degraded-ReID problem" — so one of these is the intended new default, the
other is a stale duplicate.

**Why it matters:** anyone running the pipeline through CLI / Metaflow may
silently still be on BoT-SORT, defeating the Phase 3 #4 swap. The
`docs/ml/tracker-decision.md` (newly added) doesn't help if the default
isn't actually flipped everywhere.

**Fix:** Pick one Options dataclass as the single source of truth, delete
the duplicate, and add a test that asserts the runtime default matches
the documented decision.

---

### 1.5 — Wave 2 algorithmic changes shipped without a frozen baseline — **High** *(plan violation)*

**Plan reference:** Appendix A — "Each [Phase 3 fix] ships independently
and is gated by Phase 0 regression metrics." Also: "harness produces
stable metrics across 3 consecutive runs on the same video" is the
Phase 0 acceptance test.

What actually happened in the 2026-05-13 sweep:
- D2 (metric implementations) — merged ✅
- A2 (Metaflow decomposition) — merged ✅
- A3 (service layer + SSE) — merged ✅
- B0/B3 (frontend scaffold + threshold playground + regression UI) — merged ✅
- C1 (multi-frame fusion re-enabled), C3 (DINOv2+FAISS dedup), C4 (ByteTrack swap),
  C5 (RANSAC corner refinement), C7 (active learning), A4 (Apple-Silicon paths)
  — **all merged in the same wave, no recorded `baseline_v3` or `baseline_v4`
  regression run in between.**

There is no row in `regression_baselines` because there's no database to
hold one (the only `.sqlite` files in the tree are unit-test fixtures).
The implication: we have no number to point at to say "before/after C3,
dedup ARI moved by X." Every Phase 3 fix is now retroactively
unverifiable against pre-change behaviour.

**Fix:** Two paths, both acceptable:
- (a) Check out the last pre-Wave-2 commit (`60c04919`-ish, right after
  A2 lands and before C1/C3/C4/C5 merge), run the harness on whatever
  bootstrap videos we have, and call that `baseline_v3_pre_wave2`.
  Then run the harness on `main` and record the delta in this document.
- (b) Accept that Wave 2 was a vibes-based merge and re-run *all* the
  Phase 3 algorithmic changes individually with the harness gating each.
  Costlier but actually faithful to the plan.

Recommend (a) for pragmatism — the comparison is still meaningful even
if it's retroactive. Track the delta inside this doc.

---

### 1.6 — Contract conformance test only checks route existence, not shape — **High**

**Where:** `tests/app/test_api_contract.py`.

The test asserts: (i) each of the 8 contract routes is registered, (ii)
each route appears in the OpenAPI schema, (iii) stub routes return 501.
It does *not* verify any response body matches the Contract 2 shape.

So if a future PR changes `GET /api/v1/training/datasets` to return
`{name, size}` again, this test still passes. The §1.2-class drift that
just got fixed is unprotected against the next regression.

**Fix:** Per-route shape assertions. For every route in `ROUTES_REQUIRED`,
hit the implemented routes and `jsonschema.validate(response.json(),
contract_schema_for(route))`. The Pydantic models in `app/schemas/v1.py`
can produce the schemas; the contract markdown can be linted against
those schemas separately (see §1.10).

---

### 1.7 — Heavy code landed without a CI gate — **High** *(was Medium)*

**Where:** No `.github/workflows/` directory exists.

`pytest tests/` is not running anywhere automated. ~12,000 lines of code
landed in Wave 2 across A2/A3/B0/B3/C1/C3/C4/C5/C7/A4 with zero CI
enforcement. The §1.2 and §1.3 blockers are direct evidence: bugs that
trivial test runs would catch slipped past code review.

**Why bumped to High:** Wave 2 demonstrated that without CI, contract
compliance, import errors, and typos all ship to `main`. The minimum
viable CI is a single GH Actions workflow that runs `pytest tests/` on
every PR and blocks merge on failure.

**Fix:** Add `.github/workflows/test.yml` with `python -m pytest tests/`,
pin a Python version, install `pip install -e .[harness,test]`, and
require it as a status check on `main`.

---

### 1.8 — No 409 enforcement on concurrent retrain — **Medium**

**Where:** `app/services/training_service.py` (`start_retrain`).

Contract 2's `POST /api/v1/training/retrain/{model_name}` spec says
"`409` if a retrain is already running." The implementation always
spawns a daemon thread on each call, so two simultaneous POSTs for
`fb_classifier` quietly start two training threads.

**Fix:** Track a per-model active job. If `start_retrain` is called for
a model whose latest job is still `queued` or `running`, raise
`HTTPException(409)` from the router (or a domain exception the router
maps to 409).

---

### 1.9 — Schema duplication between docs and code, no drift gate — **Medium**

**Where:** Contracts 1 and 4 are authored as prose + DDL/JSON in
`docs/contracts/`, and re-encoded as code in `migrations/*.sql` and
`harness/schema.py`. There is no test that asserts the two stay in sync.

If someone edits `harness/schema.py` to add a field, `docs/contracts/truth-schema.md`
won't be touched, and Surface B's labeling UX will produce truth files
that pass validation but mismatch the spec the rest of the team is
coding against.

**Fix candidates:**
- Generate the contract markdown from the Pydantic schema (one direction
  of truth) — lowest friction.
- Or add a CI check that round-trips: parse the DDL/JSON-schema out of
  the markdown and diff against the runtime schema. More work, but
  works for both Contract 1 (DDL) and Contract 4 (JSON).

Either way, before more contracts get added in Wave 3, lock down the
sync mechanism.

---

### 1.10 — `harness/runner.py` returns mixed `float | dataclass` metric values — **Medium**

**Where:** `harness/runner.py` (`PerVideoReport.metrics`) and `harness/cli.py`
(`_compute_deltas`).

`dedup_accuracy` returns a `DedupAccuracy` dataclass; `image_quality`
returns an `ImageQuality` dataclass; the other three metrics return
`float | None`. The per-video metrics dict is therefore heterogeneous.
The CLI's `_compute_deltas` (computing `report.metrics - baseline.metrics`)
has to special-case the dataclass branches; the JSON-persistence path in
`baseline.freeze_baseline` will currently fail to `json.dumps` the
dataclass instances unless they're explicitly converted.

**Fix:** Pick a single representation — either pure `dict[str, float | None]`
returned by each metric, or a typed `MetricResult` wrapper. Add a
`runner_smoke` test that round-trips a full `Report` through JSON.

---

### 1.11 — `harness/cli.py` ships with `config={}` TODO placeholders — **Medium**

**Where:** `harness/cli.py:90, 158`.

```python
config={},  # TODO: load current pipeline config
```

The harness records every run with an empty config blob — so when a
future operator looks back at a regression run and asks "what config
produced these numbers?", the answer is "we don't know."

**Fix:** Wire `harness.cli` to load the actual pipeline config (from
`card_capture.config` or whatever preset was used) before calling
`persist_run`. Either Surface A or Surface D owns this, but the latent
TODO must be filled in before any baseline tagged from the CLI is
trustworthy. Best fix is to delete the TODOs by making config-loading
the responsibility of `run_metrics`.

---

### 1.12 — Truth-file naming convention is not in the contract — **Medium**

**Where:** `harness/runner.py` `_find_truth` tries three filename
conventions in order: `<truth_dir>/<video_id>.truth.json`,
`<truth_dir>/<video_id>/truth.json`, `<truth_dir>/truth.json`.

Contract 4 doesn't specify the on-disk filename — only the JSON shape.
Three conventions in code means downstream tooling (labeling UX,
golden-set bootstrap scripts, future operators) will inevitably pick
different ones and we'll silently lose track of which truth file
"belongs" to which video.

**Fix:** Pick one (recommend `<truth_dir>/<video_id>.truth.json` — flat,
sortable, deterministic), add it to `docs/contracts/truth-schema.md`,
and have the runner refuse the other two with a warning that includes
the path it expected.

---

### 1.13 — Synthetic eval datasets risk masking the missing real-data path — **Low**

**Where:** `src/card_capture/ml/synthetic_eval.py`.

Synthetic F/B and dedup datasets are sensible for unblocking the C-surface
training scaffold before real labels exist. But once the F/B classifier is
wired to real `fb_labels`, anyone running tests will keep hitting the
synthetic dataset and may forget the model never trained on real cards.

**Fix:** Mark every test that uses the synthetic eval explicitly
("smoke-only, not for accuracy"), and add a CI guard that fails if the
synthetic eval is still in use after the real F/B trainer ships (a
sentinel constant like `SYNTHETIC_EVAL_ONLY = True` that flips to
`False` and breaks the smoke test once real training is wired).

---

### 1.14 — `app/web/` adds 71MB of node_modules to working trees — **Low**

**Where:** `app/web/node_modules/`.

The directory is correctly gitignored via `app/web/.gitignore`, so it
won't be committed. But every fresh checkout that runs `npm install`
will produce 71MB of dependencies, and there's no `package-lock.json`
pinning policy or `npm ci` documented. Worth noting before more JS
deps land in Wave 3 (the Svelte threshold playground will likely grow).

**Fix:** Document the install/build flow in a top-level README section
or `app/web/README.md`, and decide whether to pin to `npm ci` for
reproducibility.

---

### 1.15 — D1 PR (#37) was a no-op cleanup, not the deliverable its title claims — **Low / trivial**

**Where:** Commit `39c872ff` "feat(harness): D1 — Truth Schema + Validator".

The PR only deletes `src/card_capture.egg-info/`. The actual D1 scope
(harness schema + validator) was delivered inside D0 (#35). Not a code
issue; just makes the changelog confusing for anyone scanning commit
titles later. Mention it in the v4 retrospective.

---

## 2. Resolved concerns

### 2.1 — *(was §1.2 v1)* C0 training endpoints diverge from frozen Contract 2 — **Resolved** in PR #38 / #44

`app/api/training.py`, `app/services/training_service.py`, and
`app/schemas/v1.py` now match Contract 2's shapes:
- `GET /datasets` returns `[{model_name, total_labels, class_distribution, last_updated}]`
- `POST /retrain` accepts `{epochs, learning_rate}` and returns
  `{job_id, model_name, status, created_at}`
- `GET /jobs/{job_id}` returns `{job_id, model_name, status, progress, created_at, completed_at}`

**Note:** the conformance test guarding this is weak — see §1.6 for the
follow-up.

---

### 2.2 — *(was §1.3 v1)* `app/main.py` does not run migrations at startup — **Resolved** in PR #38

`app/main.py:create_app` now runs `Storage(db).initialize()` followed by
`apply_migrations(db_path)`. Order matters and is correct (see §1.1 for
the residual bug in the migration runner itself).

---

### 2.3 — *(was §1.4 v1)* Phase 0 is not yet self-sufficient — **Resolved** in PR #40 / #44

Metric implementations (`harness/metrics/*.py`), the `harness` CLI
(`run`, `freeze-baseline`), `harness/match.py`, and `harness/runner.py`
all shipped. What's *still* needed to call Phase 0 "done":
- 15 labeled videos (operational, not code)
- 3-run stability test (operational, not code)
- A frozen `baseline_v3` or `baseline_v4` row — **this is now §1.5**.

---

### 2.4 — *(was §1.5 v1)* `pipeline.contracts.RunContext` referenced but not implemented — **Resolved** in PR #45

`pipeline/card_capture_flow.py` exists with the FlowSpec spine; per-step
modules in `pipeline/steps/` (`detect`, `novelty`, `track`, `refine`,
`score`, `resolve`, `fuse`, `dedup`, `store`) carry the artifact wiring.

**Note:** the artifact-shape contracts in Contract 3 are not yet enforced
by tests — a `RunContext` dataclass mismatch between a step and the
contract would not currently be caught. Follow-up sits under §1.6's
contract-conformance gap.

---

### 2.5 — *(was §1.9 v1)* `TrainingService` job execution path missing — **Partially resolved** in PR #38

The threading and job state machine now exist: `start_retrain` spawns
a daemon thread that walks `queued → running → completed | failed`,
and `TrainingJob` exposes `progress`, `created_at`, `completed_at`,
`error`. The actual training body for each model is still a `pass`
inside `_run` — i.e. POSTing `/retrain/fb_classifier` will mark the
job `completed` without actually training. Tracked here so it isn't
mistaken for a working pipeline; reopen as a separate concern when
real training is required.

---

## 3. Process notes

---

## 3. Process notes

- **Concerns don't block merges by themselves.** They block the *phase
  declaration* — Phase 0 can't be called complete while §1.5 is open;
  Wave 2 can't be called complete while §1.2, §1.3, §1.4 are open.
- **Blockers in §1 mean exactly that:** §1.2 (NameError) and §1.3
  (fb_classifier typo) imply the app and the F/B trainer don't run.
  Treat them as the next two things to fix.
- **The Wave 2 sweep ran without a gating baseline (§1.5) and without
  CI (§1.7).** Re-establish *both* before any Wave 3 algorithmic
  change merges, otherwise the same drift repeats.
- **When in doubt, add the concern.** A duplicate or trivial concern
  costs nothing; a missed `Blocker` costs a re-spin.
- **The `Plan` doc (`CLAUDE.md` Appendix A) is the source of truth for
  intent. The `docs/contracts/` files are the source of truth for
  shape.** When this doc disagrees with either, this doc is wrong.
