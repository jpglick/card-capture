# v4 Concerns — Living Document

**Owner:** Josh (jpglick)
**Started:** 2026-05-12
**Last updated:** 2026-05-13 (Wave 3 sweep — implementation complete; training/labeling deferred)
**Status:** Append-only as Wave 1/2/3 implementation lands. Resolve items by linking the PR or commit that closes them; do not silently delete.

**Wave 3 framing:** The user has explicitly deferred all manual / training-data
steps (labeled videos, real F/B classifier training, dedup threshold
calibration, frozen baselines) until the code paths are fully built. Concerns
below related to those deferments are tagged **[deferred]** — they remain
real, but they're not bugs in the current build; they're work waiting on
human input. Everything else is fair game for the next dev cycle.

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

### 1.1 — Pipeline now has two parallel implementations (monolith + Metaflow) drifting asymmetrically — **High** *(new in Wave 3 review)*

*(Resolved decision-wise via ADR 0004. Execution is pending Surface A.)*
See: `docs/architecture/decisions/0004-v4-modular-pipeline.md`


**Where:** `src/card_capture/pipeline.py` (the 2,000-line monolith) and
`pipeline/card_capture_flow.py` + `pipeline/steps/*` (the Metaflow
decomposition from PR #45).

**Symptom:** Both paths are present, both are wired into entry points, and
recent commits modified *both* — but not equivalently. The 2026-05-13
fusion commit (`4729b264`) re-enabled multi-frame fusion in the monolith
(17 LOC). The 2026-05-13 Wave 3 commit (`8fa6fec6`) consolidated fusion
into `MultiFrameFuser` and touched `pipeline/steps/fuse.py` (2 LOC) plus
six other step files. There is no test that asserts the two pipelines
produce the same artifacts on the same video, and no documented "this is
the canonical path; that one is deprecated."

**Why it matters:** Threshold tuning, regression baselines, A/B configs,
and harness comparisons will produce different numbers depending on which
pipeline executed. The plan (Appendix A.1, Phase 2) was to *replace* the
monolith with Metaflow, not run both forever.

**Fix:** Decide. Either (a) mark the monolith deprecated, route every
entry point — CLI, harness CLI, headless processing, regression
playground — through the Metaflow flow, and delete or freeze the
monolith; or (b) admit the monolith is the production path and treat
Metaflow as the "future" path, in which case Wave 3 algorithmic changes
should land in the monolith first and the steps modules second. Either
way, add a smoke test that runs both on a fixture video and diffs the
top-line outputs.

---

### 1.2 — ~32 modified files sit uncommitted in the working tree — **High** *(new)*

**Where:** `git status` shows 31 modified files and 5 untracked items as
the "rest of the implementation" the user just described:

```
modified:   app/api/{cards,label,regression,training,videos}.py
modified:   app/schemas/v1.py
modified:   app/services/{cards,labeling,regression,runs,training,video}_service.py
modified:   app/web/src/lib/api.ts, app/web/src/routes/+layout.svelte
modified:   harness/{cli,match,metrics/image_quality}.py
modified:   migrations/run_migrations.py
modified:   pipeline/card_capture_flow.py
modified:   pipeline/steps/{dedup,refine,resolve,start,store}.py
modified:   src/card_capture/{ingestion,pipeline,storage}.py
modified:   src/card_capture/ml/fb_classifier.py
modified:   tests/app/test_api_contract.py, tests/test_pipeline.py
untracked:  app/web/src/routes/training/
untracked:  golden_set/videos/IMG_5872/reference_frames/  (6 PNGs)
untracked:  harness_config.json
untracked:  scripts/generate_reference_frames.py
untracked:  src/card_capture/ml/inference/fb_predict.py
```

This is the state of v4 *as reviewed*. Future `git log` won't reflect it;
bisects will be impossible; CI cannot run against it; collaborators
checking out `main` will see a different codebase.

**Fix:** Commit, in a sensible grouping (e.g. one commit per surface), or
at minimum tag the WIP as a single "v4-wave3-WIP" snapshot so the
review point is reconstructable.

---

### 1.3 — Bootstrap golden-set assets partially tracked, partially loose — **Medium** *(new)*

**Where:** `golden_set/`.

`golden_set/videos/IMG_5872/truth.json` was committed in Wave 3. The
6 reference PNGs in `golden_set/videos/IMG_5872/reference_frames/` are
untracked. `golden_set/README.md` and `_index.txt` are committed but
empty of substance.

If the reference frames are bootstrap fixtures (used for SSIM image-quality
scoring against the canonical fused output), they need to be tracked or
regenerable. If they're throwaway, the directory shouldn't exist on disk.
Currently it's ambiguous, and the policy will be set by whoever first runs
`git add -A`.

**Fix:** Decide: are the reference frames committed (then add to git),
regenerable via `scripts/generate_reference_frames.py` (then gitignore
them and document the regen command in `golden_set/README.md`), or
neither (then delete). The script `scripts/generate_reference_frames.py`
exists but is also untracked, so its provenance is unclear.

---

### 1.4 — `harness_config.json` untracked at repo root — **Medium** *(new)*

**Where:** `/Users/josh/code/card-capture/harness_config.json`.

Untracked. Repo-root config files have a way of becoming load-bearing
without anyone noticing. If this is the harness's runtime config (paths,
thresholds, baseline name to compare against), it needs to be checked in
under `harness/` (or `golden_set/`) with documented schema, not floating
at the repo root.

**Fix:** Move into `harness/config.json` or `golden_set/harness_config.json`,
add a Pydantic schema for its shape, and document it in
`docs/contracts/`.

---

### 1.5 — `FBPredictor` returns random labels when no checkpoint exists — **High** *(new)*

**Where:** `src/card_capture/ml/inference/fb_predict.py:18-22`.

```python
self.model = FBClassifier(pretrained=(checkpoint_path is None))
if checkpoint_path and Path(checkpoint_path).exists():
    ckpt = torch.load(checkpoint_path, map_location=self.device)
    …
```

When no checkpoint is provided (the deferred-training state), `FBClassifier`
is instantiated with `pretrained=True` — meaning ImageNet-pretrained
ResNet-18 with a *randomly-initialised* final `Linear(num_ftrs, 2)` head.
Predictions are effectively random; the model will confidently emit
`("front", 0.97)` or `("back", 0.98)` based on noise in the head.

The pipeline currently uses the longest-track-equals-Front heuristic per
the plan critique, but if any code path *also* calls `FBPredictor` without
a checkpoint as a fallback or augmentation, it will pollute results with
random labels.

**Fix:** Raise `FileNotFoundError` (or a domain-specific
`UntrainedModelError`) in `FBPredictor.__init__` when no checkpoint is
provided. Add a `available()` classmethod that the pipeline can probe
before deciding whether to trust predictions. Add a test that asserts
predictions are deterministic and `> 0.6` confident on a known fixture
when a checkpoint exists, and *unavailable* when one doesn't.

---

### 1.6 — `reid_embedding` column added but new runs default to ByteTrack (no ReID) — **Medium** *(new)*

**Where:** `migrations/0002_wave3_schema.sql` adds `ALTER TABLE card_instances
ADD COLUMN reid_embedding BLOB;`. Default tracker is now `bytetrack`
(`config.py:31`, `pipeline.py:205`), which doesn't produce a ReID embedding.

**Symptom:** the new column will be NULL for every new run unless someone
explicitly overrides `tracker_backend=botsort`. Cross-video dedup that
keys off this column (`deduplicator.py` ReID cosine path) will silently
fall back to the pHash path, and we won't notice because dedup will still
"work" — just worse.

**Fix:** Either (a) populate `reid_embedding` even on the ByteTrack path
by running OSNet (or DINOv2) on the rectified canonical view at store
time; or (b) explicitly drop the column from the schema for runs that
don't intend to produce it; or (c) document that the column is
conditionally populated and add an integration test that asserts the
expected NULL rate.

---

### 1.7 — Wave 3 added more algorithmic change without a frozen baseline (compounds §2.X) — **High** *(extends prior §1.5)*

The Wave 3 commit (`8fa6fec6`) shipped:

- Online adaptive thresholds for novelty and Hamming gates
- Lab-luminance switch for glare-rejection selection
- Foil-detection signal hygiene fix (operate on unregistered frames)
- Stage 9 dead-code fix (persist fused canonical)
- `MultiFrameFuser` consolidation

These are *each* the kind of change the plan said must be Phase-0-gated.
No baseline exists; the user has explicitly deferred labeling. So the
algorithmic deltas are doubly unmeasurable now: not just against the
pre-Wave-2 monolith, but also against the pre-Wave-3 Metaflow output.

**Decision needed:** is the user accepting that Wave 1/2/3 ships
unmeasured ("trust the spec, the harness will tell us afterwards"), or
will we stop algorithmic changes until at least one baseline is frozen?
If the former, this concern stays open as a *known liability* the team
has consciously accepted. If the latter, no more `feat(ml,…)` PRs until
§2.4 / golden-set bootstrap completes.

---

### 1.8 — Migration runner's silent skip needs at least a log — **Low** *(new, follow-up to resolved §2.1)*

**Where:** `migrations/run_migrations.py` (the `all_ok` patch).

The §2.1 fix is correct — the runner no longer marks itself applied on a
partial migration — but the skip is now silent. A user who runs
migrations against a fresh DB and forgets to `storage.initialize()` first
will see the migration appear to succeed (no error, no row in
`_migrations`) and only discover the failure when a downstream query
hits the missing column.

**Fix:** `print` or `logging.warning` on every "no such table" skip, with
the statement and table name. One line. Defensive instrumentation costs
nothing and saves an hour of debugging.

---

### 1.9 — No CI gate exists; manual review is the only test signal — **Resolved (see §2.12)**

**Where:** No `.github/workflows/` directory exists.

`pytest tests/` is still not running anywhere automated. Wave 2 shipped
the original §1.2 NameError and the original §1.3 typo because of this;
they were caught by humans, not by CI. Wave 3 fixed those manually too
but has now landed thousands of additional uncommitted lines (see new
§1.2) with the same zero-CI exposure.

**Fix:** Add `.github/workflows/test.yml` running
`pytest tests/` on every PR; gate merge to `main` on it. Pin a Python
version. Install via `pip install -e .[harness,test]`. This is the
single highest-leverage outstanding piece of infrastructure.

---

### 1.10 — Schema duplication between docs and code, no drift gate — **Resolved (see §2.13)**

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

Either way, before more contracts get added in Wave 4, lock down the
sync mechanism.

---

### 1.11 — `harness/runner.py` returns mixed `float | dataclass` metric values — **Medium**

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

### 1.12 — `harness/cli.py` ships with `config={}` TODO placeholders — **Medium**

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

### 1.13 — Truth-file naming convention is not in the contract — **Medium**

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

### 1.14 — Synthetic eval datasets risk masking the missing real-data path — **Low** *(deferred)*

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

### 1.15 — `app/web/` adds 71MB of node_modules to working trees — **Low**

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

### 1.16 — D1 PR (#37) was a no-op cleanup, not the deliverable its title claims — **Low / trivial**

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
real training is required. **[deferred per user]**

---

### 2.6 — *(was §1.1 v2)* Migration runner skipped pipeline_events ALTERs then marked itself applied — **Resolved** in working-tree edit

`migrations/run_migrations.py` now tracks an `all_ok` flag per file: a
statement that fails with "no such table" leaves `all_ok = False`, and
the `INSERT INTO _migrations` only fires when `all_ok` is true.
Subsequent boots will retry the migration. Follow-up §1.8 (add a log
on the skip) remains.

---

### 2.7 — *(was §1.2 v2)* `Optional` referenced but not imported in training router — **Resolved** in working-tree edit

`app/api/training.py:7` now reads `from typing import Optional`. App
boots; `/api/v1/training/hard_cases` evaluates its annotation cleanly.

---

### 2.8 — *(was §1.3 v2)* `fb_classifier.py` had a fatal `import torch.nn as annotations` typo — **Resolved** in working-tree edit

`src/card_capture/ml/fb_classifier.py:7` now reads `import torch.nn as nn`.
Follow-up §1.5 (random predictions without checkpoint) is the next
F/B concern and lives in §1.

---

### 2.9 — *(was §1.4 v2)* Tracker default disagreed between two Options dataclasses — **Resolved** in working-tree edit

Both `src/card_capture/config.py:31` and `src/card_capture/pipeline.py:205`
now say `tracker_backend: str = "bytetrack"`. Phase 3 #4's intended
swap is in effect on both paths.

---

### 2.10 — *(was §1.6 v2)* Contract conformance test only checked route existence — **Resolved** in working-tree edit

`tests/app/test_api_contract.py` now uses Pydantic `TypeAdapter` to
validate response bodies against `app/schemas/v1.py` models for every
implemented route. A future PR that changes the shape of
`/api/v1/training/datasets` (or any other validated route) will fail
the test. Shape coverage is still gated on the test being committed
(see new §1.2) and on CI running it (see new §1.9).

---

### 2.11 — *(was §1.8 v2)* No 409 enforcement on concurrent retrain — **Resolved** in working-tree edit

`TrainingService.start_retrain` now scans existing jobs under the
service's lock and raises `ValueError` if a `queued`/`running` job
for the same model exists; the router maps the exception to
`HTTPException(409)`. Matches Contract 2.

---

### 2.12 — *(was §1.9)* No CI gate exists — **Resolved** in Wave 4

Added `.github/workflows/ci.yml` to run `pytest tests/` on push and PR to `main`.

---

### 2.13 — *(was §1.10)* Schema duplication between docs and code — **Resolved** in Wave 4

Added `scripts/validate_schema_docs.py` which runs in CI to verify that `migrations/` SQL matches `docs/contracts/storage-schema.md`.

---

## 3. Process notes

- **Concerns don't block merges by themselves.** They block the *phase
  declaration*. Wave 3 ships with §1.1 (two parallel pipelines), §1.2
  (uncommitted work), §1.5 (random F/B predictor), §1.7 (no baseline),
  and §1.9 (no CI) all open — these define the next sprint.
- **The user has deferred training and labeling work.** Concerns tagged
  *[deferred]* in §1 remain real but are not blocking the next code
  cycle. Re-open them once the operator has produced labels.
- **The new top-of-priority list, in order:**
  1. Commit the working tree (§1.2). Without this, nothing else can be
     verified or bisected.
  2. Add a CI workflow (§1.9). Even just `pytest tests/` on PRs.
  3. Decide the monolith-vs-Metaflow story (§1.1) — pick a canonical
     path and route every entry point through it.
  4. Make `FBPredictor` refuse to predict without a checkpoint (§1.5).
  5. Choose a `reid_embedding` policy (§1.6) so the new column either
     gets populated on every run or gets removed.
- **The Wave 2 / Wave 3 sweep ran without a gating baseline (§1.7) and
  without CI (§1.9).** This pattern keeps repeating. Establish *both*
  before any further `feat(ml,…)` change merges.
- **When in doubt, add the concern.** A duplicate or trivial concern
  costs nothing; a missed `Blocker` costs a re-spin.
- **The `Plan` doc (`CLAUDE.md` Appendix A) is the source of truth for
  intent. The `docs/contracts/` files are the source of truth for
  shape.** When this doc disagrees with either, this doc is wrong.

---

## 4. Plan items not implemented

These are items from `CLAUDE.md` (Appendix A plan and §9/§10 critique
table) that have *not* been built and are *not* otherwise tracked as
concerns above. They are recorded here so the next dev cycle has a
single shopping list and we don't lose them between waves.

Severity here means "expected impact if missing":
- **High** — the plan called this out as a primary fix; absence is
  a known accuracy / capability ceiling.
- **Medium** — useful, not load-bearing.
- **Low** — polish.

### 4.1 — Learned quality ranker — **High**

**Plan ref:** `CLAUDE.md` §9 next steps, §10 critique row 5.

`scoring.py` still uses the hand-weighted seven-component sum
(`sharpness 25% + glare 15% + aspect 15% + size 10% + complexity 10% +
border_purity 20% + confidence 5%`). The plan calls for a learned
ranker fit on reviewer-labeled crops. No `quality_ranker.py` or training
script exists; `src/card_capture/ml/` has `fb_classifier.py` and a
`registry`/`scaffolding` skeleton, nothing for ranking.

**Closes when:** there's an `MLQualityRanker` model class, a training
script that fits on accepted/rejected card labels (sourced from the
review UI or human-labeled crops), and `QualityScorer` calls into it
behind a feature flag. Same labeling-deferred caveat as F/B applies.

---

### 4.2 — Per-pixel background model with variance — **High**

**Plan ref:** `CLAUDE.md` §9 next steps, §10 critique row 4.

`presence/background_novelty.py` still computes a single mean grayscale
across "empty" frames and gates with `|diff| / 255 > 0.08`. The plan
called for a per-pixel Gaussian (mean + variance) with a running update
so lighting drift doesn't soften the gate. Wave 3 added *adaptive
thresholds on the novelty score* (compounds the problem visibility but
doesn't fix the background-model side).

**Closes when:** the background model carries per-pixel variance, the
gate is Mahalanobis-distance based rather than absolute-difference, and
a running update over the run keeps the model fresh. Integration test
should exercise a long video with shifting lighting and confirm
candidate-drop rates stay stable.

---

### 4.3 — Per-region detector confidence / ROI-aware detection — **High**

**Plan ref:** `CLAUDE.md` §10 critique row 1.

`detectors.py` still uses a single global `corner_confidence = 0.5`
gate. The plan called this out as missing partial / corner / occluded
cards. No per-region or ROI-aware path was added in Wave 1-3.

**Closes when:** detector results include a per-region confidence (or
ROI hint from the sampler), and the candidate-acceptance threshold can
be lower in regions the sampler has flagged as likely card.

---

### 4.4 — Content-aware F/B and dedup similarity metric — **High** *(blocked on F/B classifier training)*

**Plan ref:** `CLAUDE.md` §10 critique row 2; Phase 3 #2.

`deduplicator.py` still uses pHash for Front/Back gating with a 22/64
Hamming tolerance — the plan called this "the wrong invariant" because
pHash measures contour similarity, not card-side similarity. The
intended fix is the trained F/B classifier (`fb_classifier.py` exists
but is untrained, see §1.5) plus a content-aware Front/Back metric.

DINOv2 embeddings shipped (C3) for *cross-card* dedup, but the
*within-physical-card* Front/Back match still rides on pHash.

**Closes when:** the F/B classifier is trained and the Front/Back gate
uses (a) F/B prediction + (b) content-aware similarity instead of
pHash. Tracked here rather than in §1 because the bug is the design
choice, not a defect — and the fix requires training data.

---

### 4.5 — Higher-resolution canvas (1000×1400) + Lanczos resampling — **Medium**

**Plan ref:** `CLAUDE.md` §10 critique row 8; Phase 4 implicit.

Output canvas is still fixed at 750×1050 with bilinear interpolation in
both the Kornia and vImage paths. The plan noted this throws away
sub-pixel content for grading-downstream tasks. A4 (Apple-Silicon)
landed the vImage backend but didn't bump the canvas size or switch
to Lanczos.

**Closes when:** canvas resolution is a config option (with 1000×1400
as a "quality" preset), and at least one backend (GPU Lanczos or vImage
high-quality) supports the higher-res path without doubling latency.

---

### 4.6 — YOLO26-OBB model swap — **Medium**

**Plan ref:** `CLAUDE.md` Appendix A.1 Phase 4 #1.

A4 added `src/card_capture/ml/models/apple_silicon.py` and
`coreml_detector.py` (the CoreML inference wrapper) but kept the
YOLOv8-OBB weights. The plan called for replacing the *model* with
YOLO26-OBB (newer architecture, better OBB accuracy) on the CoreML path.

**Closes when:** the CoreML detector loads a YOLO26-OBB checkpoint, the
PyTorch fallback keeps YOLOv8-OBB for cross-platform, and harness shows
no regression on detector recall.

---

### 4.7 — VideoToolbox decoder on macOS — **Medium**

**Plan ref:** `CLAUDE.md` Appendix A.1 Phase 4 #2.

A4 added vImage warp but the decoder path in `ingestion.py` still uses
OpenCV / decord. VideoToolbox would give hardware-accelerated H.264/H.265
decode on Apple silicon.

**Closes when:** decoder backend is feature-detected at startup,
VideoToolbox is used on macOS, OpenCV/decord elsewhere, and a
single-video smoke test confirms parity of decoded frame timestamps.

---

### 4.8 — Detection-conditioned sampler / objectness signal — **Medium**

**Plan ref:** `CLAUDE.md` §10 critique row 6.

The MobileNetV3 sampler classifier in `presence/classifier.py` operates
frame-globally. The plan flagged this as too permissive — bystander
cards in corners activate the same as centered cards. Fix is to
condition the sampler on detector ROI hints or a learned objectness
map.

**Closes when:** the sampler accepts a detector-conditioned signal
(either a YOLO objectness heatmap or a fast objectness head) and only
fires presence when the high-activation region is within a reasonable
ROI band.

---

### 4.9 — Multi-process error handling with structured exit codes — **Medium**

**Plan ref:** `CLAUDE.md` §10 critique row 7.

The producer/consumer subprocess split (Stages 1-3) still has the
"silent crash on MPS/CUDA init failure" failure mode. No structured
error code protocol was added. A bad GPU driver or missing weights
file produces an unhelpful timeout rather than a typed error.

**Closes when:** subprocesses report structured exit codes
(`STARTUP_OK`, `MODEL_LOAD_FAILED`, `DEVICE_INIT_FAILED`, etc.) over
a control channel, and the orchestrator surfaces them in
`run_telemetry.json` and the run-detail UI.

---

### 4.10 — User-defined config presets persistence — **Medium**

**Where:** `app/api/config.py` — `POST /api/v1/config/presets` has a
`# TODO: implement user presets in DB` and returns the payload
unchanged without persisting.

**Closes when:** there's a `config_presets` table (likely an additive
v4 migration), `create_preset` writes to it, `list_presets` unions
built-in + user presets, and the UI Settings tab can save/load them.

---

### 4.11 — A.5.2 A/B comparison view body — **Medium**

**Plan ref:** `CLAUDE.md` Appendix A.5.2.

Route `app/web/src/routes/regression/compare` exists as a directory but
needs to be verified end-to-end: pick run A and run B (same video,
different config), side-by-side cards, dedup groups, side assignments,
diff highlighting, metric-delta strip. The backend endpoint that powers
this isn't documented in `docs/contracts/v1-api.md`.

**Closes when:** the comparison endpoint is added to Contract 2,
implemented in `regression_service`, and the Svelte page renders the
diff with green/red highlights for added/removed cards.

---

### 4.12 — Sampler ML model retrain path — **Low**

The presence classifier (`presence/classifier.py`) loads
`models/presence_classifier.pt` if it exists, else falls back to Otsu.
There's no `train_presence.py` wired into the training service, so
the model is "frozen at whatever weights happened to ship." Plan
doesn't explicitly require retraining this — but if F/B classifier
training infrastructure lands (Phase 3 #2), the same plumbing should
cover presence.

**Closes when:** `TrainingService` exposes presence retraining
alongside F/B, and the `model_versions` table tracks both.

---

### 4.13 — Real ReID embeddings on BoT-SORT path — **Low**

**Plan ref:** `CLAUDE.md` §9 next steps, §10 critique row 3.

The BoT-SORT adapter was historically fed dummy images for ReID.
Wave 2 swapped the default tracker to ByteTrack (which has no ReID),
which sidesteps but doesn't fix the underlying issue. If anyone
deliberately runs `tracker_backend=botsort`, embeddings are still
degraded.

**Closes when:** `BoTSORTAdapter` is given real rectified frames for
the appearance backbone (or is officially deprecated and removed in
favour of ByteTrack-only).

---

### 4.14 — Settings tab — **Low**

`/app/web/src/routes/settings` exists but only nests the threshold
playground (B3). The plan called for a Settings section with config
preset sliders + tooltips explaining trade-offs. Currently no top-level
settings landing page beyond the playground entry.

**Closes when:** `/settings` has a list of presets with slider UI,
tooltips, and a "save as preset" button (paired with §4.10).

---

### 4.15 — A.4.1 Inbox: drag-drop video upload — **Low**

`/videos` route exists. Per the plan it should be a drag-drop inbox
with queued / processing / completed / failed states. Needs an
end-to-end check that drag-drop works and the queue UI surfaces
SSE-driven progress (the SSE channel from A3 is in place).

**Closes when:** drag-drop a video into `/videos` triggers ingestion
via `POST /api/v1/videos`, and the queue card live-updates from the
SSE feed.

---

### 4.16 — `card_capture.config` and `pipeline.py:Options` are still two competing config dataclasses — **Low**

§2.9 resolved the *tracker default* mismatch, but the two dataclasses
themselves (one in `src/card_capture/config.py`, one in
`src/card_capture/pipeline.py:Options`) still exist side-by-side and
expose overlapping fields. They've been kept in sync by hand; the
next mismatch will be the same flavour of bug as the original §1.4.

**Closes when:** there's one config dataclass, exported from
`src/card_capture/config.py`, and `pipeline.py` imports it rather than
defining its own.

---

### 4.17 — 15 labeled videos + frozen baseline + 3-run stability — **[deferred per user]**

Phase 0's operational acceptance test. The user has explicitly deferred
this work until everything else is built. Mentioned here only so the
plan-coverage list is complete; reopen when ready.
