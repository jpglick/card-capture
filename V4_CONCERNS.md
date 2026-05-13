# v4 Concerns — Living Document

**Owner:** Josh (jpglick)
**Started:** 2026-05-12
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

### 1.1 — Migration runner silently no-ops then marks itself applied — **Blocker**

**Where:** `migrations/run_migrations.py` (`apply_migrations`).

**Symptom:** On a fresh DB, `pipeline_events` doesn't exist yet when
`apply_migrations` runs (it's created later by `storage.py`). The runner
catches `no such table` on `ALTER TABLE pipeline_events ADD COLUMN …` and
`CREATE INDEX … ON pipeline_events(…)`, skips them, **and then records
`0001_v4_schema.sql` as applied in the `_migrations` table.** Subsequent
boots see the migration as already applied and never retry. Result:
`pipeline_events.stage_id` and `pipeline_events.artifact_ref` never exist,
breaking Surface A's commitment in Contract 1.

**Why the tests don't catch it:** `tests/migrations/test_schema.py`
seeds `pipeline_events` *before* calling `apply_migrations`, so the
ALTER never goes through the skip branch.

**Fix candidates:**
- (a) Run `apply_migrations` *after* `storage.py` initialises its tables
  at boot. Contract 1 already says "Surface A runs all migrations via
  `migrations/run_migrations.py` at startup" — that wiring is missing and
  should happen post-storage-init.
- (b) Track per-statement application instead of per-file; or split the
  `pipeline_events` ALTERs into a separate migration file that runs later.
- (c) Don't record the migration as applied if any statement was skipped
  due to "no such table"; let the next boot retry.

Recommend (a) + a regression test that boots a fresh DB the *real* way
(via the same code path the app uses, not by pre-seeding tables).

---

### 1.2 — C0 training endpoints diverge from frozen Contract 2 — **High**

**Where:** `app/api/training.py`, `app/services/training_service.py`.

**Contract reference:** `docs/contracts/v1-api.md` §5 Training.

| Route | Contract | Implementation |
|---|---|---|
| `GET /api/v1/training/datasets` | `[{model_name, total_labels, class_distribution, last_updated}]` keyed per **model** (e.g. `fb_classifier`) | `[{name, size}]` keyed per **table** (`fb_labels`, `dedup_clusters`) — wrong concept, not just missing fields |
| `POST /api/v1/training/retrain/{model_name}` request | `{epochs, learning_rate}` | `{dry_run: bool}` |
| `POST /api/v1/training/retrain/{model_name}` response | `{job_id, model_name, status, created_at}` | `{job_id, status}` |
| `GET /api/v1/training/jobs/{job_id}` | `{job_id, model_name, status, progress, created_at, completed_at}` | `{job_id, model_name, status, metrics, error}` |

**Why this matters:** Contract 2's stability guarantee says required
request/response fields cannot change without four-surface ack. Either
this is an intentional placeholder (then a `# TODO: contract compliance`
marker would be honest) or the contract needs an explicit amendment.

**Likely path:** A3 (`copilot/a3-fastapi-service-layer-sse`) is the
intended home for the real service layer. Recommend gating that PR on
bringing `/training/*` into contract shape and adding a thin contract
conformance test that asserts every route's request/response keys match
the spec.

---

### 1.3 — `app/main.py` does not run migrations at startup — **High**

**Where:** `app/main.py` (`create_app`).

**Contract reference:** `docs/contracts/storage-schema.md` — "Surface A
runs all migrations via `migrations/run_migrations.py` at startup."

The factory explicitly says: "*schema migrations are NOT run automatically
here*". It just creates an empty SQLite file. Combined with §1.1, this
means: the moment the app boots against a fresh DB, the new v4 tables
never get created — every `TrainingService.list_datasets()` call will
raise `OperationalError: no such table: fb_labels` until something
external runs the migrations.

**Fix:** Call `apply_migrations(db_path)` inside `create_app` after
ensuring the file exists. Add a startup test that boots against a fresh
tmp dir and hits `/api/v1/training/datasets` without exploding.

---

### 1.4 — Phase 0 is not yet self-sufficient — **High**

**Plan reference:** Appendix A, Phase 0 — "Nothing in Phases 1+ ships
without this."

| Deliverable | Status |
|---|---|
| Truth schema (`truth.json`) | ✅ #35 |
| Truth validator CLI | ✅ #35 |
| Metric definitions (docs) | ✅ #35 |
| **Metric implementations** (`harness/metrics/*.py`) | ❌ D2 not started |
| **`card-capture harness run --against <baseline.json>`** | ❌ |
| **Regression tab in UI** | ❌ (depends on A3) |
| **15 labeled videos** | ❌ |
| **Stable metric run across 3 consecutive executions** | ❌ |

Until Phase 0 actually closes, no algorithmic change in Phase 3+ is
verifiable. Recommend treating D2 as the highest-priority next merge
and not opening Phase 3 work until the harness CLI produces stable
numbers on at least the bootstrap set.

---

### 1.5 — `pipeline.contracts.RunContext` referenced but not implemented — **Medium**

**Where:** `docs/contracts/metaflow-artifacts.md` repeatedly references
`pipeline.contracts.RunContext`, `pipeline.contracts.DetectionPacket`, etc.
No `src/card_capture/pipeline/contracts.py` (or equivalent) exists yet.

This is expected — A2 hasn't started. Flagged here so that when A2 begins,
the first move is *creating the contracts module and pinning the
artifact-shape dataclasses* before anyone writes a `@step`. Otherwise
Metaflow steps will drift their artifact shapes during implementation
and the documented Contract 3 will be aspirational rather than enforced.

**Fix:** Make the Pydantic/dataclass module the first commit on the A2
branch, with a test that imports every artifact name listed in
Contract 3 and instantiates a minimal valid example.

---

### 1.6 — Schema duplication between docs and code, no drift gate — **Medium**

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

Either way, before more contracts get added in Wave 2, lock down the
sync mechanism.

---

### 1.7 — No CI gate on contract compliance — **Medium**

There is no CI job that runs the harness, the migration tests, or a
contract-conformance test set. The four merged PRs all included unit
tests, but nothing prevents the *next* PR from regressing them.

**Fix:** Minimal GH Actions workflow that runs `pytest tests/` on push
to any branch that touches `migrations/`, `harness/`, `app/`,
`src/card_capture/ml/`, or `docs/contracts/`. Add a contract-conformance
folder (`tests/contracts/`) that pins:
- migration DDL → Contract 1 table list + column types
- API routes → Contract 2 paths + verbs + required keys
- truth file → Contract 4 schema (already done by `tests/harness`)

---

### 1.8 — Synthetic eval datasets risk over-fitting model code to non-real distributions — **Low**

**Where:** `src/card_capture/ml/synthetic_eval.py`.

Synthetic F/B and dedup datasets are sensible for unblocking C-surface
training scaffold before real labels exist. But once the F/B classifier
is wired to real `fb_labels`, anyone running tests will keep hitting
the synthetic dataset and may forget the model never trained on real
cards. Treatment:
- Make the synthetic eval explicit ("smoke-only, not for accuracy") in
  every test that uses it.
- Add a TODO marker that fails CI if the synthetic eval is still in use
  after the real F/B trainer ships.

---

### 1.9 — `TrainingService` job execution path missing — **Low**

**Where:** `app/services/training_service.py`.

`TrainingJob` has the right shape but the actual training loop isn't
wired up — `start_retrain` queues a job, nothing pops it. Acceptable
for C0 (it's scaffolding), but worth noting so it doesn't get mistaken
for a working training pipeline if someone POSTs to `/retrain`.

**Fix:** Either (a) raise `NotImplementedError` when status would
transition `queued → running` until the real trainer lands, or
(b) add a placeholder background thread that immediately marks jobs
`failed` with a clear "training pipeline not yet implemented" message.

---

### 1.10 — D1 PR (#37) was a no-op cleanup, not the deliverable its title claims — **Low / trivial**

**Where:** Commit `39c872ff` "feat(harness): D1 — Truth Schema + Validator".

The PR only deletes `src/card_capture.egg-info/`. The actual D1 scope
(harness schema + validator) was delivered inside D0 (#35). Not a code
issue; just makes the changelog confusing for anyone scanning commit
titles later. Mention it in the v4 retrospective.

---

## 2. Resolved concerns

*(none yet — items move here with the PR/commit that closed them)*

---

## 3. Process notes

- **Concerns don't block merges by themselves.** They block the *phase
  declaration* — Phase 0 can't be called complete with §1.4 open;
  Wave 1 can't be called complete with §1.1 open.
- **When in doubt, add the concern.** A duplicate or trivial concern
  costs nothing; a missed `Blocker` costs a re-spin.
- **The `Plan` doc (`CLAUDE.md` Appendix A) is the source of truth for
  intent. The `docs/contracts/` files are the source of truth for
  shape.** When this doc disagrees with either, this doc is wrong.
