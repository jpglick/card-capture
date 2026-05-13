# v4 Wave 4 — Hardening Design

**Date:** 2026-05-13
**Status:** Approved for implementation
**Predecessors:** Waves 1–3 (contracts, surfaces, algorithmic gaps).
**Successor:** Wave 5 — algorithmic accuracy gains (gated on a frozen Phase-0
baseline).

This is the spec for Wave 4 of the v4 pipeline overhaul. The goal is to
stabilise the codebase so Wave 5 algorithmic work can land against a
green-CI, contract-conformant, single-canonical-pipeline foundation. No
new accuracy-shaped changes ship in this wave.

The implementation plan for each surface is written separately, under
`docs/superpowers/plans/2026-05-13-v4-wave4-surface-{a,b,c,d,e}.md`. This
spec is the source of truth for *scope, sequencing, and acceptance*; the
plans are the source of truth for *steps and code touch-points*.

The full open-concerns list lives in `/V4_CONCERNS.md`. Items referenced
by section number below (§1.x / §2.x / §4.x) refer to that doc.

---

## 1. Scope cut

**In scope (Wave 4):**

- All §1 open concerns *(carried over from prior review passes)*:
  §1.1 monolith/Metaflow drift, §1.2 uncommitted working tree
  *(pre-flight, complete)*, §1.3 golden-set bootstrap policy,
  §1.4 harness_config.json home, §1.5 FBPredictor refuse-without-checkpoint,
  §1.6 reid_embedding policy, §1.7 (deferred per user), §1.8 migration
  logging, §1.9 CI gate, §1.10 schema-drift gate, §1.11 mixed metric
  types, §1.12 cli config TODOs, §1.13 truth-file naming, §1.14 *(deferred)*,
  §1.15 npm install size *(documentation only — folded into PR conventions)*,
  §1.16 D1 misnamed PR *(retrospective only — no code change)*.
- A subset of §4 unimplemented plan items that are infrastructure/UI shaped:
  §4.10 user-defined config-preset persistence, §4.11 A/B comparison view,
  §4.14 Settings tab body, §4.15 Inbox drag-drop + SSE queue, §4.16 two
  competing Options dataclasses.
- A follow-up to §2.10 (contract conformance) — request-shape validation
  + "no undocumented routes" assertion.

**Out of scope (Wave 5):**

§4.1 learned quality ranker, §4.2 per-pixel background variance,
§4.3 per-region detector confidence, §4.4 content-aware F/B and dedup
similarity, §4.5 higher-res canvas + Lanczos, §4.6 YOLO26-OBB swap,
§4.7 VideoToolbox decoder, §4.8 detection-conditioned sampler,
§4.9 multi-process structured error codes, §4.12 sampler retrain path,
§4.13 BoT-SORT real ReID.

**Deferred per user direction:**

§4.17 (15 labeled videos + frozen baseline + 3-run stability test) and
all training-blocked items.

---

## 2. Sequencing

```
  ┌─────────────────────────────────────────┐
  │ Pre-flight: commit working tree (§1.2)  │  COMPLETE
  └────────────────┬────────────────────────┘
                   ↓
  ┌─────────────────────────────────────────┐
  │ Surface E — Foundations                  │  serial, single agent
  │ (CI, ADR, drift gates, PR conventions)   │
  └────────────────┬────────────────────────┘
                   ↓ (CI green on main)
  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
  │ Surface A │ │ Surface B │ │ Surface C │ │ Surface D │
  │ Pipeline  │ │ Frontend  │ │ ML        │ │ Harness   │
  └───────────┘ └───────────┘ └───────────┘ └───────────┘
                   (up to 4 agents in parallel)
```

Inter-surface blockers (declared in PR descriptions):

- Every A/B/C/D PR is **blocked-by E1** (CI workflow).
- **A2** (single canonical config dataclass) **blocks D1** (harness loads
  that dataclass).
- **A3** (migration-runner logging) **blocks B3** (config-preset
  migration — first new migration after the policy fix).
- **E2** (monolith/Metaflow ADR) **blocks A1** (execution of the
  decision).

---

## 3. Surface E — Foundations

Single agent, ~5 PRs. Gates everything else.

| # | Item | Closes | Spec |
|---|---|---|---|
| **E1** | `.github/workflows/test.yml` running `pytest tests/` on push + PR | §1.9 | Python 3.11; `pip install -e .[harness,test]`; must successfully run the existing 60+ test files. **Manual follow-up by user:** flip GitHub branch-protection to require the workflow as a status check on `main` — the agent can deliver the workflow but cannot configure repo settings. The agent's PR description must call this out. |
| **E2** | Monolith-vs-Metaflow decision ADR | §1.1 | New file `docs/decisions/2026-05-13-pipeline-canonical-path.md`. Picks one (Metaflow recommended per Appendix A.1) and lists every current entry point that still routes through the loser. Decision-only — no code in this PR. |
| **E3** | Schema/contract drift gate | §1.10 | `tests/contracts/test_drift.py` asserts every Pydantic field in `harness.schema.TruthFile` and every SQL column in `migrations/0001_v4_schema.sql` appears verbatim in the corresponding `docs/contracts/*.md`. Add a follow-up assertion for the truth-file naming convention (D3) once it lands. |
| **E4** | Contract-2 conformance hardening | §2.10 follow-up | Extend `tests/app/test_api_contract.py` to validate request bodies on POST/PUT/PATCH routes against the Pydantic input models. Add a "no undocumented routes" assertion: every non-internal route in the FastAPI app must appear in `ROUTES_REQUIRED`. |
| **E5** | Branch / PR conventions doc | new | One paragraph in `V4_CONCERNS.md` (or `docs/contributing.md`): branch name `wave4/{surface-letter}-{slug}`; PR title `[Wave 4 — Surface X] <imperative>`; PR description template linking the V4_CONCERNS section(s) closed; checklist for tests + contracts + drift gate. |

**Definition of done for Surface E:**

- `pytest tests/` runs green in CI on `main`.
- A deliberately-introduced field rename (made by the agent in a follow-up
  commit) fails the drift test, demonstrating it works.
- ADR is committed and linked from `V4_CONCERNS.md` §1.1.
- PR template is in place.

---

## 4. Surface A — Pipeline

Single agent, ~3 PRs. Blocked-by E1; A1 also blocked-by E2.

| # | Item | Closes | Spec |
|---|---|---|---|
| **A1** | Execute the E2 ADR | §1.1 | If Metaflow wins: route `card-capture process`, `harness.cli`'s pipeline-execution path, and `app/services/pipeline_runner.py` through `pipeline.card_capture_flow.CardCaptureFlow`. Mark `src/card_capture/pipeline.py:VideoProcessor` deprecated (`DeprecationWarning` on import; deletion is a Wave 5 task). Add a smoke test that runs both paths on a fixture video and asserts the same cards-extracted count + the same canonical-hash set. If monolith wins: inverse — delete `pipeline/card_capture_flow.py` + `pipeline/steps/*` and remove the metaflow dep from `pyproject.toml`. |
| **A2** | Two-Options consolidation | §4.16 | `src/card_capture/config.py` is canonical. Delete the `Options` dataclass in `src/card_capture/pipeline.py`. Update every caller. Add `tests/test_config.py` asserting exactly one `Options` / `Config` dataclass is exported from `card_capture`. |
| **A3** | Migration-runner log on skip | §1.8 | One-line `logging.warning(f"skipping {statement[:60]}… ({msg})")` in the `no such table` branch of `migrations/run_migrations.py`. Plus a test that captures the log and asserts it fires when migrations run against a fresh DB without `storage.initialize()` first. |

**Definition of done for Surface A:**

- A1's smoke test green; only one pipeline path exists (or the loser is
  formally deprecation-warning'd with a Wave 5 removal note).
- A2's "exactly one Options dataclass" test green.
- A3's log fires under the test condition.

---

## 5. Surface B — Frontend

Single agent, ~4 PRs. Blocked-by E1; B3 also blocked-by A3.

| # | Item | Closes | Spec |
|---|---|---|---|
| **B1** | Inbox page | §4.15 | `app/web/src/routes/videos/+page.svelte` becomes a drag-drop drop-zone wired to `POST /api/v1/videos`. A queue card list subscribes to `/events/{run_id}` (SSE channel from A3, already in place) and live-updates status: `pending → processing → completed | failed`. Add a Playwright test (or a documented manual smoke flow in the PR description) confirming a dropped video appears in the queue and the SSE event updates its card. |
| **B2** | Settings tab body | §4.14 | `/settings/+page.svelte` becomes a list of every preset from `GET /api/v1/config/presets` with slider/number inputs per threshold, a "Save as preset" button (paired with B3), and tooltips explaining the trade-off per knob. Reuse threshold widgets from the playground if already componentised. |
| **B3** | Config-preset persistence | §4.10 | (a) `migrations/0003_config_presets.sql` — additive migration adding a `config_presets` table. (b) `POST /api/v1/config/presets` writes to it; `GET` unions builtin + user presets and removes the `# TODO`. (c) B2's "Save as preset" button posts to it and refreshes. Document the new endpoint in `docs/contracts/v1-api.md` and the new table in `docs/contracts/storage-schema.md` so the E3 drift gate covers them. |
| **B4** | A/B comparison view | §4.11 | Build the empty `/regression/compare` route: pick run A and run B (same video, different config), call a new `POST /api/v1/regression/compare` endpoint that returns `{a, b, diff: {added, removed, reassigned}, metric_deltas: {…}}`. Side-by-side render with green/red highlighting. Document the new endpoint in Contract 2. |

**Definition of done for Surface B:**

- B1's smoke test (or documented manual flow) confirms drag-drop +
  SSE-driven status updates work end-to-end.
- B2 renders presets with working slider edits.
- B3 round-trips a user preset through DB and back to the UI; drift gate
  passes for the new table and the new route.
- B4 renders a diff for two real runs; new endpoint passes E4's
  request-shape validation.

---

## 6. Surface C — ML

Single agent, ~2 PRs. Blocked-by E1.

| # | Item | Closes | Spec |
|---|---|---|---|
| **C1** | `FBPredictor` refuses without checkpoint | §1.5 | `src/card_capture/ml/inference/fb_predict.py:__init__` raises `card_capture.ml.errors.UntrainedModelError` when `checkpoint_path` is `None` or the file doesn't exist. Add `FBPredictor.is_available(checkpoint_path: Path) -> bool` classmethod. Find every callsite (`grep -r FBPredictor src app pipeline`); wrap each in an `is_available` guard that falls back to the longest-track heuristic; log a single startup warning if the checkpoint is missing. Add `tests/ml/test_fb_predict.py`: (a) instantiation without checkpoint raises, (b) with a valid checkpoint loads + forward-passes, (c) callsites fall back cleanly when unavailable. |
| **C2** | `reid_embedding` policy | §1.6 | Recommended: extract the DINOv2 embedding call currently inline in `deduplicator.py` into a reusable function (`card_capture.ml.embeddings.compute_reid_embedding`) and call it from `pipeline/steps/store.py`'s persistence path. The column is then populated regardless of tracker backend. Document the policy in a docstring at the top of `deduplicator.py`. Add an integration test that runs a fixture video and asserts `reid_embedding IS NOT NULL` for every `card_instances` row. **Escalate to user before forging ahead if DINOv2 isn't actually callable as a standalone module** — scope grows if so. |

**Definition of done for Surface C:**

- C1's three tests green; running the pipeline without a checkpoint logs
  the warning and falls back to the heuristic cleanly.
- C2's integration test green; the embedding-population guarantee holds
  for ByteTrack and BoT-SORT alike.

---

## 7. Surface D — Harness

Single agent, ~4 PRs. Blocked-by E1; D1 also blocked-by A2.

| # | Item | Closes | Spec |
|---|---|---|---|
| **D1** | `harness/cli.py` config-loading | §1.12 | Replace both `config={}, # TODO` placeholders (lines 90 and 158) with real config snapshots. Add `harness.config.load_pipeline_config(preset_name: str) -> dict` returning the active config dict from the post-A2 canonical dataclass. `freeze_baseline` and `persist_run` both call it. Add a test asserting the persisted `config_json` is non-empty and round-trips through `json.loads`. |
| **D2** | Unify metric return types | §1.11 | New `harness.metrics.types.MetricResult` (Pydantic): `{name: str, value: float \| None, extras: dict}`. Refactor `harness/metrics/{card_recall, card_precision, side_accuracy}.py` to return it. Replace `DedupAccuracy` and `ImageQuality` dataclasses with `MetricResult` (their extra fields go into `extras`). Update `harness/runner.py` and `harness/cli.py:_compute_deltas`. Add `tests/harness/test_runner_roundtrip.py`: full `Report` JSON-serialises cleanly. |
| **D3** | Canonical truth-file naming | §1.13 | Pick `<truth_dir>/<video_id>.truth.json` (flat, sortable, deterministic). `harness/runner.py:_find_truth` requires it; the other two conventions emit a `DeprecationWarning` with the expected path and fall through *for now* (deletion is a Wave 5 task). Document in `docs/contracts/truth-schema.md` so E3's drift gate covers it. Move `golden_set/videos/IMG_5872/truth.json` → `golden_set/videos/IMG_5872.truth.json`; update `_index.txt`. |
| **D4** | Golden-set + harness_config policy | §1.3, §1.4 | (a) Move `harness_config.json` from repo root to `harness/config.example.json` (committed example) + a Pydantic schema in `harness/config.py`. Real instances live next to the DB, gitignored. (b) Reference frames are *regenerable*: gitignore `golden_set/**/reference_frames/`, document the regen command in `golden_set/README.md`, commit `scripts/generate_reference_frames.py` with a deterministic seed. |

**Definition of done for Surface D:**

- D1's round-trip test green; `freeze_baseline` writes a non-empty
  config snapshot.
- D2's runner-roundtrip test green; every metric returns `MetricResult`.
- D3: IMG_5872 still runs end-to-end through the harness; deprecation
  warning fires for the legacy naming conventions.
- D4: repo root no longer has `harness_config.json`; `git status` on a
  fresh clone shows no untracked golden-set assets.

---

## 8. Cross-cutting conventions

**Worktrees.** Each agent runs in its own git worktree via the
`superpowers:using-git-worktrees` skill. Branch convention:
`wave4/{surface-letter}-{slug}` — e.g. `wave4/e-ci-workflow`,
`wave4/a-monolith-deprecation`, `wave4/c-fbpredictor-refuse`.

**Test bar (per PR):**

1. A new test exercises the change. No exceptions — even a "config rename"
   PR adds an import assertion. UI work uses Playwright or a documented
   manual smoke flow in the PR description.
2. `pytest tests/` passes locally before push.
3. `pytest tests/` passes in CI (gated by E1).
4. Any new Contract-2 route is documented in `docs/contracts/v1-api.md`
   AND covered by `tests/app/test_api_contract.py`.
5. Any new schema field is documented in the matching `docs/contracts/*.md`
   AND covered by E3's drift gate.

**PR template** (delivered in E5):

```
[Wave 4 — Surface X] <imperative summary>

Closes V4_CONCERNS §X.Y
Blocked-by: <PR # or "none">
Blocks: <PR # or "none">

## Summary
<1-3 bullets>

## Test plan
- [ ] new test added: <name>
- [ ] pytest tests/ green locally
- [ ] CI green
- [ ] contract docs updated (if applicable)
- [ ] drift gate green (if applicable)
```

**Closing V4_CONCERNS.md entries.** Every PR description includes a
`Closes V4_CONCERNS §1.X` line. When merged, the agent's last commit on
the branch moves the entry to §2 (Resolved) with the PR number. This is
part of the PR, not a follow-up.

**Concurrency lanes (file-ownership for parallel agents):**

| Surface | Owns |
|---|---|
| A | `src/card_capture/pipeline.py`, `src/card_capture/config.py`, `pipeline/**`, `migrations/run_migrations.py` |
| B | `app/web/**`, `app/api/{videos,config,regression}.py`, `app/services/{video,playground,regression}_service.py`, `migrations/0003_*.sql` |
| C | `src/card_capture/ml/**`, `src/card_capture/deduplicator.py`, `src/card_capture/tracking/**`, `pipeline/steps/dedup.py`; *plus a single targeted edit to `pipeline/steps/store.py` for the C2 embedding hook — A owns `store.py` overall, so C rebases on A1 before opening the PR* |
| D | `harness/**`, `golden_set/**`, `tests/harness/**`, `scripts/generate_reference_frames.py` |
| E | `.github/**`, `docs/decisions/**`, `tests/contracts/**`, `tests/app/test_api_contract.py`, `docs/contributing.md` |

Two-way overlaps to watch:

- **A ↔ B** at `migrations/0003_*.sql` — B owns the migration content; A
  owns the runner. Coordinate via PR sequence (A3 merges first; B3 builds
  on it).
- **A ↔ C** at `pipeline/steps/store.py` — A may refactor it during A1;
  C adds the embedding hook. Sequence: A1 merges first; C2 rebases.
- **D ↔ E** at `docs/contracts/*` — D edits the truth-schema doc; E's
  drift gate reads it. D's PR adds the doc change; E rebases the drift
  gate to catch it.

---

## 9. Wave 5 placeholder

Wave 5 will cover §4.1 (learned quality ranker), §4.2 (per-pixel
background variance), §4.3 (per-region detector confidence), §4.4
(content-aware F/B + dedup similarity), §4.5 (higher-res canvas +
Lanczos), §4.6 (YOLO26-OBB), §4.7 (VideoToolbox decoder), §4.8
(detection-conditioned sampler), §4.9 (multi-process structured error
codes), §4.12 (sampler retrain path), §4.13 (BoT-SORT real ReID).

Wave 5 will not open until:

1. Wave 4 is fully merged (all PRs green in CI).
2. At least one Phase-0 baseline is frozen in `regression_baselines`
   against a labelled golden set. The user has deferred the labelling
   itself; once labelled videos exist, this gate closes.

Until both conditions are met, no `feat(ml,…)` PRs may merge. This
constraint exists because the Wave 2/3 retrospective identified
unmeasured algorithmic merges as the primary source of drift; Wave 5 is
where we stop repeating that.

---

## 10. Acceptance for the wave

Wave 4 is "done" when:

- All §1 entries except those tagged *[deferred per user]* appear in §2
  of `V4_CONCERNS.md` with the closing PR number.
- §4.10, §4.11, §4.14, §4.15, §4.16 are also closed.
- CI is green on `main`.
- One pipeline path is canonical; the other is either deleted or carries
  a `DeprecationWarning`.
- The drift gate fails when fed a deliberate contract violation
  (verified during E3 sign-off).
- The next dev cycle can be opened against a stable, measurable codebase.
