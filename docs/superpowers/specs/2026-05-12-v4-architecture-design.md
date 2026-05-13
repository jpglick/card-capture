# v4 Architecture Design

**Date:** 2026-05-12
**Status:** Approved for planning (writing-plans next)
**Relationship to CLAUDE.md Appendix A:** This spec is the authoritative v4 plan. It adopts Appendix A's scope and strategic position, fixes the open decisions in A.9 that are load-bearing for parallel agent work, and reorganizes execution by surface (not by phase) so four agents can begin simultaneously against a shared written contract.

---

## 1. Goals, Outcomes, Success Criteria

### 1.1 Goal

Ship an application — operator app, regression harness, decomposed pipeline — that lets the user process video, judge results, label outcomes, and improve models on a measured substrate. v4 is "the substrate that makes future algorithm work fast, safe, and provable." Not a rewrite; a surgical refactor of the orchestration layer plus a real application shell, with algorithmic upgrades shipped through a harness gate.

### 1.2 Outcomes

1. A SvelteKit + FastAPI application used daily for ingest, review, labeling, training, regression, and threshold tuning.
2. A regression harness with codified metrics, a golden labeled set (≥15 videos), and a CLI + Regression-tab gate for every algorithmic change.
3. `pipeline.py` decomposed into a Metaflow flow with per-step artifact persistence — the backbone of the threshold-tuning playground.
4. Five algorithmic upgrades shipped through the harness gate: multi-frame fusion verification, F/B classifier, DINOv2 + FAISS dedup, tracker swap, RANSAC corner refinement.
5. Apple-silicon fast paths (CoreML YOLO26 detector, VideoToolbox decoder, vImage warp), feature-detected at startup; cross-platform fallbacks preserved.
6. Active-learning loop wiring hard cases → training set with one-click promotion.

### 1.3 Success criteria

- Harness reports **zero metric regression** after Phase 2 refactor on the 15-video golden set.
- **F/B side accuracy** improves vs. the v4.1 baseline by ≥5 pp. The exact v4.1 baseline value is frozen as `baseline_v4.1` at Wave 1; subsequent runs report deltas against that pinned value.
- **Dedup accuracy** (Adjusted Rand Index on labeled clusters) improves measurably vs. pHash baseline.
- A **5-minute video can be labeled end-to-end in under 10 minutes** using only mouse + keyboard.
- **Threshold playground** in the Settings tab recomputes downstream stages from persisted artifacts — no full re-run of detector/sampler.

### 1.4 Out of scope

- Multi-user / auth.
- Cloud-hosted deployment.
- Replacing the algorithm modules wholesale (sampler, detectors, scoring, fusion, ECC, presence, tracking adapters are preserved).
- Cross-platform parity for Apple-silicon fast paths (they remain opt-in feature-detected).

### 1.5 Locked decisions (from A.9 and brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Frontend framework | **SvelteKit** | Labeling UX (drag-link, filmstrip, keyboard-first F/B trainer) is on the edge of HTMX+Alpine's comfort. SvelteKit's interaction density fits A.3.1–3. |
| Orchestration library | **Metaflow** | Artifact persistence is load-bearing for threshold playground; resume + `foreach` parallelism preserve value through later phases; local-first today, opt-in distributed later. |
| Apple-silicon fast paths | **In scope (Wave 3)** | Feature-detected at startup, cross-platform fallbacks preserved. |
| Training infrastructure | **Local-only on Apple Silicon (MPS)** | No cloud GPU dependency. Affects model sizing — see Surface C open decisions. |
| Tracker (BoT-SORT vs. ByteTrack) | **Deferred to Surface C** | Decision after Wave 2 baseline metrics; either choice eliminates the dummy-image ReID bug. |
| DINOv2 variant (ViT-S vs. ViT-B) | **Deferred to Surface C** | Pick after benchmarking both on labeled dedup groups. |

---

## 2. Surface Decomposition & Cross-Surface Contracts

Work is divided **by surface**, not by Appendix A phase. Each surface is owned end-to-end by one agent; surfaces interact only through the four contracts in §2.2.

### 2.1 The four surfaces

| Surface | Owns | Touches but does not own |
|---|---|---|
| **A. Orchestration / Pipeline** | `pipeline/` Metaflow decomposition; Stages 1–3 streaming subsystem; FastAPI service layer; SSE/WebSocket progress; storage extensions; CLI parity; Apple-silicon fast paths | Algorithm module internals (preserved); frontend |
| **B. Frontend / App Shell** | SvelteKit app; all left-nav routes (Inbox, Runs, Cards, Label, Train, Regression, Settings); labeling UX (A.3.1–3); threshold playground UI | API contracts (consumes); model artifacts (consumes) |
| **C. ML Models** | F/B classifier (MobileNetV3-S); DINOv2 + FAISS dedup; tracker swap decision + integration; RANSAC corner refinement; multi-frame fusion verification; model versioning + retrain pipeline | Storage schema (proposes additions; A owns); training UI (proposes interactions; B owns) |
| **D. Harness / Labeling** | `truth.json` schema; metric definitions; `card-capture harness run` CLI; regression baseline format; golden-set management; hard-case capture wire-up | Regression tab UI (B implements against D's API); training-data export (C consumes) |

### 2.2 Cross-surface contracts (frozen at Wave 1 sign-off)

These four contracts go into Wave 1 contract review. After all four agents ack, they freeze; later changes require four-way ack.

#### Contract 1 — Storage schema additions

Owned by A; A drafts after collecting proposals from D (truth, regression) and C (models, labels, clusters). Single SQLite database `cards.sqlite`; single-user; no migration tooling commitment beyond ad-hoc `ALTER TABLE` scripts in `migrations/`.

New tables:

- `truth_files` — `(video_id, schema_version, payload_json, updated_at)`.
- `regression_baselines` — `(baseline_id, name, code_sha, config_json, created_at)`. Tagged baselines (e.g. `baseline_v4.1`, `baseline_v4`) point to a frozen pipeline snapshot.
- `regression_runs` — `(run_id, baseline_id, code_sha, config_json, metrics_json, per_video_json, created_at)`.
- `fb_labels` — `(label_id, source_run_id, instance_id, frame_index, side, labeler, created_at)`.
- `dedup_clusters` — `(cluster_id, predicted_member_ids_json, confirmed_member_ids_json, status, updated_at)`.
- `model_versions` — `(version_id, model_name, training_set_hash, eval_metrics_json, checkpoint_path, created_at)`.
- `hard_cases` — `(case_id, run_id, frame_index, stage_id, reason, thumbnail_path, source_frame_path, created_at)`.

Extensions:

- `pipeline_events` adds `stage_id TEXT`, `artifact_ref TEXT` (Metaflow run-id + step-name + artifact-name).

#### Contract 2 — Service-layer API

Owned by A; consumed by B. REST + SSE, versioned `/api/v1/...`. Frozen at sign-off.

REST routes (verb-noun shorthand):
- **Videos:** `GET/POST /videos`, `GET/DELETE /videos/{id}`, `POST /videos/{id}/process`.
- **Runs:** `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/{cards,events,telemetry,rejection_log,hard_cases}`.
- **Cards:** `GET /cards` (paged, filtered), `GET /cards/{id}`, `PATCH /cards/{id}` (review state), `POST /cards/bulk` (bulk verdict).
- **Label:** `GET/PUT /label/truth/{video_id}`, `GET /label/fb/next`, `POST /label/fb` (single keypress label), `GET /label/clusters`, `PATCH /label/clusters/{id}`.
- **Training:** `GET /training/datasets`, `POST /training/retrain/{model_name}`, `GET /training/jobs/{id}`.
- **Regression:** `GET /regression/baselines`, `POST /regression/baselines` (promote), `POST /regression/run`, `GET /regression/runs/{id}`, `GET /regression/compare?a=&b=`.
- **Config:** `GET /config/presets`, `POST /config/presets`, `GET /config/playground/{run_id}` (artifact-backed slider data).

SSE: `GET /events/{run_id}` emits `stage_started`, `stage_progress`, `stage_completed`, `artifact_persisted`, `run_completed`, `run_failed`. Event payloads versioned with the REST surface.

#### Contract 3 — Metaflow artifact contract

Owned by A; consumed by B (threshold playground) and D (harness).

`pipeline/card_capture_flow.py` is a `FlowSpec` < 200 lines. Each logical stage is a `@step`. Stages 1–3 are wrapped as a single `detect` step preserving the existing `multiprocessing` + bounded `Queue` streaming subsystem. Stage 9 (per-track fusion) uses `foreach` for per-track parallelism.

Named artifacts persisted by step (B and D read these by name; A guarantees stability):

| Step | Artifacts |
|---|---|
| `detect` | `frame_samples`, `triaged_frames`, `corner_detections` |
| `novelty` | `novelty_filtered_candidates`, `background_model` |
| `track` | `tracks`, `session_resets` |
| `refine` | `rectified_crops` |
| `score` | `scored_candidates`, `pruned_tracks` |
| `resolve` | `prepared_tracks` (with `angle`, `session_id`) |
| `fuse` (foreach) | `fused_canonicals` per track |
| `dedup` | `dedup_groups`, `dedup_distances` |
| `store` | `final_cards` |

Each `@step` reads named artifacts from upstream steps via `self.<name>`; this is the substrate the threshold playground recomputes from.

#### Contract 4 — Truth file and metric schemas

Owned by D; consumed by B (labeling UX writes truth files; regression tab reads metrics) and C (training pipelines read labels and metrics).

`truth.json` per video — schema from Appendix A.0:

```json
{
  "video_id": "practice_session_03",
  "expected_cards": [
    {
      "card_id": "card_01",
      "front_present": true,
      "back_present": true,
      "approx_front_window_ms": [4200, 6100],
      "approx_back_window_ms": [6300, 7900],
      "physical_card_key": "charizard_base_4_holo_1999",
      "is_foil": true,
      "notes": "hand occlusion at 5.2s"
    }
  ]
}
```

Metrics (pure functions over `cards.sqlite` + `truth.json` — no pipeline coupling):

- **Card recall** = matched ground-truth cards / total ground-truth cards.
- **Card precision** = real detections / total detections (phantom rate complement).
- **Side accuracy** = correct front/back assignments / total assigned instances.
- **Dedup accuracy** = Adjusted Rand Index (or pair F1) on predicted vs. confirmed clusters.
- **Image quality** = SSIM (and PSNR) of fused canonical vs. reference frame.

Noise-floor thresholds (initial; calibrated after 3 baseline runs): recall ±1 pp, precision ±1 pp, side accuracy ±2 pp, dedup ARI ±0.02.

### 2.3 Contract sign-off process

At the start of Wave 1, agent A drafts Contracts 1–3 and agent D drafts Contract 4. Agents B and C ack (or push back) within a focused review pass. After ack, all four agents code in parallel. Any post-sign-off contract change requires explicit four-way ack.

---

## 3. Wave Plan & Gates

### 3.1 Wave 1 — Foundations (all four surfaces parallel)

| Agent | Deliverable | Acceptance |
|---|---|---|
| **A. Orchestration** | Metaflow decomposition; Stages 1–3 wrapped as single `detect` step; Stages 4–10 as individual steps; Stage 9 fusion as `foreach`. FastAPI service layer skeleton with all routes from Contract 2 (stubs OK where backed by deferred features). SSE channel emitting Contract 2 events. Storage migrations for the 7 new tables. CLI parity preserved. | Harness reports 0% delta on golden set after refactor. |
| **B. Frontend** | SvelteKit app shell; left-nav; route stubs for all sections; Inbox + Runs (with detail tabs) + Cards (grid + filters + bulk). SSE progress wiring. | Can ingest a video and watch per-stage progress live; can browse extracted cards with filters. |
| **C. ML** | Training-loop scaffolding on MPS (deterministic seeds, eval glue). Model-version registry tables populated by training pipeline. Synthetic eval harness for tracker + dedup until labels exist. | Scaffolding accepts a labeled dataset → produces a versioned checkpoint with eval metrics. |
| **D. Harness / Labeling** | `truth.json` schema + validator. Metric implementations. `card-capture harness run` CLI. Regression API (`/regression/*`) consumed by B. Labeling-UX backend endpoints (`/label/*`). Hard-case capture wire-up. Bootstrap-label 5 videos. | Harness runs against `baseline_v4.1` and produces stable metric reports across 3 consecutive runs (≤ noise floor variance). |

**Wave 1 gate:** all four acceptance criteria green; `baseline_v4.1` frozen as a `regression_baselines` row; all four contracts ack'd and committed.

### 3.2 Wave 2 — Features (unlocked by Wave 1 + ≥10 labeled videos)

**Front 1 — Labeling UX (B + D, parallel):** per-video truth editor (A.3.1), F/B trainer (A.3.2), dedup cluster confirmer (A.3.3). Drives label production. Auto-save every 30s. Hotkey-first throughout.

**Front 2 — Algorithmic upgrades (C, gated by harness):** ship in Appendix A.3 priority order, each as its own PR:

1. **Multi-frame median fusion verification** — confirm enabled; sweep frame-count; consider residual-region inpainting.
2. **F/B classifier (MobileNetV3-S)** — finetuned on rectified crops from `fb_labels`; confidence < 0.6 falls back to longest-track heuristic.
3. **DINOv2 + FAISS dedup** — variant chosen by C after benchmarking; cosine threshold calibrated on labeled `dedup_clusters`. Replaces pHash for cross-track / inter-instance matching (today's `_SAME_CARD_HAMMING_MAX = 22` Front/Back gate). Cheap pHash retained for within-session near-duplicate filtering (today's `_SESSION_DUPLICATE_HAMMING_MAX = 6`).
4. **Tracker swap** — BoT-SORT-with-real-ReID *or* ByteTrack-no-ReID; decision by C with regression evidence.
5. **RANSAC corner refinement** — sub-pixel corner refinement on canonical frames before rectification.

Each Front 2 PR merges only if the harness shows neutral-or-positive metrics with no per-video regression outside noise floor.

**Wave 2 gate:** F/B and dedup upgrades shipped through harness with measurable improvement; ≥15 labeled videos; tracker decision made with regression evidence.

### 3.3 Wave 3 — Speed, threshold playground, active learning

- **A:** Apple-silicon fast paths (CoreML YOLO26-OBB, VideoToolbox decoder, vImage warp) — feature-detected; cross-platform fallbacks preserved.
- **B:** Threshold-playground UI (A.5.3) reading persisted Metaflow artifacts; A/B comparison view (A.5.2); promote-to-baseline button gated on no-regression.
- **C + D:** Active-learning loop — Hard Cases tab surfaces auto-captured edges; one-click "send to training set"; retrain pipeline runs locally; new model version added to a regression run.

**Wave 3 gate:** Apple paths show throughput gain with no quality regression; threshold playground works on a persisted run; one full retrain → harness → promote-baseline cycle completed end-to-end.

### 3.4 Hard rules across all waves

1. **No algorithmic PR merges without harness evidence.** Every Wave 2/3 algorithmic change links to a harness run with per-metric deltas.
2. **No contract changes without four-surface ack.** Storage schema, API surface, Metaflow artifact names, truth-file schema — frozen at Wave 1 sign-off.
3. **Labeling bandwidth is the critical-path constraint.** Wave 2 Front 2 is bottlenecked by label flow; the spec calls this out so plans do not pretend otherwise.
4. **Behavior preservation through the Metaflow refactor.** Phase 2 (Wave 1 surface-A deliverable) is gated by 0% metric delta — the only safe way to refactor a tangled orchestration layer mid-flight.

---

## 4. Per-Surface Scope Detail

### 4.1 Surface A — Orchestration / Pipeline

**Wave 1.**
- `pipeline/card_capture_flow.py` — top-level FlowSpec < 200 lines.
- `pipeline/steps/{detect,novelty,track,refine,score,resolve,fuse,dedup,store}.py` — one module per step.
- `detect` step wraps existing producer/consumer multiprocessing for Stages 1–3 as a single unit; Metaflow sees the wrapped subsystem as one step.
- `fuse` step uses Metaflow `foreach` for per-track parallelism.
- FastAPI service layer in `app/api/` and `app/services/` wrapping `Storage` + a new `PipelineRunner` service. The existing CLI (`cli.py`) calls the same `app/services/*` — no duplicated orchestration code.
- SSE channel emits Contract 2 events.
- Storage migrations under `migrations/` (ad-hoc SQL acceptable for single-user).

**Wave 3.**
- Apple-silicon detection (`platform.system() == "Darwin"` + ANE-availability probe) at startup.
- CoreML YOLO26-OBB path with PyTorch fallback.
- VideoToolbox decoder path with OpenCV/decord fallback.
- vImage perspective warp path with Kornia fallback.
- Fast paths covered by harness on macOS CI; cross-platform fallback path covered on Linux CI (or by a forced-fallback env var on macOS if Linux CI is impractical).

**Open decisions left to A.**
- Exact adapter shape for the Stages 1–3 streaming wrap (likely a thin façade that exposes a single `run(video_path) -> CornerDetections` interface).
- Whether to introduce `alembic` for schema migrations now or stay with ad-hoc SQL through v4.
- Whether per-stage retry/backoff lives in Metaflow's `@retry` or in the existing producer/consumer.

### 4.2 Surface B — Frontend / App Shell

**Wave 1.**
- SvelteKit project under `app/web/` (served by FastAPI in prod, vite dev server in dev).
- Left-nav routing per Appendix A.4.
- Inbox: drag-drop video upload, queue with status, "Run pipeline" button with config-preset selector.
- Runs: list view + detail page with tabs (Timeline, Cards, Telemetry, Events, Rejection Log, Hard Cases).
- Cards: grid view with filters (run, video, dedup-group, review-state, side, is_foil, confidence range); bulk-actions; per-card detail.
- Settings: read-only config-preset view in Wave 1.
- SSE consumer wired to per-stage progress events.

**Wave 2.**
- Label section with three sub-tabs:
  - Per-video truth editor (A.3.1): filmstrip, three-button verdict, F/B/X hotkeys, drag-link, scrubber with detected-card markers.
  - F/B trainer (A.3.2): single-card flash-card mode, single-keypress label and advance.
  - Dedup cluster confirmer (A.3.3): cluster grid, multi-select, split/merge.
- Auto-save every 30s; backward-compatible with current `truth.json` schema.

**Wave 3.**
- Threshold playground (A.5.3): slider → `/config/playground/{run_id}` → live metric + thumbnail-strip recompute from persisted artifacts.
- A/B comparison view (A.5.2): per-run diff highlighting + metric-delta strip.
- Promote-to-baseline button gated on no-regression.

**Open decisions left to B.**
- Component library (Skeleton UI vs. shadcn-svelte vs. minimal hand-rolled).
- State management (Svelte 5 runes vs. classic stores).
- Drag-link library (svelte-dnd-action vs. native HTML5 drag).
- How aggressive optimistic updates are in the label UX (server-of-truth latency vs. perceived snappiness).

### 4.3 Surface C — ML Models

**Wave 1.**
- Training-loop scaffolding on MPS — deterministic seeds, eval glue, checkpoint serialization, `model_versions` registry writes.
- Synthetic eval harness for tracker + dedup (using rendered cards + augmentation) so C can iterate before real labels exist.
- Initial dataset loaders for `fb_labels` and `dedup_clusters` (empty-but-correct at this stage).

**Wave 2 (in Front 2 priority order).**
1. **Multi-frame fusion verification** — confirm enabled, sweep `_CANONICAL_TARGET_FRAMES`, consider residual-region inpainting on glare patches.
2. **F/B classifier** — MobileNetV3-Small finetuned on 750×1050 rectified crops; output: `front | back | uncertain`; `uncertain` defined by `confidence < 0.6` → fall back to longest-track heuristic.
3. **DINOv2 + FAISS dedup** — variant chosen after benchmarking ViT-S/14 vs. ViT-B/14 on labeled clusters; in-process FAISS index; cosine threshold calibrated against `dedup_clusters` (ARI optimum). Replaces pHash for cross-track / inter-instance matching; pHash retained for within-session near-duplicate filtering.
4. **Tracker swap** — decision between BoT-SORT-with-real-image-ReID and ByteTrack-no-ReID based on Wave 2 baseline ID-switch + session-fragmentation metrics. Whichever ships, dummy-image ReID bug is eliminated.
5. **RANSAC corner refinement** — line-fit corners on canonical frames before rectification.

**Wave 3.**
- Active-learning retrain pipeline triggered by `POST /training/retrain/{model_name}`; surfaces in `/training/jobs/{id}` SSE.
- Validation set previews (model-wrong inspection in B's Train tab).

**Open decisions left to C.**
- Tracker (BoT-SORT-fixed vs. ByteTrack). Defer per §1.5.
- DINOv2 variant (ViT-S/14 vs. ViT-B/14). Defer per §1.5.
- Whether F/B classifier uses rectified crops or raw frame crops.
- Augmentation strategy (foil-aware crops? glare jitter? rotation?).
- All training is local Apple Silicon (locked); model sizing must fit unified memory.

### 4.4 Surface D — Harness / Labeling Infrastructure

**Wave 1.**
- `truth.json` JSON Schema + validator.
- Metric implementations as pure functions over `(cards.sqlite, truth.json)`. No pipeline coupling.
- `card-capture harness run --baseline <id> [--videos <subset>]` CLI emits regression report JSON.
- Regression API: `/regression/{baselines,runs,compare}`. Reports stored in `regression_runs`; baselines in `regression_baselines`.
- Hard-case capture wire-up: existing `analysis/hard_case_capture.py` writes to `hard_cases`.
- Bootstrap-label 5 videos with backward-compatible `truth.json` (current `templates/labeling.html` continues to work during the transition).

**Wave 2.**
- Backend endpoints for label-UX writes (`PUT /label/truth/{video_id}`, `POST /label/fb`, `PATCH /label/clusters/{id}`).
- Golden-set growth to ≥15 videos (Wave 2 gate).
- Noise-floor recalibration after 3 baseline runs.

**Wave 3.**
- Training-set export endpoints feeding C's retrain pipeline.
- Hard-case promotion endpoint (`POST /training/datasets/from_hard_cases`).

**Open decisions left to D.**
- Noise-floor thresholds per metric (initial guesses in Contract 4; recalibrate empirically).
- Whether the SSIM reference frame is human-picked once (per ground-truth card) or pipeline-picked highest-quality.
- Whether harness CLI shells out to Metaflow `run` or invokes the flow programmatically (Metaflow supports both).

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Labeling bandwidth is human-bound.** Wave 2 Front 2 stalls if labels do not flow. | Bootstrap-label 5 videos in Wave 1. F/B trainer is single-keypress so 500 labels ≈ 1 hour. Front 1 ships in parallel with Front 2 specifically to keep labels flowing. |
| **Metaflow learning curve combined with 0% delta gate.** | Wave 1 agent A's entire deliverable is the refactor (no algorithm changes in the same PR). Harness gate is automated. |
| **Cross-surface contract drift.** | Contracts frozen at Wave 1 sign-off; post-sign-off changes require four-surface ack; contracts are small (4 of them, ~1 page each). |
| **Threshold playground depends on Metaflow artifact stability.** | Artifact names locked in Contract 3. A is responsible for stability; named-artifact tests in CI. |
| **Apple-silicon fast paths break cross-platform.** | Feature-detect at startup; harness covers both code paths on macOS; cross-platform fallback path covered on Linux CI. |
| **Regression noise from pipeline non-determinism.** | Harness acceptance criterion at Wave 1 is "stable across 3 consecutive runs." This forces determinism work as part of Phase 0 before anything lands. |
| **Frontend agent blocks on backend API.** | A ships service-layer stubs in Wave 1 for all Contract 2 routes; B builds against the stubs while A fills them in. |
| **C blocks on labels for F/B and dedup.** | Synthetic eval harness lets C iterate model architecture and training loop before real labels arrive. Real labels gate the *integration*, not the *scaffolding*. |

---

## 6. Implementation Plan Handoff

This spec hands off to `writing-plans` to produce four implementation plans — one per surface — each with its own task graph, dispatchable to an agent in isolation. The four plans share §2.2 contracts verbatim and reference this spec as their authority.

Order of plan production: A (orchestration) and D (harness) first because they own contracts; then B and C, which consume the contracts.

---

## Appendix — Glossary

- **Surface**: a coherent vertical of work owned end-to-end by one agent (Orchestration/Pipeline, Frontend, ML, Harness/Labeling).
- **Wave**: a parallel execution band with a hard gate at its end. Wave 1 = foundations; Wave 2 = features; Wave 3 = speed + active learning + playground.
- **Contract**: one of four written agreements between surfaces, frozen at Wave 1 sign-off (storage schema, service-layer API, Metaflow artifact contract, truth-file/metric schemas).
- **Harness gate**: a regression-harness run on the 15-video golden set showing neutral-or-positive metric deltas with no per-video regression outside the noise floor; required for every Wave 2/3 algorithmic merge.
- **Threshold playground**: A.5.3 UI that recomputes downstream pipeline stages from persisted Metaflow artifacts when a config threshold slider moves — no detector/sampler re-run.
- **Golden set**: the ≥15 manually labeled videos used by the harness for regression measurement.
