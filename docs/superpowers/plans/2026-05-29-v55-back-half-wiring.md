# V5.5 Back-Half Wiring Plan

**Status:** PLAN (no code)
**Date:** 2026-05-29
**Target branch:** `fix/ui-v55-unified-runtime` (extends PR #60)
**Author:** Claude (per josh)

---

## 1. Problem statement

The V5.5 refactor (`dc6faeb0` → squash-merged as `ba2c32c2 #59`) moved the
pipeline's architectural boundaries (DAL, platform adapters, import-linter
contracts, Metaflow removal, GpuSession) but left the **back half of the
stage bodies as TODOs**. Verified directly on `origin/main`:

| Stage | Current state | Real backing class/fn | Status |
|---|---|---|---|
| `sample`  | wired to `StrideSampler` | — | ✅ |
| `detect`  | wired to `CardcaptorUltralyticsDetector` | — | ✅ |
| `novelty` | wired to `BackgroundModel` + `quad_novelty` | — | ✅ |
| `track`   | wired to `ByteTrack`/`BoT-SORT` `.assign()` | — | ✅ |
| `refine`  | calls `KorniaNormalizer.warp_canonical_batch` only; **drops `(track_id, frame_index, score)` identity** and skips Laplacian scan / corner refinement / scoring / dedup hash / glare / ReID | `KorniaNormalizer`, `QualityScorer`, `VisualDeduplicator`, `find_glare_centroid`, `quad_novelty`, `_laplacian_select_frames`, `_select_canonical_entries`, `DinoEmbedder` | ⚠️ partial |
| `score`   | `state["scored"] = state["tracks"]` (no-op) | `QualityScorer`, novelty-gap pruning, conf-floor, stand-gate | ❌ stub |
| `resolve` | passthrough | `embedding_same_card_score`, `AdaptiveThresholdComputer`, `FBPredictor`, `_side_textiness_score`, `_appearance_vector`, hard-case capture | ❌ stub |
| `fuse`    | `[{"track_id": ..., "fused_canonical": None}]` | `MultiFrameFuser.fuse(images, foil_threshold)` | ❌ stub |
| `dedup`   | `state["final_cards"] = state.get("fused", [])` | `VisualDeduplicator`, `DinoDeduplicator`, cross-video DB query | ❌ stub |
| `store`   | hardcoded `final_cards = []` | `Storage.add_card_instance` / `update_instance_*` / `add_card_view` | ❌ stub |

Result: starting a run from the UI succeeds at the orchestration layer and
streams stage events (PR #60 fixes that), but **zero cards are persisted**.

---

## 2. Sizing

V4 step LOC living unmodified in `.worktrees/ci-fixes/pipeline/steps/`:

```
track.py     296   (already ported to stages/track.py — uses .assign())
refine.py    408   (~10% ported to stages/refine.py; rest missing)
score.py     180   (0% ported)
resolve.py   234   (0% ported)
fuse.py      121   (0% ported)
dedup.py     127   (0% ported)
store.py     155   (0% ported)
---
total      1,521 LOC step code
```

Plus helpers already present in `src/`:

- `card_capture.scoring.QualityScorer.score(image, conf, prior_frames=None, novelty=1.0) -> QualityScore` ✓
- `card_capture.fuser.MultiFrameFuser.fuse(images: List[np.ndarray], foil_threshold=None) -> np.ndarray` ✓ (accepts ndarrays directly — no path adaptation needed)
- `card_capture.deduplicator.VisualDeduplicator.compute_phash(image: np.ndarray) -> str` ✓
- `card_capture.fusion.foil_detection.detect_foil_card(frames: list[np.ndarray], threshold) -> bool` ✓
- `card_capture.fusion.median_fusion.glare_rejection_fusion(frames: list[np.ndarray]) -> np.ndarray` ✓
- `card_capture.selector._select_canonical_entries(frame_entries, deduplicator) -> list[dict]` (actually lives in `pipeline_utils.py:195`) ✓
- `card_capture.pipeline_utils._side_textiness_score`, `_appearance_vector`, `_glare_mask`, `_laplacian_heatmap`, `_compress_array`, `_laplacian_select_frames`, `decode_frames_gpu`, `_compute_laplacian_scan_indices` ✓
- `card_capture.identity.embedding_distance.embedding_same_card_score` ✓
- `card_capture.calibration.per_video_adaptive.AdaptiveThresholdComputer` ✓
- `card_capture.analysis.hard_case_capture.is_hard_case`, `capture_hard_case` ✓
- `card_capture.ml.inference.fb_predict.FBPredictor`, `card_capture.ml.registry.get_latest` ✓
- `card_capture.ml.models.dino_embedder.DinoEmbedder` ✓
- `card_capture.ml.inference.dino_dedup.DinoDeduplicator` ✓
- `card_capture.ml.embeddings.compute_reid_embedding` ✓
- `card_capture.storage.Storage.add_card_instance`, `update_instance_deduplication`, `update_instance_fusion`, `add_card_view`, `add_saved_card`, `add_track_telemetry`, `add_pipeline_event` ✓ (but `store` stage should go through `CardsRepository` per V5.5 mandate — see §5)

**Good news:** the helpers already accept the right input shapes (np.ndarray, not file paths). No helper-class API changes needed.

---

## 3. State contract (proposed)

Each stage mutates `state: dict` in place. New keys this plan adds:

```python
# After refine (new — currently dropped):
state["refined_tracks"] = [
    {
        "instance_id": str,                     # UUID string (TrackState.instance_id)
        "track_id": int,                        # tracker numeric id
        "angle": str,                            # "Unknown" initially; set in resolve
        "session_id": int,
        "first_frame_index": int,
        "frame_entries": [
            {
                "candidate": ScoredCandidate,    # carries detection_id, corners, frame_index, timestamp_ms
                "normalized": np.ndarray,        # (1050, 750, 3) uint8 BGR — IN-MEMORY (v5.5 mandate)
                "quality_score": QualityScore,
                "visual_hash": str,              # pHash
                "glare_x": float | None,
                "glare_y": float | None,
                "sharpness": float,
                "glare_mask": bytes,             # compressed
                "laplacian_heatmap": bytes,      # compressed
                "is_canonical": bool,            # set by _select_canonical_entries
                "novelty_score": float,
                "confidence": float,
                "corners": List[Tuple[float,float]],
                "width": int,                    # source frame width
                "height": int,                   # source frame height
                "timestamp_ms": int,
                "triage_metrics": dict,
                "detection_id": int,
                "frame_index": int,
            }, ...
        ],
        "canonical_detection_ids": List[int],
        "best_canonical_detection_id": int,
        "best_canonical_image": np.ndarray,      # in-memory (was best_canonical_image_path in V4)
        "reid_embedding": List[float] | None,
    }, ...
]

# After score:
state["scored_tracks"] = [...]               # refined_tracks + {pruned, median_novelty, median_quality, median_sharpness}
state["pruned_instance_ids"] = List[str]

# After resolve:
state["prepared_tracks"] = [...]             # scored_tracks (unpruned) with side_score, appearance_vector, angle, duplicate_track_index

# After fuse:
state["fused_canonicals"] = [
    {
        "instance_id": str,
        "session_id": int,
        "angle": str,
        "fused_image": np.ndarray,           # in-memory (was fused_image_path)
        "primary_hash": str,
        "quality_score": float,
        "side_score": float,
        "appearance_vector": List[float],
        "best_canonical_detection_id": int,
        "duplicate_track_index": int | None,
        "first_frame_index": int,
        "reid_embedding": List[float] | None,
    }, ...
]

# After dedup:
state["dedup_groups"] = [
    {
        "canonical_instance_id": str,
        "duplicate_instance_ids": List[str],
        "hamming_distances": Dict[str, float],
        "embedding_distances": Dict[str, float],
        "cross_video_parent_id": int | None,
    }, ...
]

# After store:
state["final_cards"] = List[CardRecord]      # populated from fused_canonicals; persisted via repos
```

### State-shape changes vs. V4

1. **`normalized` is np.ndarray, not `image_path`.** V4 spilled rectified
   crops to `crops_dir/` then re-read them. V5.5 keeps them in RAM.
   Implication: peak memory grows by `8 candidates × N_tracks × 750·1050·3 B ≈ 18 MB/track`.
   For the reference video (~10 tracks) this is ~180 MB. Acceptable.
2. **`best_canonical_image` / `fused_image` are np.ndarray, not paths.** Same
   reasoning. The `store` stage is now the only IO boundary (writes
   `crops/*.jpg` and DB rows at the end).
3. **`ScoredCandidate.image_path` is empty.** V4 used this to re-read the
   source frame; V5.5 reads from `state["sampled_frames"]` by `frame_index`.
   `image_path` stays in the dataclass for back-compat but is `""` everywhere.
4. **`reid_embedding` is computed in refine** (matches V4) but from the
   in-memory `best_canonical_image` via a new `DinoEmbedder.embed_array`
   convenience (or by writing+reading a temp file if we want to defer that
   adapter — see §6).

---

## 4. Required upstream changes

### 4.1 `refine.py` — full rewrite (45 LOC → ~250 LOC)

Port V4 `refine.py` lines 64–408 verbatim with three substitutions:

- `decoded_images[frame_index]` ← `state["sampled_frames"]` lookup by index (in-memory; no `decode_frames_gpu` call).
- `_lap_results` ← can still call `_laplacian_select_frames(video_path, ranges, decoded_frames=...)` because that helper already accepts an in-memory frame dict.
- Drop `cv2.imwrite` calls inside the per-candidate loop. Store the np.ndarray under `entry["normalized"]`. The V4 `frame_entry_paths.append({...,"image_path": str(img_path)})` becomes `frame_entry_paths.append({..., "normalized": entry["normalized"]})`.
- Replace `embedder.embed_image(best_image_path)` with either:
  - **Option A (preferred):** add `DinoEmbedder.embed_array(np.ndarray)` (5-line wrapper around the existing transform pipeline).
  - **Option B (fallback):** `cv2.imwrite` to a tempfile, embed, delete. Easier port, costs a syscall per track.

### 4.2 `track.py` — small change

Track stage currently returns `List[TrackState]` via `.assign()`, but V4 refine consumes `track_out.tracks_data: List[Dict[str, Any]]` with a `candidates` key holding per-frame dicts that include `score_total`, `confidence`, `width`, `height`, `corners`, `frame_index`, `timestamp_ms`, `detection_id`.

Need to convert `TrackState` → V4-shape dict in `track.py` (or in refine.py at the start). Conversion is mechanical — `TrackState.candidates` is already a list of `ScoredCandidate`; build dicts from those.

### 4.3 `score.py` — full port (8 LOC → ~120 LOC)

V4 `score.py:run` ported as-is, substituting:
- `refine_out.refined_tracks` → `state["refined_tracks"]`
- `ctx.novelty_floor` → `state["request"].config.get("novelty_floor", 0.30)`
- `ctx.track_confidence_floor` → `state["request"].config.get("track_confidence_floor", 0.60)`
- `ctx.stand_novelty_max` / `ctx.stand_sharpness_max` → likewise from config with V4 defaults

Returns `state["scored_tracks"]` and `state["pruned_instance_ids"]`.

### 4.4 `resolve.py` — full port (8 LOC → ~180 LOC)

V4 `resolve.py:run` ported, substituting:
- `score_out.scored_tracks` → `state["scored_tracks"]`
- `ctx.use_fb_classifier` → config flag (default True, gracefully degrades if classifier missing — V4 already does this)
- `ctx.db_path` → `state["db_path"]`
- `ctx.observed_intra_track_distances` → must be threaded from earlier stages; for now pass empty list (V4 collects this during refine; can replicate but adds noise)
- `cv2.imread(t["best_canonical_image_path"])` → `t["best_canonical_image"]` directly (already np.ndarray)

Sets `t["side_score"]`, `t["appearance_vector"]`, `t["angle"]`, `t["duplicate_track_index"]` on each track in `state["prepared_tracks"]`.

### 4.5 `fuse.py` — full port (15 LOC → ~70 LOC)

V4 `fuse.py:run` ran **once per track** (Metaflow `foreach`). In v5.5 it
runs as a plain `for` loop over `state["prepared_tracks"]`. Substitutions:
- `cv2.imread(fe["image_path"])` → `fe["normalized"]` (already np.ndarray)
- `cv2.imwrite(fused_path, fused_img)` → set `fused["fused_image"] = fused_img`
- `shutil.copy(best_path, fused_path)` (single-frame path) → `fused["fused_image"] = prepared_track["best_canonical_image"]`

Sets `state["fused_canonicals"]: List[dict]`.

### 4.6 `dedup.py` — full port (8 LOC → ~110 LOC)

V4 `dedup.py:run` ported as-is. Substitutions:
- `fused_canonicals` arg → `state["fused_canonicals"]`
- `Storage(Path(ctx.db_path))._connect()` → switch to `CardsRepository` for the cross-video query (per V5.5 mandate; needs a new `find_by_embedding_excluding_video` method on the repo, or a raw read connection — see §5).

Sets `state["dedup_groups"]: List[dict]`.

### 4.7 `store.py` — full port + DAL migration (18 LOC → ~140 LOC)

V4 `store.py:run` ported, **but** all direct `Storage` calls converted to
`CardsRepository` calls (V5.5 mandate: no raw SQL outside `card_capture.data`).
This requires adding repository methods:

```python
class CardsRepository:
    def add_card_instance(self, video_id, track_id, angle, session_id,
                          reid_embedding, run_id) -> int: ...
    def update_instance_deduplication(self, row_id, primary_hash,
                                       cross_video_parent, reid_embedding=None) -> None: ...
    def update_instance_fusion(self, row_id, fused_image_path) -> None: ...
    def add_card_view(self, card_instance_id, frame_index, timestamp_ms,
                      detection, rectified_path, quality_score, is_canonical,
                      glare_x, glare_y, sharpness, initial_confidence) -> int: ...
    def add_saved_card(self, detection_id, image_path, final_score) -> None: ...
    def find_embeddings_excluding_video(self, video_id) -> List[Tuple[int, bytes]]: ...
```

Each method wraps an existing `storage.py` SQL statement; the work is mechanical and matches the phase E "migrate SQL literals to data-layer" pattern already established in `9d2b0edd`/`5a1319fd`.

**Image-write boundary lives here** (and only here):

```python
crops_dir = state["output_root"] / "crops"
crops_dir.mkdir(parents=True, exist_ok=True)
for fused in state["fused_canonicals"]:
    path = crops_dir / f"instance_{fused['instance_id'][:8]}_fused.jpg"
    cv2.imwrite(str(path), fused["fused_image"])
    fused["fused_image_path"] = str(path)
for track in state["prepared_tracks"]:
    for fe in track["frame_entries"]:
        view_path = crops_dir / f"track_{track['instance_id'][:8]}_det_{fe['detection_id']}_rectified.jpg"
        cv2.imwrite(str(view_path), fe["normalized"])
        fe["image_path"] = str(view_path)
```

`runs_repo.mark_completed(run_id, cards_extracted=len(state["final_cards"]))` stays as the final call.

### 4.8 `PipelineRunRequest.config` keys

Currently empty / ad-hoc. Codify the keys this back-half consumes:

```
detector: "fake" | "docaligner" | "cuda"
device: "auto" | "cpu" | "mps" | "cuda"
corner_confidence: float = 0.5
detection_width: int = 640
tracker_backend: "bytetrack" | "botsort"
min_track_length: int = 3
fusion_target_frames: int = 1
foil_threshold: float | None = 50.0      # NEW; passed to MultiFrameFuser
enable_foil_aware_fusion: bool = True     # NEW
rotate_180: bool = False
use_kornia: bool = True
use_fb_classifier: bool = True            # NEW
novelty_floor: float = 0.30               # NEW
track_confidence_floor: float = 0.60      # NEW
stand_novelty_max: float = 0.35           # NEW
stand_sharpness_max: float = 0.30         # NEW
laplacian_scan_stride: int = 5            # NEW
max_corner_gap_frames: int = 30           # NEW
corner_refinement: bool = False           # NEW
kornia_device: str = "auto"               # NEW
```

These come from `PipelineConfig` defaults today; the pipeline_runner should
read its `card_capture_config.json` and merge into `request.config` before
calling the runtime. CLI already loads `PipelineConfig` via `load_config()`;
just need a `dataclasses.asdict(config)` and pass it through.

---

## 5. DAL migration scope inside store stage

Per V5.5 mandate ("all writes through `data/`"), `store.py` cannot call
`Storage` directly. Required repository additions (rough effort estimate):

| Method | Backing SQL (already in storage.py) | Effort |
|---|---|---|
| `add_card_instance` | `Storage.add_card_instance` | trivial wrap |
| `update_instance_deduplication` | `Storage.update_instance_deduplication` | trivial wrap |
| `update_instance_fusion` | `Storage.update_instance_fusion` | trivial wrap |
| `add_card_view` | `Storage.add_card_view` | trivial wrap |
| `add_saved_card` | `Storage.add_saved_card` | trivial wrap |
| `add_track_telemetry` | `Storage.add_track_telemetry` | trivial wrap |
| `add_pipeline_event` | `Storage.add_pipeline_event` | trivial wrap |
| `find_embeddings_excluding_video` | raw `SELECT id, reid_embedding FROM card_instances WHERE …` | needs new SQL constant in `sql_queries.py` |

All 8 methods follow the existing pattern in `card_capture/data/repositories/cards.py` and `runs.py`. Total ~150 LOC of repo additions + ~8 entries in `sql_queries.py`. Tests follow the existing `tests/data/test_cards_repository.py` pattern.

---

## 6. Helper-class adapters needed

| Helper | Current accepts | Needed | Effort |
|---|---|---|---|
| `MultiFrameFuser.fuse` | `List[np.ndarray]` | ✅ already correct | none |
| `VisualDeduplicator.compute_phash` | `np.ndarray` | ✅ already correct | none |
| `detect_foil_card` | `list[np.ndarray]` | ✅ already correct | none |
| `glare_rejection_fusion` | `list[np.ndarray]` | ✅ already correct | none |
| `QualityScorer.score` | `np.ndarray` | ✅ already correct | none |
| `DinoEmbedder.embed_image` | file path | needs `embed_array(np.ndarray)` companion | ~10 LOC |
| `DinoDeduplicator` | file paths | optional — V4 dedup only uses this if available; can keep file-path call by writing+reading in store stage | none if we accept the temp-file cost |
| `FBPredictor.predict` | file path | needs `predict_array(np.ndarray)` companion | ~10 LOC |
| `compute_reid_embedding` | file path | needs array variant | ~10 LOC |

Total adapter work: **~30 LOC** if we add three small array variants, **0 LOC** if we accept temp-file IO in `store` (which already does file IO for the canonical writes — adding two extra writes for embedding inputs is a minor cost).

**Recommendation:** add the three `_array` companions (DinoEmbedder, FBPredictor, compute_reid_embedding). They're small, they're called per-track (not per-frame), and they avoid disk round-trips. Cleaner than threading temp files.

---

## 7. End-to-end test plan

Currently no fixture exercises the back half. Add `tests/pipeline/test_back_half_e2e.py`:

1. **Synthetic fixture (CI-friendly).** Generate a 5-second 480p MOV with 2 simulated cards held in front of a checkerboard background (using `numpy` + `cv2.VideoWriter`). Use the `fake` detector that emits hardcoded corners around the simulated cards. Run the full pipeline (`LocalPipelineRuntime.run()`); assert:
   - `manifest.cards` is non-empty (≥ 1 card).
   - `cards.sqlite` has rows in `card_instances` keyed by run_id.
   - At least one `crops/*.jpg` exists with expected dimensions (1050×750).
   - `runs_repo.get(run_id)["cards_extracted"] > 0`.
2. **Golden-set regression (manual / opt-in).** Re-run `IMG_5872.MOV` (the existing baseline video) and compare cards_extracted / card_recall / card_precision / side_accuracy / image_quality(SSIM) against the V4 baseline at `docs/superpowers/plans/v5-5/baseline-results.md`. Tolerance: ±2 cards, ±0.05 on recall/precision, ±0.05 on SSIM.
3. **Smoke test upgrade.** Extend `tests/test_unified_runtime.py` (currently asserts stage telemetry fires) to also assert `len(result.manifest.cards) > 0` when run with the fake detector + synthetic fixture.

The synthetic fixture should land alongside the wiring as `tests/fixtures/synthetic_two_cards.MOV` (generated by `tests/conftest.py` once and cached — no binary in repo).

---

## 8. Phased commit sequence (proposed)

Each commit must leave the test suite green; each phase is one PR commit on `feat/v55-back-half-wiring` (branched from `fix/ui-v55-unified-runtime` so PR #60's runner/CLI/runtime changes are inherited).

| # | Commit | Adds | Removes from "stub" column |
|---|---|---|---|
| P1 | feat(v55-stages): codify config keys + thread through pipeline_runner | `request.config` dict shape, config merging from `PipelineConfig` | — |
| P2 | feat(v55-stages): add CardsRepository write methods + SQL constants | 8 repo methods, 8 sql_queries entries, repo unit tests | enables store stage port |
| P3 | feat(v55-stages): add array-variant helpers (Dino/FB/reid) | 3 `_array` companions + unit tests | enables refine/resolve/store array path |
| P4 | feat(v55-stages): port track → V4-dict shape + refine to full V4 logic | full refine port; `state["refined_tracks"]` shape | refine ✅ |
| P5 | feat(v55-stages): port score stage (pruning gates) | full score port; `state["scored_tracks"]` + `pruned_instance_ids` | score ✅ |
| P6 | feat(v55-stages): port resolve stage (F/B + duplicate sessions) | full resolve port; `state["prepared_tracks"]` | resolve ✅ |
| P7 | feat(v55-stages): port fuse stage (MultiFrameFuser + foil) | full fuse port; `state["fused_canonicals"]` | fuse ✅ |
| P8 | feat(v55-stages): port dedup stage (pHash + Dino cross-video) | full dedup port; `state["dedup_groups"]` | dedup ✅ |
| P9 | feat(v55-stages): port store stage via repos + write image boundary | full store port; `state["final_cards"]` populated; `crops/` written | store ✅ |
| P10 | test(v55-stages): synthetic e2e fixture + assertion of cards>0 | `tests/pipeline/test_back_half_e2e.py`; extend `test_unified_runtime.py` | — |
| P11 | feat(v55-stages): mid-stage progress events via telemetry | `PipelineTelemetry.progress(stage, pct, detail)`; stages emit it; `EventBusTelemetry` forwards | enables UI progress bars |
| P12 | test(v55-ui): UI integration assertions + DB-shape contract tests | `tests/app/test_run_to_cards.py`; assert `/api/runs/{id}/cards` returns rows after a synthetic run | — |
| P13 | docs(v55-stages): per-stage V4-vs-V5.5 audit table | `docs/superpowers/audits/2026-05-29-v55-back-half-audit.md`; one section per stage | — |
| P14 | docs(v55-stages): manual golden-set baseline re-run + CLAUDE.md status update | doc updates + numbers from real-video run | — |

**Estimated LOC delta after P1–P14:** ~1,650 added (stages + repo methods + helpers + tests + UI integration + audit doc), ~120 deleted (stub stage bodies).

**Estimated review effort:** P4 is the largest single commit (~250 LOC); the rest are 70–180 LOC each. The sequence is designed so each phase compiles + tests green independently — `LocalPipelineRuntime.run()` still completes after every commit, just with progressively less of the "card count = 0" outcome.

---

## 9. Non-goals (for this plan)

- **Multi-process / GPU-strict refactoring.** The current `LocalPipelineRuntime` is single-threaded and runs everything in the main process. The producer/worker thread split CLAUDE.md describes (`UnifiedRuntime` with `_worker` for GPU work) is **not implemented anywhere** — it's aspirational documentation. Wiring the back half does not require building that split; it can land as a follow-up phase once the stages are functional.
- **Telemetry parity.** V4 emitted detailed per-stage timing and frame-decode counters; V5.5 stage telemetry currently only fires `stage_started` / `stage_finished`. Full parity is a separate concern; this plan emits the V4-equivalent `print` lines but doesn't add OpenTelemetry counters.
- **Training-data export coverage.** The V4 `refine` step calls `storage.add_track_telemetry(...)` which feeds the presence-classifier training pipeline. This plan ports those calls verbatim, but if the training service relies on additional fields that V4 wrote inline, those become P11 follow-ups.

---

## 10. Risks

1. **No CI-runnable golden fixture.** P10 adds a synthetic fixture, but it can't catch regressions against the real `IMG_5872.MOV` baseline. Manual re-run of the harness against the golden set is required before merging this PR, and that requires the user (machine has the video; CI doesn't).
2. **`pipeline_utils.py` is large (618 LOC) and architecturally on the "to be moved" list.** The import-linter contract `layered` may complain about importing it from the stages package if the contract is tightened later. Acceptable for now; phase E left these helpers untouched.
3. **DinoEmbedder and FBPredictor are optional dependencies.** V4 already degrades gracefully when missing; ported code preserves that. Tests must use the heuristic paths so they don't depend on model checkpoints.
4. **In-memory peak.** Estimate: ~180 MB for the reference video, scaling linearly with concurrent active tracks. If a video has unusual track concurrency this could be a problem; mitigation is to spill `frame_entries` to disk between refine and score (i.e. opt back into the V4 on-disk pattern selectively — adds ~20 LOC at the refine/score boundary).
5. **`store` writes ~16 JPEGs per track (8 per frame_entry + 1 fused).** Acceptable; matches V4 output volume.
6. **SSE progress event shape may have drifted in the frontend.** The legacy Metaflow contract was `{stage_id, pct, detail}`; if the SPA was updated against a different shape during the v5.5 refactor, P11/P12 will catch it but the fix may require frontend edits not currently scoped. Mitigation: P12 includes the manual UI smoke checklist before merge.
7. **Per-stage audit cost.** The audit table in §13 has ~30 behaviors to confirm across 5 stages. Realistic: ~3–4 hours focused review per stage. The audit is the long pole; budget accordingly.
8. **Golden-set re-run requires real hardware.** P14 cannot run in CI — needs the operator (user) on a machine with `IMG_5872.MOV` and a working YOLO weights file. If the user is unavailable, P14 blocks merge.

---

## 11. Open questions

1. Do we want the array-variant ML helpers (DinoEmbedder.embed_array etc.) to land as part of P3, or as separate prep PRs that can be reviewed independently? Recommendation: P3 as proposed (small, mechanical).
2. Is there an existing synthetic test fixture I missed? Searched `tests/fixtures/` and `tests/conftest.py`; nothing 5-second-MOV-ish. If one exists in a worktree or branch, we should reuse it.
3. Should the in-memory peak mitigation (#10.4) be a separate phase or a follow-up? Recommendation: follow-up unless P10's e2e shows it's needed.
4. Does `MultiFrameFuser.fuse` need the `confidence` channel passthrough that V4 had buried in `quality_components`, or is the V4 step's `prepared_track["frame_entries"][0]["quality_score"]` simplification (line 84 of V4 fuse.py: `# simplified`) acceptable? Recommendation: accept the V4 simplification verbatim; revisit only if golden-set regression flags it.

---

## Appendix A — file-by-file reading list (for the implementer)

When executing this plan, read in this order:

1. `tests/pipeline/test_runtime_smoke.py` — current smoke contract.
2. `src/card_capture/pipeline/runtime_local.py` — orchestrator (already honors `db_path`).
3. `src/card_capture/pipeline/stages/*.py` — stubs to replace.
4. `.worktrees/ci-fixes/pipeline/steps/{track,refine,score,resolve,fuse,dedup,store}.py` — the source of truth being ported.
5. `src/card_capture/{scoring,fuser,deduplicator,selector,pipeline_utils,gpu_refinement}.py` — helpers.
6. `src/card_capture/data/repositories/{cards,runs,events}.py` — repository patterns to mirror.
7. `src/card_capture/data/sql_queries.py` — where new SQL constants live.

---

## 12. UI integration scope

The user's explicit completion criteria for "UI working again":

1. **Cards appear in the review UI after a run.** Drop-in via P9 (store stage produces rows). Verification in P12.
2. **Per-stage progress bars stream live.** Requires P11 — add a `progress(stage, pct, detail)` method to `PipelineTelemetry` and have heavy stages (refine, score, fuse) call it mid-loop.
3. **Regression UI / harness page works against new runs.** Verification in P12.

Out of scope for this plan (user did not request): exposing the new config knobs (§4.8) in the settings UI. Tracked as a separate follow-up.

### 12.1 P11 — mid-stage progress events

**Contract addition** to `card_capture.pipeline.telemetry`:

```python
class PipelineTelemetry(Protocol):
    # existing methods unchanged
    def progress(self, stage: str, pct: int, detail: str) -> None: ...

class NoopTelemetry:
    def progress(self, stage: str, pct: int, detail: str) -> None: ...
```

`InMemoryTelemetry` appends a `TelemetryEvent("progress", {"stage", "pct", "detail"})`.

`EventBusTelemetry.progress` emits the existing UI `stage_progress` event shape:

```python
def progress(self, stage, pct, detail):
    self._bus.emit(self._run_id, Event(name="stage_progress",
                                        payload={"stage_id": stage, "pct": pct, "detail": detail}))
```

**Stages that should emit progress (P11 wiring):**

| Stage | Trigger | Cadence |
|---|---|---|
| sample  | every 100 decoded frames | `pct = 100 * decoded / estimated_total` |
| detect  | every batch | `pct = 100 * batches_done / total_batches` |
| refine  | per-track | `pct = 100 * tracks_done / total_tracks` |
| score   | per-track | same |
| resolve | per-session | `pct = 100 * sessions_done / total_sessions` |
| fuse    | per-track | same as refine |
| dedup   | per-instance pair iteration | every 25% |
| store   | per-instance | `pct = 100 * stored / total` |

Stages that finish in <100ms (novelty, track) skip mid-progress emission; their `stage_started`/`stage_finished` events suffice.

**Frontend assumption check:** The existing SSE consumer in `app/web/src/` listens for `event: stage_progress` with `data: {stage_id, pct, detail}` (legacy Metaflow contract). This matches the proposed payload — no frontend code change required if my read is correct. P12 includes a frontend smoke pass to confirm.

### 12.2 P12 — UI integration assertions

New tests under `tests/app/`:

1. **`test_run_to_cards.py::test_full_run_populates_cards_endpoint`** — uses the synthetic fixture from P10. Starts a run via `POST /api/videos/{id}/process`, polls until `status=completed`, then `GET /api/runs/{run_id}/cards` and asserts a non-empty list with the expected fields (instance_id, fused_image_path, angle, quality_score, etc.).
2. **`test_run_to_cards.py::test_sse_emits_stage_progress`** — subscribes to `/api/runs/{run_id}/events`, asserts that at least one `stage_progress` event per stage arrives with valid `(stage_id, pct, detail)` payload, and that `pct` is monotonically non-decreasing within a single stage.
3. **`test_run_to_cards.py::test_regression_harness_metrics`** — runs the `card-capture harness` CLI against the synthetic fixture and the synthetic ground-truth, asserts that `card_recall ≥ 0.5` and the metrics JSON validates against `harness/performance/runner.py:PerfReport.to_json()` shape.
4. **Frontend smoke (manual, documented checklist).** Listed in §13.4; not automated because we don't have a Playwright harness wired into CI in this repo yet.

### 12.3 P12 — UI smoke checklist (manual)

To be executed by the user before merging the PR:

```
[ ] uvicorn app.main:app --reload starts cleanly
[ ] http://127.0.0.1:8000/ loads the SPA without console errors
[ ] Upload synthetic_two_cards.MOV → video appears in the list as "pending"
[ ] Click "Process" → video transitions to "processing", SSE stream opens
[ ] Stage progress bars step through sample → detect → ... → store
[ ] Pipeline completes; video shows "completed" with cards_extracted > 0
[ ] Review page lists the extracted cards with thumbnail and quality score
[ ] Labeling page lets you tag F/B without 500 errors
[ ] Regression page shows the run in the runs list
[ ] curl http://127.0.0.1:8000/api/runs/<run_id>/cards returns the expected JSON
```

---

## 13. Per-stage V4 audit (P13 deliverable)

A dedicated audit document lands as `docs/superpowers/audits/2026-05-29-v55-back-half-audit.md`. Each ported stage gets a section with the table below:

```markdown
### Stage: <name>

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/<name>.py` @ <ref-sha>
**V5.5 ported:** `src/card_capture/pipeline/stages/<name>.py` @ <commit-sha>
**Ported in commit:** <P# title>

#### Behavior parity

| V4 behavior | V5.5 behavior | Status |
|---|---|---|
| ...specific behavior line... | ...specific behavior line... | ✅ identical / ⚠️ deviates / ❌ not ported |

#### Deviations (each requires a reason)

1. **<deviation>**: <V4 did X / V5.5 does Y>. **Reason:** <e.g., in-memory mandate>. **Risk:** <none / low / medium>.

#### Removed (with reason)

- <Anything intentionally not ported, e.g., legacy `card_capture_config.json` writes>.

#### Test coverage

- Unit: <tests/.../test_<name>_stage.py> — <what's covered>
- E2E: <covered by P12 test_full_run_populates_cards_endpoint>
- Golden-set regression: <metric tolerance — covered in P14>
```

**Audit completion criteria:** every stage row in the §1 table has at least one "✅" in the audit's "Behavior parity" table for the *card-producing* behaviors. Deviations are acceptable but must each have a logged reason.

### 13.1 Specific V4 behaviors the audit must explicitly confirm or reject

| V4 behavior | Decision needed |
|---|---|
| `refine`: persist per-frame `add_track_telemetry` rows during refine | Confirm port. Tests in P12 should query `track_telemetry` after a run. |
| `refine`: write rectified crops to `crops/track_<id>_det_<id>_rectified.jpg` | Moved to store stage in P9; audit must confirm filename pattern matches. |
| `refine`: DINOv2 ReID embedding via `DinoEmbedder.embed_image(path)` | Confirm `_array` variant matches `.embed_image` byte-for-byte on the same image. |
| `score`: novelty-gate adaptive threshold (largest-gap midpoint) | Confirm calculation identical to V4 `_track_meds`/`_gaps`/`_adaptive` block. |
| `score`: `stand_gate` requires `bg_model is not None` AND `stand_nov_max > 0` | Confirm both preconditions present in port. |
| `resolve`: high-conf F/B classifier overrides textiness via `side_score = 0.8 + conf*0.2 / 0.2 - conf*0.2` | Confirm the magic numbers carry through. |
| `resolve`: hard-case capture writes `hard_cases.jsonl` to `output_dir` | Confirm port — easy to drop if `output_dir` not threaded. |
| `fuse`: foil threshold passed to `MultiFrameFuser.fuse` only when `enable_foil_aware_fusion` | Confirm config gate. |
| `fuse`: single-frame path uses `shutil.copy` (V4) vs in-memory passthrough (V5.5) | Decision logged in audit; behavior should be observably identical. |
| `dedup`: `SAME_CARD_EMB_THRESHOLD = 0.15` constant | Confirm constant identical. |
| `dedup`: cross-video query excludes the current `video_id` | Critical — easy to break. Test asserts a card from video A never appears as a duplicate of itself in video A. |
| `store`: best-canonical view points to the **fused** image path, not the rectified path | Critical — V4 comment "(A1)" at line 98. Test asserts the view row's `rectified_path` equals `fused_image_path` when `is_best`. |
| `store`: `add_saved_card` is called only when `is_canonical AND is_best` | Confirm both conditions present. |
| `store`: `Storage.update_video_status(video_id, "completed")` is the last DB write | Confirm; UI depends on this for the video card to show "completed". |

---

## 14. Test coverage expansion (post-P12)

Beyond the e2e (P10) and UI integration (P12) tests, the audit phase requires these to land:

### 14.1 Per-stage unit tests

One test file per ported stage under `tests/pipeline/stages/`:

| Test | What it asserts |
|---|---|
| `test_refine_stage.py::test_carries_identity` | After `refine.run`, each frame_entry has `(detection_id, frame_index, track instance_id)` reachable. |
| `test_refine_stage.py::test_in_memory_normalized` | `frame_entries[*]["normalized"]` is np.ndarray, not str. |
| `test_score_stage.py::test_novelty_gate_useful_thresholds` | Reproduces the V4 `_novelty_gate_useful` cases (n=5+, std>0.15, min<0.35). |
| `test_score_stage.py::test_adaptive_novelty_threshold` | Two-cluster scores → adaptive threshold falls in the gap midpoint. |
| `test_resolve_stage.py::test_longest_track_is_front` | Without FB classifier, longest track becomes Front. |
| `test_resolve_stage.py::test_same_card_via_embedding` | Two tracks with cosine distance < 0.5 are grouped. |
| `test_resolve_stage.py::test_same_card_via_phash_fallback` | Same logic when embeddings missing. |
| `test_fuse_stage.py::test_single_frame_passthrough` | `fusion_target_frames=1` → fused == best canonical. |
| `test_fuse_stage.py::test_foil_branch_invoked_when_enabled` | Foil-positive frames + `enable_foil_aware_fusion=True` → `glare_rejection_fusion` called. |
| `test_dedup_stage.py::test_intra_run_grouping` | Two near-identical tracks → grouped. |
| `test_dedup_stage.py::test_cross_video_query_excludes_self` | Critical: a card from this video never appears in `WHERE video_id != ?`. |
| `test_store_stage.py::test_best_view_points_to_fused` | The card_view with `is_best=True` has `rectified_path == fused_image_path`. |
| `test_store_stage.py::test_crops_directory_written` | `crops/` exists with expected count of JPEGs. |
| `test_store_stage.py::test_video_marked_completed_last` | DB write order: card_instances → card_views → saved_cards → video status. |

### 14.2 Repository tests (P2 deliverable)

For each new `CardsRepository` method (§5), a unit test under `tests/data/test_cards_repository.py` matching the existing pattern (`test_runs_repository.py` style).

### 14.3 Helper-class adapter tests (P3 deliverable)

| Test | What it asserts |
|---|---|
| `test_dino_embedder_array.py::test_embed_array_matches_embed_image` | `embed_array(cv2.imread(p)) == embed_image(p)` byte-for-byte on a fixed fixture. |
| `test_fb_predict_array.py::test_predict_array_matches_predict` | Same. |
| `test_reid_embeddings_array.py::test_compute_reid_embedding_array_matches_path_variant` | Same. |

### 14.4 Regression-guard against the V4 baseline

P14 runs the harness against `IMG_5872.MOV` and writes the result to `docs/superpowers/plans/v5-5/back-half-baseline.md`. Acceptance gates:

- `card_recall` within ±0.05 of V4 baseline
- `card_precision` within ±0.05
- `side_accuracy` within ±0.05
- `image_quality (SSIM)` within ±0.05
- `image_quality (PSNR)` within ±0.5

Failure of any gate blocks merge of P14 and requires audit-finding remediation.

---

## 15. Sign-off checklist (before merging the PR series)

```
P1  [ ] config keys codified; pipeline_runner threads PipelineConfig
P2  [ ] 8 CardsRepository methods + SQL constants + tests green
P3  [ ] 3 ML helper _array variants + parity tests green
P4  [ ] refine ported; pytest -q tests/pipeline/stages/test_refine_stage.py green
P5  [ ] score ported; per-stage tests green
P6  [ ] resolve ported; per-stage tests green
P7  [ ] fuse ported; per-stage tests green
P8  [ ] dedup ported; per-stage tests green
P9  [ ] store ported; per-stage tests green; cards.sqlite rows appear after run
P10 [ ] synthetic 2-card MOV fixture + e2e test asserting cards>0 lands
P11 [ ] PipelineTelemetry.progress contract added; stages emit; EventBusTelemetry forwards
P12 [ ] UI integration tests green; SSE smoke test asserts mid-stage progress
P13 [ ] per-stage audit doc complete; all card-producing behaviors marked ✅ or deviation logged
P14 [ ] manual golden-set re-run; all 5 metrics within gate; CLAUDE.md updated
    [ ] manual UI smoke checklist (§12.3) executed and pasted into PR description
    [ ] PR description links the audit doc and the back-half-baseline.md
    [ ] Existing failing tests (objc dyld, pytest-asyncio) unchanged from baseline
    [ ] All import-linter contracts still kept (5/5)
    [ ] tests/architecture/test_raw_sql_outside_data.py still passes (0 raw SQL outside data/)
```

---

## Appendix B — what `fix/ui-v55-unified-runtime` (PR #60) already established

These pieces are already on the branch and the plan assumes them:

- `PipelineRunRequest` accepts `db_path`, `video_id`, `config_preset`.
- `LocalPipelineRuntime` honors explicit `request.db_path` and threads `video_id` / `config_preset` into stage state.
- `EventBusTelemetry` adapter bridges `PipelineTelemetry` → SSE.
- `pipeline_runner.py`, `worker_core.py`, `training_service.py`, `cli.py` all drive `LocalPipelineRuntime` in-process — no more `pipeline.card_capture_flow` subprocess.

So this plan does not need to touch the runner / CLI again; only the stages and supporting repos.
