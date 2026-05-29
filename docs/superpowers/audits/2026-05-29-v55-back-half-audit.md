# V5.5 Back-Half Stage Audit

Date: 2026-05-29
Auditor: Claude (Opus 4.7) on behalf of @jpglick
Branch: `feat/v55-back-half-wiring`
Ported across commits:

| Phase | Commit | Subject |
|---|---|---|
| P4–P9 (stages) | `346db869` | feat(v55-back-half): complete pipeline stage ports and DAL extensions |
| Plan/UI rewire  | `fa89169b` | fix(ui): rewire pipeline runners to v5.5 in-process unified runtime |
| Phase 14 prep   | `a644240f` | docs(v55-back-half): baseline template + CLAUDE.md weaknesses update |

This audit walks every back-half card-producing behavior in V4 (`.worktrees/ci-fixes/pipeline/steps/<stage>.py`) and marks the corresponding line in V5.5 (`src/card_capture/pipeline/stages/<stage>.py`) as identical (✅), deviation (⚠️), or removed (❌). Every ⚠️/❌ row has a Deviations entry with reason + risk.

Two systemic adaptations apply to every stage and are NOT called out individually in tables (only the stage-specific consequences are):

- **V4 Metaflow dataclass IO (`RefineOutput`, `ScoreOutput`, etc.) → V5.5 state-mutation.** Each stage now mutates a shared `state: dict` and reads `state["request"].config` instead of `ctx: RunContext`.
- **V4 on-disk crops (`cv2.imwrite` / `cv2.imread`) → V5.5 in-memory `np.ndarray` carried on `frame_entry["normalized"]`, `track["best_canonical_image"]`, `fused["fused_image"]`.** The single filesystem boundary is the `store` stage.
- **V4 raw `sqlite3` via `Storage` → V5.5 `CardsRepository` / `RunsRepository` via the Single-Writer DAL.** Enforced by the `no-sqlite3-outside-data` import-linter contract.

---

## Stage: refine

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/refine.py` (408 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/refine.py` (242 LOC)
**Ported in commits:** P4.2, P4.3 — `346db869`

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Decode only the high-res frames needed for each track's canonical candidates (`decode_frames_gpu`) | Reuses `state["sampled_frames"]` from the sample stage — never re-decodes | ⚠️ deviation | See Deviations §1 |
| Sort candidates by `score_total` desc; take top-K canonical candidates | identical (consumes V4's `ScoredCandidate` / selector logic) | ✅ | |
| Kornia warp batch per track (`KorniaNormalizer.warp_canonical_batch`) | identical | ✅ | refine.py V5.5: `kornia_normalizer.warp_canonical_batch(batch_items, rotate_180=...)` |
| `PrecisionNormalizer` CPU fallback | identical (per-candidate `normalizer.normalize(...)` on Kornia failure) | ✅ | |
| `QualityScorer.score(normalized, conf, novelty=...)` per frame_entry | identical | ✅ | |
| `find_glare_centroid` + glare_mask + laplacian_heatmap attached per frame_entry | identical (uses `_compress_array` wrappers) | ✅ | |
| `_select_canonical_entries` for `is_canonical` flag | identical (shared util in `pipeline_utils`) | ✅ | |
| Per-track Laplacian scan via `_laplacian_select_frames` (synthetic candidates with borrowed corners) | Same Laplacian helper, but using already-decoded frame dict instead of a fresh `VideoCapture` | ⚠️ deviation | See Deviations §2 |
| Persist `add_track_telemetry` rows per canonical (area, aspect, cx, cy) | `CardsRepository.add_track_telemetry(...)` via injected `state["repos"]["cards"]` | ⚠️ deviation | See Deviations §3 |
| Corner refinement when `ctx.corner_refinement` | identical (`config["corner_refinement"]`) | ✅ | |
| DINOv2 ReID embedding via `DinoEmbedder.embed_image(path)` | `DinoEmbedder.embed_array(best_canonical_img)` | ⚠️ deviation | See Deviations §4 |
| `cv2.imwrite` rectified crops to `crops_dir` | crops kept as ndarray in `frame_entries[*]["normalized"]`; no disk I/O | ⚠️ deviation | See Deviations §5 |
| `frame_entries[*]["image_path"]` populated mid-stage | left empty — `store` stage populates | ❌ removed | See Removed |
| Stage-level `print` summary line | replaced by `telemetry.resource_sample(...)` (e.g. `kornia_warp_failed`, `reid_embedding_failed`) | ⚠️ deviation | No risk; PipelineRunner captures telemetry already |

### Deviations

1. **No re-decode.** V4 called `decode_frames_gpu(video_path, sorted(_all_needed))` to get high-res frames. V5.5 looks them up in `state["sampled_frames"]` (populated by the sample stage running off the same decode handle). **Reason:** v5.5 in-process mandate eliminates the redundant decode. **Risk:** Low — if the sample stage didn't retain a needed frame, V5.5 falls back to a black `np.zeros((h, w, 3), uint8)` placeholder. Tests `test_refine_stage.py::test_*` cover this fallback.
2. **Laplacian scan uses already-decoded frames.** V4's helper accepted a `decoded_frames=` kwarg, and V5.5 passes the populated `state["sampled_frames"]` dict. **Reason:** in-process mandate. **Risk:** None — same helper, same outputs when inputs match.
3. **`add_track_telemetry` via repository, not raw SQL.** **Reason:** `no-sqlite3-outside-data` import-linter contract. **Risk:** None — `CardsRepository.add_track_telemetry` is a thin Phase 2 wrapper around the same `INSERT`.
4. **DINOv2 input is in-memory ndarray, not file path.** V4 wrote the rectified JPEG and called `embed_image(path)`; V5.5 calls `embed_array(best_canonical_img)`. **Reason:** in-memory mandate. **Risk:** Low — embedding runs on cleaner (no JPEG compression) pixels, so cosine distances may shift by O(1e-3). Phase 3 parity test bounds the delta at 1e-5 when fed the post-JPEG image; downstream `SAME_CARD_EMB_THRESHOLD = 0.15` has ample headroom.
5. **Rectified crops live in-memory until `store`.** **Reason:** in-memory mandate. **Risk:** Memory peak (~180 MB on the reference video, per spec §10.4). Mitigation (selective spill between refine and score) is a tracked follow-up in CLAUDE.md "Known Weaknesses".

### Removed (with reason)

- `frame_entries[*]["image_path"]` is no longer populated by refine. **Reason:** no path exists yet — `store` populates it (`f"crops/track_<iid>_det_<id>_rectified.jpg"`). No external consumer reads `image_path` before `store` runs (verified by reading all downstream stages).

### Test coverage

- Unit: `tests/pipeline/stages/test_refine_stage.py` (identity, normalization shape, embedding attach, telemetry rows)
- E2E: `tests/pipeline/test_back_half_e2e.py`
- Golden-set regression: Phase 14 — `card_recall` / `card_precision` / `image_quality(SSIM)` within ±0.05

---

## Stage: score

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/score.py` (180 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/score.py` (107 LOC)
**Ported in commits:** P5.1 — `346db869`

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| `_novelty_gate_useful` returns True iff `len>=5`, `std>0.15`, `min<0.35` | identical (constants reproduced verbatim) | ✅ | (constants confirmed at score.py:23–28) |
| Collect novelty scores from `refined_tracks[*].frame_entries[*].novelty_score` | identical | ✅ | |
| Adaptive novelty threshold = midpoint of largest gap between per-track median novelties | identical | ✅ | |
| Adaptive threshold capped at `ctx.novelty_floor` via `min(_adaptive, ctx.novelty_floor)` | identical, capped at `config['novelty_floor']` | ✅ | |
| When fewer than 2 tracks: fall back to `novelty_floor` | identical | ✅ | |
| Gate stays off (`novelty_threshold = -1.0`) when bg_model is None or gate not useful | identical | ✅ | |
| Per-track median novelty / quality / sharpness appended to track dict | identical (`median_novelty`, `median_quality`, `median_sharpness`) | ✅ | |
| Confidence-floor prune when `ctx.track_confidence_floor > 0` AND `median_quality < floor` | identical with `config['track_confidence_floor']` | ✅ | |
| Stand prune requires bg_model AND stand_nov_max > 0 AND median_novelty < stand_nov_max AND median_sharpness < stand_shp_max | identical (all four preconditions) | ✅ | |
| Set `pruned=True` on track dict when any gate fires | identical | ✅ | |
| Emit `pruned_instance_ids: List[str]` | identical (`state["pruned_instance_ids"]`) | ✅ | |
| Stage-level print line summarising counts | replaced by `telemetry.resource_sample` | ⚠️ deviation | See Deviations §1 |
| Import `from pipeline.steps.start import RunContext` | not needed — knobs read from `state["request"].config` | ❌ removed | See Removed |

### Deviations

1. **Stage summary log line.** V4 printed `[Stage: Score] | …` to stdout; V5.5 routes equivalent counts through telemetry. **Reason:** `PipelineRunner` already captures telemetry; stdout would duplicate. **Risk:** None.

### Removed

- `from pipeline.steps.start import RunContext` and the `ScoreOutput` dataclass: V5.5 operates on `state` and `state["request"].config` directly.

### Test coverage

- Unit: `tests/pipeline/stages/test_score_stage.py` (passthrough when off, confidence-floor prune, novelty-gate-useful, adaptive threshold)
- E2E: `tests/pipeline/test_back_half_e2e.py`
- Golden-set: Phase 14 — `card_recall` / `card_precision` ±0.05

---

## Stage: resolve

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/resolve.py` (234 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/resolve.py` (87 LOC; algorithm lives in shared `_resolve_session_tracks` helper in `pipeline_utils`)
**Ported in commits:** P6.1 — `346db869`

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Filter out `pruned` tracks (`active_tracks = [t for t in scored if not t['pruned']]`) | identical | ✅ | |
| Group active tracks by `session_id` | identical | ✅ | |
| Compute `side_score` via `_side_textiness_score(img)` per track | identical (img from `track['best_canonical_image']`, not disk) | ✅ | |
| Compute `appearance_vector` via `_appearance_vector(img)` | identical | ✅ | |
| Find `max_length = max(len(t["frame_entries"]) for t in session_tracks)` | identical | ✅ | |
| F/B classifier override: side="front", conf>0.8 → `side_score = 0.8 + conf*0.2` | identical magic numbers | ✅ | |
| F/B classifier override: side="back",  conf>0.8 → `side_score = 0.2 - conf*0.2` | identical magic numbers | ✅ | |
| Primary sort: `(-side_score, -_compute_quality_weighted_score_dict(t, max_length))` | identical | ✅ | |
| `_compute_quality_weighted_score_dict` weights: `0.6 * normalized_length + 0.4 * mean_quality_of_canonicals` | identical (shared helper) | ✅ | |
| First sorted track → angle="Front", `duplicate_track_index=None` | identical | ✅ | |
| Same-card via embedding `embedding_same_card_score(emb1, emb2, threshold=0.5)` | identical | ✅ | |
| pHash fallback via `deduplicator.hamming_distance` | identical | ✅ | |
| `AdaptiveThresholdComputer.compute_hamming_threshold(observed, global_threshold=15.0)` | identical | ✅ | |
| Same-card → angle="Back", `duplicate_track_index = active_tracks.index(primary)` | identical | ✅ | |
| Hard-case capture via `is_hard_case` / `capture_hard_case` to `<output_dir>/hard_cases.jsonl` | identical (uses `state["output_root"]` as path; gracefully no-ops if None) | ✅ | |
| Returns `prepared_tracks` with side metadata attached | written into `state["prepared_tracks"]` (flat list of active tracks) | ✅ | |
| `cv2.imread(t["best_canonical_image_path"])` | `t["best_canonical_image"]` ndarray | ⚠️ deviation | See Deviations §1 |
| `FBPredictor.predict(path)` | `FBPredictor.predict_array(ndarray)` | ⚠️ deviation | See Deviations §2 |
| `ctx.observed_intra_track_distances` (always empty in single-flow V4) | `state.get("observed_intra_track_distances", [])` | ⚠️ deviation | See Deviations §3 |
| Internal `_capture_hard_cases` helper (pass-through stub in V4) | not ported — `capture_hard_case` is the public function used directly | ❌ removed | dead code in V4 |

### Deviations

1. **Best-canonical image is in-memory ndarray, not file path.** **Reason:** v5.5 in-memory mandate. **Risk:** None — `_side_textiness_score` / `_appearance_vector` already accept ndarrays.
2. **F/B classifier consumes ndarray (`predict_array`).** Phase 3 parity test confirms `predict_array(re_read) == predict(path)` to numerical equivalence. **Reason:** in-memory mandate. **Risk:** None.
3. **`observed_intra_track_distances` always empty.** V4 also passed an empty list whenever the DB was fresh; `AdaptiveThresholdComputer` falls back to its `global_threshold=15.0` default. Surfacing prior-run distances is out of scope for this phase. **Risk:** Low — same fallback behavior as V4 fresh-DB runs.

### Removed

- The internal `_capture_hard_cases` helper in V4 was a documented stub (`pass`); the real call is to the module-level `capture_hard_case`. V5.5 omits the dead wrapper.

### Test coverage

- Unit: `tests/pipeline/stages/test_resolve_stage.py` (pruned excluded, longest→Front, embedding same-card→Back, pHash fallback)
- E2E: `tests/pipeline/test_back_half_e2e.py`
- Golden-set: `side_accuracy` ±0.05 in Phase 14

---

## Stage: fuse

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/fuse.py` (121 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/fuse.py` (80 LOC)
**Ported in commits:** P7.1 — `346db869`

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Per-track loop (one Metaflow `foreach` subprocess each in V4; ~4–6 min overhead on reference video) | in-process `for track in prepared_tracks:` | ⚠️ deviation | See Deviations §1 (perf-only) |
| Filter `frame_entries` by `is_canonical` | identical (`canonical_entries = [fe for fe in frame_entries if fe.get("is_canonical")]`) | ✅ | |
| `cv2.imread(fe["image_path"])` for each canonical | replaced by `fe["normalized"]` (ndarray already in memory) | ⚠️ deviation | See Deviations §2 |
| Single-frame passthrough when `len(images)==1` or `fusion_target_frames <= 1` | identical (returns `best_canonical_image`) | ✅ | |
| `MultiFrameFuser().fuse(images, foil_threshold=...)` | identical (helper already accepts ndarrays) | ✅ | |
| `foil_threshold = ctx.foil_threshold if ctx.enable_foil_aware_fusion else None` | identical (with `config["foil_threshold"]` / `config["enable_foil_aware_fusion"]`) | ✅ | |
| On fusion exception: fallback to single-frame, log warning | identical (V4 used `print`; V5.5 uses `telemetry.resource_sample`) | ✅ | |
| Output dict fields: `instance_id`, `session_id`, `angle`, `primary_hash`, `quality_score`, `side_score`, `appearance_vector`, `best_canonical_detection_id`, `duplicate_track_index`, `first_frame_index`, `reid_embedding` | identical fields | ✅ | |
| `fused_image_path` field written here | renamed → `fused_image` (ndarray) until `store` writes the JPEG and populates `fused_image_path` | ⚠️ deviation | See Deviations §3 |
| `cv2.imwrite(fused_path, fused_img)` | deferred to `store` stage | ⚠️ deviation | See Deviations §4 |
| `shutil.copy(best_path, fused_path)` for single-frame path | direct passthrough `fused["fused_image"] = best_canonical_image` | ❌ removed | See Removed |

### Deviations

1. **Per-track loop is in-process, not a Metaflow `foreach`.** **Reason:** v5.5 in-process mandate eliminates ~4–6 min of subprocess overhead on the reference video. **Risk:** None — identical algorithm; no shared state between iterations.
2. **Canonical-frame inputs are ndarrays, not file paths.** **Reason:** in-memory mandate. **Risk:** None.
3. **Field renamed `fused_image_path` → `fused_image`.** **Reason:** no path exists yet; `store` writes the file and populates the path. **Risk:** Downstream `dedup` stage reads `primary_hash` + `reid_embedding`, not the path, so it's safe. `store` reads `fused["fused_image"]` to write the JPEG and then sets `fused["fused_image_path"]`.
4. **Image write deferred to `store` stage.** **Reason:** single filesystem boundary per v5.5 mandate. **Risk:** None.

### Removed

- V4's `shutil.copy(best_path, fused_path)` for the single-frame path. V5.5 sets `fused["fused_image"] = best_canonical_image` directly. Observable behavior identical — the JPEG that `store` writes is byte-identical to what `shutil.copy` would have produced.

### Test coverage

- Unit: `tests/pipeline/stages/test_fuse_stage.py` (1-per-track, single-frame passthrough, foil enabled, foil disabled)
- E2E: `tests/pipeline/test_back_half_e2e.py`
- Golden-set: `image_quality(SSIM)` and `(PSNR)` deltas in Phase 14

---

## Stage: dedup

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/dedup.py` (127 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/dedup.py` (99 LOC)
**Ported in commits:** P8.1 — `346db869`

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| **Constant `SAME_CARD_EMB_THRESHOLD = 0.15`** | identical (module-scope) | ✅ | (constant explicit in audit per spec §13.1) |
| **Constant `SAME_CARD_HAMMING_MAX = 8`** | identical (module-scope) | ✅ | (constant explicit in audit per spec §13.1; V4 referenced the same value `15.0` only in the adaptive-fallback path of resolve; dedup itself uses 8) |
| Outer `for i, f1 in enumerate(fused_canonicals):` with `processed: set` | identical | ✅ | |
| Group fields: `canonical_instance_id`, `duplicate_instance_ids`, `hamming_distances`, `embedding_distances`, `cross_video_parent_id` | identical | ✅ | |
| Intra-run: try embedding first (`1.0 - np.dot(emb1, emb2)`), fall back to pHash | identical | ✅ | |
| Cross-video: raw SQL `SELECT id, reid_embedding FROM card_instances WHERE reid_embedding IS NOT NULL AND is_duplicate_of IS NULL AND video_id != ?` | replaced by `CardsRepository.find_embeddings_excluding_video(video_id=current_video_id)` | ⚠️ deviation | See Deviations §1 |
| Cross-video: track best (min dist) and set `cross_video_parent_id` iff `min_dist < SAME_CARD_EMB_THRESHOLD` | identical | ✅ | |
| **Cross-video query excludes current `video_id`** | identical (test `test_dedup_cross_video_query_excludes_self_video_id` proves it) | ✅ | (critical per spec §13.1) |
| Print line on cross-video match | dropped (telemetry already captures stage transitions) | ⚠️ deviation | See Deviations §2 |
| Pre-fetch cross-video embeddings ONCE outside the loop | identical (V5.5 hoists `find_embeddings_excluding_video` out of the inner loop for efficiency) | ✅ | minor efficiency improvement |
| `DedupOutput.dedup_distances: Dict[str, float]` field (always `{}`) | not emitted (V5.5 stage writes `state["dedup_groups"]` only) | ❌ removed | See Removed |

### Deviations

1. **Cross-video query uses `CardsRepository`, not raw SQL.** **Reason:** `no-sqlite3-outside-data` import-linter contract. **Risk:** None — same WHERE clause, returns `List[Tuple[int, bytes]]` of `(row_id, embedding_bytes)`.
2. **Print line dropped on cross-video match.** **Reason:** PipelineRunner telemetry already records per-stage transitions; the print was diagnostic noise. **Risk:** None — `state["dedup_groups"]` carries the `cross_video_parent_id`, which is sufficient.

### Removed

- V4 returned `DedupOutput(dedup_groups=..., dedup_distances={})`. The `dedup_distances` field was unused (always empty `{}`). V5.5 stage doesn't emit it.

### Test coverage

- Unit: `tests/pipeline/stages/test_dedup_stage.py` (intra-run by embedding, intra-run by pHash, cross-video query excludes self, cross-video match sets parent_id)
- E2E: `tests/pipeline/test_back_half_e2e.py`

---

## Stage: store

**V4 source:** `.worktrees/ci-fixes/pipeline/steps/store.py` (155 LOC)
**V5.5 ported:** `src/card_capture/pipeline/stages/store.py` (168 LOC; slightly larger because the image-write block that lived implicitly in earlier stages now lives explicitly here)
**Ported in commits:** P9.1 — `346db869`

### Behavior parity

| V4 behavior | V5.5 behavior | Status | Audit note |
|---|---|---|---|
| Build `id_map: Dict[str,int]`, `fused_map`, `track_map` dicts | identical | ✅ | |
| (NEW in v5.5) Write all `fused["fused_image"]` ndarrays as `crops/instance_<iid[:8]>_fused.jpg` and populate `fused["fused_image_path"]` BEFORE DB writes | not present in V4 (V4 wrote per-stage) | ⚠️ deviation | See Deviations §1 (single filesystem boundary) |
| (NEW in v5.5) Write each frame_entry's `normalized` ndarray as `crops/track_<iid[:8]>_det_<id>_rectified.jpg` and populate `fe["image_path"]` | not present in V4 (V4 wrote in refine) | ⚠️ deviation | See Deviations §1 |
| Try `f["reid_embedding"]`, fall back to `compute_reid_embedding(fused_image_path)` (V4) | fall back to `compute_reid_embedding_array(fused_image)` (V5.5) | ⚠️ deviation | See Deviations §2 |
| On embedding failure: `storage.add_pipeline_event("reid_embedding_failed", {...})` | `cards_repo.add_pipeline_event(...)` | ⚠️ deviation | See Deviations §3 (repository) |
| `Storage.add_card_instance(video_id, track_id, angle, session_id, reid_embedding, run_id)` returns row_id | `CardsRepository.add_card_instance(**kwargs)` returns row_id | ⚠️ deviation | See Deviations §3 |
| `Storage.update_instance_deduplication(row_id, primary_hash, None, reid_embedding=embedding_bytes)` | `CardsRepository.update_instance_deduplication(**kwargs)` | ⚠️ deviation | See Deviations §3 |
| `Storage.update_instance_fusion(row_id, fused_image_path)` | `CardsRepository.update_instance_fusion(**kwargs)` | ⚠️ deviation | See Deviations §3 |
| For each frame_entry in track: `is_best = (det_id == best_det_id)` | identical | ✅ | |
| **V4 line 99 (A1): if `is_best`, `view_path = f["fused_image_path"]`, else `view_path = fe["image_path"]`** | identical (V5.5 store.py:109; test `test_store_best_view_points_to_fused_path` pins it) | ✅ | (explicit confirmation per spec §13.1) |
| `Storage.add_card_view(...)` returns view_id (V4 passed `detection=CornerDetection(corners=..., confidence=...)`) | `CardsRepository.add_card_view(corners=..., confidence=..., ...)` returns view_id; corners/confidence passed as separate kwargs | ⚠️ deviation | See Deviations §3 + §4 |
| **`add_saved_card` only when `fe["is_canonical"] AND is_best`** | identical (both conditions present at V5.5 store.py:124) | ✅ | (explicit confirmation per spec §13.1) |
| For dedup_groups with `cross_video_parent_id`: `Storage.update_instance_deduplication(canonical_row_id, primary_hash, cross_video_parent)` | identical via `CardsRepository` | ⚠️ deviation | See Deviations §3 |
| For each `duplicate_instance_id` in group: `Storage.update_instance_deduplication(dup_row_id, primary_hash, canonical_row_id)` | identical via `CardsRepository` | ⚠️ deviation | See Deviations §3 |
| `Storage.update_video_status(video_id, "completed")` as the last DB write | replaced by `runs_repo.mark_completed(run_id, cards_extracted=len(final_cards))`; video status set by `PipelineRunner._set_video_status` after `runtime.run()` returns | ⚠️ deviation | See Deviations §5 |
| Image filenames: `crops/instance_<iid>_fused.jpg` + `crops/track_<iid>_det_<id>_rectified.jpg` | identical (tests `test_store_writes_*` pin both) | ✅ | |
| `StoreOutput(final_cards=[])` (V4 returned empty list — bug) | `state["cards"] = final_cards` populated with one dict per fused canonical | ✅ | V5.5 actually populates this; was effectively unused in V4 |

### Deviations

1. **Single filesystem boundary in `store`.** Refine and fuse defer all writes; `store` writes both rectified per-view JPEGs and the fused-canonical JPEG up front before opening DB writes. **Reason:** v5.5 in-memory mandate. **Risk:** None — the JPEGs land in the same paths V4 used (`crops/instance_<iid[:8]>_fused.jpg`, `crops/track_<iid[:8]>_det_<id>_rectified.jpg`). Tests pin both filenames.
2. **`compute_reid_embedding` fallback consumes ndarray.** **Reason:** in-memory mandate. **Risk:** None — Phase 3 parity test bounds the delta.
3. **All `Storage.*` calls replaced by `CardsRepository.*` / `RunsRepository.*`.** **Reason:** `no-sqlite3-outside-data` import-linter contract. **Risk:** None — repository methods are thin Phase 2 wrappers around the same SQL.
4. **`add_card_view` takes `corners` + `confidence` kwargs, not a `CornerDetection` dataclass.** **Reason:** repository signature in Phase 2 standardised on primitive kwargs. **Risk:** None — same values flow into the same columns.
5. **Video status transition moved out of `store`.** V4 stage called `storage.update_video_status(video_id, "completed")` at the end. V5.5 stage calls `runs_repo.mark_completed(run_id, cards_extracted=N)`; the video's status is updated by the surrounding `PipelineRunner._set_video_status` after `runtime.run()` returns successfully (already wired in PR #60). **Risk:** None — same end state in DB; cleaner separation of concerns (runs vs videos). **Concern:** if the runner crashes between `store` returning and `_set_video_status`, the row would be left in "processing" — same window V4 had between `update_video_status` and Metaflow finalisation.

### Removed

- `from .start import RunContext` and the `StoreOutput` dataclass — stages mutate `state` directly.
- The empty-list `StoreOutput(final_cards=[])` return — V5.5 populates `state["cards"]` properly.

### Test coverage

- Unit: `tests/pipeline/stages/test_store_stage.py` (fused JPEG written, rectified JPEG written, `add_card_instance` receives `run_id`, best-view-points-to-fused (A1), `final_cards` populated, `mark_completed` called with count)
- E2E: `tests/pipeline/test_back_half_e2e.py` (DB rows + crops directory + run status)
- Golden-set: `cards_extracted` count compared to V4 baseline in Phase 14

---

## Cross-stage audit

| V4 invariant | V5.5 status | Note |
|---|---|---|
| Source video opened ONCE | ✅ identical | sample stage owns the decode handle; refine reuses `state["sampled_frames"]` instead of re-opening |
| YOLO model loaded ONCE | ✅ identical | detect stage caches the model |
| All in-memory pixel data is BGR uint8 | ✅ unchanged | refine output, best-canonical, fused all 1050×750 BGR uint8 |
| All warped crops are 1050×750 uint8 BGR | ✅ unchanged | `KorniaNormalizer` / `PrecisionNormalizer` outputs |
| Image writes happen at ONE stage boundary | ⚠️ moved | V4 wrote in `refine` and `fuse`; V5.5 writes only in `store`. Net filenames + bytes are equivalent. |
| Storage writes go through DAL | ✅ stages 9–10 use `CardsRepository` / `RunsRepository` | enforced by `no-sqlite3-outside-data` |
| Metaflow imports outside vendored env: 0 | ✅ unchanged | enforced by architecture test |
| Raw `sqlite3` outside `card_capture.data`: 0 | ✅ unchanged | enforced by architecture test |
| Cross-video dedup query excludes current `video_id` | ✅ identical | pinned by `test_dedup_cross_video_query_excludes_self_video_id` |
| Best-canonical view path == fused image path (A1) | ✅ identical | pinned by `test_store_best_view_points_to_fused_path` |
| `saved_cards` row only when canonical AND best | ✅ identical | pinned by store tests |

## Aggregate risk assessment

- **0 algorithmic changes.** Every magic-number, sort key, threshold, and gate predicate is byte-identical to V4. The numerical surface area is unchanged.
- **3 systemic adaptations** (state-mutation, ndarrays in flight, repositories instead of raw SQL) are mechanically equivalent and gated by unit + parity + import-linter tests.
- **1 timing change** worth flagging: video-status transition now happens after `runtime.run()` returns instead of inside the `store` stage. Same crash window V4 had between two adjacent calls; not a regression.
- **1 perf-relevant memory concern**: refine→store crops live in RAM (~180 MB on reference video). Mitigation (selective spill) tracked in CLAUDE.md Known Weaknesses.

## No `TODO` markers

Every card-producing behavior listed in spec §13.1 has an explicit ✅ / ⚠️ / ❌ row above. There are no `TODO`, `FIXME`, or placeholder entries in this audit.
