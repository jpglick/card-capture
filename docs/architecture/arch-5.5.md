# Card Capture — Pipeline Architecture v5.5

> Scope: extract clean, deduplicated 750×1050 stills of trading cards from a 4K
> portrait workspace video (cards held / placed in front of a roughly fixed
> camera). This document describes the **v5.5 in-process runtime** as it lives in
> `src/card_capture/pipeline/` and is exhaustive enough to be critiqued by a
> computer-vision practitioner. It is descriptive, not aspirational — every
> module path, threshold, and short-circuit named here exists in the merged code
> as of `v5.5.0`.
>
> v5.5 replaces the v4.1 Metaflow flow (one OS subprocess per stage) with a
> single-process runtime: stages are direct function calls that share decoded
> frames, loaded models, and GPU-resident tensors in memory. The algorithmic
> work per stage is largely carried over from v4.1; the orchestration, the
> data-access layer, and the device path are what changed. For prior versions
> see [`docs/archive/`](../archive/).

---

## 0. Pipeline at a Glance

```
                       LocalPipelineRuntime.run(request)
                       ── shared `state: dict` threaded through 10 stages ──
   video.mov ─┐
              ▼
   ┌──────────────────────────────────┐
   │ 1. sample   stages/sample.py      │ StrideSampler → FrameProducer
   │             (uniform stride, 3fps)│ (background decode thread starts)
   └───────────────┬──────────────────┘ state["frame_producer"], ["sampler"]
                   ▼
   ┌──────────────────────────────────┐
   │ 2. detect   stages/detect.py      │ drains producer, batches 16,
   │             YOLOv8-OBB, batched   │ YOLO corner detection
   └───────────────┬──────────────────┘ state["detections"] (rows)
                   ▼
   ┌──────────────────────────────────┐
   │ 3. novelty  stages/novelty.py     │ per-quad novelty score vs.
   │             (annotate only)       │ background model (Lab space)
   └───────────────┬──────────────────┘ state["novelty_scored_detections"]
                   ▼
   ┌──────────────────────────────────┐
   │ 4. track    stages/track.py       │ ByteTrack (default) / BoT-SORT;
   │             session-aware         │ detections → TrackState + dicts
   └───────────────┬──────────────────┘ state["tracks"], ["tracks_data"]
                   ▼
   ┌──────────────────────────────────┐
   │ 5. refine   stages/refine.py      │ Kornia warp → 750×1050, quality
   │             GPU warp + scoring    │ score, pHash, glare, ReID embed
   └───────────────┬──────────────────┘ state["refined_tracks"]
                   ▼
   ┌──────────────────────────────────┐
   │ 6. score    stages/score.py       │ per-track median aggregation +
   │             prune gates           │ 3 prune gates (mark, don't delete)
   └───────────────┬──────────────────┘ state["scored_tracks"], pruned ids
                   ▼
   ┌──────────────────────────────────┐
   │ 7. resolve  stages/resolve.py     │ Front/Back classifier + session
   │             F/B + identity        │ identity resolution
   └───────────────┬──────────────────┘ state["prepared_tracks"]
                   ▼
   ┌──────────────────────────────────┐
   │ 8. fuse     stages/fuse.py        │ lighting-diverse median fusion
   │             (passthrough @ 1frame)│ (single-frame passthrough default)
   └───────────────┬──────────────────┘ state["fused_canonicals"]
                   ▼
   ┌──────────────────────────────────┐
   │ 9. dedup    stages/dedup.py       │ intra-run + cross-video dedup:
   │             ReID + pHash          │ DINOv2 cosine, pHash Hamming
   └───────────────┬──────────────────┘ state["dedup_groups"], ["final_cards"]
                   ▼
   ┌──────────────────────────────────┐
   │ 10. store   stages/store.py       │ write crops to disk; persist
   │             repositories + Writer │ instances/views/links via DAL
   └──────────────────────────────────┘ state["cards"], RunManifest
```

Orchestration lives in `LocalPipelineRuntime.run()`
(`src/card_capture/pipeline/runtime_local.py`). The runtime owns the SQLite
`Writer` lifecycle and a single mutable `state: dict` that every stage reads
from and writes to. Stages are registered in the `_STAGES` tuple and invoked in
order; each is timed (`StageTiming`) and a stage exception is recorded as a
`stage_failed:<name>` contract violation before re-raising. There is **no**
inter-stage pickling and **no** subprocess boundary — the central performance
win over v4.1 (the ~52 s re-decode and the ~4–6 min fuse fan-out are gone).

The stage numbering above (1–10) is the execution order. Internally the modules
also carry their historical "Stage N" labels from v4.1 in their docstrings
(e.g. `store.py` is "Stage 10b"); this document uses execution order.

---

## 1. Runtime Model

### 1.1 `LocalPipelineRuntime`

**Module:** `pipeline/runtime_local.py`.

This is the only runtime used in production. It is constructed by:

- `src/card_capture/cli.py` (`card-capture process`),
- `app/services/pipeline_runner.py` (the web UI, per-run),
- `app/services/training_service.py` (training data generation),
- `src/card_capture/platforms/local.py :: LocalRunner` (the `PipelineRunner` adapter).

`run(request)` does the following:

1. Resolves a `db_path` — the caller's explicit `request.db_path` if given (UI,
   training), else `<output_root>/cards.sqlite` (older test callers).
2. Constructs and `start()`s a `Writer` (single-writer DB thread — see §12).
3. Builds the `state` dict, injecting `request`, `video_id`, `output_root`
   (with the `artifact://local/` prefix stripped), and a `repos` mapping of
   `RunsRepository` / `EventsRepository` / `CardsRepository`, each bound to the
   shared `Writer`.
4. Iterates `_STAGES`, calling `module.run(state, telemetry=...)` for each,
   recording a `StageTiming`. On any stage exception it appends a
   `stage_failed:<stage>` violation, emits a telemetry `contract_violation`, and
   re-raises.
5. In a `finally`, `stop()`s the `Writer` (which joins the worker thread and
   re-raises the first write error, if any).
6. Builds a `RunManifest` (run id, runtime mode, input video, output artifacts,
   card records, stage timings, violations, version `"0.5.5+phase4"`) and wraps
   it in a `PipelineRunResult`.

`LocalPipelineRuntime` is effectively synchronous. `submit()` runs the pipeline
immediately and stashes the result; `wait()` returns it; `cancel()` is a no-op.
The asynchronous `submit/wait/cancel` shape exists to satisfy the
`PipelineRunner` Protocol, not because the local backend is concurrent.

### 1.2 The `state` dict (inter-stage contract)

Stages do not pass typed objects to each other; they communicate through keys in
the shared `state` dict. The important keys, in the order they appear:

| Key | Produced by | Shape / meaning |
|---|---|---|
| `request`, `video_id`, `output_root`, `repos`, `db_path` | runtime | run inputs + injected DAL |
| `sampler`, `frame_producer`, `video_path` | sample | live `StrideSampler` + background decode thread |
| `sampled_frames` | sample (init), detect (filled) | `list[FrameSample]` — decoded frames, reused by refine (no re-decode) |
| `estimated_frame_total` | sample | header-probe estimate for the detect progress bar |
| `detections` | detect | `list[dict]` rows: `detection_id, frame_index, timestamp_ms, w, h, corners, confidence` |
| `bg_model` | novelty | `BackgroundModel` (if background proxies exist) |
| `novelty_scored_detections` | novelty | detect rows + `novelty_score` |
| `tracks` | track | `list[TrackState]` (legacy shape) |
| `tracks_data` | track | `list[dict]` — rich per-candidate dicts (the shape refine consumes) |
| `refined_tracks` | refine | per-track `frame_entries` with `normalized`, `quality_score`, `visual_hash`, `glare_*`, plus `best_canonical_image` + `reid_embedding` |
| `scored_tracks`, `pruned_instance_ids` | score | prune decision per track (median novelty/quality/sharpness) |
| `resolved_sessions`, `prepared_tracks` | resolve | F/B-resolved, session-grouped tracks |
| `fused_canonicals` | fuse | one record per output card with `fused_image`, `primary_hash`, `appearance_vector`, `reid_embedding` |
| `dedup_groups`, `final_cards` | dedup | duplicate groupings + cross-video parents |
| `cards`, `output_artifacts` | store | persisted card summaries → `RunManifest` |

Because `state` is a plain dict, stages are defensively coded: they read with
`.get(key, default)` and tolerate missing upstream keys (e.g. `novelty` falls
back to `novelty_score = 1.0`, `refine` raises a `refine_without_frames`
contract violation if `sampled_frames` is absent).

### 1.3 Runtime modes — what they actually do

`RuntimeMode = Literal["strict_gpu", "cpu_debug", "mixed_compat"]`
(`pipeline/request.py`) is part of the serialized request/manifest contract and
is preserved verbatim across the JSON boundary. **In `LocalPipelineRuntime` the
mode is cosmetic:** it is copied into the `RunManifest` but does **not** change
stage execution or device selection. Device choice is made *inside* the stages
that need a device (`detect`, `refine`) from `config["device"]` (default
`"auto"`), not from `runtime_mode`. The Mac web UI runs `cpu_debug`; this does
not force CPU in the in-process path — it is a label.

### 1.4 `StrictGpuRuntime` (dormant)

**Module:** `runtime/strict_gpu.py`.

A second runtime exists as a **Phase-2 skeleton**. `StrictGpuRuntime.run()`
selects an MPS device (or raises `MissingGpuError`), constructs a `GpuSession`,
and returns an *empty* manifest — it wires **no** stages ("Stage wiring lands in
Phase 3"). Nothing in `app/` or `src/` constructs it; it is exercised only by
its own unit tests. It and `GpuSession` (`runtime/gpu_session.py`) define the
*intended* strict-GPU capability boundary, but the v5.5 in-process stages are
**not** gated through `GpuSession` — the GPU boundary today is enforced
statically (import-linter + the GPU-strict AST scanner), not by a runtime
capability object. The stage package docstring still describes a
`run(input, GpuSession | None)` signature; the real signature is
`run(state: dict, *, telemetry)`.

### 1.5 `PipelineRunner` / `LocalRunner`

**Modules:** `pipeline/runner.py`, `platforms/local.py`.

`PipelineRunner` is a `Protocol` (`submit`/`wait`/`cancel`) with opaque
`PipelineRunHandle` and `PipelineRunStatus` value types. `platforms/local.py ::
LocalRunner` is the only implementation, delegating to `LocalPipelineRuntime`.
The provider abstraction that previously fanned out to RunPod/Beam/vast.ai was
collapsed to this single local path in the v5.5.0 cleanup; `platforms/__init__.py`
now exports only `LocalRunner`.

---

## 2. Threading & GPU Boundary

The in-process runtime is **single-threaded for compute** with two auxiliary
threads:

1. **`frame-producer` (decode) thread.** Started by the `sample` stage via
   `sampler/frame_producer.py :: FrameProducer.start()`. It runs the sampler's
   `.sample()` generator (ffmpeg/PyAV or OpenCV decode) on a daemon thread and
   pushes each `FrameSample` onto a bounded `queue.Queue(maxsize=32)`. The
   `detect` stage drains this queue (`for frame in producer`) so decode of later
   frames overlaps YOLO inference of earlier ones. Producer exceptions are
   captured and re-raised to the consumer when the queue drains; `detect` always
   calls `producer.stop()` in a `finally` to guarantee the decode thread ends.
2. **`card-capture-writer` thread.** The `Writer`'s single worker thread, the
   only holder of a write connection to SQLite (§12).

Everything else — YOLO inference (`detect`), the Kornia warp and quality scoring
(`refine`), F/B classification (`resolve`), fusion (`fuse`), and dedup — runs
**on the calling (main) thread**. There is no separate GPU "worker" thread in
the in-process path; the MPS/CoreML calls happen inline on the main thread. (The
"_worker thread" mandate in older CLAUDE.md text refers to the v4 subprocess
model and does not match v5.5.)

**Where GPU/MPS work happens:** model inference in `detect`
(`CardcaptorUltralyticsDetector`, CoreML/MPS), the perspective warp in `refine`
(`KorniaNormalizer` on MPS, with a CPU `PrecisionNormalizer` fallback), the
DINOv2 ReID embedder in `refine`/`store`, and the F/B classifier in `resolve`.
These are confined to those four stages by code organization and by the
import-linter `strict-gpu-no-image-io` and layering contracts (§12.3).

---

## 3. Stage 1 — Sample (uniform-stride producer)

**Module:** `pipeline/stages/sample.py`; sampler in `sampler/__init__.py ::
StrideSampler`; thread in `sampler/frame_producer.py`.

> **Divergence from v4.1 / CLAUDE.md.** The in-process pipeline samples with
> **`StrideSampler`**, a *uniform-stride* sampler, **not** the two-pass
> `AdaptivePresenceSampler`. The adaptive sampler (presence classifier, valley
> splits, presence windows, background-proxy heap) still exists in
> `sampler/__init__.py` but is wired only into the `card-capture sampler
> sessions` diagnostic subcommand (`cli.py`), not into `LocalPipelineRuntime`.
> Consequently the v4.1 knobs `fast_scan_fps`, `valley_drop_ratio`,
> `delta_spike_ratio`, presence windows, and forced valley splits **do not
> affect** a default `card-capture process` run.

`run()`:

1. Strips `artifact://local/` from `request.input_video` to get a filesystem path.
2. Constructs `StrideSampler(video_path=...)` (default `target_yolo_fps = 3.0`,
   `reader_backend = "auto"`, `pixel_format = "bgr24"`).
3. Starts a `FrameProducer(sampler)` and blocks on `wait_first(timeout=60)` so
   this stage's timing reflects decode startup; all remaining frames overlap
   `detect`.
4. Initializes `state["sampled_frames"] = []` (filled by `detect`) and a cheap
   header-probe `estimated_frame_total` (for the progress bar).

`StrideSampler.sample()` probes the real source FPS, computes
`stride = max(1, round(source_fps / target_yolo_fps))`, and yields every
`stride`-th frame as a `FrameSample(frame_index, timestamp_ms, image, w, h)`.
The **first** decoded frame is captured as the sole `background_proxies` entry
(used by the novelty stage). `last_inter_window_gaps_frames = [stride]` — a
single fixed gap, since there are no presence windows.

**Decode backend.** `reader_backend = "auto"` resolves (in
`ingestion._resolve_reader_backend`) to `"decord"` if the `decord` package is
importable, else `"pyav"`. `decord` is not packaged for Apple Silicon, so on the
target hardware **auto resolves to PyAV** (`_sample_with_pyav`, software HEVC
decode — fast on Apple Silicon's multithreaded decoder). The `"decord"` branch
is itself aliased to the OpenCV path. PyAV (`av`) is therefore a **runtime
dependency of the default pipeline**, shipped in the `legacy_tracking` extra.

**Critique surface.** Uniform stride at 3 fps trades the v4.1 adaptive sampler's
selectivity for simplicity and speed. It cannot place extra candidates around a
hand-swap (no valley splits) and uses a single first-frame background proxy
rather than a heap of lowest-presence frames, so the novelty model is weaker
than v4.1's. Cards shown for less than ~1/3 s can be missed entirely.

---

## 4. Stage 2 — Detect (YOLOv8-OBB corner detection)

**Module:** `pipeline/stages/detect.py`; detector in `detectors.py ::
CardcaptorUltralyticsDetector` (test double: `FakeCardDetector`).

`run()`:

1. **Model load (once).** On first entry it reads `config["detector"]` (default
   `"fake"`). `"fake"` → `FakeCardDetector`; any other value →
   `CardcaptorUltralyticsDetector(confidence_threshold=config["corner_confidence"]
   (0.5), detection_width=config["detection_width"] (640), device=...)`. The
   device is resolved by `probe_torch_device_status(config["device"] ("auto"))`
   → `.resolved` (MPS or CPU; §13). The loaded detector is cached in
   `state["yolo_model"]`.
   > Note: the *request-config* default detector is `"fake"`. Production callers
   > (CLI / UI) explicitly set a real detector; tests rely on the fake.
2. **Drain + batch.** It iterates the `FrameProducer` (overlapping decode with
   inference), appends each frame to `state["sampled_frames"]` (so `refine` can
   reuse them with no re-decode), accumulates a `batch` of `batch_size = 16`
   (hard-coded), and calls `detector.detect_batch(packets, conf)`. Progress is
   reported against `estimated_frame_total`.
3. **Rows out.** Each surviving detection becomes a dict
   `{detection_id, frame_index, timestamp_ms, width, height, corners,
   confidence, triage_metrics}` in `state["detections"]`. `detection_id` is a
   1-based running index.

The OBB model, weights, postprocessing (conf filter, polygon-length/area
sanity, scale-back to source coordinates) are carried over from v4.1 §3 —
see that document for the model card and critique. The CUDA/TensorRT fast-path
that v4.1 had in `_load_model` was removed in the v5.5.0 cleanup; only the
CoreML (MPS) and `.pt` paths remain.

---

## 5. Stage 3 — Novelty (per-quad background novelty, annotate-only)

**Module:** `pipeline/stages/novelty.py`; model in
`presence/background_novelty.py :: BackgroundModel`, `quad_novelty`.

> **Divergence from v4.1 / CLAUDE.md.** This stage **only annotates** each
> detection with a `novelty_score`; it does **not** drop anything. The actual
> empty-workspace pruning happens later in `score` (§7). CLAUDE.md's "drops
> empty stands" describes the *effect across two stages*, not this one.

`run()`:

1. Builds a `BackgroundModel.from_frames(sampler.background_proxies)` if any
   proxies exist (the single first frame from `StrideSampler`).
2. For each detection row, if a background model and source frame are available,
   computes `quad_novelty(source_frame, corners, bg_model, color_space="lab",
   lab_weights=(1.0, 0.5, 0.5))` and stores it as `novelty_score`. Missing model
   / frame ⇒ `novelty_score = 1.0` (fully novel). Rows that already carry a
   `novelty_score` (an MPS fast path) are passed through untouched.

The per-quad novelty is the mean Lab-space difference between the card's
interior quad and the background model, L weighted 1.0 and a/b weighted 0.5
(v5.5 moved from v4.1's grayscale-only difference to a luminance-weighted Lab
difference, making the gate less sensitive to pure lighting shifts and more
sensitive to chroma). Output: `state["novelty_scored_detections"]`.

---

## 6. Stage 4 — Track (session-aware tracking)

**Module:** `pipeline/stages/track.py`; adapters in
`tracking/bytetrack_adapter.py`, `tracking/botsort_adapter.py`.

`run()` reads `config["tracker_backend"]` (**default `"bytetrack"`**, changed
from v4.1's `botsort`). Common kwargs: `min_track_length = 3`,
`track_activation_threshold = 0.25`, `lost_track_buffer = 30`,
`minimum_matching_threshold = 0.8`. The BoT-SORT path additionally takes
`reid_distance_threshold = 0.6`.

The chosen adapter's `assign(detections, frames)` returns
`list[TrackState]` (→ `state["tracks"]`). The stage then builds the richer
**`state["tracks_data"]`**: one dict per track with an `instance_id`,
`track_id`, `session_id`, `first_frame_index`, and a list of per-candidate dicts
that *enrich* each `ScoredCandidate` with the originating detection's
`width/height/corners/confidence/novelty_score/timestamp` — so `refine` can warp
and score without re-decoding or re-looking-up. `angle` starts `"Unknown"`.

> Session semantics, BoT-SORT/ByteTrack internals, and the four v4.1
> session-reset signals are carried over from v4.1 §5. ByteTrack (the v5.5
> default) has no ReID and no appearance-based `pending_splits`. See the
> known-weakness note in §16 about single-track collapse when cards are shown
> one at a time.

---

## 7. Stage 5 — Refine (GPU warp, quality scoring, canonical selection)

**Module:** `pipeline/stages/refine.py`; helpers in `gpu_refinement.py ::
KorniaNormalizer`, `cropper.py :: PrecisionNormalizer`, `scoring.py ::
QualityScorer`, `deduplicator.py :: VisualDeduplicator`, `fuser.py ::
find_glare_centroid`, `pipeline_utils._select_canonical_entries`, and the
DINOv2 `ml.models.dino_embedder.DinoEmbedder` (cached singleton, `vits14`).

This is the heaviest stage. For each track in `tracks_data`:

1. **Candidate cap.** Sort the track's candidates by `score_total` desc, take
   the top **8**.
2. **Batched warp.** If `config["use_kornia"]` (default `True`), build a batch of
   `(raw_frame, corners)` (frames pulled from `state["sampled_frames"]` by
   `frame_index` — never re-decoded) and call
   `KorniaNormalizer(width, height, device=config["kornia_device"] or
   config["device"] or "auto").warp_canonical_batch(..., rotate_180=...)`,
   producing 750×1050 crops on the GPU. If Kornia construction or the warp
   fails, the stage falls back per-candidate to the CPU
   `PrecisionNormalizer.normalize(...)`.
3. **Per-crop telemetry.** For each candidate it builds a `frame_entry` with the
   `normalized` crop, a `QualityScore` (`QualityScorer.score(normalized,
   confidence, novelty=...)` — the 7-component weighted score from v4.1 §7), a
   pHash (`VisualDeduplicator.compute_phash`), glare centroid + glare mask +
   Laplacian heatmap (zlib-compressed), sharpness, etc.
4. **Canonical set.** `_select_canonical_entries(frame_entries, deduplicator)`
   chooses the canonical views (lighting-diverse selection from v4.1 §9.1);
   their `detection_id`s are marked `is_canonical`. The single best canonical
   (max `quality_score`) image is stashed as `best_canonical_image`.
5. **ReID embedding.** A DINOv2 embedding of the best canonical image
   (`embedder.embed_array(img)`, array variant — no temp file) is stored as
   `reid_embedding`.
6. **Track telemetry rows.** If a `cards` repo is present, per-canonical
   `add_track_telemetry(video_id, instance_id, frame_index, area, aspect, cx,
   cy)` rows are written (through the `Writer`).

Output: `state["refined_tracks"]` (per-track `frame_entries`,
`canonical_detection_ids`, `best_canonical_detection_id`, `best_canonical_image`,
`reid_embedding`).

> v5.5 substitutions vs v4.1 `pipeline/steps/refine.py`:
> `decoded_images[frame_index]` → `state["sampled_frames"]` lookup;
> `cv2.imwrite(...rectified.jpg)` → in-memory `frame_entry["normalized"]`;
> `embedder.embed_image(path)` → `embedder.embed_array(img)`.

---

## 8. Stage 6 — Score (median aggregation + prune gates)

**Module:** `pipeline/stages/score.py`.

This stage does **not** re-score crops (that happened in `refine`). It computes
per-track medians and applies three independent prune gates, **marking** tracks
`pruned` rather than deleting them (deletion/skipping is the consumers' job).

For each track it computes `median_novelty`, `median_quality`,
`median_sharpness` over the track's `frame_entries`, then:

1. **Novelty gate** — active only when the per-video novelty distribution is
   discriminative: `_novelty_gate_useful(scores)` requires ≥ 5 detections,
   `std > 0.15`, and `min < 0.35`. When useful, the threshold is the midpoint of
   the **largest gap** between sorted per-track median novelties, capped at
   `config["novelty_floor"]` (default **0.30**). A track is pruned if its median
   novelty is below this adaptive threshold.
2. **Confidence floor** — active when `config["track_confidence_floor"]`
   (default **0.60**) `> 0`; prunes tracks whose `median_quality` is below it.
3. **Transparent-stand gate** — active when a `bg_model` exists and
   `config["stand_novelty_max"]` (default **0.35**) `> 0`; prunes tracks that are
   *both* low-novelty (`< stand_novelty_max`) and low-sharpness
   (`median_sharpness < config["stand_sharpness_max"]`, default **0.30**) — i.e.
   an empty acrylic stand that fooled YOLO.

Output: `state["scored_tracks"]` (each annotated with `pruned` + the medians)
and `state["pruned_instance_ids"]`.

---

## 9. Stage 7 — Resolve (Front/Back + session identity)

**Module:** `pipeline/stages/resolve.py`; helper
`pipeline_utils._resolve_session_tracks`; classifier
`ml.inference.fb_predict.FBPredictor` (loaded from
`models/presence_classifier.pt` if present, cached singleton).

`run()` groups `refined_tracks` by `session_id`. For each track's
`best_canonical_image`, if `config["use_fb_classifier"]` (default `True`) and the
classifier is available, `FBPredictor.predict_array(img)` yields a
`(label, conf)` mapped into `fb_probs = [front_prob, back_prob]`. Then
`_resolve_session_tracks(tracks, deduplicator)` performs the v4.1 §8 resolution:
the longest track is Front; others are promoted to Back only if their pHash is
within the same-card Hamming bound, otherwise they remain separate Front cards
(two-card recovery). Output: `state["resolved_sessions"]`,
`state["prepared_tracks"]`.

> If the classifier weights are missing, the system falls back to the
> longest-track heuristic (a known weakness — see §16).

---

## 10. Stage 8 — Fuse (lighting-diverse median fusion)

**Module:** `pipeline/stages/fuse.py`; algorithm in `fuser.py :: MultiFrameFuser`.

For each prepared track, the canonical `frame_entries` are gathered. Behaviour
depends on `config["fusion_target_frames"]` (**default 1**):

- **`fusion_target_frames <= 1` or a single canonical** → **single-frame
  passthrough**: `fused_image = best_canonical_image`. **This is the default**,
  so out of the box no median fusion occurs — the output is the single best
  canonical crop.
- **Otherwise** → `MultiFrameFuser().fuse(images, foil_threshold=...)`:
  per-pixel median across the lighting-diverse canonical crops, with optional
  foil-aware handling when `config["enable_foil_aware_fusion"]` (default `True`,
  `foil_threshold = 50.0`). Any fusion error falls back to the best canonical.

Output: `state["fused_canonicals"]` — one record per output card with
`fused_image`, `primary_hash`, `quality_score`, `side_score`,
`appearance_vector`, `reid_embedding`, `angle`, `session_id`.

> v5.5 change: this was a Metaflow `foreach` (one subprocess per track,
> ~4–6 min overhead on the reference video); it is now a plain in-process loop.
> The fusion algorithm itself is unchanged from v4.1 §9.2.

---

## 11. Stage 9 — Dedup (intra-run + cross-video)

**Module:** `pipeline/stages/dedup.py`; `deduplicator.py :: VisualDeduplicator`;
DAL read `CardsRepository.find_embeddings_excluding_video`.

Constants (identical to v4.1): `SAME_CARD_EMB_THRESHOLD = 0.15` (DINOv2 cosine
distance), `SAME_CARD_HAMMING_MAX = 8` (pHash Hamming fallback).

1. **Cross-video prefetch.** `CardsRepository.find_embeddings_excluding_video(
   video_id)` returns `(row_id, embedding_bytes)` for all *other* videos' cards
   — pulled once. (The raw `Storage._connect()` query in v4.1 became this
   repository call so no raw SQL leaves `card_capture.data`.)
2. **Intra-run.** For each pair of fused canonicals, mark duplicates if the
   DINOv2 cosine distance `< 0.15`, else fall back to pHash Hamming `<= 8`.
   Duplicates are grouped under the first-seen canonical.
3. **Cross-video.** For each canonical with an embedding, find the nearest other
   video's card; if cosine distance `< 0.15`, record it as
   `cross_video_parent_id`.

Output: `state["dedup_groups"]` and `state["final_cards"]` (= the fused
canonicals, passed to `store`).

---

## 12. Stage 10 — Store (disk + DAL persistence)

**Module:** `pipeline/stages/store.py`; repositories in
`data/repositories/{cards,runs,events}.py`.

Image writes happen **here and only here** (the in-memory mandate). `run()`:

1. Requires a real `video_id` (raises `store_without_video_id` otherwise).
2. Writes each fused canonical to `<output_root>/crops/instance_<iid8>_fused.jpg`
   and each track view to `track_<iid8>_det_<id>_rectified.jpg`.
3. Persists per card via `CardsRepository`:
   `add_card_instance(video_id, track_id, angle, session_id, reid_embedding,
   run_id)` → `row_id`; then `update_instance_deduplication`,
   `update_instance_fusion`, and per-view `add_card_view(...)` /
   `add_saved_card(...)` for the best canonical. If a track has no
   `reid_embedding`, it recomputes one from the fused image
   (`ml.embeddings.compute_reid_embedding_array`) so the row is never NULL
   (needed by future cross-video dedup).
4. Persists dedup links (intra-run duplicate → canonical, and cross-video
   parents).
5. `RunsRepository.mark_completed(run_id, cards_extracted=len(final_cards))`.

Output: `state["cards"]`, `state["output_artifacts"]`.

---

## 13. Data-Access Layer (single-writer)

**Package:** `src/card_capture/data/`.

### 13.1 Connections

`data/connection.py`:

- `open_connection(db_path, read_only=False)` opens a URI connection
  (`mode=ro|rwc`), `isolation_level=None` (autocommit), `row_factory =
  sqlite3.Row`. For writers it sets `PRAGMA journal_mode=WAL` (with a small
  retry loop for the WAL-setup race), `PRAGMA busy_timeout=5000`, and
  `PRAGMA foreign_keys=ON`.
- `read_connection(db_path)` is a read-only context manager.

### 13.2 The `Writer` (single-writer discipline)

`data/writer.py :: Writer` enforces "WAL allows many readers, one writer." All
writes — from the pipeline runtime, the FastAPI handlers, and the harness — are
serialized through one `Writer`:

- `start()` spawns a single daemon thread `card-capture-writer` that owns the
  **only** write connection (opened inside the worker via `open_connection`).
- `submit(Write(sql, params))` is fire-and-forget; `submit_returning(Write)`
  returns a `Future[int]` resolved with `cursor.lastrowid` (for autoincrement
  reads); `flush()` blocks until the queue drains; `serialize()` is a lock for
  callers that must do a direct synchronous write.
- **Poisoning:** the first fire-and-forget write that raises records the error
  under a lock and switches the worker to *drain-and-fail* mode — subsequent
  fire-and-forget writes are discarded and returning-Futures fail with the
  recorded error (as `__cause__` of `WriterPoisonedError`) so no caller blocks
  forever. `stop()` joins the worker and re-raises the first error. This is why
  `LocalPipelineRuntime.run()` stops the writer in a `finally`.

Repositories (`RunsRepository`, `EventsRepository`, `CardsRepository`, and the
config/labeling/ml/telemetry/training/videos/batch repos) take a `Writer` + a
`db_path`: they read through their own short-lived `read_connection`s and route
every mutation through the shared `Writer`.

### 13.3 Architectural enforcement (`.importlinter`)

Five contracts (root packages `card_capture`, `app`):

| Contract | Enforces |
|---|---|
| `no-sqlite3-outside-data` | `card_capture.runtime` may not import `sqlite3` (writes go through the DAL) |
| `no-provider-sdks` | `runpod` / `beam` not imported anywhere (cloud removed) |
| `strict-gpu-no-image-io` | `runtime.strict_gpu` may not import `PIL` / `cv2` |
| `layered` | within `card_capture`, layer order `runtime → pipeline → data` |
| `no-metaflow` | `metaflow` not imported anywhere (orchestration is in-process) |

A companion architecture test (`tests/architecture/test_metaflow_absent.py`) and
the GPU-strict AST scanner back these up. `import-linter` runs in CI and as part
of the local gate.

---

## 14. Device Path (MPS → CPU)

CUDA was fully removed in v5.5.0; the only accelerator is Apple Silicon MPS.

- `detectors.py :: probe_torch_device_status(requested="auto")` resolves **MPS →
  CPU**. It still returns the `TorchDeviceStatus` dataclass tests import, but
  `cuda_available` / `cuda_built` are hard-coded `False` (the field is retained
  only for the serialized contract). `requested="mps"` with MPS unavailable
  yields a `reason="mps_unavailable"`.
- `runtime/strict_gpu.py :: _select_device()` returns `mps` or raises
  `MissingGpuError` — there is no CPU fallback on the strict route.
- `GpuSession.__post_init__` rejects a CPU device when `strict=True`.
- `detect` resolves its device via `probe_torch_device_status`; `refine` builds
  `KorniaNormalizer` on `config["kornia_device"]`/`config["device"]`/`"auto"`
  and falls back to the CPU `PrecisionNormalizer` if Kornia construction fails
  for the requested device.
- The CoreML detector path is the MPS fast path; the `.pt` path is the fallback.

---

## 15. Configuration Surface (effective v5.5 defaults)

v5.5 has no single `PipelineConfig` object on the hot path. The runtime passes
`request.config` (a plain `Mapping`) and each stage reads it with inline
`.get(key, default)`. The de-facto defaults that actually run are:

```
# sample (StrideSampler)
target_yolo_fps            = 3.0     # uniform stride sampling rate
reader_backend             = "auto"  # → pyav on Apple Silicon (decord absent)
FrameProducer maxsize      = 32      # bounded decode queue

# detect
detector                   = "fake"  # request-config default; prod sets a real one
corner_confidence          = 0.5     # YOLO conf gate
detection_width            = 640
device                     = "auto"  # → MPS → CPU
batch_size                 = 16      # hard-coded in detect.py

# novelty
quad_novelty color_space   = "lab"   # lab_weights = (1.0, 0.5, 0.5)

# track
tracker_backend            = "bytetrack"
min_track_length           = 3
track_activation_threshold = 0.25
lost_track_buffer          = 30
minimum_matching_threshold = 0.8
reid_distance_threshold    = 0.6     # BoT-SORT only

# refine
use_kornia                 = True
rotate_180                 = False   # flip for upside-down camera
top candidates per track   = 8       # hard-coded
canvas                     = 750 × 1050

# score (prune gates)
novelty_floor              = 0.30
track_confidence_floor     = 0.60
stand_novelty_max          = 0.35
stand_sharpness_max        = 0.30

# resolve
use_fb_classifier          = True

# fuse
fusion_target_frames       = 1       # single-frame passthrough by default
enable_foil_aware_fusion   = True
foil_threshold             = 50.0

# dedup (module constants)
SAME_CARD_EMB_THRESHOLD    = 0.15    # DINOv2 cosine
SAME_CARD_HAMMING_MAX      = 8       # pHash Hamming
```

> A legacy `config.py :: PipelineConfig` and the `AdaptivePresenceSampler`
> constructor still carry the v4.1 knobs (`fast_scan_fps = 12.0`,
> `valley_drop_ratio = 0.40`, `scan_width = 192`, etc.); those drive the
> `card-capture sampler sessions` diagnostic, **not** the in-process pipeline.
> CLAUDE.md's pointer to `pipeline/request.py` for these knobs is stale —
> `request.py` defines only the serialized contracts below.

---

## 16. Serialized Contracts

**Module:** `pipeline/request.py`. Everything crossing the runtime ↔ runner ↔
app ↔ harness boundary must be JSON-serializable (no tensors, model objects,
open handles).

- **`PipelineRunRequest`** (frozen): `run_id`, `input_video` (`artifact://`),
  `output_root` (`artifact://`), `runtime_mode` (`strict_gpu | cpu_debug |
  mixed_compat`), `config` (Mapping), and optional `db_path`, `video_id`,
  `config_preset`. `to_dict` / `from_dict` round-trip it.
- **`RunManifest`** (frozen): `run_id`, `runtime_mode`, `input_video`,
  `output_artifacts`, `cards: list[CardRecord]`, `stage_timings:
  list[StageTiming]`, `contract_violations: list[ContractViolation]`,
  `version`, `metadata`. `to_json` / `from_json` round-trip it
  (`json.dumps(..., sort_keys=True)`).
- **`CardRecord`**: `card_instance_id`, `front_crop`, optional `back_crop`,
  `quality`. **`StageTiming`**: `stage`, `elapsed_ms`, `metadata`.
  **`ContractViolation`**: `code`, `metadata`. **`PipelineRunResult`**:
  `manifest` + optional `manifest_path`.

The `RuntimeMode` literal set is a frozen contract: `strict_gpu` is retained
even though the in-process path ignores it, so older serialized requests/manifests
keep round-tripping.

---

## 17. What Changed from v4.1

| Concern | v4.1 | v5.5 |
|---|---|---|
| Orchestration | Metaflow flow, one subprocess per stage | `LocalPipelineRuntime`, in-process direct calls over a shared `state` dict |
| Inter-stage transport | datastore pickling + queues across processes | in-memory dict; frames/models/tensors reused |
| Sampler (pipeline) | `AdaptivePresenceSampler` (two-pass, valley splits) | `StrideSampler` (uniform 3 fps); adaptive sampler survives only as a diagnostic |
| Default tracker | BoT-SORT | ByteTrack |
| Novelty gate | grayscale mean diff | Lab-space weighted diff; gating moved to `score` with adaptive threshold |
| Fusion | per-track Metaflow `foreach` | in-process loop; single-frame passthrough by default (`fusion_target_frames=1`) |
| Persistence | `Storage` + raw SQLite | `data/` repositories behind a single-writer `Writer`, import-linter-enforced |
| Accelerator | MPS / CUDA / TensorRT | MPS → CPU only (CUDA removed) |
| Decode | decord / OpenCV+VideoToolbox | PyAV software decode (auto, Apple Silicon) |

---

## 18. Known Weaknesses (v5.5)

1. **Single-track collapse with sequential cards.** When cards are shown one at
   a time with no overlap, ByteTrack (the default) can merge them into a single
   track, because the v4.1 sampler-driven session-boundary signals (valley
   splits, inter-window gaps) are not produced by `StrideSampler` and are not
   wired into the `track` stage. Videos that present cards serially can yield a
   single output card.
2. **F/B classifier fallback.** If `models/presence_classifier.pt` is absent,
   `resolve` falls back to the longest-track heuristic, which can mislabel a
   short sharp Front against a long blurry Back.
3. **GPU refinement CPU fallback.** If `KorniaNormalizer` cannot be built for the
   requested device, `refine` silently falls back to the CPU `PrecisionNormalizer`
   per-candidate — correct output, but no telemetry distinguishes a degraded run.
4. **Fusion is off by default.** `fusion_target_frames = 1` means the default
   output is the single best canonical crop; the glare-rejecting median fusion
   only runs when a caller raises the target — so default outputs keep whatever
   glare the best single frame has.
5. **Weak background model.** `StrideSampler` contributes a single first-frame
   background proxy, so the novelty model is a one-frame mean; lighting drift or
   a non-empty opening frame degrades the empty-workspace gate.
6. **`runtime_mode` is inert in-process.** `strict_gpu` / `cpu_debug` /
   `mixed_compat` are carried for contract stability but do not change in-process
   execution; the strict-GPU enforcement (`StrictGpuRuntime` / `GpuSession`) is a
   dormant skeleton, so the only live GPU-boundary enforcement is static
   (import-linter + AST scanner), not runtime.
7. **Apple Silicon only.** MPS is the sole accelerator; there is no CUDA path.

For the algorithmic critique of the shared CV components (corner detector,
quality score weighting, pHash invariants, median-fusion alignment), see
[`docs/archive/arch-4.1.md`](../archive/arch-4.1.md) §§3–10, which remain
accurate for the helpers v5.5 reuses.
