# Pipeline Memory & Performance — Holistic Roadmap

**Date:** 2026-06-07
**Status:** Roadmap / analysis. No implementation scheduled yet. Phase 1 is shovel-ready; Phase 2 has one open architectural decision (see Phase 2, §5).
**Trigger:** Long 4K videos OOM-SIGKILL the worker in the `refine` stage on the 16 GB box (runs `run_933fd293`, `run_cb4114ed`). Investigating the kill revealed that refine is a *symptom* of two pipeline-wide facts, not the disease. See memory `refine-oom-on-long-videos`.

---

## 1. Measured profile (run_cb4114ed, 706 MB / 32 min / 60 fps)

From `run_resource_samples` and `pipeline_run_logs`. `peak RSS` = worker process resident set; `sys%` = system-wide memory utilization; engine from GPU% column + prior profiling.

| stage   | wall   | peak RSS | engine        | notes |
|---------|--------|----------|---------------|-------|
| sample  | ~0.1 s | —        | CPU (stream)  | starts the streaming producer |
| detect  | 185 s  | **9.3 GB** | CPU + ANE (0% Metal) | YOLO; **appends every full 4K frame to `state["sampled_frames"]`** |
| novelty | 101 s  | 5.1 GB   | CPU           | full-frame Lab vs background, per detection |
| track   | 158 s  | 7.0 GB   | GPU 80%       | BoT-SORT + reid |
| refine  | 69 s+  | 5.8 GB   | GPU 70%       | Kornia warp; **process died here (SIGKILL)** |

Total to refine ≈ **8.5 min**, strictly sequential. 6906 detections, 279 tracks.

**Box context:** 16 GB Mac mini, MPS-only. During runs ~8 GB is held by *other* processes (IDE, browser, vite), leaving ~3 GB headroom. `mem_used_mb` ≈ 5.6 GB worker / `mem_pct` ≈ 82% system at the time of the refine kill.

---

## 2. Root facts

**RF1 — The full 4K frame set (~5 GB) is materialized once and held resident across detect→novelty→track→refine.**
- The sampler emits full-resolution frames; the `scan_width=160` downscale (`sample/sampler/__init__.py`) is only for the internal presence/motion scan, not the emitted `FrameSample.image`.
- `detect` appends every drained frame to `state["sampled_frames"]` (`stages/detect/__init__.py:71`).
- novelty (`stages/novelty/__init__.py:38-39`), track (`stages/track/__init__.py:45`), and refine (`stages/refine/__init__.py`) all read from that buffer.
- This ~5 GB through-line is the memory ceiling. On a box with ~3 GB headroom it makes detect's 9.3 GB peak and refine's warp spike flirt with jetsam. **Nothing downstream of refine reads `sampled_frames`** (verified: score/resolve/fuse/dedup/store) — it is refine's job to release it.

**RF2 — Stages are whole-batch sequential passes.** The only overlap today is decode‖detect (the producer). detect is CPU/ANE while track/refine are GPU — different engines idle during each other's stages. Wall time ≈ sum of stages.

**RF3 — Concrete inefficiency: novelty frame lookup is O(detections × frames).** `stages/novelty/__init__.py:39` does `next(f.image for f in frames if f.frame_index == idx)` *per detection* — 6906 detections each linear-scanning the frame list. Should be a dict keyed by `frame_index` (refine already builds exactly this via `_frame_index_lookup`).

---

## 3. Opportunity map — memory

Tagged `[measured]` (this session) or `[estimate]` (prior profiling / first-principles).

- **(a) Stop materializing 4K at all.** Give each consumer only the fidelity it needs: detect 640 (already internal), novelty a downscaled frame, track a card crop, refine a full-res *card-region* crop. `[estimate]` ~5 GB → <1 GB resident. Biggest win; largest blast radius (novelty + track + refine + data contract); must ride the regression baseline.
- **(b) Disk/mmap frame store + prefetch.** Back the frame buffer with memory-mapped files, keep a small RAM LRU, prefetch ahead. **Specific benefit here: mmap'd pages are managed by the OS pager, not counted as anonymous RSS** — so they don't feed jetsam the way the current anonymous buffer does. Full fidelity preserved. Cost: ~5 GB written + re-read ~3× (hideable behind compute), store/eviction complexity.
- **(c) Refine crop pre-pass** (designed; see §5.1). Cheap; fixes only refine's tail of RF1.

## 4. Opportunity map — performance

- **novelty → dict lookup (RF3) + downscaled Lab.** `[estimate]` ~101 s → ~15 s. The Lab gate compares the card quad against background proxies; full 4K is not required.
- **Stage overlap / pipelining (RF2).** Stream decode→detect→novelty per-frame; run CPU/ANE detect concurrently with GPU track. `[estimate]` collapse ~286 s of detect+novelty toward ~190 s. A prior spike measured ~97.6% embed‖novelty overlap headroom (memory `frontend-fusion-spike-result`).
- **track** sessionizer/boxmot is the remaining ~10 s after embeddings (memory `frontend-fusion-spike-result`); over-fragmentation notes in memory `perf-bottleneck-diagnosis`.

---

## 5. Phased roadmap

### Phase 1 — quick wins (low risk, ship first)

Fixes the immediate OOM and a large slice of wall time without rearchitecting. Also a stepping stone: the refine crop pre-pass *is* the refine half of approach (a).

**5.1 Refine card-region crop pre-pass** (memory). Already designed:
- Before the warp loop, select the top-N (N=8, existing cut) candidates per track, group by `frame_index`, and **frame-ordered** crop each candidate's quad bounding box (+`refine_crop_margin_px`, default 8) as a `.copy()`, freeing each full frame immediately (`del decoded_images[idx]`). End with `decoded_images.clear()`; `sampled_frames` already emptied up front.
- Coordinate contract: `crop = frame[y0:y1, x0:x1].copy()`, `local_corners = corners − (x0, y0)`. **Invariant (keystone test):** `warp(full, corners) == warp(crop, local_corners)` pixel-for-pixel. `.copy()` is mandatory (a slice view would pin the full frame).
- The warp loop consumes `{detection_id: (crop, local_corners)}`; both the Kornia path and the `PrecisionNormalizer` fallback use it. Warp upload shrinks ~20×.
- Replace the current per-track progressive-free + `refine_frame_buffer` telemetry with one `refine_cropped` event. Keep `refine_warp_chunk_size` (default 4) and collapse the `refine_min_available_mb` gate into a single pre-pass check.
- Effect: the lethal "5 GB baseline + warp spike" overlap is eliminated; peak stays at the entry buffer (which already survived through track), with no spike on top.

**5.2 Novelty fixes** (perf): dict-keyed frame lookup (RF3) + run the Lab gate on a downscaled frame. Guard with the regression baseline (novelty scores feed scoring/pruning).

**Risk:** crop margin too small → edge interpolation artifacts (mitigated by 8 px default + the equivalence test). Novelty downscale could shift `novelty_score` → must compare against `regression_baselines`.

### Phase 2 — the memory architecture (the real lever) — OPEN DECISION

Choose the target for the resident 4K buffer (RF1):

- **(a) Eliminate the buffer** — per-consumer fidelity (downscaled frames + crops). Wins on *both* memory and perf; each per-stage change is independently beneficial. Riskier: changes novelty/track inputs → must ride `regression_baselines` to catch quality drift.
- **(b) Disk/mmap frame store + prefetch** — relieves RSS via OS paging (directly de-risks jetsam), keeps stage logic and full fidelity, adds IO. Safer; smaller perf upside.

**Recommendation:** lean (a) for the dual win, with (b)/mmap as the fallback if any stage genuinely needs full-frame random access. Decide after Phase 1 lands and we can re-measure headroom. Each option gets its own spec → plan before implementation.

### Phase 3 — overlap / pipelining (perf)

Stream decode→detect→novelty per-frame and overlap CPU/ANE detect with GPU track (RF2). Own spec.

---

## 6. Cross-cutting

- **Regression baseline guard.** Any change touching detect/novelty/track inputs (Phases 1-novelty, 2a) must be validated against `regression_baselines` / `docs/superpowers/plans/v5-5/baseline-results.md` to catch card-yield or quality drift. Pure memory-management changes (5.1, 2b) do not change outputs and need only equivalence tests.
- **Overcommit reality.** ~8 GB is held by other apps during runs. No in-pipeline fix conjures headroom that isn't there; the gate (`refine_min_available_mb`) exists to *fail cleanly* when it isn't. Closing other apps / lowering `fast_scan_fps` for long clips remains a user-side lever.
- **Already shipped (context):** the startup orphan reaper (`app/main.py:87` → `data/repositories/runs.py:48`) now flips killed `running` rows to `failed`, so runs no longer hang forever. Memory safety knobs `refine_min_available_mb` / `refine_warp_chunk_size` exist in `PipelineConfig`; the available-memory gate alone did NOT prevent the kill (it fires below ~2 GB available, but the box hovers ~2.95 GB while jetsam kills on the transient spike) — see memory `refine-oom-on-long-videos`.

## 7. Not decided here

- Phase 2 (a) vs (b) — see Phase 2, §5.
- Target downscale resolution for novelty/track in (a).
- Whether Phase 3 pipelining is worth its complexity vs the simpler per-stage wins.

## 8. Data sources

- Runs: `run_cb4114ed` (this analysis), `run_933fd293` (original kill). Tables: `run_resource_samples`, `pipeline_run_logs`, `pipeline_runs`, `pipeline_events` in `var/db/cards.sqlite`.
- Prior profiling memories: `v5-5-inprocess-stage-breakdown`, `frontend-fusion-spike-result`, `yolo-cheap-input-spike`, `decode-method-benchmark`, `perf-bottleneck-diagnosis`.
