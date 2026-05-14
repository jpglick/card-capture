# Card Capture v4 — Upgrade Roadmap

*Extracted from CLAUDE.md Appendix A. This is the strategic plan that shaped
the current codebase; phases 0–2 are substantially shipped.*

---

## Strategic Position

**Decision: preserve the algorithm library, refactor the orchestration layer,
build a real application shell on top.**

| Layer | Verdict | Why |
|---|---|---|
| Algorithm modules (sampler, detectors, scoring, fuser, foil_detection, ECC, deduplicator, presence/, tracking/, gpu_utils, cropper) | **Preserve** | Well-decomposed, encodes hard-won corner-case knowledge. Replacing gains nothing concrete. |
| `pipeline.py` (monolith) | **Decompose** → done via Metaflow | Refactored into `pipeline/steps/` with `card_capture_flow.py` orchestrator. |
| `storage.py` | **Preserve + extend** | Data model sound; wrapped in service layer. |
| UI | **Replace; keep FastAPI** | Done: Svelte SPA on FastAPI backend. |
| `cli.py` | **Keep alongside UI** | CLI stays for headless/CI. |

---

## Phase Status

| Phase | Status | Notes |
|---|---|---|
| 0 — Regression Harness | Shipped | `card-capture harness run`; truth files in `golden_set/` |
| 1 — App Shell + Labeling UX | Shipped | Svelte SPA; `/label`, `/label/fb`, `/label/clusters` |
| 2 — pipeline.py Decomposition | Shipped | Metaflow flow in `pipeline/card_capture_flow.py`; monolith deprecated (Wave 5 deletion) |
| 3 — Algorithmic Fixes | Partial | See V4_CONCERNS.md for open items |
| 4 — Speed Wins | Not started | CoreML/VideoToolbox/vImage paths |
| 5 — Active Learning Loop | Plumbing exists | Hard-case capture wired into UI |

---

## Phase 3 — Algorithmic Fixes (priority order)

1. **Trained Front/Back classifier** — MobileNetV3-Small on rectified crops; replaces longest-track heuristic. *Biggest win on side accuracy.*
2. **DINOv2 + FAISS dedup** — replaces pHash. ViT-S/14, cosine distance, in-process FAISS.
3. **ByteTrack / fixed BoT-SORT** — cleaner tracks, fewer ID switches.
4. **Corner refinement (RANSAC line-fit)** — sub-pixel corners before rectification.
5. **Multi-frame median fusion** — already in `fuser.py`; verify enabled and tune frame count.

## Phase 4 — Speed Wins

1. YOLOv8-OBB → YOLO26-OBB on CoreML (Apple silicon; PyTorch fallback elsewhere)
2. Decoder → VideoToolbox on macOS
3. Perspective warp → vImage on macOS

## What's Explicitly Not Planned

| Skipped | Why |
|---|---|
| `VNDetectRectanglesRequest` | Classical CV fails on glare, foil, occlusion |
| Single-frame neural glare removal | Ill-posed on holographic surfaces; median fusion is principled |
| Prefect / Airflow | Enterprise overhead for single-user batch pipeline |
| OpenTelemetry / Prometheus | SQLite telemetry is sufficient |
| Qdrant | FAISS in-process is sub-ms for ≤100K cards |

---

## Orchestration: Why Metaflow

- **Artifact persistence:** every `self.<name>` snapshotted per step → enables threshold-tuning playground
- **Resume:** skip expensive upstream steps (decoder, YOLO) when iterating on Stages 4–10
- **Local-first:** no infrastructure required; AWS Batch / Kubernetes is opt-in
- **Parallelism:** `foreach` + `@parallel` for per-track fusion

Stages 1–3 (streaming) remain `multiprocessing` + `Queue` internally, wrapped as a single `@step`.

---

## Open Questions

1. **DINOv2 variant:** ViT-S/14 (~22M params, speed) vs ViT-B/14 (~86M params, accuracy). Decide after benchmarking on labeled dedup groups.
2. **Tracker:** BoT-SORT with real-image ReID vs ByteTrack with no ReID. Depends on whether real ReID improves discrimination enough to justify the extra complexity.
3. **Training infra:** local Apple silicon vs cloud GPU. Decide once dataset sizes are known.
4. **Apple-specific fast paths:** feature-detect at startup or hold until cross-platform consensus confirmed.
