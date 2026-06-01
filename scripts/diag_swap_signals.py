#!/usr/bin/env python3
"""Measure which swap signal actually fires on IMG_5922.

Runs the real sample -> detect -> novelty stages, then for each consecutive
detected frame computes three candidate reset signals on the top-confidence
detection:
  - appearance_dist : DINOv2 cosine distance to the previous frame's top crop
  - gap             : number of empty sampled frames since the previous detection
  - novelty         : the novelty stage's background-novelty score

We want to know which signal has ~18-20 "spikes" (one per real card swap), so we
wire in the reset the data supports rather than guessing again.

Run: .venv/bin/python scripts/diag_swap_signals.py > out/diag_swap.log 2>&1
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from card_capture.stages import novelty, detect, sample

REPO = Path("/Users/josh/code/card-capture")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from card_capture.core.config import load_config
from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.telemetry import NoopTelemetry
from card_capture.data.writer import Writer
from card_capture.data.repositories.runs import RunsRepository
from card_capture.data.repositories.events import EventsRepository
from card_capture.data.repositories.cards import CardsRepository
from card_capture.data.repositories.videos import VideosRepository
from migrations.run_migrations import apply_migrations
from card_capture.ml.models.dino_embedder import DinoEmbedder
from card_capture.stages.track.appearance_sessionizer import (
    AppearanceObservation,
    AppearanceSessionizer,
)

VIDEO = REPO / "card_capture_uploads/bc827fce3adf4b1ea08ea8e0dec47fb8_IMG_5922.MOV"
DB = REPO / "out/diag_swap.sqlite"
OUT = REPO / "out/diag_swap_out"
OUT.mkdir(parents=True, exist_ok=True)
if DB.exists():
    DB.unlink()
apply_migrations(DB)

w = Writer(DB); w.start()
VIDEO_ID = VideosRepository(writer=w, db_path=DB).register(
    source_path=str(VIDEO), file_hash="diag", duration_ms=1, width=2160, height=3840,
)
w.flush()

cfg = load_config(REPO / "card_capture_config.json")
rc = cfg.to_request_config()
novelty_floor = float(rc.get("novelty_floor", 0.0) or 0.0)

req = PipelineRunRequest(
    run_id="diag_swap",
    input_video=f"artifact://local/{VIDEO}",
    output_root=f"artifact://local/{OUT}/",
    runtime_mode="cpu_debug",
    config=rc,
    db_path=str(DB),
    video_id=VIDEO_ID,
)

state: dict = {
    "request": req,
    "video_id": VIDEO_ID,
    "config_preset": req.config_preset,
    "db_path": DB,
    "output_root": OUT,
    "repos": {
        "runs": RunsRepository(w, DB),
        "events": EventsRepository(w, DB),
        "cards": CardsRepository(w, DB),
    },
}

tele = NoopTelemetry()
t = time.time()
sample.run(state, telemetry=tele)
detect.run(state, telemetry=tele)
novelty.run(state, telemetry=tele)
print(f"sample+detect+novelty in {time.time() - t:.1f}s", flush=True)
w.flush(); w.stop()

sampled = state["sampled_frames"]
dets = state["novelty_scored_detections"]
img_by_idx = {int(f.frame_index): f.image for f in sampled}
sampled_idx = sorted(img_by_idx)

by_frame: dict[int, list] = defaultdict(list)
for d in dets:
    by_frame[int(d["frame_index"])].append(d)
det_frames = sorted(by_frame)

embedder = DinoEmbedder(variant="vits14")


def crop_emb(img, corners):
    xs = [int(p[0]) for p in corners]
    ys = [int(p[1]) for p in corners]
    h, ww = img.shape[:2]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(ww, max(xs)), min(h, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = np.ascontiguousarray(img[y1:y2, x1:x2][:, :, ::-1])  # BGR->RGB, contiguous
    e = embedder.embed_array(crop).cpu().numpy().reshape(-1).astype(np.float32)
    n = float(np.linalg.norm(e))
    return e / n if n > 0 else e


rows = []
embedding_by_frame = {}
prev_e = None
prev_det_frame = None
for fi in det_frames:
    cands = by_frame[fi]
    top = max(cands, key=lambda d: float(d.get("confidence", 0.0)))
    nov = float(top.get("novelty_score", 1.0))
    img = img_by_idx.get(fi)
    e = crop_emb(img, top["corners"]) if (img is not None and top.get("corners")) else None
    app = None
    if e is not None and prev_e is not None:
        app = 1.0 - float(np.dot(e, prev_e))
    gap = None
    if prev_det_frame is not None:
        gap = sum(1 for s in sampled_idx if prev_det_frame < s < fi and s not in by_frame)
    rows.append((fi, gap, nov, app))
    if e is not None:
        embedding_by_frame[fi] = e
        prev_e = e
    prev_det_frame = fi


def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))] if a else 0.0


apps = [r[3] for r in rows if r[3] is not None]
gaps = [r[1] for r in rows if r[1] is not None]
novs = [r[2] for r in rows]

print(f"\ndetected_frames={len(det_frames)}  sampled_frames={len(sampled_idx)}")
print("\n--- APPEARANCE (DINOv2 cosine distance to previous top crop) ---")
print(f"  n={len(apps)} p50={pct(apps,.5):.3f} p90={pct(apps,.9):.3f} max={max(apps) if apps else 0:.3f}")
for th in (0.10, 0.15, 0.20, 0.30, 0.40):
    print(f"  spikes > {th:.2f}: {sum(1 for a in apps if a > th)}")
print("\n--- DETECTION GAP (empty sampled frames since previous detection) ---")
print(f"  p50={pct(gaps,.5):.1f} p90={pct(gaps,.9):.1f} max={max(gaps) if gaps else 0}")
for th in (1, 2, 3, 5):
    print(f"  gaps >= {th}: {sum(1 for g in gaps if g >= th)}")
print("\n--- NOVELTY (background-novelty score of top detection) ---")
print(f"  p10={pct(novs,.1):.3f} p50={pct(novs,.5):.3f} floor={novelty_floor}")
for th in (novelty_floor, 0.05, 0.08, 0.10):
    print(f"  below {th:.3f}: {sum(1 for n in novs if n < th)}")

print("\n--- PRODUCTION APPEARANCE SESSIONIZER ---")
observations = [
    AppearanceObservation(
        frame_index=fi,
        detection_id=fi,
        embedding=embedding_by_frame[fi],
        novelty_score=nov,
    )
    for fi, _gap, nov, _app in rows
    if fi in embedding_by_frame
]
result = AppearanceSessionizer().sessionize(observations)
print(f"retained_presentations={len(result.retained_plateaus)}")
print(f"suppressed_bridges={len(result.suppressed_plateaus)}")
print(f"boundary_frames={result.boundary_frame_indices}")

# Threshold sweep: the embeddings are already cached in `observations`, so this
# is instant (no re-decode / re-embed). Find the (same, change, confirm) combo
# that yields retained=26 (ground-truth card fronts) with ~18 bridges.
print("\n--- THRESHOLD SWEEP (cached embeddings; target retained=26) ---")
print(f"  {'same':>5} {'change':>6} {'confirm':>7} {'confirmed':>9} {'retained':>8} {'suppressed':>10}")
for s in (0.12, 0.15):
    for c in (0.16, 0.18, 0.20, 0.22, 0.25, 0.30):
        if c <= s:
            continue
        for k in (2, 3):
            r = AppearanceSessionizer(
                same_threshold=s, change_threshold=c, confirm_frames=k
            ).sessionize(observations)
            ret = len(r.retained_plateaus)
            sup = len(r.suppressed_plateaus)
            mark = "  <== target" if ret == 26 else ""
            print(
                f"  {s:>5.2f} {c:>6.2f} {k:>7d} "
                f"{ret + sup:>9d} {ret:>8d} {sup:>10d}{mark}"
            )

csv_path = OUT / "swap_signals.csv"
with open(csv_path, "w", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["frame_index", "gap", "novelty", "appearance_dist"])
    for fi, gap, nov, app in rows:
        cw.writerow([fi, "" if gap is None else gap, f"{nov:.4f}", "" if app is None else f"{app:.4f}"])
print(f"\nwrote {csv_path}")
print("DONE")
