#!/usr/bin/env python3
"""Diagnostic: dump per-stage funnel counts for IMG_5922 (video_id=7).

Why: every run on this video yields exactly 1 card. track_telemetry only
persists the single SURVIVING track, so we cannot see how many detections /
tracks existed before the min_track_length filter. This instruments each
stage's run() to print the size of the key state collections, plus detection
area/confidence distributions and per-track frame/centroid spread.

Run:  .venv/bin/python scripts/diag_track_funnel.py > out/diag_funnel.log 2>&1
"""
from __future__ import annotations

import statistics as S
import sys
import time
from pathlib import Path

REPO = Path("/Users/josh/code/card-capture")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from card_capture.config import load_config
from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline import runtime_local
from card_capture.pipeline.telemetry import NoopTelemetry

VIDEO = REPO / "card_capture_uploads/bc827fce3adf4b1ea08ea8e0dec47fb8_IMG_5922.MOV"
DB = REPO / "out/diag_funnel.sqlite"
OUT = REPO / "out/diag_funnel_out"
OUT.mkdir(parents=True, exist_ok=True)
if DB.exists():
    DB.unlink()

# Schema-init the throwaway DB exactly like production (so stage_metrics writes
# to pipeline_events succeed and store can persist), and register the video so
# the store stage's FK to videos(id) is satisfied.
from migrations.run_migrations import apply_migrations
apply_migrations(DB)
from card_capture.data.writer import Writer as _W
from card_capture.data.repositories.videos import VideosRepository
_w = _W(DB); _w.start()
try:
    VIDEO_ID = VideosRepository(writer=_w, db_path=DB).register(
        source_path=str(VIDEO), file_hash="diag", duration_ms=1, width=2160, height=3840,
    )
    _w.flush()
finally:
    _w.stop()


def _dist(name, vals):
    if not vals:
        print(f"      {name}: (none)")
        return
    vals = sorted(vals)
    n = len(vals)
    q = lambda p: vals[min(n - 1, int(p * n))]
    print(f"      {name}: n={n} min={vals[0]:.0f} p50={q(0.5):.0f} "
          f"p90={q(0.9):.0f} max={vals[-1]:.0f}")


def _corner_area(corners):
    if not corners or len(corners) < 3:
        return 0.0
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _centroid(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def summarize(tag, state):
    print(f"\n##### after stage: {tag} #####", flush=True)
    for k, v in state.items():
        if isinstance(v, list):
            print(f"  state[{k!r}] = list(len={len(v)})")

    if tag == "detect":
        dets = state.get("detections", [])
        frames = {d.get("frame_index") for d in dets}
        print(f"  >>> DETECT: {len(dets)} detections over {len(frames)} distinct frames")
        areas = [_corner_area(d.get("corners")) for d in dets if d.get("corners")]
        confs = [float(d.get("confidence", 0)) for d in dets]
        _dist("det_bbox_area_px2", areas)
        _dist("det_confidence(x1000)", [c * 1000 for c in confs])
        # how many frames have >1 detection?
        from collections import Counter
        per_frame = Counter(d.get("frame_index") for d in dets)
        multi = sum(1 for c in per_frame.values() if c > 1)
        print(f"      frames_with_multiple_dets={multi}; max_dets_in_a_frame="
              f"{max(per_frame.values()) if per_frame else 0}")

    if tag == "track":
        tracks = state.get("tracks", [])
        print(f"  >>> TRACK: {len(tracks)} finalized tracks (>= min_track_length)")
        lengths = []
        for i, ts in enumerate(tracks):
            cands = getattr(ts, "candidates", [])
            lengths.append(len(cands))
            fis = [c.frame_index for c in cands]
            cxs, cys, areas = [], [], []
            for c in cands:
                if getattr(c, "corners", None):
                    cx, cy = _centroid(c.corners)
                    cxs.append(cx); cys.append(cy)
                    areas.append(_corner_area(c.corners))
            cxr = (max(cxs) - min(cxs)) if cxs else 0
            cyr = (max(cys) - min(cys)) if cys else 0
            amax = max(areas) if areas else 0
            print(f"      track[{i}] len={len(cands)} "
                  f"frames[{min(fis) if fis else '-'}..{max(fis) if fis else '-'}] "
                  f"centroid_xrange={cxr:.0f} yrange={cyr:.0f} max_area={amax:.0f}")
        _dist("finalized_track_lengths", lengths)


# Wrap every stage run() to print state sizes after it executes.
for _name, _module in runtime_local._STAGES:
    _orig = _module.run

    def _make(orig, name):
        def wrapped(state, *, telemetry):
            r = orig(state, telemetry=telemetry)
            try:
                summarize(name, state)
            except Exception as e:  # never let diagnostics break the run
                print(f"  [summarize {name} error] {e!r}", flush=True)
            return r
        return wrapped

    _module.run = _make(_orig, _name)


def main():
    cfg = load_config(REPO / "card_capture_config.json")
    rc = cfg.to_request_config()
    print(f"tracker_backend={rc.get('tracker_backend')} "
          f"min_track_length={rc.get('min_track_length')} "
          f"lost_track_buffer={rc.get('lost_track_buffer')} "
          f"track_activation_threshold={rc.get('track_activation_threshold')} "
          f"corner_confidence={rc.get('corner_confidence')} "
          f"detector={rc.get('detector')}", flush=True)

    req = PipelineRunRequest(
        run_id="diag_funnel",
        input_video=f"artifact://local/{VIDEO}",
        output_root=f"artifact://local/{OUT}/",
        runtime_mode="cpu_debug",
        config=rc,
        db_path=str(DB),
        video_id=VIDEO_ID,
    )
    t = time.time()
    rt = runtime_local.LocalPipelineRuntime(telemetry=NoopTelemetry())
    try:
        res = rt.run(req)
        print(f"\nDONE in {time.time()-t:.1f}s cards={len(res.manifest.cards)} "
              f"violations={res.manifest.contract_violations}", flush=True)
    except Exception as e:
        print(f"\nRUN raised after {time.time()-t:.1f}s: {e!r} "
              f"(track-stage counts above are still valid)", flush=True)


if __name__ == "__main__":
    main()
