"""v4 pipeline orchestrated as a Metaflow FlowSpec.

Each @step is a thin call into pipeline.steps.<name>. The actual logic lives
in the step modules; this file is the orchestration spine and stays small.
"""
from __future__ import annotations

import time as _time

from metaflow import FlowSpec, Parameter, current, step

from pipeline.steps import (
    detect, novelty, track, refine, score, resolve, fuse, dedup, store,
)


def _record_stage_timing(
    db: str,
    run_id: str,
    video_id: int,
    stage: str,
    elapsed_ms: int,
    extra: dict | None = None,
) -> None:
    """Write a stage timing event to pipeline_events. Never raises.

    extra: optional dict merged into the data_json blob alongside elapsed_ms
    so callers (e.g. detect) can record device/frame-count/triage telemetry
    without needing a separate table. The runpod_handler diagnostic surfaces
    the whole blob as stage_payloads.
    """
    try:
        import sqlite3, json
        payload = {"elapsed_ms": elapsed_ms}
        if extra:
            payload.update(extra)
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_events
                    (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json)
                VALUES (?, ?, ?, 0, 0, ?, ?)
                """,
                (video_id, run_id, stage, f"stage_{stage}", json.dumps(payload)),
            )
    except Exception:
        pass


class CardCaptureFlow(FlowSpec):
    video = Parameter("video", help="Path to source video", required=True)
    output_dir = Parameter("output-dir", help="Output directory", required=True)
    db = Parameter("db", help="SQLite database path", required=True)
    detector = Parameter("detector", help="Detector backend", default="docaligner")
    config_preset = Parameter("config-preset", default="balanced")
    fusion_target_frames = Parameter("fusion-target-frames", default=1, type=int)
    corner_refinement = Parameter("corner-refinement", default=False, type=bool)
    ui_run_id = Parameter("ui-run-id", help="Run ID assigned by the app UI", default="")
    presence_threshold = Parameter("presence-threshold", default=0.4, type=float,
                                   help="Classifier confidence for card presence (0-1). Lower = more frames sampled.")
    min_track_length = Parameter("min-track-length", default=3, type=int,
                                 help="Minimum detections required to keep a track.")

    @step
    def start(self):
        from pipeline.steps.start import init_run
        self.run_context = init_run(
            self.video, self.output_dir, self.db,
            self.detector, self.config_preset,
            fusion_target_frames=self.fusion_target_frames,
            corner_refinement=self.corner_refinement,
            presence_threshold=self.presence_threshold,
            min_track_length=self.min_track_length,
        )
        self.next(self.detect)

    @step
    def detect(self):
        _t0 = _time.time()
        self.detect_out = detect.run(self.run_context)
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        _run_id = self.ui_run_id or current.run_id
        # Merge the detect-stage telemetry (yolo_device, yolo_frames, yolo_batches,
        # yolo_elapsed_s, triage_pass_rate, etc.) into the stage event's data_json.
        # The runpod_handler diagnostic surfaces this as stage_payloads.stage_detect.
        _record_stage_timing(
            ctx.db_path, _run_id, ctx.video_id or 0, "detect", _elapsed_ms,
            extra=self.detect_out.detect_telemetry or {},
        )
        self.next(self.novelty)

    @step
    def novelty(self):
        _t0 = _time.time()
        self.novelty_out = novelty.run(self.run_context, self.detect_out)
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "novelty", _elapsed_ms)
        self.next(self.track)

    @step
    def track(self):
        _t0 = _time.time()
        self.track_out = track.run(self.run_context, self.novelty_out)
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "track", _elapsed_ms)
        self.next(self.refine)

    @step
    def refine(self):
        _t0 = _time.time()
        self.refine_out = refine.run(self.run_context, self.track_out)
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        # Surface per-op timings (op_seconds + op_calls) into stage_refine
        # data_json so the handler diagnostic shows which CPU-bound step
        # dominates refine's wall time. Optional attribute — falls back to
        # bare timing if refine.run didn't set it.
        _refine_extra = getattr(self.refine_out, "refine_telemetry", None) or {}
        _record_stage_timing(
            ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0,
            "refine", _elapsed_ms, extra=_refine_extra,
        )
        self.next(self.score)

    @step
    def score(self):
        _t0 = _time.time()
        self.score_out = score.run(self.run_context, self.refine_out)
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "score", _elapsed_ms)
        self.next(self.resolve)

    @step
    def resolve(self):
        _t0 = _time.time()
        self.resolve_out = resolve.run(self.run_context, self.score_out)
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "resolve", _elapsed_ms)
        self.next(self.fuse_fanout)

    @step
    def fuse_fanout(self):
        tracks = list(self.resolve_out.prepared_tracks)
        # Metaflow foreach requires ≥1 item; sentinel None is filtered in fuse_join
        self.fanout = tracks if tracks else [None]
        self.next(self.fuse, foreach="fanout")

    @step
    def fuse(self):
        prepared_track = self.input
        self.fuse_out = fuse.run(self.run_context, prepared_track) if prepared_track is not None else None
        self.next(self.fuse_join)

    @step
    def fuse_join(self, inputs):
        self.fused_canonicals = [
            inp.fuse_out.fused_canonical for inp in inputs
            if inp.fuse_out is not None
        ]
        self.merge_artifacts(inputs, exclude=["fuse_out"])
        self.next(self.dedup)

    @step
    def dedup(self):
        _t0 = _time.time()
        self.dedup_out = dedup.run(self.run_context, self.fused_canonicals)
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "dedup", _elapsed_ms)
        self.next(self.store)

    @step
    def store(self):
        _t0 = _time.time()
        self.store_out = store.run(
            self.run_context,
            self.dedup_out.dedup_groups,
            self.fused_canonicals,
            prepared_tracks=self.resolve_out.prepared_tracks,
            run_id=self.ui_run_id or current.run_id
        )
        _elapsed_ms = int((_time.time() - _t0) * 1000)
        ctx = self.run_context
        _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "store", _elapsed_ms)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CardCaptureFlow()
