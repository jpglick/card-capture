"""v4 pipeline orchestrated as a Metaflow FlowSpec.

Each @step is a thin call into pipeline.steps.<name>. The actual logic lives
in the step modules; this file is the orchestration spine and stays small.
"""
from __future__ import annotations

from metaflow import FlowSpec, Parameter, current, step

from pipeline.steps import (
    detect, novelty, track, refine, score, resolve, fuse, dedup, store,
)


class CardCaptureFlow(FlowSpec):
    video = Parameter("video", help="Path to source video", required=True)
    output_dir = Parameter("output-dir", help="Output directory", required=True)
    db = Parameter("db", help="SQLite database path", required=True)
    detector = Parameter("detector", help="Detector backend", default="docaligner")
    config_preset = Parameter("config-preset", default="balanced")
    fusion_target_frames = Parameter("fusion-target-frames", default=1, type=int)
    corner_refinement = Parameter("corner-refinement", default=False, type=bool)

    @step
    def start(self):
        from pipeline.steps.start import init_run
        self.run_context = init_run(
            self.video, self.output_dir, self.db,
            self.detector, self.config_preset,
            fusion_target_frames=self.fusion_target_frames,
            corner_refinement=self.corner_refinement
        )
        self.next(self.detect)

    @step
    def detect(self):
        self.detect_out = detect.run(self.run_context)
        self.next(self.novelty)

    @step
    def novelty(self):
        self.novelty_out = novelty.run(self.run_context, self.detect_out)
        self.next(self.track)

    @step
    def track(self):
        self.track_out = track.run(self.run_context, self.novelty_out)
        self.next(self.refine)

    @step
    def refine(self):
        self.refine_out = refine.run(self.run_context, self.track_out)
        self.next(self.score)

    @step
    def score(self):
        self.score_out = score.run(self.run_context, self.refine_out)
        self.next(self.resolve)

    @step
    def resolve(self):
        self.resolve_out = resolve.run(self.run_context, self.score_out)
        self.next(self.fuse_fanout)

    @step
    def fuse_fanout(self):
        self.fanout = list(self.resolve_out.prepared_tracks)
        self.next(self.fuse, foreach="fanout")

    @step
    def fuse(self):
        prepared_track = self.input
        self.fuse_out = fuse.run(self.run_context, prepared_track)
        self.next(self.fuse_join)

    @step
    def fuse_join(self, inputs):
        self.fused_canonicals = [inp.fuse_out.fused_canonical for inp in inputs]
        self.merge_artifacts(inputs, exclude=["fuse_out"])
        self.next(self.dedup)

    @step
    def dedup(self):
        self.dedup_out = dedup.run(self.run_context, self.fused_canonicals)
        self.next(self.store)

    @step
    def store(self):
        self.store_out = store.run(
            self.run_context, 
            self.dedup_out.dedup_groups, 
            self.fused_canonicals,
            prepared_tracks=self.resolve_out.prepared_tracks,
            run_id=current.run_id
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CardCaptureFlow()
