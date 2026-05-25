"""fused_refine.run wires inference->novelty->track->refine and returns a RefineOutput."""
from unittest.mock import MagicMock
import pytest

def test_fused_refine_pipes_crop_cache_into_refine(tmp_path, monkeypatch):
    from pipeline.steps import fused_refine
    from pipeline.steps.start import RunContext
    from pipeline.steps.detect import DetectOutput
    from pipeline.steps.novelty import NoveltyOutput
    from pipeline.steps.track import TrackOutput
    from pipeline.steps.refine import RefineOutput

    ctx = RunContext(
        video_path="/nonexistent/video.MOV", output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"), detector="cuda",
        config_preset="balanced", crops_dir=str(tmp_path / "crops"),
        frame_dir=str(tmp_path / "frames"), video_id=1,
    )

    # Stub sampler/detector construction.
    monkeypatch.setattr(fused_refine, "_build_sampler_detector", lambda c: (MagicMock(), MagicMock()))

    captured = {}

    def _fake_cuda_inference(c, s, d, od, fd, crop_cache=None):
        crop_cache[0] = "CROP0"   # pretend one detection was warped
        return DetectOutput(
            frame_count=1, accepted_frame_count=1,
            accepted_frame_presence=[(5, 166, True)],
            detection_rows=[{ "detection_id": 0, "frame_index": 5, "corners": [],
                             "confidence": 0.9, "timestamp_ms": 166, "width": 3840,
                             "height": 2160, "source_frame_path": "", "triage_metrics": {}}],
            sampler_telemetry={"sampler_type": "CudaSampler"},
            video_id=1, detect_telemetry={"yolo_frames": 1},
        )
    monkeypatch.setattr(fused_refine, "_run_cuda_inference", _fake_cuda_inference)

    monkeypatch.setattr(
        fused_refine.novelty, "run",
        lambda c, det: NoveltyOutput(
            detection_rows=det.detection_rows, sampler_telemetry=det.sampler_telemetry,
            bg_model_path=None, accepted_frame_presence=det.accepted_frame_presence,
            frame_count=1, accepted_frame_count=1, video_id=1),
    )
    monkeypatch.setattr(
        fused_refine.track, "run",
        lambda c, nov: TrackOutput(
            tracks_data=[], frame_to_session={}, tracker_events=[],
            detection_rows=nov.detection_rows, sampler_telemetry=nov.sampler_telemetry,
            bg_model_path=None, accepted_frame_presence=nov.accepted_frame_presence,
            frame_count=1, accepted_frame_count=1, video_id=1),
    )

    def _fake_refine_run(c, trk, decoded_crops=None):
        captured["decoded_crops"] = decoded_crops
        return RefineOutput(
            refined_tracks=[], tracks_data=trk.tracks_data,
            detection_rows=trk.detection_rows, sampler_telemetry=trk.sampler_telemetry,
            bg_model_path=None, tracker_events=[], accepted_frame_presence=trk.accepted_frame_presence,
            frame_count=1, accepted_frame_count=1, video_id=1)
    monkeypatch.setattr(fused_refine.refine, "run", _fake_refine_run)

    out = fused_refine.run(ctx)

    assert isinstance(out, RefineOutput)
    # The crop cache built during inference was threaded into refine.
    assert captured["decoded_crops"] == {0: "CROP0"}
