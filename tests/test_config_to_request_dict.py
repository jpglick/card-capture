"""Phase 1 — PipelineConfig.to_request_config() returns a JSON-safe dict
of all knobs stages consume."""
import json

from card_capture.config import PipelineConfig


def test_to_request_config_includes_all_back_half_fields():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    assert d["novelty_floor"] == 0.30
    assert d["track_confidence_floor"] == 0.60
    assert d["stand_novelty_max"] == 0.35
    assert d["stand_sharpness_max"] == 0.30
    assert d["foil_threshold"] == 50.0
    assert d["enable_foil_aware_fusion"] is True
    assert d["use_fb_classifier"] is True
    assert d["laplacian_scan_stride"] == 5
    assert d["max_corner_gap_frames"] == 30
    assert d["corner_refinement"] is False


def test_to_request_config_includes_detector_and_device():
    cfg = PipelineConfig()
    cfg.detector = "docaligner"
    cfg.device = "cpu"
    d = cfg.to_request_config()
    assert d["detector"] == "docaligner"
    assert d["device"] == "cpu"


def test_to_request_config_is_json_serializable():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["novelty_floor"] == 0.30

def test_cli_run_process_passes_full_config(tmp_path, monkeypatch):
    """The CLI's _run_process must call to_request_config()."""
    from unittest.mock import patch, MagicMock
    from card_capture.cli import _run_process
    import uuid

    captured = {}

    def fake_runtime_run(self, request):
        captured["config"] = dict(request.config)
        result = MagicMock()
        result.manifest.contract_violations = []
        result.manifest.to_json.return_value = "{}"
        return result

    args = MagicMock()
    args.video_path = tmp_path / "v.mov"
    args.video_path.touch()
    args.output_dir = tmp_path / "out"
    args.db = tmp_path / "cards.sqlite"
    args.db.touch()
    args.config = tmp_path / "card_capture_config.json"
    args.detector = "fake"
    args.run_id = "r1"
    # All other CLI override fields default to None
    for f in ("tracker_backend", "fast_scan_fps", "confirm_scan_fps",
              "valley_drop_ratio", "valley_min_width_frames",
              "delta_spike_ratio", "centroid_jump_ratio",
              "centroid_jump_frames", "reid_distance_threshold",
              "presence_threshold"):
        setattr(args, f, None)

    # Storage.initialize and add_video need to succeed
    with patch("card_capture.storage.Storage") as MockStorage, \
         patch("card_capture.pipeline.runtime_local.LocalPipelineRuntime.run",
               new=fake_runtime_run):
        MockStorage.return_value.add_video.return_value = 1
        _run_process(args)

    assert "novelty_floor" in captured["config"]
    assert "foil_threshold" in captured["config"]
