from pathlib import Path
from unittest.mock import patch

import pytest

from card_capture.cli import build_parser, main
from card_capture.pipeline import ProcessingOptions
from card_capture.detectors import TorchDeviceStatus


def test_parser_rejects_missing_process_video_path():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["process"])
    assert exc.value.code == 2


def test_process_subparser_accepts_v21_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "process",
            "video.mov",
            "--config",
            "my_config.json"
        ]
    )
    assert str(args.config) == "my_config.json"
    assert str(args.video_path) == "video.mov"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--reader-backend", "opencv"),
        ("--queue-size", "0"),
        ("--queue-size", "-1"),
        ("--inference-batch-size", "0"),
        ("--corner-confidence", "-0.1"),
        ("--corner-confidence", "1.1"),
        ("--blur-threshold", "0"),
        ("--variance-threshold", "0"),
        ("--empty-pixel-threshold", "-0.1"),
        ("--empty-pixel-threshold", "1.01"),
        ("--telemetry-scope", "partial"),
    ],
)
def test_process_subparser_rejects_invalid_v21_values(flag: str, value: str):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["process", "video.mov", flag, value])
    assert exc.value.code == 2


def test_process_wires_v21_options_into_processing_options(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")

    import json
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "detector": "fake",
        "reader_backend": "decord",
        "queue_size": 7,
        "inference_batch_size": 5,
        "corner_confidence": 0.66,
        "blur_threshold": 22.0,
        "variance_threshold": 33.0,
        "empty_pixel_threshold": 0.92,
        "telemetry_scope": "canonical"
    }))

    with patch("card_capture.cli.VideoProcessor") as mock_processor:
        mock_result = mock_processor.return_value.process.return_value
        mock_result.video_id = 123
        mock_result.frame_count = 10
        mock_result.accepted_frame_count = 6
        mock_result.detection_count = 3
        mock_result.saved_instance_count = 2

        exit_code = main(
            [
                "process",
                str(video_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--db",
                str(tmp_path / "cards.sqlite"),
                "--config",
                str(config_path),
            ]
        )

    assert exit_code == 0
    _, options = mock_processor.return_value.process.call_args.args
    assert isinstance(options, ProcessingOptions)
    assert options.reader_backend == "decord"
    assert options.queue_size == 7
    assert options.inference_batch_size == 5
    assert options.corner_confidence_threshold == 0.66
    assert options.blur_threshold == 22.0
    assert options.variance_threshold == 33.0
    assert options.empty_pixel_threshold == 0.92
    assert options.telemetry_scope == "canonical"


def test_cli_process_fake_detector_writes_database(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    output_dir = tmp_path / "output"
    db_path = tmp_path / "cards.sqlite"
    config_path = tmp_path / "config.json"
    
    import json
    config_path.write_text(json.dumps({"detector": "fake"}))

    exit_code = main(
        [
            "process",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--db",
            str(db_path),
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert db_path.exists()


def test_cli_process_cancels_when_mps_unavailable_and_user_declines(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    config_path = tmp_path / "config.json"

    import json
    config_path.write_text(json.dumps({"detector": "docaligner", "device": "auto"}))

    with patch(
        "card_capture.cli.probe_torch_device_status",
        return_value=TorchDeviceStatus(
            requested="auto",
            resolved="cpu",
            mps_built=True,
            mps_available=False,
            cuda_available=False,
            reason="mps_unavailable",
        ),
    ), patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="n"):
        exit_code = main(
            [
                "process",
                str(video_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--db",
                str(tmp_path / "cards.sqlite"),
                "--config",
                str(config_path),
            ]
        )

    assert exit_code == 1


def test_cli_process_continues_when_mps_unavailable_and_user_accepts(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    config_path = tmp_path / "config.json"

    import json
    config_path.write_text(json.dumps({"detector": "docaligner", "device": "auto"}))

    with patch(
        "card_capture.cli.probe_torch_device_status",
        return_value=TorchDeviceStatus(
            requested="auto",
            resolved="cpu",
            mps_built=True,
            mps_available=False,
            cuda_available=False,
            reason="mps_unavailable",
        ),
    ), patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="y"), patch(
        "card_capture.cli.VideoProcessor"
    ) as mock_processor:
        mock_result = mock_processor.return_value.process.return_value
        mock_result.video_id = 1
        mock_result.frame_count = 1
        mock_result.accepted_frame_count = 1
        mock_result.detection_count = 1
        mock_result.saved_instance_count = 1

        exit_code = main(
            [
                "process",
                str(video_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--db",
                str(tmp_path / "cards.sqlite"),
                "--config",
                str(config_path),
            ]
        )

    assert exit_code == 0


def test_readme_mentions_reader_backend_flag():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "--reader-backend" in readme


def test_quick_reference_mentions_multiprocessing_pipeline():
    quick_reference = Path("QUICK_REFERENCE.md").read_text(encoding="utf-8").lower()
    assert "producer/consumer pipeline" in quick_reference


def test_cli_process_accepts_new_segmentation_flags():
    parser = build_parser()
    args = parser.parse_args([
        "process", "video.mov",
        "--tracker-backend", "botsort",
        "--fast-scan-fps", "15",
        "--valley-drop-ratio", "0.4",
        "--delta-spike-ratio", "0.6",
        "--centroid-jump-ratio", "0.3",
        "--centroid-jump-frames", "3",
        "--reid-distance-threshold", "0.6",
    ])
    assert args.tracker_backend == "botsort"
    assert args.fast_scan_fps == 15.0
    assert args.valley_drop_ratio == 0.4
    assert args.delta_spike_ratio == 0.6
    assert args.centroid_jump_ratio == 0.3
    assert args.centroid_jump_frames == 3
    assert args.reid_distance_threshold == 0.6


def test_cli_process_tracker_backend_choices():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["process", "video.mov", "--tracker-backend", "invalidbackend"])


def test_pipeline_config_has_new_fields():
    from card_capture.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.tracker_backend == "botsort"
    assert cfg.fast_scan_fps == 15.0
    assert cfg.confirm_scan_fps == 5.0
    assert cfg.valley_drop_ratio == 0.40
    assert cfg.valley_min_width_frames == 3
    assert cfg.delta_spike_ratio == 0.60
    assert cfg.centroid_jump_ratio == 0.30
    assert cfg.centroid_jump_frames == 3
    assert cfg.reid_distance_threshold == 0.6
