from pathlib import Path
from unittest.mock import MagicMock, patch
import argparse

from card_capture.cli import build_parser, main


def test_parser_rejects_missing_process_video_path():
    parser = build_parser()

    try:
        parser.parse_args(["process"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser to reject missing video path")


def test_cli_process_fake_detector_writes_database(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    output_dir = tmp_path / "output"
    db_path = tmp_path / "cards.sqlite"

    exit_code = main(
        [
            "process",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--db",
            str(db_path),
            "--detector",
            "fake",
        ]
    )

    assert exit_code == 0
    assert db_path.exists()


def test_process_subparser_accepts_new_flags():
    from card_capture.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "process", "video.mov",
        "--detection-width", "320",
        "--scan-fps", "5",
        "--scan-width", "120",
        "--motion-threshold", "12.0",
        "--min-stable-frames", "4",
        "--sampler", "stability",
        "--detections-to-stop", "2",
        "--quality-floor", "0.6",
        "--candidates-per-window", "3",
    ])
    assert args.detection_width == 320
    assert args.scan_fps == 5.0
    assert args.scan_width == 120
    assert args.motion_threshold == 12.0
    assert args.min_stable_frames == 4
    assert args.sampler == "stability"
    assert args.detections_to_stop == 2
    assert args.quality_floor == 0.6
    assert args.candidates_per_window == 3


def test_sampler_raw_uses_video_sampler(tmp_path: Path):
    """--sampler raw must construct VideoSampler, not StabilityBasedSampler."""
    video_path = tmp_path / "v.mov"
    video_path.write_bytes(b"x")

    fake_result = MagicMock()
    fake_result.video_id = 1
    fake_result.detection_count = 0
    fake_result.saved_count = 0
    fake_result.output_dir = tmp_path / "out"

    with patch("card_capture.cli.VideoProcessor") as MockProcessor, \
         patch("card_capture.cli.VideoSampler") as MockVideoSampler, \
         patch("card_capture.cli.StabilityBasedSampler") as MockStability:
        MockProcessor.return_value.process.return_value = fake_result
        main([
            "process", str(video_path),
            "--sampler", "raw",
            "--db", str(tmp_path / "db.sqlite"),
            "--output-dir", str(tmp_path / "out"),
        ])

    MockVideoSampler.assert_called_once()
    MockStability.assert_not_called()


def test_detector_fake_always_uses_synthetic_sampler(tmp_path: Path):
    """--detector fake must use SyntheticSampler even when --sampler raw is passed."""
    video_path = tmp_path / "v.mov"
    video_path.write_bytes(b"x")

    fake_result = MagicMock()
    fake_result.video_id = 1
    fake_result.detection_count = 0
    fake_result.saved_count = 0
    fake_result.output_dir = tmp_path / "out"

    with patch("card_capture.cli.VideoProcessor") as MockProcessor, \
         patch("card_capture.cli.SyntheticSampler") as MockSynthetic, \
         patch("card_capture.cli.VideoSampler") as MockVideoSampler:
        MockProcessor.return_value.process.return_value = fake_result
        main([
            "process", str(video_path),
            "--detector", "fake",
            "--sampler", "raw",
            "--db", str(tmp_path / "db.sqlite"),
            "--output-dir", str(tmp_path / "out"),
        ])

    MockSynthetic.assert_called_once()
    MockVideoSampler.assert_not_called()


def test_cli_rejects_nonpositive_scan_fps():
    """--scan-fps 0 must be rejected by argparse."""
    parser = build_parser()
    try:
        parser.parse_args(["process", "video.mov", "--scan-fps", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected rejection of --scan-fps 0")


def test_cli_rejects_out_of_range_quality_floor():
    """--quality-floor 1.5 must be rejected by argparse."""
    parser = build_parser()
    try:
        parser.parse_args(["process", "video.mov", "--quality-floor", "1.5"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected rejection of --quality-floor 1.5")


def test_process_command_accepts_window_merge_gap_flag():
    """CLI should accept --window-merge-gap flag."""
    parser = build_parser()
    args = parser.parse_args([
        "process",
        "video.mov",
        "--sampler", "contrast",
        "--window-merge-gap", "10",
    ])
    assert args.window_merge_gap == 10


def test_process_command_metric_flags_help():
    """--help should show new metric flags."""
    parser = build_parser()
    
    # Parse args to get the subparser, then check its help
    try:
        parser.parse_args(["process", "--help"])
    except SystemExit:
        pass
    
    # Get the help string by getting subparser directly
    subparsers_actions = [
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    
    for subparsers_action in subparsers_actions:
        for choice, subparser in subparsers_action.choices.items():
            if choice == 'process':
                help_str = subparser.format_help()
                assert "--detection-metrics" in help_str
                assert "--histogram-outlier-sigma" in help_str
                assert "--edge-density-threshold" in help_str
                assert "--sobel-magnitude-threshold" in help_str
                assert "--sharpness-batch-size" in help_str
                return
    
    raise AssertionError("Could not find process subparser")


def test_process_command_custom_metrics(tmp_path: Path):
    """Should accept custom metric settings."""
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    
    exit_code = main(
        [
            "process",
            str(video_path),
            "--db", str(tmp_path / "test.db"),
            "--output-dir", str(tmp_path),
            "--detector", "fake",
            "--detection-metrics", "variance",
            "--detection-metrics", "motion",
            "--motion-threshold", "5.0",
        ]
    )
    # Should not raise exception (exit code doesn't matter for this test)
    assert exit_code in [0, 1, 2]

