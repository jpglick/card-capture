from pathlib import Path

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
    ])
    assert args.detection_width == 320
    assert args.scan_fps == 5.0
    assert args.scan_width == 120
    assert args.motion_threshold == 12.0
    assert args.min_stable_frames == 4
    assert args.sampler == "stability"
    assert args.detections_to_stop == 2
    assert args.quality_floor == 0.6
