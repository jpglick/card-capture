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
