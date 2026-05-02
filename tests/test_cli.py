from pathlib import Path
from unittest.mock import patch

import pytest

from card_capture.cli import build_parser, main
from card_capture.pipeline import ProcessingOptions


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
            "--detector",
            "docaligner",
            "--reader-backend",
            "pyav",
            "--queue-size",
            "8",
            "--inference-batch-size",
            "4",
            "--corner-confidence",
            "0.75",
            "--blur-threshold",
            "12.5",
            "--variance-threshold",
            "45.0",
            "--empty-pixel-threshold",
            "0.9",
        ]
    )
    assert args.detector == "docaligner"
    assert args.reader_backend == "pyav"
    assert args.queue_size == 8
    assert args.inference_batch_size == 4
    assert args.corner_confidence == 0.75
    assert args.blur_threshold == 12.5
    assert args.variance_threshold == 45.0
    assert args.empty_pixel_threshold == 0.9


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
                "--detector",
                "fake",
                "--reader-backend",
                "decord",
                "--queue-size",
                "7",
                "--inference-batch-size",
                "5",
                "--corner-confidence",
                "0.66",
                "--blur-threshold",
                "22.0",
                "--variance-threshold",
                "33.0",
                "--empty-pixel-threshold",
                "0.92",
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


def test_readme_mentions_reader_backend_flag():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "--reader-backend" in readme


def test_quick_reference_mentions_multiprocessing_pipeline():
    quick_reference = Path("QUICK_REFERENCE.md").read_text(encoding="utf-8").lower()
    assert "producer/consumer pipeline" in quick_reference
