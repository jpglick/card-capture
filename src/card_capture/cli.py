from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .detectors import CardcaptorUltralyticsDetector, FakeCardDetector
from .pipeline import ProcessingOptions, VideoProcessor
from .sampler import SyntheticSampler, VideoSampler
from .storage import Storage


def _positive_float(value: str) -> float:
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value!r}")
    return f


def _positive_int(value: str) -> int:
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return i


def _unit_float(value: str) -> float:
    """Float in [0.0, 1.0]."""
    f = float(value)
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0, got {value!r}")
    return f


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="card-capture")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a local video file")
    process.add_argument("video_path", type=Path)
    process.add_argument("--output-dir", type=Path, default=Path("card_capture_output"))
    process.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    process.add_argument(
        "--detector",
        choices=["docaligner", "fake"],
        default="docaligner",
        help="docaligner for production detection, fake for smoke tests",
    )
    process.add_argument(
        "--reader-backend",
        choices=["auto", "decord", "pyav"],
        default="auto",
        help="Frame reader backend: auto (default), decord, or pyav",
    )
    process.add_argument(
        "--queue-size",
        type=_positive_int,
        default=64,
        help="Worker queue max size (default: 64)",
    )
    process.add_argument(
        "--inference-batch-size",
        type=_positive_int,
        default=16,
        help="Consumer inference batch size (default: 16)",
    )
    process.add_argument(
        "--corner-confidence",
        type=_unit_float,
        default=0.5,
        help="Minimum corner confidence threshold in [0.0, 1.0] (default: 0.5)",
    )
    process.add_argument(
        "--blur-threshold",
        type=_positive_float,
        default=30.0,
        help="Minimum blur/sharpness threshold for triage (default: 30.0)",
    )
    process.add_argument(
        "--variance-threshold",
        type=_positive_float,
        default=20.0,
        help="Minimum pixel variance threshold for triage (default: 20.0)",
    )
    process.add_argument(
        "--empty-pixel-threshold",
        type=_unit_float,
        default=0.98,
        help="Maximum empty-pixel ratio threshold in [0.0, 1.0] (default: 0.98)",
    )
    process.add_argument(
        "--detection-width",
        type=_positive_int,
        default=640,
        help="Frame width passed to the detector, proportionally scaled (default: 640)",
    )
    process.add_argument(
        "--device", default="auto",
        help="Device for model inference: auto (default, uses MPS on Mac), cpu, mps, cuda",
    )

    review = subparsers.add_parser("review", help="Start the local review UI")
    review.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "process":
        return _run_process(args)
    if args.command == "review":
        return _run_review(args)
    parser.error("unknown command")
    return 2


def _run_process(args: argparse.Namespace) -> int:
    storage = Storage(args.db)
    storage.initialize()

    if args.detector == "fake":
        detector = FakeCardDetector()
        sampler = SyntheticSampler()
    else:  # docaligner
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=args.corner_confidence,
            detection_width=args.detection_width,
            device=args.device,
        )
        sampler = VideoSampler()

    processor = VideoProcessor(storage=storage, sampler=sampler, detector=detector)
    result = processor.process(
        args.video_path,
        ProcessingOptions(
            output_dir=args.output_dir,
            reader_backend=args.reader_backend,
            queue_size=args.queue_size,
            inference_batch_size=args.inference_batch_size,
            corner_confidence_threshold=args.corner_confidence,
            blur_threshold=args.blur_threshold,
            variance_threshold=args.variance_threshold,
            empty_pixel_threshold=args.empty_pixel_threshold,
        ),
    )
    print(
        f"Processed video_id={result.video_id}: "
        f"{result.frame_count} frames ({result.accepted_frame_count} accepted), "
        f"{result.detection_count} detections, {result.saved_instance_count} saved"
    )
    return 0


def _run_review(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Review UI requires: pip install '.[review]'") from exc

    from .review import create_app

    app = create_app(args.db)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
