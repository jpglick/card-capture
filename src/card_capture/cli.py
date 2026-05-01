from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .detectors import CardcaptorUltralyticsDetector, FakeCardDetector
from .pipeline import ProcessingOptions, VideoProcessor
from .sampler import SyntheticSampler, VideoSampler
from .storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="card-capture")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a local video file")
    process.add_argument("video_path", type=Path)
    process.add_argument("--output-dir", type=Path, default=Path("card_capture_output"))
    process.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    process.add_argument("--sample-fps", type=float, default=5.0)
    process.add_argument("--max-candidates", type=int, default=10)
    process.add_argument("--confidence", type=float, default=0.25)
    process.add_argument(
        "--detector",
        choices=["cardcaptor", "fake"],
        default="cardcaptor",
        help="Use cardcaptor for real detection or fake for smoke tests",
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
    else:
        detector = CardcaptorUltralyticsDetector(confidence_threshold=args.confidence)
        sampler = VideoSampler()

    processor = VideoProcessor(storage=storage, sampler=sampler, detector=detector)
    result = processor.process(
        args.video_path,
        ProcessingOptions(
            output_dir=args.output_dir,
            sample_fps=args.sample_fps,
            max_candidates=args.max_candidates,
            confidence_threshold=args.confidence,
        ),
    )
    print(
        f"Processed video_id={result.video_id}: "
        f"{result.detection_count} detections, {result.saved_count} saved"
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
