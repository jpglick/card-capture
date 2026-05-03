from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .detectors import CardcaptorUltralyticsDetector, FakeCardDetector
from .pipeline import ProcessingOptions, VideoProcessor
from .sampler import SyntheticSampler, VideoSampler
from .storage import Storage
from .config import load_config, save_config

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="card-capture")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a local video file")
    process.add_argument("video_path", type=Path)
    process.add_argument("--output-dir", type=Path, default=Path("card_capture_output"))
    process.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    process.add_argument("--config", type=Path, default=Path("card_capture_config.json"))

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

    config = load_config(args.config)
    # Save the defaults if it didn't exist
    if not args.config.exists():
        save_config(config, args.config)

    if config.detector == "fake":
        detector = FakeCardDetector()
        sampler = SyntheticSampler()
    else:  # docaligner
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=config.corner_confidence,
            detection_width=config.detection_width,
            device=config.device,
        )
        sampler = VideoSampler(reader_backend=config.reader_backend)

    processor = VideoProcessor(storage=storage, sampler=sampler, detector=detector)
    result = processor.process(
        args.video_path,
        ProcessingOptions(
            output_dir=args.output_dir,
            reader_backend=config.reader_backend,
            queue_size=config.queue_size,
            inference_batch_size=config.inference_batch_size,
            corner_confidence_threshold=config.corner_confidence,
            blur_threshold=config.blur_threshold,
            variance_threshold=config.variance_threshold,
            empty_pixel_threshold=config.empty_pixel_threshold,
            group_gap_ms=config.group_gap_ms,
            spatial_variance_threshold=config.spatial_variance_threshold,
            telemetry_scope=config.telemetry_scope,
        ),
        debug_config=config.debug
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
