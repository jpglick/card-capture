from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .detectors import CardcaptorUltralyticsDetector, FakeCardDetector
from .pipeline import ProcessingOptions, VideoProcessor
from .sampler import ContrastBasedSampler, DetectionGuidedSampler, StabilityBasedSampler, SyntheticSampler, VideoSampler
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
    process.add_argument("--sample-fps", type=float, default=5.0)
    process.add_argument("--max-candidates", type=int, default=10)
    process.add_argument("--confidence", type=float, default=0.25)
    process.add_argument(
        "--detector",
        choices=["cardcaptor", "fake"],
        default="cardcaptor",
        help="cardcaptor for real detection, fake for smoke tests",
    )
    process.add_argument(
        "--sampler",
        choices=["stability", "detection", "contrast", "raw"],
        default="stability",
        help="stability: motion-based (default); detection: card-presence-based; contrast: color-variance-based; raw: cadence-based",
    )
    process.add_argument(
        "--scan-fps", type=_positive_float, default=10.0,
        help="Pass-1 scan cadence in frames per second (default: 10)",
    )
    process.add_argument(
        "--scan-width", type=_positive_int, default=160,
        help="Pass-1 scan frame width in pixels (default: 160)",
    )
    process.add_argument(
        "--motion-threshold", type=_positive_float, default=8.0,
        help="Max mean pixel diff (0-255) to count as stable (default: 8.0)",
    )
    process.add_argument(
        "--min-stable-frames", type=_positive_int, default=5,
        help="Min consecutive stable scan frames to form a window (default: 5)",
    )
    process.add_argument(
        "--detection-width", type=_positive_int, default=640,
        help="Frame width passed to YOLO detector, proportionally scaled (default: 640)",
    )
    process.add_argument(
        "--detections-to-stop", type=int, default=1,
        help="Stop after this many quality detections; 0 = disabled (default: 1)",
    )
    process.add_argument(
        "--quality-floor", type=_unit_float, default=0.5,
        help="Minimum quality score to count toward early stop (default: 0.5)",
    )
    process.add_argument(
        "--candidates-per-window", type=_positive_int, default=5,
        help="Candidate frames yielded per stable/detection window, evenly distributed (default: 5)",
    )
    process.add_argument(
        "--detection-scan-fps", type=_positive_float, default=3.0,
        help="Pass-1 scan cadence for detection-guided sampler (default: 3)",
    )
    process.add_argument(
        "--min-detection-frames", type=_positive_int, default=3,
        help="Min consecutive detection frames to form a detection window (default: 3)",
    )
    process.add_argument(
        "--contrast-threshold",
        type=float,
        default=1000.0,
        help="Minimum color variance to detect card presence (Pass 1). Default: 1000.0",
    )
    process.add_argument(
        "--min-presence-frames",
        type=int,
        default=3,
        help="Minimum consecutive frames to form a presence window. Default: 3",
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
    else:
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=args.confidence,
            detection_width=args.detection_width,
            device=args.device,
        )
        if args.sampler == "raw":
            sampler = VideoSampler()
        elif args.sampler == "detection":
            sampler = DetectionGuidedSampler(
                scan_fps=args.detection_scan_fps,
                scan_width=args.scan_width,
                detection_confidence=args.confidence,
                min_detection_frames=args.min_detection_frames,
                candidates_per_window=args.candidates_per_window,
                device=args.device,
            )
        elif args.sampler == "contrast":
            sampler = ContrastBasedSampler(
                video_path=args.video_path,
                scan_fps=args.scan_fps,
                scan_width=args.scan_width,
                contrast_threshold=args.contrast_threshold,
                min_presence_frames=args.min_presence_frames,
                candidates_per_window=args.candidates_per_window,
            )
        else:  # stability
            sampler = StabilityBasedSampler(
                scan_fps=args.scan_fps,
                scan_width=args.scan_width,
                motion_threshold=args.motion_threshold,
                min_stable_frames=args.min_stable_frames,
                candidates_per_window=args.candidates_per_window,
            )

    processor = VideoProcessor(storage=storage, sampler=sampler, detector=detector)
    result = processor.process(
        args.video_path,
        ProcessingOptions(
            output_dir=args.output_dir,
            sample_fps=args.sample_fps,
            max_candidates=args.max_candidates,
            confidence_threshold=args.confidence,
            detections_to_stop=args.detections_to_stop,
            quality_floor=args.quality_floor,
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

