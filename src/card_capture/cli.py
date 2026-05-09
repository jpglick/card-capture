from __future__ import annotations

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .detectors import CardcaptorUltralyticsDetector, FakeCardDetector, probe_torch_device_status
from .pipeline import ProcessingOptions, VideoProcessor
from .sampler import AdaptivePresenceSampler, SyntheticSampler
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
        device_status = probe_torch_device_status(config.device)
        if not _confirm_cpu_fallback(device_status):
            print("Cancelled before processing because GPU acceleration is unavailable.")
            return 1
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=config.corner_confidence,
            detection_width=config.detection_width,
            device=config.device,
        )
        sampler = AdaptivePresenceSampler(
            video_path=args.video_path,
            reader_backend=config.reader_backend,
            device=config.device,
        )

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
            min_track_length=config.min_track_length,
            telemetry_scope=config.telemetry_scope,
            triage_keep_percentile=config.triage_keep_percentile,
            background_frames=config.background_frames,
            background_threshold=config.background_threshold,
            null_patience_frames=config.null_patience_frames,
            rotate_180=config.rotate_180,
        ),
        debug_config=config.debug
    )
    print(
        f"Processed video_id={result.video_id}: "
        f"{result.frame_count} frames ({result.accepted_frame_count} accepted), "
        f"{result.detection_count} detections, {result.saved_instance_count} saved"
    )
    if isinstance(result.telemetry, dict) and result.telemetry:
        telemetry = result.telemetry
        t_high = telemetry.get("tracker_t_high", 0.0)
        try:
            t_high_display = f"{float(t_high):.3f}"
        except (TypeError, ValueError):
            t_high_display = "n/a"
        print(
            "Telemetry: "
            f"windows={telemetry.get('last_presence_window_count', 0)}, "
            f"selected_frames={telemetry.get('last_selected_frame_count', 0)}, "
            f"tracks={telemetry.get('tracks_finalized', 0)}, "
            f"duplicates={telemetry.get('duplicate_tracks', 0)}, "
            f"tracker_events={telemetry.get('tracker_event_count', 0)}, "
            f"t_high={t_high_display}, "
            f"status={telemetry.get('status', 'unknown')}"
        )
        telemetry_path = telemetry.get("telemetry_path")
        if telemetry_path:
            print(f"Telemetry written to {telemetry_path}")
        tracker_events_path = telemetry.get("tracker_association_events_path")
        if tracker_events_path:
            print(f"Tracker events written to {tracker_events_path}")
    return 0


def _confirm_cpu_fallback(device_status) -> bool:
    if device_status.resolved != "cpu":
        return True
    if device_status.requested not in {"auto", "mps"}:
        return True
    if not device_status.mps_built:
        return True

    message = (
        "PyTorch was built with MPS support, but MPS is unavailable in this runtime.\n"
        f"Requested device: {device_status.requested}\n"
        f"Resolved device: {device_status.resolved}\n"
        f"Reason: {device_status.reason}\n"
        "Continue on CPU? [y/N]: "
    )
    if not sys.stdin.isatty():
        raise RuntimeError(
            "MPS is unavailable and this process is non-interactive, so CPU fallback was not auto-approved."
        )
    answer = input(message).strip().lower()
    return answer in {"y", "yes"}


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
