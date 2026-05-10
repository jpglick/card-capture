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
from .presence.classifier import PresenceClassifier

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

    harness_p = subparsers.add_parser("harness", help="Run regression harness against golden corpus")
    harness_sub = harness_p.add_subparsers(dest="harness_command", required=True)

    harness_run = harness_sub.add_parser("run", help="Run pipeline on corpus and write report")
    harness_run.add_argument("--corpus", type=Path, default=Path("tests/fixtures/golden_corpus"))
    harness_run.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    harness_run.add_argument("--output-dir", type=Path, default=Path("card_capture_output"))
    harness_run.add_argument("--reports-dir", type=Path, default=Path("reports"))
    harness_run.add_argument("--baseline", type=Path, default=None,
                              help="Optional baseline JSON report to compute deltas against")

    harness_compare = harness_sub.add_parser("compare", help="Compare two existing reports")
    harness_compare.add_argument("baseline", type=Path)
    harness_compare.add_argument("current", type=Path)
    harness_compare.add_argument("--out", type=Path, default=None)

    # dataset subcommand
    dataset_p = subparsers.add_parser("dataset", help="Training dataset utilities")
    dataset_sub = dataset_p.add_subparsers(dest="dataset_command", required=True)
    ds_export = dataset_sub.add_parser("export", help="Mine positives and negatives from processed videos")
    ds_export.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    ds_export.add_argument("--out-dir", type=Path, default=Path("data/presence_dataset"))
    ds_export.add_argument("--confidence-floor", type=float, default=0.7)
    ds_export.add_argument("--negatives-per-frame", type=int, default=2)
    ds_export.add_argument("--video-id", type=int, default=None,
                           help="Limit to one video ID; default exports all videos")

    # train subcommand
    train_p = subparsers.add_parser("train", help="Model training utilities")
    train_sub = train_p.add_subparsers(dest="train_command", required=True)
    train_presence = train_sub.add_parser("presence", help="Train MobileNetV3-Small presence classifier")
    train_presence.add_argument("--data", type=Path, default=Path("data/presence_dataset"))
    train_presence.add_argument("--out", type=Path, default=Path("models/presence_classifier.pt"))
    train_presence.add_argument("--epochs", type=int, default=8)
    train_presence.add_argument("--batch-size", type=int, default=64)
    train_presence.add_argument("--lr", type=float, default=1e-3)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "process":
        return _run_process(args)
    if args.command == "review":
        return _run_review(args)
    if args.command == "harness":
        return _run_harness(args)
    if args.command == "dataset":
        return _run_dataset(args)
    if args.command == "train":
        return _run_train(args)
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
        weights = Path("models/presence_classifier.pt")
        presence_clf = PresenceClassifier(weights_path=weights, device=config.device) if weights.exists() else None
        sampler = AdaptivePresenceSampler(
            video_path=args.video_path,
            reader_backend=config.reader_backend,
            device=config.device,
            presence_classifier=presence_clf,
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


def _run_harness(args: argparse.Namespace) -> int:
    if args.harness_command == "run":
        return _run_harness_run(args)
    if args.harness_command == "compare":
        return _run_harness_compare(args)
    return 2


def _run_harness_run(args: argparse.Namespace) -> int:
    import json
    import subprocess
    import sys
    from pathlib import Path as _P
    # Ensure project root is on sys.path so `tests` package is importable from the CLI
    _project_root = str(_P(__file__).parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from tests.regression.harness import HarnessConfig, run_corpus
    from tests.regression.report import write_json_report, write_markdown_report, AggregateReport
    from tests.regression.metrics import VideoMetrics
    from tests.regression.cross_video import DedupMetrics

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sha = "unknown"

    cfg = HarnessConfig(
        corpus_dir=args.corpus,
        output_dir=args.output_dir,
        git_sha=sha,
        db_path=args.db,
    )
    report = run_corpus(cfg)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / f"{sha}.json"
    md_path = args.reports_dir / f"{sha}.md"

    baseline_report = None
    if args.baseline and args.baseline.exists():
        raw = json.loads(args.baseline.read_text())
        baseline_report = AggregateReport(
            git_sha=raw["git_sha"],
            per_video=tuple(VideoMetrics(**v) for v in raw["per_video"]),
            dedup=DedupMetrics(**raw["dedup"]),
        )

    write_json_report(report, json_path)
    write_markdown_report(report, md_path, baseline=baseline_report)
    print(f"Wrote {json_path} and {md_path}")
    return 0


def _run_harness_compare(args: argparse.Namespace) -> int:
    import json
    from tests.regression.report import AggregateReport, write_markdown_report
    from tests.regression.metrics import VideoMetrics
    from tests.regression.cross_video import DedupMetrics

    def _load(path: Path) -> AggregateReport:
        raw = json.loads(path.read_text())
        return AggregateReport(
            git_sha=raw["git_sha"],
            per_video=tuple(VideoMetrics(**v) for v in raw["per_video"]),
            dedup=DedupMetrics(**raw["dedup"]),
        )

    baseline = _load(args.baseline)
    current = _load(args.current)
    out = args.out or args.current.with_suffix(".compare.md")
    write_markdown_report(current, out, baseline=baseline)
    print(f"Wrote {out}")
    return 0


def _run_dataset(args: argparse.Namespace) -> int:
    import sqlite3
    from .presence.training_data import export_dataset

    db_path: Path = args.db
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    if args.video_id is not None:
        video_ids = [args.video_id]
    else:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id FROM videos ORDER BY id").fetchall()
        video_ids = [r[0] for r in rows]

    if not video_ids:
        print("No videos found in database.")
        return 0

    total_pos = total_neg = 0
    for vid in video_ids:
        pos, neg = export_dataset(
            db_path=db_path,
            video_id=vid,
            out_dir=args.out_dir,
            confidence_floor=args.confidence_floor,
            negatives_per_frame=args.negatives_per_frame,
        )
        print(f"video {vid}: {pos} positives, {neg} negatives")
        total_pos += pos
        total_neg += neg

    print(f"\nTotal: {total_pos} positives, {total_neg} negatives → {args.out_dir}")
    if total_pos < 200:
        print("⚠  Fewer than 200 positives. Consider processing more videos before training.")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    if args.train_command == "presence":
        from .train.presence import train
        if not args.data.exists():
            print(f"Dataset not found: {args.data}", file=sys.stderr)
            print("Run `card-capture dataset export` first.", file=sys.stderr)
            return 1
        train(
            data_dir=args.data,
            out_path=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
