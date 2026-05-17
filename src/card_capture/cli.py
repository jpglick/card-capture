from __future__ import annotations

from card_capture._warnings import install as _install_warning_filters
_install_warning_filters()

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .detectors import CardcaptorUltralyticsDetector, FakeCardDetector, probe_torch_device_status
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
    process.add_argument("--detector", choices=["docaligner", "fake"], default=None, help="Detector backend")
    process.add_argument(
        "--presence-threshold", type=float, default=None,
        help="Classifier confidence threshold for card presence (0–1). Default: 0.5. "
             "Raise to reduce phantoms; lower to increase recall.",
    )
    process.add_argument("--tracker-backend", choices=["botsort", "bytetrack"], default=None)
    process.add_argument("--fast-scan-fps", type=float, default=None, dest="fast_scan_fps")
    process.add_argument("--confirm-scan-fps", type=float, default=None, dest="confirm_scan_fps")
    process.add_argument("--valley-drop-ratio", type=float, default=None, dest="valley_drop_ratio")
    process.add_argument("--valley-min-width-frames", type=int, default=None)
    process.add_argument("--delta-spike-ratio", type=float, default=None, dest="delta_spike_ratio")
    process.add_argument("--centroid-jump-ratio", type=float, default=None, dest="centroid_jump_ratio")
    process.add_argument("--centroid-jump-frames", type=int, default=None, dest="centroid_jump_frames")
    process.add_argument("--reid-distance-threshold", type=float, default=None, dest="reid_distance_threshold")
    process.add_argument(
        "--pipeline",
        choices=["metaflow", "monolith"],
        default="metaflow",
        help="Which pipeline architecture to run. The 'monolith' path is deprecated and will be removed in Wave 5. Use --pipeline metaflow (the default).",
    )
    process.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        dest="resume_run_id",
        help="Resume a previous Metaflow run from its first failed step. "
             "Pass the run-id printed at the start of the failed run, e.g. 1778729424981206. "
             "Omit to start a fresh run.",
    )

    review = subparsers.add_parser("review", help="Start the local review UI")
    review.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8000)

    harness_p = subparsers.add_parser("harness", help="Run regression harness against golden corpus")
    harness_p.add_argument("harness_args", nargs=argparse.REMAINDER)

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

    # sampler subcommand
    sampler_p = subparsers.add_parser("sampler", help="Sampler diagnostic utilities")
    sampler_sub = sampler_p.add_subparsers(dest="sampler_command", required=True)
    sampler_sessions = sampler_sub.add_parser(
        "sessions",
        help="Scan a video and report how many sessions the tracker would create",
    )
    sampler_sessions.add_argument("video_path", type=Path)
    sampler_sessions.add_argument("--config", type=Path, default=Path("card_capture_config.json"))
    sampler_sessions.add_argument(
        "--expected", type=int, default=None,
        help="Expected number of unique cards; shown in summary for comparison",
    )

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
    if args.command == "sampler":
        return _run_sampler(args)
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

    # Apply CLI overrides for new segmentation flags
    for attr in (
        "detector", "tracker_backend", "fast_scan_fps", "confirm_scan_fps",
        "valley_drop_ratio", "valley_min_width_frames", "delta_spike_ratio",
        "centroid_jump_ratio", "centroid_jump_frames", "reid_distance_threshold",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(config, attr, val)

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
        presence_threshold = getattr(args, "presence_threshold", None) or 0.5
        sampler = AdaptivePresenceSampler(
            video_path=args.video_path,
            reader_backend=config.reader_backend,
            device=config.device,
            presence_weights_path=weights if weights.exists() else None,
            presence_threshold=presence_threshold,
        )

    if getattr(args, "pipeline", "metaflow") == "metaflow":
        import os
        import subprocess
        # Metaflow spawns a subprocess per step; those subprocesses won't find
        # the `pipeline` package (repo root, not installed) unless we pass it
        # through PYTHONPATH so Metaflow propagates it to each step worker.
        repo_root = str(Path(__file__).parent.parent.parent)
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        resume_run_id = getattr(args, "resume_run_id", None)
        if resume_run_id:
            cmd = [
                sys.executable, "-m", "pipeline.card_capture_flow",
                "resume", resume_run_id,
            ]
        else:
            cmd = [
                sys.executable, "-m", "pipeline.card_capture_flow", "run",
                "--video", str(args.video_path),
                "--output-dir", str(args.output_dir),
                "--db", str(args.db),
                "--detector", config.detector,
            ]
        res = subprocess.run(cmd, env=env)
        return res.returncode


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
    from harness.cli import harness as harness_group
    import sys
    
    # We find where 'harness' appeared in sys.argv and pass everything after it to click.
    try:
        harness_idx = sys.argv.index("harness")
        click_args = sys.argv[harness_idx + 1:]
    except ValueError:
        click_args = []
        
    try:
        # Use standalone_mode=False so it doesn't sys.exit(0) on success
        harness_group.main(args=click_args, standalone_mode=False)
    except Exception as e:
        # Click's Abort or Exit might be caught here if not handled by standalone_mode
        if hasattr(e, "exit_code") and e.exit_code == 0: # type: ignore
            return 0
        print(f"Harness error: {e}")
        return 1
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



def _run_sampler(args: argparse.Namespace) -> int:
    if args.sampler_command == "sessions":
        return _run_sampler_sessions(args)
    return 2


def _run_sampler_sessions(args: argparse.Namespace) -> int:
    """Scan a video with AdaptivePresenceSampler and report session split points.

    This runs only the scan + window-build phases (no ML inference, no frame
    decoding) so it completes in ~35 s instead of the full pipeline's 2+ min.
    """
    from .adaptive_gap import compute_session_gap_frames
    from .config import load_config

    config = load_config(args.config)
    video_path = args.video_path.resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    weights_path = Path("models/presence_classifier.pt")
    sampler = AdaptivePresenceSampler(
        presence_weights_path=weights_path if weights_path.exists() else None,
        presence_threshold=0.5,
        fast_scan_fps=getattr(config, 'fast_scan_fps', 15.0),
        valley_drop_ratio=getattr(config, 'valley_drop_ratio', 0.40),
        valley_min_width_frames=getattr(config, 'valley_min_width_frames', 3),
        delta_spike_ratio=getattr(config, 'delta_spike_ratio', 0.60),
    )

    print(f"Scanning {video_path.name} …")
    import time
    t0 = time.time()
    scan_frames = sampler._scan_video(video_path)
    sampler._scan_frames = scan_frames

    # Use the same split computation as the pipeline sampler path.
    valley_splits = sampler._compute_valley_splits(scan_frames)
    sampler.last_valley_splits = valley_splits

    windows = sampler._build_windows(scan_frames, forced_splits=valley_splits)
    elapsed = time.time() - t0

    fast_scan_count = getattr(sampler, 'last_scan_frame_count', len(scan_frames))
    print(
        f"Scan + window build: {elapsed:.1f}s | "
        f"fast_scan_frames={len(scan_frames)} | "
        f"presence_windows={len(windows)}"
    )
    if valley_splits:
        print(f"Valley splits ({len(valley_splits)}): frame indices {valley_splits}")
        print("  (sobel valleys + delta spikes — these forced window boundaries)")
    else:
        print("Valley splits: none detected")

    if not windows:
        print("No presence windows found — nothing would be tracked.")
        return 0

    # Compute inter-window gaps (same logic as pipeline.py)
    inter_window_gaps: list[int] = []
    for i in range(1, len(windows)):
        gap = windows[i].start_frame - windows[i - 1].end_frame
        if gap > 0:
            inter_window_gaps.append(gap)

    fps = getattr(sampler, "last_source_fps", None) or 30.0
    gap_dist = compute_session_gap_frames(inter_window_gaps, fps=fps)
    effective_gap = gap_dist.recommended_gap_frames
    null_patience = config.null_patience_frames
    print(
        f"\nGap stats  p50={gap_dist.p50_frames}f  p95={gap_dist.p95_frames}f  "
        f"recommended={effective_gap}f ({effective_gap/fps:.1f}s)  "
        f"null_patience={null_patience}f (was capping to {min(null_patience, effective_gap)}f before fix)"
    )

    # Simulate session boundaries.
    # A new session starts when: (a) the inter-window gap exceeds the recommended
    # gap threshold, OR (b) a valley split falls between the two windows (meaning
    # a vision signal detected a card swap regardless of gap size).
    valley_split_set = set(valley_splits)
    sessions: list[list[object]] = []
    current_session: list[object] = []
    for i, w in enumerate(windows):
        if i == 0:
            current_session.append(w)
            continue
        gap = w.start_frame - windows[i - 1].end_frame
        prev_end = windows[i - 1].end_frame
        valley_in_gap = any(prev_end <= vs <= w.start_frame for vs in valley_split_set)
        if gap > effective_gap or valley_in_gap:
            sessions.append(current_session)
            current_session = [w]
        else:
            current_session.append(w)
    if current_session:
        sessions.append(current_session)

    print(f"\nSessions predicted: {len(sessions)}")
    for idx, sess in enumerate(sessions, 1):
        first = sess[0]
        last = sess[-1]
        print(
            f"  Session {idx}: windows={len(sess)}  "
            f"frames {first.start_frame}–{last.end_frame}  "
            f"({first.start_frame/fps:.1f}s – {last.end_frame/fps:.1f}s)"
        )

    if args.expected is not None:
        diff = len(sessions) - args.expected
        status = "✅ matches" if diff == 0 else (f"⚠️  +{diff} extra" if diff > 0 else f"⚠️  {diff} missing")
        print(f"\nExpected {args.expected} unique cards → {status}")

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
