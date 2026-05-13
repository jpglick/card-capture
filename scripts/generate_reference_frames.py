"""Generate reference frames for the golden set from truth.json windows.

Usage:
    python3 scripts/generate_reference_frames.py --truth-dir golden_set/videos --video-path tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV
"""
import argparse
import json
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    args = parser.parse_args()

    video_id = args.video_path.stem
    truth_path = args.truth_dir / video_id / "truth.json"
    if not truth_path.exists():
        print(f"Truth file not found: {truth_path}")
        return

    with open(truth_path) as f:
        truth = json.load(f)

    ref_dir = args.truth_dir / video_id / "reference_frames"
    ref_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        print(f"Could not open video: {args.video_path}")
        return

    for card in truth.get("expected_cards", []):
        card_id = card["card_id"]
        
        # Check front
        if card.get("front_present") and card.get("approx_front_window_ms"):
            start, end = card["approx_front_window_ms"]
            mid = (start + end) / 2
            _save_frame(cap, mid, ref_dir / f"{card_id}_front.png")
            
        # Check back
        if card.get("back_present") and card.get("approx_back_window_ms"):
            start, end = card["approx_back_window_ms"]
            mid = (start + end) / 2
            _save_frame(cap, mid, ref_dir / f"{card_id}_back.png")

    cap.release()
    print(f"Done. Reference frames saved to {ref_dir}")


def _save_frame(cap, timestamp_ms, out_path):
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(str(out_path), frame)
        print(f"Saved {out_path} at {timestamp_ms}ms")
    else:
        print(f"Failed to read frame at {timestamp_ms}ms")


if __name__ == "__main__":
    main()
