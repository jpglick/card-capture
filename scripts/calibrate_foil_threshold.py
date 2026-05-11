"""Foil threshold calibration via sweep over labeled canonical frames.

Loads foil vs non-foil fixtures, sweeps threshold values, identifies inflection point,
and reports calibrated DEFAULT_FOIL_THRESHOLD.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from card_capture.fusion.foil_detection import compute_laplacian_variance


@dataclass(frozen=True)
class ThresholdPoint:
    """A single operating point in foil threshold space."""
    threshold: float
    foil_true_positives: int
    foil_false_negatives: int
    non_foil_true_negatives: int
    non_foil_false_positives: int

    @property
    def foil_recall(self) -> float:
        """Fraction of foil samples correctly detected."""
        total = self.foil_true_positives + self.foil_false_negatives
        return self.foil_true_positives / total if total > 0 else 0.0

    @property
    def foil_precision(self) -> float:
        """Fraction of detected-as-foil samples that are actually foil."""
        total = self.foil_true_positives + self.non_foil_false_positives
        return self.foil_true_positives / total if total > 0 else 0.0

    @property
    def specificity(self) -> float:
        """Fraction of non-foil samples correctly identified as non-foil."""
        total = self.non_foil_true_negatives + self.non_foil_false_positives
        return self.non_foil_true_negatives / total if total > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Overall classification accuracy."""
        total = (self.foil_true_positives + self.foil_false_negatives +
                 self.non_foil_true_negatives + self.non_foil_false_positives)
        correct = self.foil_true_positives + self.non_foil_true_negatives
        return correct / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 score (harmonic mean of recall and precision)."""
        if self.foil_recall + self.foil_precision == 0:
            return 0.0
        return 2 * (self.foil_recall * self.foil_precision) / (
            self.foil_recall + self.foil_precision
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "threshold": self.threshold,
            "foil_true_positives": self.foil_true_positives,
            "foil_false_negatives": self.foil_false_negatives,
            "non_foil_true_negatives": self.non_foil_true_negatives,
            "non_foil_false_positives": self.non_foil_false_positives,
            "foil_recall": self.foil_recall,
            "foil_precision": self.foil_precision,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "f1": self.f1,
        }


def load_frames_from_directory(dir_path: Path) -> List[List[np.ndarray]]:
    """Load all PNG files from subdirectories as multi-frame groups.

    Each subdirectory is treated as a frame group. PNG files within the subdirectory
    are sorted and loaded as a sequence of frames.

    Args:
        dir_path: Path to directory containing subdirectories with PNG files

    Returns:
        List of frame groups, where each group is a list of BGR frames
    """
    if not dir_path.exists():
        return []

    frames_groups = []

    # Iterate over subdirectories
    subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
    for subdir in subdirs:
        png_files = sorted(subdir.glob("*.png"))
        if not png_files:
            continue

        frames = []
        for png_file in png_files:
            # Load frame
            frame = cv2.imread(str(png_file), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"Warning: Failed to load {png_file}")
                continue
            frames.append(frame)

        if frames:
            frames_groups.append(frames)

    return frames_groups


def sweep_foil_threshold(
    foil_frames_groups: List[List[np.ndarray]],
    non_foil_frames_groups: List[List[np.ndarray]],
    thresholds: List[float],
) -> List[ThresholdPoint]:
    """Sweep threshold values and compute confusion matrix stats.

    Args:
        foil_frames_groups: List of frame groups for foil cards
        non_foil_frames_groups: List of frame groups for non-foil cards
        thresholds: List of threshold values to test

    Returns:
        List of ThresholdPoint objects for each threshold
    """
    results = []

    for threshold in thresholds:
        foil_tp = 0
        foil_fn = 0
        non_foil_tn = 0
        non_foil_fp = 0

        # Test on foil samples
        for frames in foil_frames_groups:
            if len(frames) < 2:
                foil_fn += 1
                continue
            variance = compute_laplacian_variance(frames)
            if variance > threshold:
                foil_tp += 1
            else:
                foil_fn += 1

        # Test on non-foil samples
        for frames in non_foil_frames_groups:
            if len(frames) < 2:
                non_foil_tn += 1
                continue
            variance = compute_laplacian_variance(frames)
            if variance <= threshold:
                non_foil_tn += 1
            else:
                non_foil_fp += 1

        point = ThresholdPoint(
            threshold=threshold,
            foil_true_positives=foil_tp,
            foil_false_negatives=foil_fn,
            non_foil_true_negatives=non_foil_tn,
            non_foil_false_positives=non_foil_fp,
        )
        results.append(point)

    return results


def find_calibrated_threshold(
    points: List[ThresholdPoint],
) -> Optional[ThresholdPoint]:
    """Find the threshold that maximizes F1 score (inflection point).

    Args:
        points: List of ThresholdPoint objects

    Returns:
        ThresholdPoint with highest F1 score, or None if empty
    """
    if not points:
        return None
    return max(points, key=lambda p: p.f1)


def main():
    """CLI entry point for foil threshold calibration."""
    parser = argparse.ArgumentParser(
        description="Foil threshold calibration via fixture sweep"
    )
    parser.add_argument(
        "--foil-dir",
        type=Path,
        default=Path("tests/fixtures/foil/foil"),
        help="Directory containing foil subdirectories with frame groups",
    )
    parser.add_argument(
        "--non-foil-dir",
        type=Path,
        default=Path("tests/fixtures/foil/non_foil"),
        help="Directory containing non_foil subdirectories with frame groups",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0],
        help="Threshold values to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="foil_calibration_results.json",
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Load fixtures
    print(f"Loading foil fixtures from {args.foil_dir}...")
    foil_frames_groups = load_frames_from_directory(args.foil_dir)
    print(f"  Loaded {len(foil_frames_groups)} foil groups")

    print(f"Loading non-foil fixtures from {args.non_foil_dir}...")
    non_foil_frames_groups = load_frames_from_directory(args.non_foil_dir)
    print(f"  Loaded {len(non_foil_frames_groups)} non-foil groups")

    if not foil_frames_groups or not non_foil_frames_groups:
        print("Error: No fixtures found. Please ensure directories exist with PNG files.")
        return

    # Sweep thresholds
    print(f"\nSweeping {len(args.thresholds)} threshold values...")
    points = sweep_foil_threshold(
        foil_frames_groups,
        non_foil_frames_groups,
        args.thresholds,
    )

    # Find calibrated threshold
    calibrated = find_calibrated_threshold(points)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "all_points": [p.to_dict() for p in points],
        "calibrated_threshold": calibrated.threshold if calibrated else None,
        "calibrated_metrics": calibrated.to_dict() if calibrated else None,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_path}")
    print(f"Sweep complete: {len(points)} thresholds evaluated")

    if calibrated:
        print(f"\nCalibrated threshold: {calibrated.threshold:.2f}")
        print(f"  F1 Score: {calibrated.f1:.4f}")
        print(f"  Foil Recall: {calibrated.foil_recall:.4f}")
        print(f"  Foil Precision: {calibrated.foil_precision:.4f}")
        print(f"  Specificity (Non-foil): {calibrated.specificity:.4f}")
        print(f"  Accuracy: {calibrated.accuracy:.4f}")

    # Print all points for reference
    print("\nAll threshold points:")
    for p in sorted(points, key=lambda x: x.f1, reverse=True):
        print(
            f"  threshold={p.threshold:6.2f} | "
            f"f1={p.f1:.4f} | "
            f"foil_recall={p.foil_recall:.4f} | "
            f"spec={p.specificity:.4f} | "
            f"acc={p.accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
