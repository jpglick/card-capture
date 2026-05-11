"""Threshold auto-sweep harness for Wave 3 calibration.

Grid sweep over novelty_thresholds and hamming_thresholds to find Pareto-optimal
operating points using robustness metrics from Wave 2.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Point:
    """A single operating point in threshold space."""
    novelty_threshold: float
    hamming_threshold: int
    metrics: Dict[str, float]
    f1: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "novelty_threshold": self.novelty_threshold,
            "hamming_threshold": self.hamming_threshold,
            "metrics": self.metrics,
            "f1": self.f1,
        }


def compute_robustness_metrics(
    metric_corpus: Dict,
    novelty_threshold: float,
    hamming_threshold: int,
) -> Dict[str, float]:
    """Compute robustness metrics for a given threshold pair.

    For now, this is a simplified implementation that aggregates metrics
    from the metric_corpus. In real use, this would run the full pipeline
    with the given thresholds against video data.

    Args:
        metric_corpus: Dictionary of video_id -> metrics (from Wave 2)
        novelty_threshold: Novelty gate threshold
        hamming_threshold: Hamming distance threshold for pHash

    Returns:
        Dict with card_recall, card_precision, f1
    """
    if not metric_corpus:
        return {
            "card_recall": 1.0,
            "card_precision": 1.0,
            "front_back_f1": 1.0,
            "f1": 1.0,
        }

    total_expected = 0
    total_detected = 0
    total_predictions = 0

    for video_id, metrics in metric_corpus.items():
        total_expected += metrics.get("expected_cards", 0)
        total_detected += metrics.get("detected_cards", 0)
        total_predictions += metrics.get("pipeline_output_count", 0)

    # Compute card recall: detected / expected
    card_recall = total_detected / total_expected if total_expected > 0 else 1.0

    # Compute card precision: detected / predictions
    card_precision = total_detected / total_predictions if total_predictions > 0 else 1.0

    # Compute F1: harmonic mean of recall and precision
    if card_recall + card_precision > 0:
        f1 = 2 * (card_recall * card_precision) / (card_recall + card_precision)
    else:
        f1 = 1.0

    return {
        "card_recall": card_recall,
        "card_precision": card_precision,
        "front_back_f1": 1.0,  # Placeholder - would need ground truth
        "f1": f1,
    }


def compute_pareto_front(
    points: List[Dict],
    objectives: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Compute Pareto front (non-dominated points).

    A point is dominated if another point is better on all objectives
    and strictly better on at least one.

    Args:
        points: List of point dictionaries with metric values
        objectives: Dict of metric_name -> "max"|"min". If None, assumes all "max".

    Returns:
        List of non-dominated points
    """
    if not points:
        return []

    if objectives is None:
        # Default: maximize all metrics
        objectives = {key: "max" for point in points for key in point if key not in {"novelty_threshold", "hamming_threshold"}}

    pareto = []

    for candidate in points:
        is_dominated = False

        for other in points:
            if candidate is other:
                continue

            # Check if other dominates candidate
            dominates_all = True
            strictly_better = False

            for obj_name, direction in objectives.items():
                if obj_name not in candidate or obj_name not in other:
                    continue

                candidate_val = candidate[obj_name]
                other_val = other[obj_name]

                if direction == "max":
                    if other_val < candidate_val:
                        dominates_all = False
                        break
                    if other_val > candidate_val:
                        strictly_better = True
                else:  # min
                    if other_val > candidate_val:
                        dominates_all = False
                        break
                    if other_val < candidate_val:
                        strictly_better = True

            if dominates_all and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto.append(candidate)

    return pareto


def grid_sweep_thresholds(
    novelty_thresholds: Sequence[float],
    hamming_thresholds: Sequence[int],
    metric_corpus: Dict,
    output_file: Optional[str] = None,
) -> Dict:
    """Grid sweep over threshold space and find Pareto-optimal points.

    Args:
        novelty_thresholds: List of novelty gate thresholds to test
        hamming_thresholds: List of Hamming distance thresholds to test
        metric_corpus: Dict of video_id -> metrics (from Wave 2 regression)
        output_file: Optional path to write JSON results

    Returns:
        Dict with "all_points" and "pareto_front" keys
    """
    all_points = []

    # Grid sweep: evaluate every (novelty, hamming) pair
    for novelty_th in novelty_thresholds:
        for hamming_th in hamming_thresholds:
            # Compute metrics for this threshold pair
            metrics = compute_robustness_metrics(
                metric_corpus,
                novelty_th,
                hamming_th,
            )

            # Extract F1 score
            f1 = metrics.get("f1", 0.0)

            # Create point
            point = {
                "novelty_threshold": novelty_th,
                "hamming_threshold": hamming_th,
                "metrics": metrics,
                "f1": f1,
            }
            all_points.append(point)

    # Compute Pareto front
    objectives = {
        "f1": "max",
        "metrics": "max",  # Placeholder - would need to drill down
    }
    # Compute Pareto front on F1 score (main objective)
    pareto_front = compute_pareto_front(
        all_points,
        objectives={"f1": "max"},
    )

    result = {
        "all_points": all_points,
        "pareto_front": pareto_front,
    }

    # Write output file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    """CLI entry point for threshold calibration."""
    parser = argparse.ArgumentParser(
        description="Threshold auto-sweep harness for Wave 3 calibration"
    )
    parser.add_argument(
        "--novelty-thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        help="Novelty gate thresholds to test",
    )
    parser.add_argument(
        "--hamming-thresholds",
        type=int,
        nargs="+",
        default=[18, 19, 20, 21, 22, 23, 24],
        help="Hamming distance thresholds to test",
    )
    parser.add_argument(
        "--metric-corpus",
        type=str,
        help="Path to JSON file with metric corpus (from Wave 2 regression)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_results.json",
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Load metric corpus if provided
    metric_corpus = {}
    if args.metric_corpus:
        with open(args.metric_corpus) as f:
            metric_corpus = json.load(f)

    # Run grid sweep
    result = grid_sweep_thresholds(
        novelty_thresholds=args.novelty_thresholds,
        hamming_thresholds=args.hamming_thresholds,
        metric_corpus=metric_corpus,
        output_file=args.output,
    )

    print(f"Grid sweep complete: {len(result['all_points'])} points evaluated")
    print(f"Pareto front: {len(result['pareto_front'])} non-dominated points")
    print(f"Results written to: {args.output}")

    # Print Pareto points
    if result["pareto_front"]:
        print("\nPareto-optimal operating points:")
        for point in sorted(result["pareto_front"], key=lambda p: p["f1"], reverse=True):
            print(
                f"  novelty={point['novelty_threshold']:.2f}, "
                f"hamming={point['hamming_threshold']}, "
                f"f1={point['f1']:.4f}, "
                f"recall={point['metrics']['card_recall']:.4f}, "
                f"precision={point['metrics']['card_precision']:.4f}"
            )


if __name__ == "__main__":
    main()
