"""Per-video adaptive threshold computation from online distributions."""

from typing import List
import numpy as np


class AdaptiveThresholdComputer:
    """Compute video-specific thresholds from observed in-video distributions.

    Supports:
    - Novelty threshold: p50 of candidate novelty scores, clipped ±20% of global
    - Hamming threshold: p75 of intra-track pHash distances * 1.2, clipped ±10% of global
    - Graceful fallback to global when <10 samples
    """

    MIN_SAMPLE_COUNT = 10

    def compute_novelty_threshold(
        self,
        novelty_scores: List[float],
        global_threshold: float,
    ) -> float:
        """Compute adaptive novelty threshold from p50 of observed scores.

        Args:
            novelty_scores: List of novelty scores observed in the video
            global_threshold: Global default threshold (fallback)

        Returns:
            Adaptive threshold: p50 clipped to [global * 0.8, global * 1.2]
            Falls back to global if < 10 samples
        """
        if len(novelty_scores) < self.MIN_SAMPLE_COUNT:
            return global_threshold

        p50 = float(np.median(novelty_scores))
        min_clip = global_threshold * 0.8
        max_clip = global_threshold * 1.2

        return float(np.clip(p50, min_clip, max_clip))

    def compute_hamming_threshold(
        self,
        distances: List[float],
        global_threshold: float,
    ) -> float:
        """Compute adaptive Hamming threshold from p75 of intra-track distances.

        Args:
            distances: List of Hamming distances within tracks
            global_threshold: Global default threshold (fallback)

        Returns:
            Adaptive threshold: p75 * 1.2 clipped to [global * 0.9, global * 1.1]
            Falls back to global if < 10 samples
        """
        if len(distances) < self.MIN_SAMPLE_COUNT:
            return global_threshold

        p75 = float(np.percentile(distances, 75))
        adapted = p75 * 1.2
        min_clip = global_threshold * 0.9
        max_clip = global_threshold * 1.1

        return float(np.clip(adapted, min_clip, max_clip))
