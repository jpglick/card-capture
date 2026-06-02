from __future__ import annotations

import numpy as np


def find_valley_splits(
    sobel_scores: list[float],
    delta_scores: list[float],
    frame_indices: list[int],
    valley_drop_ratio: float = 0.40,
    valley_min_width_frames: int = 3,
    delta_spike_ratio: float = 0.50,
) -> list[int]:
    """Return sorted unique frame indices where a card swap is detected.

    Two independent signals:
    - Sobel valley: score drops >= valley_drop_ratio from preceding peak and
      persists >= valley_min_width_frames frames before recovering.
      Split placed at the valley minimum frame.
    - Delta spike: frame-to-frame pixel delta >= delta_spike_ratio * max(delta_scores).
      Split placed at the spike frame.

    All three lists must be the same length. For frame 0, delta_score should be 0.0.
    """
    if not sobel_scores:
        return []

    if delta_scores and len(delta_scores) != len(sobel_scores):
        raise ValueError(
            f"delta_scores length ({len(delta_scores)}) must equal sobel_scores length ({len(sobel_scores)})"
        )
    if frame_indices and len(frame_indices) != len(sobel_scores):
        raise ValueError(
            f"frame_indices length ({len(frame_indices)}) must equal sobel_scores length ({len(sobel_scores)})"
        )

    split_frames: set[int] = set()

    # --- Sobel valley detection ---
    peak = sobel_scores[0]
    valley_start: int | None = None

    for i, score in enumerate(sobel_scores):
        if score > peak:
            peak = score

        in_valley = score < peak * (1.0 - valley_drop_ratio)

        if in_valley:
            if valley_start is None:
                valley_start = i
        else:
            if valley_start is not None:
                width = i - valley_start
                if width >= valley_min_width_frames:
                    min_idx = valley_start + int(np.argmin(sobel_scores[valley_start:i]))
                    split_frames.add(frame_indices[min_idx])
                valley_start = None

    # Handle valley that extends to end of signal
    if valley_start is not None:
        width = len(sobel_scores) - valley_start
        if width >= valley_min_width_frames:
            min_idx = valley_start + int(np.argmin(sobel_scores[valley_start:]))
            split_frames.add(frame_indices[min_idx])

    # --- Delta spike detection with peak clustering ---
    # A card swap produces a burst of high-delta frames (hand crossing camera).
    # We cluster consecutive threshold-exceeding frames and emit one split per
    # cluster at the peak, avoiding duplicate splits for the same physical event.
    if delta_scores:
        max_delta = max(delta_scores)
        if max_delta > 0:
            threshold = delta_spike_ratio * max_delta
            merge_window = max(5, valley_min_width_frames * 2)
            # Each cluster: list of (scan_idx, frame_index, delta)
            clusters: list[list[tuple[int, int, float]]] = []
            for scan_i, (delta, fi) in enumerate(zip(delta_scores, frame_indices)):
                if delta >= threshold:
                    if clusters and scan_i - clusters[-1][-1][0] <= merge_window:
                        clusters[-1].append((scan_i, fi, delta))
                    else:
                        clusters.append([(scan_i, fi, delta)])
            for cluster in clusters:
                _, peak_fi, _ = max(cluster, key=lambda x: x[2])
                split_frames.add(peak_fi)

    return sorted(split_frames)
