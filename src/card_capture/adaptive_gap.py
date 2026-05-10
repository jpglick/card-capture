from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GapDistribution:
    p50_frames: int
    p95_frames: int
    recommended_gap_frames: int


def compute_session_gap_frames(
    inter_window_gaps_frames: Sequence[int],
    *,
    fps: float = 30.0,
    floor_seconds: float = 0.5,
    cap_seconds: float = 3.0,
    safety_pad_frames: int = 2,
) -> GapDistribution:
    floor = int(round(floor_seconds * fps))
    cap = int(round(cap_seconds * fps))

    if not inter_window_gaps_frames:
        return GapDistribution(p50_frames=0, p95_frames=0, recommended_gap_frames=floor)

    arr = np.asarray(list(inter_window_gaps_frames), dtype=np.float32)
    p50 = int(np.percentile(arr, 50))
    p95 = int(np.percentile(arr, 95))
    recommended = max(floor, min(cap, p95 + safety_pad_frames))
    return GapDistribution(p50_frames=p50, p95_frames=p95, recommended_gap_frames=recommended)
