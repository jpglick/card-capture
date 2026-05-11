"""Per-video background model + per-quad novelty score.

The contract is simple: a real card replaces a rectangular patch of workspace.
Its interior pixels therefore differ from whatever was in that region of the
empty workspace. Anything that doesn't differ — an acrylic stand, a hinge, a
sticker on the table — is not a card, regardless of whether it has four
corner-like points.

This module is the foundation for v4.1's setup-agnostic gating. It is *only*
about appearance vs. the empty-workspace baseline; no learned model, no
heuristics about what a "card" looks like.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Corner = Tuple[float, float]


@dataclass
class BackgroundModel:
    """Grayscale mean of the workspace when no card is present.

    Build via `from_frames` from a list of "empty" frames (we use the
    sampler's background proxies; see `BackgroundModel.from_source_frame_paths`
    for the alternate build path used by the main process after detection)."""
    gray: np.ndarray  # float32 mean, shape (H, W)

    @classmethod
    def from_frames(cls, frames: Sequence[np.ndarray]) -> "BackgroundModel":
        if not frames:
            raise ValueError("BackgroundModel.from_frames requires at least one frame")
        acc: Optional[np.ndarray] = None
        n = 0
        target_shape: Optional[Tuple[int, int]] = None
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            if target_shape is None:
                target_shape = gray.shape[:2]
            elif gray.shape[:2] != target_shape:
                gray = cv2.resize(gray, (target_shape[1], target_shape[0]))
            if acc is None:
                acc = gray.astype(np.float32)
            else:
                acc = acc + gray.astype(np.float32)
            n += 1
        assert acc is not None
        return cls(gray=acc / float(n))

    @classmethod
    def from_source_frame_paths(cls, paths: Sequence[str], n: int = 30) -> Optional["BackgroundModel"]:
        """Build a model from the first `n` frames at the given paths.

        We use the chronologically-first detection source frames as a proxy
        for "empty workspace" when the sampler's in-process proxies aren't
        available (the producer subprocess populates them; the main process
        builds its own model here)."""
        loaded: List[np.ndarray] = []
        for p in paths[:n]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                loaded.append(img)
        if not loaded:
            return None
        return cls.from_frames(loaded)


def _polygon_mask(shape: Tuple[int, int], corners: Sequence[Corner]) -> np.ndarray:
    """Binary mask, 1 inside the quad, 0 outside. shape is (H, W)."""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in corners], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 1)
    return mask


def quad_novelty(
    frame_bgr: np.ndarray,
    corners: Sequence[Corner],
    bg: BackgroundModel,
) -> float:
    """Return a [0, 1]-clamped novelty score for the quad's interior.

    Computed as mean absolute grayscale difference between the frame's
    interior pixels and the bg model's interior pixels, divided by 255.

    If `frame_bgr` and `bg.gray` have different resolutions, the bg model
    is resized to the frame's size so coordinates align."""
    if len(corners) != 4:
        raise ValueError(f"expected 4 corners, got {len(corners)}")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    bg_gray = bg.gray
    h, w = gray.shape[:2]
    if bg_gray.shape[:2] != (h, w):
        bg_gray = cv2.resize(bg_gray.astype(np.float32), (w, h))
    mask = _polygon_mask((h, w), corners)
    if int(mask.sum()) == 0:
        return 0.0
    diff = cv2.absdiff(gray.astype(np.float32), bg_gray.astype(np.float32))
    mean_diff = float(diff[mask == 1].mean())
    return float(min(1.0, mean_diff / 255.0))


def is_quad_card_like(
    frame_bgr: np.ndarray,
    corners: Sequence[Corner],
    bg: BackgroundModel,
    threshold: float = 0.08,
) -> bool:
    """Return True when the quad's interior differs from the bg by ≥ threshold.

    Default 0.08 ≈ 20 grayscale levels on average. Calibrate against your
    workspace noise floor: typical empty-stand novelty is < 0.03; real-card
    novelty is ≥ 0.15 on the runs we have."""
    return quad_novelty(frame_bgr, corners, bg) >= threshold
