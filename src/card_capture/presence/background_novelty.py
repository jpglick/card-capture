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
    bgr: Optional[np.ndarray] = None  # float32 mean BGR, shape (H, W, 3), for Lab conversion

    @classmethod
    def from_frames(cls, frames: Sequence[np.ndarray]) -> "BackgroundModel":
        if not frames:
            raise ValueError("BackgroundModel.from_frames requires at least one frame")
        acc: Optional[np.ndarray] = None
        acc_bgr: Optional[np.ndarray] = None
        n = 0
        target_shape: Optional[Tuple[int, int]] = None
        for f in frames:
            # Ensure BGR format
            if f.ndim == 3 and f.shape[2] == 3:
                bgr = f
            else:
                # If grayscale or other format, convert to BGR
                if f.ndim == 2:
                    bgr = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = f

            # Resize if necessary
            if target_shape is None:
                target_shape = bgr.shape[:2]
            elif bgr.shape[:2] != target_shape:
                bgr = cv2.resize(bgr, (target_shape[1], target_shape[0]))

            # Accumulate grayscale
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if acc is None:
                acc = gray.astype(np.float32)
            else:
                acc = acc + gray.astype(np.float32)

            # Accumulate BGR
            if acc_bgr is None:
                acc_bgr = bgr.astype(np.float32)
            else:
                acc_bgr = acc_bgr + bgr.astype(np.float32)

            n += 1
        assert acc is not None and acc_bgr is not None
        return cls(gray=acc / float(n), bgr=acc_bgr / float(n))

    def get_mean_bgr(self) -> Optional[np.ndarray]:
        """Return the mean BGR image, computing it from gray if necessary.

        For Lab conversion, we need the color channels. If bgr was not stored
        (e.g., legacy models), this returns None."""
        return self.bgr

    @classmethod
    def from_source_frame_paths(cls, paths: Sequence[str], n: int = 30) -> Optional["BackgroundModel"]:
        """Build a model from the first `n` frames at the given paths.

        We use the chronologically-first detection source frames as a proxy
        for "empty workspace" when the sampler's in-process proxies aren't
        available (the producer subprocess populates them; the main process
        builds its own model here)."""
        loaded: List[np.ndarray] = []
        for p in paths[:n]:
            img = cv2.imread(p)  # Read in color (BGR) by default
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
    color_space: str = "grayscale",
    lab_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> float:
    """Return a [0, 1]-clamped novelty score for the quad's interior.

    Args:
        frame_bgr: Input frame in BGR format (or grayscale if 2D)
        corners: List of 4 (x, y) corner coordinates
        bg: BackgroundModel containing the reference appearance
        color_space: "grayscale" (default) or "lab"
        lab_weights: Tuple of (L_weight, a_weight, b_weight) for Lab mode.
                     Default (1.0, 0.5, 0.5) emphasizes luminance over chroma.

    If `color_space == "grayscale"`:
        Computed as mean absolute grayscale difference between the frame's
        interior pixels and the bg model's interior pixels, divided by 255.

    If `color_space == "lab"`:
        Computed as weighted mean absolute difference in Lab space:
        diff = L_weight * |L_frame - L_bg| + a_weight * |a_frame - a_bg| + b_weight * |b_frame - b_bg|
        Result is normalized by (L_weight * 100 + a_weight * 127 + b_weight * 127).

    If `frame_bgr` and `bg` have different resolutions, the bg model
    is resized to the frame's size so coordinates align."""
    if len(corners) != 4:
        raise ValueError(f"expected 4 corners, got {len(corners)}")

    if color_space == "lab":
        return _quad_novelty_lab(frame_bgr, corners, bg, lab_weights)
    elif color_space == "grayscale":
        return _quad_novelty_grayscale(frame_bgr, corners, bg)
    else:
        raise ValueError(f"unknown color_space: {color_space}")


def _quad_novelty_grayscale(
    frame_bgr: np.ndarray,
    corners: Sequence[Corner],
    bg: BackgroundModel,
) -> float:
    """Grayscale novelty (original implementation)."""
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


def _quad_novelty_lab(
    frame_bgr: np.ndarray,
    corners: Sequence[Corner],
    bg: BackgroundModel,
    lab_weights: Tuple[float, float, float],
) -> float:
    """Lab color novelty detection.

    Detects cards that match in luminance but differ in chroma.
    Weights: L (luminance, 0-100), a (red-green, -127 to +127), b (yellow-blue, -127 to +127).
    """
    # Ensure frame is BGR
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Lab novelty requires a 3-channel BGR frame")

    # Get mean BGR for background
    bg_bgr = bg.get_mean_bgr()
    if bg_bgr is None:
        raise ValueError("BackgroundModel has no BGR data; cannot compute Lab novelty. "
                        "Rebuild BackgroundModel with from_frames() on color frames.")

    h, w = frame_bgr.shape[:2]
    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = cv2.resize(bg_bgr.astype(np.float32), (w, h))

    # Convert to Lab
    frame_lab = cv2.cvtColor(frame_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    # Extract L, a, b channels
    frame_L = frame_lab[:, :, 0]
    frame_a = frame_lab[:, :, 1]
    frame_b = frame_lab[:, :, 2]

    bg_L = bg_lab[:, :, 0]
    bg_a = bg_lab[:, :, 1]
    bg_b = bg_lab[:, :, 2]

    # Compute weighted differences
    L_weight, a_weight, b_weight = lab_weights
    diff_L = L_weight * cv2.absdiff(frame_L, bg_L)
    diff_a = a_weight * cv2.absdiff(frame_a, bg_a)
    diff_b = b_weight * cv2.absdiff(frame_b, bg_b)

    # Sum to get total weighted difference
    diff_total = diff_L + diff_a + diff_b

    # Apply mask
    mask = _polygon_mask((h, w), corners)
    if int(mask.sum()) == 0:
        return 0.0

    mean_diff = float(diff_total[mask == 1].mean())

    # Normalize: OpenCV LAB uses 0-255 range for all channels
    # L ranges 0-255, a ranges 0-255, b ranges 0-255
    # Max weighted difference = L_weight * 255 + a_weight * 255 + b_weight * 255
    max_diff = (L_weight + a_weight + b_weight) * 255.0
    return float(min(1.0, mean_diff / max_diff))


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
