# src/card_capture/frame_quality.py
"""Cheap, warp-free quality scorers used to shortlist frames before the
expensive Kornia warp. Both are pure and independently testable."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple
import numpy as np

Point = Tuple[float, float]

# Card canonical short:long ratio (750x1050).
_TARGET_RATIO = 750.0 / 1050.0


def _order_corners(corners: Sequence[Point]) -> List[Point]:
    """Order 4 points counter-clockwise around their centroid (order-invariant)."""
    cx = sum(p[0] for p in corners) / 4.0
    cy = sum(p[1] for p in corners) / 4.0
    return sorted(corners, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(prev: Point, vert: Point, nxt: Point) -> float:
    """Interior angle at `vert` in degrees."""
    v1 = (prev[0] - vert[0], prev[1] - vert[1])
    v2 = (nxt[0] - vert[0], nxt[1] - vert[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def flatness_score(corners: Sequence[Point]) -> float:
    """Return how flat/fronto-parallel a detected card quad is, in [0, 1].

    Combines three deterministic terms:
      - aspect: closeness of the quad's short:long ratio to the 5:7 card ratio
      - side balance: opposite sides near-equal (no perspective foreshortening)
      - angles: interior angles near 90 degrees (rectangular, not sheared)
    Returns 0.0 for degenerate input (fewer than 4 points or zero area).
    """
    if corners is None or len(corners) != 4:
        return 0.0
    pts = _order_corners([(float(x), float(y)) for x, y in corners])
    sides = [_dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 1e-3:
        return 0.0

    s0, s1, s2, s3 = sides  # consecutive edges; (s0,s2) opposite, (s1,s3) opposite
    w = (s0 + s2) / 2.0
    h = (s1 + s3) / 2.0
    short, long_ = (w, h) if w <= h else (h, w)
    quad_ratio = short / long_  # in (0, 1]
    aspect_term = max(0.0, 1.0 - abs(quad_ratio - _TARGET_RATIO) / _TARGET_RATIO)

    bal0 = 1.0 - abs(s0 - s2) / (s0 + s2)
    bal1 = 1.0 - abs(s1 - s3) / (s1 + s3)
    side_term = (bal0 + bal1) / 2.0

    angles = [_angle(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]) for i in range(4)]
    angle_term = max(0.0, 1.0 - sum(abs(a - 90.0) for a in angles) / 4.0 / 90.0)

    score = 0.4 * aspect_term + 0.3 * side_term + 0.3 * angle_term
    return max(0.0, min(1.0, score))


def clarity_score_gpu(frame_hwc: np.ndarray, bbox: Tuple[int, int, int, int],
                      device: str = "auto") -> float:
    """Laplacian-variance sharpness of the card's bbox ROI, computed on `device`.

    `frame_hwc` is an HxWxC (BGR) uint8 array. `bbox` is (x0, y0, x1, y1) in
    pixel coords. Returns a raw variance >= 0 (higher = sharper); callers
    normalize per-track. Used as the cheap clarity signal before any warp.
    """
    import torch
    import torch.nn.functional as F
    import cv2

    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    h, w = frame_hwc.shape[:2]
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    roi = frame_hwc[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    dev = torch.device(device if device != "auto" else
                       ("mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu")))
    t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                          dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(t, kernel, padding=1)
    return float(torch.var(lap).item() * (255.0 ** 2))


def clarity_var_gpu_roi(frame_hwc, bbox: Tuple[int, int, int, int]):
    """Laplacian-variance sharpness of an ROI, kept on-device as a 0-dim tensor.

    `frame_hwc` is an HxWxC (BGR) tensor ALREADY resident on a torch device — the
    detect pass keeps full-res frames on the GPU, so this slices the bbox ROI in
    place with no full-frame download. Returns a 0-dim tensor that is NOT synced,
    so a caller can stack many detections and pull them off the GPU in a single
    `.cpu()`. Scaled by 255**2 to match `clarity_score_gpu`'s numpy path; only the
    per-track ranking matters, not the absolute value.
    """
    import torch
    import torch.nn.functional as F

    h, w = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    roi = frame_hwc[y0:y1, x0:x1]
    dev = roi.device
    if roi.numel() == 0:
        return torch.zeros((), device=dev)
    roi = roi.to(torch.float32)
    if roi.ndim == 3 and roi.shape[2] >= 3:
        # BGR -> luma (cv2 BGR2GRAY weights), matching clarity_score_gpu.
        gray = 0.114 * roi[..., 0] + 0.587 * roi[..., 1] + 0.299 * roi[..., 2]
    else:
        gray = roi if roi.ndim == 2 else roi[..., 0]
    gray = (gray / 255.0).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                          dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(gray, kernel, padding=1)
    return torch.var(lap) * (255.0 ** 2)


def appearance_grid_gpu_roi(frame_hwc, bbox: Tuple[int, int, int, int], size: int = 8):
    """Downscaled grayscale grid of the card's bbox ROI, kept on-device.

    `frame_hwc` is an HxWxC (BGR) GPU-resident tensor; `bbox` is (x0,y0,x1,y1).
    Returns a flat (size*size,) tensor of luma in [0,1] (NOT synced) — the input
    to an average hash. A fixed `size` means every detection's grid is the same
    shape, so a batch can be stacked and pulled off the GPU in a single sync.
    """
    import torch
    import torch.nn.functional as F

    h, w = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    roi = frame_hwc[y0:y1, x0:x1]
    dev = roi.device
    if roi.numel() == 0:
        return torch.zeros(size * size, device=dev)
    roi = roi.to(torch.float32)
    if roi.ndim == 3 and roi.shape[2] >= 3:
        gray = 0.114 * roi[..., 0] + 0.587 * roi[..., 1] + 0.299 * roi[..., 2]
    else:
        gray = roi if roi.ndim == 2 else roi[..., 0]
    small = F.interpolate(gray.unsqueeze(0).unsqueeze(0), size=(size, size),
                          mode="bilinear", align_corners=False)
    return small.reshape(-1) / 255.0


def ahash_from_grid(grid) -> int:
    """Average-hash a downscaled luma grid into a bit-packed int (1 bit/cell).

    Each cell is set when it is >= the grid mean. Bit `i` corresponds to cell
    `i` in row-major order. Two visually different cards differ in many bits;
    consecutive frames of the same card differ in few. Used to detect card
    swaps that happen in the same screen position (which motion-only tracking
    misses). See `ahash_hamming`.
    """
    g = np.asarray(grid, dtype=np.float32).ravel()
    if g.size == 0:
        return 0
    mean = float(g.mean())
    bits = 0
    for i, v in enumerate(g):
        if float(v) >= mean:
            bits |= (1 << i)
    return bits


def ahash_hamming(a: int, b: int) -> int:
    """Number of differing bits between two average hashes (0 = identical)."""
    return bin(int(a) ^ int(b)).count("1")
