"""image_quality metric.

Returns a MetricResult with breakdown keys:

- ``mean_ssim``: Mean Structural Similarity Index (SSIM) of each matched
  GT card's fused canonical image vs. the hand-picked reference frame.
- ``mean_psnr``: Mean Peak Signal-to-Noise Ratio (dB); reported for
  diagnostic purposes, not used as a gate metric.
- ``coverage``: Fraction of matched GT cards that have a reference frame
  available (0.0 – 1.0).

All three are ``None`` when no reference frames exist (except ``coverage``
which defaults to 0.0).

Reference frames are looked up at::

    <truth_path.parent>/reference_frames/<card_id>.png

Fused canonical images are read from ``card_instances.fused_image_path``
in the database.  Images with mismatched dimensions are resized to match the
fused image before SSIM computation (a warning is logged).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from card_capture.data.connection import read_connection
from card_capture.data.sql_queries import HARNESS_FUSED_PATHS

from harness.match import match_detections_to_truth
from harness.metrics.types import MetricResult
from harness.schema import TruthFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ImageQuality:
    """Internal result of the image_quality computation."""

    mean_ssim: Optional[float]
    mean_psnr: Optional[float]
    coverage: float  # 0.0 – 1.0


def image_quality(
    *, db_path: Path, truth_path: Path, video_id: str
) -> MetricResult:
    """Compute mean SSIM and PSNR of fused canonicals vs. reference frames.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite``.
    truth_path:
        Path to ``truth.json``.
    video_id:
        String video identifier.

    Returns
    -------
    MetricResult with breakdown keys ``mean_ssim``, ``mean_psnr``, and ``coverage``.
    """
    from skimage.metrics import (  # type: ignore[import-untyped]
        peak_signal_noise_ratio,
        structural_similarity,
    )

    truth = TruthFile.model_validate_json(truth_path.read_text())
    pairs = match_detections_to_truth(db_path, truth, video_id)

    ref_dir = truth_path.parent / "reference_frames"
    fused_map = _load_fused_paths(db_path)

    ssim_vals: list[float] = []
    psnr_vals: list[float] = []
    matched_count = 0

    matched = [p for p in pairs if p.gt_card_id is not None and p.detection_id is not None]
    for pair in matched:
        matched_count += 1
        
        # Try side-aware reference frame first
        ref_path = None
        if pair.expected_side:
            side_name = pair.expected_side.lower()
            ref_path = ref_dir / f"{pair.gt_card_id}_{side_name}.png"
            
        if ref_path is None or not ref_path.exists():
            # Fallback to generic card_id.png
            ref_path = ref_dir / f"{pair.gt_card_id}.png"
            
        if not ref_path.exists():
            continue

        fused_path_str = fused_map.get(pair.detection_id)  # type: ignore[arg-type]
        if not fused_path_str:
            continue
        fused_path = Path(fused_path_str)
        if not fused_path.exists():
            # Robust fallback for fixtures: try relative to the database directory
            # assuming the standard v4 structure: <db_dir>/crops/<name>
            fallback_path = db_path.parent / "crops" / fused_path.name
            if fallback_path.exists():
                fused_path = fallback_path
            else:
                logger.warning("Fused image not found: %s", fused_path)
                continue

        ref_img = _load_grey(ref_path)
        fused_img = _load_grey(fused_path)

        if ref_img.shape != fused_img.shape:
            logger.warning(
                "Dimension mismatch for %s: ref=%s fused=%s — resizing ref",
                pair.gt_card_id,
                ref_img.shape,
                fused_img.shape,
            )
            from skimage.transform import resize  # type: ignore[import-untyped]

            ref_img = resize(
                ref_img,
                fused_img.shape,
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.uint8)

        ssim_val = float(
            structural_similarity(fused_img, ref_img, data_range=255)
        )
        psnr_val = float(
            peak_signal_noise_ratio(fused_img, ref_img, data_range=255)
        )
        ssim_vals.append(ssim_val)
        psnr_vals.append(psnr_val)

    if not ssim_vals:
        coverage = 0.0 if matched_count == 0 else len(ssim_vals) / matched_count
        return MetricResult(breakdown={"mean_ssim": None, "mean_psnr": None, "coverage": coverage})

    coverage = len(ssim_vals) / matched_count if matched_count > 0 else 0.0
    return MetricResult(
        breakdown={
            "mean_ssim": float(np.mean(ssim_vals)),
            "mean_psnr": float(np.mean(psnr_vals)),
            "coverage": coverage,
        }
    )


def _load_fused_paths(db_path: Path) -> dict[int, Optional[str]]:
    """Return mapping from card_instance.id → fused_image_path."""
    with read_connection(db_path) as conn:
        rows = conn.execute(HARNESS_FUSED_PATHS).fetchall()
    return {int(r[0]): r[1] for r in rows}


def _load_grey(path: Path) -> np.ndarray:
    """Load an image file as an 8-bit greyscale numpy array."""
    # Use PIL to avoid a hard OpenCV dependency in the harness.
    from PIL import Image  # type: ignore[import-untyped]

    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)
