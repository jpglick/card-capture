"""image_quality metric.

Returns an ImageQuality dataclass with:

- ``mean_ssim``: Mean Structural Similarity Index (SSIM) of each matched
  GT card's fused canonical image vs. the hand-picked reference frame.
- ``mean_psnr``: Mean Peak Signal-to-Noise Ratio (dB); reported for
  diagnostic purposes, not used as a gate metric.
- ``coverage``: Fraction of matched GT cards that have a reference frame
  available (0.0 – 1.0).

All three are ``None`` when no reference frames exist.

Reference frames are looked up at::

    <truth_path.parent>/reference_frames/<card_id>.png

Fused canonical images are read from ``card_instances.fused_image_path``
in the database.  Images with mismatched dimensions are resized to match the
fused image before SSIM computation (a warning is logged).
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.match import match_detections_to_truth
from harness.schema import TruthFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageQuality:
    """Result of the image_quality metric."""

    mean_ssim: float | None
    mean_psnr: float | None
    coverage: float  # 0.0 – 1.0


def image_quality(
    *, db_path: Path, truth_path: Path, video_id: str
) -> ImageQuality:
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
    ImageQuality with mean_ssim, mean_psnr, and coverage fields.
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
        ref_path = ref_dir / f"{pair.gt_card_id}.png"
        if not ref_path.exists():
            continue

        fused_path_str = fused_map.get(pair.detection_id)  # type: ignore[arg-type]
        if not fused_path_str:
            continue
        fused_path = Path(fused_path_str)
        if not fused_path.exists():
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
        return ImageQuality(mean_ssim=None, mean_psnr=None, coverage=coverage)

    coverage = len(ssim_vals) / matched_count if matched_count > 0 else 0.0
    return ImageQuality(
        mean_ssim=float(np.mean(ssim_vals)),
        mean_psnr=float(np.mean(psnr_vals)),
        coverage=coverage,
    )


def _load_fused_paths(db_path: Path) -> dict[int, str | None]:
    """Return mapping from card_instance.id → fused_image_path."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, fused_image_path FROM card_instances"
        ).fetchall()
    return {int(r["id"]): r["fused_image_path"] for r in rows}


def _load_grey(path: Path) -> np.ndarray:
    """Load an image file as an 8-bit greyscale numpy array."""
    # Use PIL to avoid a hard OpenCV dependency in the harness.
    from PIL import Image  # type: ignore[import-untyped]

    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)
