"""Synthetic dataset generation for ML iteration prior to real labels.

Renders simple 750×1050 RGB images with controlled invariants:

- **F/B** dataset: distinct background colours + text/layout hints that make
  front vs. back visually separable.
- **Dedup** dataset: per-cluster fixed motif (background colour + ellipse)
  with per-sample Gaussian noise, so embeddings cluster correctly.

All images are saved as PNG to *out_dir*.
"""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class FBItem:
    image_path: Path
    label: str  # "front" or "back"


@dataclass
class DedupItem:
    image_path: Path
    cluster_id: int


@dataclass
class DedupDataset:
    items: list[DedupItem]
    cluster_ids: list[int]


# ---------------------------------------------------------------------------
# F/B dataset
# ---------------------------------------------------------------------------

def generate_fb_dataset(*, out_dir: Path, n_per_class: int, seed: int) -> list[FBItem]:
    """Generate *n_per_class* front and *n_per_class* back card images.

    Returns a list of :class:`FBItem` with ``label`` set to ``"front"`` or
    ``"back"``.  The list is ordered ``[front_0, back_0, front_1, back_1, …]``.
    """
    random.seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[FBItem] = []
    for i in range(n_per_class):
        items.append(_make_fb_image(out_dir, i, "front"))
        items.append(_make_fb_image(out_dir, i, "back"))
    return items


def _make_fb_image(out_dir: Path, idx: int, side: str) -> FBItem:
    bg_color = (220, 200, 120) if side == "front" else (120, 130, 200)
    img = Image.new("RGB", (750, 1050), color=bg_color)
    draw = ImageDraw.Draw(img)
    if side == "front":
        draw.rectangle([(50, 700), (700, 1000)], outline=(0, 0, 0), width=4)
        draw.text((100, 750), f"FRONT {idx}", fill=(0, 0, 0))
    else:
        draw.rectangle([(80, 80), (670, 970)], outline=(0, 0, 0), width=2)
        draw.text((250, 500), "BACK", fill=(255, 255, 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.random() * 0.5))
    path = out_dir / f"fb_{side}_{idx}.png"
    img.save(path)
    return FBItem(image_path=path, label=side)


# ---------------------------------------------------------------------------
# Dedup dataset
# ---------------------------------------------------------------------------

def generate_dedup_dataset(
    *,
    out_dir: Path,
    n_clusters: int,
    samples_per_cluster: int,
    seed: int,
) -> DedupDataset:
    """Generate *n_clusters* × *samples_per_cluster* card images.

    Within each cluster images share a base colour and ellipse motif; small
    Gaussian noise is added per sample.  Returns a :class:`DedupDataset`.
    """
    random.seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[DedupItem] = []
    for cluster_id in range(n_clusters):
        base_color: tuple[int, int, int] = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200),
        )
        for s in range(samples_per_cluster):
            img = Image.new("RGB", (750, 1050), color=base_color)
            draw = ImageDraw.Draw(img)
            draw.ellipse(
                [(200, 400), (550, 750)],
                fill=(255 - base_color[0], 0, 0),
            )
            jitter = np.random.randn(1050, 750, 3) * 5
            arr = np.clip(
                np.array(img).astype(np.float32) + jitter, 0, 255
            ).astype(np.uint8)
            jittered = Image.fromarray(arr)
            path = out_dir / f"cluster_{cluster_id}_{s}.png"
            jittered.save(path)
            items.append(DedupItem(image_path=path, cluster_id=cluster_id))
    return DedupDataset(items=items, cluster_ids=list(range(n_clusters)))
