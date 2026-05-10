from __future__ import annotations

import json as _json
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def sample_negative_patches(
    frame: np.ndarray,
    count: int,
    patch_size: int = 224,
    rng_seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Sample random patches from a frame as negative training examples."""
    h, w = frame.shape[:2]
    if h < patch_size or w < patch_size:
        return []
    rng = random.Random(rng_seed)
    out: List[np.ndarray] = []
    for _ in range(count):
        y = rng.randint(0, h - patch_size)
        x = rng.randint(0, w - patch_size)
        out.append(frame[y:y + patch_size, x:x + patch_size].copy())
    return out


def mine_positive_crops(
    frame: np.ndarray,
    corners_per_card: Sequence[Sequence[Tuple[float, float]]],
    pad_ratio: float = 0.05,
    target_size: int = 224,
) -> List[np.ndarray]:
    """For each set of card corners in the frame, extract an axis-aligned crop sized for training."""
    h, w = frame.shape[:2]
    out: List[np.ndarray] = []
    for corners in corners_per_card:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        x0 = max(0, int(min(xs) - pad_ratio * (max(xs) - min(xs))))
        x1 = min(w, int(max(xs) + pad_ratio * (max(xs) - min(xs))))
        y0 = max(0, int(min(ys) - pad_ratio * (max(ys) - min(ys))))
        y1 = min(h, int(max(ys) + pad_ratio * (max(ys) - min(ys))))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame[y0:y1, x0:x1]
        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        out.append(resized)
    return out


def export_dataset(
    db_path: Path,
    video_id: int,
    out_dir: Path,
    *,
    confidence_floor: float = 0.7,
    negatives_per_frame: int = 2,
    target_size: int = 224,
) -> Tuple[int, int]:
    """Iterate over a video's frames, write positives + negatives to disk.

    Returns (positive_count, negative_count).
    """
    from card_capture.storage import Storage
    storage = Storage(db_path)
    storage.initialize()
    out_pos = out_dir / "positives"
    out_neg = out_dir / "negatives"
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)

    pos_n = 0
    neg_n = 0
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT cv.id, cv.frame_index, cv.timestamp_ms, ef.source_frame_path,
                   cv.corners_json, cv.confidence
            FROM card_views cv
            LEFT JOIN evidence_frames ef ON ef.card_view_id = cv.id
            JOIN card_instances ci ON ci.id = cv.card_instance_id
            WHERE ci.video_id = ? AND cv.confidence >= ?
            ORDER BY cv.frame_index
            """,
            (video_id, confidence_floor),
        ).fetchall()

    rng_seed = 0
    for row in rows:
        frame_path = row["source_frame_path"]
        if not frame_path:
            continue
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        corners_list = _json.loads(row["corners_json"]) if row["corners_json"] else []
        if not corners_list or not corners_list[0]:
            continue
        corners_per_card = [corners_list] if isinstance(corners_list[0][0], (int, float)) else corners_list
        for crop in mine_positive_crops(frame, corners_per_card, target_size=target_size):
            cv2.imwrite(str(out_pos / f"v{video_id}_f{row['frame_index']}_p{pos_n}.jpg"), crop)
            pos_n += 1
        for patch in sample_negative_patches(frame, count=negatives_per_frame, patch_size=target_size, rng_seed=rng_seed):
            cv2.imwrite(str(out_neg / f"v{video_id}_f{row['frame_index']}_n{neg_n}.jpg"), patch)
            neg_n += 1
        rng_seed += 1

    return pos_n, neg_n
