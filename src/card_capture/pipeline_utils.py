"""Algorithmic helpers used by Metaflow pipeline steps.

These functions were previously part of the retired pipeline.py monolith.
They live here so individual step modules can import them without depending
on the worker subsystem (card_capture.workers).
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants used by canonical selection
# ---------------------------------------------------------------------------

_CANONICAL_TARGET_FRAMES = 3
_CANONICAL_MAX_FRAMES = 4
_SAME_APPEARANCE_HAMMING_MAX = 8


# ---------------------------------------------------------------------------
# File / array utilities
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compress_array(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Image analysis helpers
# ---------------------------------------------------------------------------

def _glare_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def _laplacian_heatmap(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return lap.astype(np.float32)


def _side_textiness_score(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    margin_h = int(height * 0.15)
    margin_w = int(width * 0.15)
    inner = image[margin_h:height - margin_h, margin_w:width - margin_w]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(edges.mean() / 255.0)
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7,
    )
    ink_ratio = float(thresholded.mean() / 255.0)
    return edge_ratio + ink_ratio


def _appearance_vector(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin_h = int(height * 0.15)
    margin_w = int(width * 0.15)
    inner = image[margin_h:height - margin_h, margin_w:width - margin_w]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    small_std = float(small.std())
    if small_std > 1e-6:
        small /= small_std
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).astype(np.float32)
    hist = hist.flatten()
    hist_sum = float(hist.sum())
    if hist_sum > 1e-6:
        hist /= hist_sum
    vector = np.concatenate([small.flatten(), hist])
    norm = float(np.linalg.norm(vector))
    if norm > 1e-6:
        vector /= norm
    return vector


# ---------------------------------------------------------------------------
# Track scoring / selection
# ---------------------------------------------------------------------------

def adaptive_min_track_length(
    detection_count: int,
    inter_gap_frames: list[float],
    min_baseline: int = 3,
) -> int:
    """Compute adaptive min_track_length from inter-detection gaps."""
    if not inter_gap_frames:
        return min_baseline
    median_gap = float(np.median(inter_gap_frames))
    return max(min_baseline, int(median_gap * 3))


def _compute_quality_weighted_score(prepared: Any, max_length: int) -> float:
    """Composite track selection score: 0.3 * norm_length + 0.7 * mean_quality."""
    if max_length <= 0:
        norm_length = 0.0
    else:
        candidates = getattr(getattr(prepared, "track", None), "candidates", None) or []
        norm_length = len(candidates) / max_length
    mean_quality = float(getattr(prepared, "mean_quality_score", 0.0))
    return 0.3 * norm_length + 0.7 * mean_quality


def _entry_quality_total(entry: dict) -> float:
    quality_score = entry.get("quality_score")
    if quality_score is not None:
        return float(quality_score.total)
    candidate = entry.get("candidate")
    if candidate is not None:
        return float(candidate.score.total)
    return 0.0


def _select_canonical_entries(frame_entries: list[dict], deduplicator: Any) -> list[dict]:
    if not frame_entries:
        return []

    scored = sorted(frame_entries, key=_entry_quality_total, reverse=True)
    anchor = scored[0]
    anchor_hash = str(anchor["visual_hash"])

    for entry in frame_entries:
        entry["_hamming_to_anchor"] = deduplicator.hamming_distance(
            str(entry["visual_hash"]), anchor_hash
        )

    same_appearance = [
        entry for entry in frame_entries
        if int(entry["_hamming_to_anchor"]) <= _SAME_APPEARANCE_HAMMING_MAX
    ]

    target = min(_CANONICAL_TARGET_FRAMES, len(frame_entries))
    if len(same_appearance) < target:
        same_appearance = sorted(
            frame_entries,
            key=lambda e: (int(e["_hamming_to_anchor"]), -_entry_quality_total(e)),
        )[:min(_CANONICAL_MAX_FRAMES, len(frame_entries))]

    ranked = sorted(same_appearance, key=_entry_quality_total, reverse=True)
    selected: list[dict] = [ranked[0]]
    while len(selected) < min(target, len(ranked)):
        best_entry = None
        best_key = None
        for entry in ranked:
            if entry in selected:
                continue
            min_gap = min(
                abs(int(entry["candidate"].timestamp_ms) - int(prev["candidate"].timestamp_ms))
                for prev in selected
            )
            key = (min_gap, _entry_quality_total(entry))
            if best_key is None or key > best_key:
                best_key = key
                best_entry = entry
        if best_entry is None:
            break
        selected.append(best_entry)

    return selected


def _build_candidates(rows: list) -> list:
    """Build ScoredCandidate list from _DetectionEnvelope rows."""
    from .selector import ScoredCandidate
    from .models import QualityScore

    candidates = []
    for index, row in enumerate(rows):
        confidence = float(row.detection_packet.corner_detection.confidence)
        score = QualityScore(total=confidence, components={"confidence": round(confidence, 6)})
        corners = row.detection_packet.corner_detection.corners
        corner_list = [(float(pt[0]), float(pt[1])) for pt in corners]
        candidates.append(
            ScoredCandidate(
                detection_id=index,
                timestamp_ms=row.detection_packet.timestamp_ms,
                image_path=row.source_frame_path,
                score=score,
                corners=corner_list,
                frame_index=row.detection_packet.frame_index,
            )
        )
    return candidates
