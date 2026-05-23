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

# _open_capture is imported at module level so it can be monkeypatched in tests.
# The two _laplacian_select_frames implementations below reference this name.
try:
    from .ingestion import _open_capture
except Exception:  # pragma: no cover — only fails in incomplete test environments
    _open_capture = None  # type: ignore[assignment]

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


# ---------------------------------------------------------------------------
# In-track sharpness selection
# ---------------------------------------------------------------------------

def _laplacian_select_frames(
    video_path,
    track_ranges: list,
    scan_stride: int = 4,
    top_k: int = 1,
    max_corner_gap: int = 15,
    decoded_frames=None,
) -> dict:
    """Single-pass Laplacian sharpness scan across all confirmed track time ranges.

    After tracking confirms which time windows contain cards, this function
    finds the sharpest source frames within each track's window. It decouples
    detection coverage (temporal stride, 3fps) from output quality (dense
    sharpness scan within confirmed holds, ~15fps for scan_stride=4 at 60fps).

    Uses ONE forward video pass covering all track ranges — no repeated seeks.
    Frames are downscaled to 640px wide before Laplacian computation to keep
    per-frame cost under 2ms.

    Args:
        video_path: Absolute path to the source video file.
        track_ranges: List of track dicts, each with:
            - "instance_id": str
            - "detections": list of (frame_index: int, corners: list) tuples
              sorted by frame_index. These are the YOLO-detected frames.
        scan_stride: Decode every Nth source frame within each range.
            4 → ~15fps for a 60fps source, covering a 2-second hold in ~30 frames.
        top_k: Number of sharpest frames to return per track (≥ 1).
        max_corner_gap: Max distance in source frames between a selected frame
            and its nearest YOLO detection when borrowing corners. If the gap
            exceeds this, fall back to the nearest detection frame itself (safe
            — always has corners from YOLO). 15 ≈ 0.25s at 60fps.
        decoded_frames: Optional dict mapping frame_index → np.ndarray. When
            provided, frames are read directly from this dict and VideoCapture
            is never opened (GPU fast path). Missing indices are silently skipped.

    Returns:
        Dict mapping instance_id → list of (frame_index, corners) tuples,
        length ≤ top_k, ordered sharpest-first. Corners are either the frame's
        own (if it was a YOLO detection) or borrowed from the nearest detection.
        Returns empty dict if track_ranges is empty or video cannot be opened.
    """
    if not track_ranges:
        return {}

    import bisect
    import cv2
    import numpy as np
    from pathlib import Path as _Path

    # Build per-track metadata and collect all frame indices to scan
    track_info: dict = {}
    all_scan_frames: set = set()

    for t in track_ranges:
        iid = t["instance_id"]
        dets = sorted(t.get("detections", []), key=lambda x: x[0])
        if not dets:
            continue
        first_frame, last_frame = dets[0][0], dets[-1][0]
        det_map = {fi: corners for fi, corners in dets}
        det_sorted = [fi for fi, _ in dets]
        scan_frames = set(range(first_frame, last_frame + 1, scan_stride))
        track_info[iid] = {
            "first": first_frame,
            "last": last_frame,
            "det_map": det_map,
            "det_sorted": det_sorted,
            "scan_frames": scan_frames,
            "scores": {},   # frame_index → laplacian variance
        }
        all_scan_frames |= scan_frames

    if not all_scan_frames:
        return {}

    # Compute Laplacian for every scan frame
    if decoded_frames is not None:
        # Fast path: frames already decoded — no VideoCapture needed
        for curr in all_scan_frames:
            frame = decoded_frames.get(curr)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            scale = 640 / w if w > 640 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            for ti in track_info.values():
                if curr in ti["scan_frames"]:
                    ti["scores"][curr] = lap_var
    else:
        # Slow path: sequential CPU decode via VideoCapture (non-CUDA fallback)
        max_scan_frame = max(all_scan_frames)
        try:
            capture = _open_capture(_Path(video_path))
        except Exception:
            return {}

        try:
            curr = 0
            while curr <= max_scan_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if curr in all_scan_frames:
                    h, w = frame.shape[:2]
                    # Downscale to 640px wide for speed (~1-2ms per frame)
                    scale = 640 / w if w > 640 else 1.0
                    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    for ti in track_info.values():
                        if curr in ti["scan_frames"]:
                            ti["scores"][curr] = lap_var
                curr += 1
        finally:
            capture.release()

    # Select top_k sharpest frames per track; map each to corners
    result: dict = {}
    for iid, ti in track_info.items():
        if not ti["scores"]:
            result[iid] = []
            continue

        ranked = sorted(ti["scores"].items(), key=lambda x: -x[1])
        selected = []
        for frame_idx, _ in ranked:
            if len(selected) >= top_k:
                break
            det_sorted = ti["det_sorted"]
            if not det_sorted:
                continue

            # Find nearest YOLO detection to borrow corners from
            pos = bisect.bisect_left(det_sorted, frame_idx)
            candidates = []
            if pos < len(det_sorted):
                candidates.append(det_sorted[pos])
            if pos > 0:
                candidates.append(det_sorted[pos - 1])
            nearest = min(candidates, key=lambda f: abs(f - frame_idx))

            if abs(nearest - frame_idx) <= max_corner_gap:
                # Close enough — use this frame with borrowed corners
                selected.append((frame_idx, ti["det_map"][nearest]))
            else:
                # Too far — fall back to the nearest detection frame itself
                selected.append((nearest, ti["det_map"][nearest]))

        result[iid] = selected

    return result


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


# ---------------------------------------------------------------------------
# In-track sharpness selection
# ---------------------------------------------------------------------------

def _laplacian_select_frames(
    video_path,
    track_ranges: list,
    scan_stride: int = 4,
    top_k: int = 1,
    max_corner_gap: int = 15,
    decoded_frames=None,
) -> dict:
    """Single-pass Laplacian sharpness scan across all confirmed track time ranges.

    After tracking confirms which time windows contain cards, this function
    finds the sharpest source frames within each track's window. It decouples
    detection coverage (temporal stride, 3fps) from output quality (dense
    sharpness scan within confirmed holds, ~15fps for scan_stride=4 at 60fps).

    Uses ONE forward video pass covering all track ranges — no repeated seeks.
    Frames are downscaled to 640px wide before Laplacian computation to keep
    per-frame cost under 2ms.

    Args:
        video_path: Absolute path to the source video file.
        track_ranges: List of track dicts, each with:
            - "instance_id": str
            - "detections": list of (frame_index: int, corners: list) tuples
              sorted by frame_index. These are the YOLO-detected frames.
        scan_stride: Decode every Nth source frame within each range.
            4 → ~15fps for a 60fps source, covering a 2-second hold in ~30 frames.
        top_k: Number of sharpest frames to return per track (≥ 1).
        max_corner_gap: Max distance in source frames between a selected frame
            and its nearest YOLO detection when borrowing corners. If the gap
            exceeds this, fall back to the nearest detection frame itself (safe
            — always has corners from YOLO). 15 ≈ 0.25s at 60fps.
        decoded_frames: Optional dict mapping frame_index → np.ndarray. When
            provided, frames are read directly from this dict and VideoCapture
            is never opened (GPU fast path). Missing indices are silently skipped.

    Returns:
        Dict mapping instance_id → list of (frame_index, corners) tuples,
        length ≤ top_k, ordered sharpest-first. Corners are either the frame's
        own (if it was a YOLO detection) or borrowed from the nearest detection.
        Returns empty dict if track_ranges is empty or video cannot be opened.
    """
    if not track_ranges:
        return {}

    import bisect
    import cv2
    import numpy as np
    from pathlib import Path as _Path

    # Build per-track metadata and collect all frame indices to scan
    track_info: dict = {}
    all_scan_frames: set = set()

    for t in track_ranges:
        iid = t["instance_id"]
        dets = sorted(t.get("detections", []), key=lambda x: x[0])
        if not dets:
            continue
        first_frame, last_frame = dets[0][0], dets[-1][0]
        det_map = {fi: corners for fi, corners in dets}
        det_sorted = [fi for fi, _ in dets]
        scan_frames = set(range(first_frame, last_frame + 1, scan_stride))
        track_info[iid] = {
            "first": first_frame,
            "last": last_frame,
            "det_map": det_map,
            "det_sorted": det_sorted,
            "scan_frames": scan_frames,
            "scores": {},   # frame_index → laplacian variance
        }
        all_scan_frames |= scan_frames

    if not all_scan_frames:
        return {}

    # Compute Laplacian for every scan frame
    if decoded_frames is not None:
        # Fast path: frames already decoded — no VideoCapture needed
        for curr in all_scan_frames:
            frame = decoded_frames.get(curr)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            scale = 640 / w if w > 640 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            for ti in track_info.values():
                if curr in ti["scan_frames"]:
                    ti["scores"][curr] = lap_var
    else:
        # Slow path: sequential CPU decode via VideoCapture (non-CUDA fallback)
        max_scan_frame = max(all_scan_frames)
        try:
            capture = _open_capture(_Path(video_path))
        except Exception:
            return {}

        try:
            curr = 0
            while curr <= max_scan_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if curr in all_scan_frames:
                    h, w = frame.shape[:2]
                    # Downscale to 640px wide for speed (~1-2ms per frame)
                    scale = 640 / w if w > 640 else 1.0
                    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    for ti in track_info.values():
                        if curr in ti["scan_frames"]:
                            ti["scores"][curr] = lap_var
                curr += 1
        finally:
            capture.release()

    # Select top_k sharpest frames per track; map each to corners
    result: dict = {}
    for iid, ti in track_info.items():
        if not ti["scores"]:
            result[iid] = []
            continue

        ranked = sorted(ti["scores"].items(), key=lambda x: -x[1])
        selected = []
        for frame_idx, _ in ranked:
            if len(selected) >= top_k:
                break
            det_sorted = ti["det_sorted"]
            if not det_sorted:
                continue

            # Find nearest YOLO detection to borrow corners from
            pos = bisect.bisect_left(det_sorted, frame_idx)
            candidates = []
            if pos < len(det_sorted):
                candidates.append(det_sorted[pos])
            if pos > 0:
                candidates.append(det_sorted[pos - 1])
            nearest = min(candidates, key=lambda f: abs(f - frame_idx))

            if abs(nearest - frame_idx) <= max_corner_gap:
                # Close enough — use this frame with borrowed corners
                selected.append((frame_idx, ti["det_map"][nearest]))
            else:
                # Too far — fall back to the nearest detection frame itself
                selected.append((nearest, ti["det_map"][nearest]))

        result[iid] = selected

    return result


# ---------------------------------------------------------------------------
# GPU frame decode
# ---------------------------------------------------------------------------

try:
    import decord as decord  # noqa: F401
except (ImportError, OSError) as _decord_import_error:
    print(f"[pipeline_utils] decord import failed: {type(_decord_import_error).__name__}: {_decord_import_error}", flush=True)
    decord = None  # type: ignore[assignment]
except Exception as _decord_import_error:
    print(f"[pipeline_utils] decord import failed (unexpected): {type(_decord_import_error).__name__}: {_decord_import_error}", flush=True)
    decord = None  # type: ignore[assignment]


def decode_frames_gpu(video_path, indices: list) -> dict:
    """Decode specific frames via NVDEC; return {frame_index: np.ndarray}.

    Hard-fails if GPU context unavailable and CC_CUDA_ALLOW_CPU_FALLBACK is not set.
    """
    import os
    if not indices:
        return {}

    allow_fallback = os.environ.get("CC_CUDA_ALLOW_CPU_FALLBACK", "0") == "1"

    # decord is None when the library couldn't be imported (e.g. missing NVIDIA driver
    # on a dev machine). Fall back to OpenCV sequential decode if allowed.
    if decord is None:
        import sys as _sys, importlib as _il
        _in_modules = _sys.modules.get('decord')
        try:
            _fresh = _il.import_module('decord')
            _fresh_repr = repr(_fresh)
        except Exception as _fe:
            _fresh_repr = f"FAILED: {type(_fe).__name__}: {_fe}"
        # __file__ tells us which copy of pipeline_utils Python actually loaded
        import card_capture.pipeline_utils as _self
        _diag = (
            f"pipeline_utils.__file__={_self.__file__}; "
            f"sys.modules['decord']={_in_modules!r}; "
            f"fresh import={_fresh_repr}"
        )
        if allow_fallback:
            return _decode_frames_opencv(video_path, indices)
        raise RuntimeError(f"decode_frames_gpu: decord is None. {_diag}")

    # decord.gpu(0) succeeds even when decord was compiled without NVDEC;
    # the real failure surfaces when VideoReader tries to open the GPU context.
    try:
        ctx = decord.gpu(0)
    except Exception:
        if allow_fallback:
            ctx = decord.cpu(0)
        else:
            raise RuntimeError(
                "decode_frames_gpu requires NVDEC (decord GPU context). "
                "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to allow CPU fallback "
                "in dev/test environments."
            )

    sorted_indices = sorted(set(indices))
    try:
        vr = decord.VideoReader(str(video_path), ctx=ctx)
    except Exception as e:
        if ("CUDA not enabled" in str(e) or "cuda" in str(e).lower()) and allow_fallback:
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        else:
            raise RuntimeError(
                f"decord GPU VideoReader failed: {e}. "
                "Ensure decord is built with USE_CUDA=ON (see Dockerfile.cuda). "
                "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to allow CPU decode in dev environments."
            ) from e
    frames = vr.get_batch(sorted_indices)
    result = {idx: frames[i].asnumpy() for i, idx in enumerate(sorted_indices)}
    del frames
    return result


def _decode_frames_opencv(video_path, indices: list) -> dict:
    """Sequential CPU fallback for decode_frames_gpu when decord is unavailable."""
    import cv2
    sorted_indices = sorted(set(indices))
    if not sorted_indices:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    result: dict = {}
    curr = 0
    max_idx = sorted_indices[-1]
    idx_set = set(sorted_indices)
    while curr <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if curr in idx_set:
            result[curr] = frame
        curr += 1
    cap.release()
    return result


def _compute_laplacian_scan_indices(track_ranges: list, scan_stride: int) -> set:
    """Return the set of frame indices that _laplacian_select_frames will scan."""
    all_scan_frames: set = set()
    for t in track_ranges:
        dets = sorted(t.get("detections", []), key=lambda x: x[0])
        if not dets:
            continue
        first_frame, last_frame = dets[0][0], dets[-1][0]
        all_scan_frames |= set(range(first_frame, last_frame + 1, scan_stride))
    return all_scan_frames
