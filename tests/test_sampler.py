from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from card_capture.models import FrameSample
from card_capture.sampler import StabilityBasedSampler, StableWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_video(tmp_path: Path, frames: list, fps: float = 30.0) -> Path:
    """Write a list of BGR numpy frames to a temporary .avi file."""
    path = tmp_path / "test.avi"
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"VideoWriter failed to open (codec unavailable?): {path}"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def gray_frames(count: int, value: int = 128) -> list:
    return [np.full((240, 320, 3), value, dtype=np.uint8) for _ in range(count)]


# ---------------------------------------------------------------------------
# _find_stable_windows
# ---------------------------------------------------------------------------

def test_finds_single_stable_window(tmp_path):
    """30 identical frames → one stable window detected."""
    path = make_video(tmp_path, gray_frames(30))
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 1
    assert isinstance(windows[0], StableWindow)
    assert windows[0].best_frame_index >= 0


def test_best_frame_index_is_source_frame_number(tmp_path):
    """best_frame_index must refer to actual source video frame position
    (passable directly to cv2.CAP_PROP_POS_FRAMES), not the scan counter."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 1
    # source_fps=30, scan_fps=10 → frame_step=3
    # frame 0 sets the initial diff reference; best_frame_index comes from
    # run_frames which starts accumulating at frame 3.
    valid_source_frames = set(range(3, 30, 3))
    assert windows[0].best_frame_index in valid_source_frames


def test_no_stable_windows_on_constant_motion(tmp_path):
    """Alternating black/white frames → no stable run reaches min_stable_frames."""
    frames = [
        np.full((240, 320, 3), 0 if i % 2 == 0 else 200, dtype=np.uint8)
        for i in range(30)
    ]
    path = make_video(tmp_path, frames)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 0


def test_finds_two_stable_windows_separated_by_motion(tmp_path):
    """Two distinct stable periods separated by a high-motion transition → 2 windows."""
    frames = gray_frames(15, value=64) + gray_frames(15, value=200)
    path = make_video(tmp_path, frames, fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 2
    assert windows[0].best_frame_index < windows[1].best_frame_index


def test_stable_window_dataclass_has_start_end_best(tmp_path):
    """StableWindow exposes start_frame, end_frame, best_frame_index, frame_candidates."""
    path = make_video(tmp_path, gray_frames(30))
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    w = windows[0]
    assert hasattr(w, "start_frame")
    assert hasattr(w, "end_frame")
    assert hasattr(w, "best_frame_index")
    assert hasattr(w, "frame_candidates")
    assert w.start_frame <= w.best_frame_index <= w.end_frame
    assert len(w.frame_candidates) >= 1


def test_frame_candidates_populated(tmp_path):
    """frame_candidates contains all (frame_index, sharpness) pairs in the window."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 1
    cands = windows[0].frame_candidates
    # Each entry is (frame_index, laplacian_variance)
    assert all(isinstance(fi, int) and isinstance(lv, float) for fi, lv in cands)
    # best_frame_index must be one of the candidates
    frame_indices = [fi for fi, _ in cands]
    assert windows[0].best_frame_index in frame_indices


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------

def test_sample_yields_candidates_per_window_frames(tmp_path):
    """sample() yields up to candidates_per_window FrameSamples per stable window."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3,
        candidates_per_window=3,
    )
    results = list(sampler.sample(path, sample_fps=5.0))

    # One stable window with 3 candidates requested
    assert len(results) == 3
    for s in results:
        assert isinstance(s, FrameSample)
        assert s.width == 320
        assert s.height == 240
        assert s.frame_index in set(range(0, 30, 3))


def test_candidates_per_window_one_yields_single_frame(tmp_path):
    """candidates_per_window=1 yields exactly one frame per stable window."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3,
        candidates_per_window=1,
    )
    results = list(sampler.sample(path, sample_fps=5.0))

    assert len(results) == 1
    assert isinstance(results[0], FrameSample)


def test_candidates_spread_across_window(tmp_path):
    """Multiple candidates from one window must not all have the same frame_index."""
    path = make_video(tmp_path, gray_frames(60), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3,
        candidates_per_window=5,
    )
    results = list(sampler.sample(path, sample_fps=5.0))

    assert len(results) >= 2
    frame_indices = [r.frame_index for r in results]
    assert len(set(frame_indices)) > 1, "candidates must be at distinct frame positions"


def test_sample_yields_empty_when_no_stable_windows(tmp_path):
    """sample() yields nothing if pass 1 finds no stable windows."""
    frames = [
        np.full((240, 320, 3), 0 if i % 2 == 0 else 200, dtype=np.uint8)
        for i in range(30)
    ]
    path = make_video(tmp_path, frames)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    results = list(sampler.sample(path, sample_fps=5.0))

    assert results == []


def test_sample_fps_argument_is_ignored(tmp_path):
    """sample_fps is accepted for interface compatibility but does not affect output."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3,
        candidates_per_window=1,
    )
    results_5 = list(sampler.sample(path, sample_fps=5.0))
    results_30 = list(sampler.sample(path, sample_fps=30.0))

    assert len(results_5) == len(results_30) == 1
    assert results_5[0].frame_index == results_30[0].frame_index


def test_raises_on_missing_video(tmp_path):
    sampler = StabilityBasedSampler()
    with pytest.raises(ValueError, match="Could not decode video"):
        list(sampler.sample(tmp_path / "nonexistent.avi", sample_fps=5.0))
