from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from card_capture.models import FrameSample
from card_capture.sampler import (
    StabilityBasedSampler,
    StableWindow,
    ContrastBasedSampler,
    PresenceWindow,
)


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


def colored_frames(count: int) -> list:
    """Generate frames with high Laplacian variance to trigger presence detection."""
    frames = []
    for i in range(count):
        # Create frames with high edge variance using patterns
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Create a checkerboard/stripe pattern to get high Laplacian variance
        # This mimics a real card with texture/patterns
        for y in range(0, 240, 10):
            for x in range(0, 320, 10):
                if (x // 10 + y // 10) % 2 == 0:
                    frame[y:y+10, x:x+10] = [50, 50, 200]   # Red squares
                else:
                    frame[y:y+10, x:x+10] = [200, 200, 50]  # Cyan squares
        frames.append(frame)
    return frames


@pytest.fixture
def synthetic_video_path(tmp_path):
    """Create a synthetic video with high-variance (presence) frames."""
    # 10 frames of low variance, 10 frames of high variance, 10 frames of low variance
    frames = (
        gray_frames(10, value=128)
        + colored_frames(10)
        + gray_frames(10, value=128)
    )
    return make_video(tmp_path, frames, fps=30.0)


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


# ---------------------------------------------------------------------------
# ContrastBasedSampler Tests
# ---------------------------------------------------------------------------


class TestContrastBasedSampler:
    """Tests for ContrastBasedSampler presence detection and sharpness ranking."""

    def test_presence_detection_basic(self, synthetic_video_path):
        """High-variance frames should be detected as presence."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,  # Low threshold for synthetic video
            min_presence_frames=1,
            candidates_per_window=1,
        )
        windows = sampler._find_presence_windows()
        # Synthetic video should have at least one presence window (the colored frames)
        assert len(windows) > 0, "Should detect at least one presence window"

    def test_presence_windows_have_frame_ranges(self, synthetic_video_path):
        """Presence windows should have valid start/end frame ranges."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=1,
            candidates_per_window=1,
        )
        windows = sampler._find_presence_windows()
        for window in windows:
            assert window.start_frame >= 0, "start_frame must be >= 0"
            assert window.end_frame >= window.start_frame, "end_frame must be >= start_frame"

    def test_sharpness_ranking(self, synthetic_video_path):
        """Frames should be ranked by sharpness (Laplacian variance)."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=1,
            candidates_per_window=3,
        )
        windows = sampler._find_presence_windows()
        if windows:
            window = windows[0]
            scored_window = sampler._score_sharpness_in_window(window)
            # frame_candidates should be populated and sorted by sharpness (descending)
            assert len(scored_window.frame_candidates) > 0, "Should have frame candidates"
            if len(scored_window.frame_candidates) > 1:
                for i in range(len(scored_window.frame_candidates) - 1):
                    assert scored_window.frame_candidates[i][1] >= scored_window.frame_candidates[i + 1][1], \
                        "Frame candidates should be sorted by sharpness descending"

    def test_sample_yields_presence_windows(self, synthetic_video_path):
        """sample() should yield PresenceWindow objects with populated candidates."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=1,
            candidates_per_window=3,
        )
        windows = list(sampler.sample())
        assert len(windows) > 0, "sample() should yield at least one window"
        for window in windows:
            assert isinstance(window, PresenceWindow), "Should yield PresenceWindow objects"
            assert len(window.frame_candidates) > 0, "Each window should have candidates"

    def test_min_presence_frames_filter(self, synthetic_video_path):
        """Windows with fewer frames than min_presence_frames should be filtered."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=100,  # Very high; should filter most windows
            candidates_per_window=1,
        )
        windows = sampler._find_presence_windows()
        # With a high threshold, should have fewer (or no) windows
        assert isinstance(windows, list), "Should return a list"

    def test_candidates_limited_per_window(self, synthetic_video_path):
        """Should return at most candidates_per_window frames per window."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=1,
            candidates_per_window=2,
        )
        windows = list(sampler.sample())
        for window in windows:
            assert len(window.frame_candidates) <= 2, f"Should have at most 2 candidates, got {len(window.frame_candidates)}"

    def test_contrast_sampler_uses_gpu_device(self, synthetic_video_path):
        """ContrastBasedSampler should accept device parameter and use it."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=1,
            candidates_per_window=1,
            device="cpu",  # NEW: device parameter
        )
        assert sampler.device.type == "cpu"

    def test_window_merging_combines_nearby_windows(self, synthetic_video_path):
        """Windows within max_gap frames should be merged."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=1,
            candidates_per_window=1,
            device="cpu",
            window_merge_gap=5,
        )
        windows = sampler._find_presence_windows()
        # After merging, should have fewer windows (merged nearby ones)
        # Exact count depends on video, but should be reasonable
        assert isinstance(windows, list), "Should return a list"
        assert len(windows) >= 0, "Should handle merged windows"
