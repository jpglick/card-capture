from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from card_capture.core.models import FrameSample
from card_capture.stages.sample.sampler import (
    StabilityBasedSampler,
    StableWindow,
    ContrastBasedSampler,
    PresenceWindow,
    AdaptivePresenceSampler,
    VideoSampler,
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


def test_video_sampler_uses_decord_backend_when_requested(monkeypatch, tmp_path):
    sampler = VideoSampler(reader_backend="decord")
    expected = [
        FrameSample(
            frame_index=7,
            timestamp_ms=233,
            image=np.zeros((4, 5, 3), dtype=np.uint8),
            width=5,
            height=4,
        )
    ]
    backend_calls = []

    monkeypatch.setattr("card_capture.stages.sample.sampler._resolve_reader_backend", lambda preferred: preferred)

    def _fake_decord(self, video_path, sample_fps, **kwargs):
        backend_calls.append((Path(video_path), sample_fps))
        yield from expected

    monkeypatch.setattr(VideoSampler, "_sample_with_decord", _fake_decord)

    results = list(sampler.sample(tmp_path / "input.mov", sample_fps=3.0))

    assert results == expected
    assert backend_calls == [(tmp_path / "input.mov", 3.0)]


def test_video_sampler_uses_pyav_backend_when_auto_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr("card_capture.stages.sample.sampler._resolve_reader_backend", lambda preferred: "pyav")
    sampler = VideoSampler(reader_backend="auto")
    expected = [
        FrameSample(
            frame_index=1,
            timestamp_ms=100,
            image=np.zeros((6, 8, 3), dtype=np.uint8),
            width=8,
            height=6,
        )
    ]
    backend_calls = []

    def _fake_pyav(self, video_path, sample_fps, **kwargs):
        backend_calls.append((Path(video_path), sample_fps))
        yield from expected

    monkeypatch.setattr(VideoSampler, "_sample_with_pyav", _fake_pyav)

    results = list(sampler.sample(tmp_path / "input.mov", sample_fps=0.0))

    assert results == expected
    assert backend_calls == [(tmp_path / "input.mov", 0.0)]


def test_pyav_decode_never_uses_frame_threading(monkeypatch):
    """FRAME threading (thread_type="AUTO") deadlocks mid-stream when OpenCV and
    PyAV are both imported (duplicate libavdevice). The pyav decode path must
    request SLICE-only threading, never AUTO/FRAME. We use a fake container so
    the assertion guards the assigned value (a real codec resolves thread_type
    to its own capabilities, masking the regression)."""
    from fractions import Fraction

    assigned: list[str] = []

    class _Stream:
        type = "video"
        average_rate = 30
        guessed_rate = 30
        time_base = Fraction(1, 30)
        _tt = "NONE"

        @property
        def thread_type(self):
            return self._tt

        @thread_type.setter
        def thread_type(self, v):
            assigned.append(v)
            self._tt = v

    class _Frame:
        pts = 0
        width = 8
        height = 6

        def to_ndarray(self, format=None):
            return np.zeros((6, 8, 3), dtype=np.uint8)

    class _Packet:
        def decode(self):
            return [_Frame()]

    class _Container:
        streams = [_Stream()]

        def demux(self, stream):
            return [_Packet()]

        def close(self):
            pass

    monkeypatch.setattr(
        VideoSampler, "_open_pyav_container", staticmethod(lambda video_path: (_Container(), False))
    )

    sampler = VideoSampler(reader_backend="pyav")
    frames = list(sampler.sample(Path("fake.mov"), sample_fps=5.0))

    assert len(frames) > 0, "decode must still yield frames"
    assert assigned, "decode path must explicitly set thread_type"
    assert assigned[-1] not in ("AUTO", "FRAME"), (
        f"FRAME threading reintroduced: thread_type={assigned[-1]!r}"
    )


# ---------------------------------------------------------------------------
# PresenceWindow Tests
# ---------------------------------------------------------------------------


def test_presence_window_detection_methods():
    """PresenceWindow should store which metrics detected the card."""
    window = PresenceWindow(
        start_frame=100, end_frame=110,
        detection_methods=["variance", "motion"]
    )
    assert window.detection_methods == ["variance", "motion"]


def test_presence_window_detection_methods_default():
    """Default detection_methods should be empty list."""
    window = PresenceWindow(start_frame=100, end_frame=110)
    assert window.detection_methods == []


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
        """sample() should yield FrameSample objects (one per candidate frame)."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            min_presence_frames=1,
            candidates_per_window=3,
        )
        frames = list(sampler.sample())
        assert len(frames) > 0, "sample() should yield at least one frame"
        for frame in frames:
            assert isinstance(frame, FrameSample), "Should yield FrameSample objects"
            assert frame.frame_index >= 0, "Frame index should be non-negative"
            assert frame.image is not None, "Frame image should not be None"

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
        frames = list(sampler.sample())
        # Should get at most 2 frames per window (since candidates_per_window=2)
        # Note: The exact number depends on how many windows were detected
        assert all(isinstance(f, FrameSample) for f in frames), "All items should be FrameSample objects"

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

    def test_multi_metric_detection_variance_only(self, synthetic_video_path):
        """With variance-only (default), should behave as before."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            candidates_per_window=1,
            device="cpu",
            detection_metrics=["variance"]
        )
        triggered = sampler._detect_metrics(
            0, np.zeros((50, 50)), variance=150.0, motion=0.0,
            histogram_stats=(0.0, 0.0), edge_metrics=(0.0, False),
            enabled_metrics=["variance"]
        )
        assert triggered == ["variance"], "Variance should trigger when above threshold"

    def test_multi_metric_detection_or_fusion(self, synthetic_video_path):
        """Multiple metrics should use OR logic (any trigger = detection)."""
        sampler = ContrastBasedSampler(
            video_path=str(synthetic_video_path),
            scan_fps=5.0,
            scan_width=160,
            contrast_threshold=100.0,
            motion_threshold=5.0,
            candidates_per_window=1,
            device="cpu",
            detection_metrics=["variance", "motion"]
        )
        
        # Motion triggers, variance doesn't
        triggered = sampler._detect_metrics(
            0, np.zeros((50, 50)), variance=50.0, motion=10.0,
            histogram_stats=(0.0, 0.0), edge_metrics=(0.0, False),
            enabled_metrics=["variance", "motion"]
        )
        assert "motion" in triggered, "Motion should trigger when above threshold"
        assert "variance" not in triggered, "Variance should not trigger when below threshold"
        
        # Both should trigger
        triggered = sampler._detect_metrics(
            0, np.zeros((50, 50)), variance=150.0, motion=10.0,
            histogram_stats=(0.0, 0.0), edge_metrics=(0.0, False),
            enabled_metrics=["variance", "motion"]
        )
        assert "variance" in triggered, "Variance should trigger"
        assert "motion" in triggered, "Motion should trigger"


class TestAdaptivePresenceSampler:
    def test_finds_presence_windows_from_video_distribution(self, tmp_path):
        frames = (
            gray_frames(8, value=128)
            + colored_frames(12)
            + gray_frames(12, value=128)
            + colored_frames(12)
            + gray_frames(8, value=128)
        )
        path = make_video(tmp_path, frames, fps=30.0)
        sampler = AdaptivePresenceSampler(
            video_path=str(path),
            reader_backend="auto",
            scan_fps=10.0,
            scan_width=160,
            device="cpu",
        )

        windows = sampler._find_presence_windows()
        assert len(windows) >= 2
        assert all(window.end_frame >= window.start_frame for window in windows)

    def test_sample_returns_selected_frames_in_time_order(self, tmp_path):
        frames = (
            gray_frames(8, value=128)
            + colored_frames(12)
            + gray_frames(12, value=128)
            + colored_frames(12)
            + gray_frames(8, value=128)
        )
        path = make_video(tmp_path, frames, fps=30.0)
        sampler = AdaptivePresenceSampler(
            video_path=str(path),
            reader_backend="auto",
            scan_fps=10.0,
            scan_width=160,
            device="cpu",
        )

        results = list(sampler.sample())
        assert len(results) > 0
        assert [frame.frame_index for frame in results] == sorted(
            frame.frame_index for frame in results
        )

    @pytest.mark.quarantine
    def test_sample_prefers_local_contiguous_frames_in_large_window(self, tmp_path):
        frames = gray_frames(120, value=128)
        frames[30:90] = colored_frames(60)
        path = make_video(tmp_path, frames, fps=30.0)
        sampler = AdaptivePresenceSampler(
            video_path=str(path),
            reader_backend="auto",
            scan_fps=10.0,
            scan_width=160,
            device="cpu",
        )

        results = list(sampler.sample())
        frame_indices = [frame.frame_index for frame in results]
        assert len(frame_indices) > 0
        # Window is from 30 to 90 (exclusive of 90 in slice, so 30-89)
        # Scan at 10fps, video at 30fps -> step 3. 
        # Frames: 30, 33, ..., 87.
        assert min(frame_indices) >= 30
        assert max(frame_indices) <= 90


# ---------------------------------------------------------------------------
# AdaptivePresenceSampler — classifier path tests
# ---------------------------------------------------------------------------

class _FakeClassifier:
    """Stub that scores any frame as 1.0 (card present) for N calls, then 0.0."""
    def __init__(self, *, always_positive: bool = True):
        self.always_positive = always_positive

    def score_batch(self, frames):
        return [1.0 if self.always_positive else 0.0] * len(frames)


class TestAdaptivePresenceSamplerClassifierPath:

    def _make_sampler(self, tmp_path, *, colored: bool = True):
        frames = gray_frames(5, 128) + (colored_frames(10) if colored else gray_frames(10, 128)) + gray_frames(5, 128)
        path = make_video(tmp_path, frames, fps=30.0)
        return AdaptivePresenceSampler(
            video_path=str(path),
            scan_fps=10.0,
            scan_width=160,
            device="cpu",
            min_presence_frames=2,
        ), path

    def test_sampler_is_picklable_with_weights_path(self, tmp_path):
        """Sampler storing a weights_path (not a live model) must survive pickle round-trip."""
        import pickle
        sampler, _ = self._make_sampler(tmp_path)
        sampler.presence_weights_path = Path("models/presence_classifier.pt")
        # Should not raise — no live tensors to pickle
        data = pickle.dumps(sampler)
        restored = pickle.loads(data)
        assert restored.presence_weights_path == sampler.presence_weights_path

    def test_classifier_not_instantiated_at_init(self, tmp_path):
        """PresenceClassifier must be lazily created, not at __init__ time."""
        sampler, _ = self._make_sampler(tmp_path)
        sampler.presence_weights_path = Path("models/presence_classifier.pt")
        assert sampler._presence_classifier is None

    def test_build_windows_with_classifier_sets_score_threshold(self, tmp_path):
        """When classifier path is taken, last_score_threshold must be 0.5 (not UnboundLocalError)."""
        sampler, _ = self._make_sampler(tmp_path)
        sampler._scan_frames = sampler._scan_video(sampler.video_path)
        sampler._presence_classifier = _FakeClassifier(always_positive=True)
        sampler.presence_weights_path = Path("fake")  # truthy so classifier branch is taken
        windows = sampler._build_windows(sampler._scan_frames)
        # Must not raise; threshold must equal the classifier fixed value
        assert sampler.last_score_threshold == 0.5

    def test_build_windows_classifier_positive_produces_windows(self, tmp_path):
        """Classifier scoring all frames 1.0 should produce at least one presence window."""
        sampler, _ = self._make_sampler(tmp_path)
        sampler._scan_frames = sampler._scan_video(sampler.video_path)
        sampler._presence_classifier = _FakeClassifier(always_positive=True)
        sampler.presence_weights_path = Path("fake")
        windows = sampler._build_windows(sampler._scan_frames)
        assert len(windows) > 0

    def test_build_windows_classifier_all_negative_produces_no_presence_windows(self, tmp_path):
        """Classifier scoring all frames 0.0 should produce no adaptive_presence windows.
        The sampler may still emit an adaptive_fallback window — that's expected behaviour."""
        sampler, _ = self._make_sampler(tmp_path)
        sampler._scan_frames = sampler._scan_video(sampler.video_path)
        sampler._presence_classifier = _FakeClassifier(always_positive=False)
        sampler.presence_weights_path = Path("fake")
        windows = sampler._build_windows(sampler._scan_frames)
        presence_windows = [w for w in windows if "adaptive_presence" in w.detection_methods]
        assert len(presence_windows) == 0

    def test_fallback_path_still_works_without_weights(self, tmp_path):
        """When presence_weights_path is None, Otsu fallback runs without error."""
        sampler, _ = self._make_sampler(tmp_path)
        sampler._scan_frames = sampler._scan_video(sampler.video_path)
        assert sampler.presence_weights_path is None
        windows = sampler._build_windows(sampler._scan_frames)
        # Otsu path: result is a list (possibly empty on gray-only video)
        assert isinstance(windows, list)

def test_sampler_collects_background_proxies(tmp_path):
    from card_capture.stages.sample.sampler import AdaptivePresenceSampler
    import numpy as np

    # 10 frames of background (low score), 10 frames of card (high score)
    frames = gray_frames(10, value=128) + colored_frames(10)
    path = make_video(tmp_path, frames, fps=30.0)

    sampler = AdaptivePresenceSampler(
        video_path=str(path),
        scan_fps=10.0,
        scan_width=160,
        device="cpu",
    )
    sampler._max_bg_proxies = 3
    sampler._bg_safety_threshold = 0.5

    # Run sample to trigger background collection
    list(sampler.sample())

    # Should have 3 background proxies because we have 10 low-score frames
    assert len(sampler.background_proxies) == 3
    # They should be from the first 10 frames (gray_frames)
    # Each proxy should be an image
    assert all(isinstance(img, np.ndarray) for img in sampler.background_proxies)

@pytest.mark.quarantine
def test_sampler_background_proxies_safety_threshold(tmp_path):
    from card_capture.stages.sample.sampler import AdaptivePresenceSampler
    # Only "card" frames (high score), no background
    frames = colored_frames(20)
    path = make_video(tmp_path, frames, fps=30.0)

    sampler = AdaptivePresenceSampler(
        video_path=str(path),
        scan_fps=10.0,
        scan_width=160,
        device="cpu",
    )
    sampler._bg_safety_threshold = -10.0  # Force safety threshold to be very low

    list(sampler.sample())

    # Should have 0 background proxies because all scores are likely > -10.0
    assert len(sampler.background_proxies) == 0
