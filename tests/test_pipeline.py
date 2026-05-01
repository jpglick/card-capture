from pathlib import Path

import numpy as np

from card_capture.models import CardDetection, FrameSample
from card_capture.pipeline import ProcessingOptions, VideoProcessor
from card_capture.storage import Storage


class FakeSampler:
    def sample(self, video_path, sample_fps):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:90, 10:90] = 180
        yield FrameSample(
            frame_index=1,
            timestamp_ms=100,
            image=image,
            width=100,
            height=100,
        )


class FakeDetector:
    runtime = "fake"
    model_name = "fake-card-detector"

    def detect(self, frame):
        return [
            CardDetection(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                polygon=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),
                confidence=0.95,
            )
        ]


def test_video_processor_saves_best_crop_and_metadata(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(),
        detector=FakeDetector(),
    ).process(
        video_path,
        ProcessingOptions(output_dir=tmp_path / "output", sample_fps=5, max_candidates=5),
    )

    saved_cards = storage.list_saved_cards()

    assert result.saved_count == 1
    assert len(saved_cards) == 1
    assert Path(saved_cards[0]["image_path"]).exists()
    assert saved_cards[0]["final_score"] > 0


def test_early_stop_halts_after_first_qualifying_detection(tmp_path: Path):
    """Pipeline breaks out of the frame loop once detections_to_stop
    detections exceed quality_floor, without consuming further frames."""
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    processed_frames = []

    class CountingSampler:
        def sample(self, video_path, sample_fps):
            for i in range(5):
                processed_frames.append(i)
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[10:90, 10:90] = 180
                yield FrameSample(
                    frame_index=i,
                    timestamp_ms=i * 200,
                    image=image,
                    width=100,
                    height=100,
                )

    VideoProcessor(
        storage=storage,
        sampler=CountingSampler(),
        detector=FakeDetector(),
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            sample_fps=5,
            max_candidates=5,
            detections_to_stop=1,
            quality_floor=0.2,  # FakeDetector crop scores ≈ 0.249 > 0.2
        ),
    )

    # Generator is closed by `break` after the first frame — frame 1 never appended.
    assert len(processed_frames) == 1


def test_early_stop_disabled_when_zero(tmp_path: Path):
    """detections_to_stop=0 processes all sampled frames."""
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    processed_frames = []

    class CountingSampler:
        def sample(self, video_path, sample_fps):
            for i in range(3):
                processed_frames.append(i)
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[10:90, 10:90] = 180
                yield FrameSample(
                    frame_index=i,
                    timestamp_ms=i * 200,
                    image=image,
                    width=100,
                    height=100,
                )

    VideoProcessor(
        storage=storage,
        sampler=CountingSampler(),
        detector=FakeDetector(),
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            sample_fps=5,
            max_candidates=5,
            detections_to_stop=0,
            quality_floor=0.2,
        ),
    )

    assert len(processed_frames) == 3
