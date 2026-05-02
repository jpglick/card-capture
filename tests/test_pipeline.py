from pathlib import Path

import numpy as np

from card_capture.models import CardDetection, FrameSample
from card_capture.pipeline import ProcessingOptions, VideoProcessor
from card_capture.storage import Storage


class FakeSampler:
    def __init__(self, frame_count: int = 1, timestamp_step_ms: int = 100):
        self.frame_count = frame_count
        self.timestamp_step_ms = timestamp_step_ms

    def sample(self, video_path, sample_fps):
        for i in range(self.frame_count):
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[10:90, 10:90] = 180
            yield FrameSample(
                frame_index=i,
                timestamp_ms=i * self.timestamp_step_ms,
                image=image,
                width=100,
                height=100,
            )


class FakeDetector:
    runtime = "fake"
    model_name = "fake-card-detector"

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def detect(self, frame):
        return [
            CardDetection(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                polygon=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),
                confidence=self.confidence,
            )
        ]


def test_video_processor_saves_best_crop_and_metadata(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=1),
        detector=FakeDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output"))

    saved_cards = storage.list_saved_cards()

    assert result.frame_count == 1
    assert result.accepted_frame_count == 1
    assert result.detection_count == 1
    assert result.saved_instance_count == 1
    assert len(saved_cards) == 1
    assert Path(saved_cards[0]["image_path"]).exists()
    assert saved_cards[0]["final_score"] > 0


def test_corner_confidence_threshold_filters_detections(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=3),
        detector=FakeDetector(confidence=0.40),
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            corner_confidence_threshold=0.50,
        ),
    )

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 0
    assert result.saved_instance_count == 0


def test_pipeline_processes_all_sampled_frames(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=3),
        detector=FakeDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output"))

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 3


def test_saved_instance_count_tracks_selected_outputs(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=2, timestamp_step_ms=1200),
        detector=FakeDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output"))

    # Two detections in separate selector groups should each be saved.
    assert result.detection_count == 2
    assert result.saved_instance_count == 2
