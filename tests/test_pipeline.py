from pathlib import Path
import re
import numpy as np
import pytest
from queue import Empty, Full

from card_capture.models import CardDetection, CornerDetection, DetectionPacket, FramePacket, FrameSample
from card_capture.pipeline import (
    ProcessingOptions,
    VideoProcessor,
    _SENTINEL,
    _drain_detection_queue,
    _put_with_retry,
)
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

class FakeBatchDetector:
    runtime = "fake"
    model_name = "fake-corner-detector"
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
    def detect_batch(self, frames: list[FramePacket], confidence_threshold: float):
        if self.confidence < confidence_threshold:
            return []
        detections = []
        for frame in frames:
            detections.append(
                DetectionPacket(
                    frame_index=frame.frame_index,
                    timestamp_ms=frame.timestamp_ms,
                    width=frame.width,
                    height=frame.height,
                    corner_detection=CornerDetection(
                        corners=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),
                        confidence=self.confidence,
                        metadata={"runtime": self.runtime, "model": self.model_name},
                    ),
                )
            )
        return detections

class FakeLegacyDetector:
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
                metadata={"runtime": self.runtime, "model": self.model_name},
            )
        ]

def _row_count(storage: Storage, table: str) -> int:
    with storage._connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
    assert row is not None
    return int(row["c"])

def test_pipeline_v3_persists_telemetry_and_groups_instances(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    # frames_per_instance=1 for easier assertion
    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=3),
        detector=FakeBatchDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output", queue_size=4, frames_per_instance=1))

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 3
    assert result.saved_instance_count == 1 # All 3 frames grouped into 1 instance

    assert _row_count(storage, "card_instances") == 1
    assert _row_count(storage, "card_views") == 3
    assert _row_count(storage, "performance_logs") == 3 # Telemetry logged for each frame

def test_pipeline_corner_confidence_threshold_filters_detections(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=3),
        detector=FakeBatchDetector(confidence=0.40),
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            corner_confidence_threshold=0.50,
            queue_size=4,
        ),
    )

    assert result.detection_count == 0
    assert _row_count(storage, "card_instances") == 0

def test_pipeline_telemetry_logging(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=1),
        detector=FakeBatchDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output"))

    with storage._connect() as conn:
        log = conn.execute("SELECT * FROM performance_logs").fetchone()
    assert log is not None
    assert log["t_ingest"] > 0
    assert log["t_detect"] > 0
