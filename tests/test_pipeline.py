from pathlib import Path

import numpy as np
import pytest

from card_capture.models import CardDetection, CornerDetection, DetectionPacket, FramePacket, FrameSample
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


class ExplodingBatchDetector:
    runtime = "fake"
    model_name = "exploding-corner-detector"

    def detect_batch(self, frames: list[FramePacket], confidence_threshold: float):
        raise RuntimeError("simulated detector crash")


def _row_count(storage: Storage, table: str) -> int:
    with storage._connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
    assert row is not None
    return int(row["c"])


def test_pipeline_persists_v21_rows_and_result_counts(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=3),
        detector=FakeBatchDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output", queue_size=4))

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 3
    assert result.saved_instance_count == 1

    assert _row_count(storage, "card_instances") == 3
    assert _row_count(storage, "card_views") == 3
    assert _row_count(storage, "evidence_frames") == 3

    with storage._connect() as conn:
        canonical_views = conn.execute(
            "SELECT COUNT(*) AS c FROM card_views WHERE is_canonical = 1"
        ).fetchone()
        evidence_rows = conn.execute(
            "SELECT source_frame_path FROM evidence_frames ORDER BY id"
        ).fetchall()
    assert canonical_views is not None
    assert int(canonical_views["c"]) == 1
    assert len(evidence_rows) == 3
    assert all(Path(row["source_frame_path"]).exists() for row in evidence_rows)


def test_corner_confidence_threshold_filters_detections(tmp_path: Path):
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

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 0
    assert result.saved_instance_count == 0
    assert _row_count(storage, "card_instances") == 0
    assert _row_count(storage, "card_views") == 0
    assert _row_count(storage, "evidence_frames") == 0


def test_pipeline_falls_back_to_legacy_detect_contract(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=2, timestamp_step_ms=1200),
        detector=FakeLegacyDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output", queue_size=4))

    assert result.frame_count == 2
    assert result.accepted_frame_count == 2
    assert result.detection_count == 2
    assert result.saved_instance_count == 2
    assert _row_count(storage, "card_instances") == 2
    assert _row_count(storage, "card_views") == 2
    assert _row_count(storage, "evidence_frames") == 2


def test_pipeline_propagates_consumer_errors(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    processor = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=2),
        detector=ExplodingBatchDetector(),
    )

    with pytest.raises(RuntimeError, match="consumer"):
        processor.process(video_path, ProcessingOptions(output_dir=tmp_path / "output", queue_size=2))
