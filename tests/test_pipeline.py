from pathlib import Path
import json
import re

import numpy as np
import pytest
from queue import Empty, Full

from card_capture.models import CardDetection, CornerDetection, DetectionPacket, FramePacket, FrameSample, QualityScore
from card_capture.pipeline import (
    ProcessingOptions,
    VideoProcessor,
    _PreparedTrack,
    _SENTINEL,
    _appearance_vector,
    _drain_detection_queue,
    _resolve_session_tracks,
    _side_textiness_score,
    _put_with_retry,
    _select_canonical_entries,
)
from card_capture.deduplicator import VisualDeduplicator
from card_capture.selector import ScoredCandidate
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
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            queue_size=4,
            background_frames=0,
            min_track_length=3,
            triage_keep_percentile=1.0,
        ),
    )

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 3
    assert result.saved_instance_count == 1
    assert result.telemetry["tracker_event_count"] == 3
    assert Path(result.telemetry["tracker_association_events_path"]).exists()

    assert _row_count(storage, "card_instances") == 1
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
    assert int(canonical_views["c"]) == 3
    assert len(evidence_rows) == 3
    assert all(Path(row["source_frame_path"]).exists() for row in evidence_rows)
    tracker_events = json.loads(Path(result.telemetry["tracker_association_events_path"]).read_text())
    assert [event["action"] for event in tracker_events] == [
        "new_track",
        "assigned_existing",
        "assigned_existing",
    ]


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
    # v3 Stage 2 verification drops tracks shorter than 3 frames.
    assert result.saved_instance_count == 0
    assert _row_count(storage, "card_instances") == 0
    assert _row_count(storage, "card_views") == 0
    assert _row_count(storage, "evidence_frames") == 0


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


class _QueueFullNTimes:
    def __init__(self, full_count: int):
        self.full_count = full_count
        self.items = []
        self.put_calls = 0

    def put(self, item, timeout):
        self.put_calls += 1
        if self.put_calls <= self.full_count:
            raise Full
        self.items.append(item)


class _QueueAlwaysFull:
    def put(self, item, timeout):
        raise Full


class _QueueAlwaysEmpty:
    def get(self, timeout):
        raise Empty


class _ErrorQueueAlwaysEmpty:
    def get_nowait(self):
        raise Empty


class _FakeProcess:
    def __init__(self, alive: bool = True):
        self._alive = alive

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False


def test_put_with_retry_guarantees_sentinel_delivery_after_full():
    queue = _QueueFullNTimes(full_count=3)

    _put_with_retry(queue, _SENTINEL, timeout=0.001)

    assert queue.items == [_SENTINEL]
    assert queue.put_calls == 4


def test_put_with_retry_guarantees_error_payload_delivery_after_full():
    queue = _QueueFullNTimes(full_count=2)
    error_payload = {"worker": "consumer", "message": "boom"}

    _put_with_retry(queue, error_payload, timeout=0.001)

    assert queue.items == [error_payload]
    assert queue.put_calls == 3


def test_put_with_retry_fails_fast_after_retry_deadline():
    queue = _QueueAlwaysFull()

    with pytest.raises(RuntimeError, match="timed out"):
        _put_with_retry(queue, _SENTINEL, timeout=0.001, max_wait_s=0.02)


def test_drain_detection_queue_times_out_when_worker_is_wedged():
    detection_queue = _QueueAlwaysEmpty()
    error_queue = _ErrorQueueAlwaysEmpty()
    producer = _FakeProcess(alive=False)
    consumer = _FakeProcess(alive=True)

    with pytest.raises(RuntimeError, match="timed out"):
        _drain_detection_queue(
            detection_queue=detection_queue,
            error_queue=error_queue,
            producer=producer,
            consumer=consumer,
            idle_timeout_s=0.02,
        )


def test_select_canonical_entries_prefers_same_appearance_cluster():
    deduplicator = VisualDeduplicator()
    entries = [
        {
            "candidate": ScoredCandidate(1, 100, "a.jpg", QualityScore(0.92, {})),
            "sharpness": 300.0,
            "visual_hash": "0000000000000000",
        },
        {
            "candidate": ScoredCandidate(2, 200, "b.jpg", QualityScore(0.90, {})),
            "sharpness": 280.0,
            "visual_hash": "0000000000000003",
        },
        {
            "candidate": ScoredCandidate(3, 300, "c.jpg", QualityScore(0.88, {})),
            "sharpness": 260.0,
            "visual_hash": "0000000000000007",
        },
        {
            "candidate": ScoredCandidate(4, 400, "d.jpg", QualityScore(0.91, {})),
            "sharpness": 320.0,
            "visual_hash": "ffffffffffffffff",
        },
    ]

    selected = _select_canonical_entries(entries, deduplicator)
    selected_ids = {entry["candidate"].detection_id for entry in selected}

    assert len(selected) == 3
    assert selected_ids == {1, 2, 3}


def test_resolve_session_tracks_merges_visually_identical_clusters():
    deduplicator = VisualDeduplicator()
    image = np.full((120, 80, 3), 180, dtype=np.uint8)
    image[20:100, 20:60] = 30
    prepared = []
    for idx, frame_index in enumerate((100, 120), start=1):
        prepared.append(
            _PreparedTrack(
                track=type("Track", (), {"instance_id": f"t{idx}", "candidates": []})(),
                session_id=1,
                first_frame_index=frame_index,
                angle="Front",
                frame_entries=[],
                canonical_entries=[],
                candidate_hashes=[deduplicator.compute_phash(image)],
                primary_hash=deduplicator.compute_phash(image),
                side_score=_side_textiness_score(image),
                appearance_vector=_appearance_vector(image),
                canonical_detection_ids=set(),
            )
        )

    _resolve_session_tracks(prepared, deduplicator)

    assert prepared[0].duplicate_track_index is None
    assert prepared[1].duplicate_track_index == 0
    assert prepared[0].angle == "Front"
    assert prepared[1].angle == "Back"

def test_pyproject_declares_pipeline_v21_runtime_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r"pipeline_v21\s*=\s*\[(.*?)\]", pyproject, re.DOTALL)

    assert match is not None
    runtime_block = match.group(1)
    for dep_name in ("onnxruntime", "av"):
        assert f'"{dep_name}"' in runtime_block
    assert '"decord"' not in runtime_block
