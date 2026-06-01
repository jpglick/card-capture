from pathlib import Path
import json
import re

import numpy as np
import pytest
from queue import Empty, Full

from card_capture.core.models import CardDetection, CornerDetection, DetectionPacket, FramePacket, FrameSample, QualityScore, ScoredCandidate, TrackState
from card_capture.shared.pipeline_utils import (
    _appearance_vector,
    _side_textiness_score,
    _select_canonical_entries,
)
from card_capture.core.workers import (
    ProcessingOptions,
    _SENTINEL,
    _drain_detection_queue,
    _put_with_retry,
)
from card_capture.stages.dedup.deduplicator import VisualDeduplicator
from card_capture.storage import Storage


class FakeSampler:
    def __init__(self, frame_count: int = 1, timestamp_step_ms: int = 100):
        self.frame_count = frame_count
        self.timestamp_step_ms = timestamp_step_ms
        self.background_proxies = []

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

class AlternatingFakeSampler(FakeSampler):
    def sample(self, video_path, sample_fps):
        for i in range(self.frame_count):
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            # Frame 0, 2, 4... are "full" (with a box)
            # Frame 1, 3, 5... are "empty" (all zeros)
            if i % 2 == 0:
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
            # Only detect on "full" frames (those with the box)
            # In our tests, "full" frames have mean > 0
            if frame.image.mean() > 0:
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
        if frame.image.mean() > 0:
            return [
                CardDetection(
                    frame_index=frame.frame_index,
                    timestamp_ms=frame.timestamp_ms,
                    polygon=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),
                    confidence=self.confidence,
                    metadata={"runtime": self.runtime, "model": self.model_name},
                )
            ]
        return []


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


@pytest.mark.skip(reason="VideoProcessor retired with monolith")
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
            background_frames=10, # Warmup longer than test
            min_track_length=3,
            triage_keep_percentile=1.0,
        ),
    )

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 3
    # BoT-SORT emits tentative tracks immediately; the pipeline-level
    # min_track_length gate is the single confirmation gate.
    assert result.saved_instance_count == 1
    assert result.telemetry["tracker_event_count"] == 3
    assert Path(result.telemetry["tracker_association_events_path"]).exists()

    assert _row_count(storage, "card_instances") == 1
    assert _row_count(storage, "card_views") == 3
    assert _row_count(storage, "evidence_frames") == 3


@pytest.mark.skip(reason="VideoProcessor retired with monolith")
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
            background_frames=10,
        ),
    )

    assert result.frame_count == 3
    assert result.accepted_frame_count == 3
    assert result.detection_count == 0
    assert result.saved_instance_count == 0
    assert _row_count(storage, "card_instances") == 0
    assert _row_count(storage, "card_views") == 0
    assert _row_count(storage, "evidence_frames") == 0


@pytest.mark.skip(reason="VideoProcessor retired with monolith")
def test_pipeline_filters_empty_workspace_with_pre_warmed_detector(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    # Pre-warm with black frames (zeros)
    sampler = AlternatingFakeSampler(frame_count=10)
    sampler.background_proxies = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5

    # AlternatingFakeSampler produces:
    # 0: full, 1: empty, 2: full, 3: empty, 4: full, 5: empty, 6: full, 7: empty, 8: full, 9: empty
    # Total 10 frames. 5 are empty (all zeros).
    # Since background is pre-warmed with zeros, those 5 should be filtered.

    result = VideoProcessor(
        storage=storage,
        sampler=sampler,
        detector=FakeBatchDetector(confidence=0.95),
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            queue_size=4,
            background_frames=5,
            background_threshold=1.0,
            min_track_length=1,
            triage_keep_percentile=1.0,
        ),
    )

    # 10 frames sampled, 5 should be filtered out because they are identical to pre-warmed background
    assert result.frame_count == 10
    assert result.accepted_frame_count == 5
    assert result.detection_count == 5


@pytest.mark.skip(reason="VideoProcessor retired with monolith")
def test_pipeline_falls_back_to_legacy_detect_contract(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(frame_count=2, timestamp_step_ms=1200),
        detector=FakeLegacyDetector(confidence=0.95),
    ).process(video_path, ProcessingOptions(output_dir=tmp_path / "output", queue_size=4, background_frames=10))

    assert result.frame_count == 2
    assert result.accepted_frame_count == 2
    assert result.detection_count == 2
    # v3 Stage 2 verification drops tracks shorter than 3 frames.
    assert result.saved_instance_count == 0
    assert _row_count(storage, "card_instances") == 0
    assert _row_count(storage, "card_views") == 0
    assert _row_count(storage, "evidence_frames") == 0


@pytest.mark.skip(reason="VideoProcessor retired with monolith")
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
        processor.process(video_path, ProcessingOptions(output_dir=tmp_path / "output", queue_size=2, background_frames=10))


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


@pytest.mark.skip(reason="_resolve_session_tracks retired with monolith")
def test_resolve_session_tracks_merges_visually_identical_clusters(tmp_path):
    """v4.1 NoveltyGate: visually identical tracks still merge front+back."""
    import cv2
    deduplicator = VisualDeduplicator()
    image = np.full((120, 80, 3), 180, dtype=np.uint8)
    image[20:100, 20:60] = 30

    # Save the image so we have a path to read from
    img_path = str(tmp_path / "identical.jpg")
    cv2.imwrite(img_path, image)

    prepared = []
    for idx, frame_index in enumerate((100, 120), start=1):
        # Create a track with a candidate that has an image path
        track = type("Track", (), {"instance_id": f"t{idx}", "candidates": []})()
        track.candidates.append(ScoredCandidate(
            detection_id=idx, timestamp_ms=idx * 33, image_path=img_path,
            score=QualityScore(total=0.7, components={}),
            corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
            frame_index=frame_index,
        ))
        prepared.append(
            _PreparedTrack(
                track=track,
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
                best_canonical_detection_id=0,
                fused_canonical=image,
            )
        )

    _resolve_session_tracks(prepared, deduplicator)

    assert prepared[0].duplicate_track_index is None
    assert prepared[1].duplicate_track_index == 0
    assert prepared[0].angle == "Front"
    assert prepared[1].angle == "Back"


@pytest.mark.skip(reason="_resolve_session_tracks retired with monolith")
def test_adaptive_hamming_threshold_flips_dedup_decision(tmp_path):
    """E2: Adaptive Hamming threshold flips dedup decision for borderline cases.

    Construct two _PreparedTrack objects with controlled Hamming distances such that:
    1. Hamming distance H > _SAME_CARD_HAMMING_MAX (22)
    2. But H <= adaptive_threshold when 10+ smaller intra-track distances exist

    Assert:
    - Without context (no adaptation): classified as different cards
    - With context (after collecting samples): classified as same card

    The adaptive threshold is computed as p75*1.2 from collected distances,
    clipped to [global*0.9, global*1.1] = [19.8, 24.2]. So we need a test case
    where H > 22 but fits within the clipping range, OR we pre-populate the
    context to raise the threshold high enough.
    """
    from card_capture.shared.pipeline_utils import PipelineContext, _side_textiness_score, _appearance_vector
    import cv2

    deduplicator = VisualDeduplicator()

    # Create a base image
    image1 = np.full((120, 80, 3), 180, dtype=np.uint8)
    image1[20:100, 20:60] = 30
    img_path1 = str(tmp_path / "image1.jpg")
    cv2.imwrite(img_path1, image1)

    # Create a second image with small pixel differences to get a distance > 22
    # Strategy: make a mostly-identical image with some pixel flips
    image2 = image1.copy()
    # Flip some pixels to induce hash changes
    image2[50:60, 30:40] = 255 - image2[50:60, 30:40]  # Invert a small region
    image2[70:75, 50:55] = 255 - image2[70:75, 50:55]  # Invert another region
    img_path2 = str(tmp_path / "image2.jpg")
    cv2.imwrite(img_path2, image2)

    # Compute actual hashes and distance
    hash1 = deduplicator.compute_phash(image1)
    hash2 = deduplicator.compute_phash(image2)
    actual_distance = deduplicator.hamming_distance(hash1, hash2)

    # Create two prepared tracks: one with image1, one with image2
    prepared_1 = _PreparedTrack(
        track=type("Track", (), {
            "instance_id": "t1",
            "candidates": [ScoredCandidate(
                detection_id=1, timestamp_ms=100, image_path=img_path1,
                score=QualityScore(total=0.9, components={}),
                corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
                frame_index=100,
            )]
        })(),
        session_id=1,
        first_frame_index=100,
        angle="Front",
        frame_entries=[],
        canonical_entries=[],
        candidate_hashes=[hash1],
        primary_hash=hash1,
        side_score=_side_textiness_score(image1),
        appearance_vector=_appearance_vector(image1),
        canonical_detection_ids=set(),
        best_canonical_detection_id=0,
        fused_canonical=image1,
        embedding=None,  # Force pHash path
    )

    prepared_2 = _PreparedTrack(
        track=type("Track", (), {
            "instance_id": "t2",
            "candidates": [ScoredCandidate(
                detection_id=2, timestamp_ms=200, image_path=img_path2,
                score=QualityScore(total=0.9, components={}),
                corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
                frame_index=120,
            )]
        })(),
        session_id=1,
        first_frame_index=120,
        angle="Front",
        frame_entries=[],
        canonical_entries=[],
        candidate_hashes=[hash2],
        primary_hash=hash2,
        side_score=_side_textiness_score(image2),
        appearance_vector=_appearance_vector(image2),
        canonical_detection_ids=set(),
        best_canonical_detection_id=0,
        fused_canonical=image2,
        embedding=None,  # Force pHash path
    )

    prepared_without_context = [prepared_1, prepared_2]

    # Test WITHOUT context: distance > 22 should NOT merge (using global threshold)
    _resolve_session_tracks(prepared_without_context, deduplicator, context=None)
    assert prepared_without_context[0].duplicate_track_index is None
    # If distance <= 22, they will merge; if > 22, they won't.
    # Since we don't have control over exact distance, check the logic works:
    if actual_distance <= 22:
        assert prepared_without_context[1].duplicate_track_index == 0
    else:
        assert prepared_without_context[1].duplicate_track_index is None
    assert prepared_without_context[0].angle == "Front"

    # Test WITH context: collect samples, then merge
    # Create fresh prepared tracks for the context test
    prepared_1_ctx = _PreparedTrack(
        track=type("Track", (), {
            "instance_id": "t1",
            "candidates": [ScoredCandidate(
                detection_id=1, timestamp_ms=100, image_path=img_path1,
                score=QualityScore(total=0.9, components={}),
                corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
                frame_index=100,
            )]
        })(),
        session_id=1,
        first_frame_index=100,
        angle="Front",
        frame_entries=[],
        canonical_entries=[],
        candidate_hashes=[hash1],
        primary_hash=hash1,
        side_score=_side_textiness_score(image1),
        appearance_vector=_appearance_vector(image1),
        canonical_detection_ids=set(),
        best_canonical_detection_id=0,
        fused_canonical=image1,
        embedding=None,  # Force pHash path
    )

    prepared_2_ctx = _PreparedTrack(
        track=type("Track", (), {
            "instance_id": "t2",
            "candidates": [ScoredCandidate(
                detection_id=2, timestamp_ms=200, image_path=img_path2,
                score=QualityScore(total=0.9, components={}),
                corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
                frame_index=120,
            )]
        })(),
        session_id=1,
        first_frame_index=120,
        angle="Front",
        frame_entries=[],
        canonical_entries=[],
        candidate_hashes=[hash2],
        primary_hash=hash2,
        side_score=_side_textiness_score(image2),
        appearance_vector=_appearance_vector(image2),
        canonical_detection_ids=set(),
        best_canonical_detection_id=0,
        fused_canonical=image2,
        embedding=None,  # Force pHash path
    )

    prepared_with_context = [prepared_1_ctx, prepared_2_ctx]

    # Pre-populate context with intra-track distances.
    # Strategy: add distances at or slightly above the actual_distance to ensure the adaptive
    # threshold rises at least to the clipping max (24.2 for global=22).
    # With p75*1.2 clipped to [19.8, 24.2], we want enough samples at high values to reach the cap.
    context = PipelineContext()

    # Add samples that will push p75 to ~22-23, giving threshold clipped to 24.2
    # Mix of values centered around actual_distance (or the max clipping if distance is too high)
    threshold_max = 22 * 1.1  # 24.2
    sample_base = min(actual_distance - 2, threshold_max - 2)  # Go slightly below

    sample_distances = []
    for i in range(10):
        sample_distances.append(sample_base + i * 0.5)  # Spread samples across range
    # Ensure we have enough samples above the threshold to try to hit the clipping max
    sample_distances.extend([threshold_max - 1] * 5)

    for dist in sample_distances:
        context.add_intra_track_distance(float(dist))

    # Now resolve with context: adaptive threshold should be raised
    _resolve_session_tracks(prepared_with_context, deduplicator, context=context)

    # Verify adaptive threshold was consulted by checking if merge happened
    # If actual_distance <= the adaptive threshold (at least 24.2 due to clipping), they'll merge
    assert prepared_with_context[0].duplicate_track_index is None
    if actual_distance <= 24.2:  # Clipping max for global=22
        assert prepared_with_context[1].duplicate_track_index == 0, \
            f"With adaptive threshold at {24.2}, expected merge at distance={actual_distance}"
        assert prepared_with_context[0].angle == "Front"
        assert prepared_with_context[1].angle == "Back"
    else:
        # Distance is too high even for clipping max; just verify no crash occurred
        assert prepared_with_context[0].duplicate_track_index is None

def test_pipeline_processing_options_has_tracker_backend():
    from card_capture.core.workers import ProcessingOptions
    from pathlib import Path
    opts = ProcessingOptions(output_dir=Path("/tmp"))
    assert opts.tracker_backend == "bytetrack"
    assert opts.centroid_jump_ratio == 0.30
    assert opts.centroid_jump_frames == 3


def test_pipeline_processing_options_accepts_bytetrack():
    from card_capture.core.workers import ProcessingOptions
    from pathlib import Path
    opts = ProcessingOptions(output_dir=Path("/tmp"), tracker_backend="bytetrack")
    assert opts.tracker_backend == "bytetrack"


def test_pyproject_declares_legacy_tracking_runtime_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r"legacy_tracking\s*=\s*\[(.*?)\]", pyproject, re.DOTALL)

    assert match is not None
    runtime_block = match.group(1)
    for dep_name in ("onnxruntime", "av"):
        assert f'"{dep_name}"' in runtime_block
    assert '"decord"' not in runtime_block


@pytest.mark.skip(reason="_filter_candidates_by_novelty retired with monolith")
def test_candidate_filter_drops_background_quads(tmp_path):
    """_filter_candidates_by_novelty must remove candidates whose quad interior
    matches the workspace baseline."""
    from card_capture.shared.pipeline_utils import _filter_candidates_by_novelty
    from card_capture.stages.novelty.background_novelty import BackgroundModel
    import cv2

    # Build a flat background and write it to disk
    bg_arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    bg_path = tmp_path / "bg.jpg"
    cv2.imwrite(str(bg_path), bg_arr)

    # Build a "card" frame and write it to disk
    card_arr = bg_arr.copy()
    card_arr[40:160, 40:160] = 20
    card_path = tmp_path / "card.jpg"
    cv2.imwrite(str(card_path), card_arr)

    bg = BackgroundModel.from_frames([bg_arr])

    cands = [
        # Background-only candidate: corners over a flat region.
        ScoredCandidate(
            detection_id=0, timestamp_ms=0, image_path=str(bg_path),
            score=QualityScore(total=0.8, components={}),
            corners=[(40, 40), (160, 40), (160, 160), (40, 160)],
            frame_index=0,
        ),
        # Card candidate: corners over the painted patch.
        ScoredCandidate(
            detection_id=1, timestamp_ms=33, image_path=str(card_path),
            score=QualityScore(total=0.8, components={}),
            corners=[(40, 40), (160, 40), (160, 160), (40, 160)],
            frame_index=1,
        ),
    ]
    kept = _filter_candidates_by_novelty(cands, bg, threshold=0.08)
    kept_ids = [c.detection_id for c in kept]
    assert kept_ids == [1], kept_ids


@pytest.mark.skip(reason="_collect_candidate_novelty_scores retired with monolith")
def test_candidate_novelty_collection_does_not_drop_pre_tracking_candidates(tmp_path):
    from card_capture.shared.pipeline_utils import _collect_candidate_novelty_scores, PipelineContext
    from card_capture.stages.novelty.background_novelty import BackgroundModel
    import cv2

    bg_arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    bg_path = tmp_path / "bg.jpg"
    cv2.imwrite(str(bg_path), bg_arr)

    card_arr = bg_arr.copy()
    card_arr[40:160, 40:160] = 20
    card_path = tmp_path / "card.jpg"
    cv2.imwrite(str(card_path), card_arr)

    cands = [
        ScoredCandidate(
            detection_id=0, timestamp_ms=0, image_path=str(bg_path),
            score=QualityScore(total=0.8, components={}),
            corners=[(40, 40), (160, 40), (160, 160), (40, 160)],
            frame_index=0,
        ),
        ScoredCandidate(
            detection_id=1, timestamp_ms=33, image_path=str(card_path),
            score=QualityScore(total=0.8, components={}),
            corners=[(40, 40), (160, 40), (160, 160), (40, 160)],
            frame_index=1,
        ),
    ]

    context = PipelineContext()
    scored = _collect_candidate_novelty_scores(cands, BackgroundModel.from_frames([bg_arr]), context)

    assert scored == 2
    assert len(cands) == 2
    assert len(context.observed_novelty_scores) == 2


@pytest.mark.skip(reason="_PreparedTrack retired with monolith")
def test_prune_empty_workspace_tracks_drops_low_novelty_tracks(tmp_path):
    """A finalized track whose medoid frame's quad matches the background is removed."""
    from card_capture.shared.pipeline_utils import _prune_empty_workspace_tracks, _PreparedTrack
    from card_capture.core.models import TrackState
    from card_capture.stages.novelty.background_novelty import BackgroundModel
    import cv2

    bg_arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    bg_path = tmp_path / "bg.jpg"
    cv2.imwrite(str(bg_path), bg_arr)
    bg = BackgroundModel.from_frames([bg_arr])

    fake_corners = [(40, 40), (160, 40), (160, 160), (40, 160)]

    # Empty-workspace track (3 background-matching candidates)
    empty_track = TrackState(instance_id="empty")
    for i in range(3):
        empty_track.candidates.append(ScoredCandidate(
            detection_id=i, timestamp_ms=i * 33, image_path=str(bg_path),
            score=QualityScore(total=0.7, components={}),
            corners=fake_corners, frame_index=i,
        ))

    # Real-card track (3 painted-frame candidates)
    card_arr = bg_arr.copy()
    card_arr[40:160, 40:160] = 20
    card_path = tmp_path / "card.jpg"
    cv2.imwrite(str(card_path), card_arr)
    card_track = TrackState(instance_id="card")
    for i in range(3):
        card_track.candidates.append(ScoredCandidate(
            detection_id=10 + i, timestamp_ms=i * 33, image_path=str(card_path),
            score=QualityScore(total=0.7, components={}),
            corners=fake_corners, frame_index=i,
        ))

    prepared = [
        _PreparedTrack(
            track=empty_track, session_id=1, first_frame_index=0, angle="Front",
            frame_entries=[], canonical_entries=[], candidate_hashes=[],
            primary_hash="", side_score=0.0, appearance_vector=np.array([]),
            canonical_detection_ids=set(), best_canonical_detection_id=0,
            fused_canonical=None, duplicate_track_index=None
        ),
        _PreparedTrack(
            track=card_track, session_id=1, first_frame_index=0, angle="Back",
            frame_entries=[], canonical_entries=[], candidate_hashes=[],
            primary_hash="", side_score=0.0, appearance_vector=np.array([]),
            canonical_detection_ids=set(), best_canonical_detection_id=10,
            fused_canonical=None, duplicate_track_index=0
        ),
    ]
    kept = _prune_empty_workspace_tracks(prepared, bg, threshold=0.08)
    kept_ids = [pt.track.instance_id for pt in kept]
    assert kept_ids == ["card"], kept_ids


class SingleFrameSampler:
    def sample(self, video_path, sample_fps):
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        image[20:80, 20:80] = 30 # A "card"
        yield FrameSample(frame_index=0, timestamp_ms=0, image=image, width=100, height=100)

# Detector returns one detection
class SingleDetectionDetector:
    runtime = "fake"
    model_name = "fake"
    def detect_batch(self, frames, threshold=None, confidence_threshold=None):
        return [DetectionPacket(
            frame_index=0, timestamp_ms=0, width=100, height=100,
            corner_detection=CornerDetection(
                corners=((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)),
                confidence=0.9, metadata={}
            )
        )]

@pytest.mark.skip(reason="VideoProcessor retired with monolith")
def test_pipeline_integrates_novelty_score(tmp_path):
    """Pipeline must compute novelty and store it in card_views quality_score_json."""
    from card_capture.shared.pipeline_utils import VideoProcessor, ProcessingOptions
    from card_capture.storage import Storage
    import cv2
    import json

    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    # Pre-warm background with different content
    bg_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    sampler = SingleFrameSampler()
    sampler.background_proxies = [bg_image]

    processor = VideoProcessor(storage=storage, sampler=sampler, detector=SingleDetectionDetector())
    processor.process(video_path, ProcessingOptions(
        output_dir=tmp_path / "output",
        background_frames=1,
        min_track_length=1,
        triage_keep_percentile=1.0,
    ))
    # Check the database for the novelty score in quality_score_json
    with storage._connect() as conn:
        row = conn.execute("SELECT quality_score_json FROM card_views").fetchone()
    
    assert row is not None
    quality_score = json.loads(row["quality_score_json"])
    assert "novelty" in quality_score
    # Novelty should be > 0 because we have a painted patch vs flat background
    assert quality_score["novelty"] > 0
