from pathlib import Path

import numpy as np

from card_capture.models import (
    CornerDetection,
    DetectionPacket,
    FramePacket,
    ProcessingResult,
)


def test_frame_packet_has_triage_metrics():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    triage_metrics = {"blur_score": 12.5, "variance": 42.0}

    packet = FramePacket(
        frame_index=3,
        timestamp_ms=120,
        image=image,
        width=48,
        height=32,
        triage_metrics=triage_metrics,
    )

    assert packet.triage_metrics == triage_metrics


def test_detection_packet_wraps_corner_detection():
    corners = ((1.0, 1.0), (11.0, 1.0), (11.0, 9.0), (1.0, 9.0))
    corner_detection = CornerDetection(
        corners=corners,
        confidence=0.88,
        metadata={"source": "unit-test"},
    )

    packet = DetectionPacket(
        frame_index=8,
        timestamp_ms=320,
        width=1280,
        height=720,
        corner_detection=corner_detection,
    )

    assert packet.corner_detection is corner_detection
    assert packet.corner_detection.corners == corners


def test_processing_result_has_v21_counts():
    result = ProcessingResult(
        video_id=21,
        frame_count=100,
        accepted_frame_count=24,
        detection_count=17,
        saved_instance_count=6,
        output_dir=Path("/tmp/out"),
    )

    assert result.frame_count == 100
    assert result.accepted_frame_count == 24
    assert result.detection_count == 17
    assert result.saved_instance_count == 6


def test_packets_can_carry_telemetry():
    from card_capture.models import PerformanceTelemetry

    telemetry = PerformanceTelemetry(t_ingest=0.1, t_detect=0.2)

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    frame_packet = FramePacket(
        frame_index=1,
        timestamp_ms=10,
        image=image,
        width=48,
        height=32,
        triage_metrics={},
        telemetry=telemetry,
    )
    assert frame_packet.telemetry == telemetry

    corners = ((1.0, 1.0), (11.0, 1.0), (11.0, 9.0), (1.0, 9.0))
    detection_packet = DetectionPacket(
        frame_index=1,
        timestamp_ms=10,
        width=1280,
        height=720,
        corner_detection=CornerDetection(corners=corners, confidence=0.9),
        telemetry=telemetry,
    )
    assert detection_packet.telemetry == telemetry
