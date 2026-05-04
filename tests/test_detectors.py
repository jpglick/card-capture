from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from card_capture.detectors import (
    CardcaptorUltralyticsDetector,
    FakeCornerDetector,
    probe_torch_device_status,
)
from card_capture.models import DetectionPacket, FramePacket, FrameSample


def _make_frame(height: int, width: int) -> FrameSample:
    return FrameSample(
        frame_index=0,
        timestamp_ms=0,
        image=np.zeros((height, width, 3), dtype=np.uint8),
        width=width,
        height=height,
    )


def _empty_model():
    """YOLO model mock that returns no detections."""
    m = MagicMock()
    m.return_value = []
    return m


def test_detector_downscales_wide_frame_before_inference():
    """Model receives a 640-wide image when original frame is wider."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    frame = _make_frame(height=960, width=1280)

    model_mock = _empty_model()
    with patch.object(detector, "_load_model", return_value=model_mock):
        detector.detect(frame)

    passed_image = model_mock.call_args[0][0][0]
    assert passed_image.shape[1] == 640   # width
    assert passed_image.shape[0] == 480   # height: 960 * 640/1280 = 480


def test_detector_skips_resize_for_frame_already_small():
    """No resize when original_width <= detection_width."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    frame = _make_frame(height=480, width=320)

    model_mock = _empty_model()
    with patch.object(detector, "_load_model", return_value=model_mock):
        detector.detect(frame)

    passed_image = model_mock.call_args[0][0][0]
    assert passed_image.shape[1] == 320
    assert passed_image.shape[0] == 480


def test_detector_rescales_polygon_to_original_frame_space():
    """Polygon coordinates from detection space must be scaled back to original
    frame dimensions using separate x and y scale factors."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    # 1280x960 → detection at 640x480 → scale_x = scale_y = 2.0
    frame = _make_frame(height=960, width=1280)

    obb_mock = MagicMock()
    obb_mock.conf.cpu.return_value.numpy.return_value = np.array([0.9])
    obb_mock.cls.cpu.return_value.numpy.return_value = np.array([0])
    # Points in 640x480 detection space: a 100x100 square
    obb_mock.xyxyxyxy.cpu.return_value.numpy.return_value = np.array(
        [[[100.0, 100.0], [500.0, 100.0], [500.0, 300.0], [100.0, 300.0]]],
        dtype=np.float32,
    )
    result_mock = MagicMock()
    result_mock.obb = obb_mock
    model_mock = MagicMock()
    model_mock.return_value = [result_mock]

    with patch.object(detector, "_load_model", return_value=model_mock):
        detections = detector.detect(frame)

    assert len(detections) == 1
    poly = detections[0].polygon
    # scale_x = 1280/640 = 2.0, scale_y = 960/480 = 2.0
    assert poly[0] == (200.0, 200.0)
    assert poly[1] == (1000.0, 200.0)
    assert poly[2] == (1000.0, 600.0)
    assert poly[3] == (200.0, 600.0)


def test_detector_rescales_polygon_with_asymmetric_scale_factors():
    """Verifies scale_x is applied to x and scale_y to y independently."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    # 1280x721 -> scaled_w=640, scaled_h=round(721*640/1280)=round(360.5)=360
    # scale_x = 1280/640 = 2.0, scale_y = 721/360 ≈ 2.00278
    frame = _make_frame(height=721, width=1280)

    obb_mock = MagicMock()
    obb_mock.conf.cpu.return_value.numpy.return_value = np.array([0.9])
    obb_mock.cls.cpu.return_value.numpy.return_value = np.array([0])
    obb_mock.xyxyxyxy.cpu.return_value.numpy.return_value = np.array(
        [[[100.0, 100.0], [500.0, 100.0], [500.0, 300.0], [100.0, 300.0]]],
        dtype=np.float32,
    )
    result_mock = MagicMock()
    result_mock.obb = obb_mock
    model_mock = MagicMock()
    model_mock.return_value = [result_mock]

    with patch.object(detector, "_load_model", return_value=model_mock):
        detections = detector.detect(frame)

    poly = detections[0].polygon
    # x: 100 * (1280/640) = 100 * 2.0 = 200.0
    # y: 100 * (721/360) ≈ 200.2778
    assert poly[0][0] == pytest.approx(200.0)
    assert poly[0][1] == pytest.approx(200.27777777777777)


def test_detector_skips_resize_when_frame_exactly_matches_detection_width():
    """original_w == detection_width must NOT resize (guard is >, not >=)."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    frame = _make_frame(height=480, width=640)

    model_mock = _empty_model()
    with patch.object(detector, "_load_model", return_value=model_mock):
        detector.detect(frame)

    passed_image = model_mock.call_args[0][0][0]
    assert passed_image.shape[1] == 640
    assert passed_image.shape[0] == 480
    assert passed_image is frame.image


def test_detector_detect_batch_returns_detection_packets():
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    frames = [
        FramePacket(
            frame_index=7,
            timestamp_ms=70,
            image=np.zeros((960, 1280, 3), dtype=np.uint8),
            width=1280,
            height=960,
            triage_metrics={},
        )
    ]

    obb_mock = MagicMock()
    obb_mock.conf.cpu.return_value.numpy.return_value = np.array([0.9])
    obb_mock.cls.cpu.return_value.numpy.return_value = np.array([0])
    obb_mock.xyxyxyxy.cpu.return_value.numpy.return_value = np.array(
        [[[100.0, 100.0], [500.0, 100.0], [500.0, 300.0], [100.0, 300.0]]],
        dtype=np.float32,
    )
    result_mock = MagicMock()
    result_mock.obb = obb_mock
    model_mock = MagicMock()
    model_mock.return_value = [result_mock]

    with patch.object(detector, "_load_model", return_value=model_mock):
        detections = detector.detect_batch(frames, confidence_threshold=0.5)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.frame_index == 7
    assert detection.timestamp_ms == 70
    assert detection.corner_detection.corners[0] == (200.0, 200.0)


def test_detector_raises_on_nonpositive_detection_width():
    """detection_width <= 0 must raise ValueError at construction time."""
    with pytest.raises(ValueError, match="detection_width"):
        CardcaptorUltralyticsDetector(detection_width=0)
    with pytest.raises(ValueError, match="detection_width"):
        CardcaptorUltralyticsDetector(detection_width=-1)


def test_fake_corner_detector_returns_one_detection_packet_per_frame_at_threshold():
    detector = FakeCornerDetector(confidence=0.99)
    frames = [
        FramePacket(
            frame_index=10,
            timestamp_ms=111,
            image=np.zeros((240, 320, 3), dtype=np.uint8),
            width=320,
            height=240,
            triage_metrics={},
        ),
        FramePacket(
            frame_index=11,
            timestamp_ms=222,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            width=640,
            height=480,
            triage_metrics={},
        ),
    ]

    detections = detector.detect_batch(frames, confidence_threshold=0.99)

    assert len(detections) == len(frames)
    for frame, detection in zip(frames, detections):
        assert isinstance(detection, DetectionPacket)
        assert detection.frame_index == frame.frame_index
        assert detection.timestamp_ms == frame.timestamp_ms
        assert detection.width == frame.width
        assert detection.height == frame.height
        assert detection.corner_detection.confidence == 0.99


def test_fake_corner_detector_returns_empty_list_below_threshold():
    detector = FakeCornerDetector(confidence=0.49)
    frames = [
        FramePacket(
            frame_index=0,
            timestamp_ms=0,
            image=np.zeros((240, 320, 3), dtype=np.uint8),
            width=320,
            height=240,
            triage_metrics={},
        )
    ]

    detections = detector.detect_batch(frames, confidence_threshold=0.5)

    assert detections == []


def test_probe_torch_device_status_reports_mps_unavailable_state():
    class _MPS:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_built():
            return True

    class _Backends:
        mps = _MPS()

    class _Torch:
        backends = _Backends()
        class cuda:
            @staticmethod
            def is_available():
                return False

    with patch("platform.system", return_value="Darwin"), patch(
        "platform.machine", return_value="arm64"
    ), patch.dict("sys.modules", {"torch": _Torch()}):
        status = probe_torch_device_status("auto")

    assert status.resolved == "cpu"
    assert status.mps_built is True
    assert status.mps_available is False
    assert status.reason == "mps_unavailable"
