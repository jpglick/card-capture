from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

from card_capture.detectors import CardcaptorUltralyticsDetector
from card_capture.models import FrameSample


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

    passed_image = model_mock.call_args[0][0]
    assert passed_image.shape[1] == 640   # width
    assert passed_image.shape[0] == 480   # height: 960 * 640/1280 = 480


def test_detector_skips_resize_for_frame_already_small():
    """No resize when original_width <= detection_width."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    frame = _make_frame(height=480, width=320)

    model_mock = _empty_model()
    with patch.object(detector, "_load_model", return_value=model_mock):
        detector.detect(frame)

    passed_image = model_mock.call_args[0][0]
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
        [[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]],
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
    assert poly[1] == (400.0, 200.0)
    assert poly[2] == (400.0, 400.0)
    assert poly[3] == (200.0, 400.0)
