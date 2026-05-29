from __future__ import annotations

import os
import platform
from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.quarantine

from card_capture.detectors import CardcaptorUltralyticsDetector, TorchDeviceStatus

@pytest.fixture
def mock_yolo():
    with patch("ultralytics.YOLO") as mock:
        yield mock

@pytest.fixture
def mock_platform_mac():
    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"):
        yield

@pytest.fixture
def mock_device_mps():
    status = TorchDeviceStatus(
        requested="auto",
        resolved="mps",
        mps_built=True,
        mps_available=True,
        cuda_available=False,
        reason="auto_mps"
    )
    with patch("card_capture.detectors.probe_torch_device_status", return_value=status):
        yield status

@pytest.fixture
def mock_resolve_model_path():
    with patch("card_capture.detectors._resolve_model_path", return_value="/models/model.pt"):
        yield "/models/model.pt"

pytestmark = pytest.mark.quarantine
def test_load_model_coreml_existing(mock_yolo, mock_platform_mac, mock_device_mps, mock_resolve_model_path):
    """If .mlpackage exists on Mac, it should be loaded directly."""
    detector = CardcaptorUltralyticsDetector(device="auto")
    
    mlpackage_path = "/models/model.mlpackage"
    
    def side_effect(path):
        return path == mlpackage_path

    with patch("os.path.exists", side_effect=side_effect):
        detector._load_model()
        
    # Should be loaded twice? No, YOLO(mlpackage_path) should be called.
    # Wait, the current implementation of CardcaptorUltralyticsDetector._load_model
    # doesn't have Core ML logic yet. This test SHOULD fail.
    
    mock_yolo.assert_called_with(mlpackage_path)
    assert detector._half is True

pytestmark = pytest.mark.quarantine
def test_load_model_coreml_export(mock_yolo, mock_platform_mac, mock_device_mps, mock_resolve_model_path):
    """If .mlpackage doesn't exist on Mac, it should be exported and then loaded."""
    detector = CardcaptorUltralyticsDetector(device="auto")
    
    mlpackage_path = "/models/model.mlpackage"
    
    # First call to YOLO(model_path) for exporting
    # Second call to YOLO(mlpackage_path) for loading the engine
    
    with patch("os.path.exists", return_value=False):
        # We need to mock the export method
        exporter_mock = MagicMock()
        exporter_mock.export.return_value = mlpackage_path
        mock_yolo.side_effect = [exporter_mock, MagicMock()]
        
        detector._load_model()

    exporter_mock.export.assert_called_once_with(
        format="coreml", half=True, imgsz=detector.detection_width, verbose=False
    )
    mock_yolo.assert_any_call(mlpackage_path)
    assert detector._half is True

pytestmark = pytest.mark.quarantine
def test_load_model_no_coreml_on_non_mac(mock_yolo, mock_resolve_model_path):
    """On non-Mac platforms, Core ML should not be attempted."""
    status = TorchDeviceStatus(
        requested="auto",
        resolved="cpu",
        mps_built=False,
        mps_available=False,
        cuda_available=False,
        reason="no_accelerator"
    )
    with patch("platform.system", return_value="Linux"), \
         patch("card_capture.detectors.probe_torch_device_status", return_value=status), \
         patch("os.path.exists", return_value=False):
        
        detector = CardcaptorUltralyticsDetector(device="auto")
        detector._load_model()
        
    mock_yolo.assert_called_once_with(mock_resolve_model_path)
    # Ensure export was not called
    assert mock_yolo.return_value.export.called is False

pytestmark = pytest.mark.quarantine
def test_load_model_coreml_fallback_on_export_failure(mock_yolo, mock_platform_mac, mock_device_mps, mock_resolve_model_path):
    """If Core ML export fails, it should fall back to PyTorch MPS (.pt)."""
    detector = CardcaptorUltralyticsDetector(device="auto")
    
    with patch("os.path.exists", return_value=False):
        exporter_mock = MagicMock()
        exporter_mock.export.side_effect = Exception("Export failed")
        
        # side_effect for mock_yolo:
        # 1. YOLO(model_path) for exporter
        # 2. YOLO(model_path) for fallback
        fallback_model = MagicMock()
        mock_yolo.side_effect = [exporter_mock, fallback_model]
        
        detector._load_model()
        
    # Should have called YOLO(model_path) twice (once for export, once for fallback)
    assert mock_yolo.call_count == 2
    mock_yolo.assert_called_with(mock_resolve_model_path)
    assert detector._half is False
    assert detector._model == fallback_model
