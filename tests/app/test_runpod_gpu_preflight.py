import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


def _import_handler(monkeypatch):
    runpod = types.SimpleNamespace(
        serverless=types.SimpleNamespace(start=MagicMock())
    )
    boto3 = types.SimpleNamespace(client=MagicMock())

    class _Config:
        def __init__(self, *args, **kwargs):
            pass

    botocore_config = types.SimpleNamespace(Config=_Config)

    monkeypatch.setitem(sys.modules, "runpod", runpod)
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    monkeypatch.delitem(sys.modules, "app.runpod_handler", raising=False)
    return importlib.import_module("app.runpod_handler")


def test_gpu_preflight_requires_video_capability(monkeypatch):
    mod = _import_handler(monkeypatch)

    info = {
        "capabilities": {
            "compute": True,
            "utility": True,
            "video": False,
        },
        "video": {
            "libnvcuvid_present": False,
            "decord_nvdec_probe": False,
            "error": "libnvcuvid.so.1 not found",
        },
    }

    with pytest.raises(RuntimeError, match="video"):
        mod._assert_gpu_preflight(info)


def test_gpu_preflight_passes_when_all_capabilities_are_confirmed(monkeypatch):
    mod = _import_handler(monkeypatch)

    info = {
        "capabilities": {
            "compute": True,
            "utility": True,
            "video": True,
        },
        "video": {
            "libnvcuvid_present": True,
            "decord_nvdec_probe": True,
        },
    }

    mod._assert_gpu_preflight(info)


def test_check_gpu_reports_video_from_real_probe_result(monkeypatch):
    mod = _import_handler(monkeypatch)

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    fake_torch.cuda.get_device_name.return_value = "RTX 4090"
    fake_torch.cuda.get_device_properties.return_value.total_memory = 24_000_000_000
    fake_torch.version.cuda = "12.4"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    smi_result = MagicMock(returncode=0, stdout="1, 508, 24564\n")
    monkeypatch.setattr(mod._sp, "run", MagicMock(return_value=smi_result))
    monkeypatch.setattr(
        mod,
        "_check_nvdec_video_capability",
        MagicMock(return_value={
            "libnvcuvid_present": True,
            "decord_gpu_context": True,
            "decord_nvdec_probe": True,
        }),
    )

    info = mod._check_gpu()

    assert info["capabilities"] == {
        "compute": True,
        "utility": True,
        "video": True,
    }
