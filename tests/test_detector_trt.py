# tests/test_detector_trt.py
"""Detector picks a cached .engine, else exports it, else falls back to .pt FP16."""
from unittest.mock import MagicMock, patch
import card_capture.detectors as det


def _make_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "_resolve_model_path", lambda repo, fn: str(tmp_path / "model.pt"))
    (tmp_path / "model.pt").write_bytes(b"x")
    d = det.CardcaptorUltralyticsDetector(device="cuda")
    return d


def test_loads_existing_engine(tmp_path, monkeypatch):
    d = _make_detector(tmp_path, monkeypatch)
    (tmp_path / "model.engine").write_bytes(b"e")  # pretend a cached engine exists
    made = {}
    def _yolo(path, **kwargs):
        made["path"] = path
        m = MagicMock(); m.stride = 32; return m
    monkeypatch.setattr("ultralytics.YOLO", _yolo)
    monkeypatch.setattr(d, "_resolve_device", lambda: "cuda")
    d._load_model()
    assert made["path"].endswith("model.engine")  # preferred over .pt


def test_exports_engine_when_missing(tmp_path, monkeypatch):
    d = _make_detector(tmp_path, monkeypatch)
    calls = {"export": 0, "load": []}
    class _M:
        stride = 32
        def export(self, **kw):
            calls["export"] += 1
            assert kw["format"] == "engine" and kw["half"] is True
            (tmp_path / "model.engine").write_bytes(b"e")
            return str(tmp_path / "model.engine")
        def to(self, d): return self
    def _yolo(path, **kwargs):
        calls["load"].append(path)
        return _M()
    monkeypatch.setattr("ultralytics.YOLO", _yolo)
    monkeypatch.setattr(d, "_resolve_device", lambda: "cuda")
    d._load_model()
    assert calls["export"] == 1
    assert calls["load"][-1].endswith("model.engine")


def test_falls_back_to_pt_half_on_export_failure(tmp_path, monkeypatch):
    d = _make_detector(tmp_path, monkeypatch)
    class _M:
        stride = 32
        half_called = False
        def export(self, **kw):
            raise RuntimeError("no tensorrt")
        def to(self, d): return self
        def half(self): _M.half_called = True; return self
    monkeypatch.setattr("ultralytics.YOLO", lambda p, **kwargs: _M())
    monkeypatch.setattr(d, "_resolve_device", lambda: "cuda")
    d._load_model()
    # On export failure we keep the .pt model (engine load not attempted with a bad path)
    assert d._model is not None
