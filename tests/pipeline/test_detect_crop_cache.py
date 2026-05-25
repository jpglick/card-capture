"""_run_cuda_inference warps from the GPU-resident tensor and fills the crop cache."""
import numpy as np
import pytest
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")


def _make_ctx(tmp_path):
    from pipeline.steps.start import RunContext
    return RunContext(
        video_path="/nonexistent/video.MOV",
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"),
        detector="cuda",
        config_preset="balanced",
        crops_dir=str(tmp_path / "crops"),
        frame_dir=str(tmp_path / "frames"),
        rotate_180=False,
        kornia_device="cpu",
        video_id=1,
    )


def test_run_cuda_inference_warps_from_gpu_tensor(tmp_path, monkeypatch):
    from pipeline.steps import detect
    from card_capture.models import FrameSample, CornerDetection

    H, W = 2160, 3840
    gpu_batch = torch.zeros((1, H, W, 3), dtype=torch.uint8)
    thumb = np.zeros((360, 640, 3), dtype=np.uint8)
    frames = [FrameSample(frame_index=5, timestamp_ms=166, image=thumb, width=W, height=H)]

    sampler = MagicMock()
    sampler.last_selected_frame_count = 1
    sampler.last_source_fps = 30.0
    sampler.sample_gpu_batches.return_value = iter([(gpu_batch, frames)])

    corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 140.0), (0.0, 140.0)]
    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detection_width = 640

    def _detect_batch(packets, conf):
        from card_capture.models import DetectionPacket
        out = []
        for p in packets:
            out.append(DetectionPacket(
                frame_index=p.frame_index,
                timestamp_ms=p.timestamp_ms,
                width=p.width,
                height=p.height,
                corner_detection=CornerDetection(corners=corners, confidence=0.9),
            ))
        return out
    detector.detect_batch.side_effect = _detect_batch

    captured = {}

    class _FakeNorm:
        def warp_canonical_batch_gpu(self, batch_items, rotate_180=False, return_gpu=False):
            captured["items"] = batch_items
            return [np.full((1050, 750, 3), i + 1, dtype=np.uint8) for i in range(len(batch_items))]
    monkeypatch.setattr(detect, "KorniaNormalizer", lambda **k: _FakeNorm())

    ctx = _make_ctx(tmp_path)
    crop_cache: dict = {}
    out = detect._run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path, crop_cache=crop_cache)

    assert len(out.detection_rows) == 1
    det_id = out.detection_rows[0]["detection_id"]
    assert det_id in crop_cache
    assert crop_cache[det_id].shape == (1050, 750, 3)
    # The warp received a GPU-tensor slice (not a numpy frame).
    assert isinstance(captured["items"][0][0], torch.Tensor)
