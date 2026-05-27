# tests/pipeline/test_detect_prefetch.py
"""Prefetched inference yields the same detections, in frame order, as serial."""
import numpy as np, pytest
from unittest.mock import MagicMock
torch = pytest.importorskip("torch")


def test_prefetch_preserves_order_and_detections(tmp_path, monkeypatch):
    from pipeline.steps import detect
    from pipeline.steps.start import RunContext
    from card_capture.models import FrameSample, DetectionPacket, CornerDetection

    ctx = RunContext(video_path="/x.MOV", output_dir=str(tmp_path), db_path=str(tmp_path/"c.sqlite"),
                     detector="cuda", config_preset="balanced", crops_dir=str(tmp_path/"crops"),
                     frame_dir=str(tmp_path/"frames"), rotate_180=False, kornia_device="cpu", video_id=1)

    H, W = 64, 64
    def _frame(fi): return FrameSample(frame_index=fi, timestamp_ms=fi*16,
                                       image=np.zeros((H,W,3),dtype=np.uint8), width=W, height=H)
    batches = [(torch.zeros((2,H,W,3),dtype=torch.uint8), [_frame(0),_frame(1)]),
               (torch.zeros((2,H,W,3),dtype=torch.uint8), [_frame(2),_frame(3)])]
    sampler = MagicMock(); sampler.last_selected_frame_count=4; sampler.last_source_fps=30.0
    sampler.sample_gpu_batches.return_value = iter(batches)

    cd = lambda: CornerDetection(corners=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)], confidence=0.9)
    detector = MagicMock(); detector.confidence_threshold=0.5; detector.detection_width=64
    detector.detect_batch.side_effect = lambda packets, conf: [
        DetectionPacket(frame_index=p.frame_index, timestamp_ms=p.timestamp_ms,
                        width=p.width, height=p.height, corner_detection=cd()) for p in packets]
    monkeypatch.setattr(detect, "KorniaNormalizer", lambda **k: MagicMock(
        warp_canonical_batch_gpu=lambda items, rotate_180=False, return_gpu=False: [np.zeros((1050,750,3),np.uint8) for _ in items]))

    crop_cache = {}
    out = detect._run_fused_inference(ctx, sampler, detector, tmp_path, tmp_path, crop_cache=crop_cache)
    assert [r["frame_index"] for r in out.detection_rows] == [0,1,2,3]   # order preserved
    assert out.frame_count == 4
    assert len(crop_cache) == 4


def test_prefetch_propagates_errors(tmp_path, monkeypatch):
    from pipeline.steps import detect
    from pipeline.steps.start import RunContext

    ctx = RunContext(video_path="/x.MOV", output_dir=str(tmp_path), db_path=str(tmp_path/"c.sqlite"),
                     detector="cuda", config_preset="balanced", crops_dir=str(tmp_path/"crops"),
                     frame_dir=str(tmp_path/"frames"), rotate_180=False, kornia_device="cpu", video_id=1)

    sampler = MagicMock()
    def _fail(**k):
        yield (None, [])
        raise RuntimeError("decode failed")
    sampler.sample_gpu_batches.side_effect = _fail

    detector = MagicMock()
    detector.detect_batch.return_value = []

    with pytest.raises(RuntimeError, match="decode failed"):
        detect._run_fused_inference(ctx, sampler, detector, tmp_path, tmp_path)
