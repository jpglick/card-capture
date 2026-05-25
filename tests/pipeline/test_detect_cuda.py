"""Tests for _run_cuda_inference — mocked sampler and detector."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

os.environ.setdefault("CC_CUDA_ALLOW_CPU_FALLBACK", "1")


def _make_frame_sample(frame_index: int):
    import numpy as np
    from card_capture.models import FrameSample
    return FrameSample(
        frame_index=frame_index,
        timestamp_ms=frame_index * 16,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        width=64,
        height=64,
    )


def _make_ctx(tmp_path, batch_size=4):
    from pipeline.steps.start import RunContext
    return RunContext(
        video_path=str(tmp_path / "video.mp4"),
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"),
        detector="cuda",
        config_preset="balanced",
        cuda_batch_size=batch_size,
        cuda_stride=2,
    )


def test_detect_output_frame_count(tmp_path):
    """DetectOutput.frame_count matches number of sampled frames."""
    from pipeline.steps.detect import _run_cuda_inference
    from card_capture.sampler.cuda_sampler import CudaSampler

    ctx = _make_ctx(tmp_path)
    frames = [_make_frame_sample(i) for i in range(10)]

    sampler = MagicMock(spec=CudaSampler)
    def _gpu_batches(batch_size=32, thumbnail_width=640, video_path=None):
        # CudaSampler now yields (gpu_tensor_batch, [FrameSample]); _run_cuda_inference
        # only indexes the tensor when a crop_cache is passed (not in these tests),
        # so a placeholder stands in for the GPU batch.
        from unittest.mock import MagicMock as _MM
        for i in range(0, len(frames), batch_size):
            yield _MM(), frames[i:i+batch_size]
    sampler.sample_gpu_batches.side_effect = _gpu_batches
    sampler.last_selected_frame_count = len(frames)
    sampler.last_source_fps = 60.0

    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detection_width = 640
    detector.detect_batch.return_value = []  # no detections

    out = _run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path)
    assert out.frame_count == 10
    assert out.accepted_frame_count == 10
    assert out.detection_rows == []


def test_detect_output_batching(tmp_path):
    """With batch_size=4 and 10 frames, detect_batch is called 3 times."""
    from pipeline.steps.detect import _run_cuda_inference
    from card_capture.sampler.cuda_sampler import CudaSampler

    ctx = _make_ctx(tmp_path, batch_size=4)
    frames = [_make_frame_sample(i) for i in range(10)]

    sampler = MagicMock(spec=CudaSampler)
    def _gpu_batches(batch_size=32, thumbnail_width=640, video_path=None):
        # CudaSampler now yields (gpu_tensor_batch, [FrameSample]); _run_cuda_inference
        # only indexes the tensor when a crop_cache is passed (not in these tests),
        # so a placeholder stands in for the GPU batch.
        from unittest.mock import MagicMock as _MM
        for i in range(0, len(frames), batch_size):
            yield _MM(), frames[i:i+batch_size]
    sampler.sample_gpu_batches.side_effect = _gpu_batches
    sampler.last_selected_frame_count = len(frames)
    sampler.last_source_fps = 60.0

    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detection_width = 640
    detector.detect_batch.return_value = []

    _run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path)
    # 10 frames / batch_size=4 → ceil(10/4) = 3 calls
    assert detector.detect_batch.call_count == 3


def test_detect_output_has_detection_rows(tmp_path):
    """Detections returned by detect_batch appear in detection_rows."""
    from pipeline.steps.detect import _run_cuda_inference
    from card_capture.sampler.cuda_sampler import CudaSampler
    from card_capture.models import DetectionPacket, FramePacket, CornerDetection

    ctx = _make_ctx(tmp_path, batch_size=8)
    frames = [_make_frame_sample(0)]

    sampler = MagicMock(spec=CudaSampler)
    def _gpu_batches(batch_size=32, thumbnail_width=640, video_path=None):
        # CudaSampler now yields (gpu_tensor_batch, [FrameSample]); _run_cuda_inference
        # only indexes the tensor when a crop_cache is passed (not in these tests),
        # so a placeholder stands in for the GPU batch.
        from unittest.mock import MagicMock as _MM
        for i in range(0, len(frames), batch_size):
            yield _MM(), frames[i:i+batch_size]
    sampler.sample_gpu_batches.side_effect = _gpu_batches
    sampler.last_selected_frame_count = 1
    sampler.last_source_fps = 60.0

    # Fake detection packet
    cd = MagicMock()
    cd.corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    cd.confidence = 0.85
    pkt = MagicMock()
    pkt.frame_index = 0
    pkt.timestamp_ms = 0
    pkt.width = 64
    pkt.height = 64
    pkt.corner_detection = cd

    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detection_width = 640
    detector.detect_batch.return_value = [pkt]

    out = _run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path)
    assert len(out.detection_rows) == 1
    assert out.detection_rows[0]["confidence"] == pytest.approx(0.85)
    assert out.detection_rows[0]["frame_index"] == 0
