"""Stage 1: Adaptive Presence Sampler (streaming producer).

Starts a background decode thread (``FrameProducer``) so the ``detect`` stage can
run inference while later frames are still decoding. Frames are NOT materialized
here; ``detect`` drains the producer and fills ``state["sampled_frames"]``.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from card_capture.shared.stage_metrics import emit_stage_metrics
from card_capture.stages.sample.sampler import StrideSampler
from card_capture.stages.sample.sampler.frame_producer import FrameProducer


def _estimate_selected_count(video_path: str, target_fps: float) -> int:
    """Cheap header probe -> estimated kept-frame count (for detect progress)."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if src_fps <= 0 or total <= 0 or target_fps <= 0:
        return 0
    stride = max(1, round(src_fps / target_fps))
    return int(total // stride)


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    video_path = request.input_video.replace("artifact://local/", "")
    telemetry.resource_sample({"event": "decode_open", "path": video_path})

    sampler = StrideSampler(video_path=Path(video_path))
    # stall_timeout: if decode wedges mid-stream (no frame for this long while the
    # thread is alive), detect raises and the run fails loudly instead of hanging.
    producer = FrameProducer(sampler, stall_timeout=120.0).start()
    # Block until the first frame is decoded so this stage's timing reflects
    # decode startup; all remaining frames overlap the detect stage.
    producer.wait_first(timeout=60.0)

    state["sampler"] = sampler
    state["video_path"] = video_path
    state["frame_producer"] = producer
    state["sampled_frames"] = []  # filled by the detect stage as it drains
    state["estimated_frame_total"] = _estimate_selected_count(
        video_path, sampler.target_yolo_fps
    )
    emit_stage_metrics(
        state,
        stage="sample",
        metrics={"estimated_frames": int(state["estimated_frame_total"])},
    )
