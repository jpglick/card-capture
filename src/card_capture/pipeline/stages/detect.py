"""Stage 3: YOLO Corner Detection (drains the FrameProducer).

Consumes frames from ``state["frame_producer"]`` as they decode (overlapping
decode with inference), appends each to ``state["sampled_frames"]`` for
downstream stages, batches them, and runs the detector. Loads the model once.
"""
from __future__ import annotations

from card_capture.detectors import (
    CardcaptorUltralyticsDetector,
    FakeCardDetector,
    probe_torch_device_status,
)
from card_capture.models import FramePacket


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    config = request.config

    if "yolo_model" not in state:
        telemetry.resource_sample({"event": "model_load", "model": "yolo_obb"})
        detector_name = config.get("detector", "fake")
        if detector_name == "fake":
            detector = FakeCardDetector()
        else:
            device = config.get("device", "auto")
            device_status = probe_torch_device_status(device)
            detector = CardcaptorUltralyticsDetector(
                confidence_threshold=config.get("corner_confidence", 0.5),
                detection_width=config.get("detection_width", 640),
                device=device_status.resolved,
            )
        state["yolo_model"] = detector

    detector = state["yolo_model"]
    conf = config.get("corner_confidence", 0.5)
    producer = state["frame_producer"]
    frames_out = state["sampled_frames"]
    estimated_total = max(1, int(state.get("estimated_frame_total", 0)) or 1)

    batch_size = 16
    detections = []
    batch: list = []
    processed = 0

    def _flush() -> None:
        nonlocal processed
        if not batch:
            return
        packets = [
            FramePacket(
                frame_index=f.frame_index,
                timestamp_ms=f.timestamp_ms,
                image=f.image,
                width=f.width,
                height=f.height,
                triage_metrics={},
            )
            for f in batch
        ]
        detections.extend(detector.detect_batch(packets, conf))
        processed += len(batch)
        pct = min(99, int(100 * processed / estimated_total))
        telemetry.progress("detect", pct, f"{processed} frames")
        batch.clear()

    try:
        for frame in producer:  # overlaps decode with inference
            frames_out.append(frame)
            batch.append(frame)
            if len(batch) >= batch_size:
                _flush()
        _flush()
    finally:
        producer.stop()  # guarantee the decode thread ends

    telemetry.progress("detect", 100, f"{processed} frames")

    rows = []
    for i, p in enumerate(detections):
        rows.append(
            {
                "detection_id": i + 1,
                "frame_index": p.frame_index,
                "timestamp_ms": p.timestamp_ms,
                "width": p.width,
                "height": p.height,
                "corners": p.corner_detection.corners,
                "confidence": p.corner_detection.confidence,
                "triage_metrics": {},
            }
        )

    state["detections"] = rows
