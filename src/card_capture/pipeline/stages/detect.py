"""Stage 3: YOLO Corner Detection.

Reuses `state["sampled_frames"]` produced by the sample stage. Loads the
YOLO model once on first call and stashes it in state for any later stage
that needs it (none currently; refine uses its own model).
"""
from __future__ import annotations

from card_capture.detectors import FakeCardDetector, CardcaptorUltralyticsDetector, probe_torch_device_status
from card_capture.models import FramePacket


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    config = request.config
    
    if "yolo_model" not in state:
        telemetry.resource_sample({"event": "model_load", "model": "yolo_obb"})
        # In a real implementation we would select the detector based on config
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

    frames = state["sampled_frames"]
    # Convert FrameSamples to FramePackets for detection
    packets = [
        FramePacket(
            frame_index=f.frame_index,
            timestamp_ms=f.timestamp_ms,
            image=f.image,
            width=f.width,
            height=f.height,
            triage_metrics={},
        )
        for f in frames
    ]
    # Execute
    detections = state["yolo_model"].detect_batch(packets, config.get("corner_confidence", 0.5))

    rows = []
    for i, p in enumerate(detections):
        rows.append({
            "detection_id": i + 1,
            "frame_index": p.frame_index,
            "timestamp_ms": p.timestamp_ms,
            "width": p.width,
            "height": p.height,
            "corners": p.corner_detection.corners,
            "confidence": p.corner_detection.confidence,
            "triage_metrics": {},
        })
        
    state["detections"] = rows
