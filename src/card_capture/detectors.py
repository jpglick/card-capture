from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Protocol

import cv2

from .models import (
    CardDetection,
    CornerDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
    Polygon,
)


class CardDetector(Protocol):
    runtime: str
    model_name: str

    def detect(self, frame: FrameSample) -> Iterable[CardDetection]:
        ...


class CornerDetector(Protocol):
    def detect_batch(
        self, frames: list[FramePacket], confidence_threshold: float
    ) -> list[DetectionPacket]:
        ...


class FakeCardDetector:
    runtime = "fake"
    model_name = "fake-card-detector"

    def detect(self, frame: FrameSample) -> List[CardDetection]:
        x0 = frame.width * 0.12
        x1 = frame.width * 0.88
        y0 = frame.height * 0.12
        y1 = frame.height * 0.88
        return [
            CardDetection(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                polygon=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                confidence=0.99,
                metadata={"runtime": self.runtime, "model": self.model_name},
            )
        ]


class FakeCornerDetector:
    runtime = "fake"
    model_name = "fake-corner-detector"

    def __init__(self, confidence: float = 0.99):
        self.confidence = confidence

    def detect_batch(
        self, frames: list[FramePacket], confidence_threshold: float
    ) -> list[DetectionPacket]:
        if self.confidence < confidence_threshold:
            return []

        detections: list[DetectionPacket] = []
        for frame in frames:
            x0 = frame.width * 0.12
            x1 = frame.width * 0.88
            y0 = frame.height * 0.12
            y1 = frame.height * 0.88
            detections.append(
                DetectionPacket(
                    frame_index=frame.frame_index,
                    timestamp_ms=frame.timestamp_ms,
                    width=frame.width,
                    height=frame.height,
                    corner_detection=CornerDetection(
                        corners=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                        confidence=self.confidence,
                        metadata={"runtime": self.runtime, "model": self.model_name},
                    ),
                )
            )
        return detections


class CardcaptorUltralyticsDetector:
    runtime = "ultralytics"
    model_name = "AlecKarfonta/cardcaptor-v3"

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        repo_id: str = "AlecKarfonta/cardcaptor-v3",
        filename: str = "weights/cardcaptor_v3_best.pt",
        detection_width: int = 640,
        device: str = "auto",
    ):
        if detection_width <= 0:
            raise ValueError(f"detection_width must be a positive integer, got {detection_width!r}")
        self.confidence_threshold = confidence_threshold
        self.repo_id = repo_id
        self.filename = filename
        self.detection_width = detection_width
        self.device = device
        self._model = None

    def detect(self, frame: FrameSample) -> List[CardDetection]:
        model = self._load_model()
        original_h, original_w = frame.image.shape[:2]

        if original_w > self.detection_width:
            scaled_w = self.detection_width
            scaled_h = max(1, int(round(original_h * self.detection_width / original_w)))
            detect_image = cv2.resize(frame.image, (scaled_w, scaled_h))
            scale_x = original_w / scaled_w
            scale_y = original_h / scaled_h
        else:
            detect_image = frame.image
            scale_x = 1.0
            scale_y = 1.0

        results = model(detect_image, conf=self.confidence_threshold, verbose=False)
        detections: List[CardDetection] = []
        for result in results:
            obb = getattr(result, "obb", None)
            if obb is None or obb.conf is None:
                continue
            polygons = obb.xyxyxyxy.cpu().numpy()
            confidences = obb.conf.cpu().numpy()
            labels = (
                obb.cls.cpu().numpy()
                if obb.cls is not None
                else [0] * len(confidences)
            )
            for polygon_array, confidence, label in zip(polygons, confidences, labels):
                confidence_float = float(confidence)
                if confidence_float < self.confidence_threshold:
                    continue
                polygon = tuple(
                    (float(point[0]) * scale_x, float(point[1]) * scale_y)
                    for point in polygon_array
                )
                if len(polygon) != 4:
                    continue

                import numpy as np
                import cv2
                frame_area = frame.width * frame.height
                poly_area = cv2.contourArea(np.array(polygon, dtype=np.float32))
                if not (0.1 * frame_area <= poly_area <= 0.8 * frame_area):
                    continue

                detections.append(
                    CardDetection(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        polygon=polygon,  # type: ignore[arg-type]
                        confidence=confidence_float,
                        metadata={
                            "runtime": self.runtime,
                            "model": self.model_name,
                            "class_id": int(label),
                        },
                    )
                )
        return detections

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Real card detection requires optional dependencies. "
                "Install with: pip install '.[model]'"
            ) from exc

        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
        self._model = YOLO(model_path)
        self._model.to(self._resolve_device())
        return self._model

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
