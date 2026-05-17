from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol
import platform

import cv2
import numpy as np

from .models import (
    CardDetection,
    CornerDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
    Polygon,
)

from huggingface_hub import hf_hub_download
from huggingface_hub import try_to_load_from_cache


def _try_local_cache(repo_id: str, filename: str):
    """Return the local cached path for (repo_id, filename), else None."""
    try:
        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    except Exception:
        return None
    if isinstance(cached, str):
        return cached
    return None


def _resolve_model_path(repo_id: str, filename: str) -> str:
    """Prefer a locally cached weights file. Only hit the Hub if no cache exists."""
    cached = _try_local_cache(repo_id, filename)
    if cached:
        return cached
    return hf_hub_download(repo_id=repo_id, filename=filename)


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


@dataclass(frozen=True)
class TorchDeviceStatus:
    requested: str
    resolved: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    reason: str


def probe_torch_device_status(requested: str = "auto") -> TorchDeviceStatus:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported device request: {requested!r}")

    try:
        import torch
    except ImportError:
        return TorchDeviceStatus(
            requested=normalized,
            resolved="cpu",
            mps_built=False,
            mps_available=False,
            cuda_available=False,
            reason="torch_not_installed",
        )

    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())

    if normalized == "cpu":
        return TorchDeviceStatus(normalized, "cpu", mps_built, mps_available, cuda_available, "forced_cpu")
    if normalized == "mps":
        return TorchDeviceStatus(
            normalized,
            "mps" if mps_available else "cpu",
            mps_built,
            mps_available,
            cuda_available,
            "forced_mps" if mps_available else "mps_unavailable",
        )
    if normalized == "cuda":
        return TorchDeviceStatus(
            normalized,
            "cuda" if cuda_available else "cpu",
            mps_built,
            mps_available,
            cuda_available,
            "forced_cuda" if cuda_available else "cuda_unavailable",
        )

    if platform.system() == "Darwin" and platform.machine() == "arm64" and mps_available:
        return TorchDeviceStatus(normalized, "mps", mps_built, mps_available, cuda_available, "auto_mps")
    if cuda_available:
        return TorchDeviceStatus(normalized, "cuda", mps_built, mps_available, cuda_available, "auto_cuda")
    return TorchDeviceStatus(
        normalized,
        "cpu",
        mps_built,
        mps_available,
        cuda_available,
        "mps_unavailable" if mps_built else "no_accelerator",
    )


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
        detections = self.detect_batch(
            [
                FramePacket(
                    frame_index=frame.frame_index,
                    timestamp_ms=frame.timestamp_ms,
                    image=frame.image,
                    width=frame.width,
                    height=frame.height,
                    triage_metrics={},
                )
            ],
            confidence_threshold=self.confidence_threshold,
        )
        return [
            CardDetection(
                frame_index=detection.frame_index,
                timestamp_ms=detection.timestamp_ms,
                polygon=detection.corner_detection.corners,
                confidence=detection.corner_detection.confidence,
                metadata=dict(detection.corner_detection.metadata),
            )
            for detection in detections
        ]

    def detect_batch(
        self, frames: list[FramePacket], confidence_threshold: float
    ) -> list[DetectionPacket]:
        if not frames:
            return []
        model = self._load_model()
        detect_images: list[np.ndarray] = []
        scale_factors: list[tuple[float, float, int, int]] = []
        for frame in frames:
            packet_h, packet_w = frame.image.shape[:2]
            if packet_w > self.detection_width:
                scaled_w = self.detection_width
                scaled_h = max(1, int(round(packet_h * self.detection_width / packet_w)))
                detect_image = cv2.resize(frame.image, (scaled_w, scaled_h))
            else:
                detect_image = frame.image
                scaled_w = packet_w
                scaled_h = packet_h
                
            scale_x = frame.width / scaled_w
            scale_y = frame.height / scaled_h
            
            detect_images.append(detect_image)
            scale_factors.append((scale_x, scale_y, frame.width, frame.height))

        results = model(detect_images, conf=confidence_threshold, verbose=False)
        detections: List[DetectionPacket] = []
        for frame, result, (scale_x, scale_y, frame_width, frame_height) in zip(frames, results, scale_factors):
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
                if confidence_float < confidence_threshold:
                    continue
                polygon = tuple(
                    (float(point[0]) * scale_x, float(point[1]) * scale_y)
                    for point in polygon_array
                )
                if len(polygon) != 4:
                    continue

                poly_arr = np.array(polygon, dtype=np.float32)
                poly_area = cv2.contourArea(poly_arr)
                frame_area = frame.width * frame.height
                if not (0.1 * frame_area <= poly_area <= 0.8 * frame_area):
                    continue

                # Check aspect ratio (short_side / long_side should be ~0.714)
                # Relaxed gate (0.50 - 0.95) to allow for hand-held cards with perspective.
                width_top = np.linalg.norm(poly_arr[1] - poly_arr[0])
                width_bottom = np.linalg.norm(poly_arr[2] - poly_arr[3])
                height_right = np.linalg.norm(poly_arr[2] - poly_arr[1])
                height_left = np.linalg.norm(poly_arr[3] - poly_arr[0])
                w = max(width_top, width_bottom)
                h = max(height_right, height_left)
                ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                if not (0.50 <= ratio <= 0.95):
                    continue

                detections.append(
                    DetectionPacket(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        width=frame.width,
                        height=frame.height,
                        corner_detection=CornerDetection(
                            corners=polygon,  # type: ignore[arg-type]
                            confidence=confidence_float,
                            metadata={
                                "runtime": self.runtime,
                                "model": self.model_name,
                                "class_id": int(label),
                            },
                        ),
                    )
                )
        return detections

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Real card detection requires optional dependencies. "
                "Install with: pip install '.[model]'"
            ) from exc

        # Backend selection notes (benchmarked on M4, YOLO11m-OBB, batch=16):
        #   MPS PyTorch:          94ms/frame  ← winner
        #   ONNX+CoreML EP b=1:  325ms/frame  (ANE for 419/441 ops, but loop overhead dominates)
        #   ONNX CPU b=16:       296ms/frame  (CoreML EP rejects b>1, falls back to CPU)
        # MPS wins by 3×. ANE excels at lightweight mobile architectures; YOLO11m's
        # 134 layers saturate the GPU more efficiently. ONNX kept in models/ as a
        # verified-correct fallback for non-MPS environments.

        # Primary: MPS / CPU
        model_path = _resolve_model_path(self.repo_id, self.filename)
        self._model = YOLO(model_path)
        self._device = self._resolve_device()
        self._model.to(self._device)
        print(f"[detector] backend={self._device}", flush=True)
        return self._model

    def _resolve_device(self) -> str:
        return probe_torch_device_status(self.device).resolved
