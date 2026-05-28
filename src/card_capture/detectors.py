from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union

import cv2
import numpy as np
import torch

from .interfaces import CardDetector
from .models import (
    CardDetection,
    CornerDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
    Point,
    TorchDeviceStatus,
)


class FakeCornerDetector:
    """Dummy detector that finds a card in the center of every frame with configurable confidence."""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def detect(self, frame: FrameSample) -> List[CardDetection]:
        h, w = frame.image.shape[:2]
        return [
            CardDetection(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                polygon=((w // 4, h // 4), (3 * w // 4, h // 4), (3 * w // 4, 3 * h // 4), (w // 4, 3 * h // 4)),
                confidence=self.confidence,
            )
        ]

    def detect_batch(
        self,
        frames: list[FramePacket],
        confidence_threshold: float,
        tensor_input: Optional["torch.Tensor"] = None,
    ) -> list[DetectionPacket]:
        if self.confidence < confidence_threshold:
            return []
        results = []
        for f in frames:
            h, w = f.height, f.width
            cd = CornerDetection(
                corners=((w // 4, h // 4), (3 * w // 4, h // 4), (3 * w // 4, 3 * h // 4), (w // 4, 3 * h // 4)),
                confidence=self.confidence,
            )
            results.append(
                DetectionPacket(
                    frame_index=f.frame_index,
                    timestamp_ms=f.timestamp_ms,
                    width=w,
                    height=h,
                    corner_detection=cd,
                )
            )
        return results


class FakeCardDetector(CardDetector):
    """Dummy detector that finds a card in the center of every frame."""

    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: FrameSample) -> List[CardDetection]:
        h, w = frame.image.shape[:2]
        return [
            CardDetection(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                polygon=((w // 4, h // 4), (3 * w // 4, h // 4), (3 * w // 4, 3 * h // 4), (w // 4, 3 * h // 4)),
                confidence=0.95,
            )
        ]

    def detect_batch(
        self,
        frames: list[FramePacket],
        confidence_threshold: float,
        tensor_input: Optional["torch.Tensor"] = None,
    ) -> list[DetectionPacket]:
        results = []
        for f in frames:
            h, w = f.height, f.width
            cd = CornerDetection(
                corners=((w // 4, h // 4), (3 * w // 4, h // 4), (3 * w // 4, 3 * h // 4), (w // 4, 3 * h // 4)),
                confidence=0.95,
            )
            results.append(
                DetectionPacket(
                    frame_index=f.frame_index,
                    timestamp_ms=f.timestamp_ms,
                    width=w,
                    height=h,
                    corner_detection=cd,
                )
            )
        return results


class CardcaptorUltralyticsDetector(CardDetector):
    """OBB detector using Ultralytics YOLOv8/v11."""

    def __init__(
        self,
        repo_id: str = "AlecKarfonta/cardcaptor-v3",
        filename: str = "cardcaptor_v3_best.pt",
        detection_width: int = 640,
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ):
        if detection_width <= 0:
            raise ValueError("detection_width must be positive")
        self.repo_id = repo_id
        self.filename = filename
        self.detection_width = detection_width
        self._conf_thresh = confidence_threshold
        self.device = device
        self._model = None
        self._half = False
        self._is_coreml = False
        self._device = "cpu"  # Default before load

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
        self,
        frames: list[FramePacket],
        confidence_threshold: float,
        tensor_input: Optional["torch.Tensor"] = None,
    ) -> list[DetectionPacket]:
        if not frames:
            return []
        model = self._load_model()
        
        # Scale factors: (scale_x, scale_y) to map detection coordinates back to original resolution.
        scale_factors: list[tuple[float, float, int, int]] = []
        
        if tensor_input is not None:
            # Optimized tensor path (e.g. CUDA)
            if self._is_coreml:
                # COREML STABILITY FIX: Sequential inference for batch tensors.
                # Ultralytics CoreML predictor is unstable with batch_size > 1.
                results = []
                for i in range(tensor_input.shape[0]):
                    res = model(tensor_input[i : i + 1], conf=confidence_threshold, verbose=False)
                    if res:
                        results.append(res[0])
            else:
                results = model(tensor_input, conf=confidence_threshold, verbose=False)

            for frame in frames:
                sx = frame.width / tensor_input.shape[3]
                sy = frame.height / tensor_input.shape[2]
                scale_factors.append((sx, sy, frame.width, frame.height))
        else:
            # Standard path: feed numpy images to YOLO.
            detect_images = []
            for f in frames:
                h, w = f.image.shape[:2]
                if max(h, w) > self.detection_width:
                    # Manual resize to detection_width (maintaining aspect ratio) as expected by tests
                    scale = self.detection_width / max(h, w)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    resized = cv2.resize(f.image, (new_w, new_h))
                    detect_images.append(resized)
                    
                    # sx/sy scale from the resized image to the original resolution
                    sx = f.width / new_w
                    sy = f.height / new_h
                else:
                    # No resize needed
                    detect_images.append(f.image)
                    sx = 1.0
                    sy = 1.0
                
                scale_factors.append((sx, sy, f.width, f.height))
            
            if self._is_coreml:
                results = []
                for img in detect_images:
                    res = model(img, conf=confidence_threshold, imgsz=self.detection_width, verbose=False)
                    if res:
                        results.append(res[0])
            else:
                results = model(detect_images, conf=confidence_threshold, imgsz=self.detection_width, 
                                half=getattr(self, "_half", False), verbose=False)
        
        detections: List[DetectionPacket] = []
        for frame, result, (sx, sy, orig_w, orig_h) in zip(frames, results, scale_factors):
            obb = getattr(result, "obb", None)
            if obb is None or obb.conf is None:
                continue
            
            # Use xyxyxyxy (coordinates relative to the input image resolution)
            polygons = obb.xyxyxyxy.cpu().numpy()
            confidences = obb.conf.cpu().numpy()
            labels = (
                obb.cls.cpu().numpy()
                if obb.cls is not None
                else [0] * len(confidences)
            )

            frame_area = orig_w * orig_h

            for polygon_array, confidence, label in zip(polygons, confidences, labels):
                confidence_float = float(confidence)
                if confidence_float < confidence_threshold:
                    continue
                
                # Apply scale factors to map back to original resolution
                scaled_polygon = []
                for pt in polygon_array:
                    scaled_polygon.append((float(pt[0]) * sx, float(pt[1]) * sy))
                
                # Area gate: ignore tiny specks or giant background blobs
                poly_area = cv2.contourArea(np.array(scaled_polygon, dtype=np.float32))
                if not (0.01 * frame_area <= poly_area <= 0.95 * frame_area):
                    continue

                if len(scaled_polygon) != 4:
                    continue

                detections.append(
                    DetectionPacket(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        width=orig_w,
                        height=orig_h,
                        corner_detection=CornerDetection(
                            corners=tuple(scaled_polygon),  # type: ignore[arg-type]
                            confidence=confidence_float,
                            metadata={
                                "runtime": self.runtime,
                                "label_index": int(label),
                                "area_fraction": float(poly_area / frame_area),
                            },
                        ),
                    )
                )
        return detections

    @property
    def confidence_threshold(self) -> float:
        return self._conf_thresh

    @property
    def runtime(self) -> str:
        return "coreml" if self._is_coreml else self._device

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

        self._device = self._resolve_device()
        model_path = _resolve_model_path(self.repo_id, self.filename)

        # TensorRT fast path (CUDA only)
        if self._device == "cuda":
            import os
            engine_path = os.path.splitext(model_path)[0] + ".engine"
            try:
                if not os.path.exists(engine_path):
                    exporter = YOLO(model_path, task="obb")
                    out = exporter.export(format="engine", half=True, dynamic=True,
                                          imgsz=self.detection_width, device=0, verbose=False,
                                          task="obb")
                    engine_path = str(out) if out else engine_path
                self._model = YOLO(engine_path, task="obb")
                print(f"[detector] backend=tensorrt engine={engine_path}", flush=True)
                self._half = True
                return self._model
            except Exception as e:
                print(f"[detector] TensorRT unavailable ({e}); falling back to .pt FP16", flush=True)

        # CoreML fast path (Apple Silicon only)
        if self._device == "mps" and platform.system() == "Darwin" and platform.machine() == "arm64":
            import os
            mlpackage_path = os.path.splitext(model_path)[0] + ".mlpackage"
            try:
                if not os.path.exists(mlpackage_path):
                    exporter = YOLO(model_path, task="obb")
                    out = exporter.export(format="coreml", half=True,
                                          imgsz=self.detection_width, verbose=False,
                                          task="obb")
                    mlpackage_path = str(out) if out else mlpackage_path
                self._model = YOLO(mlpackage_path, task="obb")
                print(f"[detector] backend=coreml engine={mlpackage_path}", flush=True)
                self._half = True
                self._is_coreml = True
                return self._model
            except Exception as e:
                print(f"[detector] CoreML unavailable ({e}); falling back to .pt MPS", flush=True)

        self._model = YOLO(model_path, task="obb")
        self._model.to(self._device)
        self._half = False
        if self._device == "cuda":
            try:
                self._model.half()
                self._half = True
            except Exception:
                pass
        print(f"[detector] backend={self._device} half={self._half}", flush=True)
        return self._model

    def _resolve_device(self) -> str:
        return probe_torch_device_status(self.device).resolved


def _resolve_model_path(repo_id: str, filename: str) -> str:
    """Download and return the local path to the YOLO model."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install '.[model]'"
        ) from exc

    return hf_hub_download(repo_id=repo_id, filename=f"weights/{filename}")


def probe_torch_device_status(requested: str = "auto") -> TorchDeviceStatus:
    """Determine the best available PyTorch device."""
    import torch
    
    mps_built = False
    mps_available = False
    try:
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()
    except AttributeError:
        pass

    reason = None
    if requested == "mps" and not mps_available:
        reason = "mps_unavailable"
    elif requested == "cuda" and not torch.cuda.is_available():
        reason = "cuda_unavailable"
    elif requested == "auto" and not (torch.cuda.is_available() or mps_available):
        reason = "mps_unavailable"  # Default reason if neither available on Mac/Linux? Test expects mps_unavailable

    cuda_avail = torch.cuda.is_available()
    if requested == "cuda":
        return TorchDeviceStatus(requested, "cuda" if cuda_avail else "cpu", cuda_avail, mps_built=mps_built, mps_available=mps_available, cuda_available=cuda_avail, reason=reason)
    if requested == "mps":
        return TorchDeviceStatus(requested, "mps" if mps_available else "cpu", mps_available, mps_built=mps_built, mps_available=mps_available, cuda_available=cuda_avail, reason=reason)
    
    if cuda_avail:
        return TorchDeviceStatus(requested, "cuda", True, mps_built=mps_built, mps_available=mps_available, cuda_available=cuda_avail, reason=reason)
    if mps_available:
        return TorchDeviceStatus(requested, "mps", True, mps_built=mps_built, mps_available=mps_available, cuda_available=cuda_avail, reason=reason)
        
    return TorchDeviceStatus(requested, "cpu", True, mps_built=mps_built, mps_available=mps_available, cuda_available=cuda_avail, reason=reason)
