from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class CardDetection:
    frame_index: int
    timestamp_ms: int
    polygon: Polygon
    confidence: float
    label: str = "trading_card"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray
    width: int
    height: int
    polygon: Polygon


@dataclass(frozen=True)
class QualityScore:
    total: float
    components: Dict[str, float]


@dataclass(frozen=True)
class ProcessingResult:
    video_id: int
    frame_count: int
    accepted_frame_count: int
    detection_count: int
    saved_instance_count: int
    output_dir: Path


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    width: int
    height: int
    triage_metrics: Dict[str, float]
    telemetry: Optional[PerformanceTelemetry] = None


@dataclass(frozen=True)
class CornerDetection:
    corners: Polygon
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionPacket:
    frame_index: int
    timestamp_ms: int
    width: int
    height: int
    corner_detection: CornerDetection
    telemetry: Optional[PerformanceTelemetry] = None


@dataclass(frozen=True)
class PerformanceTelemetry:
    t_ingest: float = 0.0
    t_detect: float = 0.0
    t_refine: float = 0.0
    t_io: float = 0.0
    queue_wait: float = 0.0
