from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]


@dataclass
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
    """Consolidated quality metrics for a single detection."""
    total: float
    components: Dict[str, float]


@dataclass(frozen=True)
class ScoredCandidate:
    """A detection with an associated quality score and metadata."""
    detection_id: int
    timestamp_ms: int
    image_path: str
    score: QualityScore
    corners: Optional[Polygon] = None
    frame_index: Optional[int] = None


@dataclass
class TrackState:
    """Mutable state for a temporal track of card presentations."""
    instance_id: str
    candidates: List[ScoredCandidate] = field(default_factory=list)
    last_centroid: Optional[Point] = None
    last_frame_index: Optional[int] = None
    missed_frames: int = 0
    active: bool = True
    angle: str = "Front"
    reid_embedding: Optional[np.ndarray] = None
    session_id: int = 0


@dataclass(frozen=True)
class TorchDeviceStatus:
    """Device capabilities and status for PyTorch execution."""
    requested: str
    resolved: str
    is_available: bool = False
    mps_built: bool = False
    mps_available: bool = False
    cuda_built: bool = False
    cuda_available: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class ProcessingResult:
    video_id: int
    frame_count: int
    accepted_frame_count: int
    detection_count: int
    saved_instance_count: int
    output_dir: Path
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
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
