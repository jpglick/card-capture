"""Typed artifact contracts for the Metaflow pipeline steps.

Each Pydantic model defines the public fields that one Metaflow step
produces and the next step consumes.  Where the existing pipeline uses
``np.ndarray`` directly, the contract uses ``image_path: str``
(artifact-friendly) so that Metaflow can serialise/deserialise without
shipping raw pixel data through its metadata store.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class FrameSample(BaseModel):
    """One sampled frame written to disk by the producer subprocess."""

    frame_index: int
    timestamp_ms: int
    image_path: str
    w: int
    h: int


class TriagedFrame(BaseModel):
    """A frame that passed the triage filter (non-empty, not excessively blurry)."""

    frame_index: int
    timestamp_ms: int
    image_path: str
    triage_metrics: Dict[str, float] = Field(default_factory=dict)


class CornerDetection(BaseModel):
    """Four OBB corner points detected by YOLOv8-OBB, plus confidence."""

    frame_index: int
    timestamp_ms: int
    corners: List[Tuple[float, float]]  # 4 (x, y) pairs
    confidence: float
    source_frame_path: str
    triage_metrics: Dict[str, float] = Field(default_factory=dict)
    width: int = 0
    height: int = 0
    detection_id: int = 0  # positional index within the run


class NoveltyFilteredCandidate(BaseModel):
    """A CornerDetection that passed (or was not gated by) the background novelty check."""

    detection_id: int
    frame_index: int
    timestamp_ms: int
    corners: List[Tuple[float, float]]
    confidence: float
    source_frame_path: str
    triage_metrics: Dict[str, float] = Field(default_factory=dict)
    novelty_score: float = 1.0
    width: int = 0
    height: int = 0


class Track(BaseModel):
    """A finalized tracker track after session-aware tracking."""

    instance_id: str
    track_id: int
    angle: str
    session_id: int
    candidate_detection_ids: List[int]
    first_frame_index: int = -1


class RectifiedCrop(BaseModel):
    """A 750×1050 perspective-rectified card crop saved to disk."""

    detection_id: int
    instance_id: str
    image_path: str
    corners: List[Tuple[float, float]]
    frame_index: int


class ScoredCandidate(BaseModel):
    """A rectified crop with quality scores attached."""

    detection_id: int
    instance_id: str
    image_path: str
    quality_score: float
    quality_components: Dict[str, float] = Field(default_factory=dict)
    is_canonical: bool = False
    glare_x: Optional[float] = None
    glare_y: Optional[float] = None
    sharpness: float = 0.0
    visual_hash: str = ""
    frame_index: int = 0


class PreparedTrack(BaseModel):
    """All data needed to fuse one track into a canonical card image."""

    instance_id: str
    session_id: int
    angle: str
    first_frame_index: int
    # Paths to rectified candidate images (canonical entries)
    canonical_image_paths: List[str] = Field(default_factory=list)
    canonical_detection_ids: List[int] = Field(default_factory=list)
    best_canonical_detection_id: int = 0
    best_canonical_image_path: str = ""
    # pHash strings
    primary_hash: str = ""
    candidate_hashes: List[str] = Field(default_factory=list)
    # Scalar metrics
    side_score: float = 0.0
    mean_quality_score: float = 0.0
    # Serialised appearance vector (list of floats, not np.ndarray)
    appearance_vector: List[float] = Field(default_factory=list)
    # Glare centroids per canonical entry (for quadrant selection in fuse)
    glare_centroids: List[Tuple[Optional[float], Optional[float]]] = Field(default_factory=list)
    sharpness_scores: List[float] = Field(default_factory=list)
    duplicate_track_index: Optional[int] = None


class FusedCanonical(BaseModel):
    """The final fused card image for one PreparedTrack."""

    instance_id: str
    session_id: int
    angle: str
    fused_image_path: str
    primary_hash: str
    quality_score: float
    side_score: float
    appearance_vector: List[float] = Field(default_factory=list)
    candidate_hashes: List[str] = Field(default_factory=list)
    best_canonical_detection_id: int = 0
    duplicate_track_index: Optional[int] = None
    first_frame_index: int = -1


class DedupGroup(BaseModel):
    """One deduplication group: a canonical instance plus its duplicates."""

    canonical_instance_id: int
    duplicate_instance_ids: List[int] = Field(default_factory=list)
    hamming_distance: Optional[float] = None


class FinalCard(BaseModel):
    """A saved card record returned by the store step."""

    instance_id: int
    view_ids: List[int] = Field(default_factory=list)
    canonical_image_path: str = ""
    quality_score: float = 0.0
    angle: str = ""
