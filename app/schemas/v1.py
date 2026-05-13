"""Pydantic v2 models for the Card Capture v1 API (Contract 2).

Every model has a ``model_config`` with a ``json_schema_extra["example"]``
block so that the OpenAPI docs are self-documenting and tests can validate
round-trip serialisation with a single call.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

class Video(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "practice_session_03",
                "filename": "practice_session_03.mov",
                "duration_ms": 312000,
                "status": "completed",
                "created_at": "2026-05-12T14:00:00Z",
            }
        }
    )

    video_id: str
    filename: str
    duration_ms: int
    status: str  # pending | processing | completed | failed
    created_at: str


class VideoCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "practice_session_03.mov",
                "file_path": "/data/videos/practice_session_03.mov",
            }
        }
    )

    filename: str
    file_path: Optional[str] = None


class VideoList(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"items": []}}
    )

    items: list[Video]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "video_id": "practice_session_03",
                "status": "completed",
                "cards_extracted": 12,
                "elapsed_ms": 87430,
                "created_at": "2026-05-12T14:00:00Z",
            }
        }
    )

    run_id: str
    video_id: str
    status: str
    cards_extracted: int = 0
    elapsed_ms: int = 0
    created_at: str


class Run(RunSummary):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "video_id": "practice_session_03",
                "status": "completed",
                "cards_extracted": 12,
                "elapsed_ms": 87430,
                "created_at": "2026-05-12T14:00:00Z",
            }
        }
    )


class RunDetail(RunSummary):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "video_id": "practice_session_03",
                "status": "completed",
                "config_json": {"detector": "docaligner", "config_preset": "balanced"},
                "cards_extracted": 12,
                "elapsed_ms": 87430,
                "created_at": "2026-05-12T14:00:00Z",
                "stage_timings": [
                    {"stage_id": "detect", "elapsed_ms": 12000},
                    {"stage_id": "novelty", "elapsed_ms": 300},
                ],
            }
        }
    )

    config_json: Optional[dict[str, Any]] = None
    stage_timings: list[dict[str, Any]] = []


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config_preset": "balanced",
                "detector": "docaligner",
                "output_dir": "/tmp/out",
            }
        }
    )

    config_preset: str = "balanced"
    detector: str = "docaligner"
    output_dir: Optional[str] = None


class RunEvent(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event": "stage_started",
                "data": {"stage_id": "detect", "run_id": "CardCaptureFlow/1715523601234567"},
                "ts": "2026-05-12T14:00:05Z",
            }
        }
    )

    event: str
    data: dict[str, Any]
    ts: str


class RunTelemetry(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "total_frames_sampled": 1420,
                "total_detections": 384,
                "stages": [
                    {
                        "stage_id": "detect",
                        "frames_in": 1420,
                        "frames_out": 384,
                        "elapsed_ms": 12000,
                        "throughput_fps": 32.1,
                    }
                ],
            }
        }
    )

    run_id: str
    total_frames_sampled: int = 0
    total_detections: int = 0
    stages: list[dict[str, Any]] = []


class RunRejection(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frame_index": 4201,
                "stage_id": "novelty",
                "reason": "novelty_below_threshold",
                "thumbnail_url": "/static/runs/.../rejections/frame_4201_thumb.jpg",
            }
        }
    )

    frame_index: int
    stage_id: str
    reason: str
    thumbnail_url: Optional[str] = None


class RunHardCase(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "case_id": 7,
                "frame_index": 8830,
                "stage_id": "score",
                "reason": "low_confidence",
                "thumbnail_url": "/static/runs/.../hard_cases/case_7_thumb.jpg",
            }
        }
    )

    case_id: int
    frame_index: int
    stage_id: str
    reason: str
    thumbnail_url: Optional[str] = None


class RunCardSummary(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "card_id": "a1b2c3d4",
                "run_id": "CardCaptureFlow/1715523601234567",
                "side": "front",
                "confidence": 0.91,
            }
        }
    )

    card_id: str
    run_id: str
    side: str
    confidence: float


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

class QualityScoreDetail(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sharpness": 0.82,
                "glare": 0.76,
                "aspect_ratio": 0.98,
                "size": 0.90,
                "complexity": 0.65,
                "border_purity": 0.88,
                "confidence": 0.91,
                "total": 0.84,
            }
        }
    )

    sharpness: float = 0.0
    glare: float = 0.0
    aspect_ratio: float = 0.0
    size: float = 0.0
    complexity: float = 0.0
    border_purity: float = 0.0
    confidence: float = 0.0
    total: float = 0.0


class Card(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "card_id": "a1b2c3d4-...",
                "instance_id": "e5f6g7h8-...",
                "video_id": "practice_session_03",
                "run_id": "CardCaptureFlow/1715523601234567",
                "side": "front",
                "is_foil": False,
                "confidence": 0.91,
                "review_state": "pending",
                "canonical_url": "/static/crops/a1b2c3d4_canonical.jpg",
                "fused_url": "/static/crops/a1b2c3d4_fused.jpg",
                "created_at": "2026-05-12T14:01:10Z",
            }
        }
    )

    card_id: str
    instance_id: str
    video_id: str
    run_id: str
    side: str
    is_foil: bool = False
    confidence: float
    review_state: str = "pending"
    canonical_url: Optional[str] = None
    fused_url: Optional[str] = None
    created_at: str


class CardDetail(Card):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "card_id": "a1b2c3d4-...",
                "instance_id": "e5f6g7h8-...",
                "video_id": "practice_session_03",
                "run_id": "CardCaptureFlow/1715523601234567",
                "side": "front",
                "is_foil": False,
                "confidence": 0.91,
                "review_state": "pending",
                "canonical_url": "/static/crops/a1b2c3d4_canonical.jpg",
                "fused_url": "/static/crops/a1b2c3d4_fused.jpg",
                "quality_score": {
                    "sharpness": 0.82,
                    "glare": 0.76,
                    "aspect_ratio": 0.98,
                    "size": 0.90,
                    "complexity": 0.65,
                    "border_purity": 0.88,
                    "confidence": 0.91,
                    "total": 0.84,
                },
                "source_frame_indices": [4201, 4235, 4268],
                "dedup_group_id": 3,
                "created_at": "2026-05-12T14:01:10Z",
            }
        }
    )

    quality_score: Optional[QualityScoreDetail] = None
    source_frame_indices: list[int] = []
    dedup_group_id: Optional[int] = None


class CardFilter(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": None,
                "video_id": None,
                "review_state": "pending",
                "side": "front",
                "page": 1,
                "page_size": 50,
            }
        }
    )

    run_id: Optional[str] = None
    video_id: Optional[str] = None
    dedup_group_id: Optional[int] = None
    review_state: Optional[str] = None
    side: Optional[str] = None
    is_foil: Optional[bool] = None
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    page: int = 1
    page_size: int = 50


class CardBulkAction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "card_ids": ["a1b2c3d4-...", "b2c3d4e5-..."],
                "review_state": "rejected",
            }
        }
    )

    card_ids: list[str]
    review_state: str


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

class LabelTruthExpectedCard(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "card_id": "card_001",
                "front_present": True,
                "back_present": True,
            }
        }
    )

    card_id: str
    front_present: bool = True
    back_present: bool = False


class LabelTruth(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "practice_session_03",
                "schema_version": 1,
                "expected_cards": [
                    {"card_id": "card_001", "front_present": True, "back_present": True}
                ],
            }
        }
    )

    video_id: str
    schema_version: int = 1
    expected_cards: list[LabelTruthExpectedCard] = []


class LabelFBNext(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instance_id": "e5f6g7h8-...",
                "frame_index": 4201,
                "canonical_url": "/static/crops/e5f6g7h8_canonical.jpg",
                "video_id": "practice_session_03",
                "run_id": "CardCaptureFlow/1715523601234567",
                "labels_collected": 247,
                "labels_target": 500,
            }
        }
    )

    instance_id: str
    frame_index: int
    canonical_url: Optional[str] = None
    video_id: str
    run_id: str
    labels_collected: int
    labels_target: int


class LabelFB(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instance_id": "e5f6g7h8-...",
                "frame_index": 4201,
                "side": "front",
            }
        }
    )

    instance_id: str
    frame_index: int
    side: str  # front | back | uncertain


class LabelFBResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label_id": 248,
                "instance_id": "e5f6g7h8-...",
                "side": "front",
                "created_at": "2026-05-12T15:00:01Z",
            }
        }
    )

    label_id: int
    instance_id: str
    side: str
    created_at: str


class DedupCluster(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cluster_id": 5,
                "status": "unverified",
                "predicted_member_ids": ["e5f6g7h8-...", "a1b2c3d4-..."],
                "confirmed_member_ids": None,
                "member_thumbnails": ["/static/crops/e5f6_thumb.jpg"],
                "updated_at": "2026-05-12T14:01:10Z",
            }
        }
    )

    cluster_id: int
    status: str
    predicted_member_ids: list[str] = []
    confirmed_member_ids: Optional[list[str]] = None
    member_thumbnails: list[str] = []
    updated_at: str


class DedupClusterUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "confirmed",
                "confirmed_member_ids": ["e5f6g7h8-...", "a1b2c3d4-..."],
            }
        }
    )

    status: str
    confirmed_member_ids: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainingDataset(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "fb_classifier",
                "total_labels": 247,
                "class_distribution": {"front": 130, "back": 112, "uncertain": 5},
                "last_updated": "2026-05-12T15:00:01Z",
            }
        }
    )

    model_name: str
    total_labels: int = 0
    class_distribution: dict[str, int] = {}
    last_updated: Optional[str] = None


class TrainingJob(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "retrain-fb_classifier-1715527201",
                "model_name": "fb_classifier",
                "status": "queued",
                "progress": None,
                "created_at": "2026-05-12T15:00:01Z",
                "completed_at": None,
            }
        }
    )

    job_id: str
    model_name: str
    status: str
    progress: Optional[dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None


class TrainingRetrainRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "epochs": 20,
                "learning_rate": 0.001,
            }
        }
    )

    epochs: int = 20
    learning_rate: float = 0.001


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class RegressionBaseline(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "baseline_id": 1,
                "name": "baseline_v4.1",
                "code_sha": "e34f40d",
                "created_at": "2026-05-12T14:00:00Z",
            }
        }
    )

    baseline_id: int
    name: str
    code_sha: str
    created_at: str


class RegressionBaselineCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "baseline_v4.2",
                "code_sha": "abc1234",
            }
        }
    )

    name: str
    code_sha: str


class RegressionMetric(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "card_recall": 0.92,
                "card_precision": 0.88,
                "side_accuracy": 0.79,
                "dedup_ari": 0.81,
                "image_quality_ssim": 0.87,
            }
        }
    )

    card_recall: float = 0.0
    card_precision: float = 0.0
    side_accuracy: float = 0.0
    dedup_ari: float = 0.0
    image_quality_ssim: float = 0.0


class RegressionPerVideoDelta(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "practice_session_03",
                "card_recall": 0.91,
                "card_precision": 0.90,
                "side_accuracy": 0.80,
                "regressions": [],
            }
        }
    )

    video_id: str
    card_recall: float = 0.0
    card_precision: float = 0.0
    side_accuracy: float = 0.0
    regressions: list[str] = []


class RegressionRun(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": 7,
                "baseline_id": 1,
                "status": "completed",
                "metrics": {
                    "card_recall": 0.92,
                    "card_precision": 0.88,
                    "side_accuracy": 0.79,
                    "dedup_ari": 0.81,
                    "image_quality_ssim": 0.87,
                },
                "per_video": [],
                "created_at": "2026-05-12T16:00:00Z",
            }
        }
    )

    run_id: int
    baseline_id: int
    status: str
    metrics: Optional[RegressionMetric] = None
    per_video: list[RegressionPerVideoDelta] = []
    created_at: str


class RegressionRunRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "baseline_id": 1,
                "video_subset": ["practice_session_03"],
            }
        }
    )

    baseline_id: int
    video_subset: Optional[list[str]] = None


class RegressionCompare(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_a": 6,
                "run_b": 7,
                "metric_deltas": {
                    "card_recall": 0.02,
                    "card_precision": -0.01,
                    "side_accuracy": 0.05,
                    "dedup_ari": 0.03,
                    "image_quality_ssim": 0.00,
                },
                "regressions": [],
                "per_video_deltas": [],
            }
        }
    )

    run_a: int
    run_b: int
    metric_deltas: dict[str, float] = {}
    regressions: list[str] = []
    per_video_deltas: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigPreset(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "preset_name": "balanced",
                "description": "Default balanced trade-off between speed and quality",
                "config": {
                    "corner_confidence": 0.5,
                    "background_novelty_threshold": 0.08,
                    "centroid_jump_ratio": 0.30,
                    "valley_drop_ratio": 0.40,
                    "foil_threshold": 50.0,
                },
            }
        }
    )

    preset_name: str
    description: str = ""
    config: dict[str, Any] = {}


class ConfigPlayground(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "available_steps": ["novelty", "score", "dedup"],
                "slider_params": [
                    {
                        "param": "background_novelty_threshold",
                        "current": 0.08,
                        "min": 0.02,
                        "max": 0.30,
                        "step": 0.01,
                        "affects_steps": ["novelty", "score"],
                    }
                ],
            }
        }
    )

    run_id: str
    available_steps: list[str] = []
    slider_params: list[dict[str, Any]] = []


class ConfigPlaygroundSliderUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "param": "background_novelty_threshold",
                "value": 0.10,
            }
        }
    )

    param: str
    value: float


# ---------------------------------------------------------------------------
# SSE events
# ---------------------------------------------------------------------------

class SSEEvent(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event": "stage_started",
                "data": {"stage_id": "detect", "run_id": "CardCaptureFlow/1715523601234567"},
                "ts": "2026-05-12T14:00:05Z",
            }
        }
    )

    event: str
    data: dict[str, Any]
    ts: Optional[str] = None


class SSEStageStarted(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "stage_id": "detect",
                "ts": "2026-05-12T14:00:05Z",
            }
        }
    )

    run_id: str
    stage_id: str
    ts: str


class SSEStageProgress(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "stage_id": "detect",
                "pct": 42,
                "detail": "Sampling frame 600/1420",
                "ts": "2026-05-12T14:00:11Z",
            }
        }
    )

    run_id: str
    stage_id: str
    pct: int
    detail: str = ""
    ts: str


class SSEStageCompleted(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "stage_id": "detect",
                "elapsed_ms": 12100,
                "ts": "2026-05-12T14:00:17Z",
            }
        }
    )

    run_id: str
    stage_id: str
    elapsed_ms: int
    ts: str


class SSEArtifactPersisted(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "stage_id": "detect",
                "artifact_name": "corner_detections",
                "artifact_ref": "CardCaptureFlow/1715523601234567/detect/corner_detections",
                "ts": "2026-05-12T14:00:17Z",
            }
        }
    )

    run_id: str
    stage_id: str
    artifact_name: str
    artifact_ref: str
    ts: str


class SSERunCompleted(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "cards_extracted": 12,
                "elapsed_ms": 87430,
                "ts": "2026-05-12T14:01:27Z",
            }
        }
    )

    run_id: str
    cards_extracted: int
    elapsed_ms: int
    ts: str


class SSERunFailed(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "CardCaptureFlow/1715523601234567",
                "stage_id": "refine",
                "error": "GPU out of memory",
                "ts": "2026-05-12T14:00:55Z",
            }
        }
    )

    run_id: str
    stage_id: Optional[str] = None
    error: str
    ts: str

# ---------------------------------------------------------------------------
# Labeling (Contract 4 based)
# ---------------------------------------------------------------------------

class LabelTruthExpectedCard(BaseModel):
    model_config = ConfigDict(frozen=True)
    card_id: str
    front_present: bool = True
    back_present: bool = False
    physical_card_key: str
    is_foil: bool = False
    approx_front_window_ms: Optional[tuple[int, int]] = None
    approx_back_window_ms: Optional[tuple[int, int]] = None
    notes: Optional[str] = None


class LabelTruth(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    schema_version: int = 1
    expected_cards: list[LabelTruthExpectedCard] = []


class LabelFBNext(BaseModel):
    instance_id: str
    frame_index: int
    canonical_url: Optional[str] = None
    video_id: str
    run_id: str
    labels_collected: int
    labels_target: int


class LabelFB(BaseModel):
    instance_id: str
    frame_index: int
    side: str  # front | back | uncertain


class LabelFBResult(BaseModel):
    label_id: int
    instance_id: str
    side: str
    created_at: str


class DedupCluster(BaseModel):
    cluster_id: int
    status: str
    predicted_member_ids: list[str] = Field(default_factory=list)
    confirmed_member_ids: Optional[list[str]] = None
    member_thumbnails: list[str] = Field(default_factory=list)
    updated_at: str


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class DatasetSummary(BaseModel):
    model_name: str
    total_labels: int
    class_distribution: dict[str, int]
    last_updated: str


class RetrainRequest(BaseModel):
    epochs: int = 20
    learning_rate: float = 0.001


class TrainingJobSummary(BaseModel):
    job_id: str
    model_name: str
    status: str
    created_at: str


class TrainingJobDetail(BaseModel):
    job_id: str
    model_name: str
    status: str
    progress: Optional[dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class Baseline(BaseModel):
    baseline_id: int
    name: str
    code_sha: str
    created_at: str


class RegressionRun(BaseModel):
    run_id: int
    baseline_id: int
    status: str
    created_at: str


class RegressionRunDetail(BaseModel):
    run_id: int
    baseline_id: int
    status: str
    metrics: dict[str, Any]
    per_video: list[dict[str, Any]]
    created_at: str


class RegressionCompare(BaseModel):
    run_a: int
    run_b: int
    metric_deltas: dict[str, Any]
    regressions: list[Any] = Field(default_factory=list)
    per_video_deltas: list[dict[str, Any]] = Field(default_factory=list)
