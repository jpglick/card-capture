"""Pydantic models for the Card Capture v1 API (Contract 2).
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


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
