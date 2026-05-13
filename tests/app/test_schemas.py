"""Verify that all Pydantic v2 schema classes are importable and self-consistent.

Each class must declare a ``model_config["json_schema_extra"]["example"]`` dict
that can be round-tripped through ``model_validate``.
"""
from __future__ import annotations

import pytest

from app.schemas.v1 import (
    Card,
    CardBulkAction,
    CardDetail,
    CardFilter,
    ConfigPlayground,
    ConfigPreset,
    DedupCluster,
    LabelFB,
    LabelFBNext,
    LabelFBResult,
    LabelTruth,
    LabelTruthExpectedCard,
    QualityScoreDetail,
    RegressionBaseline,
    RegressionBaselineCreate,
    RegressionCompare,
    RegressionMetric,
    RegressionPerVideoDelta,
    RegressionRun,
    RegressionRunRequest,
    Run,
    RunCardSummary,
    RunDetail,
    RunEvent,
    RunHardCase,
    RunRejection,
    RunSummary,
    RunTelemetry,
    SSEArtifactPersisted,
    SSEEvent,
    SSERunCompleted,
    SSERunFailed,
    SSEStageCompleted,
    SSEStageProgress,
    SSEStageStarted,
    TrainingDataset,
    TrainingJob,
    TrainingRetrainRequest,
    Video,
    VideoCreate,
)

_SCHEMAS_WITH_EXAMPLES = [
    Video,
    VideoCreate,
    Run,
    RunDetail,
    RunSummary,
    RunCardSummary,
    RunEvent,
    RunTelemetry,
    RunRejection,
    RunHardCase,
    Card,
    CardDetail,
    CardFilter,
    CardBulkAction,
    LabelTruth,
    LabelTruthExpectedCard,
    LabelFB,
    LabelFBNext,
    LabelFBResult,
    DedupCluster,
    TrainingDataset,
    TrainingJob,
    TrainingRetrainRequest,
    RegressionBaseline,
    RegressionBaselineCreate,
    RegressionMetric,
    RegressionPerVideoDelta,
    RegressionRun,
    RegressionRunRequest,
    RegressionCompare,
    ConfigPreset,
    ConfigPlayground,
    SSEEvent,
    SSEStageStarted,
    SSEStageProgress,
    SSEStageCompleted,
    SSEArtifactPersisted,
    SSERunCompleted,
    SSERunFailed,
    QualityScoreDetail,
]


def test_all_v1_schemas_importable_and_have_examples():
    """Each schema must declare a worked example in model_config."""
    missing = []
    for cls in _SCHEMAS_WITH_EXAMPLES:
        example = cls.model_config.get("json_schema_extra", {}).get("example")
        if example is None:
            missing.append(cls.__name__)
    assert not missing, f"schemas missing json_schema_extra example: {missing}"


@pytest.mark.parametrize("cls", _SCHEMAS_WITH_EXAMPLES, ids=lambda c: c.__name__)
def test_schema_example_validates(cls):
    """The worked example in each schema must round-trip through model_validate."""
    example = cls.model_config["json_schema_extra"]["example"]
    instance = cls.model_validate(example)
    assert instance is not None
