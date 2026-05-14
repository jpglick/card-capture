"""Unified return type for all harness metrics."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class MetricResult(BaseModel):
    """Unified return type for harness metric functions.

    Simple metrics set ``value`` and leave ``breakdown`` empty.
    Compound metrics (dedup_accuracy, image_quality) set ``value=None``
    and populate ``breakdown`` with named sub-scores.
    """

    value: Optional[float] = None
    breakdown: dict[str, Optional[float]] = {}

    model_config = ConfigDict(frozen=True)
