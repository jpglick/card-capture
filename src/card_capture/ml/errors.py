"""ML error types."""
from __future__ import annotations


class UntrainedModelError(RuntimeError):
    """Raised when a model is asked to predict without a trained checkpoint."""
