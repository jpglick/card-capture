"""failures.py exposes stable categories and a mapping helper."""
from __future__ import annotations

from card_capture.platforms.failures import (
    PROVIDER_FAILURE_CATEGORIES,
    ProviderFailure,
    map_provider_failure,
)


def test_categories_are_stable_strings():
    for c in (
        "preflight_failed",
        "submission_failed",
        "execution_failed",
        "result_invalid",
        "cancelled",
        "unknown",
    ):
        assert c in PROVIDER_FAILURE_CATEGORIES


def test_map_unknown_provider_returns_unknown_category():
    failure = map_provider_failure(provider="runpod", raw="<garbled blob>")
    assert failure.category == "unknown"
    assert failure.provider == "runpod"
    assert failure.raw == "<garbled blob>"


def test_map_well_known_runpod_phrases():
    assert map_provider_failure(provider="runpod", raw="endpoint not found").category == "preflight_failed"
    assert map_provider_failure(provider="runpod", raw="JOB FAILED: out of memory").category == "execution_failed"
    assert map_provider_failure(provider="runpod", raw="cancelled by user").category == "cancelled"


def test_map_well_known_beam_phrases():
    assert map_provider_failure(provider="beam", raw="deployment not found").category == "preflight_failed"
    assert map_provider_failure(provider="beam", raw="task failed during exec").category == "execution_failed"


def test_provider_failure_validates_category():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        ProviderFailure(provider="runpod", category="bogus", raw="x")
