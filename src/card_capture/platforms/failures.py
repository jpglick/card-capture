"""Stable failure categories for provider runs."""
from __future__ import annotations

import dataclasses
import re
from typing import FrozenSet


PROVIDER_FAILURE_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "preflight_failed",
        "submission_failed",
        "execution_failed",
        "result_invalid",
        "cancelled",
        "unknown",
    }
)


@dataclasses.dataclass(frozen=True)
class ProviderFailure:
    provider: str
    category: str
    raw: str

    def __post_init__(self) -> None:
        if self.category not in PROVIDER_FAILURE_CATEGORIES:
            raise ValueError(
                f"unknown category {self.category!r}; "
                f"must be one of {sorted(PROVIDER_FAILURE_CATEGORIES)}"
            )


_RUNPOD_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"endpoint\s+not\s+found", re.I), "preflight_failed"),
    (re.compile(r"unauthorized|invalid\s+api\s+key", re.I), "preflight_failed"),
    (re.compile(r"timeout\s+during\s+submit", re.I), "submission_failed"),
    (re.compile(r"job\s+failed", re.I), "execution_failed"),
    (re.compile(r"out\s+of\s+memory|oom", re.I), "execution_failed"),
    (re.compile(r"cancell?ed", re.I), "cancelled"),
)

_BEAM_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"deployment\s+not\s+found", re.I), "preflight_failed"),
    (re.compile(r"app\s+id\s+missing|missing\s+credentials", re.I), "preflight_failed"),
    (re.compile(r"timeout", re.I), "submission_failed"),
    (re.compile(r"task\s+failed|error\s+in\s+task", re.I), "execution_failed"),
    (re.compile(r"cancell?ed", re.I), "cancelled"),
)


def map_provider_failure(*, provider: str, raw: str) -> ProviderFailure:
    patterns = {
        "runpod": _RUNPOD_PATTERNS,
        "beam": _BEAM_PATTERNS,
        "local": (),
    }.get(provider, ())
    for pattern, category in patterns:
        if pattern.search(raw):
            return ProviderFailure(provider=provider, category=category, raw=raw)
    return ProviderFailure(provider=provider, category="unknown", raw=raw)
