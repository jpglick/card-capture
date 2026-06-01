"""Shared dataclasses for the dead-code toolchain."""
from __future__ import annotations

import dataclasses
from typing import Literal, Optional

Kind = Literal["whole_module", "symbol", "branch", "non_mps_path"]
Decision = Literal["remove", "keep", "investigate"]


@dataclasses.dataclass
class Finding:
    path: str          # path relative to repo root, e.g. "src/card_capture/foo.py"
    lineno: int
    name: str          # symbol name vulture reported
    kind: str          # vulture's category: "function", "method", "import", ...
    confidence: int    # vulture 0-100


@dataclasses.dataclass
class ManifestEntry:
    id: str
    kind: Kind
    path: str
    symbol: Optional[str]
    vulture_confidence: int
    adjusted_confidence: int
    decision: Decision
    tier: int
    reason: str


@dataclasses.dataclass
class GateResult:
    passed: bool
    stage: str         # "tests" | "video_smoke" | "metric_regression" | "all"
    detail: str
