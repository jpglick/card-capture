"""Serializable contracts passed between runtime, runner, app, and harness.

Values passed across this boundary must remain JSON-serializable. They must
not include tensors, model objects, open video handles, or process-local
resources.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Literal, Mapping


RuntimeMode = Literal["strict_gpu", "cpu_debug", "mixed_compat"]


@dataclasses.dataclass(frozen=True)
class PipelineRunRequest:
    run_id: str
    input_video: str            # artifact:// reference
    output_root: str            # artifact:// reference
    runtime_mode: RuntimeMode
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    # Optional callers (UI, training) pass these explicitly so the
    # runtime doesn't have to infer SQLite location from output_root or guess
    # a video_id for FK constraints. All fields stay JSON-serializable.
    db_path: str | None = None
    video_id: int | None = None
    config_preset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["config"] = dict(self.config)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineRunRequest":
        return cls(
            run_id=data["run_id"],
            input_video=data["input_video"],
            output_root=data["output_root"],
            runtime_mode=data["runtime_mode"],
            config=dict(data.get("config", {})),
            db_path=data.get("db_path"),
            video_id=data.get("video_id"),
            config_preset=data.get("config_preset"),
        )


@dataclasses.dataclass(frozen=True)
class StageTiming:
    stage: str
    elapsed_ms: float
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ContractViolation:
    code: str                   # stable, machine-readable category
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class CardRecord:
    """Provider-neutral card output. Refined in Phase 3 to match storage."""
    card_instance_id: str
    front_crop: str             # artifact:// reference
    back_crop: str | None = None
    quality: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RunManifest:
    run_id: str
    runtime_mode: RuntimeMode
    input_video: str
    output_artifacts: list[str]
    cards: list[CardRecord]
    stage_timings: list[StageTiming]
    contract_violations: list[ContractViolation]
    version: str
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "RunManifest":
        data = json.loads(blob)
        return cls(
            run_id=data["run_id"],
            runtime_mode=data["runtime_mode"],
            input_video=data["input_video"],
            output_artifacts=list(data.get("output_artifacts", [])),
            cards=[CardRecord(**c) for c in data.get("cards", [])],
            stage_timings=[StageTiming(**s) for s in data.get("stage_timings", [])],
            contract_violations=[ContractViolation(**v) for v in data.get("contract_violations", [])],
            version=data["version"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclasses.dataclass(frozen=True)
class PipelineRunResult:
    manifest: RunManifest
    manifest_path: str | None = None
