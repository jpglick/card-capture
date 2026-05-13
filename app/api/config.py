"""Config routes — `/api/v1/config`.

Stubs only for playground; preset listing returns the three built-in presets.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.v1 import ConfigPlayground, ConfigPreset

router = APIRouter()

_BUILTIN_PRESETS: list[ConfigPreset] = [
    ConfigPreset(
        preset_name="fast",
        description="Lower quality thresholds optimised for throughput",
        config={
            "corner_confidence": 0.40,
            "background_novelty_threshold": 0.06,
            "centroid_jump_ratio": 0.35,
            "valley_drop_ratio": 0.35,
            "foil_threshold": 50.0,
        },
    ),
    ConfigPreset(
        preset_name="balanced",
        description="Default balanced trade-off between speed and quality",
        config={
            "corner_confidence": 0.50,
            "background_novelty_threshold": 0.08,
            "centroid_jump_ratio": 0.30,
            "valley_drop_ratio": 0.40,
            "foil_threshold": 50.0,
        },
    ),
    ConfigPreset(
        preset_name="quality",
        description="Higher quality thresholds at the cost of throughput",
        config={
            "corner_confidence": 0.60,
            "background_novelty_threshold": 0.10,
            "centroid_jump_ratio": 0.25,
            "valley_drop_ratio": 0.45,
            "foil_threshold": 50.0,
        },
    ),
]


@router.get("/presets", response_model=list[ConfigPreset])
def list_presets():
    """Return all available config presets (including user-defined ones)."""
    return _BUILTIN_PRESETS


@router.post("/presets", response_model=ConfigPreset, status_code=201)
def create_preset(payload: ConfigPreset):
    # TODO: implement user presets in DB
    return payload


@router.get("/playground/{run_id}", response_model=ConfigPlayground)
def get_playground(run_id: str, request: Request):
    """Load initial playground data for a run."""
    svc = request.app.state.playground_service
    artifacts = svc.get_run_artifacts(run_id)
    ctx = artifacts["run_context"]
    
    # Extract interesting thresholds
    config = {
        "corner_confidence_threshold": ctx.corner_confidence_threshold,
        "background_novelty_threshold": ctx.background_novelty_threshold,
        "centroid_jump_ratio": ctx.centroid_jump_ratio,
        "min_track_length": ctx.min_track_length,
    }
    
    return ConfigPlayground(run_id=run_id, config=config)


@router.post("/playground/{run_id}/recompute")
def recompute_playground(run_id: str, body: dict, request: Request):
    """Recompute metrics based on new thresholds."""
    svc = request.app.state.playground_service
    return svc.recompute(run_id, body)
