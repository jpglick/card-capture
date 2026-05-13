"""Config routes — `/api/v1/config`.

Stubs only for playground; preset listing returns the three built-in presets.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
    raise HTTPException(status_code=501, detail="not implemented yet")


@router.get("/playground/{run_id}", response_model=ConfigPlayground)
def get_playground(run_id: str):
    raise HTTPException(status_code=501, detail="not implemented yet")
