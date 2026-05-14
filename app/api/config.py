"""Config routes — `/api/v1/config`.

Built-in presets are served from a static list. User-defined presets are
persisted to the ``config_presets`` table in cards.sqlite and unioned with
the built-ins on GET.
"""
from __future__ import annotations

import json
import sqlite3

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

# Built-in names are reserved and cannot be overwritten by user presets.
_BUILTIN_NAMES = {p.preset_name for p in _BUILTIN_PRESETS}


def _get_user_presets(db_path) -> list[ConfigPreset]:
    """Load user-defined presets from the database."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT preset_name, description, config_json FROM config_presets ORDER BY created_at"
            ).fetchall()
        return [
            ConfigPreset(
                preset_name=r["preset_name"],
                description=r["description"],
                config=json.loads(r["config_json"]),
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        # Table may not exist yet if migration hasn't run.
        return []


@router.get("/presets", response_model=list[ConfigPreset])
def list_presets(request: Request):
    """Return all available config presets (built-in + user-defined)."""
    user = _get_user_presets(request.app.state.db_path)
    return _BUILTIN_PRESETS + user


@router.post("/presets", response_model=ConfigPreset, status_code=201)
def create_preset(payload: ConfigPreset, request: Request):
    """Create a new user-defined config preset."""
    if payload.preset_name in _BUILTIN_NAMES:
        raise HTTPException(
            status_code=409,
            detail=f"Preset name '{payload.preset_name}' is reserved for built-in presets.",
        )
    db_path = request.app.state.db_path
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO config_presets (preset_name, description, config_json) VALUES (?, ?, ?)",
                (payload.preset_name, payload.description, json.dumps(payload.config)),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail=f"Preset '{payload.preset_name}' already exists.",
        )
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
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
