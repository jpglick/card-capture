"""Config routes — `/api/v1/config`.

Built-in presets are served from a static list. User-defined presets are
persisted to the ``config_presets`` table in cards.sqlite and unioned with
the built-ins on GET.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

_CONFIG_PATH = Path(__file__).parent.parent.parent / "card_capture_config.json"

# Fields exposed via /config/pipeline — performance knobs only
_PIPELINE_FIELDS = {
    "inference_batch_size": int,
    "triage_keep_percentile": float,
    "queue_size": int,
    "corner_confidence": float,
    "presence_threshold": float,
}

_COMPUTE_FIELDS = {
    "pipeline_backend": str,
    # Beam
    "beam_api_key": str,
    "beam_volume_id": str,
    "beam_endpoint_id": str,
    # RunPod
    "runpod_api_key": str,
    "runpod_endpoint_id": str,
    # Cloudflare R2 (shared transfer storage for RunPod)
    "r2_account_id": str,
    "r2_bucket": str,
    "r2_access_key_id": str,
    "r2_secret_access_key": str,
}

_COMPUTE_DEFAULTS = {
    "pipeline_backend": "mps",
    "beam_api_key": "",
    "beam_volume_id": "",
    "beam_endpoint_id": "",
    "runpod_api_key": "",
    "runpod_endpoint_id": "",
    "r2_account_id": "",
    "r2_bucket": "",
    "r2_access_key_id": "",
    "r2_secret_access_key": "",
}

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


def _get_user_presets(config_repo) -> list[ConfigPreset]:
    """Load user-defined presets from the database."""
    rows = config_repo.list_presets()
    return [
        ConfigPreset(
            preset_name=r["preset_name"],
            description=r["description"],
            config=r["config"],
        )
        for r in rows
    ]


@router.get("/presets", response_model=list[ConfigPreset])
def list_presets(request: Request):
    """Return all available config presets (built-in + user-defined)."""
    user = _get_user_presets(request.app.state.config_repo)
    return _BUILTIN_PRESETS + user


@router.post("/presets", response_model=ConfigPreset, status_code=201)
def create_preset(payload: ConfigPreset, request: Request):
    """Create a new user-defined config preset."""
    if payload.preset_name in _BUILTIN_NAMES:
        raise HTTPException(
            status_code=409,
            detail=f"Preset name '{payload.preset_name}' is reserved for built-in presets.",
        )
    existing = request.app.state.config_repo.get_preset(payload.preset_name)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Preset name '{payload.preset_name}' already exists.",
        )
    try:
        request.app.state.config_repo.upsert_preset(
            name=payload.preset_name,
            description=payload.description,
            config=payload.config,
        )
        request.app.state.writer.flush()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    return payload


@router.get("/pipeline")
def get_pipeline_config(_request: Request):
    """Return the current pipeline performance config."""
    try:
        data = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: data.get(k) for k in _PIPELINE_FIELDS if k in data}


@router.patch("/pipeline")
def patch_pipeline_config(body: dict, _request: Request):
    """Update one or more pipeline performance fields in card_capture_config.json."""
    unknown = set(body) - set(_PIPELINE_FIELDS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown fields: {sorted(unknown)}")
    try:
        data = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
        for key, cast in _PIPELINE_FIELDS.items():
            if key in body:
                data[key] = cast(body[key])
        _CONFIG_PATH.write_text(json.dumps(data, indent=4))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: data.get(k) for k in _PIPELINE_FIELDS if k in data}


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


@router.get("/compute")
def get_compute_config(_request: Request):
    """Return the current compute backend config."""
    try:
        data = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: data.get(k, _COMPUTE_DEFAULTS[k]) for k in _COMPUTE_FIELDS}


@router.patch("/compute")
def patch_compute_config(body: dict, _request: Request):
    """Update compute backend fields in card_capture_config.json."""
    unknown = set(body) - set(_COMPUTE_FIELDS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown fields: {sorted(unknown)}")
    try:
        data = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
        for key, cast in _COMPUTE_FIELDS.items():
            if key in body:
                data[key] = cast(body[key])
        _CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: data.get(k) for k in _COMPUTE_FIELDS}
