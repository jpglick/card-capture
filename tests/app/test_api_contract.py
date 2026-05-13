"""Contract 2 conformance tests — every required route must be registered.

These tests fail before ``app/main.py`` and the router stubs exist; they
pass once the app factory wires all routers.
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from app.main import create_app
from app.schemas.v1 import (
    Video, RunSummary, Card, DatasetSummary,
    Baseline, RegressionCompare, ConfigPreset
)

ROUTES_REQUIRED = [
    ("GET", "/api/v1/videos", list[Video]),
    ("POST", "/api/v1/videos", Video),
    ("GET", "/api/v1/runs", list[RunSummary]),
    ("GET", "/api/v1/cards", dict), # Paginated dict {"total": ..., "items": [...]}
    ("GET", "/api/v1/label/fb/next", dict), # Shape varies if empty (204)
    ("GET", "/api/v1/training/datasets", list[DatasetSummary]),
    ("GET", "/api/v1/regression/baselines", list[Baseline]),
    ("GET", "/api/v1/config/presets", list[ConfigPreset]),
]


def _registered_routes(app):
    return {
        (list(r.methods)[0], r.path)
        for r in app.routes
        if hasattr(r, "methods") and r.methods
    }


def test_all_required_routes_registered():
    """Every Contract-2 route must appear in the FastAPI route table."""
    client = TestClient(create_app())
    registered = _registered_routes(client.app)
    for method, path, _ in ROUTES_REQUIRED:
        assert (method, path) in registered, f"missing route: {method} {path}"


def test_openapi_includes_v1_routes():
    """The OpenAPI schema must document all Contract-2 routes."""
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for method, path, _ in ROUTES_REQUIRED:
        assert path in paths, f"missing from OpenAPI: {path}"
        assert method.lower() in paths[path], (
            f"method {method} missing from OpenAPI path {path}"
        )


def test_implemented_route_shapes():
    """Verify that implemented routes return the correct Contract-2 shapes."""
    client = TestClient(create_app())
    
    # POST bodies for routes that need them
    post_bodies: dict[str, dict] = {
        "/api/v1/videos": {"filename": "test.mov"},
    }

    implemented = {
        "/api/v1/config/presets",
        "/api/v1/videos",
        "/api/v1/runs",
        "/api/v1/cards",
        "/api/v1/training/datasets",
        "/api/v1/regression/baselines",
        "/api/v1/label/fb/next",
        "/api/v1/label/truth",
    }

    for method, path, schema in ROUTES_REQUIRED:
        if path not in implemented:
            continue
            
        func = getattr(client, method.lower())
        kwargs: dict = {}
        if method == "POST" and path in post_bodies:
            kwargs["json"] = post_bodies[path]
            
        response = func(path, **kwargs)
        assert response.status_code in (200, 201, 204), f"Error hitting {path}: {response.text}"
        
        if response.status_code == 204:
            return

        # Validate shape
        try:
            adapter = TypeAdapter(schema)
            adapter.validate_python(response.json())
        except Exception as exc:
            pytest.fail(f"Route {path} returned invalid shape: {exc}\nResponse: {response.json()}")


def test_config_presets_returns_builtin_presets():
    """GET /api/v1/config/presets returns the three built-in preset names."""
    client = TestClient(create_app())
    response = client.get("/api/v1/config/presets")
    assert response.status_code == 200
    names = {p["preset_name"] for p in response.json()}
    assert {"fast", "balanced", "quality"} == names


def test_stub_routes_return_501():
    """Unimplemented routes must return 501 Not Implemented."""
    client = TestClient(create_app(), raise_server_exceptions=False)

    # Minimal valid request bodies for POST routes
    post_bodies: dict[str, dict] = {
        "/api/v1/videos": {"filename": "test.mov"},
    }

    implemented = {
        "/api/v1/config/presets",
        "/api/v1/videos",
        "/api/v1/runs",
        "/api/v1/cards",
        "/api/v1/training/datasets",
        "/api/v1/regression/baselines",
        "/api/v1/label/fb/next",
        "/api/v1/label/truth",
    }

    for method, path, _ in ROUTES_REQUIRED:
        if path in implemented:
            continue
        func = getattr(client, method.lower())
        kwargs: dict = {}
        if method == "POST" and path in post_bodies:
            kwargs["json"] = post_bodies[path]
        response = func(path, **kwargs)
        assert response.status_code == 501, (
            f"expected 501 for {method} {path}, got {response.status_code}"
        )
