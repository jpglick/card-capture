"""Contract 2 conformance tests — every required route must be registered.

These tests fail before ``app/main.py`` and the router stubs exist; they
pass once the app factory wires all routers.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app

ROUTES_REQUIRED = [
    ("GET", "/api/v1/videos"),
    ("POST", "/api/v1/videos"),
    ("GET", "/api/v1/runs"),
    ("GET", "/api/v1/cards"),
    ("GET", "/api/v1/label/fb/next"),
    ("GET", "/api/v1/training/datasets"),
    ("GET", "/api/v1/regression/baselines"),
    ("GET", "/api/v1/config/presets"),
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
    for method, path in ROUTES_REQUIRED:
        assert (method, path) in registered, f"missing route: {method} {path}"


def test_openapi_includes_v1_routes():
    """The OpenAPI schema must document all Contract-2 routes."""
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for method, path in ROUTES_REQUIRED:
        assert path in paths, f"missing from OpenAPI: {path}"
        assert method.lower() in paths[path], (
            f"method {method} missing from OpenAPI path {path}"
        )


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

    # Minimal valid request bodies for POST routes (so FastAPI validates
    # before our stub handler raises 501 instead of returning 422).
    post_bodies: dict[str, dict] = {
        "/api/v1/videos": {"filename": "test.mov"},
    }

    implemented = {
        "/api/v1/config/presets",
        "/api/v1/videos",
        "/api/v1/runs",
        "/api/v1/cards",
        "/api/v1/label/fb/next",
        "/api/v1/training/datasets",
        "/api/v1/regression/baselines",
    }

    for method, path in ROUTES_REQUIRED:
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
