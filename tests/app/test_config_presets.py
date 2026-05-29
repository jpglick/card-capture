"""Tests for config preset persistence (B3)."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "cards.sqlite"
    app = create_app(db_path=db)
    with TestClient(app) as c:
        yield c


def test_list_presets_returns_builtins(client):
    r = client.get("/api/v1/config/presets")
    assert r.status_code == 200
    names = [p["preset_name"] for p in r.json()]
    assert "fast" in names
    assert "balanced" in names
    assert "quality" in names


def test_create_and_list_user_preset(client):
    payload = {
        "preset_name": "my_preset",
        "description": "Custom test preset",
        "config": {"corner_confidence": 0.55},
    }
    r = client.post("/api/v1/config/presets", json=payload)
    assert r.status_code == 201
    assert r.json()["preset_name"] == "my_preset"

    # Should appear in GET list
    r2 = client.get("/api/v1/config/presets")
    names = [p["preset_name"] for p in r2.json()]
    assert "my_preset" in names


def test_create_preset_conflicts_with_builtin(client):
    payload = {
        "preset_name": "fast",
        "description": "Try to overwrite builtin",
        "config": {},
    }
    r = client.post("/api/v1/config/presets", json=payload)
    assert r.status_code == 409


def test_create_duplicate_preset_is_conflict(client):
    payload = {
        "preset_name": "unique_preset",
        "description": "First",
        "config": {"corner_confidence": 0.5},
    }
    client.post("/api/v1/config/presets", json=payload)
    r2 = client.post("/api/v1/config/presets", json=payload)
    assert r2.status_code == 409
