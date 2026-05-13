"""Integration tests for /api/v1/training/* endpoints."""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from migrations.run_migrations import apply_migrations


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """TestClient with a fresh migrated database."""
    db = tmp_path / "test.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    return TestClient(create_app(db_path=db))


def test_list_datasets(client: TestClient):
    r = client.get("/api/v1/training/datasets")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    names = {d["name"] for d in data}
    assert "fb_labels" in names
    assert "dedup_clusters" in names


def test_retrain_returns_job_id(client: TestClient):
    r = client.post("/api/v1/training/retrain/fb_classifier", json={"dry_run": True})
    assert r.status_code in (200, 202)
    body = r.json()
    assert "job_id" in body
    assert body["status"] in ("queued", "running", "completed")


def test_get_job(client: TestClient):
    j = client.post(
        "/api/v1/training/retrain/fb_classifier", json={"dry_run": True}
    ).json()
    r = client.get(f"/api/v1/training/jobs/{j['job_id']}")
    assert r.status_code == 200
    assert r.json()["model_name"] == "fb_classifier"


def test_get_job_not_found(client: TestClient):
    r = client.get("/api/v1/training/jobs/does_not_exist")
    assert r.status_code == 404
