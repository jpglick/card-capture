"""Tests for vastai_worker — uses FastAPI TestClient."""
import asyncio
import io
import json
import tarfile
from fastapi.testclient import TestClient

# Import lazily to avoid triggering startup in other tests
def _app():
    import importlib, sys
    if "app.vastai_worker" in sys.modules:
        del sys.modules["app.vastai_worker"]
    from app.vastai_worker import app
    return app


def test_health_returns_ok():
    client = TestClient(_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_submit_job_enqueues():
    client = TestClient(_app())
    r = client.post("/jobs", json={
        "job_id": "test-job-1",
        "video_path": "/tmp/fake.mp4",
        "config_preset": "balanced",
    })
    assert r.status_code == 200
    assert r.json()["job_id"] == "test-job-1"


def test_status_pending_after_submit():
    client = TestClient(_app())
    client.post("/jobs", json={"job_id": "test-job-2", "video_path": "/tmp/fake.mp4"})
    r = client.get("/jobs/test-job-2")
    assert r.status_code == 200
    assert r.json()["status"] in ("pending", "running")


def test_status_404_for_unknown_job():
    client = TestClient(_app())
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def test_results_409_when_not_complete():
    client = TestClient(_app())
    client.post("/jobs", json={"job_id": "not-done", "video_path": "/tmp/fake.mp4"})
    r = client.get("/jobs/not-done/results")
    assert r.status_code == 409


def test_confirm_deletes_job():
    client = TestClient(_app())
    client.post("/jobs", json={"job_id": "to-delete", "video_path": "/tmp/fake.mp4"})
    r = client.delete("/jobs/to-delete")
    assert r.status_code == 200
    assert client.get("/jobs/to-delete").status_code == 404
