"""Tests for POST /api/v1/runs/batch."""
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import pytest


def _make_client():
    from app.main import create_app
    from pathlib import Path
    import tempfile, sqlite3
    tmp = tempfile.mkdtemp()
    db = Path(tmp) / "cards.sqlite"
    sqlite3.connect(db).close()
    app = create_app(db_path=db)
    return TestClient(app)


def test_batch_requires_video_ids():
    client = _make_client()
    r = client.post("/api/v1/runs/batch", json={})
    assert r.status_code == 422


def test_batch_returns_batch_id():
    client = _make_client()
    with patch("app.api.batch.asyncio") as mock_asyncio:
        mock_asyncio.create_task = MagicMock()
        r = client.post("/api/v1/runs/batch", json={"video_ids": ["1", "2"]})
    assert r.status_code == 202
    assert "batch_id" in r.json()


def test_batch_status_404_unknown():
    client = _make_client()
    r = client.get("/api/v1/runs/batch/does-not-exist")
    assert r.status_code == 404
