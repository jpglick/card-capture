"""Tests for InstanceWorkerClient — uses httpx mock transport."""
import json
from pathlib import Path
import httpx
import pytest

from app.services.worker_client import InstanceWorkerClient


def _make_transport(responses: dict):
    """Build a mock httpx transport from {path: (status, body)} dict."""
    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            key = f"{request.method} {request.url.path}"
            status, body = responses.get(key, (404, b"not found"))
            return httpx.Response(status, content=body)
    return MockTransport()


@pytest.mark.asyncio
async def test_health_check_returns_true():
    t = _make_transport({"GET /health": (200, b'{"status":"ok"}')})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    assert await client.health_check() is True
    await client.close()


@pytest.mark.asyncio
async def test_health_check_returns_false_on_error():
    t = _make_transport({})  # no routes → 404
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    assert await client.health_check() is False
    await client.close()


@pytest.mark.asyncio
async def test_submit_job():
    t = _make_transport({"POST /jobs": (200, b'{"job_id":"run-1"}')})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    await client.submit_job("run-1", "/tmp/video.mp4", {"config_preset": "balanced"})
    await client.close()


@pytest.mark.asyncio
async def test_poll_status_returns_dict():
    body = json.dumps({"status": "running", "progress_pct": 50}).encode()
    t = _make_transport({"GET /jobs/run-1": (200, body)})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    status = await client.poll_status("run-1")
    assert status["status"] == "running"
    await client.close()


@pytest.mark.asyncio
async def test_confirm_downloaded():
    t = _make_transport({"DELETE /jobs/run-1": (200, b'{"deleted":"run-1"}')})
    client = InstanceWorkerClient("http://1.2.3.4:8765", transport=t)
    await client.confirm_downloaded("run-1")
    await client.close()
