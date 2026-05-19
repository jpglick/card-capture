"""Tests for VastAIRunner — all external calls mocked."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.vast_runner import VastAIRunner


def _make_runner(tmp_path):
    bus = MagicMock()
    bus.emit = MagicMock()
    return VastAIRunner(
        bus=bus,
        db_path=tmp_path / "cards.sqlite",
        output_base=tmp_path,
        api_key="test-key",
        gpu_type="RTX 4090",
        template_id="pytorch/pytorch:latest",
    )


@pytest.mark.asyncio
async def test_run_async_emits_started_and_completed(tmp_path):
    runner = _make_runner(tmp_path)

    # Mock collaborators
    runner._client = MagicMock()
    runner._client.search_offers.return_value = [{"id": 1, "dph_total": 0.5}]
    runner._client.provision.return_value = {"id": 42}
    runner._client.get_instance_details.return_value = {
        "public_ipaddr": "1.2.3.4",
        "ssh_host": "ssh1.vast.ai",
        "ssh_port": 12345,
    }

    worker = AsyncMock()
    worker.health_check.return_value = True
    worker.upload_video.return_value = "/tmp/video.mp4"
    worker.poll_status.return_value = {"status": "complete"}
    worker.download_results = AsyncMock()
    worker.confirm_downloaded = AsyncMock()
    worker.close = AsyncMock()

    importer = MagicMock()
    importer.import_tarball.return_value = 3
    runner._importer = importer

    # Mock subprocess so no real SSH is attempted
    fake_proc = MagicMock()
    fake_proc.returncode = None  # tunnel still running (not exited)
    fake_proc.terminate = MagicMock()
    fake_proc.wait = AsyncMock()

    mock_writer = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    with patch("app.services.vast_runner.InstanceWorkerClient", return_value=worker), \
         patch("app.services.vast_runner._save_active_instance"), \
         patch("app.services.vast_runner._clear_active_instance"), \
         patch("asyncio.create_subprocess_exec", return_value=fake_proc), \
         patch("asyncio.open_connection", return_value=(AsyncMock(), mock_writer)):
        (tmp_path / "run-1").mkdir()
        await runner.run_async(
            "run-1",
            video=str(tmp_path / "video.mp4"),
            output_dir=str(tmp_path / "run-1"),
            db=str(tmp_path / "cards.sqlite"),
        )

    event_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_started" in event_names
    assert "run_completed" in event_names


@pytest.mark.asyncio
async def test_run_async_emits_failed_on_error(tmp_path):
    runner = _make_runner(tmp_path)
    runner._client = MagicMock()
    runner._client.search_offers.side_effect = RuntimeError("no offers")

    with patch("app.services.vast_runner._save_active_instance"):
        with patch("app.services.vast_runner._clear_active_instance"):
            with pytest.raises(RuntimeError):
                await runner.run_async(
                    "run-fail",
                    video="/tmp/v.mp4",
                    output_dir=str(tmp_path),
                    db=str(tmp_path / "cards.sqlite"),
                )

    event_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_failed" in event_names


@pytest.mark.asyncio
async def test_destroy_calls_client(tmp_path):
    runner = _make_runner(tmp_path)
    runner._instance_id = 42
    runner._client = MagicMock()
    runner._worker = AsyncMock()
    runner._worker.close = AsyncMock()

    with patch("app.services.vast_runner._clear_active_instance"):
        await runner.destroy_instance()

    runner._client.destroy.assert_called_once_with(42)
    assert runner._instance_id is None
