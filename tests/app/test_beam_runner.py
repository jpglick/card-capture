"""Tests for BeamRunner — all HTTP calls mocked."""
import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.services.beam_runner import BeamRunner


def _make_runner(tmp_path):
    bus = MagicMock()
    bus.emit = MagicMock()
    return BeamRunner(
        bus=bus,
        db_path=tmp_path / "cards.sqlite",
        output_base=tmp_path,
        api_key="beam-test-key",
        volume_id="vol-abc123",
        endpoint_id="ep-xyz789",
    )


def _make_db(tmp_path):
    db = str(tmp_path / "cards.sqlite")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_runs "
            "(run_id TEXT PRIMARY KEY, video_id INTEGER, status TEXT, "
            "cards_extracted INTEGER, finished_at TEXT)"
        )
    return db


@pytest.mark.asyncio
async def test_run_async_emits_started_and_completed(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._importer = MagicMock()
    runner._importer.import_tarball.return_value = 5
    runner._upload_to_volume = AsyncMock()
    runner._invoke_endpoint = AsyncMock(return_value="task-001")
    runner._poll_task = AsyncMock(return_value="complete")
    runner._download_from_volume = AsyncMock(
        side_effect=lambda key, dest: dest.write_bytes(b"fake_tarball")
    )
    runner._cleanup_volume = AsyncMock()

    await runner.run_async(
        "run-1",
        video=str(tmp_path / "video.mov"),
        output_dir=str(tmp_path / "out"),
        db=db,
        config_preset="balanced",
    )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_started" in emitted_names
    assert "run_completed" in emitted_names
    runner._upload_to_volume.assert_called_once()
    runner._invoke_endpoint.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_emits_run_failed_on_error(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._upload_to_volume = AsyncMock(side_effect=RuntimeError("upload failed"))
    runner._cleanup_volume = AsyncMock()

    with pytest.raises(RuntimeError, match="upload failed"):
        await runner.run_async(
            "run-fail",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_failed" in emitted_names


@pytest.mark.asyncio
async def test_run_async_raises_on_beam_task_failure(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._upload_to_volume = AsyncMock()
    runner._invoke_endpoint = AsyncMock(return_value="task-002")
    runner._poll_task = AsyncMock(return_value="failed")
    runner._cleanup_volume = AsyncMock()

    with pytest.raises(RuntimeError, match="task-002"):
        await runner.run_async(
            "run-fail2",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )


def test_destroy_instance_is_noop(tmp_path):
    runner = _make_runner(tmp_path)
    asyncio.get_event_loop().run_until_complete(runner.destroy_instance())


def test_satisfies_gpu_runner_protocol(tmp_path):
    from app.services.gpu_runner import GPURunner
    runner = _make_runner(tmp_path)
    assert isinstance(runner, GPURunner)
