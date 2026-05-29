"""Tests for RunPodRunner — all HTTP and S3 calls mocked."""
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.services.runpod_runner import RunPodRunner


def _make_runner(tmp_path):
    bus = MagicMock()
    bus.emit = MagicMock()
    return RunPodRunner(
        bus=bus,
        db_path=tmp_path / "cards.sqlite",
        output_base=tmp_path,
        api_key="rp-test-key",
        endpoint_id="ep-runpod-001",
        r2_account_id="test-account",
        r2_bucket="cc-runpod-bucket",
        r2_access_key_id="AKIATEST",
        r2_secret_access_key="secret",
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
    runner._importer.import_tarball.return_value = 3
    runner._upload_video = MagicMock()
    runner._download_results = MagicMock(
        side_effect=lambda key, dest: dest.write_bytes(b"fake_tarball")
    )
    runner._submit_job = AsyncMock(return_value="rp-job-001")
    # _poll_job now returns (status, body) so callers can capture body["output"]
    runner._poll_job = AsyncMock(return_value=("COMPLETED", {"output": {"diagnostics": {}}}))
    runner._cleanup_r2 = AsyncMock()

    await runner.run_async(
        "run-rp-1",
        video=str(tmp_path / "video.mov"),
        output_dir=str(tmp_path / "out"),
        db=db,
        config_preset="balanced",
    )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_started" in emitted_names
    assert "run_completed" in emitted_names
    runner._submit_job.assert_called_once()
    runner._importer.import_handler_output.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_emits_run_failed_on_upload_error(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._upload_video = MagicMock(side_effect=RuntimeError("S3 upload failed"))
    runner._cleanup_r2 = AsyncMock()

    with pytest.raises(RuntimeError, match="S3 upload failed"):
        await runner.run_async(
            "run-rp-fail",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )

    emitted_names = [call.args[1].name for call in runner.bus.emit.call_args_list]
    assert "run_failed" in emitted_names


@pytest.mark.asyncio
async def test_run_async_raises_on_runpod_job_failure(tmp_path):
    runner = _make_runner(tmp_path)
    (tmp_path / "video.mov").write_bytes(b"fake_video")
    db = _make_db(tmp_path)

    runner._upload_video = MagicMock()
    runner._submit_job = AsyncMock(return_value="rp-job-002")
    # _poll_job now returns (status, body)
    runner._poll_job = AsyncMock(return_value=("FAILED", {"error": "test failure"}))
    runner._cleanup_r2 = AsyncMock()

    with pytest.raises(RuntimeError, match="rp-job-002"):
        await runner.run_async(
            "run-rp-fail2",
            video=str(tmp_path / "video.mov"),
            output_dir=str(tmp_path / "out"),
            db=db,
        )


@pytest.mark.asyncio
async def test_destroy_instance_is_noop(tmp_path):
    runner = _make_runner(tmp_path)
    await runner.destroy_instance()


def test_satisfies_gpu_runner_protocol(tmp_path):
    from app.services.gpu_runner import GPURunner
    runner = _make_runner(tmp_path)
    assert isinstance(runner, GPURunner)
