"""Tests for worker_core — shared GPU pipeline execution logic."""
import io
import json
import sqlite3
import tarfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.worker_core import (
    CUDA_CONFIG_OVERRIDES,
    apply_cuda_config,
    restore_config,
    package_results,
    parse_metaflow_start_stage,
)


def test_apply_cuda_config_writes_overrides(tmp_path, monkeypatch):
    config_file = tmp_path / "card_capture_config.json"
    config_file.write_text(json.dumps({"corner_confidence": 0.5}))
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", config_file)

    original = apply_cuda_config()

    written = json.loads(config_file.read_text())
    assert written["device"] == "cuda"
    assert written["pipeline_backend"] == "cuda"
    assert written["corner_confidence"] == 0.5  # untouched key preserved
    assert original["device"] is None  # key was absent before


def test_apply_cuda_config_returns_original_values(tmp_path, monkeypatch):
    config_file = tmp_path / "card_capture_config.json"
    config_file.write_text(json.dumps({"device": "mps", "cuda_stride": 1}))
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", config_file)

    original = apply_cuda_config()

    assert original["device"] == "mps"
    assert original["cuda_stride"] == 1


def test_restore_config_restores_originals(tmp_path, monkeypatch):
    config_file = tmp_path / "card_capture_config.json"
    config_file.write_text(json.dumps({"device": "cuda", "corner_confidence": 0.5}))
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", config_file)

    restore_config({"device": "mps", "cuda_stride": None})

    written = json.loads(config_file.read_text())
    assert written["device"] == "mps"
    assert written["cuda_stride"] is None
    assert written["corner_confidence"] == 0.5  # untouched


def test_restore_config_noop_when_no_file(tmp_path, monkeypatch):
    missing = tmp_path / "nonexistent.json"
    monkeypatch.setattr("app.worker_core._CONFIG_PATH", missing)
    restore_config({"device": "mps"})  # must not raise


def test_package_results_returns_valid_gzipped_tarball(tmp_path):
    output_dir = tmp_path / "output"
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True)
    (crops_dir / "card1.jpg").write_bytes(b"fake_image_data")
    db_path = tmp_path / "cards.sqlite"  # does not exist — cards list will be empty

    result = package_results("job123", output_dir, db_path)

    assert isinstance(result, bytes)
    assert len(result) > 0
    buf = io.BytesIO(result)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        names = tar.getnames()
    assert "export.json" in names
    assert any(n.startswith("crops") for n in names)


def test_package_results_export_json_is_empty_list_when_no_db(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    db_path = tmp_path / "cards.sqlite"  # does not exist

    result = package_results("job999", output_dir, db_path)

    buf = io.BytesIO(result)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        export_member = tar.getmember("export.json")
        f = tar.extractfile(export_member)
        cards = json.loads(f.read())
    assert cards == []


def test_package_results_includes_worker_database(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    db_path = output_dir / "cards.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE card_instances (run_id TEXT, track_id TEXT, session_id TEXT, fused_image_path TEXT, angle TEXT)")

    result = package_results("job-db", output_dir, db_path)

    buf = io.BytesIO(result)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        assert "cards.sqlite" in tar.getnames()


def test_parse_metaflow_start_stage_accepts_pid_format():
    line = "2026-05-24 22:13:27.434 [1779660806748398/detect/2 (pid 1254)] Task is starting."

    assert parse_metaflow_start_stage(line) == "detect"


def test_parse_metaflow_start_stage_accepts_legacy_format():
    line = "2026-05-24 22:13:27.434 [1779660806748398/refine/5] Task is starting."

    assert parse_metaflow_start_stage(line) == "refine"
