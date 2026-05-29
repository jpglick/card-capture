from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import RunManifest
from card_capture.platforms.manifests import export_manifest, import_manifest


def _sample_manifest() -> RunManifest:
    return RunManifest(
        run_id="r1",
        runtime_mode="cpu_debug",
        input_video="artifact://local/x.MOV",
        output_artifacts=["artifact://local/r1/cards/"],
        cards=[],
        stage_timings=[],
        contract_violations=[],
        version="0.5.5+phaseD",
    )


def test_export_then_import_roundtrip(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    path = export_manifest(manifest, tmp_path / "manifest.json")
    assert path.exists()
    loaded = import_manifest(path)
    assert loaded == manifest


def test_import_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_manifest(tmp_path / "nope.json")


def test_export_creates_parent_directories(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    path = export_manifest(manifest, tmp_path / "nested" / "subdir" / "m.json")
    assert path.exists()
