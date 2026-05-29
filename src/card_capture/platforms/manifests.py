"""Manifest import/export helpers shared by platform adapters."""
from __future__ import annotations

from pathlib import Path

from card_capture.pipeline.request import RunManifest


def export_manifest(manifest: RunManifest, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.to_json())
    return p


def import_manifest(path: Path | str) -> RunManifest:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    return RunManifest.from_json(p.read_text())
