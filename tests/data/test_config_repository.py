"""ConfigRepository read/write tests against the production schema."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.config import ConfigRepository
from card_capture.data.writer import Writer


def test_set_and_get_preset(prod_db: Path) -> None:
    writer = Writer(prod_db)
    writer.start()
    try:
        repo = ConfigRepository(writer=writer, db_path=prod_db)
        repo.upsert_preset(name="balanced", config={"corner_confidence": 0.5})
        writer.flush()
        loaded = repo.get_preset("balanced")
    finally:
        writer.stop()
    assert loaded == {"corner_confidence": 0.5}


def test_get_missing_returns_none(prod_db: Path) -> None:
    repo = ConfigRepository(writer=None, db_path=prod_db)
    assert repo.get_preset("does-not-exist") is None


def test_list_presets_returns_created_order(prod_db: Path) -> None:
    writer = Writer(prod_db)
    writer.start()
    try:
        repo = ConfigRepository(writer=writer, db_path=prod_db)
        repo.upsert_preset(name="b", config={})
        repo.upsert_preset(name="a", config={})
        repo.upsert_preset(name="c", config={})
        writer.flush()
        presets = repo.list_presets()
    finally:
        writer.stop()
    assert [p["preset_name"] for p in presets] == ["b", "a", "c"]

