"""config_presets repository (production schema: migrations/0003_config_presets.sql)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class ConfigRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def upsert_preset(self, *, name: str, description: str = "", config: Mapping[str, object]) -> None:
        if self._writer is None:
            raise RuntimeError("ConfigRepository.upsert_preset requires a Writer")
        self._writer.submit(Write(
            sql="INSERT OR REPLACE INTO config_presets(preset_name, description, config_json) VALUES (?, ?, ?)",
            params=(name, description, json.dumps(dict(config))),
        ))

    def get_preset(self, name: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT config_json FROM config_presets WHERE preset_name=?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def list_presets(self) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT preset_name, description, config_json FROM config_presets ORDER BY created_at"
            ).fetchall()
        return [
            {
                "preset_name": r[0],
                "description": r[1],
                "config": json.loads(r[2]),
            }
            for r in rows
        ]
