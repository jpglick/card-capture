"""Unpack a results tarball from the instance and import cards into local SQLite."""
from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from pathlib import Path


class ResultImporter:
    def __init__(self, db_path: Path, output_base: Path) -> None:
        self.db_path = db_path
        self.output_base = output_base

    def import_tarball(self, tarball_path: Path, run_id: str) -> int:
        """Unpack tarball, copy crops, import card rows. Returns count of new cards."""
        run_dir = self.output_base / run_id
        crops_dir = run_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tarball_path, "r:gz") as tar:
            # Extract crop files into run_dir/crops/
            for member in tar.getmembers():
                if member.name.startswith("crops/") and not member.isdir():
                    fname = member.name.split("/", 1)[1]
                    data = tar.extractfile(member)
                    if data:
                        (crops_dir / fname).write_bytes(data.read())

            # Load export.json
            export_f = tar.extractfile("export.json")
            if not export_f:
                raise ValueError("Tarball missing export.json")
            cards: list[dict] = json.loads(export_f.read())

        count = 0
        with sqlite3.connect(self.db_path) as conn:
            for card in cards:
                fname = Path(card["fused_image_path"]).name
                local_path = str(crops_dir / fname)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO card_instances
                           (run_id, track_id, session_id, fused_image_path, angle)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            run_id,
                            card.get("track_id", ""),
                            card.get("session_id", 0),
                            local_path,
                            card.get("side", "Front"),
                        ),
                    )
                    count += conn.execute("SELECT changes()").fetchone()[0]
                except sqlite3.Error:
                    pass
            conn.commit()
        return count
