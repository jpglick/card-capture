"""batch_jobs repository (production schema: migrations/0013_v55_repository_schema.sql)."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class BatchRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def enqueue(self, *, batch_id: str, total: int, status: str = "queued") -> None:
        if self._writer is None:
            raise RuntimeError("BatchRepository.enqueue requires a Writer")
        self._writer.submit(Write(
            sql="INSERT INTO batch_jobs(batch_id, total, status) VALUES (?, ?, ?)",
            params=(batch_id, total, status),
        ))

    def update_progress(self, *, batch_id: str, completed: int = 0, failed: int = 0, status: str | None = None) -> None:
        if self._writer is None:
            raise RuntimeError("BatchRepository.update_progress requires a Writer")
        
        updates = []
        params = []
        if completed is not None:
            updates.append("completed = ?")
            params.append(completed)
        if failed is not None:
            updates.append("failed = ?")
            params.append(failed)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            
        if not updates:
            return
            
        params.append(batch_id)
        self._writer.submit(Write(
            sql=f"UPDATE batch_jobs SET {', '.join(updates)} WHERE batch_id = ?",
            params=tuple(params),
        ))

    def get(self, batch_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT batch_id, status, total, completed, failed, created_at FROM batch_jobs WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
        if row is None:
            return None
        keys = ("batch_id", "status", "total", "completed", "failed", "created_at")
        return dict(zip(keys, row))

    def list_pending(self) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT batch_id, status, total, completed, failed, created_at "
                "FROM batch_jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
            ).fetchall()
        keys = ("batch_id", "status", "total", "completed", "failed", "created_at")
        return [dict(zip(keys, r)) for r in rows]
