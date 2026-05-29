"""BeamRunner — orchestrates Beam endpoint invocation for GPU pipeline runs."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import httpx

from app.services.event_bus import Event, EventBus
from app.services.result_importer import ResultImporter
from app.services import _event_bus_registry
from card_capture.data.connection import open_connection

_BEAM_BASE = "https://app.beam.cloud"


class BeamRunner:
    def __init__(
        self,
        bus: EventBus,
        db_path: Path,
        output_base: Path,
        api_key: str,
        volume_id: str,
        endpoint_id: str,
    ) -> None:
        self.bus = bus
        self._api_key = api_key
        self._volume_id = volume_id
        self._endpoint_id = endpoint_id
        self._importer = ResultImporter(db_path=db_path, output_base=output_base)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **kw,
    ) -> None:
        video_id: Optional[int] = kw.get("video_id")
        _event_bus_registry.set(run_id, self.bus)

        video_key = f"runs/{run_id}/input.mov"
        results_key = f"runs/{run_id}/results.tar.gz"

        try:
            self._record_run_start(run_id, video_id, db)
            self.bus.emit(run_id, Event(name="run_started"))

            self.bus.emit(run_id, Event(name="log", payload={"line": "Uploading video to Beam…"}))
            print(f"[{run_id}] beam: uploading video…", flush=True)
            await self._upload_to_volume(video_key, Path(video))

            self.bus.emit(run_id, Event(name="log", payload={"line": "Invoking Beam endpoint…"}))
            task_id = await self._invoke_endpoint(run_id, video_key, results_key, config_preset)
            print(f"[{run_id}] beam: task {task_id} submitted", flush=True)

            poll_count = 0
            while True:
                status = await self._poll_task(task_id)
                if status == "complete":
                    break
                if status in ("failed", "error", "cancelled"):
                    raise RuntimeError(f"Beam task {task_id} ended with status: {status}")
                if poll_count % 10 == 0:
                    msg = "Processing on Beam GPU…"
                    self.bus.emit(run_id, Event(name="log", payload={"line": msg}))
                    print(f"[{run_id}] beam: {msg}", flush=True)
                poll_count += 1
                await asyncio.sleep(3)

            self.bus.emit(run_id, Event(name="log", payload={"line": "Downloading results…"}))
            tarball = Path(output_dir) / f"{run_id}_results.tar.gz"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            await self._download_from_volume(results_key, tarball)
            n_cards = self._importer.import_tarball(tarball, run_id)
            tarball.unlink(missing_ok=True)

            self._record_run_finish(run_id, n_cards, db)
            print(f"[{run_id}] beam: done — {n_cards} cards imported", flush=True)
            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            print(f"[{run_id}] beam: FAILED — {exc}", flush=True)
            self._record_run_fail(run_id, db)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            _event_bus_registry.clear(run_id)
            await self._cleanup_volume(run_id)

    async def run_batch_async(self, jobs: list[dict]) -> None:
        for job in jobs:
            await self.run_async(**job)

    async def destroy_instance(self) -> None:
        pass  # Beam manages container lifecycle

    async def _upload_to_volume(self, key: str, path: Path) -> None:
        url = f"{_BEAM_BASE}/volume/files/{self._volume_id}/{key}"
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.put(
                url,
                content=path.read_bytes(),
                headers={**self._headers, "Content-Type": "application/octet-stream"},
            )
            r.raise_for_status()

    async def _download_from_volume(self, key: str, dest: Path) -> None:
        url = f"{_BEAM_BASE}/volume/files/{self._volume_id}/{key}"
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            dest.write_bytes(r.content)

    async def _invoke_endpoint(
        self, run_id: str, video_key: str, results_key: str, config_preset: str
    ) -> str:
        url = f"{_BEAM_BASE}/endpoint/{self._endpoint_id}/"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                url,
                json={
                    "run_id": run_id,
                    "video_volume_path": f"/volumes/{self._volume_id}/{video_key}",
                    "results_volume_path": f"/volumes/{self._volume_id}/{results_key}",
                    "config_preset": config_preset,
                },
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()["task_id"]

    async def _poll_task(self, task_id: str) -> str:
        url = f"{_BEAM_BASE}/task/{task_id}/status/"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            return r.json().get("status", "unknown")

    async def _cleanup_volume(self, run_id: str) -> None:
        for key in [f"runs/{run_id}/input.mov", f"runs/{run_id}/results.tar.gz"]:
            try:
                url = f"{_BEAM_BASE}/volume/files/{self._volume_id}/{key}"
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.delete(url, headers=self._headers)
            except Exception:
                pass

    def _record_run_start(self, run_id: str, video_id: Optional[int], db: str) -> None:
        try:
            with open_connection(db) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status)"
                    " VALUES (?, ?, 'running')",
                    (run_id, video_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, n_cards: int, db: str) -> None:
        try:
            with open_connection(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='completed', cards_extracted=?,"
                    " finished_at=datetime('now') WHERE run_id=?",
                    (n_cards, run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _record_run_fail(self, run_id: str, db: str) -> None:
        try:
            with open_connection(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='failed',"
                    " finished_at=datetime('now') WHERE run_id=?",
                    (run_id,),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run failure: {exc}", flush=True)
