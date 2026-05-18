"""Orchestrates vast.ai instance lifecycle + job execution."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Optional

from app.services.event_bus import Event, EventBus
from app.services.vast_client import VastAIClient
from app.services.worker_client import InstanceWorkerClient
from app.services.result_importer import ResultImporter
from app.services import _event_bus_registry

_CONFIG_PATH = Path(__file__).parent.parent.parent / "card_capture_config.json"


def _save_active_instance(instance_id: int) -> None:
    data: dict = {}
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
    data["active_vast_instance"] = instance_id
    _CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _clear_active_instance() -> None:
    if not _CONFIG_PATH.exists():
        return
    try:
        data = json.loads(_CONFIG_PATH.read_text())
        data["active_vast_instance"] = None
        _CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


class VastAIRunner:
    """Mirrors PipelineRunner.run_async interface for drop-in substitution."""

    def __init__(
        self,
        bus: EventBus,
        db_path: Path,
        output_base: Path,
        api_key: str,
        gpu_type: str = "RTX 4090",
        template_id: str = "",
        branch: str = "main",
        idle_timeout_s: int = 300,
    ) -> None:
        self.bus = bus
        self._client = VastAIClient(api_key)
        self._gpu_type = gpu_type
        self._template_id = template_id
        self._branch = branch
        self._idle_timeout_s = idle_timeout_s
        self._instance_id: Optional[int] = None
        self._worker: Optional[InstanceWorkerClient] = None
        self._importer = ResultImporter(db_path=db_path, output_base=output_base)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **_kw,
    ) -> None:
        """Provision instance (if needed), run job, download results, destroy."""
        video_id: Optional[int] = _kw.get("video_id")

        # Register event bus so SSE stream can deliver events to the browser
        _event_bus_registry.set(run_id, self.bus)

        try:
            # Create pipeline_runs record so the run appears in the UI immediately
            self._record_run_start(run_id, video_id, db)

            print(f"[{run_id}] vast.ai: provisioning {self._gpu_type} instance…", flush=True)
            await self._ensure_instance()
            self.bus.emit(run_id, Event(name="run_started"))
            self.bus.emit(run_id, Event(name="log", payload={"line": f"Instance ready — uploading video…"}))
            print(f"[{run_id}] vast.ai: instance ready, uploading video…", flush=True)

            server_path = await self._worker.upload_video(Path(video))
            self.bus.emit(run_id, Event(name="log", payload={"line": "Upload complete — running CUDA pipeline…"}))
            print(f"[{run_id}] vast.ai: video uploaded, submitting job…", flush=True)
            await self._worker.submit_job(run_id, server_path, {"config_preset": config_preset})

            # Poll until complete or failed
            poll_count = 0
            while True:
                status = await self._worker.poll_status(run_id)
                state = status.get("status", "unknown")
                if state == "complete":
                    break
                if state == "failed":
                    raise RuntimeError(f"Remote job failed: {status.get('error', 'unknown')}")
                if poll_count % 10 == 0:  # log every 30s
                    pct = status.get("progress_pct", 0)
                    msg = f"Processing on cloud GPU… {pct}%"
                    self.bus.emit(run_id, Event(name="log", payload={"line": msg}))
                    print(f"[{run_id}] vast.ai: {msg}", flush=True)
                poll_count += 1
                await asyncio.sleep(3)

            print(f"[{run_id}] vast.ai: job complete, downloading results…", flush=True)
            self.bus.emit(run_id, Event(name="log", payload={"line": "Pipeline complete — downloading results…"}))

            # Download and import results
            tarball = Path(output_dir) / f"{run_id}_results.tar.gz"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            await self._worker.download_results(run_id, tarball)
            await self._worker.confirm_downloaded(run_id)
            n_cards = self._importer.import_tarball(tarball, run_id)
            tarball.unlink(missing_ok=True)

            self._record_run_finish(run_id, n_cards, db)
            print(f"[{run_id}] vast.ai: done — {n_cards} cards imported", flush=True)
            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            print(f"[{run_id}] vast.ai: FAILED — {exc}", flush=True)
            self._record_run_fail(run_id, db)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            _event_bus_registry.clear(run_id)
            await self.destroy_instance()

    def _record_run_start(self, run_id: str, video_id: Optional[int], db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status) VALUES (?, ?, 'running')",
                    (run_id, video_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, n_cards: int, db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='completed', cards_extracted=?, finished_at=datetime('now') WHERE run_id=?",
                    (n_cards, run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _record_run_fail(self, run_id: str, db: str) -> None:
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='failed', finished_at=datetime('now') WHERE run_id=?",
                    (run_id,),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run failure: {exc}", flush=True)

    async def run_batch_async(self, jobs: list[dict]) -> None:
        """Process multiple videos on one instance, destroy when all done."""
        try:
            await self._ensure_instance()
            for job in jobs:
                await self.run_async(**job)
        finally:
            await self.destroy_instance()

    async def destroy_instance(self) -> None:
        if self._instance_id is None:
            return
        if self._worker:
            await self._worker.close()
            self._worker = None
        self._client.destroy(self._instance_id)
        _clear_active_instance()
        self._instance_id = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_instance(self) -> None:
        if self._worker is not None:
            return

        # Find cheapest offer
        offers = self._client.search_offers(self._gpu_type)
        if not offers:
            raise RuntimeError(f"No vast.ai offers found for GPU type: {self._gpu_type}")
        offer_id = offers[0]["id"]

        # Provision
        result = self._client.provision(offer_id, self._template_id)
        self._instance_id = result["id"]
        _save_active_instance(self._instance_id)

        # Wait for IP (up to 3 minutes — image pull can delay container start)
        ip: Optional[str] = None
        for _ in range(36):
            await asyncio.sleep(5)
            ip = self._client.get_instance_ip(self._instance_id)
            if ip:
                break
        if not ip:
            raise RuntimeError("Instance did not receive an IP within 3 minutes")

        self._worker = InstanceWorkerClient(f"http://{ip}:8765")

        # Fetch SSH connection details so the user can debug via console if needed
        try:
            details = self._client.get_instance_details(self._instance_id)
            ssh_host = details.get("ssh_host", "")
            ssh_port = details.get("ssh_port", "")
            if ssh_host and ssh_port:
                print(f"[vast.ai] SSH: ssh -p {ssh_port} root@{ssh_host}", flush=True)
                print(f"[vast.ai] Worker log: cat /tmp/worker.log", flush=True)
        except Exception:
            pass

        # Wait for health (up to 8 minutes — first pull of 12-15 GB image takes time)
        for i in range(96):
            healthy = await self._worker.health_check()
            if healthy:
                return
            if i % 6 == 0:  # every 30s
                print(f"[vast.ai] Waiting for worker on {ip}:8765… ({i*5}s)", flush=True)
            await asyncio.sleep(5)
        raise RuntimeError(
            f"Instance worker did not become healthy within 8 minutes. "
            f"SSH in to debug: ssh -p {ssh_port} root@{ssh_host} "
            f"then check: cat /tmp/worker.log"
        )
