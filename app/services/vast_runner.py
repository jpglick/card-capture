"""Orchestrates vast.ai instance lifecycle + job execution."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from app.services.event_bus import Event, EventBus
from app.services.vast_client import VastAIClient
from app.services.worker_client import InstanceWorkerClient
from app.services.result_importer import ResultImporter

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
        try:
            await self._ensure_instance()
            self.bus.emit(run_id, Event(name="run_started"))

            server_path = await self._worker.upload_video(Path(video))
            await self._worker.submit_job(run_id, server_path, {"config_preset": config_preset})

            # Poll until complete or failed
            while True:
                status = await self._worker.poll_status(run_id)
                if status["status"] == "complete":
                    break
                if status["status"] == "failed":
                    raise RuntimeError(f"Remote job failed: {status.get('error', 'unknown')}")
                await asyncio.sleep(3)

            # Download and import results
            tarball = Path(output_dir) / f"{run_id}_results.tar.gz"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            await self._worker.download_results(run_id, tarball)
            await self._worker.confirm_downloaded(run_id)
            self._importer.import_tarball(tarball, run_id)
            tarball.unlink(missing_ok=True)

            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            await self.destroy_instance()

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
        result = self._client.provision(offer_id, self._template_id, self._branch)
        self._instance_id = result["id"]
        _save_active_instance(self._instance_id)

        # Wait for IP (up to 2 minutes)
        ip: Optional[str] = None
        for _ in range(24):
            await asyncio.sleep(5)
            ip = self._client.get_instance_ip(self._instance_id)
            if ip:
                break
        if not ip:
            raise RuntimeError("Instance did not receive an IP within 2 minutes")

        self._worker = InstanceWorkerClient(f"http://{ip}:8765")

        # Wait for health (up to 3 more minutes)
        for _ in range(36):
            if await self._worker.health_check():
                return
            await asyncio.sleep(5)
        raise RuntimeError("Instance worker did not become healthy within 5 minutes")
