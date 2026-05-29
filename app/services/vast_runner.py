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
from app.services import _event_bus_registry
from card_capture.data.connection import open_connection

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
        idle_timeout_s: int = 600,
    ) -> None:
        self.bus = bus
        self._client = VastAIClient(api_key)
        self._gpu_type = gpu_type
        self._template_id = template_id
        self._branch = branch
        self._idle_timeout_s = idle_timeout_s
        self._instance_id: Optional[int] = None
        self._worker: Optional[InstanceWorkerClient] = None
        self._ssh_tunnel: Optional[asyncio.subprocess.Process] = None
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
            with open_connection(db) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status) VALUES (?, ?, 'running')",
                    (run_id, video_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, n_cards: int, db: str) -> None:
        try:
            with open_connection(db) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='completed', cards_extracted=?, finished_at=datetime('now') WHERE run_id=?",
                    (n_cards, run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _record_run_fail(self, run_id: str, db: str) -> None:
        try:
            with open_connection(db) as conn:
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
        if self._ssh_tunnel:
            try:
                self._ssh_tunnel.terminate()
                await asyncio.wait_for(self._ssh_tunnel.wait(), timeout=3)
            except Exception:
                pass
            self._ssh_tunnel = None
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

        # Wait for IP + SSH details (both required — SSH details sometimes lag by 10-30s)
        ssh_host: str = ""
        ssh_port: int = 0
        ip: Optional[str] = None
        for i in range(48):  # up to 4 minutes
            await asyncio.sleep(5)
            details = self._client.get_instance_details(self._instance_id)
            ip = details.get("public_ipaddr") or None
            ssh_host = details.get("ssh_host", "")
            ssh_port = int(details.get("ssh_port", 0) or 0)
            if ip and ssh_host and ssh_port:
                break
            if i % 6 == 0:
                print(f"[vast.ai] Waiting for instance details… ip={ip} ssh={ssh_host}:{ssh_port}", flush=True)
        if not ip:
            raise RuntimeError("Instance did not receive an IP within 4 minutes")

        print(f"[vast.ai] Instance {self._instance_id} up", flush=True)
        print(f"[vast.ai] IP: {ip}  SSH: ssh -p {ssh_port} root@{ssh_host}", flush=True)
        print(f"[vast.ai] Tunnel cmd: ssh -p {ssh_port} root@{ssh_host} -N -L 8765:localhost:8765", flush=True)

        # vast.ai blocks arbitrary ports via their firewall — port 8765 is not
        # directly accessible. Create an SSH tunnel: Mac localhost:8765 → instance:8765.
        if ssh_host and ssh_port:
            # Wait for the SSH proxy port to accept connections — it lags behind
            # the instance IP assignment by 30-90 seconds on vast.ai's infrastructure.
            # Use asyncio.open_connection (non-blocking) to avoid stalling the event loop.
            print(f"[vast.ai] Waiting for SSH proxy {ssh_host}:{ssh_port}…", flush=True)
            for i in range(120):  # up to 10 minutes
                try:
                    _, w = await asyncio.wait_for(
                        asyncio.open_connection(ssh_host, ssh_port), timeout=4
                    )
                    w.close()
                    await w.wait_closed()
                    print(f"[vast.ai] SSH proxy ready after {i*5}s", flush=True)
                    break
                except Exception as e:
                    if i % 6 == 0:
                        print(f"[vast.ai] SSH proxy not ready yet ({i*5}s): {type(e).__name__}", flush=True)
                await asyncio.sleep(5)
            else:
                raise RuntimeError(
                    f"SSH proxy {ssh_host}:{ssh_port} not accepting connections after 10 minutes"
                )

            print(f"[vast.ai] Opening SSH tunnel via {ssh_host}:{ssh_port}…", flush=True)
            tunnel_cmd = [
                "ssh", "-N",
                "-L", "8765:localhost:8765",
                "-p", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ServerAliveInterval=30",
                "-o", "ConnectTimeout=60",
                f"root@{ssh_host}",
            ]
            print(f"[vast.ai] Running: {' '.join(tunnel_cmd)}", flush=True)
            self._ssh_tunnel = await asyncio.create_subprocess_exec(
                *tunnel_cmd,
                stderr=asyncio.subprocess.PIPE,
            )
            # Give tunnel time to establish
            await asyncio.sleep(6)
            if self._ssh_tunnel.returncode is not None:
                # Read stderr for diagnosis
                ssh_err = ""
                try:
                    ssh_err_bytes = await asyncio.wait_for(
                        self._ssh_tunnel.stderr.read(2000), timeout=2
                    )
                    ssh_err = ssh_err_bytes.decode(errors="replace").strip()
                except Exception:
                    pass
                print(f"[vast.ai] SSH tunnel stderr: {ssh_err}", flush=True)
                raise RuntimeError(
                    f"SSH tunnel exited (code {self._ssh_tunnel.returncode}): {ssh_err or 'no output'}. "
                    f"Fix: chmod 600 ~/.ssh/id_* && chmod 700 ~/.ssh  "
                    f"Test: ssh -v -p {ssh_port} root@{ssh_host} echo ok"
                )
            self._worker = InstanceWorkerClient("http://localhost:8765")
        else:
            print(f"[vast.ai] No SSH details — trying direct {ip}:8765 (may be blocked by firewall)", flush=True)
            self._worker = InstanceWorkerClient(f"http://{ip}:8765")

        # Wait for health (up to 8 minutes — image pull + startup takes time)
        for i in range(96):
            healthy = await self._worker.health_check()
            if healthy:
                print(f"[vast.ai] Worker healthy after {i*5}s ✓", flush=True)
                return
            if i % 6 == 0:
                print(f"[vast.ai] Waiting for worker… ({i*5}s)", flush=True)
            await asyncio.sleep(5)
        raise RuntimeError(
            f"Instance worker did not become healthy within 8 minutes. "
            f"SSH in: ssh -p {ssh_port} root@{ssh_host}  |  cat /tmp/worker.log"
        )
