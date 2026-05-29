"""RunPodRunner — orchestrates RunPod serverless endpoint for GPU pipeline runs.

File transfer uses Cloudflare R2 (S3-compatible, zero egress fees).
R2 credentials are also injected into the RunPod endpoint as env vars so the
worker handler can access the same bucket without embedding secrets in job payloads.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import boto3
import httpx
from botocore.config import Config as BotocoreConfig

from app.services.event_bus import Event, EventBus
from app.services.result_importer import ResultImporter
from app.services import _event_bus_registry
from card_capture.data.connection import open_connection
from card_capture.data.sql_queries import (
    PIPELINE_RUN_INSERT_START,
    PIPELINE_RUN_MARK_COMPLETED,
    PIPELINE_RUN_MARK_FAILED,
)

_RUNPOD_API = "https://api.runpod.ai/v2"


def _r2_client(account_id: str, access_key_id: str, secret_access_key: str):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=BotocoreConfig(
            connect_timeout=30,
            read_timeout=300,
            retries={"max_attempts": 2},
            s3={"addressing_style": "path"},
        ),
    )


class RunPodRunner:
    def __init__(
        self,
        bus: EventBus,
        db_path: Path,
        output_base: Path,
        api_key: str,
        endpoint_id: str,
        r2_account_id: str,
        r2_bucket: str,
        r2_access_key_id: str,
        r2_secret_access_key: str,
    ) -> None:
        self.bus = bus
        self._api_key = api_key
        self._endpoint_id = endpoint_id
        self._r2_bucket = r2_bucket
        self._r2_account_id = r2_account_id
        self._r2_access_key_id = r2_access_key_id
        self._r2_secret_access_key = r2_secret_access_key
        self._s3 = _r2_client(r2_account_id, r2_access_key_id, r2_secret_access_key)
        self._importer = ResultImporter(db_path=db_path, output_base=output_base)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

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
        import time as _time
        video_id: Optional[int] = kw.get("video_id")
        _event_bus_registry.set(run_id, self.bus)

        video_key = f"runs/{run_id}/input.mov"
        results_key = f"runs/{run_id}/results.tar.gz"
        video_mb = Path(video).stat().st_size / 1_048_576

        t_start = _time.time()

        try:
            self._record_run_start(run_id, video_id, db)
            self.bus.emit(run_id, Event(name="run_started"))

            self.bus.emit(run_id, Event(name="log", payload={"line": f"Uploading {video_mb:.0f} MB video to R2…"}))
            print(f"[{run_id}] runpod: uploading {video_mb:.0f} MB video to R2…", flush=True)
            t_upload_start = _time.time()
            await asyncio.get_event_loop().run_in_executor(
                None, self._upload_video, video_key, Path(video)
            )
            t_upload = _time.time() - t_upload_start
            print(f"[{run_id}] runpod: R2 upload done in {t_upload:.1f}s ({video_mb/t_upload:.1f} MB/s)", flush=True)

            self.bus.emit(run_id, Event(name="log", payload={"line": "Submitting RunPod job…"}))
            t_submit = _time.time()
            job_id = await self._submit_job(run_id, video_key, results_key, config_preset)
            print(f"[{run_id}] runpod: job {job_id} submitted", flush=True)

            poll_count = 0
            handler_output: dict = {}
            while True:
                status, body = await self._poll_job(job_id)
                if status == "COMPLETED":
                    # Capture the handler's full return value (timings,
                    # stage_payloads, detect_telemetry, GPU info, db_diag) so
                    # we can persist it locally — RunPod auto-deletes job data
                    # after a short window so this is our only chance.
                    handler_output = body.get("output") or {}
                    break
                if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    raise RuntimeError(f"RunPod job {job_id} ended with status: {status}")
                elapsed = _time.time() - t_submit
                if poll_count % 10 == 0:
                    msg = f"Processing on RunPod GPU… ({elapsed:.0f}s)"
                    self.bus.emit(run_id, Event(name="log", payload={"line": msg}))
                    print(f"[{run_id}] runpod: {msg}", flush=True)
                poll_count += 1
                await asyncio.sleep(3)

            t_gpu = _time.time() - t_submit
            print(f"[{run_id}] runpod: GPU job completed in {t_gpu:.1f}s", flush=True)

            # Persist the handler output dict locally + emit it as an event so
            # the UI can show timings/telemetry. File path mirrors the tarball
            # location so debugging tools can find it next to crops.
            try:
                import json as _json
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                ho_path = Path(output_dir) / f"{run_id}_handler_output.json"
                ho_path.write_text(_json.dumps(handler_output, indent=2))
                print(f"[{run_id}] runpod: saved handler output to {ho_path}", flush=True)
            except Exception as exc:
                print(f"[{run_id}] runpod: could not save handler output: {exc}", flush=True)
            self.bus.emit(run_id, Event(
                name="handler_output", payload={"output": handler_output}
            ))

            self.bus.emit(run_id, Event(name="log", payload={"line": "Downloading results from R2…"}))
            tarball = Path(output_dir) / f"{run_id}_results.tar.gz"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            t_dl_start = _time.time()
            await asyncio.get_event_loop().run_in_executor(
                None, self._download_results, results_key, tarball
            )
            t_dl = _time.time() - t_dl_start
            result_mb = tarball.stat().st_size / 1_048_576
            print(f"[{run_id}] runpod: R2 download done in {t_dl:.1f}s ({result_mb:.1f} MB)", flush=True)

            n_cards = self._importer.import_tarball(tarball, run_id)
            tarball.unlink(missing_ok=True)
            try:
                self._importer.import_handler_output(handler_output, run_id)
            except Exception as exc:
                print(f"[{run_id}] runpod: could not import handler diagnostics: {exc}", flush=True)

            t_total = _time.time() - t_start
            self._record_run_finish(run_id, n_cards, db)
            print(
                f"[{run_id}] runpod: done — {n_cards} cards | "
                f"upload={t_upload:.1f}s gpu={t_gpu:.1f}s download={t_dl:.1f}s total={t_total:.1f}s",
                flush=True,
            )
            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            print(f"[{run_id}] runpod: FAILED — {exc}", flush=True)
            self._record_run_fail(run_id, db)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            _event_bus_registry.clear(run_id)
            await self._cleanup_r2(run_id)

    async def run_batch_async(self, jobs: list[dict]) -> None:
        for job in jobs:
            await self.run_async(**job)

    async def destroy_instance(self) -> None:
        pass  # RunPod manages container lifecycle

    def _upload_video(self, key: str, path: Path) -> None:
        self._s3.upload_file(str(path), self._r2_bucket, key)

    def _download_results(self, key: str, dest: Path) -> None:
        self._s3.download_file(self._r2_bucket, key, str(dest))

    async def _submit_job(
        self, run_id: str, video_key: str, results_key: str, config_preset: str
    ) -> str:
        url = f"{_RUNPOD_API}/{self._endpoint_id}/run"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                url,
                json={
                    "input": {
                        "run_id": run_id,
                        "video_r2_key": video_key,
                        "results_r2_key": results_key,
                        "r2_bucket": self._r2_bucket,
                        "config_preset": config_preset,
                    }
                },
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()["id"]

    async def _poll_job(self, job_id: str) -> tuple[str, dict]:
        """Return (status, full body) so callers can capture output on COMPLETED.

        Previously returned only the status string and threw away body["output"]
        — which contains the handler's full diagnostics (stage_payloads,
        detect_telemetry, timings, GPU info). That data never reached the local
        app; cards landed via the tarball but metrics were silently dropped.
        """
        url = f"{_RUNPOD_API}/{self._endpoint_id}/status/{job_id}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            body = r.json()
            status = body.get("status", "UNKNOWN")
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                error = body.get("error") or body.get("output", {})
                print(f"[runpod] job {job_id} {status}: {error}", flush=True)
            return status, body

    async def _cleanup_r2(self, run_id: str) -> None:
        loop = asyncio.get_event_loop()
        for key in [f"runs/{run_id}/input.mov", f"runs/{run_id}/results.tar.gz"]:
            try:
                await loop.run_in_executor(
                    None,
                    lambda k=key: self._s3.delete_object(Bucket=self._r2_bucket, Key=k),
                )
            except Exception:
                pass

    def _record_run_start(self, run_id: str, video_id: Optional[int], db: str) -> None:
        try:
            with open_connection(db) as conn:
                conn.execute(
                    PIPELINE_RUN_INSERT_START,
                    (run_id, video_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, n_cards: int, db: str) -> None:
        try:
            with open_connection(db) as conn:
                conn.execute(
                    PIPELINE_RUN_MARK_COMPLETED,
                    (n_cards, run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _record_run_fail(self, run_id: str, db: str) -> None:
        try:
            with open_connection(db) as conn:
                conn.execute(
                    PIPELINE_RUN_MARK_FAILED,
                    (run_id,),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run failure: {exc}", flush=True)
