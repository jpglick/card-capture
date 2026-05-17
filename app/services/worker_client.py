"""HTTP client for the instance-side vastai_worker FastAPI app."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx


class InstanceWorkerClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    async def health_check(self) -> bool:
        try:
            r = await self._client.get("/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def upload_video(self, video_path: Path) -> str:
        """Upload the video file; returns the server-side path string."""
        with open(video_path, "rb") as f:
            r = await self._client.post(
                "/upload",
                files={"file": (video_path.name, f, "video/mp4")},
                timeout=600.0,
            )
        r.raise_for_status()
        return r.json()["path"]

    async def submit_job(self, job_id: str, server_video_path: str, params: dict) -> None:
        r = await self._client.post(
            "/jobs",
            json={"job_id": job_id, "video_path": server_video_path, **params},
        )
        r.raise_for_status()

    async def poll_status(self, job_id: str) -> dict:
        r = await self._client.get(f"/jobs/{job_id}")
        r.raise_for_status()
        return r.json()

    async def download_results(self, job_id: str, dest: Path) -> None:
        async with self._client.stream("GET", f"/jobs/{job_id}/results", timeout=300.0) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)

    async def confirm_downloaded(self, job_id: str) -> None:
        r = await self._client.delete(f"/jobs/{job_id}")
        r.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()
