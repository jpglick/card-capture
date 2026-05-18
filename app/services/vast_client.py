"""Vast.ai REST API client for instance provisioning.

Uses httpx to call the vast.ai v0 API directly instead of the CLI,
which requires Python 3.10+ and is incompatible with this project's venv.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx

GPU_TYPE_QUERIES: dict[str, str] = {
    "RTX 4090": "gpu_name=RTX+4090&num_gpus=1&reliability2>0.95",
    "Flagship": "num_gpus=1&reliability2>0.99&rentable=true&order=flops_per_dphtotal-",
    "RTX 5060 Ti": "gpu_name=RTX+5060+Ti&num_gpus=1&reliability2>0.95",
}

_BASE = "https://console.vast.ai/api/v0"

_BOOT_SCRIPT = (
    "cd /workspace/card-capture && "
    "git pull origin {branch} -q && "
    "pip install -e '.[app]' -q && "
    "nohup uvicorn app.vastai_worker:app --host 0.0.0.0 --port 8765 &"
)


class VastAIClient:
    """Thin wrapper around the vast.ai v0 REST API."""

    def __init__(self, api_key: str) -> None:
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _get(self, path: str, params: Optional[dict] = None) -> object:
        r = httpx.get(f"{_BASE}{path}", headers=self._headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _put(self, path: str, body: dict) -> object:
        r = httpx.put(f"{_BASE}{path}", headers=self._headers, json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> None:
        r = httpx.delete(f"{_BASE}{path}", headers=self._headers, timeout=30)
        r.raise_for_status()

    def search_offers(self, gpu_type: str) -> list[dict]:
        """Return available offers matching the GPU type, cheapest first."""
        query = GPU_TYPE_QUERIES.get(gpu_type, f"gpu_name={gpu_type.replace(' ', '+')}&num_gpus=1")
        # vast.ai search endpoint accepts query params
        r = httpx.get(
            f"{_BASE}/bundles/",
            headers=self._headers,
            params={"q": f"{{\"rentable\":{{\"eq\":true}},\"num_gpus\":{{\"eq\":1}},\"verified\":{{\"eq\":true}}}}", "order": "dph_total", "type": "on-demand"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        offers = data.get("offers", [])
        # Filter by GPU name if a known type
        gpu_name = gpu_type.replace("RTX ", "RTX ").strip()
        if gpu_type in ("RTX 4090", "RTX 5060 Ti"):
            offers = [o for o in offers if gpu_name.lower() in o.get("gpu_name", "").lower()]
        offers.sort(key=lambda o: o.get("dph_total", 999))
        return offers

    def provision(
        self,
        offer_id: int,
        template_id: str,
        branch: str = "main",
    ) -> dict:
        """Launch an instance. Returns the instance dict with at least {"id": int}."""
        script = _BOOT_SCRIPT.format(branch=branch)
        body = {
            "client_id": "me",
            "image": template_id,
            "onstart": script,
            "runtype": "ssh",
            "disk": 40,
            "extra_env": {"open_ports": "8765/tcp"},
        }
        r = httpx.put(
            f"{_BASE}/asks/{offer_id}/",
            headers=self._headers,
            json=body,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        # API returns {"success": true, "new_contract": 12345}
        instance_id = data.get("new_contract") or data.get("id")
        return {"id": instance_id, **data}

    def destroy(self, instance_id: int) -> None:
        """Destroy a running instance. Billing stops immediately."""
        try:
            r = httpx.delete(
                f"{_BASE}/instances/{instance_id}/",
                headers=self._headers,
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"[VastAIClient] Warning: destroy {instance_id} failed: {e}")

    def get_instance_ip(self, instance_id: int) -> Optional[str]:
        """Return the public IP of a running instance, or None if not yet assigned."""
        try:
            r = httpx.get(
                f"{_BASE}/instances/",
                headers=self._headers,
                timeout=30,
            )
            r.raise_for_status()
            instances = r.json().get("instances", [])
        except Exception:
            return None
        for inst in instances:
            if inst.get("id") == instance_id:
                return inst.get("public_ipaddr") or None
        return None
