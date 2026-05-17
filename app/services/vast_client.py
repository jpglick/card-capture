"""Thin wrapper around the vastai CLI for instance provisioning."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

GPU_TYPE_QUERIES: dict[str, str] = {
    "RTX 4090": "gpu_name=RTX_4090 num_gpus=1 reliability>0.95",
    "Flagship": "num_gpus=1 reliability>0.99",   # sorted by TFLOPS at provision time
    "RTX 5060 Ti": "gpu_name=RTX_5060_Ti num_gpus=1 reliability>0.95",
}

_BOOT_SCRIPT = (
    "cd /workspace/card-capture && "
    "git pull origin {branch} -q && "
    "pip install -e '.[app]' -q && "
    "nohup uvicorn app.vastai_worker:app --host 0.0.0.0 --port 8765 &"
)


class VastAIClient:
    """Wraps the vastai CLI. All calls require VAST_API_KEY in the environment."""

    def __init__(self, api_key: str) -> None:
        self._env = {**os.environ, "VAST_API_KEY": api_key}

    def _run(self, *args: str) -> object:
        result = subprocess.run(
            ["vastai", *args, "--raw"],
            capture_output=True, text=True, env=self._env, check=True,
        )
        return json.loads(result.stdout)

    def search_offers(self, gpu_type: str) -> list[dict]:
        """Return available offers matching the GPU type, cheapest first."""
        query = GPU_TYPE_QUERIES.get(gpu_type, gpu_type)
        offers = self._run("search", "offers", query)
        if isinstance(offers, list):
            offers.sort(key=lambda o: o.get("dph_total", 999))
        return offers if isinstance(offers, list) else []

    def provision(
        self,
        offer_id: int,
        template_id: str,
        branch: str = "main",
    ) -> dict:
        """Launch an instance. Returns the instance dict with at least {"id": int}."""
        script = _BOOT_SCRIPT.format(branch=branch)
        result = self._run(
            "create", "instance", str(offer_id),
            "--image", template_id,
            "--onstart", script,
            "--ports", "8765",
        )
        return result if isinstance(result, dict) else {"id": result}

    def destroy(self, instance_id: int) -> None:
        """Destroy a running instance. Billing stops immediately."""
        self._run("destroy", "instance", str(instance_id))

    def get_instance_ip(self, instance_id: int) -> Optional[str]:
        """Return the public IP of a running instance, or None if not yet assigned."""
        try:
            instances = self._run("show", "instances")
        except subprocess.CalledProcessError:
            return None
        if not isinstance(instances, list):
            return None
        for inst in instances:
            if inst.get("id") == instance_id:
                return inst.get("public_ipaddr") or None
        return None
