"""GPURunner — structural protocol for switchable GPU compute backends."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GPURunner(Protocol):
    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        config_preset: str = "balanced",
        **kw,
    ) -> None: ...

    async def run_batch_async(self, jobs: list[dict]) -> None: ...

    async def destroy_instance(self) -> None: ...
