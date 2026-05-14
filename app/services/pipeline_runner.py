"""Async wrapper around CardCaptureFlow that emits SSE events per step.

In production the runner invokes the Metaflow flow in a thread-pool executor
so it doesn't block the asyncio event loop.  Each Metaflow step module reads
``EVENT_BUS_RUN_ID`` from the environment and calls into ``_event_bus_registry``
to emit ``stage_started`` / ``stage_completed`` events.

For testing, pass a *flow_cls* that accepts keyword arguments and drives the
bus directly.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Type

# Repo root is three levels up from app/services/pipeline_runner.py
_REPO_ROOT = str(Path(__file__).parent.parent.parent)

from app.services.event_bus import Event, EventBus
from app.services import _event_bus_registry


class PipelineRunner:
    """Bridges an asynchronous HTTP handler to a synchronous Metaflow run."""

    def __init__(self, bus: EventBus, flow_cls: Optional[Type] = None) -> None:
        self.bus = bus
        self.flow_cls = flow_cls

    async def run_async(
        self,
        run_id: str,
        *,
        video: str,
        output_dir: str,
        db: str,
        detector: str = "docaligner",
        config_preset: str = "balanced",
    ) -> None:
        """Run the pipeline in a thread-pool executor, emitting SSE events."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._run_blocking,
            run_id,
            video,
            output_dir,
            db,
            detector,
            config_preset,
        )

    def _run_blocking(
        self,
        run_id: str,
        video: str,
        output_dir: str,
        db: str,
        detector: str,
        config_preset: str,
    ) -> None:
        """Blocking implementation; called from the thread-pool executor."""
        os.environ["EVENT_BUS_RUN_ID"] = run_id
        os.environ["EVENT_BUS_INPROC"] = "1"

        _event_bus_registry.set(run_id, self.bus)

        try:
            self.bus.emit(run_id, Event(name="run_started"))

            if self.flow_cls is not None:
                # Testable path: call the injected flow class directly.
                self.flow_cls(
                    run_id=run_id,
                    bus=self.bus,
                    video=video,
                    output_dir=output_dir,
                    db=db,
                    detector=detector,
                    config_preset=config_preset,
                )
            else:
                # Production path: subprocess — same as the CLI, avoids Runner's
                # PYTHONPATH and working-directory issues.
                env = os.environ.copy()
                env["PYTHONPATH"] = _REPO_ROOT + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                cmd = [
                    sys.executable, "-m", "pipeline.card_capture_flow", "run",
                    "--video", video,
                    "--output-dir", output_dir,
                    "--db", db,
                    "--detector", detector,
                    "--config-preset", config_preset,
                ]
                result = subprocess.run(cmd, env=env, cwd=_REPO_ROOT)
                if result.returncode != 0:
                    raise RuntimeError(f"Pipeline subprocess exited with code {result.returncode}")

            self.bus.emit(run_id, Event(name="run_completed"))
        except Exception as exc:
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            raise
        finally:
            _event_bus_registry.clear(run_id)
            os.environ.pop("EVENT_BUS_RUN_ID", None)
            os.environ.pop("EVENT_BUS_INPROC", None)
