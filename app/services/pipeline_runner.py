from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Type

_REPO_ROOT = str(Path(__file__).parent.parent.parent)
logger = logging.getLogger(__name__)

from app.services.event_bus import Event, EventBus
from app.services import _event_bus_registry


class PipelineRunner:
    def __init__(self, bus: EventBus, flow_cls: Optional[Type] = None, db_path: Optional[Path] = None) -> None:
        self.bus = bus
        self.flow_cls = flow_cls
        self.db_path = db_path  # for updating video status on failure

    async def run_async(
        self,
        run_id: str,
        *,
        video_id: int,
        video: str,
        output_dir: str,
        db: str,
        detector: str = "docaligner",
        config_preset: str = "balanced",
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._run_blocking,
            run_id,
            video_id,
            video,
            output_dir,
            db,
            detector,
            config_preset,
        )

    def _run_blocking(
        self,
        run_id: str,
        video_id: int,
        video: str,
        output_dir: str,
        db: str,
        detector: str,
        config_preset: str,
    ) -> None:
        os.environ["EVENT_BUS_RUN_ID"] = run_id
        os.environ["EVENT_BUS_INPROC"] = "1"
        _event_bus_registry.set(run_id, self.bus)

        try:
            self.bus.emit(run_id, Event(name="run_started"))
            logger.info("[%s] pipeline starting — video=%s", run_id, video)

            if self.flow_cls is not None:
                self.flow_cls(
                    run_id=run_id, bus=self.bus, video=video,
                    output_dir=output_dir, db=db,
                    detector=detector, config_preset=config_preset,
                )
            else:
                env = os.environ.copy()
                env["PYTHONPATH"] = _REPO_ROOT + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                # Use absolute db path so the subprocess always finds it
                abs_db = str(Path(db).resolve())
                abs_video = str(Path(video).resolve())
                abs_output = str((Path(_REPO_ROOT) / output_dir).resolve())
                cmd = [
                    sys.executable, "-m", "pipeline.card_capture_flow", "run",
                    "--video", abs_video,
                    "--output-dir", abs_output,
                    "--db", abs_db,
                    "--detector", detector,
                    "--config-preset", config_preset,
                ]
                logger.info("[%s] running: %s", run_id, " ".join(cmd))

                proc = subprocess.Popen(
                    cmd, env=env, cwd=_REPO_ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info("[%s] %s", run_id, line)
                        self.bus.emit(run_id, Event(name="log", payload={"line": line}))
                proc.wait()

                if proc.returncode != 0:
                    raise RuntimeError(f"Pipeline exited with code {proc.returncode}")

            logger.info("[%s] pipeline completed", run_id)
            self.bus.emit(run_id, Event(name="run_completed"))
            self._set_video_status(video_id, "completed")

        except Exception as exc:
            logger.error("[%s] pipeline failed: %s", run_id, exc)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            self._set_video_status(video_id, "failed")
            raise
        finally:
            _event_bus_registry.clear(run_id)
            os.environ.pop("EVENT_BUS_RUN_ID", None)
            os.environ.pop("EVENT_BUS_INPROC", None)

    def _set_video_status(self, video_id: int, status: str) -> None:
        if not self.db_path:
            return
        try:
            from card_capture.storage import Storage
            Storage(self.db_path).update_video_status(video_id, status)
        except Exception as exc:
            logger.warning("Could not update video %s status to %s: %s", video_id, status, exc)
