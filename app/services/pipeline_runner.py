from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Type

_REPO_ROOT = str(Path(__file__).parent.parent.parent)
logger = logging.getLogger(__name__)

# macOS objc runtime prints duplicate-dylib warnings when cv2 and av both ship
# libavdevice. They're harmless noise — filter before logging or persisting.
_NOISE_SUBSTRINGS = (
    "objc[",
    "implemented in both",
    "One of the duplicates must be removed",
    "This may cause spurious casting failures",
)


def _is_noise(line: str) -> bool:
    return any(s in line for s in _NOISE_SUBSTRINGS)

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
        try:
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
        except Exception:
            # Already handled in _run_blocking (status updated, event emitted).
            # Swallow here so the exception doesn't propagate through Starlette's
            # ASGI background task layer and generate a spurious 500 traceback.
            pass

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
            self._record_run_start(run_id, video_id)
            print(f"[{run_id}] pipeline starting — video={video}", flush=True)

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
                    sys.executable, "-m", "pipeline.card_capture_flow",
                    "--no-pylint", "run",
                    "--max-num-splits", "500",
                    "--video", abs_video,
                    "--output-dir", abs_output,
                    "--db", abs_db,
                    "--detector", detector,
                    "--config-preset", config_preset,
                    "--ui-run-id", run_id,
                ]
                print(f"[{run_id}] running: {' '.join(cmd)}", flush=True)

                start_time = time.time()
                sampler = None
                if self.db_path:
                    from app.services.resource_sampler import ResourceSampler
                    sampler = ResourceSampler(run_id, self.db_path, time.monotonic())
                    sampler.start()
                try:
                    proc = subprocess.Popen(
                        cmd, env=env, cwd=_REPO_ROOT,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line and not _is_noise(line):
                            print(f"[{run_id}] {line}", flush=True)
                            self.bus.emit(run_id, Event(name="log", payload={"line": line}))
                            self._persist_log(run_id, line)
                    proc.wait()
                finally:
                    if sampler is not None:
                        sampler.stop()

                if proc.returncode != 0:
                    raise RuntimeError(f"Pipeline exited with code {proc.returncode}")

            print(f"[{run_id}] pipeline completed", flush=True)
            self.bus.emit(run_id, Event(name="run_completed"))
            self._set_video_status(video_id, "completed")
            self._record_run_finish(run_id, "completed")
            self._sample_presence_frames(run_id, video_id, video)

        except Exception as exc:
            print(f"[{run_id}] pipeline failed: {exc}", flush=True)
            self.bus.emit(run_id, Event(name="run_failed", payload={"error": str(exc)}))
            self._set_video_status(video_id, "failed")
            self._record_run_finish(run_id, "failed")
            raise
        finally:
            _event_bus_registry.clear(run_id)
            os.environ.pop("EVENT_BUS_RUN_ID", None)
            os.environ.pop("EVENT_BUS_INPROC", None)

    def _record_run_start(self, run_id: str, video_id: int) -> None:
        if not self.db_path:
            return
        try:
            import sqlite3
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status) VALUES (?, ?, 'running')",
                    (run_id, video_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run start: {exc}", flush=True)
        try:
            import sqlite3, json as _json
            from app.services.resource_sampler import get_host_info
            host_info = get_host_info()
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET host_info_json = ? WHERE run_id = ?",
                    (_json.dumps(host_info), run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record host info: {exc}", flush=True)

    def _record_run_finish(self, run_id: str, status: str) -> None:
        if not self.db_path:
            return
        try:
            import sqlite3
            with sqlite3.connect(str(self.db_path)) as conn:
                cards = conn.execute(
                    "SELECT COUNT(*) FROM card_instances WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
                conn.execute(
                    "UPDATE pipeline_runs SET status=?, cards_extracted=?, finished_at=datetime('now') WHERE run_id=?",
                    (status, cards, run_id),
                )
        except Exception as exc:
            print(f"[{run_id}] could not record run finish: {exc}", flush=True)

    def _sample_presence_frames(self, run_id: str, video_id: int, video_path: str) -> None:
        if not self.db_path:
            return
        try:
            from app.services.presence_sampler import sample_presence_frames
            from pathlib import Path as _Path
            base_output = _Path(self.db_path).parent
            n = sample_presence_frames(
                video_path=_Path(video_path),
                run_id=run_id,
                video_id=video_id,
                output_dir=base_output,
                db_path=_Path(self.db_path),
            )
            print(f"[{run_id}] queued {n} presence frames for labeling", flush=True)
        except Exception as exc:
            print(f"[{run_id}] presence sampling skipped: {exc}", flush=True)

    def _persist_log(self, run_id: str, line: str) -> None:
        if not self.db_path:
            return
        try:
            import sqlite3
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT INTO pipeline_run_logs (run_id, line) VALUES (?, ?)",
                    (run_id, line),
                )
        except Exception:
            pass  # never let a DB error kill the pipeline

    def _set_video_status(self, video_id: int, status: str) -> None:
        if not self.db_path:
            return
        try:
            from card_capture.storage import Storage
            Storage(self.db_path).update_video_status(video_id, status)
        except Exception as exc:
            logger.warning("Could not update video %s status to %s: %s", video_id, status, exc)
